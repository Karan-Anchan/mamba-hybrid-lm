"""Mamba-2 mixer, written in plain PyTorch (no mamba-ssm kernel — see D-ARCH-03 / E-0002).

I use the SSD "dual" form: because a selective SSM with a scalar decay per head is equivalent to a
masked attention, I can compute the whole thing as an L x L matrix instead of looping over time. It's
O(L^2) like attention (fine at our lengths, and attention layers pay the same), and it's fully
parallel so the GPU is happy. A chunked O(L) version is the obvious later optimization if throughput
ever bites.

Shapes I keep in my head:
    x   (B, L, H, P)   H mamba heads, P = headdim
    dt  (B, L, H)      per-head timestep (after softplus)
    A   (H,)           scalar decay per head, negative
    B,C (B, L, N)      shared across heads since n_groups = 1
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.config import ModelConfig
from src.model.norm import RMSNorm


def ssd(x, dt, A, B, C, D):
    """The scan, done as a masked-attention matmul. Runs in float32 for stability."""
    b, l, h, p = x.shape
    x, dt, B, C = x.float(), dt.float(), B.float(), C.float()
    A = A.float()

    # log-decay accumulated along time; cumA_i - cumA_j = sum of dt*A over (j, i]
    dtA = dt * A                                   # (b, l, h), <= 0
    cumA = torch.cumsum(dtA, dim=1).transpose(1, 2)  # (b, h, l)
    decay = cumA[..., :, None] - cumA[..., None, :]  # (b, h, i, j) = cumA_i - cumA_j
    # Mask the non-causal half to -inf BEFORE exp. If I exp first and then mask (torch.where),
    # the j>i entries blow up to +inf and inf*0 turns into NaN gradients in backward. Masking
    # first makes exp(-inf)=0 cleanly, which cost me a debugging session (E-0003).
    causal = torch.tril(torch.ones(l, l, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~causal, float("-inf")).exp()  # (b, h, l, l)

    # C_i · B_j over the state dim; same for every head (one group)
    cb = torch.einsum("bin,bjn->bij", C, B)        # (b, l, l)
    m = cb[:, None] * decay                         # (b, h, i, j)

    xdt = x * dt[..., None]                          # fold dt_j into x_j
    y = torch.einsum("bhij,bjhp->bihp", m, xdt)      # (b, l, h, p)
    y = y + x * D.float()[None, None, :, None]       # D skip connection
    return y


class Mamba2Mixer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_inner = cfg.d_inner
        self.nheads = cfg.n_mamba_heads
        self.headdim = cfg.mamba_headdim
        self.d_state = cfg.d_state
        self.ngroups = cfg.n_groups
        gN = self.ngroups * self.d_state
        self.conv_dim = self.d_inner + 2 * gN

        # one projection makes z (gate), xBC (goes through conv), and dt
        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner + 2 * gN + self.nheads, bias=False)
        # depthwise causal conv over x, B and C; left-pad so it can't look ahead
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=cfg.d_conv,
                                groups=self.conv_dim, padding=cfg.d_conv - 1, bias=True)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(self.d_inner)   # gated: normalizes y * silu(z)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

        self._reset_ssm_params()

    def _reset_ssm_params(self):
        # A in [1, 16] like the paper, stored as log; dt_bias set so softplus(dt) starts small
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
            dt = torch.empty(self.nheads).uniform_(0.001, 0.1)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, _ = u.shape
        z, xBC, dt = self.in_proj(u).split(
            [self.d_inner, self.conv_dim, self.nheads], dim=-1)

        xBC = self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2)  # causal conv
        xBC = F.silu(xBC)
        x, Bm, Cm = xBC.split([self.d_inner, self.ngroups * self.d_state,
                               self.ngroups * self.d_state], dim=-1)

        x = x.view(B, L, self.nheads, self.headdim)
        Bm = Bm.view(B, L, self.d_state)   # ngroups = 1
        Cm = Cm.view(B, L, self.d_state)
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)

        y = ssd(x, dt, A, Bm, Cm, self.D).reshape(B, L, self.d_inner)
        y = self.norm(y * F.silu(z))       # z gates, then normalize
        return self.out_proj(y.to(u.dtype))
