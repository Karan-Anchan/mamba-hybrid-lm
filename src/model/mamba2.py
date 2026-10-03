"""Mamba-2 mixer with explicit, checkpoint-compatible scan engines.

I use the SSD "dual" form: because a selective SSM with a scalar decay per head is equivalent to a
masked attention, I can compute the whole thing as an L x L matrix instead of looping over time. It's
O(L^2) like attention. This remains the historical default training engine. A bounded
PyTorch scan and an optional official fused scan can be selected at runtime and must
pass parity checks before performance comparisons. Stateful inference is bounded.

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
from src.model.inference import Mamba2State
from src.model.norm import RMSNorm
from src.model.scan_backend import resolve_scan_backend


def ssd(x, dt, A, B, C, D):
    """Masked-attention scan with float32 inputs; autocast can lower matmul precision.

    Keep this historical execution unchanged. Recurrent memory is float32, but
    float32 inputs alone do not disable autocast for the einsum contractions.
    """
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


def ssd_stateful(x, dt, A, B, C, D, initial_state):
    """Evaluate one bounded chunk and carry its exact recurrent state forward."""
    _, l, _, _ = x.shape
    x, dt, B, C = x.float(), dt.float(), B.float(), C.float()
    A, initial_state = A.float(), initial_state.float()

    dtA = dt * A
    cumA = torch.cumsum(dtA, dim=1).transpose(1, 2)
    decay = cumA[..., :, None] - cumA[..., None, :]
    causal = torch.tril(torch.ones(l, l, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~causal, float("-inf")).exp()

    cb = torch.einsum("bin,bjn->bij", C, B)
    xdt = x * dt[..., None]
    local_y = torch.einsum("bhij,bjhp->bihp", cb[:, None] * decay, xdt)

    carry_scale = cumA.exp()
    carried_y = torch.einsum("bhl,bhpn,bln->blhp", carry_scale, initial_state, C)
    y = local_y + carried_y + x * D.float()[None, None, :, None]

    end_scale = (cumA[..., -1, None] - cumA).exp()
    local_state = torch.einsum("bhl,blhp,bln->bhpn", end_scale, xdt, B)
    next_state = carry_scale[..., -1, None, None] * initial_state + local_state
    return y, next_state


class Mamba2Mixer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_groups != 1:
            raise ValueError("n_groups must be 1; grouped Mamba B/C are not implemented")
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
        self.scan_backend = resolve_scan_backend("reference")

        self._reset_ssm_params()

    def init_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype
    ) -> Mamba2State:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("Mamba inference batch size must be a positive integer")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Mamba convolution state dtype must be a supported floating dtype")
        try:
            device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("Mamba inference-state device must be a valid torch device") from exc
        conv_tail = self.conv1d.kernel_size[0] - 1
        return Mamba2State(
            conv=torch.zeros(batch_size, self.conv_dim, conv_tail, device=device, dtype=dtype),
            ssm=torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=device,
                dtype=torch.float32,
            ),
        )

    def _validate_forward(
        self, u: torch.Tensor, state: Mamba2State | None, chunk_size: int
    ) -> None:
        """Check the request before touching memory retained from earlier tokens."""
        if not isinstance(u, torch.Tensor) or u.ndim != 3:
            raise ValueError("Mamba input must have shape (batch, tokens, d_model)")
        if u.shape[0] <= 0 or u.shape[1] <= 0 or u.shape[2] != self.in_proj.in_features:
            raise ValueError("Mamba input needs positive batch/tokens and the configured d_model")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("Mamba inference chunk size must be a positive integer")
        supported_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        if u.dtype not in supported_dtypes:
            raise ValueError("Mamba input must use a supported floating dtype")
        weight = self.in_proj.weight
        if u.device != weight.device:
            raise ValueError("Mamba input device must match the mixer device")
        autocast = torch.is_autocast_enabled(u.device.type)
        # Autocast changes the projection dtype, but does not convert float64 operands.
        autocast_projection = autocast and torch.float64 not in (u.dtype, weight.dtype)
        if u.dtype != weight.dtype and not autocast_projection:
            raise ValueError("Mamba input dtype must match mixer weights outside autocast")
        projected_dtype = torch.get_autocast_dtype(u.device.type) if autocast_projection else weight.dtype
        if state is None:
            return
        if not isinstance(state, Mamba2State):
            raise ValueError("Mamba inference state must be a Mamba2State")
        if not isinstance(state.conv, torch.Tensor) or not isinstance(state.ssm, torch.Tensor):
            raise ValueError("Mamba inference state must contain convolution and SSM tensors")
        expected_conv = (u.shape[0], self.conv_dim, self.conv1d.kernel_size[0] - 1)
        expected_ssm = (u.shape[0], self.nheads, self.headdim, self.d_state)
        if state.conv.shape != expected_conv or state.ssm.shape != expected_ssm:
            raise ValueError("Mamba inference-state shape does not match this input and mixer")
        if state.conv.device != u.device or state.ssm.device != u.device:
            raise ValueError("Mamba inference-state device must match the input device")
        if state.conv.dtype != projected_dtype:
            raise ValueError(
                f"Mamba convolution state uses {state.conv.dtype}, projected input uses "
                f"{projected_dtype}"
            )
        if state.ssm.dtype != torch.float32:
            raise ValueError("Mamba SSM state must use torch.float32 for recurrent accumulation")

    def _reset_ssm_params(self):
        # A in [1, 16] like the paper, stored as log; dt_bias set so softplus(dt) starts small
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
            dt = torch.empty(self.nheads).uniform_(0.001, 0.1)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def forward(
        self, u: torch.Tensor, state: Mamba2State | None = None, chunk_size: int | None = None
    ) -> torch.Tensor:
        chunk_size = self.scan_backend.chunk_size if chunk_size is None else chunk_size
        self._validate_forward(u, state, chunk_size)
        if self.scan_backend.name == "fused_mamba" and chunk_size != self.scan_backend.chunk_size:
            raise ValueError("fused chunk size is fixed by configure_scan_backend; per-call overrides are unsupported")
        batch, L, _ = u.shape
        z, xBC, dt = self.in_proj(u).split(
            [self.d_inner, self.conv_dim, self.nheads], dim=-1)

        if state is None:
            xBC = self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2)
        else:
            projected = xBC.transpose(1, 2)
            # Check the actual result too, in case a caller installs a custom projection.
            if projected.dtype != state.conv.dtype:
                raise ValueError(
                    f"Mamba convolution state uses {state.conv.dtype}, projected input uses "
                    f"{projected.dtype}"
                )
            combined = torch.cat([state.conv, projected], dim=2)
            xBC = F.conv1d(
                combined,
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.conv_dim,
            ).transpose(1, 2)
            tail = state.conv.shape[2]

        xBC = F.silu(xBC)
        x, Bm, Cm = xBC.split([self.d_inner, self.ngroups * self.d_state,
                               self.ngroups * self.d_state], dim=-1)

        x = x.view(batch, L, self.nheads, self.headdim)
        Bm = Bm.view(batch, L, self.d_state)   # ngroups = 1
        Cm = Cm.view(batch, L, self.d_state)
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)

        y, carried = self.scan_backend.scan(
            x, dt, A, Bm, Cm, self.D, None if state is None else state.ssm,
            reference_scan=ssd, stateful_scan=ssd_stateful,
            inference_chunk_size=chunk_size,
        )

        y = y.reshape(batch, L, self.d_inner)
        y = self.norm(y * F.silu(z))       # z gates, then normalize
        result = self.out_proj(y.to(u.dtype))
        if state is not None:
            # Commit both memories only after this mixer has produced its output successfully.
            if tail:
                state.conv.copy_(combined[..., -tail:])
            state.ssm.copy_(carried)
        return result
