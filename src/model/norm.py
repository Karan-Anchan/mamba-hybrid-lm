"""RMSNorm — the norm I use everywhere in the stack.

Cheaper than LayerNorm (no mean-subtraction, no bias) and it's what Mamba/Llama-style models use.
I do the normalize step in float32 even under bf16 autocast, since the variance is the one bit that
actually hurts if it's low-precision.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)
