"""SwiGLU MLP — the feed-forward that follows every mixer.

Two projections up (gate + value), a SiLU gate, one projection back down. With hidden ~= 8/3 * d
this lands at roughly 8*d^2 params, same ballpark as a vanilla 4x GELU MLP but it trains a bit better.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.mlp_hidden
        self.gate = nn.Linear(cfg.d_model, h, bias=cfg.mlp_bias)
        self.up = nn.Linear(cfg.d_model, h, bias=cfg.mlp_bias)
        self.down = nn.Linear(h, cfg.d_model, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
