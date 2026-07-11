"""HybridBlock — one layer of the stack. The mixer is either Mamba-2 or attention, picked by the
ratio pattern; everything else (pre-norm, residuals, the MLP) is identical so the two layer types
are drop-in swappable.
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.attention import CausalAttention
from src.model.config import ModelConfig
from src.model.mamba2 import Mamba2Mixer
from src.model.mlp import SwiGLU
from src.model.norm import RMSNorm


class HybridBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_type: str):
        super().__init__()
        self.is_attn = layer_type == "attention"
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = CausalAttention(cfg) if self.is_attn else Mamba2Mixer(cfg)

        self.has_mlp = cfg.mlp_on_every_layer
        if self.has_mlp:
            self.norm2 = RMSNorm(cfg.d_model)
            self.mlp = SwiGLU(cfg)

    def forward(self, x, cos=None, sin=None, cache=None):
        # attention needs positions + optional kv cache; mamba just needs the sequence
        if self.is_attn:
            x = x + self.mixer(self.norm1(x), cos, sin, cache)
        else:
            x = x + self.mixer(self.norm1(x))
        if self.has_mlp:
            x = x + self.mlp(self.norm2(x))
        return x
