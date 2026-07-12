"""HybridLM — embedding, the stack of HybridBlocks, a final norm, and a tied LM head.

This is the thing I train. The layer_types list from the config decides which blocks are attention
vs Mamba, so switching the 1:3 / 1:7 / 1:15 variant is just a different config, not different code.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.attention import RotaryEmbedding
from src.model.block import HybridBlock
from src.model.config import ModelConfig
from src.model.norm import RMSNorm


class HybridLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.grad_checkpointing = False  # trainer flips this on to trade compute for VRAM
        self.blocks = nn.ModuleList(HybridBlock(cfg, t) for t in cfg.layer_types)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight  # share one weight, counted once

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the residual-path output projections so deep stacks stay stable
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("out.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, L = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope(L)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                # recompute the block in backward instead of storing its activations
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # parameters() dedupes tied weights
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
