"""Causal self-attention mixer — the "expensive but global" half of the hybrid.

RoPE for positions, and I lean on torch's scaled_dot_product_attention so I get the fused/flash
kernel for free when it's available (no hard flash-attn dependency, which matters since I couldn't
build the custom kernels on this box). There's an optional KV cache so generation can reuse past
keys/values instead of recomputing them — that's also what I'll measure for the memory-vs-context plot.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Precompute cos/sin tables once; slice out the positions I need per forward."""

    def __init__(self, head_dim: int, max_seq: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)          # (max_seq, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (max_seq, head_dim)
        # buffers so they move with .to(device) / .cuda() but aren't trained
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        # offset lets me grab the right positions when decoding with a cache
        return self.cos[offset : offset + seq_len], self.sin[offset : offset + seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # cos/sin: (L, hd) -> broadcast over batch and heads
    cos, sin = cos[None, None], sin[None, None]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.nheads = cfg.n_heads
        self.hd = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.attn_bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.attn_bias)

    def forward(self, x, cos, sin, cache: dict | None = None):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        # (B, L, nheads, hd) -> (B, nheads, L, hd)
        q = q.view(B, L, self.nheads, self.hd).transpose(1, 2)
        k = k.view(B, L, self.nheads, self.hd).transpose(1, 2)
        v = v.view(B, L, self.nheads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None and "k" in cache:
            # prepend what we've already seen, then this step attends to the whole thing
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v

        # is_causal only when doing a full parallel pass; with a cache the new tokens
        # legitimately see all cached positions, so I drop the causal mask there
        is_causal = cache is None or k.shape[2] == L
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).reshape(B, L, -1)
        return self.out(y)
