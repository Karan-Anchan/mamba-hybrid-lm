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
from src.model.inference import AttentionCache, HybridInferenceState
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
        # Training used 512 positions; the larger table lets Week 4 evaluate an 8K prompt plus decode.
        self.rope = RotaryEmbedding(cfg.head_dim, max_seq=16384)

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

    def init_inference_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        cache_dtype: torch.dtype | None = None,
    ) -> HybridInferenceState:
        if batch_size <= 0:
            raise ValueError("inference batch size must be positive")
        device = self.embed.weight.device if device is None else torch.device(device)
        cache_dtype = self.embed.weight.dtype if cache_dtype is None else cache_dtype
        layers = []
        for block in self.blocks:
            if block.is_attn:
                layers.append(AttentionCache())
            else:
                layers.append(block.mixer.init_state(batch_size, device, cache_dtype))
        return HybridInferenceState(
            layers=layers,
            batch_size=batch_size,
            max_seq_len=self.rope.cos.shape[0],
        )

    def forward(
        self,
        idx,
        targets=None,
        *,
        inference_state: HybridInferenceState | None = None,
        logits_to_keep: int | None = None,
    ):
        B, L = idx.shape
        if inference_state is not None:
            if targets is not None:
                raise ValueError("stateful inference does not accept training targets")
            if len(inference_state.layers) != len(self.blocks):
                raise ValueError("inference-state layer count does not match the model")
            inference_state.validate_step(L, B)
        if logits_to_keep is not None and logits_to_keep <= 0:
            raise ValueError("logits_to_keep must be positive")

        x = self.embed(idx)
        offset = inference_state.position if inference_state is not None else 0
        cos, sin = self.rope(L, offset=offset)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for layer_index, blk in enumerate(self.blocks):
            if self.grad_checkpointing and self.training:
                if inference_state is not None:
                    raise RuntimeError("gradient checkpointing cannot mutate inference state")
                # recompute the block in backward instead of storing its activations
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                state = None if inference_state is None else inference_state.layers[layer_index]
                x = blk(x, cos, sin, state)
        if inference_state is not None:
            inference_state.advance(L)

        x = self.norm_f(x)
        if logits_to_keep is not None:
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if logits_to_keep is not None:
                raise ValueError("logits_to_keep cannot be used when computing loss")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def prefill(
        self, idx: torch.Tensor, *, cache_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, HybridInferenceState]:
        state = self.init_inference_state(
            idx.shape[0], device=idx.device, cache_dtype=cache_dtype
        )
        logits, _ = self(idx, inference_state=state, logits_to_keep=1)
        return logits, state

    @torch.no_grad()
    def decode(self, idx: torch.Tensor, state: HybridInferenceState) -> torch.Tensor:
        logits, _ = self(idx, inference_state=state, logits_to_keep=1)
        return logits

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # parameters() dedupes tied weights
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
