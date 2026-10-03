"""Model config, plus how I turn a ratio into a list of layer types.

Kept torch-free so count_params.py can size things without touching the GPU. The dataclass is the
one place a variant's shape lives — the real nn.Module (Week 2) reads the same ModelConfig.

Ratio convention: "1:3" is one attention layer per three Mamba-2 layers, so the pattern repeats
every 4 layers (1:7 -> every 8, 1:15 -> every 16). I drop the attention layer at the end of each
period (SSM layers first) — starting a stack with attention isn't the usual hybrid setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _positive_int(name: str, value: object) -> None:
    """Reject malformed shapes before any module or checkpoint is constructed."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ratio_parts(ratio: str) -> tuple[int, int]:
    if not isinstance(ratio, str):
        raise ValueError("ratio must be an 'attention:ssm' string of nonnegative integers")
    parts = ratio.split(":")
    if len(parts) != 2 or any(not part.strip().isascii() or not part.strip().isdecimal()
                              for part in parts):
        raise ValueError("ratio must contain two nonnegative integers, e.g. '1:3'")
    attention, mamba = (int(part) for part in parts)
    if attention + mamba == 0:
        raise ValueError("ratio must have a positive period; '0:0' has no layers")
    return attention, mamba


def build_layer_pattern(n_layers: int, ratio: str) -> list[str]:
    """Build the per-layer type list, e.g. ['mamba','mamba','mamba','attention', ...].

    ratio is "a:s" (attention:ssm). Period is a+s, with the s SSM layers first and the a attention
    layer(s) after. An incomplete final period keeps that same order, so the realized counts
    can differ from the requested ratio. '1:0' and '0:1' are the pure-attention and pure-SSM
    controls, respectively.
    """
    _positive_int("n_layers", n_layers)
    a, s = _ratio_parts(ratio)
    period = a + s
    pattern: list[str] = []
    for i in range(n_layers):
        pos = i % period
        # first s slots of each period are SSM, the rest are attention
        pattern.append("attention" if pos >= s else "mamba")
    return pattern


@dataclass
class ModelConfig:
    """Shape of one hybrid-LM variant. All three ratio variants share every field except
    ``ratio`` (which changes only *which* layers are attention vs. Mamba, not the depth)."""

    # identity
    name: str = "unnamed"
    ratio: str = "1:7"          # attention:ssm

    # core dims
    vocab_size: int = 50257     # GPT-2 BPE by default; 16k custom BPE is a candidate (D-ARCH-01)
    d_model: int = 768
    n_layers: int = 8
    tie_embeddings: bool = True  # LM head shares the token-embedding weight

    # attention mixer
    head_dim: int = 64          # n_heads = d_model // head_dim
    attn_bias: bool = False

    # mamba-2 mixer
    expand: int = 2             # d_inner = expand * d_model
    d_state: int = 128
    mamba_headdim: int = 64     # n_mamba_heads = d_inner // mamba_headdim
    d_conv: int = 4
    n_groups: int = 1

    # mlp (SwiGLU); set mlp_on_every_layer=False for Mamba-style mixer-only SSM layers
    mlp_ratio: float = 8 / 3    # hidden ~= mlp_ratio * d_model, keeps params ~8*d_model^2
    mlp_multiple_of: int = 64
    mlp_bias: bool = False
    mlp_on_every_layer: bool = True

    # bookkeeping (derived)
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "d_model", "n_layers", "head_dim", "expand", "d_state",
            "mamba_headdim", "d_conv", "n_groups", "mlp_multiple_of",
        ):
            _positive_int(name, getattr(self, name))
        for name in ("tie_embeddings", "attn_bias", "mlp_bias", "mlp_on_every_layer"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        _ratio_parts(self.ratio)  # Validate the recorded ratio even with an explicit layer list.
        if self.d_model % self.head_dim != 0:
            raise ValueError("d_model must be divisible by head_dim")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position embeddings")
        d_inner = self.expand * self.d_model
        if d_inner % self.mamba_headdim != 0:
            raise ValueError("d_inner must be divisible by mamba_headdim")
        if self.n_groups != 1:
            raise ValueError("n_groups must be 1; grouped Mamba B/C are not implemented")
        if (
            not isinstance(self.mlp_ratio, (int, float))
            or isinstance(self.mlp_ratio, bool)
            or not math.isfinite(self.mlp_ratio)
            or self.mlp_ratio <= 0
        ):
            raise ValueError("mlp_ratio must be a finite positive number")
        hidden = self.mlp_ratio * self.d_model
        if not math.isfinite(hidden) or hidden < 1:
            raise ValueError("mlp_ratio * d_model must produce at least one finite hidden unit")
        if not isinstance(self.layer_types, list):
            raise ValueError("layer_types must be a list")
        if not self.layer_types:
            self.layer_types = build_layer_pattern(self.n_layers, self.ratio)
        if len(self.layer_types) != self.n_layers:
            raise ValueError("layer_types must contain exactly n_layers entries")
        if any(not isinstance(value, str) or value not in ("attention", "mamba")
               for value in self.layer_types):
            raise ValueError("layer_types entries must be 'attention' or 'mamba'")

    # convenience accessors -------------------------------------------------
    @property
    def n_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_mamba_heads(self) -> int:
        return self.d_inner // self.mamba_headdim

    @property
    def n_attention_layers(self) -> int:
        return sum(t == "attention" for t in self.layer_types)

    @property
    def n_mamba_layers(self) -> int:
        return sum(t == "mamba" for t in self.layer_types)

    @property
    def realized_ratio(self) -> str:
        """Actual attention:SSM counts, including a partial period or explicit placement."""
        return f"{self.n_attention_layers}:{self.n_mamba_layers}"

    @property
    def mlp_hidden(self) -> int:
        h = int(self.mlp_ratio * self.d_model)
        m = self.mlp_multiple_of
        return ((h + m - 1) // m) * m
