"""Model configuration and layer-pattern construction for the hybrid LM.

Pure-Python (no torch import) so it can be used by tooling like ``scripts/count_params.py``
without a GPU or the SSM kernels. The dataclass here is the single source of truth for a
model's shape; the actual ``nn.Module`` (Week 2) will consume the same ``ModelConfig``.

Ratio convention
----------------
``attn:ssm`` = 1:3 means one causal-attention layer for every three Mamba-2 layers, i.e. a
repeating period of length 4. 1:7 -> period 8, 1:15 -> period 16. The attention layer is
placed at the *end* of each period (SSM layers first, then the attention layer), matching the
common hybrid convention of not starting the stack with attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def build_layer_pattern(n_layers: int, ratio: str) -> list[str]:
    """Return a list like ['mamba', 'mamba', 'mamba', 'attention', ...] of length ``n_layers``.

    ``ratio`` is a string ``"a:s"`` (attention:ssm). The period is ``a + s``; within each
    period the ``s`` SSM layers come first, followed by the ``a`` attention layer(s). If
    ``n_layers`` is not a whole number of periods, the remainder is filled with SSM layers
    (the cheaper, more numerous type) so the requested depth is honoured exactly.
    """
    a_str, s_str = ratio.split(":")
    a, s = int(a_str), int(s_str)
    period = a + s
    pattern: list[str] = []
    for i in range(n_layers):
        pos = i % period
        # SSM layers occupy positions [0, s); attention occupies [s, period)
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
        if not self.layer_types:
            self.layer_types = build_layer_pattern(self.n_layers, self.ratio)
        assert self.d_model % self.head_dim == 0, "d_model must be divisible by head_dim"
        d_inner = self.expand * self.d_model
        assert d_inner % self.mamba_headdim == 0, "d_inner must be divisible by mamba_headdim"

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
    def mlp_hidden(self) -> int:
        h = int(self.mlp_ratio * self.d_model)
        m = self.mlp_multiple_of
        return ((h + m - 1) // m) * m
