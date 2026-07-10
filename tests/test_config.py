"""Checks on the ratio -> layer-pattern logic and the param counts.

Mostly guarding against the thing that bit me in D-ARCH-01: a ratio silently collapsing to zero
attention layers because the depth is too small for the period.
"""

from src.model.config import ModelConfig, build_layer_pattern


def test_pattern_lengths_and_counts():
    # 16 layers is the depth I locked, so all three ratios should come out clean
    assert build_layer_pattern(16, "1:3").count("attention") == 4    # 4 attn / 12 mamba
    assert build_layer_pattern(16, "1:7").count("attention") == 2    # 2 / 14
    assert build_layer_pattern(16, "1:15").count("attention") == 1   # 1 / 15
    for ratio in ("1:3", "1:7", "1:15"):
        assert len(build_layer_pattern(16, ratio)) == 16


def test_attention_sits_at_end_of_period():
    # with 1:3 the period is 4 and attention is the last slot of each: indices 3, 7, 11, 15
    pat = build_layer_pattern(16, "1:3")
    assert [i for i, t in enumerate(pat) if t == "attention"] == [3, 7, 11, 15]


def test_1_15_needs_the_depth():
    # the D-ARCH-01 trap: at only 8 layers a 1:15 can't fit a single attention layer
    assert build_layer_pattern(8, "1:15").count("attention") == 0
    assert build_layer_pattern(16, "1:15").count("attention") == 1


def test_config_derived_dims():
    cfg = ModelConfig(ratio="1:7", d_model=448, n_layers=16, vocab_size=16000)
    assert cfg.n_heads == 7            # 448 / 64
    assert cfg.d_inner == 896          # expand 2
    assert cfg.n_mamba_heads == 14     # 896 / 64
    assert cfg.n_attention_layers == 2 and cfg.n_mamba_layers == 14
