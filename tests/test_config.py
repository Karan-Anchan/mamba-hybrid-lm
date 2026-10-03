"""Checks on the ratio -> layer-pattern logic and the param counts.

Mostly guarding against the thing that bit me in D-ARCH-01: a ratio silently collapsing to zero
attention layers because the depth is too small for the period.
"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("ratio, layer_type", [("1:0", "attention"), ("0:1", "mamba")])
def test_pure_controls_are_valid(ratio, layer_type):
    cfg = ModelConfig(n_layers=5, ratio=ratio)
    assert cfg.layer_types == [layer_type] * 5
    assert cfg.realized_ratio == ("5:0" if layer_type == "attention" else "0:5")


def test_incomplete_period_and_explicit_placement_report_actual_counts():
    cfg = ModelConfig(n_layers=6, ratio="1:3")
    assert cfg.realized_ratio == "1:5"
    explicit = ModelConfig(n_layers=3, ratio="1:3", layer_types=["attention", "mamba", "attention"])
    assert explicit.realized_ratio == "2:1"


@pytest.mark.parametrize("ratio", ["0:0", "-1:3", "1:-3", "1", "1:3:4", "a:3", "1.0:3", " :3", 13, None])
def test_invalid_ratio_is_rejected_even_with_explicit_placement(ratio):
    with pytest.raises(ValueError, match="ratio"):
        build_layer_pattern(4, ratio)
    with pytest.raises(ValueError, match="ratio"):
        ModelConfig(n_layers=1, ratio=ratio, layer_types=["mamba"])


@pytest.mark.parametrize("field", [
    "vocab_size", "d_model", "n_layers", "head_dim", "expand", "d_state",
    "mamba_headdim", "d_conv", "n_groups", "mlp_multiple_of",
])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "16", None])
def test_dimensions_require_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("n_layers", [0, -1, 1.5, True])
def test_pattern_depth_requires_a_positive_integer(n_layers):
    with pytest.raises(ValueError, match="n_layers"):
        build_layer_pattern(n_layers, "1:3")


@pytest.mark.parametrize("options, message", [
    ({"d_model": 127}, "divisible by head_dim"),
    ({"mamba_headdim": 65}, "divisible by mamba_headdim"),
    ({"d_model": 126, "head_dim": 63, "mamba_headdim": 63}, "even"),
    ({"n_groups": 2}, "grouped Mamba"),
])
def test_unsupported_geometry_is_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**options)


@pytest.mark.parametrize("layer_types, message", [
    (["mamba"], "exactly n_layers"),
    (["mamba"] * 9, "exactly n_layers"),
    (["mamba"] * 7 + ["ssm"], "entries"),
    (["mamba"] * 7 + [None], "entries"),
    (["mamba"] * 7 + [1], "entries"),
    (("mamba",) * 8, "list"),
    ("mamba", "list"),
    (None, "list"),
])
def test_explicit_layer_list_is_validated(layer_types, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(layer_types=layer_types)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf"), True, "2.0", 1e-10])
def test_mlp_width_requires_finite_positive_units(value):
    with pytest.raises(ValueError, match="mlp_ratio"):
        ModelConfig(mlp_ratio=value)


@pytest.mark.parametrize("field", ["tie_embeddings", "attn_bias", "mlp_bias", "mlp_on_every_layer"])
def test_boolean_flags_reject_yaml_strings_and_integers(field):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: "false"})
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: 1})


def test_historical_config_roundtrip_keeps_checkpoint_field_shape():
    cfg = ModelConfig(name="hybrid-1:7", ratio="1:7", d_model=448, n_layers=16, vocab_size=16000)
    recorded = asdict(cfg)
    # Checkpoint/run signatures use this dictionary. Derived explanations must not add fields.
    assert set(recorded) == {
        "name", "ratio", "vocab_size", "d_model", "n_layers", "tie_embeddings",
        "head_dim", "attn_bias", "expand", "d_state", "mamba_headdim", "d_conv",
        "n_groups", "mlp_ratio", "mlp_multiple_of", "mlp_bias", "mlp_on_every_layer", "layer_types",
    }
    assert asdict(ModelConfig(**json.loads(json.dumps(recorded)))) == recorded


def test_contract_checks_survive_python_optimization():
    code = """
from src.model.config import ModelConfig, build_layer_pattern
invalid = [
    lambda: ModelConfig(d_model=127),
    lambda: ModelConfig(n_groups=2),
    lambda: ModelConfig(layer_types=['mamba']),
    lambda: ModelConfig(d_conv=0),
    lambda: build_layer_pattern(4, '0:0'),
]
for make in invalid:
    try:
        make()
    except ValueError:
        continue
    raise RuntimeError('invalid configuration passed with Python -O')
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
