"""Count params analytically, before I've written any model code.

Runs on plain Python (no torch, no GPU) so I can size configs early. The per-module formulas below
match the parameter inventory the real nn.Module will have in Week 2 — I'll double-check them against
sum(p.numel()) once the model exists.

    python scripts/count_params.py                 # the candidate-config sweep table
    python scripts/count_params.py --config configs/ratio_1_3.yaml
    python scripts/count_params.py --target 50e6   # compare against a param target

Why I need it: D-ARCH-01 — d_model=768, ~50M params, and a clean 1:15 ratio (which needs a 16-layer
period) can't all hold at once. Putting real numbers on the table lets me pick a config on evidence.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# allow "python scripts/count_params.py" from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.config import ModelConfig  # noqa: E402


# --- per-module parameter formulas (bias-free unless noted) -------------------------------

def attn_params(cfg: ModelConfig) -> int:
    """Causal MHA: fused QKV (3*d^2) + output proj (d^2) = 4*d^2. RoPE adds no params."""
    d = cfg.d_model
    p = 3 * d * d + d * d
    if cfg.attn_bias:
        p += 3 * d + d
    return p


def mamba2_params(cfg: ModelConfig) -> int:
    """Mamba-2 (SSD) block, matching the official module's parameter inventory."""
    d = cfg.d_model
    d_inner = cfg.d_inner
    nheads = cfg.n_mamba_heads
    gN = cfg.n_groups * cfg.d_state
    conv_dim = d_inner + 2 * gN

    in_proj = d * (2 * d_inner + 2 * gN + nheads)   # [z, x, B, C, dt], bias=False
    conv1d = conv_dim * cfg.d_conv + conv_dim        # depthwise conv (groups=conv_dim) + bias
    dt_bias = nheads
    a_log = nheads                                   # scalar A per head (Mamba-2)
    d_skip = nheads                                  # D skip-connection param per head
    gated_norm = d_inner                             # RMSNorm before out_proj
    out_proj = d_inner * d                           # bias=False
    return in_proj + conv1d + dt_bias + a_log + d_skip + gated_norm + out_proj


def mlp_params(cfg: ModelConfig) -> int:
    """SwiGLU: gate + up + down = 3 * d * hidden (hidden ~= 8/3 d keeps this ~8 d^2)."""
    d, h = cfg.d_model, cfg.mlp_hidden
    p = 3 * d * h
    if cfg.mlp_bias:
        p += 2 * h + d
    return p


def count(cfg: ModelConfig) -> dict:
    d = cfg.d_model
    per_layer_norm = 2 * d          # pre-mixer + pre-mlp RMSNorm weights
    mixer = mlp = norms = 0
    for t in cfg.layer_types:
        mixer += attn_params(cfg) if t == "attention" else mamba2_params(cfg)
        if cfg.mlp_on_every_layer:
            mlp += mlp_params(cfg)
        norms += per_layer_norm
    norms += d                       # final RMSNorm
    embedding = cfg.vocab_size * d   # tied -> LM head shares this weight
    non_embed = mixer + mlp + norms
    total = non_embed + embedding
    return {
        "embedding": embedding,
        "mixer": mixer,
        "mlp": mlp,
        "norms": norms,
        "non_embed": non_embed,
        "total": total,
    }


# --- candidate configs for D-ARCH-01 ------------------------------------------------------

def candidates() -> list[ModelConfig]:
    """The configs we weigh for the ratio sweep. Each is expanded over all three ratios."""
    presets = [
        # (name, d_model, n_layers, vocab)  -- honouring d_model=768
        ("A-768x8-gpt2", 768, 8, 50257),
        ("A-768x6-gpt2", 768, 6, 50257),
        # clean-ratio-friendly (16-layer period supports a true 1:15)
        ("B-512x16-16k", 512, 16, 16000),
        ("B-448x16-16k", 448, 16, 16000),
        ("B-512x16-gpt2", 512, 16, 50257),
    ]
    cfgs: list[ModelConfig] = []
    for name, d, n, v in presets:
        for ratio in ("1:3", "1:7", "1:15"):
            cfgs.append(ModelConfig(name=f"{name} [{ratio}]", ratio=ratio,
                                    d_model=d, n_layers=n, vocab_size=v))
    return cfgs


def fmt(n: int) -> str:
    return f"{n/1e6:6.1f}M"


def print_table(cfgs: list[ModelConfig]) -> None:
    hdr = f"{'config':22} {'ratio':6} {'A/M layers':10} {'total':>8} {'non-emb':>8} {'embed':>8}"
    print(hdr)
    print("-" * len(hdr))
    for cfg in cfgs:
        c = count(cfg)
        base = cfg.name.split(" [")[0]
        am = f"{cfg.n_attention_layers}/{cfg.n_mamba_layers}"
        print(f"{base:22} {cfg.ratio:6} {am:10} {fmt(c['total']):>8} "
              f"{fmt(c['non_embed']):>8} {fmt(c['embedding']):>8}")


def load_yaml(path: str) -> ModelConfig:
    import yaml  # local import so the sweep works without pyyaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="path to a single YAML config to size")
    ap.add_argument("--target", type=float, default=None, help="param target to compare against (e.g. 50e6)")
    args = ap.parse_args()

    if args.config:
        cfg = load_yaml(args.config)
        c = count(cfg)
        print(f"{cfg.name}  (d_model={cfg.d_model}, n_layers={cfg.n_layers}, vocab={cfg.vocab_size})")
        print(f"  layer pattern : {cfg.layer_types}")
        for k in ("embedding", "mixer", "mlp", "norms", "non_embed", "total"):
            print(f"  {k:10}: {fmt(c[k])}  ({c[k]:,})")
        if args.target:
            print(f"  vs target {args.target/1e6:.0f}M non-embed: {c['non_embed']/args.target:.2f}x")
        return

    print("D-ARCH-01 candidate sweep (SwiGLU MLP on every layer):\n")
    print_table(candidates())
    print("\nNote: 'clean 1:15' requires >=16 layers; A-768* configs only approximate it.")


if __name__ == "__main__":
    main()
