"""Run the attention:SSM ratio sweep — train 1:3, 1:7, 1:15 at matched compute and collect results.

"Matched compute" here = identical token budget and identical everything-else across the three
variants; the only thing that changes is which layers are attention vs Mamba. Since the effective
batch and step count are the same for all three, they all see the same number of tokens, so any
difference in perplexity / throughput / VRAM is down to the ratio — which is the whole question.

    python scripts/run_sweep.py --data-dir data/openwebtext --max-steps 8000
    # writes results/sweep_results.json + results/sweep_table.md, one checkpoint per variant

I train them sequentially (one 12 GB GPU), freeing the model between runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "src..." imports work when run as a script
from src.train.train import TrainConfig, run  # noqa: E402

CONFIGS = ["configs/ratio_1_3.yaml", "configs/ratio_1_7.yaml", "configs/ratio_1_15.yaml"]


def to_markdown(results: list[dict]) -> str:
    cols = ["ratio", "params_m", "n_attention", "best_val_ppl", "avg_tok_per_s", "peak_vram_mb", "tokens_seen"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in results]
    return "\n".join([head, sep, *rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/openwebtext")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="on by default the sweep runs WITHOUT checkpointing for speed; pass this to force it on")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    sweep_t0 = time.time()
    for cfg_path in CONFIGS:
        print(f"\n{'='*70}\n  {cfg_path}\n{'='*70}")
        tcfg = TrainConfig(
            model_config=cfg_path, data_dir=args.data_dir,
            max_steps=args.max_steps, warmup_steps=args.warmup_steps,
            block_size=args.block_size, batch_size=args.batch_size, grad_accum=args.grad_accum,
            lr=args.lr, eval_interval=args.eval_interval,
            grad_checkpointing=args.grad_checkpointing, wandb=args.wandb,
        )
        results.append(run(tcfg))
        # save after each variant so a crash on run 3 doesn't lose runs 1-2
        (out / "sweep_results.json").write_text(json.dumps(results, indent=2))
        (out / "sweep_table.md").write_text(to_markdown(results) + "\n")

    mins = (time.time() - sweep_t0) / 60
    print(f"\nsweep done in {mins:.1f} min\n")
    print(to_markdown(results))


if __name__ == "__main__":
    main()
