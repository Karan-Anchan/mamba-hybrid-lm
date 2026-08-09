"""Train the 1:3, 1:7, and 1:15 variants at a matched token budget.

Every invocation gets a namespace. Results go to ``results/<run-id>/`` and training artifacts go
to ``checkpoints/<run-id>/<variant>/``. Reusing the same explicit ``--run-id`` resumes an interrupted
variant and skips variants that already have a validated completed result.

The 8,000-step default is a 131.072M-token reduced run at batch 8, accumulation 4, block 512. The
authoritative 700M-token Week-3 run uses 42,725 steps:

    python scripts/run_sweep.py --data-dir data/openwebtext-5b --run-id week3-700m-v1 \
        --max-steps 42725 --warmup-steps 855
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.train import (  # noqa: E402
    TrainConfig,
    atomic_write_json,
    atomic_write_text,
    read_json,
    run,
    validate_run_id,
)
from src.data.prepare_data import validate_prepared_dataset  # noqa: E402

CONFIGS = ["configs/ratio_1_3.yaml", "configs/ratio_1_7.yaml", "configs/ratio_1_15.yaml"]


def to_markdown(results: list[dict]) -> str:
    cols = ["ratio", "params_m", "n_attention", "best_val_ppl", "avg_tok_per_s", "peak_vram_mb", "tokens_seen"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(result[column]) for column in cols) + " |" for result in results]
    return "\n".join([head, sep, *rows])


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("sweep-%Y%m%d-%H%M%S")


def sweep_output_dir(root: str | Path, run_id: str) -> Path:
    return Path(root) / validate_run_id(run_id)


def tokens_for_steps(steps: int, batch_size: int, grad_accum: int, block_size: int) -> int:
    return steps * batch_size * grad_accum * block_size


def steps_for_tokens(target_tokens: int, batch_size: int, grad_accum: int, block_size: int) -> int:
    tokens_per_step = batch_size * grad_accum * block_size
    return (target_tokens + tokens_per_step - 1) // tokens_per_step


def warmup_steps_for_fraction(steps: int, fraction: float) -> int:
    """Round a planned warmup fraction up to a whole optimizer step."""
    if steps < 0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("steps must be non-negative and fraction must be between 0 and 1")
    return math.ceil(steps * fraction)


def _sweep_signature(args: argparse.Namespace, run_id: str, data_signature: str) -> str:
    payload = {
        "configs": CONFIGS,
        "data_signature": data_signature,
        "run_id": run_id,
        "settings": {
            key: value for key, value in vars(args).items()
            if key not in {"resume", "wandb", "out"}
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", required=True,
        help="verified prepared-dataset directory; explicit to prevent accidental preview-data runs",
    )
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="optimizer steps; default is a reduced 131.072M-token run, not the 700M target")
    ap.add_argument(
        "--warmup-steps", type=int, default=200,
        help="linear-warmup steps; the authoritative 42,725-step run requires 855 (2%%)",
    )
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--checkpoint-interval", type=int, default=500)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="the sweep defaults to no checkpointing for speed; pass this to enable it")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default="results", help="root for namespaced sweep summaries")
    ap.add_argument("--checkpoint-root", default="checkpoints", help="root for namespaced training artifacts")
    ap.add_argument("--run-id", help="stable namespace; reuse it to resume, omit it to create a timestamped run")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    data_manifest = validate_prepared_dataset(Path(args.data_dir))
    data_signature = data_manifest["signature"]
    print(f"verified prepared data: {Path(args.data_dir).resolve()} ({data_signature})")
    run_id = validate_run_id(args.run_id or default_run_id())
    out = sweep_output_dir(args.out, run_id)
    out.mkdir(parents=True, exist_ok=True)
    print(f"run id: {run_id}")

    tokens_per_step = tokens_for_steps(1, args.batch_size, args.grad_accum, args.block_size)
    manifest_path = out / "sweep_manifest.json"
    signature = _sweep_signature(args, run_id, data_signature)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"sweep manifest must be a JSON object: {manifest_path}")
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite existing sweep: {out}")
        if manifest.get("signature") != signature:
            raise RuntimeError(f"sweep settings differ from the existing run namespace: {out}")
        manifest.update({"status": "running", "resumed_at": datetime.now(timezone.utc).isoformat()})
    else:
        manifest = {
            "schema": 1,
            "run_id": run_id,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "configs": CONFIGS,
            "arguments": vars(args),
            "data_signature": data_signature,
            "tokens_per_step": tokens_per_step,
            "tokens_per_variant": tokens_per_step * args.max_steps,
            "signature": signature,
        }
    atomic_write_json(manifest_path, manifest)

    results: list[dict] = []
    sweep_t0 = time.time()
    for config_path in CONFIGS:
        print(f"\n{'='*70}\n  {config_path}\n{'='*70}")
        train_cfg = TrainConfig(
            model_config=config_path,
            data_dir=args.data_dir,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            block_size=args.block_size,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            grad_checkpointing=args.grad_checkpointing,
            wandb=args.wandb,
            ckpt_dir=args.checkpoint_root,
            run_id=run_id,
            resume=args.resume,
        )
        results.append(run(train_cfg))

    elapsed_minutes = (time.time() - sweep_t0) / 60
    # Variant result.json files are already durable. Replace aggregate summaries only after all three
    # validate, so reconstructing a run can never truncate a previously complete table to one row.
    atomic_write_json(out / "sweep_results.json", results)
    atomic_write_text(out / "sweep_table.md", to_markdown(results) + "\n")
    manifest.update({
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_minutes_this_invocation": elapsed_minutes,
        "completed_variants": [result["ratio"] for result in results],
    })
    atomic_write_json(manifest_path, manifest)
    print(f"\nsweep done in {elapsed_minutes:.1f} min\n")
    print(to_markdown(results))


if __name__ == "__main__":
    main()
