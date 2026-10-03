"""Train configurable architecture/initialization arms at a matched token budget.

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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.train import (  # noqa: E402
    TrainConfig,
    RunLock,
    atomic_write_json,
    atomic_write_text,
    read_json,
    run,
    load_model_config,
    validate_run_id,
    variant_slug,
)
from src.data.prepare_data import validate_prepared_dataset  # noqa: E402
from src.model.scan_backend import resolve_scan_backend  # noqa: E402

CONFIGS = ["configs/ratio_1_3.yaml", "configs/ratio_1_7.yaml", "configs/ratio_1_15.yaml"]


def to_markdown(results: list[dict]) -> str:
    cols = ["ratio", "params_m", "n_attention", "best_val_ppl", "avg_tok_per_s", "peak_vram_mb", "tokens_seen"]
    if any("model_seed" in result for result in results):
        cols = ["name", "model_seed", "data_seed", "eval_seed", *cols]
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


def _sweep_signature(
    args: argparse.Namespace, run_id: str, data_signature: str, matrix: list[dict], backend_plan: dict,
) -> str:
    payload = {
        "matrix": matrix,
        "scan_backend_plan": backend_plan,
        "data_signature": data_signature,
        "run_id": run_id,
        "settings": {
            key: value for key, value in vars(args).items()
            if key not in {"resume", "wandb", "out", "dry_run"}
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def argument_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--configs", nargs="+", help="model YAML files; default: the three historical ratios")
    ap.add_argument("--model-seeds", nargs="+", type=int,
                    help="independent initialization seeds; each gets a separate training namespace")
    ap.add_argument("--data-seed", type=int, help="common training-window seed across all arms (default: 1337)")
    ap.add_argument("--eval-seed", type=int, help="common development-window seed across all arms (default: 1337)")
    ap.add_argument("--scan-backend", choices=("reference", "torch_chunked", "fused_mamba"), default="reference",
                    help="explicit scan implementation; unavailable fused requests fail without fallback")
    ap.add_argument("--scan-chunk-size", type=int, default=128,
                    help="bounded scan chunk length; fused mode requires a power of two")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify data/configs and print a JSON matrix without creating outputs or training")
    return ap


def _validate_settings(args: argparse.Namespace) -> None:
    for name in (
        "max_steps", "block_size", "batch_size", "grad_accum", "eval_interval",
        "eval_iters", "log_interval", "checkpoint_interval", "scan_chunk_size",
    ):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= args.warmup_steps <= args.max_steps:
        raise ValueError("warmup_steps must be between zero and max_steps")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    seeds = list(args.model_seeds or [])
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate model seeds would create duplicate campaign arms")
    for name, values in (
        ("model_seeds", seeds), ("data_seed", [args.data_seed]), ("eval_seed", [args.eval_seed]),
    ):
        for value in values:
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32
            ):
                raise ValueError(f"{name} must contain integers between 0 and 2^32 - 1")


def prepare_matrix(args: argparse.Namespace, run_id: str, data_manifest: dict) -> list[dict]:
    """Load every arm and freeze its identity before the first output directory is created."""
    configs = []
    names = set()
    slugs = set()
    paths = set()
    for config_path in args.configs or CONFIGS:
        path = Path(config_path)
        if path.resolve() in paths:
            raise ValueError(f"duplicate config path would create duplicate arms: {config_path}")
        paths.add(path.resolve())
        model = load_model_config(config_path)
        if not isinstance(model.name, str) or not model.name.strip():
            raise ValueError("model name must be a non-empty string")
        slug = variant_slug(model.name)  # Validate eventual checkpoint-directory names read-only.
        if model.name in names:
            raise ValueError(f"duplicate model name would create duplicate arms: {model.name}")
        if slug.casefold() in slugs:
            raise ValueError(f"model names would share a checkpoint directory on Windows: {model.name}")
        names.add(model.name)
        slugs.add(slug.casefold())
        if model.vocab_size != data_manifest["tokenizer"]["vocab_size"]:
            raise ValueError(f"tokenizer/model vocab mismatch: {config_path}")
        configs.append((config_path, model, hashlib.sha256(path.read_bytes()).hexdigest()))
    for split in ("train", "val"):
        # get_batch's exclusive upper bound needs more than block_size + 1 positions.
        if data_manifest["outputs"][split]["tokens"] <= args.block_size + 1:
            raise ValueError(f"{split} data is too short for block_size={args.block_size}")

    explicit_seeds = args.model_seeds is not None
    legacy_seed = TrainConfig().seed
    matrix = []
    for seed in args.model_seeds if explicit_seeds else [legacy_seed]:
        arm_run_id = validate_run_id(f"{run_id}-seed-{seed}" if explicit_seeds else run_id)
        for config_path, model, config_sha in configs:
            matrix.append({
                "arm_id": f"{model.name}@seed-{seed}",
                "config_path": config_path,
                "config_sha256": config_sha,
                "model_config": asdict(model),
                "name": model.name,
                "ratio": model.ratio,
                "realized_ratio": model.realized_ratio,
                "run_id": arm_run_id,
                "model_seed": seed,
                "data_seed": legacy_seed if args.data_seed is None else args.data_seed,
                "eval_seed": legacy_seed if args.eval_seed is None else args.eval_seed,
                "tokens": tokens_for_steps(args.max_steps, args.batch_size, args.grad_accum, args.block_size),
            })
    return matrix


def _execute_sweep(
    args: argparse.Namespace, run_id: str, data_signature: str, matrix: list[dict], out: Path, backend_plan: dict,
) -> None:
    tokens_per_step = tokens_for_steps(1, args.batch_size, args.grad_accum, args.block_size)
    manifest_path = out / "sweep_manifest.json"
    signature = _sweep_signature(args, run_id, data_signature, matrix, backend_plan)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"sweep manifest must be a JSON object: {manifest_path}")
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite existing sweep: {out}")
        if manifest.get("signature") != signature:
            raise RuntimeError(f"sweep settings differ from the existing run namespace: {out}")
        if (manifest.get("schema") != 2 or manifest.get("matrix") != matrix
                or manifest.get("scan_backend_plan") != backend_plan
                or manifest.get("data_signature") != data_signature
                or manifest.get("tokens_per_step") != tokens_per_step
                or manifest.get("tokens_per_variant") != tokens_per_step * args.max_steps
                or manifest.get("tokens_total") != sum(arm["tokens"] for arm in matrix)):
            raise RuntimeError(f"sweep manifest matrix/budget differs from its signed identity: {out}")
        manifest.update({"status": "running", "resumed_at": datetime.now(timezone.utc).isoformat()})
    else:
        manifest = {
            "schema": 2,
            "run_id": run_id,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "configs": args.configs or CONFIGS,
            "matrix": matrix,
            "scan_backend_plan": backend_plan,
            "arguments": vars(args),
            "data_signature": data_signature,
            "tokens_per_step": tokens_per_step,
            "tokens_per_variant": tokens_per_step * args.max_steps,
            "tokens_total": sum(arm["tokens"] for arm in matrix),
            "budget_kind": "token_positions",
            "signature": signature,
        }
    atomic_write_json(manifest_path, manifest)

    results: list[dict] = []
    sweep_t0 = time.time()
    for arm in matrix:
        if hashlib.sha256(Path(arm["config_path"]).read_bytes()).hexdigest() != arm["config_sha256"]:
            raise RuntimeError(f"model config changed after matrix preparation: {arm['config_path']}")
        print(f"\narm: {arm['arm_id']} ({arm['config_path']})")
        train_cfg = TrainConfig(
            model_config=arm["config_path"],
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
            run_id=arm["run_id"],
            resume=args.resume,
            model_seed=arm["model_seed"] if args.model_seeds is not None else None,
            data_seed=arm["data_seed"] if args.model_seeds is not None else args.data_seed,
            eval_seed=arm["eval_seed"] if args.model_seeds is not None else args.eval_seed,
            scan_backend=args.scan_backend,
            scan_chunk_size=args.scan_chunk_size,
        )
        result = dict(run(train_cfg))
        result["sweep_arm"] = arm
        if (args.configs is not None or args.model_seeds is not None
                or args.data_seed is not None or args.eval_seed is not None):
            result.update({name: arm[name] for name in ("model_seed", "data_seed", "eval_seed")})
        results.append(result)

    elapsed_minutes = (time.time() - sweep_t0) / 60
    # Variant result.json files are already durable. Replace aggregate summaries only after all arms
    # validate, so reconstructing a run can never truncate a previously complete table to one row.
    atomic_write_json(out / "sweep_results.json", results)
    atomic_write_text(out / "sweep_table.md", to_markdown(results) + "\n")
    manifest.update({
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_minutes_this_invocation": elapsed_minutes,
        "completed_variants": [result["ratio"] for result in results],
        "completed_arms": [arm["arm_id"] for arm in matrix],
    })
    atomic_write_json(manifest_path, manifest)
    print(f"\nsweep done in {elapsed_minutes:.1f} min\n")
    print(to_markdown(results))


def main(argv: list[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    _validate_settings(args)
    backend_plan = {
        **resolve_scan_backend(args.scan_backend, args.scan_chunk_size).metadata(),
        "execution_status": "unexecuted",
        "observed_paths": dict.fromkeys(("training", "prefill", "decode")),
    }
    run_id = validate_run_id(args.run_id or default_run_id())
    data_manifest = validate_prepared_dataset(Path(args.data_dir))
    matrix = prepare_matrix(args, run_id, data_manifest)
    data_signature = data_manifest["signature"]
    if args.dry_run:
        print(json.dumps({
            "schema": 2, "status": "dry_run", "run_id": run_id,
            "budget_kind": "token_positions", "data_signature": data_signature,
            "tokens_per_step": tokens_for_steps(1, args.batch_size, args.grad_accum, args.block_size),
            "tokens_per_arm": matrix[0]["tokens"], "tokens_total": sum(arm["tokens"] for arm in matrix),
            "matrix": matrix, "scan_backend_plan": backend_plan,
            "signature": _sweep_signature(args, run_id, data_signature, matrix, backend_plan),
        }, indent=2, sort_keys=True, allow_nan=False))
        return
    print(f"verified prepared data: {Path(args.data_dir).resolve()} ({data_signature})")
    print(f"run id: {run_id}; {len(matrix)} token-matched arms")
    out = sweep_output_dir(args.out, run_id)
    with RunLock(out / ".sweep.lock"):
        _execute_sweep(args, run_id, data_signature, matrix, out, backend_plan)


if __name__ == "__main__":
    main()
