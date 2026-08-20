"""Run or resume the certified Week 4 inference and retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

# Running this file directly puts scripts/ first on sys.path; add the repository root for src imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare_data import validate_prepared_dataset
from src.eval.report import atomic_write_json, write_artifacts
from src.eval.suite import (
    WEEK3_RATIOS,
    Week4Protocol,
    discover_checkpoints,
    evaluate_variant,
    evaluation_source_hashes,
    file_sha256,
    git_provenance,
    read_json,
    runtime_provenance,
)
from src.data.dataset import load_split


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_shape(value: Any) -> Any:
    """Normalize tuples and dataclass primitives to the representation written in JSON."""
    return json.loads(json.dumps(value, allow_nan=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="week4-eval-v1")
    parser.add_argument("--training-run-id", default="week3-700m-v1")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/openwebtext-5b"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/openwebtext.json"))
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("outputs/week4-eval"))
    parser.add_argument("--ratio", action="append", choices=WEEK3_RATIOS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_partial(
    partial: dict[str, Any],
    ratio: str,
    protocol: dict[str, Any],
    git: dict[str, Any],
    checkpoint_sha256: str,
    source_sha256: dict[str, str],
) -> dict[str, Any]:
    if (
        partial.get("schema") != 1
        or partial.get("ratio") != ratio
        or partial.get("protocol") != protocol
        or partial.get("git_commit") != git["commit"]
        or partial.get("checkpoint_sha256") != checkpoint_sha256
        or partial.get("source_sha256") != source_sha256
        or not isinstance(partial.get("result"), dict)
    ):
        raise RuntimeError(f"partial result for {ratio} does not match this evaluation")
    return partial["result"]


def main() -> None:
    args = parse_args()
    if not ROOT.samefile(Path.cwd()):
        raise RuntimeError("run the Week 4 evaluator from the repository root")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")

    protocol = Week4Protocol.smoke() if args.smoke else Week4Protocol()
    protocol.validate()
    protocol_json = json_shape(asdict(protocol))
    ratios = tuple(args.ratio or WEEK3_RATIOS)
    if len(set(ratios)) != len(ratios):
        raise ValueError("each ratio may be selected only once")

    git = git_provenance(ROOT)
    if args.require_clean and git["dirty"]:
        raise RuntimeError("the final evaluation requires a clean Git worktree")
    source_hashes = evaluation_source_hashes(ROOT)
    data_manifest = validate_prepared_dataset(args.data_dir)
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer is missing: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    validation_data = load_split(args.data_dir, "val")
    checkpoints = discover_checkpoints(args.checkpoint_root, args.training_run_id, ratios)

    out_root = args.out_root or (Path("outputs") if args.smoke else Path("results"))
    output_dir = out_root / args.run_id
    if output_dir.exists() and (output_dir / "manifest.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"completed output already exists: {output_dir}; pass --overwrite to replace it"
        )
    work_dir = args.work_root / args.run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    created_at = utc_now()
    for ratio in ratios:
        checkpoint = checkpoints[ratio]
        partial_path = work_dir / f"ratio-{ratio.replace(':', '-')}.json"
        if partial_path.is_file() and not args.overwrite:
            variant = validate_partial(
                read_json(partial_path),
                ratio,
                protocol_json,
                git,
                checkpoint.checkpoint_sha256,
                source_hashes,
            )
            print(f"resume: loaded completed Week 4 partial for {ratio}")
        else:
            print(f"evaluate: {ratio} from {checkpoint.best_path}")
            variant = evaluate_variant(
                checkpoint, validation_data, tokenizer, protocol, device
            )
            atomic_write_json(partial_path, {
                "schema": 1,
                "ratio": ratio,
                "protocol": protocol_json,
                "git_commit": git["commit"],
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "source_sha256": source_hashes,
                "completed_at": utc_now(),
                "result": variant,
            })
        variants.append(variant)

    result = {
        "schema": 1,
        "run_id": args.run_id,
        "status": "completed",
        "smoke": args.smoke,
        "created_at": created_at,
        "completed_at": utc_now(),
        "training_run_id": args.training_run_id,
        "protocol": protocol_json,
        "git": git,
        "runtime": runtime_provenance(device),
        "source_sha256": source_hashes,
        "data": {
            "path": args.data_dir.as_posix(),
            "signature": data_manifest["signature"],
            "manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "validation_tokens": len(validation_data),
        },
        "tokenizer": {
            "path": args.tokenizer.as_posix(),
            "sha256": file_sha256(args.tokenizer),
            "vocab_size": tokenizer.get_vocab_size(),
        },
        "variants": variants,
    }
    manifest = write_artifacts(result, output_dir)
    print(f"completed: {output_dir}")
    print(f"artifacts: {len(manifest['artifact_sha256'])}")


if __name__ == "__main__":
    main()
