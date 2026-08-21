"""Benchmark real sampled generation and retain the exact completions as demo evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.report import atomic_write_json
from src.eval.suite import discover_checkpoints, file_sha256, load_variant_model
from src.generation import SamplingSettings, generate_text
from src.serve.registry import RATIOS, parameter_count


PROMPTS = (
    "A practical reason to compare attention with state-space layers is",
    "In a small language model, memory usage matters because",
    "The experiment showed that",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="week5-generation-cuda-v1")
    parser.add_argument("--training-run-id", default="week3-700m-v1")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/openwebtext.json"))
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratio", action="append", choices=RATIOS)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt-limit", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    return {"commit": commit, "branch": branch, "dirty": dirty}


def runtime_state(device: torch.device) -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
        result["device_capability"] = list(torch.cuda.get_device_capability(device))
    else:
        result["device_name"] = platform.processor() or "CPU"
    return result


def summarize(samples: list[dict]) -> dict:
    metrics = [sample["metrics"] for sample in samples]
    median = lambda key: statistics.median(float(item[key]) for item in metrics)
    return {
        "sample_count": len(samples),
        "median_tokens_per_second": median("tokens_per_second"),
        "median_decode_tokens_per_second": median("decode_tokens_per_second"),
        "median_time_to_first_token_seconds": median("time_to_first_token_seconds"),
        "median_peak_vram_mib": None if metrics[0]["peak_vram_mib"] is None else median("peak_vram_mib"),
        "median_logical_state_mib": median("logical_state_mib"),
    }


def markdown(result: dict) -> str:
    lines = [
        f"# {result['run_id']}",
        "",
        "Measured sampled generation from the certified Week 3 checkpoints. Speed includes prompt prefill,",
        "sampling, and recurrent decode; decode speed is reported separately. Each completion is retained in JSON.",
        "",
        "| Ratio | Samples | End-to-end tok/s | Decode tok/s | TTFT (s) | Peak VRAM (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in result["variants"]:
        summary = variant["summary"]
        peak = summary["median_peak_vram_mib"]
        lines.append(
            f"| {variant['ratio']} | {summary['sample_count']} | "
            f"{summary['median_tokens_per_second']:.2f} | "
            f"{summary['median_decode_tokens_per_second']:.2f} | "
            f"{summary['median_time_to_first_token_seconds']:.3f} | "
            f"{'n/a' if peak is None else f'{peak:.1f}'} |"
        )
    lines.extend([
        "",
        f"Device: `{result['runtime']['device_name']}`. Git commit: `{result['git']['commit']}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not ROOT.samefile(Path.cwd()):
        raise RuntimeError("run the generation benchmark from the repository root")
    git = git_state()
    if args.require_clean and git["dirty"]:
        raise RuntimeError("the final generation benchmark requires a clean Git worktree")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer is missing: {args.tokenizer}")
    output_dir = args.out_root / args.run_id
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    ratios = tuple(args.ratio or RATIOS)
    checkpoints = discover_checkpoints(args.checkpoint_root, args.training_run_id, ratios)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompts = PROMPTS[:args.prompt_limit]
    started = utc_now()
    variants = []
    for ratio in ratios:
        checkpoint = checkpoints[ratio]
        model = load_variant_model(checkpoint, device)
        warmup = SamplingSettings(temperature=0.8, top_k=1, max_new_tokens=4, seed=9000)
        generate_text(
            model, tokenizer, prompts[0], warmup, device=device, ratio=ratio,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
        )
        samples = []
        for prompt_index, prompt in enumerate(prompts):
            settings = SamplingSettings(
                temperature=args.temperature,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                seed=1701 + prompt_index,
            )
            sample = generate_text(
                model, tokenizer, prompt, settings, device=device, ratio=ratio,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
            ).to_dict()
            sample["prompt_id"] = f"P{prompt_index + 1}"
            samples.append(sample)
            print(
                f"{ratio} P{prompt_index + 1}: "
                f"{sample['metrics']['tokens_per_second']:.2f} tok/s, "
                f"{sample['metrics']['generated_tokens']} tokens"
            )
        variants.append({
            "ratio": ratio,
            "parameters": parameter_count(checkpoint.model_config),
            "checkpoint": {
                "path": str(checkpoint.best_path),
                "sha256": checkpoint.checkpoint_sha256,
                "step": checkpoint.step,
                "recorded_val_loss": checkpoint.val_loss,
            },
            "summary": summarize(samples),
            "samples": samples,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema": 1,
        "run_id": args.run_id,
        "training_run_id": args.training_run_id,
        "created_at": started,
        "completed_at": utc_now(),
        "git": git,
        "runtime": runtime_state(device),
        "protocol": {
            "ratios": list(ratios),
            "prompts": list(prompts),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "warmup_new_tokens": 4,
            "seed_by_prompt": [1701 + index for index in range(len(prompts))],
            "batch_size": 1,
        },
        "variants": variants,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "generation_results.json"
    table_path = output_dir / "generation_table.md"
    atomic_write_json(result_path, result)
    table_path.write_text(markdown(result), encoding="utf-8", newline="\n")
    manifest = {
        "schema": 1,
        "run_id": args.run_id,
        "git_commit": git["commit"],
        "artifacts": {
            result_path.name: file_sha256(result_path),
            table_path.name: file_sha256(table_path),
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
