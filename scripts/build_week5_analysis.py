"""Build the joined Week 4 and Week 5 analysis bundle from committed measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.report import atomic_write_json, atomic_write_text
from src.eval.suite import file_sha256
from src.eval.week5_report import (
    analysis_markdown,
    build_analysis,
    ratio_tradeoffs_svg,
    runtime_comparison_svg,
    state_crossover_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week4", type=Path, default=Path("results/week4-eval-v1/evaluation_results.json"))
    parser.add_argument("--cuda", type=Path, default=Path("results/week5-generation-cuda-v1/generation_results.json"))
    parser.add_argument("--cpu", type=Path, default=Path("results/week5-generation-cpu-matched-v1/generation_results.json"))
    parser.add_argument("--output", type=Path, default=Path("results/week5-analysis-v1"))
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    if not ROOT.samefile(Path.cwd()):
        raise RuntimeError("run the Week 5 analysis builder from the repository root")
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    if args.require_clean and dirty:
        raise RuntimeError("the final Week 5 analysis requires a clean Git worktree")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    analysis = build_analysis(read_json(args.week4), read_json(args.cuda), read_json(args.cpu))
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "analysis.json": lambda: atomic_write_json(args.output / "analysis.json", analysis),
        "README.md": lambda: atomic_write_text(args.output / "README.md", analysis_markdown(analysis)),
        "plots/ratio_tradeoffs.svg": lambda: atomic_write_text(
            args.output / "plots/ratio_tradeoffs.svg", ratio_tradeoffs_svg(analysis)
        ),
        "plots/runtime_comparison.svg": lambda: atomic_write_text(
            args.output / "plots/runtime_comparison.svg", runtime_comparison_svg(analysis)
        ),
        "plots/state_crossover.svg": lambda: atomic_write_text(
            args.output / "plots/state_crossover.svg", state_crossover_svg(analysis)
        ),
    }
    for write in artifacts.values():
        write()
    manifest = {
        "schema": 1,
        "source_sha256": {
            str(args.week4): file_sha256(args.week4),
            str(args.cuda): file_sha256(args.cuda),
            str(args.cpu): file_sha256(args.cpu),
        },
        "artifact_sha256": {
            name: file_sha256(args.output / name) for name in artifacts
        },
    }
    atomic_write_json(args.output / "manifest.json", manifest)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
