"""Deterministic Week 4 JSON, Markdown, and code-native SVG reporting."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Callable

from src.eval.suite import file_sha256


RATIO_COLORS = {"1:3": "#2b8cbe", "1:7": "#7b6fd0", "1:15": "#d46b5f"}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _fmt_rate(value: float) -> str:
    return f"{value:,.1f}"


def _fmt_mib(value: int) -> str:
    return f"{value / 2**20:,.2f}"


def comparison_table(result: dict[str, Any]) -> str:
    longest = result["protocol"]["context_lengths"][-1]
    lines = [
        "# Week 4 comparison",
        "",
        f"All speed and state-memory columns below use the {longest:,}-token context. ",
        "Logical tensor memory is kept separate from CUDA allocator deltas.",
        "",
        "| Ratio | Validation PPL | Prefill tok/s | Decode tok/s | Attention KV MiB | Mamba state MiB | Total state MiB | Needle exact match |",
        "|:--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for variant in result["variants"]:
        row = next(
            item for item in variant["inference"] if item["context_length"] == longest
        )
        mamba_bytes = row["mamba_conv_bytes"] + row["mamba_ssm_bytes"]
        retrieval = variant["needle_retrieval"]
        lines.append(
            f"| {variant['ratio']} | {variant['validation']['perplexity']:.3f} | "
            f"{_fmt_rate(row['prefill_tokens_per_second'])} | "
            f"{_fmt_rate(row['decode_tokens_per_second'])} | "
            f"{_fmt_mib(row['attention_kv_bytes'])} | {_fmt_mib(mamba_bytes)} | "
            f"{_fmt_mib(row['logical_state_bytes'])} | "
            f"{retrieval['matches']}/{retrieval['trials']} ({retrieval['accuracy']:.1%}) |"
        )
    lines.extend([
        "",
        "The Mamba state column is recurrent convolution plus SSM state and is constant with context length. "
        "Attention KV grows linearly with the number of cached tokens and attention layers.",
        "",
    ])
    return "\n".join(lines)


def result_readme(result: dict[str, Any]) -> str:
    contexts = ", ".join(f"{value:,}" for value in result["protocol"]["context_lengths"])
    return f"""# Week 4 inference and long-context evaluation

This directory contains the certified `{result['run_id']}` evaluation of the three Week 3 checkpoints.
The run rechecks fixed-window validation quality, separates prefill from token-by-token decode speed, measures
logical recurrent/KV state alongside CUDA allocation deltas, and runs controlled exact-match retrieval.

## Protocol

- Checkpoints: best-validation artifacts from `{result['training_run_id']}` after manifest SHA-256 validation
- Context lengths: {contexts} tokens
- Decode: {result['protocol']['decode_tokens']} timed greedy steps after each prefill
- Throughput repeats: {result['protocol']['throughput_repeats']} after warmup; medians are reported
- Validation: {result['protocol']['validation_iters']} fixed batches × {result['protocol']['validation_batch_size']} × {result['protocol']['validation_block_size']} tokens
- Retrieval depths: {', '.join(f"{depth:.0%}" for depth in result['protocol']['needle_depths'])}
- Precision: bf16 CUDA compute with fp32 recurrent SSM state

## Files

- `evaluation_results.json`: complete machine-readable measurements and trial records
- `evaluation_table.md`: compact 8K comparison derived from the JSON
- `manifest.json`: protocol, source/checkpoint provenance, and generated-artifact hashes
- `plots/validation_perplexity.svg`: quality comparison
- `plots/decode_throughput.svg`: decode speed across prompt lengths
- `plots/logical_state_memory.svg`: recurrent plus KV state scaling
- `plots/needle_retrieval.svg`: exact-match retrieval by context

## Interpretation boundary

These measurements describe this repository's plain-PyTorch recurrent/chunked implementation on the recorded
RTX 5070 runtime. Logical state bytes are exact tensor sizes. Allocated and peak deltas include framework and
kernel workspaces and therefore answer a different question. Retrieval uses one pre-registered code at each
context/depth pair; it is a controlled diagnostic, not a broad language-understanding benchmark.

See `evaluation_table.md` for the headline comparison and `evaluation_results.json` for every raw timing and
generated answer.
"""


def _line_chart(
    result: dict[str, Any],
    title: str,
    subtitle: str,
    value_at: Callable[[dict[str, Any], int], float],
    y_label: str,
    formatter: Callable[[float], str],
) -> str:
    width, height = 960, 560
    left, right, top, bottom = 92, 44, 104, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    contexts = result["protocol"]["context_lengths"]
    series = {
        variant["ratio"]: [value_at(variant, context) for context in contexts]
        for variant in result["variants"]
    }
    values = [value for rows in series.values() for value in rows]
    y_min = 0.0 if min(values) >= 0 and max(values) > min(values) * 1.8 else min(values) * 0.96
    y_max = max(values) * 1.06 if max(values) else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_pos(index: int) -> float:
        return left + (plot_w * index / max(1, len(contexts) - 1))

    def y_pos(value: float) -> float:
        return top + plot_h * (1.0 - (value - y_min) / (y_max - y_min))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{left}" y="42" font-family="Segoe UI, sans-serif" font-size="25" font-weight="700" fill="#172033">{html.escape(title)}</text>',
        f'<text x="{left}" y="70" font-family="Segoe UI, sans-serif" font-size="13" fill="#59677d">{html.escape(subtitle)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        parts.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#dce3eb" stroke-width="1"/>',
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#66758a">{html.escape(formatter(value))}</text>',
        ])
    for index, context in enumerate(contexts):
        x = x_pos(index)
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#4a5870">{context:,}</text>'
        )
    parts.extend([
        f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="12" fill="#4a5870">prompt context (tokens)</text>',
        f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="12" fill="#4a5870">{html.escape(y_label)}</text>',
    ])
    for series_index, (ratio, rows) in enumerate(series.items()):
        color = RATIO_COLORS[ratio]
        points = " ".join(f"{x_pos(i):.2f},{y_pos(value):.2f}" for i, value in enumerate(rows))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for index, value in enumerate(rows):
            parts.append(
                f'<circle cx="{x_pos(index):.2f}" cy="{y_pos(value):.2f}" r="5" fill="#f8fafc" stroke="{color}" stroke-width="3"/>'
            )
        legend_x = left + plot_w - 210 + series_index * 72
        parts.extend([
            f'<line x1="{legend_x}" y1="92" x2="{legend_x + 20}" y2="92" stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x + 26}" y="96" font-family="Consolas, monospace" font-size="12" fill="#263247">{ratio}</text>',
        ])
    parts.append("</svg>\n")
    return "".join(parts)


def _perplexity_bars(result: dict[str, Any]) -> str:
    width, height = 760, 500
    values = [(variant["ratio"], variant["validation"]["perplexity"]) for variant in result["variants"]]
    y_min = min(value for _, value in values) * 0.995
    y_max = max(value for _, value in values) * 1.005
    plot_top, plot_bottom = 108, 410
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Fixed-window validation perplexity</title>',
        '<desc id="desc">Lower is better; all variants reuse the Week 3 fixed-window protocol.</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="72" y="44" font-family="Segoe UI, sans-serif" font-size="25" font-weight="700" fill="#172033">Fixed-window validation perplexity</text>',
        '<text x="72" y="72" font-family="Segoe UI, sans-serif" font-size="13" fill="#59677d">Lower is better · identical validation windows</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        parts.extend([
            f'<line x1="72" y1="{y:.2f}" x2="710" y2="{y:.2f}" stroke="#dce3eb"/>',
            f'<text x="60" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="11" fill="#66758a">{value:.2f}</text>',
        ])
    for index, (ratio, value) in enumerate(values):
        x, bar_w = 132 + index * 196, 112
        y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        parts.extend([
            f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{plot_bottom - y:.2f}" rx="8" fill="{RATIO_COLORS[ratio]}"/>',
            f'<text x="{x + bar_w / 2}" y="{y - 12:.2f}" text-anchor="middle" font-family="Consolas, monospace" font-size="15" font-weight="700" fill="#263247">{value:.3f}</text>',
            f'<text x="{x + bar_w / 2}" y="442" text-anchor="middle" font-family="Consolas, monospace" font-size="14" fill="#263247">{ratio}</text>',
        ])
    parts.append("</svg>\n")
    return "".join(parts)


def write_artifacts(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    atomic_write_json(output_dir / "evaluation_results.json", result)
    atomic_write_text(output_dir / "evaluation_table.md", comparison_table(result))
    atomic_write_text(output_dir / "README.md", result_readme(result))
    atomic_write_text(plots / "validation_perplexity.svg", _perplexity_bars(result))
    atomic_write_text(plots / "decode_throughput.svg", _line_chart(
        result,
        "Greedy decode throughput",
        "Median synchronized batch-1 throughput after one warmup",
        lambda variant, context: next(
            row["decode_tokens_per_second"] for row in variant["inference"]
            if row["context_length"] == context
        ),
        "decode tokens / second",
        lambda value: f"{value:,.0f}",
    ))
    atomic_write_text(plots / "logical_state_memory.svg", _line_chart(
        result,
        "Logical inference-state memory",
        "Attention KV plus fixed recurrent Mamba convolution and SSM state",
        lambda variant, context: next(
            row["logical_state_bytes"] / 2**20 for row in variant["inference"]
            if row["context_length"] == context
        ),
        "logical state (MiB)",
        lambda value: f"{value:.1f}",
    ))
    atomic_write_text(plots / "needle_retrieval.svg", _line_chart(
        result,
        "Needle retrieval exact match",
        "One pre-registered code at each context/depth pair",
        lambda variant, context: next(
            row["accuracy"] * 100 for row in variant["needle_retrieval"]["by_context"]
            if row["context_length"] == context
        ),
        "exact match (%)",
        lambda value: f"{value:.0f}%",
    ))

    artifact_paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": 1,
        "run_id": result["run_id"],
        "status": "completed",
        "created_at": result["created_at"],
        "completed_at": result["completed_at"],
        "training_run_id": result["training_run_id"],
        "protocol": result["protocol"],
        "git": result["git"],
        "runtime": result["runtime"],
        "source_sha256": result["source_sha256"],
        "checkpoint_evidence": [variant["checkpoint"] for variant in result["variants"]],
        "artifact_sha256": {
            path.relative_to(output_dir).as_posix(): file_sha256(path) for path in artifact_paths
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
