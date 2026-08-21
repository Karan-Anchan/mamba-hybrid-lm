"""Derive Week 5 comparison tables, findings, and precise SVG plots from committed evidence."""

from __future__ import annotations

import html
import math
import statistics
from typing import Iterable


RATIOS = ("1:3", "1:7", "1:15")
COLORS = {"1:3": "#168aad", "1:7": "#7667d8", "1:15": "#e36f5c"}


def _variants(result: dict) -> dict[str, dict]:
    return {variant["ratio"]: variant for variant in result["variants"]}


def _require_clean(result: dict, name: str) -> None:
    if result.get("git", {}).get("dirty") is not False:
        raise ValueError(f"{name} was not measured from a clean worktree")


def build_analysis(week4: dict, cuda: dict, cpu: dict) -> dict:
    """Join the evaluation and generation protocols, rejecting mismatched runtime comparisons."""
    _require_clean(week4, "Week 4 evaluation")
    _require_clean(cuda, "CUDA generation")
    _require_clean(cpu, "CPU generation")
    cuda_protocol = cuda["protocol"]
    cpu_protocol = cpu["protocol"]
    matched_keys = ("prompts", "temperature", "top_k", "max_new_tokens", "seed_by_prompt")
    if any(cuda_protocol[key] != cpu_protocol[key] for key in matched_keys):
        raise ValueError("CPU and CUDA generation protocols are not matched")

    eval_by_ratio = _variants(week4)
    cuda_by_ratio = _variants(cuda)
    cpu_by_ratio = _variants(cpu)
    if tuple(eval_by_ratio) != RATIOS or tuple(cuda_by_ratio) != RATIOS:
        raise ValueError("the ratio comparison must contain 1:3, 1:7, and 1:15 in order")
    if tuple(cpu_by_ratio) != ("1:3",):
        raise ValueError("the CPU deployment comparison must contain only the selected 1:3 model")

    ratios = []
    state_curves = {}
    for ratio in RATIOS:
        evaluation = eval_by_ratio[ratio]
        generation = cuda_by_ratio[ratio]
        longest = evaluation["inference"][-1]
        summary = generation["summary"]
        sample_contexts = [
            sample["metrics"]["prompt_tokens"] + sample["metrics"]["generated_tokens"] - 1
            for sample in generation["samples"]
        ]
        short_context = int(statistics.median(sample_contexts))
        fixed_state = longest["mamba_conv_bytes"] + longest["mamba_ssm_bytes"]
        kv_per_token = longest["attention_kv_bytes"] / longest["context_length"]
        state_curves[ratio] = {
            "fixed_mamba_bytes": fixed_state,
            "attention_kv_bytes_per_token": kv_per_token,
        }
        ratios.append({
            "ratio": ratio,
            "parameters": evaluation["parameters"],
            "attention_layers": evaluation["attention_layers"],
            "mamba_layers": evaluation["mamba_layers"],
            "perplexity": evaluation["validation"]["perplexity"],
            "generation_tokens_per_second": summary["median_tokens_per_second"],
            "generation_decode_tokens_per_second": summary["median_decode_tokens_per_second"],
            "time_to_first_token_seconds": summary["median_time_to_first_token_seconds"],
            "peak_vram_mib": summary["median_peak_vram_mib"],
            "short_context_tokens": short_context,
            "short_logical_state_mib": summary["median_logical_state_mib"],
            "context_8k_logical_state_mib": longest["logical_state_bytes"] / 2**20,
            "context_8k_prefill_tokens_per_second": longest["prefill_tokens_per_second"],
            "context_8k_decode_tokens_per_second": longest["decode_tokens_per_second"],
            "retrieval_matches": evaluation["needle_retrieval"]["matches"],
            "retrieval_trials": evaluation["needle_retrieval"]["trials"],
        })

    by_ratio = {row["ratio"]: row for row in ratios}
    curve_13, curve_115 = state_curves["1:3"], state_curves["1:15"]
    crossover = (
        curve_115["fixed_mamba_bytes"] - curve_13["fixed_mamba_bytes"]
    ) / (
        curve_13["attention_kv_bytes_per_token"]
        - curve_115["attention_kv_bytes_per_token"]
    )
    gpu_rate = by_ratio["1:3"]["generation_tokens_per_second"]
    cpu_rate = cpu_by_ratio["1:3"]["summary"]["median_tokens_per_second"]
    calculations = {
        "ppl_cost_1_15_vs_1_3": by_ratio["1:15"]["perplexity"] - by_ratio["1:3"]["perplexity"],
        "state_saving_1_15_vs_1_3_at_8k_percent": 100 * (
            1 - by_ratio["1:15"]["context_8k_logical_state_mib"]
            / by_ratio["1:3"]["context_8k_logical_state_mib"]
        ),
        "generation_gain_1_3_vs_1_7_percent": 100 * (
            gpu_rate / by_ratio["1:7"]["generation_tokens_per_second"] - 1
        ),
        "generation_gap_1_7_vs_1_15_percent": 100 * abs(
            by_ratio["1:7"]["generation_tokens_per_second"]
            / by_ratio["1:15"]["generation_tokens_per_second"] - 1
        ),
        "cpu_share_of_cuda_1_3_percent": 100 * cpu_rate / gpu_rate,
        "cuda_speedup_over_cpu_1_3": gpu_rate / cpu_rate,
        "state_crossover_1_3_vs_1_15_tokens": crossover,
        "short_state_saving_1_3_vs_1_15_percent": 100 * (
            1 - by_ratio["1:3"]["short_logical_state_mib"]
            / by_ratio["1:15"]["short_logical_state_mib"]
        ),
    }
    return {
        "schema": 1,
        "source_runs": {
            "week4": week4["run_id"],
            "cuda": cuda["run_id"],
            "cpu": cpu["run_id"],
        },
        "protocol": {
            "generation": {key: cuda_protocol[key] for key in matched_keys},
            "long_context_tokens": week4["protocol"]["context_lengths"],
        },
        "ratios": ratios,
        "runtime_comparison": {
            "ratio": "1:3",
            "cuda": {
                "device": cuda["runtime"]["device_name"],
                **cuda_by_ratio["1:3"]["summary"],
            },
            "cpu": {
                "device": cpu["runtime"]["device_name"],
                **cpu_by_ratio["1:3"]["summary"],
            },
        },
        "state_curves": state_curves,
        "calculations": calculations,
    }


def analysis_markdown(analysis: dict) -> str:
    rows = analysis["ratios"]
    calc = analysis["calculations"]
    lines = [
        "# Week 5 analysis",
        "",
        "This report joins certified Week 4 evaluation with protocol-matched Week 5 sampled generation.",
        "",
        "| Ratio | PPL | GPU generation tok/s | TTFT (ms) | Peak VRAM (MiB) | State near 57 tokens (MiB) | State at 8K (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ratio']} | {row['perplexity']:.3f} | "
            f"{row['generation_tokens_per_second']:.2f} | "
            f"{1000 * row['time_to_first_token_seconds']:.1f} | "
            f"{row['peak_vram_mib']:.1f} | {row['short_logical_state_mib']:.2f} | "
            f"{row['context_8k_logical_state_mib']:.2f} |"
        )
    lines.extend([
        "",
        "## Findings",
        "",
        f"1. **Quality and long-context memory form the clearest trade-off.** 1:15 saves "
        f"{calc['state_saving_1_15_vs_1_3_at_8k_percent']:.1f}% logical state at 8K while adding "
        f"{calc['ppl_cost_1_15_vs_1_3']:.3f} perplexity versus 1:3.",
        f"2. **Short sampled generation favors 1:3 on this implementation.** Its median end-to-end rate is "
        f"{calc['generation_gain_1_3_vs_1_7_percent']:.1f}% above 1:7. The 1:7 and 1:15 rates differ by only "
        f"{calc['generation_gap_1_7_vs_1_15_percent']:.2f}% in this nine-sample protocol.",
        f"3. **State ordering reverses with context.** Near 57 cached tokens, 1:3 uses "
        f"{calc['short_state_saving_1_3_vs_1_15_percent']:.1f}% less logical state than 1:15 because each Mamba "
        f"layer owns a fixed recurrent state. Their calculated curves cross near "
        f"{calc['state_crossover_1_3_vs_1_15_tokens']:.0f} tokens; beyond that, attention KV growth dominates.",
        f"4. **Local CPU serving is viable for this narrow demo.** The Ryzen 7700 reaches "
        f"{calc['cpu_share_of_cuda_1_3_percent']:.1f}% of the RTX 5070's 1:3 end-to-end rate under the matched "
        f"48-token protocol. This is a local result, not a claim about a two-core cloud host.",
        "5. **Generation quality remains the limiting factor.** The samples are locally coherent but often drift, "
        "repeat phrases, or make unsupported factual statements. The demo therefore presents the model as a "
        "controlled systems experiment, not a production assistant.",
        "",
        "All findings are descriptive: one training seed and three generation prompts are not statistical "
        "evidence of a universal ratio ordering.",
        "",
    ])
    return "\n".join(lines)


def _svg_shell(width: int, height: int, title: str, description: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title>
<desc id="desc">{html.escape(description)}</desc>
<rect width="100%" height="100%" rx="24" fill="#fbfaf7"/>
<style>text{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:#182033}}.muted{{fill:#667085}}.grid{{stroke:#dfe4ea;stroke-width:1}}.label{{font-size:14px}}.value{{font-size:17px;font-weight:700}}.heading{{font-size:24px;font-weight:750}}</style>
{body}
</svg>'''


def ratio_tradeoffs_svg(analysis: dict) -> str:
    metrics = (
        ("Perplexity", "perplexity", "lower is better", "{:.3f}"),
        ("Generation", "generation_tokens_per_second", "end-to-end tok/s", "{:.2f}"),
        ("8K logical state", "context_8k_logical_state_mib", "MiB · lower is better", "{:.2f}"),
    )
    body = ['<text x="44" y="48" class="heading">Ratio trade-offs on one fixed experiment</text>']
    for panel, (title, key, subtitle, formatter) in enumerate(metrics):
        x0 = 44 + panel * 386
        values = [row[key] for row in analysis["ratios"]]
        maximum = max(values)
        body.append(f'<text x="{x0}" y="90" class="value">{title}</text>')
        body.append(f'<text x="{x0}" y="112" class="label muted">{subtitle}</text>')
        for index, row in enumerate(analysis["ratios"]):
            y = 144 + index * 72
            width = 260 * row[key] / maximum
            body.append(f'<text x="{x0}" y="{y + 21}" class="label">{row["ratio"]}</text>')
            body.append(f'<rect x="{x0 + 48}" y="{y}" width="260" height="30" rx="8" fill="#e9edf2"/>')
            body.append(
                f'<rect x="{x0 + 48}" y="{y}" width="{width:.2f}" height="30" rx="8" '
                f'fill="{COLORS[row["ratio"]]}"/>'
            )
            body.append(
                f'<text x="{x0 + 318}" y="{y + 21}" class="value">{formatter.format(row[key])}</text>'
            )
    return _svg_shell(
        1200, 390, "Ratio trade-offs",
        "Three panels compare perplexity, measured sampled-generation throughput, and 8K logical state.",
        "\n".join(body),
    )


def runtime_comparison_svg(analysis: dict) -> str:
    comparison = analysis["runtime_comparison"]
    gpu = comparison["cuda"]["median_tokens_per_second"]
    cpu = comparison["cpu"]["median_tokens_per_second"]
    maximum = max(gpu, cpu)
    body = [
        '<text x="42" y="48" class="heading">1:3 local serving · matched 48-token protocol</text>',
        '<text x="42" y="78" class="label muted">End-to-end speed includes prefill, sampling, and recurrent decode.</text>',
    ]
    for index, (label, value, color, device) in enumerate((
        ("RTX 5070 · bf16", gpu, "#168aad", comparison["cuda"]["device"]),
        ("Ryzen 7700 · fp32", cpu, "#25a67a", comparison["cpu"]["device"]),
    )):
        y = 124 + index * 104
        body.append(f'<text x="42" y="{y}" class="value">{label}</text>')
        body.append(f'<rect x="42" y="{y + 18}" width="560" height="42" rx="11" fill="#e9edf2"/>')
        body.append(
            f'<rect x="42" y="{y + 18}" width="{560 * value / maximum:.2f}" height="42" '
            f'rx="11" fill="{color}"/>'
        )
        body.append(f'<text x="620" y="{y + 48}" class="value">{value:.2f} tok/s</text>')
        body.append(f'<text x="42" y="{y + 82}" class="label muted">{html.escape(device)}</text>')
    return _svg_shell(
        840, 360, "CPU and GPU generation comparison",
        "Protocol-matched local 1:3 end-to-end generation throughput on an RTX 5070 and Ryzen 7700.",
        "\n".join(body),
    )


def state_crossover_svg(analysis: dict) -> str:
    width, height = 1000, 500
    left, top, plot_w, plot_h = 84, 94, 850, 330
    contexts = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
    maximum_mib = 66

    def x_pos(context: float) -> float:
        return left + (math.log2(context) - 5) / 8 * plot_w

    def y_pos(mib: float) -> float:
        return top + plot_h - mib / maximum_mib * plot_h

    body = [
        '<text x="42" y="46" class="heading">Logical inference state changes winner near 260 tokens</text>',
        '<text x="42" y="72" class="label muted">Exact bf16 state equations: fixed Mamba state + attention KV bytes per token.</text>',
    ]
    for value in (0, 16, 32, 48, 64):
        y = y_pos(value)
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        body.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" class="label muted">{value}</text>')
    for context in contexts:
        x = x_pos(context)
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>')
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="label muted">{context if context < 1024 else str(context // 1024) + "K"}</text>')
    for ratio in RATIOS:
        curve = analysis["state_curves"][ratio]
        points = []
        for context in contexts:
            mib = (curve["fixed_mamba_bytes"] + curve["attention_kv_bytes_per_token"] * context) / 2**20
            points.append(f"{x_pos(context):.2f},{y_pos(mib):.2f}")
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[ratio]}" '
            f'stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    crossover = analysis["calculations"]["state_crossover_1_3_vs_1_15_tokens"]
    cross_x = x_pos(crossover)
    body.append(f'<line x1="{cross_x:.2f}" y1="{top}" x2="{cross_x:.2f}" y2="{top + plot_h}" stroke="#182033" stroke-width="2" stroke-dasharray="6 6"/>')
    body.append(f'<text x="{cross_x + 10:.2f}" y="{top + 20}" class="value">≈ {crossover:.0f} tokens</text>')
    for index, ratio in enumerate(RATIOS):
        legend_x = 620 + index * 105
        body.append(f'<circle cx="{legend_x}" cy="48" r="6" fill="{COLORS[ratio]}"/>')
        body.append(f'<text x="{legend_x + 11}" y="53" class="label">{ratio}</text>')
    body.append('<text x="24" y="260" class="label muted" transform="rotate(-90 24 260)">Logical state (MiB)</text>')
    body.append('<text x="509" y="486" text-anchor="middle" class="label muted">Cached context tokens · log₂ scale</text>')
    return _svg_shell(
        width, height, "Logical state crossover",
        "The 1:3 variant uses less state at very short context, while 1:15 uses less after about 260 tokens.",
        "\n".join(body),
    )
