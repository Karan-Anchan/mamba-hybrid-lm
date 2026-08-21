"""Joined Week 4 and Week 5 analysis contracts."""

import copy

import pytest

from src.eval.week5_report import (
    build_analysis,
    ratio_tradeoffs_svg,
    runtime_comparison_svg,
    state_crossover_svg,
)


def fixtures():
    week4 = {"run_id": "w4", "git": {"dirty": False}, "protocol": {"context_lengths": [512, 8192]}, "variants": []}
    cuda = {
        "run_id": "cuda", "git": {"dirty": False}, "runtime": {"device_name": "GPU"},
        "protocol": {"prompts": ["a"], "temperature": 0.8, "top_k": 40,
                     "max_new_tokens": 48, "seed_by_prompt": [1]}, "variants": [],
    }
    cpu = copy.deepcopy(cuda)
    cpu.update({"run_id": "cpu", "runtime": {"device_name": "CPU"}})
    for index, ratio in enumerate(("1:3", "1:7", "1:15")):
        attention = (4, 2, 1)[index]
        fixed = (5_500_000, 6_300_000, 6_900_000)[index]
        kv_per_token = attention * 1792
        week4["variants"].append({
            "ratio": ratio, "parameters": 52_000_000 + index, "attention_layers": attention,
            "mamba_layers": 16 - attention, "validation": {"perplexity": 26.3 + index * 0.1},
            "inference": [{"context_length": 512}, {
                "context_length": 8192, "logical_state_bytes": fixed + kv_per_token * 8192,
                "attention_kv_bytes": kv_per_token * 8192, "mamba_conv_bytes": 100_000,
                "mamba_ssm_bytes": fixed - 100_000, "prefill_tokens_per_second": 12_000 - index,
                "decode_tokens_per_second": 48 - index,
            }],
            "needle_retrieval": {"matches": 3, "trials": 15},
        })
        cuda["variants"].append({
            "ratio": ratio,
            "summary": {"median_tokens_per_second": 52 - index * 2,
                        "median_decode_tokens_per_second": 51 - index,
                        "median_time_to_first_token_seconds": 0.02,
                        "median_peak_vram_mib": 240 + index,
                        "median_logical_state_mib": 5.7 + index * 0.5,
                        "sample_count": 1},
            "samples": [{"metrics": {"prompt_tokens": 10, "generated_tokens": 48}}],
        })
    cpu["variants"] = [copy.deepcopy(cuda["variants"][0])]
    cpu["variants"][0]["summary"]["median_tokens_per_second"] = 40
    return week4, cuda, cpu


def test_analysis_joins_ratios_and_calculates_state_crossover():
    analysis = build_analysis(*fixtures())
    assert [row["ratio"] for row in analysis["ratios"]] == ["1:3", "1:7", "1:15"]
    assert analysis["calculations"]["cpu_share_of_cuda_1_3_percent"] == pytest.approx(100 * 40 / 52)
    assert analysis["calculations"]["state_crossover_1_3_vs_1_15_tokens"] > 0


def test_analysis_rejects_unmatched_runtime_protocols():
    week4, cuda, cpu = fixtures()
    cpu["protocol"]["max_new_tokens"] = 24
    with pytest.raises(ValueError, match="not matched"):
        build_analysis(week4, cuda, cpu)


def test_all_plots_are_accessible_svg_documents():
    analysis = build_analysis(*fixtures())
    for svg in (ratio_tradeoffs_svg(analysis), runtime_comparison_svg(analysis), state_crossover_svg(analysis)):
        assert svg.startswith("<svg")
        assert "role=\"img\"" in svg
        assert "<title" in svg and "<desc" in svg
