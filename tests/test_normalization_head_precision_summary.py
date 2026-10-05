"""Adversarial standard-library evidence tests; no CUDA or private binaries."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import summarize_normalization_head_precision as summary

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/research/normalization-head-protocol-2026-10-05.json"


def identity(shape, dtype="torch.float32"):
    return {"shape": shape, "dtype": dtype, "sha256": "a" * 64}


def layout(shape):
    strides, current = [], 1
    for size in reversed(shape):
        strides.append(current)
        current *= size
    return {"shape": shape, "stride": strides[::-1], "storage_offset": 0, "dtype": "torch.float32"}


def comparison(shape, actual="torch.float32", reference="torch.float32", *, error=0.):
    sample = {"coordinate": [0] * len(shape), "actual": error, "reference": 0., "absolute_error": error, "tolerance_ratio": error / 3e-5}
    failed = error > 3e-5
    return {"finite": True, "passed": not failed, "shape": shape, "actual_dtype": actual, "reference_dtype": reference,
            "comparison_dtype": "torch.float64" if "torch.float64" in (actual, reference) else "torch.float32",
            "tolerance": deepcopy(summary.base.TOLERANCES["float32"]), "elements": math.prod(shape), "nonzero_error_count": int(error > 0),
            "violation_count": int(failed), "max_absolute_error": error, "max_tolerance_ratio": error / 3e-5,
            "first_nonzero": deepcopy(sample) if error else None, "first_violation": deepcopy(sample) if failed else None,
            "worst_absolute_error": deepcopy(sample), "worst_tolerance_ratio": deepcopy(sample), "violation_samples": [deepcopy(sample)] if failed else []}


def stages(cfg, batch, length):
    rows = [{"layer": layer, "stage": stage, **comparison(shape)} for layer, stage, shape in summary.isolation.stage_inventory(cfg, batch, length)]
    rows = [{**{key: value for key, value in row.items() if key not in summary.SAMPLES}, "sample_policy": "passing_compact", "first_nonzero_coordinate": None} for row in rows]
    return {"stages": rows, "first_nonzero": None, "first_violation": None, "passed": True}


def state_identities(cfg, batch, length):
    return [{"field": name, **identity(shape, dtype), "nonzero_elements": 1} for name, (shape, dtype) in summary.isolation.state_inventory(cfg, batch, length).items()]


def state_bundle(cfg, batch, length):
    return {"tolerance": deepcopy(summary.base.TOLERANCES["float32"]), "fields": [{"field": name, **comparison(shape)}
            for name, (shape, _dtype) in summary.isolation.state_inventory(cfg, batch, length).items()], "passed": True}


def effect(targets, batch, length, start=0):
    zeros = {"mean": 0., "minimum": 0., "maximum": 0., "mean_absolute": 0., "p95_absolute": 0.}
    ones = {key: 1. for key in zeros}
    return {"scope": "descriptive arithmetic effects; no quality-equivalence cutoff", "scored_tokens": len(targets), "position_start": start,
            "position_stop_exclusive": start + length, "metric_dtype": "torch.float32", "actual_mean_nll": 1., "reference_mean_nll": 1., "mean_nll_delta": 0.,
            "true_next_token_logprob_delta": zeros, "greedy_agreement": 1., "greedy_disagreements": 0,
            "actual_top1_top2_margin": deepcopy(ones), "reference_top1_top2_margin": deepcopy(ones),
            "per_token": {"target_ids": targets, "actual_logprob": [-1.] * len(targets), "reference_logprob": [-1.] * len(targets),
                          "logprob_delta": [0.] * len(targets), "greedy_agreement": [True] * len(targets)},
            "batch_size": batch, "positions_per_row": length, "flatten_order": "batch-major; position span describes each row", "targets_sha256": summary.target_hash(targets)}


def route_paths(cfg, batch, length, tokenwise=False):
    positions = range(length) if tokenwise else (0,)
    scans, attention = [], []
    for position in positions:
        for layer, kind in enumerate(cfg["layer_types"]):
            if kind == "mamba":
                scans.append({"layer": layer, "position": position, "length": 1 if tokenwise else length, "stateful": tokenwise,
                              "path": "torch.chunked_ssd" if tokenwise else "torch.quadratic_ssd", "backend": "reference", "chunk_size": 128})
            else:
                query, keys = (1, position + 1) if tokenwise else (length, length)
                attention.append({"layer": layer, "position": position, "query_length": query, "key_length": keys, "is_causal": position == 0,
                                  "mask_present": False, "mask_shape": None, "operand_layout": {name: layout([batch, cfg["d_model"] // cfg["head_dim"], query if name == "q" else keys, cfg["head_dim"]]) for name in ("q", "k", "v")}})
    return {"scan": scans, "attention": attention}


def prefix(cfg, batch, length, prefix_length, targets):
    suffix = length - prefix_length
    output_shape = [batch, suffix, cfg["vocab_size"]]
    return {"prefix_position": prefix_length, "suffix_length": suffix, "synthetic": False, "prefix_unchanged": True,
            "prefix_fields": state_identities(cfg, batch, prefix_length), "prefix_logits": identity([batch, 1, cfg["vocab_size"]]),
            "prefix_comparisons": {anchor: comparison([batch, 1, cfg["vocab_size"]]) for anchor in summary.ANCHORS},
            "suffix_logits": {route: identity(output_shape) for route in summary.SUFFIX_ROUTES},
            "suffix_comparisons": {route: {anchor: comparison(output_shape) for anchor in summary.ANCHORS} for route in summary.SUFFIX_ROUTES},
            "route_comparisons": {route: comparison(output_shape) for route in ("chunked128_vs_one_shot", "tokenwise_vs_one_shot")},
            "retained_state": {route: {anchor: state_bundle(cfg, batch, length) for anchor in summary.ANCHORS} for route in summary.SUFFIX_ROUTES},
            "final_positions": {route: length for route in summary.SUFFIX_ROUTES}, "suffix_state_fields": {route: state_identities(cfg, batch, length) for route in summary.SUFFIX_ROUTES},
            "chunked_schedule": [min(128, suffix - start) for start in range(0, suffix, 128)],
            "next_token_effects_vs_original": {route: effect(summary.suffix_targets(targets, batch, length, prefix_length), batch, suffix, prefix_length) for route in summary.SUFFIX_ROUTES}, "passed": True}


def cell(cell_id, cfg, batch, length, prefix_length, targets):
    shape = [batch, length, cfg["vocab_size"]]
    return {"cell_id": cell_id, "normalization_fp64_coefficients": cell_id[1] == "1", "head_fp64_matmul": cell_id[3] == "1", "execution_status": "completed",
            "whole_model_comparisons": {key: comparison(shape) for key in summary.ENDPOINTS}, "stage_comparisons": {key: stages(cfg, batch, length) for key in summary.PAIRINGS},
            "retained_state": {key: state_bundle(cfg, batch, length) for key in summary.ANCHORS}, "stateful_full_retained_state": {key: state_bundle(cfg, batch, length) for key in summary.ANCHORS[1:]},
            "actual_prefix": prefix(cfg, batch, length, prefix_length, targets), "final_positions": {"tokenwise": length, "stateful_full": length},
            "logits": {key: identity(shape) for key in ("full", "tokenwise", "chunked128")},
            "next_token_effects": {key: effect(targets, batch, length) for key in ("full_vs_original", "tokenwise_vs_original", "tokenwise_vs_own_full")},
            "actual_paths": {"full": route_paths(cfg, batch, length), "tokenwise": route_paths(cfg, batch, length, True)},
            "backward_executed": False, "BF16_executed": False, "whole_model_passed": True, "all_recorded_comparisons_passed": True}


def norm_probe(site):
    name, input_stage, output_stage, shape = site
    stat_shape = [*shape[:2], 1]
    checks = {key: comparison(shape, reference="torch.float64" if key.endswith("vs_fp64") else "torch.float32") for key in summary.NORM_OUTPUT_CHECKS}
    stats = {key: comparison(shape if key.startswith("squared_") else stat_shape, actual="torch.float64" if key.startswith("mean_square_n1_") else "torch.float32", reference="torch.float64") for key in summary.NORM_STAT_CHECKS}
    return {"module": name, "source_input_stage": input_stage, "source_output_stage": output_stage, "input_layout": layout(shape), "output_layout": layout(shape),
            "input": identity(shape), "observed_output": identity(shape), "weight": identity([shape[-1]]), "epsilon": 1e-5, "replay_layout": layout(shape),
            "tokenwise_shapes": [shape[0], 1, shape[-1]], "identical_operands": True, "input_unchanged": True,
            "oracle": "independent detached CPU NumPy FP64 feature sum / reciprocal sqrt / weight multiplication", "output_checks": checks, "statistic_checks": stats,
            "statistics": {route: {key: identity(shape if key == "squared" else stat_shape, "torch.float64" if route == "fp64_oracle" or route.startswith("n1_") and key == "mean_square" else "torch.float32")
                                   for key in ("squared", "mean_square", "coefficient")} for route in ("original_full", "original_tokenwise", "n1_full", "n1_tokenwise", "fp64_oracle")},
            "passed": True, "exploratory": True}


def head(cfg, batch, length):
    input_shape, shape = [batch, length, cfg["d_model"]], [batch, length, cfg["vocab_size"]]
    return {"source_input_stage": [None, "lm_head.input"], "source_output_stage": [None, "lm_head.output"], "input_layout": layout(input_shape), "output_layout": layout(shape),
            "input": identity(input_shape), "observed_output": identity(shape), "weight": identity([cfg["vocab_size"], cfg["d_model"]]), "bias": None,
            "replay_layout": layout(input_shape), "tokenwise_shapes": [batch, 1, cfg["d_model"]], "identical_operands": True, "input_unchanged": True,
            "oracle": "independent detached CPU NumPy FP64 matrix multiplication and optional bias addition", "oracle_output": identity(shape, "torch.float64"),
            "checks": {key: comparison(shape, reference="torch.float64" if key.endswith("vs_fp64") else "torch.float32") for key in summary.HEAD_CHECKS},
            "outputs": {key: identity(shape) for key in ("h0_full", "h0_tokenwise", "h1_full", "h1_tokenwise")}, "passed": True, "exploratory": True}


@pytest.fixture
def toy():
    # Synthetic metadata exercises every site/route without allocating tensors.
    cfg = {"d_model": 8, "expand": 2, "d_state": 4, "mamba_headdim": 4, "d_conv": 4, "head_dim": 4, "vocab_size": 16,
           "mlp_ratio": 2., "mlp_multiple_of": 4, "mlp_on_every_layer": True, "n_layers": 2, "layer_types": ["mamba", "attention"]}
    batch, length, prefix_length, targets = 2, 4, 2, [1, 2, 3, 4, 5, 6, 7, 8]
    shared = {"case_id": "toy", "batch_size": batch, "length": length, "prefix_length": prefix_length, "gradient_check": True,
              "targets_sha256": summary.target_hash(targets), "windows": [{"targets_sha256": summary.target_hash(targets[row * length:(row + 1) * length])} for row in range(batch)]}
    cells = [cell(name, cfg, batch, length, prefix_length, targets) for name in summary.CELLS]
    policy = {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "deterministic_algorithms": False, "deterministic_warn_only": False,
              "cudnn_deterministic": False, "cudnn_benchmark": False, "cublas_workspace_config": None}
    shape = [batch, length, cfg["vocab_size"]]
    original = {"logits": identity(shape), "loss": identity([]), "stateful_logits": identity(shape), "state_fields": state_identities(cfg, batch, length),
                "stateful_vs_stateless": comparison(shape), "repeat": {"exact_equal": True, "comparison": comparison(shape)}, "actual_paths": route_paths(cfg, batch, length)}
    previous = {"weights_sha256": "a" * 64, "whole_model_comparisons": {key: comparison(shape) for key in summary.PRIOR_MAP.values()}}
    control = {"executed": True, "exact_fields": summary.PRIOR_FIELDS, "comparisons": {key: True for key in summary.PRIOR_MAP}, "exact_equal": True,
               "scope": "N0H0 repeated historical full/tokenwise/stateful endpoint statistics; separate from tolerance acceptance"}
    operators = [{"stage": "original.stateful_full", "layer": 0, "length": length, "calls": 1, "path": "torch.chunked_ssd", "autocast": False, "input_dtypes": ["torch.float32"] * 7}]
    for name in summary.CELLS:
        for stage in ("stateful_full", "full_chunked128", "prefix_prefill", "suffix_one_shot", "suffix_chunked128", "suffix_tokenwise"):
            span, calls = (1, length - prefix_length) if stage == "suffix_tokenwise" else (prefix_length if stage == "prefix_prefill" else length - prefix_length if stage.startswith("suffix_") else length, 1)
            operators.append({"stage": name + "." + stage, "layer": 0, "length": span, "calls": calls, "path": "torch.chunked_ssd", "autocast": False, "input_dtypes": ["torch.float32"] * 7})
    case = {"ratio": "1:3", **shared, "weights_sha256": "a" * 64, "execution_status": "completed", "cells": cells, "failed_cell": None,
            "frozen_norms": [norm_probe(site) for site in summary.norm_inventory(cfg, batch, length)], "frozen_head": head(cfg, batch, length),
            "backward_executed": False, "BF16_executed": False, "production_defaults_changed": False, "optimizer_executed": False,
            "runtime_policy": {"before": deepcopy(policy), "active": deepcopy(policy), "restored": deepcopy(policy), "restoration_exact": True},
            "tied_weight_integrity": {"before": {"same_parameter": True, "same_storage": True}, "after": {"same_parameter": True, "same_storage": True},
                                      "parameter_identity_unchanged": True, "data_ptr_unchanged": True, "passed": True},
            "post_model_weights_unchanged": True, "operator_observations": operators, "attribution_control_stable": True, "all_recorded_comparisons_passed": True,
            "processing_seconds": 1., "cuda_peak_allocated_bytes": 1024, "original_anchor": original,
            "decay_baseline_anchor": {"logits": identity(shape), "state_fields": state_identities(cfg, batch, length), "cell": "N0H0"}, "prior_reproduction": control}
    return case, shared, {"ratio": "1:3", "model_config": cfg}, previous, {key: value for key, value in policy.items() if "allow_tf32" not in key}


def validate(toy):
    summary.validate_case(*toy)


def test_complete_metadata_exercises_all_four_cells_and_independent_replays(toy):
    validate(toy)
    case, _, cp, _, _ = toy
    compact = summary.compact_case(case, cp["model_config"])
    assert len(compact["cells"]) == 4 and compact["missing_cells"] == []
    assert compact["frozen_comparison_counts"]["head"] == {"passed": 8, "total": 8, "worst_tolerance_ratio": 0.}
    assert compact["frozen_comparison_counts"]["normalization_statistics"]["total"] == 6 * 9
    assert not compact["frozen_coverage"]["independent_repetitions_per_cell"]
    assert "per_token" not in compact["cells"][0]["next_token_effects"]["full_vs_original"]
    assert compact["cells"][0]["actual_paths"]["tokenwise"]["scan_calls"] == 4


@pytest.mark.parametrize("change", [
    lambda case: case["cells"].pop(),
    lambda case: case["cells"].__setitem__(1, deepcopy(case["cells"][0])),
    lambda case: case["cells"][0]["whole_model_comparisons"].pop("stateful_full_vs_original"),
    lambda case: case["cells"][0]["stateful_full_retained_state"].pop("original"),
    lambda case: case["cells"][0]["stage_comparisons"].pop(summary.PAIRINGS[-1]),
    lambda case: case["cells"][0]["stage_comparisons"][summary.PAIRINGS[0]]["stages"].pop(),
    lambda case: case["frozen_norms"].pop(),
    lambda case: case["frozen_head"]["checks"].pop(summary.HEAD_CHECKS[-1]),
    lambda case: case["frozen_head"].__setitem__("oracle", "shared Torch double linear"),
    lambda case: case["frozen_head"]["oracle_output"].__setitem__("dtype", "torch.float32"),
    lambda case: case["cells"][0]["actual_paths"]["tokenwise"]["scan"].pop(),
    lambda case: case["operator_observations"].pop(),
    lambda case: case["cells"][0]["actual_paths"]["tokenwise"]["attention"][2].__setitem__("key_length", 1),
    lambda case: case["cells"][0]["actual_prefix"].__setitem__("synthetic", True),
    lambda case: case["cells"][0]["actual_prefix"]["final_positions"].__setitem__("tokenwise", 3),
    lambda case: case["cells"][0]["actual_prefix"]["suffix_comparisons"]["one_shot"].pop("original"),
    lambda case: next(row for row in case["cells"][0]["actual_prefix"]["prefix_fields"] if row["field"].endswith(".ssm")).__setitem__("nonzero_elements", 0),
    lambda case: case["runtime_policy"]["active"].__setitem__("cuda_matmul_allow_tf32", True),
    lambda case: case["tied_weight_integrity"].__setitem__("data_ptr_unchanged", False),
    lambda case: case.__setitem__("post_model_weights_unchanged", False),
    lambda case: case["cells"][0]["next_token_effects"]["full_vs_original"]["per_token"]["target_ids"].__setitem__(0, 15),
    lambda case: case["frozen_norms"][0]["statistic_checks"]["mean_square_n1_full_vs_fp64"].__setitem__("actual_dtype", "torch.float32"),
    lambda case: case["frozen_norms"][0]["statistics"]["n1_full"]["coefficient"].__setitem__("dtype", "torch.float64"),
    lambda case: case["frozen_norms"][0]["replay_layout"].__setitem__("storage_offset", 1),
    lambda case: case["frozen_head"]["checks"][summary.HEAD_CHECKS[0]]["tolerance"].__setitem__("atol", .001),
    lambda case: case["frozen_head"]["checks"][summary.HEAD_CHECKS[0]].__setitem__("elements", True),
])
def test_missing_tampered_cells_anchors_precision_prefix_paths_rejected(toy, change):
    change(toy[0])
    with pytest.raises(ValueError):
        validate(toy)


def test_unknown_pass_leaf_cannot_inflate_explicit_frozen_or_case_counts(toy):
    case = toy[0]
    before = summary.isolation.count(summary.case_leaves(case))
    case["frozen_head"]["synthetic_leaf"] = {"passed": True, "finite": True, "max_tolerance_ratio": 0.}
    assert summary.isolation.count(summary.case_leaves(case)) == before
    with pytest.raises(ValueError, match="frozen head schema"):
        validate(toy)


def test_hidden_failed_intermediate_stage_remains_separate_from_passing_scores(toy):
    case = toy[0]
    trace = case["cells"][1]["stage_comparisons"]["tokenwise_vs_original"]
    row = trace["stages"][-3]  # final normalization, before the passing head scores
    failed = {"layer": row["layer"], "stage": row["stage"], **comparison(row["shape"], error=6e-5), "sample_policy": "full_failed"}
    trace["stages"][-3] = failed
    trace["first_nonzero"] = {"layer": failed["layer"], "stage": failed["stage"], "sample": deepcopy(failed["first_nonzero"])}
    trace["first_violation"] = {"layer": failed["layer"], "stage": failed["stage"], "sample": deepcopy(failed["first_violation"])}
    trace["passed"] = case["cells"][1]["all_recorded_comparisons_passed"] = case["all_recorded_comparisons_passed"] = False
    validate(toy)
    assert case["cells"][1]["whole_model_passed"]
    compact = summary.compact_case(case, toy[2]["model_config"])
    assert compact["cells"][1]["stage_comparisons"]["tokenwise_vs_original"]["stages"][-3]["violation_samples"] == failed["violation_samples"]
    trace["first_violation"] = None
    with pytest.raises(ValueError):
        validate(toy)


def test_failed_prior_exact_control_is_not_relabelled_tolerance_failure_or_pass(toy):
    toy[3]["whole_model_comparisons"][next(iter(summary.PRIOR_MAP.values()))]["max_absolute_error"] = 1e-8
    control = toy[0]["prior_reproduction"]
    control["comparisons"][next(iter(summary.PRIOR_MAP))] = control["exact_equal"] = toy[0]["attribution_control_stable"] = toy[0]["all_recorded_comparisons_passed"] = False
    validate(toy)
    assert all(cell["whole_model_passed"] for cell in toy[0]["cells"])
    control["exact_equal"] = True
    with pytest.raises(ValueError):
        validate(toy)


def test_partial_frozen_failure_retains_completed_n0_cell_without_future_passes(toy):
    case = toy[0]
    case["cells"] = case["cells"][:1]
    case["frozen_norms"] = case["frozen_norms"][:2]
    case["frozen_head"] = None
    case["operator_observations"] = [row for row in case["operator_observations"] if row["stage"].startswith(("original.", "N0H0."))]
    case.update(execution_status="incomplete", failed_cell="N0H0", reason="MemoryError: bounded frozen replay", all_recorded_comparisons_passed=False)
    validate(toy)
    compact = summary.compact_case(case, toy[2]["model_config"])
    assert compact["missing_cells"] == ["N1H0", "N0H1", "N1H1"]
    assert compact["frozen_comparison_counts"]["head"]["total"] == 0
    assert compact["frozen_coverage"]["executed_norm_sites"] == 2


def test_post_case_restoration_failure_keeps_four_completed_cells_in_incomplete_workflow(toy):
    case = toy[0]
    case.update(execution_status="incomplete", failed_cell=None, reason="model weights changed during diagnostic",
                post_model_weights_unchanged=False, all_recorded_comparisons_passed=False)
    validate(toy)
    raw = {"execution_status": "incomplete", "cases": [case], "reason": "RuntimeError: model weights changed during diagnostic",
           "failed_case": {"ratio": case["ratio"], "case_id": case["case_id"], "cell": "N1H1", "stage": "full/tokenwise/stateful/continuation"}}
    summary.validate_coverage(raw, [(case["ratio"], case["case_id"])])
    assert len(summary.compact_case(case, toy[2]["model_config"])["cells"]) == 4
    # This narrowly described restoration edge cannot excuse an ordinary
    # mismatched failed cell after otherwise unchanged model/runtime state.
    case["post_model_weights_unchanged"] = True
    with pytest.raises(ValueError, match="failed cell differs"):
        summary.validate_coverage(raw, [(case["ratio"], case["case_id"])])


@pytest.mark.parametrize("change", [
    lambda value: value["worst_tolerance_ratio"].__setitem__("tolerance_ratio", 0.1),
    lambda value: value["worst_absolute_error"]["coordinate"].__setitem__(0, 999),
    lambda value: value.__setitem__("max_absolute_error", float("inf")),
    lambda value: value.__setitem__("reference_dtype", "torch.bfloat16"),
])
def test_coordinate_arithmetic_and_dtype_cannot_be_resigned(change):
    value = comparison([2, 3, 4], reference="torch.float64")
    change(value)
    with pytest.raises(ValueError):
        summary.detailed(value, [2, 3, 4], reference="torch.float64")


def test_exact_six_case_coverage_rejects_duplicates_and_missing_completed_cells():
    expected = [(ratio, case_id) for ratio in summary.RATIOS for case_id in summary.CASE_IDS]
    report = {"cases": [{"ratio": ratio, "case_id": case_id, "execution_status": "completed"} for ratio, case_id in expected], "execution_status": "completed", "reason": None, "failed_case": None}
    summary.validate_coverage(report, expected)
    report["cases"][1] = deepcopy(report["cases"][0])
    with pytest.raises(ValueError):
        summary.validate_coverage(report, expected)


def test_budgeted_final_audit_null_groups_and_actual_hash_failure_remain_honest():
    registry = {group: [(f"{group}.txt", "a" * 64)] for group in summary.AUDIT_GROUPS}
    files = [{"group": group, "path": entries[0][0], "path_scope": "repository-relative", "expected_sha256": "a" * 64, "actual_sha256": "a" * 64, "passed": True} for group, entries in registry.items()]
    raw = {"post_execution_integrity": {group: True for group in registry}, "post_execution_audit_files": files}
    assert summary.final_audit(raw, registry, True)
    raw["post_execution_integrity"]["sources"] = None
    assert not summary.final_audit(raw, registry, False)
    with pytest.raises(ValueError):
        summary.final_audit(raw, registry, True)
    raw["post_execution_integrity"]["sources"] = False
    files[-1]["actual_sha256"], files[-1]["passed"] = "b" * 64, False
    assert not summary.final_audit(raw, registry, False)
    files[-1]["passed"] = True
    with pytest.raises(ValueError):
        summary.final_audit(raw, registry, False)


def test_lf_crlf_equivalence_allows_only_source_text(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"a = 1\r\nb = 2\r\n")
    recorded = hashlib.sha256(b"a = 1\nb = 2\n").hexdigest()
    summary.isolation.source_check({"source.py": recorded}, tmp_path, {"source.py"})
    path.write_bytes(b"a = 3\nb = 2\n")
    with pytest.raises(ValueError):
        summary.isolation.source_check({"source.py": recorded}, tmp_path, {"source.py"})


def test_atomic_publication_is_lf_and_exclusive(tmp_path):
    path = tmp_path / "new.json"
    summary.base.write_summary(path, {"status": "incomplete", "certified": False})
    before = path.read_bytes()
    assert before.endswith(b"\n") and b"\r" not in before
    with pytest.raises(ValueError, match="exists"):
        summary.base.write_summary(path, {"certified": True})
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def projected_source(toy):
    case, _shared, checkpoint, _previous, _policy = toy
    return {"schema": 1, "kind": summary.KIND + "_summary", "certified": False, "cases": [summary.compact_case(case, checkpoint["model_config"])],
            "gate_counts": {"example": summary.isolation.count(summary.case_leaves(case))}, "coverage": {"expected_cells": 24, "completed_cells": 4},
            "protocol": {"tolerances": summary.base.TOLERANCES}, "declaration": {"path": "declaration.json", "sha256": "b" * 64},
            "raw_report": {"path": "raw.json", "sha256": "c" * 64}, "post_execution_integrity": {group: True for group in summary.AUDIT_GROUPS},
            "execution_identities_available": True, "final_audit_passed": True}


def test_website_projection_keeps_gates_anchors_counts_and_source_binding(toy):
    validate(toy)
    full = projected_source(toy)
    before = deepcopy(full)
    source = {"path": "docs/research/full-summary.json", "sha256": "d" * 64}
    view = summary.project_website_view(full, source)
    assert full == before  # A new artifact never mutates the full source.
    assert view["kind"] == summary.KIND + "_website_view" and view["source_summary"] == source
    for key in ("gate_counts", "coverage", "protocol", "declaration", "raw_report", "post_execution_integrity", "final_audit_passed"):
        assert view[key] == full[key]
    original, projected = full["cases"][0], view["cases"][0]
    assert projected["prior_reproduction"] == original["prior_reproduction"]
    for cell, current in zip(original["cells"], projected["cells"]):
        assert current["whole_model_comparisons"] == cell["whole_model_comparisons"]
        assert current["next_token_effects"] == cell["next_token_effects"]
        for name, pair in cell["stage_comparisons"].items():
            projected_pair = current["stage_comparisons"][name]
            assert projected_pair["counts"] == pair["counts"]
            assert len(projected_pair["stages"]) < pair["counts"]["total"]
            assert {(row["layer"], row["stage"]) for row in projected_pair["stages"]} == {(0, "block_output"), (1, "block_output"), (None, "final_norm"), (None, "lm_head.output")}
            assert projected_pair["projection_coverage"]["full_stage_count"] == len(pair["stages"])
        for bundle in current["retained_state"].values():
            for leaf in bundle["fields"]:
                assert not summary.SAMPLES.intersection(leaf)
                assert {"tolerance", "finite", "shape", "actual_dtype", "reference_dtype", "elements", "max_absolute_error", "max_tolerance_ratio", "nonzero_error_count", "violation_count"} <= set(leaf)
    assert projected["frozen_head"]["checks"][summary.HEAD_CHECKS[-1]]["reference_dtype"] == "torch.float64"
    assert not summary.SAMPLES.intersection(projected["frozen_head"]["checks"][summary.HEAD_CHECKS[-1]])


def test_website_projection_retains_every_failed_stage_memory_and_frozen_sample(toy):
    case = toy[0]
    trace = case["cells"][1]["stage_comparisons"]["tokenwise_vs_original"]
    original = trace["stages"][3]
    negative = {"layer": original["layer"], "stage": original["stage"], **comparison(original["shape"], error=6e-5), "sample_policy": "full_failed"}
    trace["stages"][3] = negative
    trace.update(passed=False, first_nonzero={"layer": negative["layer"], "stage": negative["stage"], "sample": deepcopy(negative["first_nonzero"])},
                 first_violation={"layer": negative["layer"], "stage": negative["stage"], "sample": deepcopy(negative["first_violation"])})
    bundle = case["cells"][1]["stateful_full_retained_state"]["original"]
    memory = {"field": bundle["fields"][0]["field"], **comparison(bundle["fields"][0]["shape"], error=6e-5)}
    bundle["fields"][0], bundle["passed"] = memory, False
    norm = case["frozen_norms"][0]
    frozen = comparison(norm["input"]["shape"], error=6e-5)
    norm["output_checks"][summary.NORM_OUTPUT_CHECKS[0]], norm["passed"] = frozen, False
    case["cells"][1]["whole_model_passed"] = case["cells"][1]["all_recorded_comparisons_passed"] = case["all_recorded_comparisons_passed"] = False
    validate(toy)
    full = projected_source(toy)
    view = summary.project_website_view(full, {"path": "full.json", "sha256": "e" * 64})
    projected = view["cases"][0]
    full_pair = full["cases"][0]["cells"][1]["stage_comparisons"]["tokenwise_vs_original"]
    pair = projected["cells"][1]["stage_comparisons"]["tokenwise_vs_original"]
    assert pair["first_nonzero"] == full_pair["first_nonzero"] and pair["first_violation"] == full_pair["first_violation"]
    assert next(row for row in pair["stages"] if not row["passed"]) == full_pair["stages"][3]
    assert projected["cells"][1]["stateful_full_retained_state"]["original"]["fields"][0] == {**full["cases"][0]["cells"][1]["stateful_full_retained_state"]["original"]["fields"][0], "tolerance": summary.base.TOLERANCES["float32"]}
    assert projected["frozen_norms"][0]["output_checks"][summary.NORM_OUTPUT_CHECKS[0]] == frozen
    assert view["gate_counts"] == full["gate_counts"]


def test_website_projection_retains_first_drift_stage_even_when_passing_and_not_plotted(toy):
    case = toy[0]
    pair = case["cells"][0]["stage_comparisons"]["tokenwise_vs_own_full"]
    detail = comparison(pair["stages"][0]["shape"], error=1e-8)
    pair["stages"][0] = {"layer": None, "stage": "embedding", **{key: value for key, value in detail.items() if key not in summary.SAMPLES}, "sample_policy": "passing_compact", "first_nonzero_coordinate": detail["first_nonzero"]["coordinate"]}
    pair["first_nonzero"] = {"layer": None, "stage": "embedding", "sample": detail["first_nonzero"]}
    validate(toy)
    full = projected_source(toy)
    view = summary.project_website_view(full, {"path": "full.json", "sha256": "e" * 64})
    result = view["cases"][0]["cells"][0]["stage_comparisons"]["tokenwise_vs_own_full"]
    assert result["stages"][0]["stage"] == "embedding" and result["first_nonzero"] == pair["first_nonzero"]
    assert result["counts"]["total"] == full["cases"][0]["cells"][0]["stage_comparisons"]["tokenwise_vs_own_full"]["counts"]["total"]


def test_website_view_revalidates_source_summary_instead_of_trusting_resigned_gates(toy, tmp_path, monkeypatch):
    full = projected_source(toy)
    full["gate_counts"]["example"]["passed"] = 0
    path = tmp_path / "source-summary.json"
    path.write_text(json.dumps(full))
    expected = projected_source(toy)
    monkeypatch.setattr(summary, "build_summary", lambda *args, **kwargs: expected)
    with pytest.raises(ValueError, match="independently validated raw evidence"):
        summary.build_website_view(path, root=tmp_path)


def test_website_view_compact_atomic_lf_no_overwrite(toy, tmp_path):
    view = summary.project_website_view(projected_source(toy), {"path": "full.json", "sha256": "e" * 64})
    path = tmp_path / "view.json"
    summary.write_website_view(path, view)
    before = path.read_bytes()
    assert before.count(b"\n") == 1 and b"\r" not in before
    assert json.loads(before) == view
    with pytest.raises(ValueError, match="exists"):
        summary.write_website_view(path, view)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_import_does_not_load_torch_numpy_or_producer():
    code = "import sys; from scripts import summarize_normalization_head_precision; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'scripts.study_normalization_head_precision' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


@pytest.fixture(scope="module")
def public_declaration():
    if not DECLARATION.exists():
        pytest.skip("new declaration has not been bound")
    declaration, _ = summary.base.read_json(DECLARATION)
    prior_plan, _ = summary.base.read_json(ROOT / declaration["prior_declaration"]["path"])
    prior, _ = summary.base.read_json(ROOT / declaration["prior_report"]["path"])
    return declaration, prior_plan, prior


def test_exact_title_annotation_is_preserved_without_computational_change(public_declaration):
    declaration, prior_plan, prior = public_declaration
    summary.validate_declaration(declaration, prior_plan, prior, ROOT)
    assert declaration["study"] == "Fixed-decay normalization coefficient and final projection precision factorial"
    altered = deepcopy(declaration)
    altered["study"] = "Numerical policy certified"
    with pytest.raises(ValueError, match="declared scope differs: study"):
        summary.validate_declaration(altered, prior_plan, prior, ROOT)


@pytest.mark.parametrize("change", [
    lambda declaration: declaration["cells"].pop(),
    lambda declaration: declaration["shared_inputs"].__setitem__(1, deepcopy(declaration["shared_inputs"][0])),
    lambda declaration: declaration["data"]["tokenizer"].__setitem__("sha256", "f" * 64),
    lambda declaration: declaration["checkpoints"][0].__setitem__("sha256", "f" * 64),
    lambda declaration: declaration["source_sha256"].__setitem__("scripts/study_normalization_head_precision.py", "f" * 64),
    lambda declaration: declaration["observed_inference_policy"].__setitem__("deterministic_algorithms", True),
    lambda declaration: declaration.__setitem__("new_policy", {"passed": True}),
])
def test_public_declaration_priors_sources_and_observed_policy_cannot_change(public_declaration, change):
    declaration, prior_plan, prior = public_declaration
    altered = deepcopy(declaration)
    change(altered)
    with pytest.raises(ValueError):
        summary.validate_declaration(altered, prior_plan, prior, ROOT)


def test_actual_export_when_declared_cuda_report_is_available():
    if not DECLARATION.exists():
        pytest.skip("new declaration has not been bound")
    declared = json.loads(DECLARATION.read_text())
    if not (ROOT / declared["output"]).exists():
        pytest.skip("declared CUDA report has not been published")
    result = summary.build_summary(DECLARATION)
    assert result["certified"] is False and result["execution_identities_available"]
    assert result["coverage"]["expected_cells"] == 24 and not result["coverage"]["backward_executed"]
    if result["execution_status"] == "completed":
        assert result["final_audit_passed"] and result["coverage"]["completed_cells"] == 24 and result["coverage"]["completed_cases"] == 6
        assert result["coverage"]["missing_cells"] == result["coverage"]["missing_cases"] == []
        assert all(len(case["cells"]) == 4 for case in result["cases"])
    assert all(set(cell["whole_model_comparisons"]) == set(summary.ENDPOINTS) for case in result["cases"] for cell in case["cells"])
    source_path = ROOT / "docs/research/normalization-head-summary-2026-10-05.json"
    binding = {"path": source_path.relative_to(ROOT).as_posix(), "sha256": summary.base.file_sha256(source_path)} if source_path.exists() else {"path": "full.json", "sha256": "d" * 64}
    view = summary.project_website_view(result, binding)
    assert view["gate_counts"] == result["gate_counts"] and view["coverage"] == result["coverage"]
    for case, projected in zip(result["cases"], view["cases"]):
        for cell, current in zip(case["cells"], projected["cells"]):
            for name, pair in cell["stage_comparisons"].items():
                actual = current["stage_comparisons"][name]
                assert actual["counts"] == pair["counts"]
                assert [row for row in actual["stages"] if not row["passed"]] == [row for row in pair["stages"] if not row["passed"]]
                assert actual["first_nonzero"] == pair["first_nonzero"] and actual["first_violation"] == pair["first_violation"]
                assert [row["layer"] for row in actual["stages"] if row["stage"] == "block_output"] == list(range(16))
