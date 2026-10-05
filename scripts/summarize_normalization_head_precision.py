"""Independent standard-library validation of the declared norm/head factorial.

No producer module, Torch, CUDA, checkpoint binary or corpus binary is loaded.
Original, decay-only and own score/memory anchors remain separate. Missing cells
and exploratory frozen operations cannot become whole-model qualifications.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_checkpoint_numerics as base  # noqa: E402
from scripts import summarize_tokenwise_isolation as isolation  # noqa: E402

KIND = "trained_checkpoint_normalization_head_precision"
RATIOS, CASES, CASE_IDS = isolation.RATIOS, isolation.CASES, isolation.CASE_IDS
CELLS = ["N0H0", "N1H0", "N0H1", "N1H1"]
PAIRINGS = ["full_vs_original", "full_vs_decay_baseline", "tokenwise_vs_own_full", "tokenwise_vs_original", "tokenwise_vs_decay_baseline"]
ENDPOINTS = [*PAIRINGS, "stateful_full_vs_own_full", "stateful_full_vs_original", "stateful_full_vs_decay_baseline", "full_chunked_vs_own_full", "full_chunked_vs_original", "full_chunked_vs_decay_baseline"]
ANCHORS = ["own", "original", "decay_baseline"]
SUFFIX_ROUTES = ["one_shot", "chunked128", "tokenwise"]
NORM_OUTPUT_CHECKS = ["original_full_vs_observed", "original_tokenwise_vs_full", "original_full_vs_fp64", "original_tokenwise_vs_fp64",
                      "n1_full_vs_original_full", "n1_tokenwise_vs_n1_full", "n1_full_vs_fp64", "n1_tokenwise_vs_fp64"]
NORM_STAT_CHECKS = ["squared_fp32_vs_fp64", "mean_square_original_full_vs_fp64", "mean_square_original_tokenwise_vs_fp64",
                    "mean_square_n1_full_vs_fp64", "mean_square_n1_tokenwise_vs_fp64", "coefficient_original_full_vs_fp64",
                    "coefficient_original_tokenwise_vs_fp64", "coefficient_n1_full_vs_fp64", "coefficient_n1_tokenwise_vs_fp64"]
HEAD_CHECKS = ["h0_full_vs_observed", "h0_tokenwise_vs_h0_full", "h0_full_vs_fp64", "h0_tokenwise_vs_fp64",
               "h1_full_vs_h0_full", "h1_tokenwise_vs_h1_full", "h1_full_vs_fp64", "h1_tokenwise_vs_fp64"]
SOURCES = isolation.SOURCES | {"scripts/study_normalization_head_precision.py"}
SAMPLES = {"first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples"}
DETAIL_KEYS = isolation.DETAIL_KEYS
same = isolation.same


def closed(value, expected, label):
    base.require(isinstance(value, dict), label + " must be an object")
    same(sorted(value), sorted(expected), label + " schema differs")


def relative(value):
    base.relative_path(value)
    same(value, PurePosixPath(value).as_posix(), "noncanonical repository-relative path")
    return value


def coordinate(indices, shape):
    base.require(isinstance(indices, list) and len(indices) == len(shape), "coordinate rank differs")
    position = 0
    for index, size in zip(indices, shape):
        base.integer(index, "coordinate")
        base.require(index < size, "coordinate outside tensor")
        position = position * size + index
    return position


def sample(value, shape, maximum_error, maximum_ratio):
    closed(value, ["coordinate", "actual", "reference", "absolute_error", "tolerance_ratio"], "coordinate sample")
    position = coordinate(value["coordinate"], shape)
    for key in ("actual", "reference", "absolute_error", "tolerance_ratio"):
        base.number(value[key], key, None if key in {"actual", "reference"} else 0)
    error = abs(value["actual"] - value["reference"])
    base.require(math.isclose(error, value["absolute_error"], rel_tol=2e-6, abs_tol=1e-10), "coordinate absolute error differs")
    threshold = 3e-5 + 3e-4 * abs(value["reference"])
    base.require(math.isclose(value["absolute_error"] / threshold, value["tolerance_ratio"], rel_tol=2e-6, abs_tol=1e-9), "coordinate tolerance ratio differs")
    base.require(value["absolute_error"] <= maximum_error and value["tolerance_ratio"] <= maximum_ratio, "coordinate exceeds maximum")
    return position


def metric_header(value, shape, *, actual="torch.float32", reference="torch.float32"):
    base.metric(value, "comparison")
    base.require(value["finite"] is True, "nonfinite completed comparison")
    same(value["shape"], shape, "comparison shape differs")
    for size in shape:
        base.integer(size, "tensor dimension", 1)
    same(value["elements"], math.prod(shape), "comparison element coverage differs")
    same(value["actual_dtype"], actual, "actual comparison dtype differs")
    same(value["reference_dtype"], reference, "reference comparison dtype differs")
    same(value["comparison_dtype"], "torch.float64" if "torch.float64" in (actual, reference) else "torch.float32", "comparison arithmetic differs")
    same(value["tolerance"], base.TOLERANCES["float32"], "FP32 acceptance tolerance changed")
    for name in ("nonzero_error_count", "violation_count"):
        base.integer(value[name], name)
    base.require(value["violation_count"] <= value["nonzero_error_count"] <= value["elements"], "impossible error counts")
    base.flag(value["passed"], value["violation_count"] == 0, "violation count")
    base.require((value["max_absolute_error"] == 0) == (value["nonzero_error_count"] == 0), "nonzero count differs")


def detailed(value, shape, *, actual="torch.float32", reference="torch.float32"):
    closed(value, DETAIL_KEYS, "detailed comparison")
    metric_header(value, shape, actual=actual, reference=reference)
    violations = value["violation_samples"]
    base.require(isinstance(violations, list) and len(violations) == min(8, value["violation_count"]), "bounded violation coverage differs")
    indices = [sample(row, shape, value["max_absolute_error"], value["max_tolerance_ratio"]) for row in violations]
    base.require(indices == sorted(set(indices)), "violation coordinates duplicate/out of order")
    base.require(all(row["tolerance_ratio"] > 1 for row in violations), "nonviolating violation sample")
    for key, count in (("first_nonzero", value["nonzero_error_count"]), ("first_violation", value["violation_count"])):
        base.require((value[key] is None) == (count == 0), "missing/extra first coordinate")
    for key in ("first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio"):
        if value[key] is not None:
            sample(value[key], shape, value["max_absolute_error"], value["max_tolerance_ratio"])
    if value["first_nonzero"] is not None:
        base.require(value["first_nonzero"]["absolute_error"] > 0, "first nonzero drift is zero")
    if violations:
        same(value["first_violation"], violations[0], "first violation differs from bounded samples")
        base.require(coordinate(value["first_nonzero"]["coordinate"], shape) <= indices[0], "first drift occurs after first violation")
    same(value["worst_absolute_error"]["absolute_error"], value["max_absolute_error"], "worst absolute sample differs")
    same(value["worst_tolerance_ratio"]["tolerance_ratio"], value["max_tolerance_ratio"], "worst ratio sample differs")
    return value["passed"]


def trace(value, cfg, batch, length, endpoint):
    closed(value, ["stages", "first_nonzero", "first_violation", "passed"], "aligned trace")
    inventory = isolation.stage_inventory(cfg, batch, length)
    same([(row["layer"], row["stage"]) for row in value["stages"]], [(layer, stage) for layer, stage, _shape in inventory], "trace names/order/coverage differ")
    for row, (_layer, _stage, shape) in zip(value["stages"], inventory):
        if row["passed"]:
            closed(row, (DETAIL_KEYS - SAMPLES) | {"layer", "stage", "sample_policy", "first_nonzero_coordinate"}, "passing stage")
            same(row["sample_policy"], "passing_compact", "passing stage policy differs")
            metric_header(row, shape)
            base.require((row["first_nonzero_coordinate"] is None) == (row["nonzero_error_count"] == 0), "passing first-coordinate count differs")
            if row["first_nonzero_coordinate"] is not None:
                coordinate(row["first_nonzero_coordinate"], shape)
        else:
            closed(row, DETAIL_KEYS | {"layer", "stage", "sample_policy"}, "failed stage")
            same(row["sample_policy"], "full_failed", "failed stage policy differs")
            detailed({key: item for key, item in row.items() if key in DETAIL_KEYS}, shape)
    for key, count_key in (("first_nonzero", "nonzero_error_count"), ("first_violation", "violation_count")):
        first = next((row for row in value["stages"] if row[count_key]), None)
        marker = value[key]
        if first is None:
            base.require(marker is None, "extra pair-first marker")
            continue
        closed(marker, ["layer", "stage", "sample"], "pair-first marker")
        same((marker["layer"], marker["stage"]), (first["layer"], first["stage"]), "pair-first stage differs")
        sample(marker["sample"], first["shape"], first["max_absolute_error"], first["max_tolerance_ratio"])
        if first["passed"]:
            same(marker["sample"]["coordinate"], first["first_nonzero_coordinate"], "first compact coordinate differs")
            base.require(marker["sample"]["absolute_error"] > 0, "pair-first drift is zero")
        else:
            same(marker["sample"], first[key], "pair-first numeric sample differs")
    base.flag(value["passed"], all(row["passed"] for row in value["stages"]), "trace aggregate")
    final = value["stages"][-1]
    if final["passed"]:
        same({key: item for key, item in final.items() if key in DETAIL_KEYS}, {key: endpoint[key] for key in DETAIL_KEYS - SAMPLES}, "endpoint differs from compact final stage")
        same(final["first_nonzero_coordinate"], None if endpoint["first_nonzero"] is None else endpoint["first_nonzero"]["coordinate"], "endpoint first coordinate differs")
    else:
        same({key: item for key, item in final.items() if key in DETAIL_KEYS}, endpoint, "endpoint differs from failed final stage")


def norm_inventory(cfg, batch, length):
    result = []
    for layer, kind in enumerate(cfg["layer_types"]):
        result.append((f"blocks.{layer}.norm1", [layer, "block_input"], [layer, "norm1_output"], cfg["d_model"]))
        if kind == "mamba":
            result.append((f"blocks.{layer}.mixer.norm", [layer, "gated_norm_input"], [layer, "gated_norm_output"], cfg["d_model"] * cfg["expand"]))
        if cfg["mlp_on_every_layer"]:
            result.append((f"blocks.{layer}.norm2", [layer, "mixer_residual"], [layer, "norm2_output"], cfg["d_model"]))
    result.append(("norm_f", [cfg["n_layers"] - 1, "block_output"], [None, "final_norm"], cfg["d_model"]))
    return [(name, input_stage, output_stage, [batch, length, width]) for name, input_stage, output_stage, width in result]


def norm_probe(value, inventory):
    name, input_stage, output_stage, shape = inventory
    closed(value, ["module", "input_layout", "output_layout", "source_input_stage", "source_output_stage", "input", "observed_output", "weight", "epsilon",
                   "replay_layout", "tokenwise_shapes", "identical_operands", "input_unchanged", "oracle", "output_checks", "statistic_checks", "statistics", "passed", "exploratory"], "frozen normalization")
    same(value["module"], name, "norm site differs")
    same(value["source_input_stage"], input_stage, "frozen norm input stage differs")
    same(value["source_output_stage"], output_stage, "frozen norm output stage differs")
    same(value["epsilon"], 1e-5, "normalization epsilon differs")
    for key in ("input_layout", "output_layout", "replay_layout"):
        isolation.operand_layout(value[key], shape)
    same(value["replay_layout"], value["input_layout"], "frozen norm replay layout differs")
    same(value["tokenwise_shapes"], [shape[0], 1, shape[2]], "frozen norm token shape differs")
    for key in ("input", "observed_output"):
        isolation.identities(value[key], shape)
    isolation.identities(value["weight"], [shape[-1]])
    base.require(value["identical_operands"] is True and value["input_unchanged"] is True and value["exploratory"] is True, "unfrozen/promoted norm probe")
    same(value["oracle"], "independent detached CPU NumPy FP64 feature sum / reciprocal sqrt / weight multiplication", "norm oracle path differs")
    same(list(value["output_checks"]), NORM_OUTPUT_CHECKS, "norm output leaves differ")
    same(list(value["statistic_checks"]), NORM_STAT_CHECKS, "norm statistic leaves differ")
    for key, leaf in value["output_checks"].items():
        detailed(leaf, shape, reference="torch.float64" if key.endswith("vs_fp64") else "torch.float32")
    for key, leaf in value["statistic_checks"].items():
        stat_shape = shape if key.startswith("squared_") else [shape[0], shape[1], 1]
        detailed(leaf, stat_shape, actual="torch.float64" if key.startswith("mean_square_n1_") else "torch.float32", reference="torch.float64")
    routes = ["original_full", "original_tokenwise", "n1_full", "n1_tokenwise", "fp64_oracle"]
    same(list(value["statistics"]), routes, "frozen norm statistic identities differ")
    for route, identities in value["statistics"].items():
        closed(identities, ["squared", "mean_square", "coefficient"], "norm statistic route")
        for key, identity in identities.items():
            expected_dtype = "torch.float64" if route == "fp64_oracle" or route.startswith("n1_") and key == "mean_square" else "torch.float32"
            isolation.identities(identity, shape if key == "squared" else [shape[0], shape[1], 1], expected_dtype)
    for suffix in ("full", "tokenwise"):
        same(value["statistics"]["n1_" + suffix]["squared"], value["statistics"]["original_" + suffix]["squared"], "N1 changed FP32 squaring")
    base.flag(value["passed"], all(row["passed"] for row in norm_leaves(value)), "frozen norm aggregate")


def norm_leaves(value):
    return [value["output_checks"][key] for key in NORM_OUTPUT_CHECKS] + [value["statistic_checks"][key] for key in NORM_STAT_CHECKS]


def state_bundle(value, cfg, batch, length):
    closed(value, ["tolerance", "fields", "passed"], "retained-state bundle")
    same(value["tolerance"], base.TOLERANCES["float32"], "memory tolerance changed")
    inventory = isolation.state_inventory(cfg, batch, length)
    same([row["field"] for row in value["fields"]], list(inventory), "memory field coverage/order differs")
    for row in value["fields"]:
        closed(row, DETAIL_KEYS | {"field"}, "memory comparison")
        detailed({key: row[key] for key in DETAIL_KEYS}, inventory[row["field"]][0])
    base.flag(value["passed"], all(row["passed"] for row in value["fields"]), "memory bundle aggregate")


def target_hash(targets):
    return hashlib.sha256(b"".join(struct.pack("<q", value) for value in targets)).hexdigest()


EFFECT_KEYS = {"scope", "scored_tokens", "position_start", "position_stop_exclusive", "metric_dtype", "actual_mean_nll", "reference_mean_nll", "mean_nll_delta",
               "true_next_token_logprob_delta", "greedy_agreement", "greedy_disagreements", "actual_top1_top2_margin", "reference_top1_top2_margin", "per_token",
               "batch_size", "positions_per_row", "flatten_order", "targets_sha256"}


def effect(value, batch, length, start, targets):
    closed(value, EFFECT_KEYS, "descriptive next-token effect")
    same((value["batch_size"], value["positions_per_row"], value["position_start"], value["position_stop_exclusive"]), (batch, length, start, start + length), "per-row scoring positions differ")
    same(value["flatten_order"], "batch-major; position span describes each row", "scored flatten order differs")
    same(value["targets_sha256"], target_hash(targets), "scored target checksum differs")
    # The older arithmetic validator uses a flat one-row position span. Enforce
    # the actual B×L per-row span above, then reuse only its flattened arithmetic.
    arithmetic = {**value, "position_stop_exclusive": start + len(targets)}
    base.validate_effect(arithmetic, targets, start, "norm/head effect")


def suffix_targets(targets, batch, length, prefix):
    return [token for row in range(batch) for token in targets[row * length + prefix:(row + 1) * length]]


def head_probe(value, cfg, batch, length, baseline_logits):
    closed(value, ["input_layout", "output_layout", "source_input_stage", "source_output_stage", "input", "observed_output", "weight", "bias",
                   "replay_layout", "tokenwise_shapes", "identical_operands", "input_unchanged", "oracle", "oracle_output", "checks", "outputs", "passed", "exploratory"], "frozen head")
    input_shape, output_shape = [batch, length, cfg["d_model"]], [batch, length, cfg["vocab_size"]]
    same(value["source_input_stage"], [None, "lm_head.input"], "head input stage differs")
    same(value["source_output_stage"], [None, "lm_head.output"], "head output stage differs")
    for key in ("input_layout", "replay_layout"):
        isolation.operand_layout(value[key], input_shape)
    isolation.operand_layout(value["output_layout"], output_shape)
    same(value["replay_layout"], value["input_layout"], "head replay layout differs")
    same(value["tokenwise_shapes"], [batch, 1, cfg["d_model"]], "head token shape differs")
    for key, shape, dtype in (("input", input_shape, "torch.float32"), ("observed_output", output_shape, "torch.float32"),
                              ("weight", [cfg["vocab_size"], cfg["d_model"]], "torch.float32"), ("oracle_output", output_shape, "torch.float64")):
        isolation.identities(value[key], shape, dtype)
    same(value["observed_output"], baseline_logits, "frozen head did not observe N0H0 scores")
    base.require(value["bias"] is None, "historical head bias changed")
    base.require(value["identical_operands"] is True and value["input_unchanged"] is True and value["exploratory"] is True, "unfrozen/promoted head probe")
    same(value["oracle"], "independent detached CPU NumPy FP64 matrix multiplication and optional bias addition", "head oracle path differs")
    same(list(value["checks"]), HEAD_CHECKS, "head check coverage differs")
    for key, leaf in value["checks"].items():
        detailed(leaf, output_shape, reference="torch.float64" if key.endswith("vs_fp64") else "torch.float32")
    same(list(value["outputs"]), ["h0_full", "h0_tokenwise", "h1_full", "h1_tokenwise"], "head replay output coverage differs")
    for identity in value["outputs"].values():
        isolation.identities(identity, output_shape)
    base.flag(value["passed"], all(row["passed"] for row in value["checks"].values()), "head aggregate")


POLICY_KEYS = {"cuda_matmul_allow_tf32", "cudnn_allow_tf32", "deterministic_algorithms", "deterministic_warn_only", "cudnn_deterministic", "cudnn_benchmark", "cublas_workspace_config"}


def policy_flags(value):
    closed(value, POLICY_KEYS, "runtime policy flags")
    base.require(all(type(value[key]) is bool for key in POLICY_KEYS - {"cublas_workspace_config"}), "nonboolean runtime factor")
    base.require(value["cublas_workspace_config"] is None or isinstance(value["cublas_workspace_config"], str), "invalid CUBLAS environment")


def runtime_policy(value, declared, *, nested=False):
    closed(value, ["before", "active", "restored", "restoration_exact"], "runtime policy")
    for flags in (value["before"], value["active"], value["restored"]):
        policy_flags(flags)
    for key, expected in declared.items():
        same(value["before"][key], expected, "runtime factor differs from declaration: " + key)
        same(value["active"][key], expected, "active runtime factor differs: " + key)
    for key in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
        base.require(value["active"][key] is False, "actual TF32 enabled")
        if nested:
            base.require(value["before"][key] is False, "case outside outer TF32 scope")
    base.flag(value["restoration_exact"], value["before"] == value["restored"], "runtime restoration")


def tied_integrity(value):
    closed(value, ["before", "after", "parameter_identity_unchanged", "data_ptr_unchanged", "passed"], "tied weights")
    for key in ("before", "after"):
        closed(value[key], ["same_parameter", "same_storage"], "tied weight observation")
        base.require(all(type(flag) is bool for flag in value[key].values()), "nonboolean tied weight observation")
    for key in ("parameter_identity_unchanged", "data_ptr_unchanged"):
        base.require(type(value[key]) is bool, "nonboolean identity observation")
    base.require(all(value["before"].values()), "historical embedding/head tying missing before intervention")
    base.flag(value["passed"], value["before"] == value["after"] and value["parameter_identity_unchanged"] and value["data_ptr_unchanged"], "tied parameter restoration")


def route_paths(value, cfg, batch, length, tokenwise=False):
    closed(value, ["scan", "attention"], "actual route paths")
    mamba = [i for i, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]
    attention_layers = [i for i, kind in enumerate(cfg["layer_types"]) if kind == "attention"]
    positions = range(length) if tokenwise else (0,)
    scan = [{"layer": layer, "position": position, "length": 1 if tokenwise else length, "stateful": tokenwise,
             "path": "torch.chunked_ssd" if tokenwise else "torch.quadratic_ssd", "backend": "reference", "chunk_size": 128}
            for position in positions for layer in mamba]
    same(value["scan"], scan, "actual scan route/positions differ")
    attention = [{"layer": layer, "position": position, "query_length": 1 if tokenwise else length,
                  "key_length": position + 1 if tokenwise else length, "is_causal": position == 0, "mask_present": False, "mask_shape": None}
                 for position in positions for layer in attention_layers]
    same([{key: item for key, item in event.items() if key != "operand_layout"} for event in value["attention"]], attention, "actual attention route/positions differ")
    for event in value["attention"]:
        closed(event, [*attention[0], "operand_layout"], "attention event")
        same(list(event["operand_layout"]), ["q", "k", "v"], "attention operands differ")
        for name, layout in event["operand_layout"].items():
            isolation.operand_layout(layout, [batch, cfg["d_model"] // cfg["head_dim"], event["query_length"] if name == "q" else event["key_length"], cfg["head_dim"]])
    return {"backend": "reference", "scan_paths": sorted({row["path"] for row in scan}), "scan_calls": len(scan), "attention_calls": len(attention),
            "query_policy": "one token with all causal prefix keys" if tokenwise else "full causal sequence"}


def prefix_leaves(value):
    return [*value["prefix_comparisons"].values(), *(leaf for routes in value["suffix_comparisons"].values() for leaf in routes.values()),
            *value["route_comparisons"].values(), *(leaf for routes in value["retained_state"].values() for bundle in routes.values() for leaf in bundle["fields"])]


def prefix_probe(value, cfg, batch, length, prefix, targets):
    closed(value, ["prefix_position", "suffix_length", "synthetic", "prefix_unchanged", "prefix_fields", "prefix_logits", "prefix_comparisons", "suffix_logits",
                   "suffix_comparisons", "route_comparisons", "retained_state", "final_positions", "suffix_state_fields", "chunked_schedule", "next_token_effects_vs_original", "passed"], "actual prefix")
    same((value["prefix_position"], value["suffix_length"]), (prefix, length - prefix), "genuine prefix positions differ")
    base.require(value["synthetic"] is False and value["prefix_unchanged"] is True, "synthetic or mutated prefix")
    isolation.state_identities(value["prefix_fields"], cfg, batch, prefix)
    isolation.identities(value["prefix_logits"], [batch, 1, cfg["vocab_size"]])
    same(list(value["prefix_comparisons"]), ANCHORS, "prefix score anchors differ")
    for leaf in value["prefix_comparisons"].values():
        detailed(leaf, [batch, 1, cfg["vocab_size"]])
    for key in ("suffix_logits", "suffix_comparisons", "retained_state", "final_positions", "suffix_state_fields", "next_token_effects_vs_original"):
        same(list(value[key]), SUFFIX_ROUTES, "prefix suffix route coverage differs: " + key)
    suffix = length - prefix
    for route in SUFFIX_ROUTES:
        isolation.identities(value["suffix_logits"][route], [batch, suffix, cfg["vocab_size"]])
        same(list(value["suffix_comparisons"][route]), ANCHORS, "suffix score anchors differ")
        for leaf in value["suffix_comparisons"][route].values():
            detailed(leaf, [batch, suffix, cfg["vocab_size"]])
        same(list(value["retained_state"][route]), ANCHORS, "suffix state anchors differ")
        for bundle in value["retained_state"][route].values():
            state_bundle(bundle, cfg, batch, length)
        same(value["final_positions"][route], length, "suffix final position differs")
        isolation.state_identities(value["suffix_state_fields"][route], cfg, batch, length)
        effect(value["next_token_effects_vs_original"][route], batch, suffix, prefix, suffix_targets(targets, batch, length, prefix))
    same(list(value["route_comparisons"]), ["chunked128_vs_one_shot", "tokenwise_vs_one_shot"], "suffix route comparisons differ")
    for leaf in value["route_comparisons"].values():
        detailed(leaf, [batch, suffix, cfg["vocab_size"]])
    same(value["chunked_schedule"], [min(128, suffix - start) for start in range(0, suffix, 128)], "suffix chunk boundaries differ")
    base.flag(value["passed"], all(leaf["passed"] for leaf in prefix_leaves(value)), "prefix aggregate")


def cell_leaves(value):
    return [*value["whole_model_comparisons"].values(), *(row for trace in value["stage_comparisons"].values() for row in trace["stages"]),
            *(row for bundle in value["retained_state"].values() for row in bundle["fields"]),
            *(row for bundle in value["stateful_full_retained_state"].values() for row in bundle["fields"]), *prefix_leaves(value["actual_prefix"])]


def validate_cell(value, cell_id, cfg, identity):
    closed(value, ["cell_id", "normalization_fp64_coefficients", "head_fp64_matmul", "execution_status", "whole_model_comparisons", "stage_comparisons",
                   "retained_state", "stateful_full_retained_state", "actual_prefix", "final_positions", "logits", "next_token_effects", "actual_paths", "backward_executed", "BF16_executed",
                   "whole_model_passed", "all_recorded_comparisons_passed"], "completed factorial cell")
    same(value["cell_id"], cell_id, "cell order differs")
    same((value["normalization_fp64_coefficients"], value["head_fp64_matmul"]), (cell_id[1] == "1", cell_id[3] == "1"), "cell factor flags differ")
    base.require(value["execution_status"] == "completed" and value["backward_executed"] is False and value["BF16_executed"] is False, "cell execution scope differs")
    batch, length, prefix = (identity[key] for key in ("batch_size", "length", "prefix_length"))
    targets = value["next_token_effects"]["full_vs_original"]["per_token"]["target_ids"]
    base.require(isinstance(targets, list) and len(targets) == batch * length, "public scored target coverage differs")
    for token in targets:
        base.integer(token, "target token")
        base.require(token < cfg["vocab_size"], "target outside vocabulary")
    same(target_hash(targets), identity["targets_sha256"], "paired immutable target checksum differs")
    same(len(identity["windows"]), batch, "paired input window coverage differs")
    for index, window in enumerate(identity["windows"]):
        same(target_hash(targets[index * length:(index + 1) * length]), window["targets_sha256"], "per-row immutable target checksum differs")
    shape = [batch, length, cfg["vocab_size"]]
    same(list(value["whole_model_comparisons"]), ENDPOINTS, "eleven score endpoint coverage differs")
    for leaf in value["whole_model_comparisons"].values():
        detailed(leaf, shape)
    same(list(value["stage_comparisons"]), PAIRINGS, "five trace pairings differ")
    for name, row in value["stage_comparisons"].items():
        trace(row, cfg, batch, length, value["whole_model_comparisons"][name])
    same(list(value["retained_state"]), ANCHORS, "three retained-memory anchors differ")
    for row in value["retained_state"].values():
        state_bundle(row, cfg, batch, length)
    same(list(value["stateful_full_retained_state"]), ANCHORS[1:], "stateful full memory anchors differ")
    for row in value["stateful_full_retained_state"].values():
        state_bundle(row, cfg, batch, length)
    prefix_probe(value["actual_prefix"], cfg, batch, length, prefix, targets)
    same(value["final_positions"], {"tokenwise": length, "stateful_full": length}, "cell final positions differ")
    same(list(value["logits"]), ["full", "tokenwise", "chunked128"], "cell score identities differ")
    for row in value["logits"].values():
        isolation.identities(row, shape)
    same(list(value["next_token_effects"]), ["full_vs_original", "tokenwise_vs_original", "tokenwise_vs_own_full"], "descriptive effect anchors differ")
    for row in value["next_token_effects"].values():
        effect(row, batch, length, 0, targets)
    same(list(value["actual_paths"]), ["full", "tokenwise"], "cell actual path coverage differs")
    route_paths(value["actual_paths"]["full"], cfg, batch, length)
    route_paths(value["actual_paths"]["tokenwise"], cfg, batch, length, True)
    base.flag(value["whole_model_passed"], all(row["passed"] for row in value["whole_model_comparisons"].values())
              and all(row["passed"] for row in value["retained_state"].values()) and all(row["passed"] for row in value["stateful_full_retained_state"].values())
              and value["actual_prefix"]["passed"], "cell whole-model aggregate")
    base.flag(value["all_recorded_comparisons_passed"], all(row["passed"] for row in cell_leaves(value)), "cell recorded comparison aggregate")


PRIOR_MAP = {"full_vs_original": "candidate_full_vs_original_stateless", "tokenwise_vs_own_full": "candidate_tokenwise_vs_candidate_full",
             "tokenwise_vs_original": "candidate_tokenwise_vs_original_stateless", "stateful_full_vs_own_full": "candidate_stateful_full_vs_candidate_stateless"}
PRIOR_FIELDS = ["shape", "passed", "tolerance", "nonzero_error_count", "violation_count", "max_absolute_error", "max_tolerance_ratio",
                "first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples"]


def prior_reproduction(value, current, previous):
    closed(value, ["executed", "exact_fields", "comparisons", "exact_equal", "scope"], "N0H0 prior reproduction")
    base.require(value["executed"] is True, "historical N0H0 repeat unexecuted")
    same(value["exact_fields"], PRIOR_FIELDS, "prior control compared fields differ")
    same(list(value["comparisons"]), list(PRIOR_MAP), "prior repeated endpoint coverage differs")
    for name, old_name in PRIOR_MAP.items():
        base.flag(value["comparisons"][name], all(current[name][field] == previous["whole_model_comparisons"][old_name][field] for field in PRIOR_FIELDS), "exact prior endpoint " + name)
    base.flag(value["exact_equal"], all(value["comparisons"].values()), "exact prior reproduction aggregate")
    same(value["scope"], "N0H0 repeated historical full/tokenwise/stateful endpoint statistics; separate from tolerance acceptance", "prior control scope differs")


def original_anchor(value, cfg, batch, length):
    closed(value, ["logits", "loss", "stateful_logits", "state_fields", "stateful_vs_stateless", "repeat", "actual_paths"], "original anchor")
    shape = [batch, length, cfg["vocab_size"]]
    for key in ("logits", "stateful_logits"):
        isolation.identities(value[key], shape)
    isolation.identities(value["loss"], [])
    isolation.state_identities(value["state_fields"], cfg, batch, length)
    detailed(value["stateful_vs_stateless"], shape)
    closed(value["repeat"], ["exact_equal", "comparison"], "original exact repeat")
    detailed(value["repeat"]["comparison"], shape)
    base.flag(value["repeat"]["exact_equal"], value["repeat"]["comparison"]["nonzero_error_count"] == 0, "original exact repeat")
    route_paths(value["actual_paths"], cfg, batch, length)


def partition(length):
    return Counter(min(128, length - start) for start in range(0, length, 128))


def operators(value, cfg, length, prefix, cells, failed_cell, has_original):
    expected = {}
    layers = [i for i, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]
    def add(stage, counts):
        for layer in layers:
            for span, calls in counts.items():
                expected[(stage, layer, span)] = calls
    add("original.stateful_full", partition(length))
    stages = ["stateful_full", "full_chunked128", "prefix_prefill", "suffix_one_shot", "suffix_chunked128", "suffix_tokenwise"]
    for cell in CELLS:
        for stage in stages:
            counts = Counter({1: length - prefix}) if stage == "suffix_tokenwise" else partition(prefix if stage == "prefix_prefill" else length - prefix if stage.startswith("suffix_") else length)
            add(cell + "." + stage, counts)
    observed = {}
    allowed_cells = set(CELLS[:max(len(cells), 0 if failed_cell is None else CELLS.index(failed_cell) + 1)])
    for event in value:
        closed(event, ["stage", "layer", "path", "length", "autocast", "input_dtypes", "calls"], "scan observation")
        key = event["stage"], event["layer"], event["length"]
        base.require(key not in observed and key in expected, "unknown/duplicate scan observation")
        base.require(event["stage"] == "original.stateful_full" or event["stage"].split(".")[0] in allowed_cells, "unexecuted future-cell operator")
        same(event["path"], "torch.chunked_ssd", "cached/chunked path differs")
        base.require(event["autocast"] is False, "autocast operator observed")
        same(event["input_dtypes"], ["torch.float32"] * 7, "scan operands changed dtype")
        base.integer(event["calls"], "scan calls", 1)
        base.require(event["calls"] <= expected[key], "excess scan calls")
        observed[key] = event["calls"]
    required = {key: calls for key, calls in expected.items() if key[0].split(".")[0] in {row["cell_id"] for row in cells}
                or has_original and key[0] == "original.stateful_full"}
    base.require(all(observed.get(key) == calls for key, calls in required.items()), "completed route missing actual operator coverage")


def case_leaves(value):
    leaves = [row for cell in value["cells"] for row in cell_leaves(cell)]
    leaves.extend(row for probe in value["frozen_norms"] for row in norm_leaves(probe))
    if value["frozen_head"] is not None:
        leaves.extend(value["frozen_head"]["checks"].values())
    if "original_anchor" in value:
        leaves += [value["original_anchor"]["stateful_vs_stateless"], value["original_anchor"]["repeat"]["comparison"]]
    return leaves


def validate_case(value, identity, checkpoint, previous, declared_policy):
    required = {"ratio", *identity, "weights_sha256", "execution_status", "cells", "failed_cell", "frozen_norms", "frozen_head",
                "backward_executed", "BF16_executed", "production_defaults_changed", "optimizer_executed", "runtime_policy", "tied_weight_integrity",
                "post_model_weights_unchanged", "operator_observations", "attribution_control_stable", "all_recorded_comparisons_passed", "processing_seconds", "cuda_peak_allocated_bytes"}
    optional = {"reason", "original_anchor", "decay_baseline_anchor", "prior_reproduction"}
    base.require(isinstance(value, dict) and required <= set(value) <= required | optional, "case schema differs")
    for key, expected in identity.items():
        same(value[key], expected, "shared case identity differs: " + key)
    same(value["ratio"], checkpoint["ratio"], "case/checkpoint ratio differs")
    same(value["weights_sha256"], previous["weights_sha256"], "loaded weights differ from immutable prior")
    base.digest(value["weights_sha256"], "loaded weights")
    cfg = checkpoint["model_config"]
    batch, length = identity["batch_size"], identity["length"]
    complete = value["execution_status"] == "completed"
    base.require(value["execution_status"] in {"completed", "incomplete"}, "case execution status differs")
    for key in ("backward_executed", "BF16_executed", "production_defaults_changed", "optimizer_executed"):
        base.require(value[key] is False, "unapproved executed scope: " + key)
    runtime_policy(value["runtime_policy"], declared_policy, nested=True)
    tied_integrity(value["tied_weight_integrity"])
    base.require(type(value["post_model_weights_unchanged"]) is bool, "invalid model restoration flag")
    if complete:
        base.require(value["failed_cell"] is None and "reason" not in value, "complete case has failure metadata")
        base.require(value["post_model_weights_unchanged"] and value["tied_weight_integrity"]["passed"] and value["runtime_policy"]["restoration_exact"], "complete case did not restore model/runtime")
    else:
        base.require(isinstance(value.get("reason"), str) and value["reason"], "incomplete case missing reason")
        base.require(value["failed_cell"] is None or value["failed_cell"] in CELLS, "unknown failed cell")
    base.number(value["processing_seconds"], "case processing time", 0)
    base.integer(value["cuda_peak_allocated_bytes"], "case allocation high-water mark", 1)
    base.require(isinstance(value["cells"], list) and len(value["cells"]) <= 4, "extra factorial cell")
    same([row["cell_id"] for row in value["cells"]], CELLS if complete else CELLS[:len(value["cells"])], "cell coverage/order differs")
    if "original_anchor" in value:
        original_anchor(value["original_anchor"], cfg, batch, length)
    base.require(not value["cells"] or {"original_anchor", "decay_baseline_anchor", "prior_reproduction"} <= set(value), "executed cells missing anchors/control")
    if "decay_baseline_anchor" in value:
        anchor = value["decay_baseline_anchor"]
        closed(anchor, ["logits", "state_fields", "cell"], "decay baseline anchor")
        same(anchor["cell"], "N0H0", "baseline cell differs")
        isolation.identities(anchor["logits"], [batch, length, cfg["vocab_size"]])
        isolation.state_identities(anchor["state_fields"], cfg, batch, length)
    for row, cell in zip(value["cells"], CELLS):
        validate_cell(row, cell, cfg, identity)
    if value["cells"]:
        baseline = value["cells"][0]
        same(baseline["logits"]["full"], value["decay_baseline_anchor"]["logits"], "baseline scores do not belong to N0H0")
        same(baseline["whole_model_comparisons"]["full_vs_decay_baseline"]["nonzero_error_count"], 0, "N0H0 self anchor differs")
        prior_reproduction(value["prior_reproduction"], baseline["whole_model_comparisons"], previous)
    elif "prior_reproduction" in value:
        # The producer may finish N0H0 endpoints before its first trace/cell fails.
        closed(value["prior_reproduction"], ["executed", "exact_fields", "comparisons", "exact_equal", "scope"], "partial prior control")
        same(value["prior_reproduction"]["exact_fields"], PRIOR_FIELDS, "partial prior control fields differ")
        same(list(value["prior_reproduction"]["comparisons"]), list(PRIOR_MAP), "partial prior control coverage differs")
        base.require(value["prior_reproduction"]["executed"] is True and all(type(flag) is bool for flag in value["prior_reproduction"]["comparisons"].values()), "partial control flags invalid")
        base.flag(value["prior_reproduction"]["exact_equal"], all(value["prior_reproduction"]["comparisons"].values()), "partial prior aggregate")
    inventory = norm_inventory(cfg, batch, length)
    base.require(isinstance(value["frozen_norms"], list) and len(value["frozen_norms"]) <= len(inventory), "extra frozen norm")
    same([row["module"] for row in value["frozen_norms"]], [row[0] for row in inventory[:len(value["frozen_norms"])]], "norm site coverage/order differs")
    base.require(not value["frozen_norms"] or bool(value["cells"]), "frozen operands without executed N0H0")
    for probe, site in zip(value["frozen_norms"], inventory):
        norm_probe(probe, site)
    if value["frozen_head"] is not None:
        base.require(len(value["frozen_norms"]) == len(inventory) and "decay_baseline_anchor" in value, "head executed before required norm sites")
        head_probe(value["frozen_head"], cfg, batch, length, value["decay_baseline_anchor"]["logits"])
    if complete:
        base.require(len(value["frozen_norms"]) == len(inventory) and value["frozen_head"] is not None, "completed case lacks required frozen inventory")
    operators(value["operator_observations"], cfg, length, identity["prefix_length"], value["cells"], value["failed_cell"], "original_anchor" in value)
    stable = value.get("original_anchor", {}).get("repeat", {}).get("exact_equal", False) and value.get("prior_reproduction", {}).get("exact_equal", False)
    base.flag(value["attribution_control_stable"], stable, "attribution control")
    base.flag(value["all_recorded_comparisons_passed"], complete and stable and value["tied_weight_integrity"]["passed"] and all(row["passed"] for row in case_leaves(value)), "case recorded comparison aggregate")


REPORT_SCHEMA = {"version": 1,
    "passing_stage_rows": "complete names/shapes/dtypes/tolerances/counts/maxima and first_nonzero_coordinate; redundant numeric samples omitted",
    "failed_stage_rows": "complete detailed comparison including bounded first/worst/violation samples",
    "pair_first_markers": "complete first_nonzero and first_violation numeric samples", "coordinate_limit": 8, "serialized_activations": False,
    "frozen_norm_output_checks": NORM_OUTPUT_CHECKS, "frozen_norm_stat_checks": NORM_STAT_CHECKS, "frozen_head_checks": HEAD_CHECKS,
    "target_raw_bytes": 50 * 1024**2}
FIXED_DECLARATION = {"schema": 1, "date": "2026-10-05", "study": "Fixed-decay normalization coefficient and final projection precision factorial", "status_at_declaration": "planned_before_execution", "device": "cuda", "seed": 2027,
    "ratios": RATIOS, "cases": CASES, "cells": CELLS, "decay_treatment": "fp64_cumsum_decay_coefficients",
    "normalization_treatment": "fp64_rms_reduction_coefficients",
    "normalization_scope": "all RMSNorms; FP32 input/square, FP64 mean+epsilon+rsqrt, FP32 coefficient then original FP32 input/weight products",
    "head_treatment": "fp64_final_head_matmul",
    "head_scope": "instance lm_head only; FP64 input/weight matrix calculation then FP32 output; tied parameter unchanged",
    "chunk_size": 128, "tolerances": base.TOLERANCES, "report_schema": REPORT_SCHEMA,
    "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
    "historical_policy_coverage": "prior records deterministic_algorithms; other factors recorded before this execution, not retrospectively certified",
    "time_budget_seconds": 900, "minimum_large_case_free_bytes": 8 * 1024**3,
    "backward_executed": False, "BF16_executed": False, "production_defaults_changed": False, "optimizer_executed": False}
AUDIT_GROUPS = ["checkpoints", "training_manifests", "prepared_data_and_tokenizer", "prior_and_declaration", "sources"]
ALLOWANCE_SCOPE = "runner entry through preflight/model work/final audits; immutable publication excluded"


def reference(value):
    closed(value, ["path", "path_scope", "sha256"], "public file identity")
    relative(value["path"])
    same(value["path_scope"], "repository-relative", "private file identity scope")
    base.digest(value["sha256"], "public file")


def validate_declaration(value, prior_plan, prior, source_root):
    closed(value, {*FIXED_DECLARATION, "shared_inputs", "observed_inference_policy", "prior_declaration", "prior_report", "output", "source_sha256", "data", "checkpoints"}, "declaration")
    for key, expected in FIXED_DECLARATION.items():
        same(value[key], expected, "declared scope differs: " + key)
    policy = value["observed_inference_policy"]
    closed(policy, POLICY_KEYS - {"cuda_matmul_allow_tf32", "cudnn_allow_tf32"}, "declared inference factors")
    policy_flags({**value["precision_flags"], **policy})
    same(policy["deterministic_algorithms"], prior["protocol"]["runtime"]["precision_flags"]["deterministic_algorithms"], "historical deterministic factor differs")
    for key in ("prior_declaration", "prior_report"):
        reference(value[key])
    relative(value["output"])
    same(value["shared_inputs"], prior["protocol"]["shared_inputs"], "prior shared inputs differ")
    same(value["shared_inputs"], prior_plan["shared_inputs"], "prior plan inputs differ")
    same([row["case_id"] for row in value["shared_inputs"]], CASE_IDS, "three shared shapes differ")
    for identity, cfg in zip(value["shared_inputs"], CASES):
        for key, expected in cfg.items():
            same(identity[key], expected, "shared input geometry differs")
    for key in ("data", "checkpoints"):
        same(value[key], prior["protocol"][key], "immutable prior registry differs: " + key)
    same([row["ratio"] for row in value["checkpoints"]], RATIOS, "checkpoint arm order differs")
    isolation.source_check(value["source_sha256"], source_root, SOURCES)


def audit_registry(declaration, declaration_path, declaration_hash):
    data, checkpoints = declaration["data"], declaration["checkpoints"]
    return {
        "checkpoints": [(row["path"], row["sha256"]) for row in checkpoints],
        "training_manifests": [(row["training_manifest"]["path"], row["training_manifest"]["sha256"]) for row in checkpoints],
        "prepared_data_and_tokenizer": [(data["manifest"]["path"], data["manifest"]["sha256"]), (data["tokenizer"]["path"], data["tokenizer"]["sha256"]),
                                        *[(item["path"], item["sha256"]) for item in data["artifacts"].values()]],
        "prior_and_declaration": [(declaration_path, declaration_hash), *[(declaration[key]["path"], declaration[key]["sha256"]) for key in ("prior_declaration", "prior_report")]],
        "sources": list(declaration["source_sha256"].items())}


def final_audit(value, registry, complete):
    integrity, files = value["post_execution_integrity"], value["post_execution_audit_files"]
    same(list(integrity), AUDIT_GROUPS, "final audit group coverage/order differs")
    base.require(all(type(flag) is bool or flag is None for flag in integrity.values()), "invalid/unexecuted audit flag")
    expected = [(group, path, digest) for group, entries in registry.items() for path, digest in entries]
    base.require(isinstance(files, list) and len(files) <= len(expected), "extra final audit file")
    for row, (group, path, digest) in zip(files, expected):
        closed(row, ["group", "path", "path_scope", "expected_sha256", "actual_sha256", "passed"], "final audit file")
        same((row["group"], row["path"], row["expected_sha256"]), (group, path, digest), "audit file registry differs")
        relative(row["path"])
        same(row["path_scope"], "repository-relative", "audit file path scope differs")
        base.digest(row["expected_sha256"], "expected audit file")
        base.digest(row["actual_sha256"], "actual audit file")
        base.flag(row["passed"], row["actual_sha256"] == digest, "actual file hash audit")
    stopped = False
    for group, entries in registry.items():
        rows = [row for row in files if row["group"] == group]
        if stopped:
            base.require(integrity[group] is None and not rows, "audit resumed after an unexecuted group")
        elif integrity[group] is None:
            # A final file hash can finish then exhaust the budget before the
            # group assignment; even full recorded file coverage supplies no
            # completed group flag in that case.
            stopped = True
        else:
            same(len(rows), len(entries), "completed audit group lacks file coverage")
            base.flag(integrity[group], all(row["passed"] for row in rows), "file group aggregate")
    if complete:
        base.require(len(files) == len(expected) and all(flag is True for flag in integrity.values()), "completed workflow lacks stable final audit")
    return len(files) == len(expected) and all(flag is True for flag in integrity.values())


def validate_coverage(value, expected):
    rows = value["cases"]
    base.require(isinstance(rows, list) and len(rows) <= len(expected), "extra recorded case")
    same([(row["ratio"], row["case_id"]) for row in rows], expected[:len(rows)], "six-case coverage/order differs")
    complete = value["execution_status"] == "completed"
    if complete:
        same(len(rows), 6, "completed study missing cases")
        base.require(all(row["execution_status"] == "completed" for row in rows), "completed study contains incomplete case")
        base.require(value["reason"] is None and value["failed_case"] is None, "completed workflow has failure metadata")
    else:
        base.require(isinstance(value["reason"], str) and value["reason"], "incomplete workflow missing reason")
        base.require(all(row["execution_status"] == "completed" for row in rows[:-1]), "workflow continued after incomplete case")
        failed = value["failed_case"]
        closed(failed, ["ratio", "case_id", "cell", "stage"], "workflow interruption")
        if len(rows) == 6 and all(row["execution_status"] == "completed" for row in rows):
            same(failed, {"ratio": None, "case_id": None, "cell": None, "stage": "final audit / allowance"}, "completed cases hide final workflow failure")
        elif rows and rows[-1]["execution_status"] == "incomplete":
            case = rows[-1]
            same((failed["ratio"], failed["case_id"]), (case["ratio"], case["case_id"]), "failed recorded case differs")
            base.require(failed["stage"] in {"original anchors", "full/tokenwise/stateful/continuation", "frozen normalization/head"}, "unknown recorded-case failure stage")
            post_case_safeguard = (len(case["cells"]) == 4 and case["failed_cell"] is None
                and case.get("reason") == "model weights changed during diagnostic"
                and not (case["post_model_weights_unchanged"] and case["tied_weight_integrity"]["passed"] and case["runtime_policy"]["restoration_exact"]))
            if post_case_safeguard:
                # The completed cell loop cleared failed_cell before the final
                # restoration safeguard failed. The runner's last active cell
                # still names N1H1; do not discard that legitimate partial raw.
                same((failed["cell"], failed["stage"]), ("N1H1", "full/tokenwise/stateful/continuation"), "post-case safeguard interruption differs")
            else:
                same(failed["cell"], case["failed_cell"], "failed cell differs from case")
        else:
            ratio, case_id = expected[len(rows)]
            same(failed["ratio"], ratio, "next unexecuted ratio differs")
            if failed["stage"] == "load model":
                base.require(failed["case_id"] is None and failed["cell"] is None and case_id == CASE_IDS[0], "invalid model-load failure")
            else:
                same(failed, {"ratio": ratio, "case_id": case_id, "cell": None, "stage": "case preflight"}, "next unexecuted case differs")


def validate(value, declaration, declaration_hash, declaration_path, prior, source_root):
    closed(value, ["schema", "kind", "certified", "execution_status", "status", "reason", "failed_case", "timestamp_utc", "protocol", "protocol_sha256", "cases",
                   "large_case_headroom", "post_execution_integrity", "post_execution_audit_files", "runtime_policy", "execution_allowance", "limits"], "audited report")
    same(value["schema"], 1, "raw schema differs")
    base.require(value["kind"] == KIND and value["certified"] is False and value["execution_status"] in {"completed", "incomplete"}, "unknown/promoted raw identity")
    protocol = value["protocol"]
    closed(protocol, {*declaration, "declaration", "runtime", "runtime_sha256", "quality_equivalence_margin"}, "executed protocol")
    same(value["protocol_sha256"], base.canonical_sha256(protocol), "protocol canonical fingerprint differs")
    for key, expected in declaration.items():
        same(protocol[key], expected, "executed protocol differs from declaration: " + key)
    same(protocol["declaration"], {"path": declaration_path, "path_scope": "repository-relative", "sha256": declaration_hash}, "declaration byte identity differs")
    base.require(protocol["quality_equivalence_margin"] is None, "invented quality equivalence margin")
    runtime = protocol["runtime"]
    same(sorted(runtime), sorted([*prior["protocol"]["runtime"], "inference_policy"]), "runtime metadata coverage differs")
    same({key: runtime[key] for key in prior["protocol"]["runtime"]}, prior["protocol"]["runtime"], "measured runtime differs from immutable control")
    same(protocol["runtime_sha256"], base.canonical_sha256(runtime), "runtime canonical fingerprint differs")
    runtime_policy(value["runtime_policy"], declaration["observed_inference_policy"])
    same(runtime["inference_policy"], value["runtime_policy"]["active"], "active policy metadata differs")
    for key in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "deterministic_algorithms"):
        same(runtime["precision_flags"][key], runtime["inference_policy"][key], "runtime precision observation differs")
    expected = [(ratio, identity["case_id"]) for ratio in RATIOS for identity in declaration["shared_inputs"]]
    validate_coverage(value, expected)
    identities = {row["case_id"]: row for row in declaration["shared_inputs"]}
    checkpoints = {row["ratio"]: row for row in declaration["checkpoints"]}
    prior_cases = {(row["ratio"], row["case_id"]): row for row in prior["cases"]}
    for case in value["cases"]:
        validate_case(case, identities[case["case_id"]], checkpoints[case["ratio"]], prior_cases[(case["ratio"], case["case_id"])], declaration["observed_inference_policy"])
    complete = value["execution_status"] == "completed"
    same(value["status"], "incomplete" if not complete else "completed" if all(row["all_recorded_comparisons_passed"] for row in value["cases"]) else "completed_with_parity_failures", "workflow status hides recorded negatives")
    audit_ok = final_audit(value, audit_registry(declaration, declaration_path, declaration_hash), complete)
    allowance = value["execution_allowance"]
    closed(allowance, ["allowance_seconds", "elapsed_seconds", "overshoot_seconds", "scope"], "execution allowance")
    same(allowance["allowance_seconds"], 900, "cooperative allowance changed")
    same(allowance["scope"], ALLOWANCE_SCOPE, "allowance measurement scope differs")
    base.number(allowance["elapsed_seconds"], "elapsed time", 0)
    base.close_double(allowance["overshoot_seconds"], max(0, allowance["elapsed_seconds"] - 900), "in-flight wall overshoot")
    base.require(not complete or allowance["elapsed_seconds"] < 900, "completed workflow exceeded declared allowance")
    base.require(not complete or value["runtime_policy"]["restoration_exact"], "completed workflow did not restore runtime")
    seen = []
    required = [(row["ratio"], row["case_id"]) for row in value["cases"] if row["batch_size"] == 2]
    for row in value["large_case_headroom"]:
        closed(row, ["ratio", "case_id", "free_bytes", "total_bytes", "minimum_free_bytes"], "large-case headroom")
        pair = row["ratio"], row["case_id"]
        base.require(pair not in seen and pair in [(ratio, CASE_IDS[-1]) for ratio in RATIOS], "duplicate/unknown memory guard")
        seen.append(pair)
        for key in ("free_bytes", "total_bytes", "minimum_free_bytes"):
            base.integer(row[key], "headroom " + key, 0 if key == "free_bytes" else 1)
        same(row["minimum_free_bytes"], 8 * 1024**3, "headroom guard changed")
        base.require(row["free_bytes"] <= row["total_bytes"] and (pair not in required or row["free_bytes"] >= 8 * 1024**3), "executed large case lacked headroom")
    failed = value["failed_case"]
    extra = [] if complete or failed is None or failed["case_id"] != CASE_IDS[-1] else [(failed["ratio"], CASE_IDS[-1])]
    base.require(seen == required or seen == required + [pair for pair in extra if pair not in required], "headroom coverage differs")
    base.require(isinstance(value["timestamp_utc"], str) and bool(value["timestamp_utc"]) and isinstance(value["limits"], list) and all(isinstance(row, str) for row in value["limits"]), "public report annotations invalid")
    return audit_ok


def compact_stage(value):
    result = {key: deepcopy(value[key]) for key in ("layer", "stage", "passed", "finite", "shape", "actual_dtype", "reference_dtype", "comparison_dtype",
                                                   "elements", "nonzero_error_count", "violation_count", "max_absolute_error", "max_tolerance_ratio")}
    if value["passed"]:
        result["first_nonzero_coordinate"] = deepcopy(value["first_nonzero_coordinate"])
    else:
        result.update({key: deepcopy(value[key]) for key in SAMPLES})
    return result


def compact_bundle(value):
    return {"passed": value["passed"], "fields": [{"field": row["field"], **isolation.compact_detail(row)} for row in value["fields"]],
            "counts": isolation.count(value["fields"])}


def compact_effect(value):
    return {key: deepcopy(item) for key, item in value.items() if key != "per_token"}


def compact_cell(value, cfg, batch, length):
    result = {key: deepcopy(item) for key, item in value.items() if key not in {"actual_paths", "stage_comparisons", "retained_state", "stateful_full_retained_state", "actual_prefix", "next_token_effects"}}
    result["stage_comparisons"] = {name: {"passed": row["passed"], "first_nonzero": deepcopy(row["first_nonzero"]), "first_violation": deepcopy(row["first_violation"]),
                                          "counts": isolation.count(row["stages"]), "stages": [compact_stage(stage) for stage in row["stages"]]}
                                   for name, row in value["stage_comparisons"].items()}
    for key in ("retained_state", "stateful_full_retained_state"):
        result[key] = {name: compact_bundle(row) for name, row in value[key].items()}
    prefix = value["actual_prefix"]
    result["actual_prefix"] = {key: deepcopy(item) for key, item in prefix.items() if key not in {"retained_state", "next_token_effects_vs_original"}}
    result["actual_prefix"]["retained_state"] = {route: {anchor: compact_bundle(row) for anchor, row in bundles.items()} for route, bundles in prefix["retained_state"].items()}
    result["actual_prefix"]["next_token_effects_vs_original"] = {route: compact_effect(row) for route, row in prefix["next_token_effects_vs_original"].items()}
    result["actual_paths"] = {"full": route_paths(value["actual_paths"]["full"], cfg, batch, length), "tokenwise": route_paths(value["actual_paths"]["tokenwise"], cfg, batch, length, True)}
    result["next_token_effects"] = {key: compact_effect(row) for key, row in value["next_token_effects"].items()}
    result["stage_comparison_counts"] = isolation.count(row for pair in value["stage_comparisons"].values() for row in pair["stages"])
    return result


def compact_case(value, cfg):
    result = {key: deepcopy(item) for key, item in value.items() if key not in {"cells", "original_anchor", "operator_observations", "processing_seconds", "cuda_peak_allocated_bytes"}}
    batch, length = value["batch_size"], value["length"]
    result["cells"] = [compact_cell(row, cfg, batch, length) for row in value["cells"]]
    if "original_anchor" in value:
        result["original_anchor"] = {key: deepcopy(item) for key, item in value["original_anchor"].items() if key != "actual_paths"}
        result["original_anchor"]["actual_paths"] = route_paths(value["original_anchor"]["actual_paths"], cfg, batch, length)
    result["operator_observations"] = deepcopy(value["operator_observations"])
    result["resources"] = {"processing_seconds": value["processing_seconds"], "cuda_peak_allocated_bytes": value["cuda_peak_allocated_bytes"],
                           "scope": "case combined full/tokenwise/cached routes, stage comparisons and frozen CPU/GPU replay/oracle work; allocation peak reset before each case; not throughput"}
    norms = value["frozen_norms"]
    result["frozen_comparison_counts"] = {
        "normalization_outputs": isolation.count(row for probe in norms for row in probe["output_checks"].values()),
        "normalization_statistics": isolation.count(row for probe in norms for row in probe["statistic_checks"].values()),
        "head": isolation.count([] if value["frozen_head"] is None else value["frozen_head"]["checks"].values())}
    result["frozen_coverage"] = {"expected_norm_sites": len(norm_inventory(cfg, batch, length)), "executed_norm_sites": len(norms), "head_executed": value["frozen_head"] is not None,
                                 "common_input_cell": "N0H0", "independent_repetitions_per_cell": False}
    result["missing_cells"] = CELLS[len(value["cells"]):]
    result["prior_reproduction_endpoints_available"] = bool(value["cells"])
    return result


def validate_prior_chain(declaration, root, source_root):
    prior_plan, plan_hash = base.read_json(root / relative(declaration["prior_declaration"]["path"]))
    prior, raw_hash = base.read_json(root / relative(declaration["prior_report"]["path"]))
    same(plan_hash, declaration["prior_declaration"]["sha256"], "immutable prior declaration bytes differ")
    same(raw_hash, declaration["prior_report"]["sha256"], "immutable prior raw bytes differ")
    base.require(prior["kind"] == isolation.KIND and prior["certified"] is False and prior["execution_status"] == "completed", "incomplete/unknown/certified prior")
    same(prior["protocol_sha256"], base.canonical_sha256(prior["protocol"]), "immutable prior canonical protocol differs")
    same(prior["protocol"]["declaration"], declaration["prior_declaration"], "immutable prior declaration chain differs")
    older_plan, older_plan_hash = base.read_json(root / relative(prior_plan["breadth_declaration"]["path"]))
    older, older_hash = base.read_json(root / relative(prior_plan["breadth_report"]["path"]))
    same(older_plan_hash, prior_plan["breadth_declaration"]["sha256"], "prior breadth declaration bytes differ")
    same(older_hash, prior_plan["breadth_report"]["sha256"], "prior breadth raw bytes differ")
    base.require(older["kind"] == isolation.breadth.KIND and older["certified"] is False and older["execution_status"] == "completed", "invalid prior breadth identity")
    same(older["protocol_sha256"], base.canonical_sha256(older["protocol"]), "prior breadth canonical protocol differs")
    same(older["protocol"]["declaration"], prior_plan["breadth_declaration"], "prior breadth declaration chain differs")
    isolation.source_check(older["protocol"]["source_sha256"], source_root, isolation.breadth.SOURCES)
    same([(row["ratio"], row["case_id"]) for row in older["cases"]], [(ratio, isolation.breadth.identifier(case)) for ratio in base.RATIOS for case in isolation.breadth.CASES], "prior breadth complete matrix differs")
    isolation.validate_declaration(prior_plan, older_plan, older, source_root)
    isolation.validate(prior, prior_plan, plan_hash, declaration["prior_declaration"]["path"], older, source_root)
    return prior_plan, prior


def build_summary(declaration_path, raw_path=None, *, root=ROOT, source_root=ROOT):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    declaration, declaration_hash = base.read_json(declaration_path)
    declared_raw = root / relative(declaration["output"])
    raw_path = declared_raw if raw_path is None else Path(raw_path).resolve()
    same(str(raw_path.resolve()), str(declared_raw.resolve()), "raw path differs from declaration")
    report, raw_hash = base.read_json(raw_path)
    prior_plan, prior = validate_prior_chain(declaration, root, source_root)
    validate_declaration(declaration, prior_plan, prior, source_root)
    declaration_relative = declaration_path.relative_to(root).as_posix()
    audited = "protocol" in report
    final_ok = False
    if audited:
        final_ok = validate(report, declaration, declaration_hash, declaration_relative, prior, source_root)
    else:
        closed(report, ["schema", "kind", "certified", "status", "execution_status", "reason", "cases"], "pre-audit failure")
        same(report["schema"], 1, "pre-audit schema differs")
        base.require(report["kind"] == KIND and report["certified"] is False and report["status"] == report["execution_status"] == "incomplete"
                     and report["cases"] == [] and isinstance(report["reason"], str) and report["reason"], "invalid pre-audit incomplete report")
    configs = {row["ratio"]: row["model_config"] for row in declaration["checkpoints"]}
    cases = [compact_case(row, configs[row["ratio"]]) for row in report["cases"]]
    raw_cells = [cell for case in report["cases"] for cell in case["cells"]]
    expected = [(ratio, identity["case_id"]) for ratio in RATIOS for identity in declaration["shared_inputs"]]
    completed = {(case["ratio"], case["case_id"]) for case in cases if case["execution_status"] == "completed"}
    missing_cells = [{"ratio": ratio, "case_id": case_id, "cell_id": cell_id} for ratio, case_id in expected for cell_id in CELLS
                     if not any(case["ratio"] == ratio and case["case_id"] == case_id and any(row["cell_id"] == cell_id for row in case["cells"]) for case in cases)]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"], "execution_status": report["execution_status"],
        "certified": False, "reason": report["reason"], "failed_case": report.get("failed_case"), "execution_identities_available": audited, "final_audit_passed": final_ok,
        "declaration": {"path": declaration_relative, "sha256": declaration_hash}, "raw_report": {"path": declaration["output"], "sha256": raw_hash},
        "protocol_sha256": report.get("protocol_sha256"), "protocol": report.get("protocol"),
        "coverage": {"expected_cases": 6, "recorded_cases": len(cases), "completed_cases": len(completed), "expected_cells": 24, "completed_cells": len(raw_cells),
                     "missing_cases": [{"ratio": ratio, "case_id": case_id} for ratio, case_id in expected if (ratio, case_id) not in completed], "missing_cells": missing_cells,
                     "execution_identities_available": audited, "backward_executed": False, "BF16_executed": False, "optimizer_executed": False, "production_defaults_changed": False},
        "gate_counts": {"cells": {cell_id: {"whole_model_comparisons": {name: isolation.count(cell["whole_model_comparisons"][name] for cell in raw_cells if cell["cell_id"] == cell_id) for name in ENDPOINTS},
                                             "whole_model": {"passed": sum(cell["whole_model_passed"] for cell in raw_cells if cell["cell_id"] == cell_id), "total": sum(cell["cell_id"] == cell_id for cell in raw_cells)},
                                             "all_recorded": {"passed": sum(cell["all_recorded_comparisons_passed"] for cell in raw_cells if cell["cell_id"] == cell_id), "total": sum(cell["cell_id"] == cell_id for cell in raw_cells)}} for cell_id in CELLS},
                       "original_repeat": {"exact_passed": sum(case["original_anchor"]["repeat"]["exact_equal"] for case in cases if "original_anchor" in case), "total": sum("original_anchor" in case for case in cases)},
                       "prior_reproduction": {"exact_passed": sum(case["prior_reproduction"]["exact_equal"] for case in cases if case["cells"]), "total": sum(bool(case["cells"]) for case in cases)},
                       "frozen_normalization_outputs": isolation.count(row for case in report["cases"] for probe in case["frozen_norms"] for row in probe["output_checks"].values()),
                       "frozen_normalization_statistics": isolation.count(row for case in report["cases"] for probe in case["frozen_norms"] for row in probe["statistic_checks"].values()),
                       "frozen_head": isolation.count(row for case in report["cases"] if case["frozen_head"] is not None for row in case["frozen_head"]["checks"].values())},
        "cases": cases, "runtime_policy": report.get("runtime_policy"), "execution_allowance": report.get("execution_allowance"), "large_case_headroom": report.get("large_case_headroom", []),
        "post_execution_integrity": report.get("post_execution_integrity"), "post_execution_audit_files": report.get("post_execution_audit_files", []),
        "limits": [*report.get("limits", []), "The standard-library exporter loads public JSON and text source only; producer file audits bind checkpoint/corpus binaries without reloading them.",
                   "Every aggregate uses explicitly declared executed comparison leaves; unknown nested pass fields, missing cells and missing audits supply no passes.",
                   "All stage maxima/counts/shapes/dtypes and pair-first samples remain; full coordinate samples remain for every failed stage and all endpoints/frozen probes.",
                   "Frozen normalization/head checks use common N0H0 inputs once per case, not four independent per-cell repetitions; local passes cannot promote a whole-model policy.",
                   "Partial N0H0 controls without a completed cell lack exported endpoint statistics for independent exact recomputation and are excluded from control totals.",
                   "Descriptive next-token changes have no quality-equivalence threshold, independent test-pool conclusion or production certification."]}


WEBSITE_PROJECTION = {
    "version": 1, "source_kind": KIND + "_summary", "full_summary_immutable": True,
    "stage_policy": "all block_output stages, final_norm and lm_head.output; every failed stage and both pair-first-marker stages; original order",
    "counts_scope": "all executed stages in the full summary, including omitted passing stages",
    "memory_tolerance": "unchanged validated FP32 protocol tolerance attached to memory fields compacted without a local tolerance in the full summary",
    "passing_samples": "first/worst/violation samples omitted only for passing memory and frozen comparisons; full details remain in the bound source summary",
    "failed_samples": "all failed stages, endpoints, memory and frozen bounded coordinate samples retained unchanged",
    "frozen_input_scope": "common N0H0 input once per case; unrendered statistic/weight identities omitted; input/oracle/check identities and gates retained"}


def website_comparison(value):
    result = {key: deepcopy(item) for key, item in value.items() if not value["passed"] or key not in SAMPLES}
    result.setdefault("tolerance", deepcopy(base.TOLERANCES["float32"]))
    return result


def website_bundle(value):
    return {**{key: deepcopy(item) for key, item in value.items() if key != "fields"}, "fields": [website_comparison(row) for row in value["fields"]]}


def project_website_view(validated_summary, source_summary):
    """Project already validated evidence; never recompute a gate from a subset."""
    base.require(validated_summary["kind"] == KIND + "_summary" and validated_summary["certified"] is False, "unknown/promoted projection source")
    closed(source_summary, ["path", "sha256"], "source summary binding")
    relative(source_summary["path"])
    base.digest(source_summary["sha256"], "full source summary")
    result = {key: deepcopy(item) for key, item in validated_summary.items() if key != "cases"}
    result.update(kind=KIND + "_website_view", source_summary=deepcopy(source_summary), projection=deepcopy(WEBSITE_PROJECTION), cases=[])
    for case in validated_summary["cases"]:
        projected = {key: deepcopy(item) for key, item in case.items() if key not in {"cells", "frozen_norms", "frozen_head"}}
        projected["cells"] = []
        for cell in case["cells"]:
            current = {key: deepcopy(item) for key, item in cell.items() if key not in {"stage_comparisons", "retained_state", "stateful_full_retained_state", "actual_prefix"}}
            current["stage_comparisons"] = {}
            for name, pairing in cell["stage_comparisons"].items():
                first_sites = {(marker["layer"], marker["stage"]) for key in ("first_nonzero", "first_violation") if (marker := pairing[key]) is not None}
                retained = [row for row in pairing["stages"] if row["stage"] == "block_output" or (row["layer"] is None and row["stage"] in {"final_norm", "lm_head.output"})
                            or not row["passed"] or (row["layer"], row["stage"]) in first_sites]
                current["stage_comparisons"][name] = {**{key: deepcopy(item) for key, item in pairing.items() if key != "stages"}, "stages": deepcopy(retained),
                    "projection_coverage": {"full_stage_count": len(pairing["stages"]), "retained_stage_count": len(retained),
                                            "omitted_passing_stages": len(pairing["stages"]) - len(retained), "retained_all_failures": True}}
            for key in ("retained_state", "stateful_full_retained_state"):
                current[key] = {name: website_bundle(row) for name, row in cell[key].items()}
            prefix = cell["actual_prefix"]
            current["actual_prefix"] = {key: deepcopy(item) for key, item in prefix.items() if key != "retained_state"}
            current["actual_prefix"]["retained_state"] = {route: {anchor: website_bundle(row) for anchor, row in bundles.items()} for route, bundles in prefix["retained_state"].items()}
            projected["cells"].append(current)
        projected["frozen_norms"] = []
        for probe in case["frozen_norms"]:
            current = {key: deepcopy(item) for key, item in probe.items() if key not in {"statistics", "weight", "output_checks", "statistic_checks"}}
            for key in ("output_checks", "statistic_checks"):
                current[key] = {name: website_comparison(row) for name, row in probe[key].items()}
            projected["frozen_norms"].append(current)
        head = case["frozen_head"]
        projected["frozen_head"] = None if head is None else {**{key: deepcopy(item) for key, item in head.items() if key != "checks"},
                                                            "checks": {name: website_comparison(row) for name, row in head["checks"].items()}}
        result["cases"].append(projected)
    return result


def build_website_view(summary_path, *, root=ROOT, source_root=ROOT):
    """Read and independently revalidate the full immutable summary before projection."""
    root = Path(root).resolve()
    summary_path = Path(summary_path).resolve()
    full, digest = base.read_json(summary_path)
    base.require(full.get("kind") == KIND + "_summary", "website source is not a full summary")
    expected = build_summary(root / relative(full["declaration"]["path"]), root=root, source_root=source_root)
    same(full, expected, "full source summary differs from independently validated raw evidence")
    return project_website_view(full, {"path": summary_path.relative_to(root).as_posix(), "sha256": digest})


def write_website_view(path, value):
    """Exclusive atomic publication, compact UTF-8 JSON with one LF terminator."""
    path = Path(path)
    base.require(not path.exists(), "website view output already exists")
    encoded = base.json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/normalization-head-protocol-2026-10-05.json")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--website-view-from", type=Path, help="Validate this full immutable summary and project a separate compact website view")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.website_view_from is not None:
        base.require(args.raw is None, "website projection takes its raw identity from the source summary")
        write_website_view(args.output, build_website_view(args.website_view_from))
        print("Validated normalization/head precision website view written.")
    else:
        base.write_summary(args.output, build_summary(args.declaration, args.raw))
        print("Validated normalization/head precision summary written.")


if __name__ == "__main__":
    main()
