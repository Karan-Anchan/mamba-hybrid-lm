"""Validate the declared read-only isolation report before exporting website data."""
from __future__ import annotations

import argparse
import math
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_checkpoint_numerics import (SOURCES, TOLERANCES, canonical_sha256,
    digest, file_sha256, flag, integer, metric, number, read_json, relative_path, require, write_summary)


def detailed(value):
    passed = metric(value, "detailed comparison")
    require(value.get("finite") is True and value.get("tolerance") == TOLERANCES["float32"], "changed tolerance/finite flag")
    shape = value["shape"]
    require(isinstance(shape, list) and len(shape) > 0, "empty shape")
    for size in shape:
        integer(size, "shape", 1)
    require(value["elements"] == math.prod(shape), "element coverage differs")
    for key in ("nonzero_error_count", "violation_count"):
        integer(value[key], key)
    require(0 <= value["violation_count"] <= value["nonzero_error_count"] <= value["elements"], "impossible error counts")
    flag(value["passed"], value["violation_count"] == 0, "violation count")
    require((value["max_absolute_error"] == 0) == (value["nonzero_error_count"] == 0), "inconsistent nonzero count")
    require(value["comparison_dtype"] in ("torch.float32", "torch.float64"), "unknown comparison dtype")
    for key, count in (("first_nonzero", value["nonzero_error_count"]), ("first_violation", value["violation_count"])):
        require((value[key] is None) == (count == 0), "first coordinate missing/extra")
    samples = [value[key] for key in ("first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio") if value[key] is not None]
    samples += value["violation_samples"]
    require(len(value["violation_samples"]) == min(8, value["violation_count"]), "violation sample coverage differs")
    for sample in samples:
        require(len(sample["coordinate"]) == len(shape), "coordinate rank differs")
        for index, size in zip(sample["coordinate"], shape):
            integer(index, "coordinate")
            require(index < size, "coordinate outside tensor")
        for key in ("actual", "reference", "absolute_error", "tolerance_ratio"):
            number(sample[key], key)
        error = abs(sample["actual"] - sample["reference"])
        # Stored FP32 metrics round their subtraction/reduction. This validates
        # reporting arithmetic only; it never alters the tensor pass threshold.
        require(math.isclose(error, sample["absolute_error"], rel_tol=2e-6, abs_tol=1e-10), "coordinate error differs")
        ratio = sample["absolute_error"] / (3e-5 + 3e-4 * abs(sample["reference"]))
        require(math.isclose(ratio, sample["tolerance_ratio"], rel_tol=2e-6, abs_tol=1e-9), "coordinate ratio differs")
    require(all(sample["tolerance_ratio"] > 1 for sample in value["violation_samples"]), "nonviolating violation sample")
    require(value["worst_absolute_error"]["absolute_error"] == value["max_absolute_error"]
            and value["worst_tolerance_ratio"]["tolerance_ratio"] == value["max_tolerance_ratio"], "worst coordinate differs from maximum")
    return passed


def expected_stages(cfg):
    stages = [(None, "embedding")]
    for layer, kind in enumerate(cfg["layer_types"]):
        names = ["block_input", "norm1_output"]
        if kind == "mamba":
            names += ["in_projection", "convolution", *["scan_input." + name for name in ("x", "dt", "A", "B", "C", "D")],
                      "scan_output", "gated_norm_input", "gated_norm_output"]
        else:
            names += ["attention_qkv"]
        names += ["mixer_output"]
        if cfg["mlp_on_every_layer"]:
            names += ["norm2_output", "mlp_output"]
        names += ["block_output"]
        stages.extend((layer, name) for name in names)
    return [*stages, (None, "final_norm"), (None, "logits")]


def first_record(trace, field):
    item = next((item for item in trace if item[field] > 0), None)
    keys = ("layer", "stage", "axes", "nonzero_error_count", "violation_count", "first_nonzero", "first_violation")
    return None if item is None else {key: item[key] for key in keys}


def validate(report, declaration, declaration_hash, baseline, source_root=ROOT):
    require(report.get("schema") == 1 and report.get("kind") == "trained_checkpoint_operation_isolation"
            and report.get("execution_status") == "completed" and report.get("certified") is False, "incomplete or unknown report")
    p = report["protocol"]
    require(report["protocol_sha256"] == canonical_sha256(p), "protocol fingerprint differs")
    require(p["isolation_declaration"]["sha256"] == declaration_hash, "declaration fingerprint differs")
    require(declaration["status_at_declaration"] == "planned_before_execution", "undeclared study")
    for key in ("seed", "length", "prefix_length", "chunk_size", "coordinate_limit", "ratios", "device", "tolerances"):
        require(p[key] == declaration[key], "declaration differs: " + key)
    require(p["tolerances"] == TOLERANCES and p["coordinate_limit"] == 8, "changed reporting policy")
    require(p["production_defaults_changed"] is False and p["optimizer_executed"] is False, "unexpected mutation/training")
    require(p["input"] == baseline["protocol"]["shared_inputs"][0], "baseline input differs")
    for key in ("start_token", "tokens_sha256", "targets_sha256"):
        require(p["input"][key] == declaration[key], "declared input differs")
    require(p["baseline_report"]["sha256"] == declaration["baseline_sha256"], "baseline fingerprint differs")
    require(p["baseline_declaration"]["path"] == declaration["baseline_declaration"]
            and p["baseline_declaration"]["sha256"] == file_sha256(Path(source_root) / declaration["baseline_declaration"]), "baseline declaration differs")
    require(p["data"] == baseline["protocol"]["data"], "baseline data identity differs")
    require(p["runtime_sha256"] == canonical_sha256(p["runtime"]), "runtime fingerprint differs")
    require(p["runtime"]["precision_flags"]["cuda_matmul_allow_tf32"] is False
            and p["runtime"]["precision_flags"]["cudnn_allow_tf32"] is False, "TF32 enabled")
    require(set(p["source_sha256"]) == SOURCES | {"scripts/isolate_checkpoint_numerics.py"}, "source coverage differs")
    for name, recorded in p["source_sha256"].items():
        digest(recorded, name)
        raw = (Path(source_root) / relative_path(name)).read_bytes().replace(b"\r\n", b"\n")
        require(recorded in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()}, "stale measured source: " + name)
    checkpoints = {item["ratio"]: item for item in baseline["protocol"]["checkpoints"]}
    require(p["checkpoints"] == [checkpoints[ratio] for ratio in p["ratios"]], "checkpoint identity differs")
    cases = report["cases"]
    require([item["ratio"] for item in cases] == p["ratios"], "missing/extra/unordered cases")
    for case in cases:
        require(all(case[key] == p["input"][key] for key in p["input"]), "case input identity differs")
        old = next(item for item in baseline["cases"] if item["ratio"] == case["ratio"])
        require(case["weights_sha256"] == old["weights_sha256"], "weight identity differs")
        require(case["baseline_fp32_logits"] == old["fp32_backend"]["logits"], "baseline logits differ")
        trace = case["trace"]
        require([(item["layer"], item["stage"]) for item in trace] == expected_stages(checkpoints[case["ratio"]]["model_config"]), "trace coverage differs")
        for item in trace:
            detailed(item)
        require(case["logits"] == trace[-1], "endpoint differs from trace")
        require(case["first_nonzero_drift"] == first_record(trace, "nonzero_error_count"), "first drift differs")
        require(case["first_tolerance_violation"] == first_record(trace, "violation_count"), "first violation differs")
        flag(case["endpoint_passed"], trace[-1]["passed"], "endpoint")
        flag(case["baseline_pass_flag_reproduced"], trace[-1]["passed"] == old["fp32_backend"]["logits"]["passed"], "reproduced endpoint")
        flag(case["all_trace_checks_passed"], all(item["passed"] for item in trace), "whole trace")
        layers = case["isolation_layers"]
        require(1 <= len(layers) <= 2 and len(set(layers)) == len(layers), "invalid replay layers")
        require([item["layer"] for item in case["frozen_replays"]] == layers, "replay coverage differs")
        frozen_passes = []
        for replay in case["frozen_replays"]:
            require(replay["real_prefix"]["position"] == 128 and replay["real_prefix"]["synthetic"] is False
                    and replay["real_prefix"]["unchanged_after_cloned_continuation"] is True, "not genuine untouched prefix")
            require(set(replay["operand_pair_checks"]) == {"x", "dt", "A", "B", "C", "D"}, "operand coverage differs")
            for item in replay["operand_pair_checks"].values():
                detailed(item)
            for scope in ("full_zero_state", "real_prefix_continuation"):
                frozen = replay[scope]
                suffix = scope == "real_prefix_continuation"
                expected = {"stateful_one_shot", "chunked128", "tokenwise"} | (set() if suffix else {"quadratic"})
                require(set(frozen["treatments"]) == expected, "frozen route coverage differs")
                require(frozen["initial_state"]["synthetic_random"] is False, "synthetic prefix")
                require(frozen["initial_state"]["sha256"] == replay["real_prefix"]["ssm_sha256"] if suffix else frozen["initial_state"]["nonzero_elements"] == 0, "initial memory differs")
                comparison_passes = []
                for name, item in frozen["treatments"].items():
                    comparison_passes.append(detailed(item["output_vs_fp64"]))
                    require((item["state_vs_fp64"] is None) == (name == "quadratic"), "missing/extra state comparison")
                    if item["state_vs_fp64"] is not None:
                        comparison_passes.append(detailed(item["state_vs_fp64"]))
                for item in [*frozen["internal_pairs"].values(), frozen["replay_vs_observed_scan"], *frozen["operation_probes"].values()]:
                    comparison_passes.append(detailed(item))
                contrasts = frozen["exploratory_decay_contrasts"]
                require(set(contrasts) == (set() if suffix else {"fp64_cumsum_decay", "fp64_product_cumsum_decay"}), "decay contrast coverage differs")
                for item in contrasts.values():
                    require(item["exploratory"] is True and item["production"] is False, "promoted exploratory treatment")
                    comparison_passes.append(detailed(item["output_vs_fp64"]))
                    comparison_passes.append(detailed(item["output_vs_quadratic"]))
                flag(frozen["all_comparisons_passed"], all(comparison_passes), "frozen aggregate")
                frozen_passes.append(all(comparison_passes))
        flag(case["all_frozen_checks_passed"], all(frozen_passes), "case frozen aggregate")
        expected_candidates = []
        for replay in case["frozen_replays"]:
            frozen = replay["full_zero_state"]
            baseline_error = frozen["treatments"]["quadratic"]["output_vs_fp64"]["max_absolute_error"]
            for name, contrast in frozen["exploratory_decay_contrasts"].items():
                error = contrast["output_vs_fp64"]["max_absolute_error"]
                if error < baseline_error:
                    expected_candidates.append((replay["layer"], name, baseline_error, error))
        require([(item["layer"], item["candidate"], item["baseline_max_absolute_error"], item["candidate_max_absolute_error"])
                 for item in case["candidate_experiments"]] == expected_candidates, "candidate evidence differs")
        require(all(item["approved_for_production"] is False for item in case["candidate_experiments"]), "unexpected candidate approval")
    expected_status = "completed" if all(item["all_trace_checks_passed"] and item["all_frozen_checks_passed"] for item in cases) else "completed_with_parity_failures"
    require(report["status"] == expected_status, "report status differs")


def build_summary(declaration_path, root=ROOT, source_root=ROOT):
    declaration, declaration_hash = read_json(declaration_path)
    raw_path = Path(root) / relative_path(declaration["output"])
    report, raw_hash = read_json(raw_path)
    baseline_path = Path(root) / relative_path(declaration["baseline_report"])
    baseline, baseline_hash = read_json(baseline_path)
    require(baseline_hash == declaration["baseline_sha256"], "baseline bytes differ")
    validate(report, declaration, declaration_hash, baseline, source_root)
    rows = [{key: case[key] for key in ("ratio", "role", "weights_sha256", "first_nonzero_drift", "first_tolerance_violation",
             "logits", "isolation_layers", "frozen_replays", "candidate_experiments", "processing_seconds", "cuda_peak_allocated_bytes")}
            for case in report["cases"]]
    return {"schema": 1, "kind": "trained_checkpoint_operation_isolation_summary", "date": declaration["date"],
            "status": report["status"], "execution_status": "completed", "certified": False,
            "declaration": {"path": Path(declaration_path).resolve().relative_to(Path(root).resolve()).as_posix(), "sha256": declaration_hash},
            "raw_report": {"path": declaration["output"], "sha256": raw_hash}, "protocol_sha256": report["protocol_sha256"],
            "protocol": report["protocol"], "rows": rows, "limits": report["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/numerical-isolation-protocol-2026-10-04.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_summary(args.output, build_summary(args.declaration))
    print("Validated isolation summary written.")


if __name__ == "__main__":
    main()
