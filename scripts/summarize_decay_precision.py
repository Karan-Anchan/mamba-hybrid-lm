"""Validate the declared FP64 decay contrast and export uncertified evidence.

Standard library only. Internal agreement and agreement with the original FP32
reference are separate checks. A null self-anchor is never counted as a pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_checkpoint_numerics as base  # noqa: E402
from scripts import summarize_numerical_isolation as isolation  # noqa: E402

KIND = "trained_checkpoint_decay_precision"
TREATMENT = "fp64_cumsum_decay_coefficients"
TREATMENTS = ["original", TREATMENT]
BASE_DECLARATION = "docs/research/checkpoint-numerics-protocol-2026-10-04.json"
ISOLATION_DECLARATION = "docs/research/numerical-isolation-protocol-2026-10-04.json"
SOURCES = base.SOURCES | {"scripts/study_decay_precision.py", "scripts/isolate_checkpoint_numerics.py"}
BF16_STAGES = ["bf16_full", "bf16_stateful_full", "bf16_prefix_prefill", "bf16_suffix_one_shot",
               "bf16_suffix_segmented", "bf16_suffix_tokenwise", "bf16_full_tokenwise"]
POLICY = {"dt_times_A": "original FP32 product", "cumsum_subtraction_exp": "FP64",
          "decay_carry_end_coefficients": "cast to FP32 before unchanged contractions",
          "contractions_and_autocast": "original policy preserved", "production": False}
ANCHOR_SCOPE = "separate comparison against original reference FP32 full forward/backward"


def same(actual, expected, label):
    """Canonical identities also distinguish JSON booleans from numbers."""
    base.require(base.canonical_sha256(actual) == base.canonical_sha256(expected), label)


def public_binding(binding, path, digest, label, protocol_digest=None):
    expected = {"path_scope": "repository-relative", "path": path, "sha256": digest}
    if protocol_digest is not None:
        expected["protocol_sha256"] = protocol_digest
    same(binding, expected, label + " identity differs")


def gradients(value, cfg, label):
    same(value["tolerance"], base.TOLERANCES["float32"], label + " tolerance changed")
    inventory = base.parameter_inventory(cfg)
    checks = value["checks"]
    names = [check["parameter"] for check in checks]
    base.require(len(names) == len(set(names)) and set(names) == set(inventory),
                 label + " missing/duplicate unique parameter gradients")
    base.integer(value["parameter_tensors"], label + " parameter count", 1)
    base.require(value["parameter_tensors"] == len(inventory), label + " parameter count differs")
    passes = []
    for check in checks:
        same(check["shape"], inventory[check["parameter"]], label + " parameter shape differs")
        passes.append(base.metric(check, label + "." + check["parameter"]))
        if check["finite"]:
            base.number(check["actual_l2"], label + " actual norm", 0)
            base.number(check["reference_l2"], label + " reference norm", 0)
            if check["actual_l2"] and check["reference_l2"]:
                base.number(check["cosine_similarity"], label + " cosine")
                base.require(-1 <= check["cosine_similarity"] <= 1, label + " cosine outside range")
            else:
                base.require(check["cosine_similarity"] is None, label + " zero norm cosine must be null")
        else:
            base.require(all(check[name] is None for name in ("actual_l2", "reference_l2", "cosine_similarity")),
                         label + " nonfinite gradient statistics must be null")
    same(value["failed_parameters"], [check["parameter"] for check in checks if not check["passed"]],
         label + " failed parameter names differ")
    base.flag(value["all_passed"], all(passes), label + " aggregate")
    return passes


def comparison(value, cfg, targets, label):
    same(value["tolerance"], base.TOLERANCES["float32"], label + " tolerance changed")
    logits, loss = base.metric(value["logits"], label + " logits"), base.metric(value["loss"], label + " loss")
    grad = gradients(value["gradients"], cfg, label + " gradients")
    base.validate_effect(value["next_token_effects"], targets, 0, label + " descriptive token effects")
    passed = logits and loss and all(grad)
    base.flag(value["passed"], passed, label + " aggregate")
    return passed


def operators(arm, cfg, length, chunk):
    stages = {"fp32_reference_forward": Counter({length: 1}),
              "fp32_torch_chunked_forward": Counter(min(chunk, length - start) for start in range(0, length, chunk))}
    layers = [layer for layer, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]
    seen = {}
    paths = {}
    for event in arm["operator_observations"]:
        same(sorted(event), sorted(["autocast", "input_dtypes", "layer", "length", "path", "stage", "calls"]),
             "operator fields differ")
        stage, layer = event["stage"], event["layer"]
        base.require(stage in stages and type(layer) is int and layer in layers, "unexpected operator stage/layer")
        path = "torch.quadratic_ssd" if stage == "fp32_reference_forward" else "torch.chunked_ssd"
        base.require(event["path"] == path and event["autocast"] is False, "operator path/precision differs")
        same(event["input_dtypes"], ["torch.float32"] * (7 if path == "torch.chunked_ssd" else 6),
             "operator operand dtypes differ")
        base.integer(event["length"], "operator length", 1)
        base.integer(event["calls"], "operator calls", 1)
        key = (stage, layer, event["length"])
        base.require(key not in seen, "duplicate operator observation")
        seen[key] = event["calls"]
        paths[stage] = [path]
    expected = {(stage, layer, size): count for stage, parts in stages.items()
                for layer in layers for size, count in parts.items()}
    base.require(seen == expected, "missing/incorrect actual scan calls")
    paths.update(dict.fromkeys(BF16_STAGES))
    same(arm["actual_paths"], paths, "actual/unexecuted paths differ")


def anchor_identity(value, cfg):
    same(sorted(value), sorted(["treatment", "backend", "logits_sha256", "loss_sha256", "parameter_tensors", "gradients"]),
         "original anchor identity fields differ")
    base.require(value["treatment"] == "original" and value["backend"] == "reference", "anchor is not original reference")
    for field in ("logits_sha256", "loss_sha256"):
        base.digest(value[field], "original anchor " + field)
    inventory = base.parameter_inventory(cfg)
    base.integer(value["parameter_tensors"], "original anchor gradient count", 1)
    base.require(value["parameter_tensors"] == len(inventory), "anchor gradient count differs")
    names = [item["parameter"] for item in value["gradients"]]
    base.require(len(names) == len(set(names)) and set(names) == set(inventory), "anchor unique gradient coverage differs")
    for item in value["gradients"]:
        same(sorted(item), sorted(["parameter", "shape", "dtype", "sha256"]), "anchor gradient fields differ")
        same(item["shape"], inventory[item["parameter"]], "anchor gradient shape differs")
        base.require(item["dtype"] == "torch.float32", "anchor gradient dtype differs")
        base.digest(item["sha256"], "original anchor gradient")


def validate_case(case, protocol, checkpoint, old, targets):
    cfg = checkpoint["model_config"]
    for name, value in protocol["input"].items():
        same(case[name], value, "case input identity differs")
    same(case["weights_sha256"], old["weights_sha256"], "case weight identity differs")
    base.number(case["processing_seconds"], "diagnostic processing seconds", 0)
    base.integer(case["cuda_peak_allocated_bytes"], "diagnostic CUDA peak", 1)
    anchor_identity(case["common_original_fp32_anchor"], cfg)
    same(case["baseline_fp32_logits"], old["fp32_backend"]["logits"], "saved baseline logits differ")
    arms = case["treatments"]
    same([arm["treatment_id"] for arm in arms], TREATMENTS, "missing/duplicate/unordered treatment matrix")
    same(arms[0]["fp32_backend"], old["fp32_backend"], "original full FP32 numerical fields differ from baseline")
    original_chunked = arms[0]["common_original_fp32_anchor"]["torch_chunked"]
    for name in ("logits", "loss", "gradients", "next_token_effects"):
        same(original_chunked[name], old["fp32_backend"][name], "original chunked anchor fields differ from baseline")
    original_logprobs = old["fp32_backend"]["next_token_effects"]["per_token"]["reference_logprob"]
    for arm in arms:
        same(arm["weights_sha256"], case["weights_sha256"], "treatment changed weight identity")
        base.require(arm["include_bf16"] is False and arm["bf16_cached"] is None, "BF16 must remain explicitly unexecuted")
        same(arm["unexecuted_stages"], BF16_STAGES, "unexecuted BF16 coverage differs")
        bindings = {"reference_scan": "src.model.mamba2.ssd", "stateful_scan": "src.model.mamba2.ssd_stateful"}
        if arm["treatment_id"] == TREATMENT:
            module = arm["scan_function_bindings"]["reference_scan"].rsplit(".", 1)[0]
            base.require(module in ("__main__", "scripts.study_decay_precision"), "unknown treatment function module")
            bindings = {"reference_scan": module + ".ssd_decay_fp64", "stateful_scan": module + ".ssd_stateful_decay_fp64"}
        same(arm["scan_function_bindings"], bindings, "scan treatment bindings differ")
        fp32 = arm["fp32_backend"]
        base.validate_backend(fp32["reference_backend"], "reference", cfg, protocol["chunk_size"])
        base.validate_backend(fp32["candidate_backend"], "torch_chunked", cfg, protocol["chunk_size"])
        same(fp32["observed_paths"], {"prefill": None, "decode": None,
             "backward": "autograd through each recorded forward scan; no separate kernel probe"}, "stateful qualification was not executed")
        internal_pass = comparison(fp32, cfg, targets, arm["treatment_id"] + " internal")
        operators(arm, cfg, protocol["length"], protocol["chunk_size"])
        anchors = arm["common_original_fp32_anchor"]
        same(sorted(anchors), ["reference", "torch_chunked"], "missing original anchor routes")
        anchor_passes = []
        for route, check in anchors.items():
            if arm["treatment_id"] == "original" and route == "reference":
                base.require(check is None, "original reference self-anchor must remain null, not a measured pass")
                continue
            base.require(isinstance(check, dict) and check["scope"] == ANCHOR_SCOPE, "anchor scope differs")
            anchor_passes.append(comparison(check, cfg, targets, arm["treatment_id"] + " original anchor " + route))
            same(check["next_token_effects"]["per_token"]["reference_logprob"], original_logprobs,
                 "common original FP32 anchor token probabilities differ")
        internal_effect = fp32["next_token_effects"]["per_token"]
        same(internal_effect["actual_logprob"], anchors["torch_chunked"]["next_token_effects"]["per_token"]["actual_logprob"],
             "chunked treatment token identities differ between internal and anchor comparisons")
        if arm["treatment_id"] == TREATMENT:
            same(internal_effect["reference_logprob"], anchors["reference"]["next_token_effects"]["per_token"]["actual_logprob"],
                 "reference treatment token identities differ between internal and anchor comparisons")
        base.flag(arm["passed"], internal_pass and all(anchor_passes), "treatment aggregate")
    reproduced = arms[0]["fp32_backend"]["logits"]["passed"] == old["fp32_backend"]["logits"]["passed"]
    base.flag(case["baseline_pass_flag_reproduced"], reproduced, "baseline reproduction")
    base.flag(case["passed"], all(arm["passed"] for arm in arms) and reproduced, "case aggregate")


def validate(report, declaration, declaration_hash, declaration_relative, baseline, isolation_report, isolation_hash, source_root):
    base.require(type(report["schema"]) is int and report["schema"] == 1 and report["kind"] == KIND
                 and report["execution_status"] == "completed" and report["certified"] is False, "incomplete/unknown/certified report")
    expected = {"schema": 1, "status_at_declaration": "planned_before_execution", "device": "cuda", "seed": 2027,
                "length": 257, "prefix_length": 128, "chunk_size": 128, "batch_size": 1, "ratios": base.RATIOS,
                "include_bf16": False, "treatment_id": TREATMENT, "tolerances": base.TOLERANCES,
                "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}}
    for name, value in expected.items():
        same(declaration[name], value, "declaration differs: " + name)
    p = report["protocol"]
    same(report["protocol_sha256"], base.canonical_sha256(p), "protocol fingerprint differs")
    for name in ("device", "seed", "length", "prefix_length", "chunk_size", "ratios", "include_bf16", "treatment_id", "tolerances"):
        same(p[name], expected[name], "protocol differs: " + name)
    base.require(type(p["version"]) is int and p["version"] == 1 and p["full_tokenwise"] is None,
                 "protocol version or unexecuted tokenwise flag differs")
    same(p["treatment_policy"], POLICY, "arithmetic treatment identity differs")
    base.require(p["optimizer_executed"] is False and p["production_defaults_changed"] is False
                 and p["quality_equivalence_margin"] is None, "unexpected training/promotion/quality margin")
    public_binding(p["declaration"], declaration_relative, declaration_hash, "declaration")
    public_binding(p["baseline_declaration"], BASE_DECLARATION,
                   base.file_sha256(Path(source_root) / BASE_DECLARATION), "baseline declaration")
    base.require(declaration["baseline_declaration"] == BASE_DECLARATION, "baseline declaration path differs")
    public_binding(p["baseline_report"], declaration["baseline_report"], declaration["baseline_sha256"],
                   "baseline", baseline["protocol_sha256"])
    public_binding(p["isolation_report"], declaration["isolation_report"], isolation_hash,
                   "isolation", isolation_report["protocol_sha256"])
    same(p["input"], baseline["protocol"]["shared_inputs"][0], "baseline shared input differs")
    for name in ("start_token", "tokens_sha256", "targets_sha256"):
        same(p["input"][name], declaration[name], "declared input differs: " + name)
    for name in ("data", "checkpoints", "runtime"):
        same(p[name], baseline["protocol"][name], "baseline " + name + " identity differs")
    same(p["runtime"], isolation_report["protocol"]["runtime"], "isolation runtime identity differs")
    same(p["runtime_sha256"], base.canonical_sha256(p["runtime"]), "runtime fingerprint differs")
    same(sorted(p["source_sha256"]), sorted(SOURCES), "source coverage differs")
    for name, recorded in p["source_sha256"].items():
        base.digest(recorded, "source " + name)
        raw = (Path(source_root) / base.relative_path(name)).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        base.require(recorded in {hashlib.sha256(item).hexdigest() for item in (raw, lf, lf.replace(b"\n", b"\r\n"))},
                     "stale measured source: " + name)
        prior = isolation_report["protocol"]["source_sha256"] if name == "scripts/isolate_checkpoint_numerics.py" else baseline["protocol"]["source_sha256"]
        if name in prior:
            same(recorded, prior[name], "measured source differs from prior evidence: " + name)
    same([case["ratio"] for case in report["cases"]], base.RATIOS, "missing/duplicate/unordered model matrix")
    checkpoints = {item["ratio"]: item for item in p["checkpoints"]}
    targets = baseline["cases"][0]["fp32_backend"]["next_token_effects"]["per_token"]["target_ids"]
    for case, old in zip(report["cases"], baseline["cases"]):
        validate_case(case, p, checkpoints[case["ratio"]], old, targets)
    status = "completed" if all(case["passed"] for case in report["cases"]) else "completed_with_parity_failures"
    base.require(report["status"] == status, "report status hides failed original checks")


def compact(value):
    if value is None:
        return None
    return {"logits": value["logits"], "loss": value["loss"], "gradients": {
        **base.count_checks(value["gradients"]["checks"]), "failed_parameters": value["gradients"]["failed_parameters"]},
        "passed": value["passed"]}


def build_summary(declaration_path, raw_path=None, root=ROOT, source_root=ROOT):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    declaration, declaration_hash = base.read_json(declaration_path)
    declaration_relative = declaration_path.relative_to(root).as_posix()
    declared_raw = root / base.relative_path(declaration["output"])
    raw_path = declared_raw if raw_path is None else Path(raw_path).resolve()
    base.require(raw_path == declared_raw.resolve(), "raw path differs from immutable declaration")
    report, raw_hash = base.read_json(raw_path)
    baseline, baseline_hash = base.read_json(root / base.relative_path(declaration["baseline_report"]))
    same(baseline_hash, declaration["baseline_sha256"], "immutable baseline bytes differ")
    baseline_declaration, _ = base.read_json(root / BASE_DECLARATION)
    base.validate_report(baseline, baseline_declaration, source_root)
    isolated, isolation_hash = base.read_json(root / base.relative_path(declaration["isolation_report"]))
    same(isolation_hash, declaration["isolation_sha256"], "immutable isolation bytes differ")
    isolated_declaration, isolated_declaration_hash = base.read_json(root / ISOLATION_DECLARATION)
    base.require(isolated_declaration["output"] == declaration["isolation_report"], "isolation report path differs")
    same(isolated_declaration["baseline_sha256"], baseline_hash, "isolation baseline bytes differ")
    public_binding(isolated["protocol"]["baseline_report"], declaration["baseline_report"], baseline_hash,
                   "isolation baseline", baseline["protocol_sha256"])
    public_binding(isolated["protocol"]["isolation_declaration"], ISOLATION_DECLARATION,
                   isolated_declaration_hash, "isolation declaration")
    isolation.validate(isolated, isolated_declaration, isolated_declaration_hash, baseline, source_root)
    validate(report, declaration, declaration_hash, declaration_relative, baseline, isolated, isolation_hash, source_root)
    rows = []
    for case in report["cases"]:
        for arm in case["treatments"]:
            anchors = arm["common_original_fp32_anchor"]
            rows.append({"ratio": case["ratio"], "treatment": arm["treatment_id"], "weights_sha256": case["weights_sha256"],
                         "internal": compact(arm["fp32_backend"]),
                         "original_fp32_anchor": {route: compact(value) for route, value in anchors.items()},
                         "next_token_effects": {"internal": base.compact_effect(arm["fp32_backend"]["next_token_effects"]),
                            **{"original_anchor_" + route: None if value is None else base.compact_effect(value["next_token_effects"])
                               for route, value in anchors.items()}}, "passed": arm["passed"]})
    resources = [{"ratio": case["ratio"], "processing_seconds": case["processing_seconds"],
                  "peak_allocated_mib": case["cuda_peak_allocated_bytes"] / 1048576,
                  "scope": "combined original and treatment audit/forward/backward diagnostics; not training or throughput"}
                 for case in report["cases"]]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"],
            "execution_status": "completed", "certified": False, "production_defaults_changed": False,
            "optimizer_executed": False, "quality_equivalence_margin": None,
            "declaration": {"path": declaration_relative, "sha256": declaration_hash},
            "raw_report": {"path": declaration["output"], "sha256": raw_hash, "cases": len(report["cases"])},
            "baseline_report": report["protocol"]["baseline_report"], "isolation_report": report["protocol"]["isolation_report"],
            "protocol_sha256": report["protocol_sha256"], "protocol": report["protocol"],
            "original_controls": [{"ratio": case["ratio"], "exact_fp32_fields_equal": True,
                                   "original_anchor_identity": case["common_original_fp32_anchor"]} for case in report["cases"]],
            "rows": rows, "treatment_counts": {treatment: {"passed": sum(row["passed"] for row in rows if row["treatment"] == treatment),
                                                   "total": sum(row["treatment"] == treatment for row in rows)} for treatment in TREATMENTS},
            "coverage": {"models": 3, "treatments_per_model": 2, "batch_size": 1, "length": 257, "chunk_size": 128,
                         "fp32_full_forward_and_backward": True, "bf16_executed": False,
                         "retained_state_comparisons_executed": False, "cached_continuation_executed": False,
                         "original_reference_self_anchor": None, "unexecuted_bf16_stages": BF16_STAGES},
            "diagnostic_resources": resources,
            "limits": [*report["limits"],
                       "Original reference self-anchor is null: comparing a tensor with itself supplies no independent measurement.",
                       "All unique learned parameter tensors are checked; a gradient pass does not replace the forward-logit gate.",
                       "Saved tensor metrics cannot be recomputed without tensors; saved token effects and counts are independently validated.",
                       "Whole-model passes on this one declared FP32 workload do not certify a backend or qualify BF16/cache routes."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/decay-precision-protocol-2026-10-04.json")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        base.write_summary(args.output, build_summary(args.declaration, args.raw, args.root, args.source_root))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Validated decay summary written; certified=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
