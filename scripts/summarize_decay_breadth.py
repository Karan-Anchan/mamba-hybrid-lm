"""Strict standard-library exporter for declared, uncertified FP32 breadth evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_checkpoint_numerics as base  # noqa: E402
from scripts import summarize_decay_precision as decay  # noqa: E402

KIND = "trained_checkpoint_decay_breadth"
SOURCES = decay.SOURCES | {"scripts/check_decay_breadth.py"}
CASES = [{"batch_size": batch, "length": length, "prefix_length": min(128, length // 2), "gradient_check": batch == 2}
         for batch, length in ((1, 127), (1, 128), (1, 129), (1, 257), (2, 512))]
CACHED = ["stateful_full", "prefill_full", "prefix", "suffix_one_shot", "suffix_chunked", "suffix_tokenwise", "full_tokenwise"]
STATEFUL = ["stateful_full", "prefill_full", "suffix_one_shot", "suffix_chunked", "suffix_tokenwise", "full_tokenwise"]
INTERNAL = [*CACHED, "suffix_chunked_vs_one_shot", "suffix_tokenwise_vs_one_shot"]


def same(actual, expected, label):
    decay.same(actual, expected, label)


def identifier(case):
    return f"b{case['batch_size']}-l{case['length']}-p{case['prefix_length']}"


def tensor_digest(values):
    return hashlib.sha256(struct.pack("<" + "q" * len(values), *values)).hexdigest()


@lru_cache(maxsize=64)
def recover_first_token(targets, expected_hash, vocabulary):
    """Cache only a public, content-addressed vocabulary search across exports."""
    tail = struct.pack("<" + "q" * (len(targets) - 1), *targets[:-1])
    return next((token for token in range(vocabulary)
                 if hashlib.sha256(struct.pack("<q", token) + tail).hexdigest() == expected_hash), None)


def batch_slice(values, batch, length, start, stop):
    base.require(len(values) == batch * length, "flattened batch coverage differs")
    return [value for row in range(batch) for value in values[row * length + start:row * length + stop]]


def validate_effect(effect, targets, batch, start, span, label):
    """Positions apply to EACH row; saved arrays flatten rows in batch-major order."""
    same(effect["scored_tokens"], batch * span, label + " batch token count differs")
    same(effect["position_start"], start, label + " position start differs")
    same(effect["position_stop_exclusive"], start + span, label + " per-row position stop differs")
    base.require(len(targets) == batch * span, label + " shifted target slice differs")
    # Reuse all standard-library arithmetic checks after adapting only the helper's
    # single-row span expectation. No stored numeric metric or model gate changes.
    normalized = deepcopy(effect)
    normalized["position_stop_exclusive"] = start + len(targets)
    base.validate_effect(normalized, targets, start, label)


def inventory(cfg, batch, length):
    return {name: ([batch, *shape[1:]], "torch.float32") for name, (shape, _dtype) in base.state_inventory(cfg, length).items()}


def state_check(value, cfg, batch, length, label):
    same(value["tolerance"], base.TOLERANCES["float32"], label + " state tolerance changed")
    return base.validate_state(value, inventory(cfg, batch, length), label)


def identity_fields(fields, cfg, batch, length, label):
    expected = inventory(cfg, batch, length)
    names = [field["field"] for field in fields]
    base.require(len(names) == len(set(names)) and set(names) == set(expected), label + " identity fields missing/duplicate")
    for field in fields:
        same(sorted(field), sorted(["field", "shape", "dtype", "sha256", "nonzero_elements"]), label + " identity schema differs")
        shape, dtype = expected[field["field"]]
        same(field["shape"], shape, label + " state shape differs")
        same(field["dtype"], dtype, label + " state dtype differs")
        base.digest(field["sha256"], label + " state")
        base.integer(field["nonzero_elements"], label + " nonzero count")
        base.require(field["nonzero_elements"] <= math.prod(shape), label + " impossible nonzero count")
        if field["field"].endswith(".ssm"):
            base.require(field["nonzero_elements"] > 0, label + " genuine prefix/anchor SSM memory is zero-only")


def comparison(value, cfg, targets, case, label):
    same(value["tolerance"], base.TOLERANCES["float32"], label + " tolerance changed")
    checks = [base.metric(value[name], label + "." + name) for name in ("logits", "loss")]
    if case["gradient_check"]:
        checks.extend(decay.gradients(value["gradients"], cfg, label + " gradients"))
    else:
        base.require(value["gradients"] is None, label + " gradients must be unexecuted")
    validate_effect(value["next_token_effects"], targets, case["batch_size"], 0, case["length"], label + " effects")
    base.flag(value["passed"], all(checks), label)
    return all(checks)


def bounds(name, length, prefix):
    if name == "prefill_full":
        return length - 1, length
    if name == "prefix":
        return prefix - 1, prefix
    return (prefix, length) if name.startswith("suffix_") else (0, length)


def operators(arm, cfg, case, chunk):
    length, prefix = case["length"], case["prefix_length"]
    suffix = length - prefix
    partition = lambda size: Counter(min(chunk, size - start) for start in range(0, size, chunk))
    stages = {"fp32_reference_full": Counter({length: 1}), "fp32_torch_chunked_full": partition(length),
              "fp32_stateful_full": partition(length), "fp32_prefill_full": partition(length),
              "fp32_prefix_prefill": partition(prefix), "fp32_suffix_one_shot": partition(suffix),
              "fp32_suffix_chunked": partition(suffix), "fp32_suffix_tokenwise": Counter({1: suffix}),
              "fp32_full_tokenwise": Counter({1: length})}
    layers = [layer for layer, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]
    seen, paths = {}, {}
    for event in arm["operator_observations"]:
        same(sorted(event), sorted(["stage", "layer", "path", "length", "autocast", "input_dtypes", "calls"]), "operator schema differs")
        stage, layer = event["stage"], event["layer"]
        base.require(stage in stages and type(layer) is int and layer in layers, "unexpected actual scan stage/layer")
        path = "torch.quadratic_ssd" if stage == "fp32_reference_full" else "torch.chunked_ssd"
        same(event["path"], path, "actual scan route differs")
        base.require(event["autocast"] is False, "unexpected autocast")
        same(event["input_dtypes"], ["torch.float32"] * (6 if path == "torch.quadratic_ssd" else 7), "operand precision differs")
        base.integer(event["length"], "scan chunk length", 1)
        base.integer(event["calls"], "scan calls", 1)
        key = stage, layer, event["length"]
        base.require(key not in seen, "duplicate scan observation")
        seen[key], paths[stage] = event["calls"], [path]
    expected = {(stage, layer, size): calls for stage, sizes in stages.items() for layer in layers for size, calls in sizes.items()}
    base.require(seen == expected, "actual scan coverage differs")
    same(arm["actual_paths"], paths, "actual stage paths differ")


def validate_input_targets(targets, shared, vocabulary):
    batch, length = shared["batch_size"], shared["length"]
    base.require(isinstance(targets, list) and len(targets) == batch * length, "full target batch coverage differs")
    base.require(all(type(token) is int and 0 <= token < vocabulary for token in targets), "target IDs outside vocabulary")
    same(tensor_digest(targets), shared["targets_sha256"], "aggregate shifted-target hash differs")
    reconstructed = []
    for row, window in enumerate(shared["windows"]):
        values = targets[row * length:(row + 1) * length]
        same(tensor_digest(values), window["targets_sha256"], "window shifted-target hash differs")
        # The shifted targets reveal every input token except the first. Recover
        # that bounded vocabulary ID by matching the already-declared input SHA.
        first = recover_first_token(tuple(values), window["tokens_sha256"], vocabulary)
        base.require(first is not None, "window input hash does not match shifted targets")
        reconstructed.extend([first, *values[:-1]])
    same(tensor_digest(reconstructed), shared["tokens_sha256"], "aggregate input hash differs")


def validate_arm(arm, cfg, case, targets, original_lp):
    batch, length, prefix = case["batch_size"], case["length"], case["prefix_length"]
    policy = arm["treatment_id"]
    expected_bindings = {"reference_scan": "src.model.mamba2.ssd", "stateful_scan": "src.model.mamba2.ssd_stateful"}
    if policy == decay.TREATMENT:
        expected_bindings = {"reference_scan": "scripts.study_decay_precision.ssd_decay_fp64",
                             "stateful_scan": "scripts.study_decay_precision.ssd_stateful_decay_fp64"}
    same(arm["scan_function_bindings"], expected_bindings, "arithmetic treatment bindings differ")
    for name, backend in (("reference_backend", "reference"), ("candidate_backend", "torch_chunked")):
        base.validate_backend(arm[name], backend, cfg, 128)
    checks = [comparison(arm["fp32_backend"], cfg, targets, case, policy + " internal")]
    anchors = arm["common_original_fp32_anchor"]
    same(sorted(anchors), ["reference", "torch_chunked"], "missing full original anchor routes")
    for route, check in anchors.items():
        if policy == "original" and route == "reference":
            base.require(check is None, "original reference self-anchor must be null")
            continue
        checks.append(comparison(check, cfg, targets, case, policy + " original anchor " + route))
        same(check["next_token_effects"]["per_token"]["reference_logprob"], original_lp, "common original logprob anchor differs")
    internal_lp = arm["fp32_backend"]["next_token_effects"]["per_token"]
    same(internal_lp["actual_logprob"], anchors["torch_chunked"]["next_token_effects"]["per_token"]["actual_logprob"], "chunked logprob identity differs")
    own_lp = internal_lp["reference_logprob"]
    if policy == decay.TREATMENT:
        same(own_lp, anchors["reference"]["next_token_effects"]["per_token"]["actual_logprob"], "reference logprob identity differs")
    else:
        same(arm["fp32_backend"], anchors["torch_chunked"], "original full internal and common-anchor fields differ")
    cached = arm["fp32_cached"]
    same(cached["tolerance"], base.TOLERANCES["float32"], "cached tolerance changed")
    same(cached["cache_dtype"], "torch.float32", "cached dtype changed")
    same(sorted(cached["internal_logits"]), sorted(INTERNAL), "cached internal route coverage differs")
    same(sorted(cached["original_fp32_anchor_logits"]), sorted(CACHED), "cached original anchor coverage differs")
    checks += [base.metric(value, "cached internal " + name) for name, value in cached["internal_logits"].items()]
    checks += [base.metric(value, "cached original anchor " + name) for name, value in cached["original_fp32_anchor_logits"].items()]
    if policy == "original":
        same({name: cached["internal_logits"][name] for name in CACHED}, cached["original_fp32_anchor_logits"],
             "original cached internal and common stateless-anchor fields differ")
    same(sorted(cached["retained_state"]), sorted(STATEFUL[1:]), "retained state bundle coverage differs")
    checks += [state_check(value, cfg, batch, length, "retained " + name) for name, value in cached["retained_state"].items()]
    common_state = cached["original_reference_stateful_anchor"]
    if policy == "original":
        base.require(common_state is None, "original stateful self-anchor must be null")
    else:
        same(sorted(common_state), sorted(STATEFUL), "common original stateful coverage differs")
        checks += [state_check(value, cfg, batch, length, "original memory " + name) for name, value in common_state.items()]
    prefix_state = cached["real_prefix"]
    base.require(prefix_state["synthetic"] is False and prefix_state["unchanged_after_cloned_continuations"] is True, "prefix is synthetic or mutated")
    same(prefix_state["position"], prefix, "prefix position differs")
    identity_fields(prefix_state["fields"], cfg, batch, prefix, "actual prefix")
    same(cached["suffix_length"], length - prefix, "suffix length differs")
    same(cached["chunked_schedule"], [min(128, length - prefix - start) for start in range(0, length - prefix, 128)], "continuation schedule differs")
    base.require(cached["one_shot_vs_tokenwise_degenerate"] is False, "degenerate continuation")
    same(cached["final_positions"], dict.fromkeys(STATEFUL, length), "actual final positions differ")
    same(sorted(cached["next_token_effects"]), sorted(CACHED), "cached descriptive coverage differs")
    for name, effect in cached["next_token_effects"].items():
        start, stop = bounds(name, length, prefix)
        slice_targets = batch_slice(targets, batch, length, start, stop)
        validate_effect(effect, slice_targets, batch, start, stop - start, "cached " + name)
        same(effect["per_token"]["reference_logprob"], batch_slice(original_lp, batch, length, start, stop), "cached common original logprob slice differs")
    same(arm["unexecuted"], ["BF16 paths", *([] if case["gradient_check"] else ["all parameter gradients"])], "unexecuted scope differs")
    operators(arm, cfg, case, 128)
    base.flag(cached["passed"], all(value["passed"] for value in [*cached["internal_logits"].values(), *cached["original_fp32_anchor_logits"].values(),
                  *cached["retained_state"].values(), *((common_state or {}).values())]), "cached aggregate")
    base.flag(arm["passed"], all(checks), "policy aggregate")
    return all(checks)


def validate_case(case, shared, checkpoint, old, narrow):
    for name, value in shared.items():
        same(case[name], value, "case shared input differs: " + name)
    same(case["weights_sha256"], old["weights_sha256"], "unchanged weight identity differs")
    base.number(case["processing_seconds"], "case processing seconds", 0)
    base.integer(case["cuda_peak_allocated_bytes"], "CUDA peak", 1)
    cfg, batch, length = checkpoint["model_config"], shared["batch_size"], shared["length"]
    arms = case["treatments"]
    same([arm["treatment_id"] for arm in arms], decay.TREATMENTS, "missing/duplicate policy matrix")
    original_lp = arms[0]["fp32_backend"]["next_token_effects"]["per_token"]["reference_logprob"]
    targets = arms[0]["fp32_backend"]["next_token_effects"]["per_token"]["target_ids"]
    validate_input_targets(targets, shared, cfg["vocab_size"])
    anchor = case["original_anchor"]
    same(sorted(anchor), sorted(["treatment", "backend", "logits_sha256", "loss_sha256", "retained_state_scope", "retained_state", "gradient_parameter_tensors"]), "anchor schema differs")
    base.require(anchor["treatment"] == "original" and anchor["backend"] == "reference"
                 and anchor["retained_state_scope"] == "original reference stateful_full", "anchor scope differs")
    base.digest(anchor["logits_sha256"], "original full logits")
    base.digest(anchor["loss_sha256"], "original full loss")
    same(anchor["gradient_parameter_tensors"], len(base.parameter_inventory(cfg)) if shared["gradient_check"] else None, "anchor gradient scope differs")
    identity_fields(anchor["retained_state"], cfg, batch, length, "original stateful anchor")
    if batch == 1 and length == 257:
        for name in ("logits_sha256", "loss_sha256"):
            same(anchor[name], narrow["common_original_fp32_anchor"][name], "257-token original anchor does not reproduce narrow decay")
        for arm, previous_arm in zip(arms, narrow["treatments"]):
            for name in ("logits", "loss", "next_token_effects"):
                same(arm["fp32_backend"][name], previous_arm["fp32_backend"][name],
                     "257-token full numerical fields do not reproduce narrow decay")
                for route in ("reference", "torch_chunked"):
                    current, previous = arm["common_original_fp32_anchor"][route], previous_arm["common_original_fp32_anchor"][route]
                    if current is not None and previous is not None:
                        same(current[name], previous[name], "257-token common anchor fields do not reproduce narrow decay")
    passes = [validate_arm(arm, cfg, shared, targets, original_lp) for arm in arms]
    base.flag(case["passed"], all(passes), "case aggregate")


def validate_declaration(declaration):
    expected = {"schema": 1, "status_at_declaration": "planned_before_execution", "device": "cuda", "seed": 2027,
                "chunk_size": 128, "ratios": base.RATIOS, "treatment_id": decay.TREATMENT, "tolerances": base.TOLERANCES,
                "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}, "cases": CASES,
                "time_budget_seconds": 900, "minimum_large_case_free_bytes": 8 * 1024**3}
    for name, value in expected.items():
        same(declaration[name], value, "declaration differs: " + name)
    same(declaration["baseline_declaration"], decay.BASE_DECLARATION, "unexpected baseline declaration")
    return expected


def validate(report, declaration, declaration_hash, declaration_path, baseline, previous, baseline_declaration, source_root):
    base.require(type(report["schema"]) is int and report["schema"] == 1 and report["kind"] == KIND
                 and report["certified"] is False and report["execution_status"] in ("completed", "incomplete"), "unknown/certified breadth report")
    expected = validate_declaration(declaration)
    p = report["protocol"]
    same(report["protocol_sha256"], base.canonical_sha256(p), "protocol fingerprint differs")
    for name in ("device", "seed", "chunk_size", "ratios", "treatment_id", "tolerances", "cases", "time_budget_seconds", "minimum_large_case_free_bytes"):
        same(p[name], expected[name], "protocol differs: " + name)
    base.require(type(p["version"]) is int and p["version"] == 1 and p["BF16_executed"] is False
                 and p["production_defaults_changed"] is False and p["optimizer_executed"] is False
                 and p["quality_equivalence_margin"] is None, "unexpected precision/promotion/training")
    decay.public_binding(p["declaration"], declaration_path, declaration_hash, "breadth declaration")
    decay.public_binding(p["baseline_declaration"], declaration["baseline_declaration"], base.file_sha256(Path(source_root) / declaration["baseline_declaration"]), "baseline declaration")
    for name in ("baseline", "decay"):
        decay.public_binding(p[name + "_report"], declaration[name + "_report"], declaration[name + "_sha256"], name)
    same(p["data"], baseline["protocol"]["data"], "data identity differs")
    same(p["checkpoints"], baseline["protocol"]["checkpoints"], "checkpoint registry differs")
    same(p["runtime"], previous["protocol"]["runtime"], "runtime differs from measured decay")
    same(p["runtime_sha256"], base.canonical_sha256(p["runtime"]), "runtime fingerprint differs")
    same(p["runtime"]["precision_flags"]["cuda_matmul_allow_tf32"], False, "TF32 enabled")
    same(p["runtime"]["precision_flags"]["cudnn_allow_tf32"], False, "TF32 enabled")
    declared_cps = {item["ratio"]: item for item in baseline_declaration["checkpoints"]}
    same([item["ratio"] for item in p["checkpoints"]], base.RATIOS, "checkpoint order differs")
    for item in p["checkpoints"]:
        for name in ("path", "sha256"):
            same(item[name], declared_cps[item["ratio"]][name], "declared checkpoint differs")
    same(sorted(p["source_sha256"]), sorted(SOURCES), "missing measured source coverage")
    for name, recorded in p["source_sha256"].items():
        base.digest(recorded, "source " + name)
        raw = (Path(source_root) / base.relative_path(name)).read_bytes().replace(b"\r\n", b"\n")
        base.require(recorded in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()}, "stale measured source: " + name)
        if name in previous["protocol"]["source_sha256"]:
            same(recorded, previous["protocol"]["source_sha256"][name], "source differs from prior decay: " + name)
    shared = p["shared_inputs"]
    same(shared, declaration["shared_inputs"], "declared paired inputs differ")
    same([{name: row[name] for name in CASES[0]} for row in shared], CASES, "shared case matrix differs")
    same([row["case_id"] for row in shared], [identifier(case) for case in CASES], "shared case identities differ")
    for row in shared:
        same(len(row["windows"]), row["batch_size"], "window batch coverage differs")
        for index, window in enumerate(row["windows"]):
            same(window["window"], index, "window order differs")
            same(window["length"], row["length"], "window length differs")
            same(window["target_policy"], "y[i] = validation[start_token+i+1]", "target shift policy differs")
            base.integer(window["start_token"], "natural window start")
            for name in ("tokens_sha256", "targets_sha256"):
                base.digest(window[name], "window " + name)
                base.digest(row[name], "batch " + name)
    same(shared[3]["windows"], baseline["protocol"]["shared_inputs"], "257-token baseline pairing differs")
    ordered = [(ratio, row["case_id"]) for ratio in base.RATIOS for row in shared]
    actual = [(row["ratio"], row["case_id"]) for row in report["cases"]]
    complete = report["execution_status"] == "completed"
    same(actual, ordered if complete else ordered[:len(actual)], "missing/duplicate/out-of-order case coverage")
    base.require(len(actual) <= len(ordered), "extra completed cases")
    if complete:
        base.require(report["reason"] is None and report["failed_case"] is None, "complete report has failure metadata")
    else:
        base.require(isinstance(report["reason"], str) and report["reason"], "incomplete reason missing")
        failed = report["failed_case"]
        if len(actual) < len(ordered):
            base.require(isinstance(failed, dict), "failed case missing")
            ratio, case_id = ordered[len(actual)]
            base.require(failed["ratio"] == ratio and failed["case_id"] in (None, case_id), "failed case is not next uncompleted case")
            base.require(failed["stage"] in ("model load", "full/cached model diagnostic"), "unknown failure stage")
            if failed["stage"] == "model load":
                base.require(failed["case_id"] is None and case_id == shared[0]["case_id"], "invalid model-load failure location")
            else:
                base.require(failed["case_id"] == case_id, "missing active case identity")
        else:
            base.require(failed is None or (failed["ratio"], failed["case_id"]) == actual[-1]
                         and failed["stage"] == "full/cached model diagnostic", "post-completion failure location differs")
    checkpoints = {item["ratio"]: item for item in p["checkpoints"]}
    originals = {item["ratio"]: item for item in baseline["cases"]}
    narrow_cases = {item["ratio"]: item for item in previous["cases"]}
    identities = {item["case_id"]: item for item in shared}
    for case in report["cases"]:
        validate_case(case, identities[case["case_id"]], checkpoints[case["ratio"]], originals[case["ratio"]], narrow_cases[case["ratio"]])
    expected_status = "incomplete" if not complete else "completed" if all(case["passed"] for case in report["cases"]) else "completed_with_parity_failures"
    same(report["status"], expected_status, "status hides failures or missing work")
    integrity = report["post_execution_integrity"]
    same(sorted(integrity), sorted(["checkpoints", "declaration_and_prior_evidence", "measured_sources"]), "post-execution audit coverage differs")
    base.require(all(type(value) is bool for value in integrity.values()), "invalid integrity flags")
    base.require(not complete or all(integrity.values()), "completed report has unstable evidence")
    allowance = report["execution_allowance"]
    same(allowance["allowance_seconds"], 900, "time allowance differs")
    base.number(allowance["elapsed_seconds"], "elapsed time", 0)
    base.close_double(allowance["overshoot_seconds"], max(0, allowance["elapsed_seconds"] - 900), "cooperative overshoot")
    same(allowance["scope"], "cooperative diagnostic model execution; in-flight work can exceed the allowance", "time scope differs")
    expected_headroom = [(row["ratio"], row["case_id"]) for row in report["cases"] if row["gradient_check"]]
    seen = []
    for entry in report["large_case_headroom"]:
        pair = entry["ratio"], entry["case_id"]
        base.require(pair not in seen and pair in [(ratio, shared[-1]["case_id"]) for ratio in base.RATIOS], "duplicate/unknown headroom case")
        seen.append(pair)
        for key in ("free_bytes", "total_bytes", "minimum_free_bytes"):
            base.integer(entry[key], "CUDA headroom " + key, 0 if key == "free_bytes" else 1)
        same(entry["minimum_free_bytes"], 8 * 1024**3, "headroom guard changed")
        base.require(entry["free_bytes"] <= entry["total_bytes"], "impossible CUDA free memory")
        if pair in actual:
            base.require(entry["free_bytes"] >= entry["minimum_free_bytes"], "executed large case lacked guarded headroom")
    extra = []
    if not complete and report["failed_case"] and report["failed_case"]["case_id"] == shared[-1]["case_id"]:
        extra = [(report["failed_case"]["ratio"], shared[-1]["case_id"])]
    base.require(seen == expected_headroom or seen == expected_headroom + extra, "headroom coverage differs")


def counts(checks):
    checks = dict(checks)
    return {"passed": sum(value["passed"] for value in checks.values()), "total": len(checks),
            "failed_checks": [name for name, value in checks.items() if not value["passed"]],
            "worst_tolerance_ratio": max((value["max_tolerance_ratio"] for value in checks.values() if value["max_tolerance_ratio"] is not None), default=None)}


def states(value):
    if value is None:
        return None
    bundles = {name: {"passed": item["passed"], "max_tolerance_ratio": max((field["max_tolerance_ratio"] for field in item["fields"] if field["max_tolerance_ratio"] is not None), default=None)} for name, item in value.items()}
    return {"bundles": counts(bundles), "fields": counts({name + "." + field["field"]: field for name, item in value.items() for field in item["fields"]})}


def compact_comparison(value):
    if value is None:
        return None
    gradients = value["gradients"]
    return {"logits": value["logits"], "loss": value["loss"], "gradients": None if gradients is None else {
        **counts({check["parameter"]: check for check in gradients["checks"]}), "failed_parameters": gradients["failed_parameters"]}, "passed": value["passed"]}


def effect(value, batch):
    return {**base.compact_effect(value), "batch_size": batch,
            "position_interpretation": "position_start/stop apply to each row; per-token arrays are batch-major flattened"}


def gate_counts(report, policy):
    arms = [arm for case in report["cases"] for arm in case["treatments"] if arm["treatment_id"] == policy]
    def count(values):
        values = list(values)
        return {"passed": sum(value["passed"] for value in values), "total": len(values)}
    cached = [arm["fp32_cached"] for arm in arms]
    full_anchors = [value for arm in arms for value in arm["common_original_fp32_anchor"].values() if value is not None]
    state_anchors = [value for item in cached for value in (item["original_reference_stateful_anchor"] or {}).values()]
    return {"full_backend": count(arm["fp32_backend"] for arm in arms), "cached": count(cached),
            "full_original_stateless_anchor": count(full_anchors),
            "cached_internal_logits": count(value for item in cached for value in item["internal_logits"].values()),
            "cached_original_stateless_logits": count(value for item in cached for value in item["original_fp32_anchor_logits"].values()),
            "retained_state_bundles": count(value for item in cached for value in item["retained_state"].values()),
            "original_stateful_anchor_bundles": None if policy == "original" else count(state_anchors),
            "internal_unique_parameter_gradients": count(check for arm in arms if arm["fp32_backend"]["gradients"] is not None
                                                       for check in arm["fp32_backend"]["gradients"]["checks"])}


def unexecuted_summary(report, declaration, declaration_hash, relative, raw_hash):
    """A pre-audit failure has no measured protocol or passing policy rows."""
    validate_declaration(declaration)
    same(sorted(report), sorted(["schema", "kind", "certified", "status", "execution_status", "reason", "cases"]),
         "pre-audit incomplete report schema differs")
    base.require(type(report["schema"]) is int and report["schema"] == 1 and report["kind"] == KIND
                 and report["certified"] is False and report["status"] == report["execution_status"] == "incomplete"
                 and report["cases"] == [] and isinstance(report["reason"], str) and report["reason"],
                 "invalid pre-audit incomplete evidence")
    expected = [{"ratio": ratio, "case_id": identifier(case)} for ratio in base.RATIOS for case in CASES]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": "incomplete",
            "execution_status": "incomplete", "certified": False, "reason": report["reason"], "failed_case": None,
            "declaration": {"path": relative, "sha256": declaration_hash},
            "raw_report": {"path": declaration["output"], "sha256": raw_hash, "completed_cases": 0},
            "protocol_sha256": None, "protocol": None, "rows": [],
            "coverage": {"expected_cases": len(expected), "completed_cases": 0, "completed_policy_rows": 0,
                         "missing_cases": expected, "BF16_executed": False, "gradients_only_batch2_length512": True,
                         "execution_identities_available": False},
            "treatment_counts": {policy: {"passed": 0, "completed": 0, "expected": len(expected)} for policy in decay.TREATMENTS},
            "gate_counts": {policy: None for policy in decay.TREATMENTS}, "diagnostic_resources": [],
            "execution_allowance": None, "large_case_headroom": [], "post_execution_integrity": None,
            "limits": [*declaration["limits"],
                       "Execution failed before an audited protocol was recorded; runtime, source, input and checkpoint identities are unavailable for this run.",
                       "The declaration is a plan, not measured evidence. No missing case or policy supplies a pass."]}


def build_summary(declaration_path, raw_path=None, root=ROOT, source_root=ROOT):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    declaration, declaration_hash = base.read_json(declaration_path)
    declared_raw = (root / base.relative_path(declaration["output"])).resolve()
    raw_path = declared_raw if raw_path is None else Path(raw_path).resolve()
    same(str(raw_path), str(declared_raw), "raw file differs from declared immutable path")
    report, raw_hash = base.read_json(raw_path)
    priors = {}
    for name in ("baseline", "decay"):
        prior, digest = base.read_json(root / base.relative_path(declaration[name + "_report"]))
        same(digest, declaration[name + "_sha256"], "immutable " + name + " bytes differ")
        same(prior["protocol_sha256"], base.canonical_sha256(prior["protocol"]), "prior protocol fingerprint differs")
        base.require(prior["execution_status"] == "completed" and prior["certified"] is False, "invalid prior evidence")
        priors[name] = prior
    same(priors["decay"]["protocol"]["baseline_report"]["sha256"], declaration["baseline_sha256"], "prior baseline chain differs")
    same(priors["decay"]["protocol"]["checkpoints"], priors["baseline"]["protocol"]["checkpoints"], "prior checkpoint chain differs")
    baseline_declaration, _ = base.read_json(root / base.relative_path(declaration["baseline_declaration"]))
    relative = declaration_path.relative_to(root).as_posix()
    if "protocol" not in report:
        return unexecuted_summary(report, declaration, declaration_hash, relative, raw_hash)
    validate(report, declaration, declaration_hash, relative, priors["baseline"], priors["decay"], baseline_declaration, source_root)
    rows = []
    for case in report["cases"]:
        for arm in case["treatments"]:
            cached = arm["fp32_cached"]
            rows.append({"ratio": case["ratio"], "case_id": case["case_id"], "batch_size": case["batch_size"],
                "length": case["length"], "prefix_length": case["prefix_length"], "gradient_check": case["gradient_check"],
                "treatment": arm["treatment_id"], "weights_sha256": case["weights_sha256"],
                "internal": {"full": compact_comparison(arm["fp32_backend"]), "cached_logits": counts(cached["internal_logits"]), "retained_state": states(cached["retained_state"])},
                "original_stateless_anchor": {"full": {route: compact_comparison(value) for route, value in arm["common_original_fp32_anchor"].items()}, "cached_logits": counts(cached["original_fp32_anchor_logits"])},
                "original_stateful_anchor": states(cached["original_reference_stateful_anchor"]),
                "real_prefix": cached["real_prefix"], "passed": arm["passed"], "unexecuted": arm["unexecuted"],
                "next_token_effects": {"internal_full": effect(arm["fp32_backend"]["next_token_effects"], case["batch_size"]),
                    "cached_original_anchor": {name: effect(value, case["batch_size"]) for name, value in cached["next_token_effects"].items()}}})
    expected = [(ratio, item["case_id"]) for ratio in base.RATIOS for item in declaration["shared_inputs"]]
    completed = [(row["ratio"], row["case_id"]) for row in report["cases"]]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"],
            "execution_status": report["execution_status"], "certified": False, "reason": report["reason"], "failed_case": report["failed_case"],
            "declaration": {"path": relative, "sha256": declaration_hash}, "raw_report": {"path": declaration["output"], "sha256": raw_hash, "completed_cases": len(completed)},
            "protocol_sha256": report["protocol_sha256"], "protocol": report["protocol"], "rows": rows,
            "coverage": {"expected_cases": len(expected), "completed_cases": len(completed), "completed_policy_rows": len(rows),
                         "missing_cases": [{"ratio": ratio, "case_id": case_id} for ratio, case_id in expected[len(completed):]],
                         "BF16_executed": False, "gradients_only_batch2_length512": True},
            "treatment_counts": {policy: {"passed": sum(row["passed"] for row in rows if row["treatment"] == policy),
                                           "completed": sum(row["treatment"] == policy for row in rows), "expected": len(expected)} for policy in decay.TREATMENTS},
            "gate_counts": {policy: gate_counts(report, policy) for policy in decay.TREATMENTS},
            "original_controls": [{"ratio": case["ratio"], "case_id": case["case_id"], "narrow_full_fields_and_anchor_hashes_equal": True}
                                  for case in report["cases"] if case["batch_size"] == 1 and case["length"] == 257],
            "diagnostic_resources": [{"ratio": case["ratio"], "case_id": case["case_id"],
                                      "processing_seconds": case["processing_seconds"],
                                      "peak_allocated_mib": case["cuda_peak_allocated_bytes"] / 1048576,
                                      "scope": "combined original/treatment full and cached diagnostics; gradients only where declared; not throughput"}
                                     for case in report["cases"]],
            "execution_allowance": report["execution_allowance"], "large_case_headroom": report["large_case_headroom"],
            "post_execution_integrity": report["post_execution_integrity"],
            "limits": [*report["limits"], "Batch-two scored_tokens equals batch_size times the per-row position span; per-token arrays flatten rows in batch-major order.",
                       "Original stateless logits and original stateful memory are distinct anchors; null self-anchors are not measured passes.",
                       "Missing/incomplete cases supply no passes; failed original and treatment findings remain visible.",
                       "Public fingerprints bind the producer's binary/data audit; this exporter does not load private checkpoint or corpus binaries."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/decay-breadth-protocol-2026-10-04.json")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        base.write_summary(args.output, build_summary(args.declaration, args.raw, args.root, args.source_root))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Validated breadth summary written; certified=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
