"""Validate declared RoPE evidence and export a compact, uncertified summary.

Uses only the standard library. Original control fields must exactly match the
immutable natural-window baseline; matched frozen operands are not a model gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_checkpoint_numerics as base  # noqa: E402

KIND = "rope_precision_isolation"
POLICIES = {
    "original": "Only cached attention rounds FP32 rotation tables to the query dtype",
    "shared_fp32": "Both routes rotate in FP32, then explicitly cast query/key back to the original dtype",
    "shared_bf16": "Both routes round rotation tables to BF16 before rotation",
}
BASE_DECLARATION = "docs/research/checkpoint-numerics-protocol-2026-10-04.json"
SOURCES = base.SOURCES | {"scripts/study_rope_precision.py"}
INTERNAL = {"stateful_full", "prefix", "suffix_one_shot", "suffix_segmented", "suffix_tokenwise", "full_tokenwise",
            "suffix_tokenwise_vs_one_shot", "suffix_segmented_vs_one_shot"}
STATES = {"suffix_one_shot", "suffix_segmented", "suffix_tokenwise", "full_tokenwise"}
EFFECTS = {"stateful_full", "prefix", "suffix_one_shot", "suffix_segmented", "suffix_tokenwise", "full_tokenwise", "full_vs_fp32_anchor"}
CONTROL_KEYS = ("internal_logits", "retained_state", "fp32_anchor", "next_token_effects")


def identity(value, shape, dtype, label):
    base.require(set(value) == {"shape", "dtype", "sha256"} and value["shape"] == shape
                 and all(type(size) is int for size in value["shape"])
                 and value["dtype"] == dtype, "frozen operand/output shape or dtype differs: " + label)
    base.digest(value["sha256"], label)


def validate_frozen(probes, cfg, length):
    layers = [layer for layer, kind in enumerate(cfg["layer_types"]) if kind == "attention"]
    base.require(isinstance(probes, list) and [probe["layer"] for probe in probes] == layers,
                 "missing/duplicate/out-of-order actual attention frozen layers")
    qshape = [1, cfg["d_model"] // cfg["head_dim"], length, cfg["head_dim"]]
    tshape = [length, cfg["head_dim"]]
    table_identities = []
    for probe in probes:
        base.require(probe["scope"] == "actual frozen pre-RoPE operands; no recomputation of upstream layers",
                     "frozen scope differs")
        operands = probe["operands"]
        base.require(set(operands) == {"q", "k", "cos", "sin"}, "frozen operand coverage differs")
        for name in ("q", "k"):
            identity(operands[name], qshape, "torch.bfloat16", name)
        for name in ("cos", "sin"):
            identity(operands[name], tshape, "torch.float32", name)
        table_identities.append({name: operands[name] for name in ("cos", "sin")})
        rows = probe["policies"]
        base.require([row["policy"] for row in rows] == list(POLICIES), "frozen policy coverage differs")
        for row in rows:
            policy = row["policy"]
            rotation, cast = row["post_rotation"], row["post_cast"]
            base.require(set(rotation) == {"full", "cached", "comparisons", "bitwise_equal"}, "frozen postrotation fields differ")
            for route in ("full", "cached"):
                dtype = "torch.float32" if policy == "shared_fp32" or policy == "original" and route == "full" else "torch.bfloat16"
                base.require(set(rotation[route]) == {"q", "k"}, "frozen output coverage differs")
                for name in ("q", "k"):
                    identity(rotation[route][name], qshape, dtype, policy + "." + route + "." + name)
            base.require(set(cast) == {"comparison_dtype", "scope", "comparisons", "bitwise_equal"}
                         and cast["comparison_dtype"] == "torch.bfloat16"
                         and cast["scope"] == "explicit replay cast; the original stateless model has no added explicit cast",
                         "postcast must retain its explicit comparison-only scope")
            for label, phase in (("postrotation", rotation), ("postcast", cast)):
                base.require(set(phase["comparisons"]) == {"q", "k"}, "frozen comparison coverage differs")
                for name, metric in phase["comparisons"].items():
                    base.require(metric["finite"] is True, "frozen replay must retain finite recorded operands/results")
                    base.metric(metric, "frozen." + policy + "." + label + "." + name)
                equal = all(metric["max_absolute_error"] == 0 for metric in phase["comparisons"].values())
                base.flag(phase["bitwise_equal"], equal, "frozen exact agreement")
                if policy != "original":
                    base.require(equal, "matched frozen policy must retain exact agreement by construction")
            if policy != "original":
                base.require(rotation["full"] == rotation["cached"], "matched frozen output hashes differ")
        original, fp32, bf16 = rows
        base.require(fp32["post_rotation"]["full"] == original["post_rotation"]["full"]
                     and bf16["post_rotation"]["full"] == original["post_rotation"]["cached"],
                     "frozen counterfactual does not reuse original full/cached output identities")
    base.require(all(item == table_identities[0] for item in table_identities), "rotation tables differ across attention layers")
    return table_identities[0]


def validate_observations(row, cfg, length, prefix, chunk):
    suffix = length - prefix
    attention_stages = {"bf16_full": (Counter({length: 1}), False), "bf16_stateful_full": (Counter({length: 1}), True),
        "bf16_prefix_prefill": (Counter({prefix: 1}), True), "bf16_suffix_one_shot": (Counter({suffix: 1}), True),
        "bf16_suffix_segmented": (Counter(min(chunk, suffix - start) for start in range(0, suffix, chunk)), True),
        "bf16_suffix_tokenwise": (Counter({1: suffix}), True), "bf16_full_tokenwise": (Counter({1: length}), True)}
    attention_layers = [layer for layer, kind in enumerate(cfg["layer_types"]) if kind == "attention"]
    seen = {}
    for event in row["attention_observations"]:
        base.require(set(event) == {"layer", "stage", "policy", "cached", "length", "tables_dtype_before_treatment", "calls"},
                     "attention observation fields differ")
        stage, layer = event["stage"], event["layer"]
        base.require(stage in attention_stages and type(layer) is int and layer in attention_layers,
                     "unexpected attention stage/layer")
        base.require(event["policy"] == row["policy"] and event["tables_dtype_before_treatment"] == "torch.float32",
                     "attention policy/table identity differs")
        base.flag(event["cached"], attention_stages[stage][1], "attention cached route")
        base.integer(event["length"], "attention length", 1)
        base.integer(event["calls"], "attention calls", 1)
        key = (stage, layer, event["length"])
        base.require(key not in seen, "duplicate attention observation")
        seen[key] = event["calls"]
    expected = {(stage, layer, size): count for stage, (parts, _cached) in attention_stages.items()
                for layer in attention_layers for size, count in parts.items()}
    base.require(seen == expected, "missing/incorrect attention route calls")
    full_chunks = Counter(min(chunk, length - start) for start in range(0, length, chunk))
    suffix_chunks = Counter(min(chunk, suffix - start) for start in range(0, suffix, chunk))
    scan_stages = {"bf16_full": Counter({length: 1}), "bf16_stateful_full": full_chunks,
        "bf16_prefix_prefill": Counter(min(chunk, prefix - start) for start in range(0, prefix, chunk)),
        "bf16_suffix_one_shot": suffix_chunks, "bf16_suffix_segmented": suffix_chunks,
        "bf16_suffix_tokenwise": Counter({1: suffix}), "bf16_full_tokenwise": Counter({1: length})}
    mamba_layers = [layer for layer, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]
    seen = {}
    for event in row["scan_observations"]:
        stage, layer = event["stage"], event["layer"]
        base.require(stage in scan_stages and type(layer) is int and layer in mamba_layers, "unexpected scan stage/layer")
        path = "torch.quadratic_ssd" if stage == "bf16_full" else "torch.chunked_ssd"
        dtypes = ["torch.bfloat16", "torch.float32", "torch.float32", "torch.bfloat16", "torch.bfloat16", "torch.float32"]
        if path == "torch.chunked_ssd":
            dtypes += ["torch.float32"]
        base.require(event["path"] == path and event["autocast"] is True and event["input_dtypes"] == dtypes,
                     "RoPE experiment changed the scan path or arithmetic policy")
        base.integer(event["length"], "scan length", 1)
        base.integer(event["calls"], "scan calls", 1)
        key = (stage, layer, event["length"])
        base.require(key not in seen, "duplicate scan observation")
        seen[key] = event["calls"]
    expected = {(stage, layer, size): count for stage, parts in scan_stages.items()
                for layer in mamba_layers for size, count in parts.items()}
    base.require(seen == expected, "missing/incorrect actual scan calls")


def validate_policy(row, protocol, checkpoint, baseline_case, targets):
    cfg = checkpoint["model_config"]
    length, prefix, chunk = protocol["length"], protocol["prefix_length"], protocol["chunk_size"]
    base.require(row["description"] == POLICIES[row["policy"]] and row["execution_status"] == "completed",
                 "policy description or execution coverage differs")
    base.number(row["processing_seconds"], "diagnostic processing seconds", 0)
    base.integer(row["cuda_peak_allocated_bytes"], "diagnostic CUDA peak", 1)
    value = row["bf16_cached"]
    base.require(isinstance(value, dict) and value["tolerance"] == base.TOLERANCES["bfloat16"], "missing BF16 result or changed tolerance")
    base.validate_backend(value["backend"], "reference", cfg, chunk)
    base.require(set(value["internal_logits"]) == INTERNAL, "missing/extra eight internal logit checks")
    logit_passes = [base.metric(metric, row["policy"] + "." + name) for name, metric in value["internal_logits"].items()]
    base.require(set(value["retained_state"]) == STATES, "missing/extra four retained-state bundles")
    state_passes = [base.validate_state(metric, base.state_inventory(cfg, length), name)
                    for name, metric in value["retained_state"].items()]
    base.flag(value["internal_passed"], all(logit_passes) and all(state_passes), "BF16 internal aggregate")
    base.require(value["fp32_anchor"]["scope"] == "separate cross-precision full-logit comparison", "anchor scope differs")
    anchor = base.metric(value["fp32_anchor"], "unchanged FP32 anchor")
    base.flag(value["passed"], value["internal_passed"] and anchor, "BF16 whole policy")
    base.require(value["cache_dtype"] == "torch.bfloat16" and value["suffix_length"] == length - prefix
                 and value["one_shot_vs_tokenwise_degenerate"] is False and value["unexecuted_stages"] == []
                 and value["segmented_schedule"] == [128, 1], "real continuation schedule differs")
    base.require(value["final_positions"] == {name: length for name in ("stateful_full", *sorted(STATES))}, "continuation final positions differ")
    real = value["real_prefix"]
    base.require(real["position"] == prefix and real["synthetic"] is False
                 and real["unchanged_after_cloned_continuations"] is True, "real cloned prefix identity differs")
    inventory = base.state_inventory(cfg, prefix)
    names = [field["field"] for field in real["fields"]]
    base.require(len(names) == len(set(names)) and set(names) == set(inventory), "real prefix memory coverage differs")
    for field in real["fields"]:
        shape, dtype = inventory[field["field"]]
        base.require(field["shape"] == shape and field["dtype"] == dtype, "real prefix shape/dtype differs")
        base.digest(field["sha256"], "real prefix memory")
        base.integer(field["nonzero_elements"], "real prefix nonzero count")
        base.require(field["nonzero_elements"] <= prod(shape), "prefix nonzero count exceeds shape")
        if field["field"].endswith(".ssm"):
            base.require(field["nonzero_elements"] > 0, "prefix SSM memory is zero-only")
    effects = value["next_token_effects"]
    base.require(set(effects) == EFFECTS, "next-token effect coverage differs")
    for name, effect in effects.items():
        start = prefix - 1 if name == "prefix" else prefix if name.startswith("suffix_") else 0
        stop = prefix if name == "prefix" else length
        base.validate_effect(effect, targets[start:stop], start, row["policy"] + "." + name)
    full = effects["full_vs_fp32_anchor"]
    base.require(full["per_token"]["reference_logprob"]
                 == baseline_case["fp32_backend"]["next_token_effects"]["per_token"]["reference_logprob"],
                 "common unchanged FP32 anchor probabilities differ from baseline")
    for name, effect in effects.items():
        if name != "full_vs_fp32_anchor":
            start, stop = effect["position_start"], effect["position_stop_exclusive"]
            base.require(effect["per_token"]["reference_logprob"] == full["per_token"]["actual_logprob"][start:stop],
                         "internal full-BF16 probability reference differs")
    validate_observations(row, cfg, length, prefix, chunk)


def validate_report(report, declaration, baseline, declaration_hash, baseline_hash, declaration_reference, source_root):
    base.require(type(report["schema"]) is int and report["schema"] == 1 and report["kind"] == KIND and report["certified"] is False
                 and report["execution_status"] == "completed", "incomplete or certified RoPE diagnostic")
    protocol = report["protocol"]
    base.require(report["protocol_sha256"] == base.canonical_sha256(protocol), "RoPE protocol hash differs")
    base.require(protocol["kind"] == KIND and protocol["version"] == 1 and protocol["policies"] == POLICIES,
                 "RoPE kind or policies differ")
    expected = {"device": "cuda", "seed": 2027, "length": 257, "prefix_length": 128,
                "chunk_size": 128, "batch_size": 1, "ratios": base.RATIOS}
    base.require(type(declaration["schema"]) is int and declaration["schema"] == 1 and declaration["status_at_declaration"] == "planned_before_execution"
                 and declaration["policies"] == POLICIES, "declaration policy/status differs")
    base.require(all(type(protocol[name]) is type(declaration[name]) is type(value)
                     and protocol[name] == declaration[name] == value for name, value in expected.items()), "declared RoPE workload differs")
    base.require(protocol["windows"] == 1 and protocol["full_model"] is True and protocol["full_tokenwise"] is True
                 and protocol["training_run_id"] == "week3-700m-v1", "full-model route grid is incomplete")
    base.require(protocol["tolerances"] == declaration["tolerances"] == base.TOLERANCES
                 and protocol["quality_equivalence_margin"] is None, "tolerance or quality margin changed")
    base.require(protocol["declaration"] == {"path": declaration_reference, "path_scope": "repository-relative", "sha256": declaration_hash},
                 "bound declaration identity differs")
    base.require(protocol["baseline_report"] == {"path": declaration["baseline_report"], "path_scope": "repository-relative", "sha256": baseline_hash}
                 and declaration["baseline_sha256"] == baseline_hash, "bound baseline hash/path differs")
    prior = baseline["protocol"]
    for name in ("runtime", "runtime_sha256", "data", "checkpoints", "shared_inputs"):
        base.require(base.canonical_sha256(protocol[name]) == base.canonical_sha256(prior[name]), "baseline shared identity differs: " + name)
    base.require(protocol["runtime_sha256"] == base.canonical_sha256(protocol["runtime"]), "runtime hash differs")
    base.require(declaration["precision_flags"] == {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}, "declared TF32 identity differs")
    shared = protocol["shared_inputs"][0]
    base.require(all(type(shared[name]) is type(declaration[name]) and shared[name] == declaration[name]
                     for name in ("start_token", "tokens_sha256", "targets_sha256")), "declared token/target window differs")
    sources = protocol["source_sha256"]
    base.require(set(sources) == SOURCES and all(sources[name] == prior["source_sha256"][name] for name in base.SOURCES),
                 "source coverage or baseline code identity differs")
    for path, value in sources.items():
        base.digest(value, "RoPE source")
        raw = (Path(source_root) / base.relative_path(path)).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        base.require(value in {hashlib.sha256(content).hexdigest() for content in (raw, lf, lf.replace(b"\n", b"\r\n"))},
                     "recorded source hash differs from source checkout")
    base.require([case["ratio"] for case in report["cases"]] == base.RATIOS, "missing/duplicate/out-of-order checkpoint rows")
    tables = []
    for case, checkpoint, baseline_case in zip(report["cases"], protocol["checkpoints"], baseline["cases"]):
        base.require(case["ratio"] == checkpoint["ratio"] == baseline_case["ratio"]
                     and all(case[name] == shared[name] for name in ("window", "start_token", "length", "tokens_sha256", "targets_sha256", "target_policy")),
                     "checkpoint case input identity differs")
        base.require(case["weights_sha256"] == baseline_case["weights_sha256"], "same learned weights differ from baseline")
        base.digest(case["fp32_anchor"]["logits_sha256"], "FP32 anchor logits")
        base.require(case["fp32_anchor"]["scope"] == "unchanged original stateless reference, autocast disabled", "FP32 anchor scope differs")
        base.require([row["policy"] for row in case["policies"]] == list(POLICIES), "missing/duplicate full-model policies")
        tables.append(validate_frozen(case["frozen_probes"], checkpoint["model_config"], protocol["length"]))
        targets = baseline_case["bf16_cached"]["next_token_effects"]["full_vs_fp32_anchor"]["per_token"]["target_ids"]
        for row in case["policies"]:
            validate_policy(row, protocol, checkpoint, baseline_case, targets)
        original = case["policies"][0]["bf16_cached"]
        base.require(all(original[key] == baseline_case["bf16_cached"][key] for key in CONTROL_KEYS),
                     "original full-model numerical fields no longer equal immutable baseline")
        original_hash = base.canonical_sha256({key: original[key] for key in CONTROL_KEYS})
        prior_hash = base.canonical_sha256({key: baseline_case["bf16_cached"][key] for key in CONTROL_KEYS})
        comparison = case["baseline_control_comparison"]
        base.require(comparison == {"scope": "exact saved numerical fields; equality is descriptive, not certification",
                     "original_fields_sha256": original_hash, "baseline_fields_sha256": prior_hash, "exact_fields_equal": True},
                     "original-control comparison hashes/flag differ")
        base.flag(case["full_model_passed"], all(row["bf16_cached"]["passed"] for row in case["policies"]), "full-model policy aggregate")
    base.require(all(table == tables[0] for table in tables), "shared FP32 rotation table identities differ across checkpoints")
    status = "completed" if all(case["full_model_passed"] for case in report["cases"]) else "completed_with_parity_failures"
    base.require(report["status"] == status, "RoPE overall status hides failed endpoints")


def build_summary(protocol_path: Path, report_path: Path | None = None, root: Path = ROOT,
                  source_root: Path = ROOT) -> dict:
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    declaration_reference = protocol_path.relative_to(root).as_posix()
    declaration, declaration_hash = base.read_json(protocol_path)
    expected_path = base.relative_path(declaration["output"])
    report_path = root / expected_path if report_path is None else Path(report_path).resolve()
    base.require(report_path == root / expected_path, "raw report path differs from declaration")
    baseline_path = root / base.relative_path(declaration["baseline_report"])
    baseline, baseline_hash = base.read_json(baseline_path)
    base.require(baseline_hash == declaration["baseline_sha256"], "immutable baseline bytes differ from declaration")
    baseline_declaration, _ = base.read_json(root / BASE_DECLARATION)
    base.validate_report(baseline, baseline_declaration, source_root)
    report, report_hash = base.read_json(report_path)
    validate_report(report, declaration, baseline, declaration_hash, baseline_hash, declaration_reference, source_root)
    rows = []
    for case in report["cases"]:
        for policy in case["policies"]:
            value = policy["bf16_cached"]
            frozen = [next(row for row in probe["policies"] if row["policy"] == policy["policy"]) for probe in case["frozen_probes"]]
            frozen_counts = {}
            for name in ("post_rotation", "post_cast"):
                metrics = [metric for row in frozen for metric in row[name]["comparisons"].values()]
                frozen_counts[name] = {**base.count_checks(metrics), "exact_layers": sum(row[name]["bitwise_equal"] for row in frozen),
                    "total_layers": len(frozen), "checks": [{"layer": probe["layer"], **row[name]} for probe, row in zip(case["frozen_probes"], frozen)]}
            rows.append({"ratio": case["ratio"], "policy": policy["policy"], "description": policy["description"],
                "passed": value["passed"], "weights_sha256": case["weights_sha256"],
                "internal": {**base.count_checks(value["internal_logits"].values()), "checks": value["internal_logits"]},
                "retained_state": {**base.count_checks(value["retained_state"].values()),
                    "fields": base.count_checks(field for group in value["retained_state"].values() for field in group["fields"]),
                    "failed_fields": [{"comparison": name, **field} for name, group in value["retained_state"].items()
                                      for field in group["fields"] if not field["passed"]]},
                "fp32_anchor": value["fp32_anchor"], "frozen": {"attention_layers": len(frozen), **frozen_counts},
                "next_token_effects": {name: base.compact_effect(effect) for name, effect in value["next_token_effects"].items()},
                "real_prefix": {name: value["real_prefix"][name] for name in ("position", "synthetic", "unchanged_after_cloned_continuations")},
                "suffix_length": value["suffix_length"], "segmented_schedule": value["segmented_schedule"],
                "processing_seconds": policy["processing_seconds"], "peak_allocated_mib": policy["cuda_peak_allocated_bytes"] / 1048576})
    protocol = report["protocol"]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"],
        "execution_status": "completed", "certified": False, "rows": rows,
        "declaration": {"path": declaration_reference, "sha256": declaration_hash, "canonical_sha256": base.canonical_sha256(declaration)},
        "raw_report": {"path": expected_path, "sha256": report_hash, "canonical_sha256": base.canonical_sha256(report),
                       "protocol_sha256": report["protocol_sha256"], "cases": len(report["cases"]), "policies": len(rows)},
        "baseline_report": protocol["baseline_report"], "source_sha256": protocol["source_sha256"], "data": protocol["data"],
        "protocol_sha256": report["protocol_sha256"],
        "runtime": protocol["runtime"], "runtime_sha256": protocol["runtime_sha256"],
        "shared_inputs": protocol["shared_inputs"], "checkpoints": protocol["checkpoints"],
        "tolerances": protocol["tolerances"], "quality_equivalence_margin": None,
        "original_controls": [{"ratio": case["ratio"], **case["baseline_control_comparison"]} for case in report["cases"]],
        "limits": declaration["limits"] + ["Frozen tensor identities and recorded scalar statistics are validated; tensor values are not embedded or recomputed by this exporter.",
            "Matched frozen policy agreement is by construction; the full-model gate remains a separate endpoint.",
            "BF16 RoPE differences do not explain or approve the separate FP32 scan comparison.",
            "Per-policy resource measurements include inference, replay, hashing and checks; no throughput ranking."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = build_summary(args.protocol, args.report, args.root, args.source_root)
        base.write_summary(args.output, result)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(f"invalid RoPE evidence: {error}")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "certified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
