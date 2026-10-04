"""Diagnostic breadth gate for the temporary cumulative-decay coefficient policy.

Natural inputs are paired across policies and checkpoints. Boundary cases record
FP32 full/cached logits and genuine continuation memory; only batch-two length512
also records every full-model parameter gradient. No production policy is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import scripts.study_checkpoint_numerics as base  # noqa: E402
import scripts.study_decay_precision as decay  # noqa: E402
from scripts.isolate_checkpoint_numerics import EXPECTED_INPUT, _baseline_identities  # noqa: E402
from src.model.scan_backend import BackendUnavailableError  # noqa: E402

CASES = tuple({"batch_size": batch, "length": length, "prefix_length": min(128, length // 2),
               "gradient_check": batch == 2} for batch, length in ((1, 127), (1, 128), (1, 129), (1, 257), (2, 512)))
MINIMUM_FREE_BYTES = 8 * 1024**3


def case_id(case):
    return f"b{case['batch_size']}-l{case['length']}-p{case['prefix_length']}"


def prepare_inputs(validation):
    """Read-only input matrix, available before declaration/model allocation."""
    prepared = []
    for case in CASES:
        windows = base.shared_windows(validation, case["length"], case["batch_size"], 2027)
        x, y = (torch.cat([row[index] for row in windows], dim=0).contiguous() for index in (0, 1))
        identity = {"case_id": case_id(case), **case, "windows": [row[2] for row in windows],
                    "tokens_sha256": base.tensor_sha256(x), "targets_sha256": base.tensor_sha256(y)}
        prepared.append((x, y, identity))
    return prepared


class DiagnosticLimit:
    def __init__(self, seconds=900., *, clock=time.monotonic):
        if type(seconds) not in (int, float) or not 0 < seconds <= 900:
            raise ValueError("diagnostic allowance must be finite, positive and at most 900 seconds")
        self.seconds, self.clock, self.started = seconds, clock, clock()

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise RuntimeError("diagnostic time allowance exhausted; no reduced-shape fallback")

    def metadata(self):
        elapsed = max(0., self.clock() - self.started)
        return {"allowance_seconds": self.seconds, "elapsed_seconds": elapsed,
                "overshoot_seconds": max(0., elapsed - self.seconds),
                "scope": "cooperative diagnostic model execution; in-flight work can exceed the allowance"}


def _full_pass(model, x, y, backend, chunk_size, gradients, recorder, check_limit):
    check_limit()
    metadata = model.configure_scan_backend(backend, chunk_size)
    recorder["stage"] = "fp32_" + backend + "_full"
    model.zero_grad(set_to_none=True)
    try:
        with torch.set_grad_enabled(gradients), base.observe_scans(model, recorder), torch.autocast(x.device.type, enabled=False):
            logits, loss = model(x, y)
        base.require_finite(logits, (*x.shape, model.cfg.vocab_size), "full FP32 logits")
        base.require_finite(loss, (), "full FP32 loss")
        stored = None
        if gradients:
            check_limit()
            loss.backward()
            stored = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.grad is None or parameter.grad.shape != parameter.shape:
                    raise RuntimeError("missing or wrong-shaped gradient: " + name)
                stored[name] = parameter.grad.detach().cpu().clone()
        return (logits.detach().cpu(), loss.detach().cpu(), stored), metadata
    finally:
        model.zero_grad(set_to_none=True)


def pass_comparison(actual, anchor, targets):
    if (actual[2] is None) != (anchor[2] is None):
        raise ValueError("gradient scopes differ")
    checks = {"logits": base.compare(actual[0], anchor[0], base.TOLERANCES["float32"]),
              "loss": base.compare(actual[1], anchor[1], base.TOLERANCES["float32"]),
              "gradients": None if actual[2] is None else base.gradient_comparisons(actual[2], anchor[2])}
    return {"tolerance": dict(base.TOLERANCES["float32"]), **checks,
            "next_token_effects": base.token_effects(actual[0], anchor[0], targets),
            "passed": checks["logits"]["passed"] and checks["loss"]["passed"]
                      and (checks["gradients"] is None or checks["gradients"]["all_passed"])}


def state_comparison(actual, expected):
    """Every retained field uses the original FP32 tolerance, including attention."""
    if list(actual) != list(expected):
        raise RuntimeError("retained-state names/order differ")
    fields = [{"field": name, "shape": list(value.shape),
               **base.compare(value, expected[name], base.TOLERANCES["float32"])}
              for name, value in actual.items() if value.numel()]
    return {"tolerance": dict(base.TOLERANCES["float32"]), "fields": fields,
            "passed": all(field["passed"] for field in fields)}


def state_identity(state):
    return [{"field": name, "shape": list(value.shape), "dtype": str(value.dtype),
             "sha256": base.tensor_sha256(value), "nonzero_elements": int(torch.count_nonzero(value))}
            for name, value in state.items()]


@torch.no_grad()
def cached_probe(model, x, y, own_full, original_full, prefix_length, chunk_size, recorder,
                 check_limit, original_state=None):
    model.configure_scan_backend("reference", chunk_size)
    logits, states, positions = {}, {}, {}

    def capture(name, values, state=None):
        check_limit()
        length = 1 if name in ("prefill_full", "prefix") else x.shape[1] - prefix_length if name.startswith("suffix_") else x.shape[1]
        base.require_finite(values, (x.shape[0], length, model.cfg.vocab_size), "cached " + name)
        logits[name] = values.detach().cpu()
        if state is not None:
            states[name] = base.state_snapshot(state)
            positions[name] = state.position

    with base.observe_scans(model, recorder), torch.autocast(x.device.type, enabled=False):
        recorder["stage"] = "fp32_stateful_full"
        state = model.init_inference_state(x.shape[0], device=x.device, cache_dtype=torch.float32)
        capture("stateful_full", model(x, inference_state=state)[0], state)
        del state
        recorder["stage"] = "fp32_prefill_full"
        value, state = model.prefill(x, cache_dtype=torch.float32)
        capture("prefill_full", value, state)
        del state
        recorder["stage"] = "fp32_prefix_prefill"
        check_limit()
        value, prefix = model.prefill(x[:, :prefix_length], cache_dtype=torch.float32)
        capture("prefix", value)
        prefix_snapshot = base.state_snapshot(prefix)
        suffix = x[:, prefix_length:]
        schedule = [min(chunk_size, suffix.shape[1] - start) for start in range(0, suffix.shape[1], chunk_size)]
        for name, size in (("suffix_one_shot", suffix.shape[1]), ("suffix_chunked", chunk_size), ("suffix_tokenwise", 1)):
            recorder["stage"] = "fp32_" + name
            state, parts = prefix.clone(), []
            for start in range(0, suffix.shape[1], size):
                check_limit()
                idx = suffix[:, start:start + size]
                value = model.decode(idx, state) if size == 1 else model(idx, inference_state=state)[0]
                parts.append(value.detach().cpu())
            capture(name, torch.cat(parts, 1), state)
            del state, parts
        recorder["stage"] = "fp32_full_tokenwise"
        state, parts = model.init_inference_state(x.shape[0], device=x.device, cache_dtype=torch.float32), []
        for index in range(x.shape[1]):
            check_limit()
            parts.append(model.decode(x[:, index:index + 1], state).detach().cpu())
        capture("full_tokenwise", torch.cat(parts, 1), state)
        del state, parts
        unchanged = base.state_snapshot(prefix)
        if prefix.position != prefix_length or any(not torch.equal(value, unchanged[name]) for name, value in prefix_snapshot.items()):
            raise RuntimeError("a cloned continuation changed the original prefix memory")
    if any(position != x.shape[1] for position in positions.values()):
        raise RuntimeError("cached final position differs from input length")
    expected = {name: (own_full[:, -1:] if name == "prefill_full" else own_full[:, prefix_length - 1:prefix_length]
                      if name == "prefix" else own_full[:, prefix_length:] if name.startswith("suffix_") else own_full)
                for name in logits}
    original_expected = {name: (original_full[:, -1:] if name == "prefill_full" else original_full[:, prefix_length - 1:prefix_length]
                               if name == "prefix" else original_full[:, prefix_length:] if name.startswith("suffix_") else original_full)
                         for name in logits}
    internal = {name: base.compare(value, expected[name], base.TOLERANCES["float32"]) for name, value in logits.items()}
    internal.update({name + "_vs_one_shot": base.compare(logits[name], logits["suffix_one_shot"], base.TOLERANCES["float32"])
                     for name in ("suffix_chunked", "suffix_tokenwise")})
    anchored = {name: base.compare(value, original_expected[name], base.TOLERANCES["float32"]) for name, value in logits.items()}
    retained = {name: state_comparison(value, states["stateful_full"]) for name, value in states.items() if name != "stateful_full"}
    common_states = None if original_state is None else {name: state_comparison(value, original_state) for name, value in states.items()}
    effects = {}
    for name, value in logits.items():
        start = x.shape[1] - 1 if name == "prefill_full" else prefix_length - 1 if name == "prefix" else prefix_length if name.startswith("suffix_") else 0
        effects[name] = base.token_effects(value, original_expected[name], y[:, start:start + value.shape[1]], start)
    return {"tolerance": dict(base.TOLERANCES["float32"]), "cache_dtype": "torch.float32",
            "internal_logits": internal, "original_fp32_anchor_logits": anchored, "retained_state": retained,
            "original_reference_stateful_anchor": common_states, "next_token_effects": effects,
            "real_prefix": {"position": prefix_length, "fields": state_identity(prefix_snapshot), "synthetic": False,
                            "unchanged_after_cloned_continuations": True}, "suffix_length": suffix.shape[1],
            "chunked_schedule": schedule, "one_shot_vs_tokenwise_degenerate": suffix.shape[1] == 1,
            "final_positions": positions, "passed": all(item["passed"] for item in [*internal.values(), *anchored.values(),
                                 *retained.values(), *((common_states or {}).values())])}, states["stateful_full"]


def study_case(model, x, y, *, prefix_length, gradient_check=False, chunk_size=128, check_limit=lambda: None):
    if (x.ndim != 2 or x.shape != y.shape or type(gradient_check) is not bool or type(chunk_size) is not int or chunk_size <= 0
            or type(prefix_length) is not int or not 0 < prefix_length < x.shape[1]
            or x.dtype != torch.int64 or y.dtype != torch.int64
            or x.shape[0] <= 0 or not torch.equal(x[:, 1:], y[:, :-1])):
        raise ValueError("valid paired int64 tokens, substantive prefix and boolean gradient scope required")
    identity, treatments, original, original_state = base.weight_sha256(model), [], None, None
    with base.preserve_model_execution(model):
        for policy in ("original", decay.TREATMENT):
            # No global override for the original control; the candidate context is
            # the same already-tested temporary policy as the narrow decay study.
            from contextlib import nullcontext
            with decay.temporary_decay_treatment() if policy != "original" else nullcontext():
                recorder = {"stage": "unexecuted", "events": {}}
                reference, reference_backend = _full_pass(model, x, y, "reference", chunk_size, gradient_check, recorder, check_limit)
                candidate, candidate_backend = _full_pass(model, x, y, "torch_chunked", chunk_size, gradient_check, recorder, check_limit)
                internal = pass_comparison(candidate, reference, y)
                if original is None:
                    original = reference
                common = {"reference": None if policy == "original" else pass_comparison(reference, original, y),
                          "torch_chunked": pass_comparison(candidate, original, y)}
                cached, state = cached_probe(model, x, y, reference[0], original[0], prefix_length, chunk_size,
                                              recorder, check_limit, original_state)
                if original_state is None:
                    original_state = state
                events = [{**json.loads(key), "calls": calls} for key, calls in sorted(recorder["events"].items())]
                paths = {stage: sorted({event["path"] for event in events if event["stage"] == stage})
                         for stage in sorted({event["stage"] for event in events})}
                treatments.append({"treatment_id": policy, "reference_backend": reference_backend, "candidate_backend": candidate_backend,
                    "scan_function_bindings": decay._bindings(), "fp32_backend": internal,
                    "common_original_fp32_anchor": common, "fp32_cached": cached,
                    "actual_paths": paths, "operator_observations": events,
                    "unexecuted": ["BF16 paths", *([] if gradient_check else ["all parameter gradients"])],
                    "passed": internal["passed"] and cached["passed"] and all(item["passed"] for item in common.values() if item is not None)})
                del reference, candidate, state
    if base.weight_sha256(model) != identity:
        raise RuntimeError("breadth diagnostic changed model weights")
    return {"weights_sha256": identity, "treatments": treatments,
            "original_anchor": {"treatment": "original", "backend": "reference", "logits_sha256": base.tensor_sha256(original[0]),
                "loss_sha256": base.tensor_sha256(original[1].reshape(-1)), "retained_state_scope": "original reference stateful_full",
                "retained_state": state_identity(original_state), "gradient_parameter_tensors": None if original[2] is None else len(original[2])},
            "passed": all(arm["passed"] for arm in treatments)}


def validate_declaration(path, *, device, baseline_declaration, baseline_report, decay_report, seconds, minimum_free):
    planned = base.read_json(path)
    expected = {"schema": 1, "status_at_declaration": "planned_before_execution", "device": device, "seed": 2027,
                "chunk_size": 128, "ratios": list(decay.RATIOS), "treatment_id": decay.TREATMENT,
                "tolerances": base.TOLERANCES, "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
                "cases": list(CASES), "time_budget_seconds": seconds, "minimum_large_case_free_bytes": minimum_free}
    if any(planned.get(key) != value for key, value in expected.items()):
        raise ValueError("breadth declaration differs from the exact case/policy matrix")
    for name, report in (("baseline", baseline_report), ("decay", decay_report)):
        if planned.get(name + "_report") != base.public_reference(report)["path"] or planned.get(name + "_sha256") != base.file_sha256(report):
            raise ValueError("declared " + name + " report path/hash differs")
    if planned.get("baseline_declaration") != base.public_reference(baseline_declaration)["path"]:
        raise ValueError("declared baseline declaration path differs")
    return planned, {**base.public_reference(path), "sha256": base.file_sha256(path)}


def run_study(device="cuda", *, declaration=ROOT / "docs/research/decay-breadth-protocol-2026-10-04.json",
              baseline_declaration=ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json",
              baseline_report=ROOT / "docs/research/checks/trained-numerics-2026-10-04/natural-257-seed-2027.json",
              decay_report=ROOT / "docs/research/checks/decay-precision-2026-10-04/natural-257-seed-2027.json",
              checkpoint_root=Path("checkpoints"), training_run_id="week3-700m-v1",
              data_dir=Path("data/openwebtext-5b"), tokenizer=Path("data/tokenizer/openwebtext.json"),
              time_budget_seconds=900, minimum_large_case_free_bytes=MINIMUM_FREE_BYTES):
    if device not in ("cpu", "cuda") or type(minimum_large_case_free_bytes) is not int or minimum_large_case_free_bytes <= 0:
        raise ValueError("choose cpu/cuda and a positive headroom guard")
    DiagnosticLimit(time_budget_seconds)
    _, baseline = _baseline_identities(baseline_declaration, baseline_report)
    planned, declaration_identity = validate_declaration(declaration, device=device, baseline_declaration=baseline_declaration,
        baseline_report=baseline_report, decay_report=decay_report, seconds=time_budget_seconds, minimum_free=minimum_large_case_free_bytes)
    previous = base.read_json(decay_report)
    if (previous.get("kind") != "trained_checkpoint_decay_precision" or previous.get("execution_status") != "completed"
            or previous.get("certified") is not False or previous["protocol"]["input"] != EXPECTED_INPUT
            or previous["protocol_sha256"] != base.canonical_sha256(previous["protocol"])
            or previous["protocol"]["baseline_report"]["sha256"] != base.file_sha256(baseline_report)):
        raise ValueError("prior decay evidence differs from the exact baseline")
    checkpoints, data_identity, checkpoint_identity = base.audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, decay.RATIOS)
    prepared = prepare_inputs(base.load_split(data_dir, "val"))
    input_identities = [row[2] for row in prepared]
    if (planned.get("shared_inputs") != input_identities or data_identity != baseline["protocol"]["data"]
            or checkpoint_identity != baseline["protocol"]["checkpoints"]
            or prepared[3][2]["windows"] != [EXPECTED_INPUT]):
        raise ValueError("prepared breadth input/data/checkpoint identities differ")
    sources = base.source_metadata()
    for name in ("scripts/study_decay_precision.py", "scripts/isolate_checkpoint_numerics.py", "scripts/check_decay_breadth.py"):
        sources[name] = base.file_sha256(ROOT / name)
    for name, recorded in previous["protocol"]["source_sha256"].items():
        raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if recorded not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()}:
            raise ValueError("measured decay source changed")
    evidence = {Path(path): base.file_sha256(path) for path in (declaration, baseline_declaration, baseline_report, decay_report)}
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA unavailable; no fallback permitted")
    limit, rows, headroom, failure, active_case = DiagnosticLimit(time_budget_seconds), [], [], None, None
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), base.tf32_disabled():
        runtime = base.runtime_metadata(device)
        try:
            for ratio in decay.RATIOS:
                active_case = {"ratio": ratio, "case_id": None, "stage": "model load"}
                limit.check()
                model = base.load_variant_model(checkpoints[ratio], torch.device(device))
                try:
                    old = next(case for case in baseline["cases"] if case["ratio"] == ratio)
                    if base.weight_sha256(model) != old["weights_sha256"]:
                        raise ValueError("loaded model weights differ from baseline")
                    for x, y, identity in prepared:
                        active_case = {"ratio": ratio, "case_id": identity["case_id"], "stage": "full/cached model diagnostic"}
                        limit.check()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            if identity["gradient_check"]:
                                free, total = torch.cuda.mem_get_info()
                                headroom.append({"ratio": ratio, "case_id": identity["case_id"], "free_bytes": free,
                                                 "total_bytes": total, "minimum_free_bytes": minimum_large_case_free_bytes})
                                if free < minimum_large_case_free_bytes:
                                    raise RuntimeError("insufficient declared free headroom for batch2/512; no reduced-shape fallback")
                        started = time.perf_counter()
                        result = study_case(model, x.to(device), y.to(device), prefix_length=identity["prefix_length"],
                                            gradient_check=identity["gradient_check"], check_limit=limit.check)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        rows.append({"ratio": ratio, **identity, **result, "processing_seconds": time.perf_counter() - started,
                                     "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
                finally:
                    del model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
            limit.check()
            active_case = None
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            failure = (f"{type(exc).__name__}: {exc}").replace(str(ROOT), ".").replace(str(Path.home()), "<home>")
    # An incomplete run must also audit its partial evidence; inability to recheck
    # or a changed file is recorded instead of silently claiming stable sources.
    integrity = {}
    for name, entries in (("checkpoints", [(checkpoints[item["ratio"]].best_path, item["sha256"]) for item in checkpoint_identity]),
                          ("declaration_and_prior_evidence", list(evidence.items())),
                          ("measured_sources", [(ROOT / name, recorded) for name, recorded in sources.items()])):
        try:
            integrity[name] = all(base.file_sha256(path) == recorded for path, recorded in entries)
        except OSError:
            integrity[name] = False
    if not all(integrity.values()):
        failure = (failure + "; " if failure else "") + "post-execution file-integrity check failed"
    protocol = {"version": 1, "device": device, "seed": 2027, "ratios": list(decay.RATIOS), "cases": list(CASES),
                "chunk_size": 128, "treatment_id": decay.TREATMENT, "shared_inputs": input_identities,
                "checkpoints": checkpoint_identity, "data": data_identity, "tolerances": base.TOLERANCES,
                "declaration": declaration_identity, "baseline_declaration": {**base.public_reference(baseline_declaration), "sha256": evidence[Path(baseline_declaration)]},
                "baseline_report": {**base.public_reference(baseline_report), "sha256": evidence[Path(baseline_report)]},
                "decay_report": {**base.public_reference(decay_report), "sha256": evidence[Path(decay_report)]},
                "runtime": runtime, "runtime_sha256": base.canonical_sha256(runtime), "source_sha256": sources,
                "time_budget_seconds": time_budget_seconds, "minimum_large_case_free_bytes": minimum_large_case_free_bytes,
                "BF16_executed": False, "production_defaults_changed": False, "optimizer_executed": False, "quality_equivalence_margin": None}
    return {"schema": 1, "kind": "trained_checkpoint_decay_breadth", "certified": False,
            "execution_status": "incomplete" if failure is not None else "completed",
            "status": "incomplete" if failure is not None else "completed" if all(row["passed"] for row in rows) else "completed_with_parity_failures",
            "reason": failure, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "failed_case": active_case if failure is not None else None,
            "protocol": protocol, "protocol_sha256": base.canonical_sha256(protocol), "cases": rows,
            "execution_allowance": limit.metadata(), "large_case_headroom": headroom,
            "post_execution_integrity": integrity,
            "limits": ["Natural windows from the checkpoint-selection pool; no independent quality conclusion.",
                       "Internal consistency, original stateless FP32 logits and original stateful FP32 memory are separate anchors.",
                       "Only batch2/512 records full-model gradients; other gradients and every BF16 route are unexecuted.",
                       "Prefix memory comes from each policy's actual model.prefill and is cloned before each continuation.",
                       "Original tolerances are unchanged; descriptive probability effects have no quality cutoff.",
                       "A free-memory guard is conservative and does not guarantee the peak allocation fits.",
                       "The time allowance is cooperative; a running operation can overshoot. Incomplete cases are not passing cases.",
                       "No production promotion, optimizer, throughput benchmark or backend certification."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/decay-breadth-protocol-2026-10-04.json")
    parser.add_argument("--time-budget-seconds", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new immutable breadth artifact")
    try:
        report = run_study(args.device, declaration=args.declaration, time_budget_seconds=args.time_budget_seconds)
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        reason = f"{type(exc).__name__}: {exc}".replace(str(ROOT), ".").replace(str(Path.home()), "<home>")
        report = {"schema": 1, "kind": "trained_checkpoint_decay_breadth", "certified": False,
                  "status": "incomplete", "execution_status": "incomplete", "reason": reason, "cases": []}
    base.write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "execution_status", "certified")}
                     | {"completed_cases": len(report["cases"])}))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
