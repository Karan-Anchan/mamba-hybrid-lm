"""A named, temporary FP64 cumulative-decay treatment on unchanged trained weights.

The original FP32 dt*A product is retained. Only cumulative sums, differences and
exponentials use FP64; coefficients return to FP32 before unchanged contractions.
No production defaults or optimizer are changed. BF16 is explicitly omitted unless
requested in a matching pre-execution declaration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
import src.model.mamba2 as mamba  # noqa: E402
from scripts.isolate_checkpoint_numerics import EXPECTED_INPUT, _baseline_identities  # noqa: E402
from src.model.scan_backend import BackendUnavailableError  # noqa: E402

TREATMENT = "fp64_cumsum_decay_coefficients"
RATIOS = ("1:3", "1:7", "1:15")
BF16_STAGES = ("bf16_full", "bf16_stateful_full", "bf16_prefix_prefill",
               "bf16_suffix_one_shot", "bf16_suffix_segmented", "bf16_suffix_tokenwise",
               "bf16_full_tokenwise")


def _decay_coefficients(dt, A):
    # Keep the historical product and its derivative in FP32. The conversion to
    # FP64 stays connected to autograd; there is no detached coefficient cache.
    cumulative = (dt * A).double().cumsum(1).transpose(1, 2)
    length = dt.shape[1]
    causal = torch.tril(torch.ones(length, length, device=dt.device, dtype=torch.bool))
    local = cumulative[..., :, None] - cumulative[..., None, :]
    local = local.masked_fill(~causal, float("-inf")).exp().float()
    carry = cumulative.exp().float()
    end = (cumulative[..., -1, None] - cumulative).exp().float()
    return local, carry, end


def ssd_decay_fp64(x, dt, A, B, C, D):
    """Original quadratic scan, with only cumulative-decay coefficients changed."""
    x, dt, A, B, C = x.float(), dt.float(), A.float(), B.float(), C.float()
    decay, _, _ = _decay_coefficients(dt, A)
    cb = torch.einsum("bin,bjn->bij", C, B)
    xdt = x * dt[..., None]
    y = torch.einsum("bhij,bjhp->bihp", cb[:, None] * decay, xdt)
    return y + x * D.float()[None, None, :, None]


def ssd_stateful_decay_fp64(x, dt, A, B, C, D, initial_state):
    """Original carried scan with FP32 decay, carry and final-state coefficients."""
    x, dt, A, B, C = x.float(), dt.float(), A.float(), B.float(), C.float()
    initial_state = initial_state.float()
    decay, carry_scale, end_scale = _decay_coefficients(dt, A)
    cb = torch.einsum("bin,bjn->bij", C, B)
    xdt = x * dt[..., None]
    local_y = torch.einsum("bhij,bjhp->bihp", cb[:, None] * decay, xdt)
    carried_y = torch.einsum("bhl,bhpn,bln->blhp", carry_scale, initial_state, C)
    y = local_y + carried_y + x * D.float()[None, None, :, None]
    local_state = torch.einsum("bhl,blhp,bln->bhpn", end_scale, xdt, B)
    next_state = carry_scale[..., -1, None, None] * initial_state + local_state
    return y, next_state


@contextmanager
def temporary_decay_treatment():
    """Serial diagnostic override; restore both module bindings even on failure."""
    original, original_stateful = mamba.ssd, mamba.ssd_stateful
    try:
        mamba.ssd, mamba.ssd_stateful = ssd_decay_fp64, ssd_stateful_decay_fp64
        yield
    finally:
        mamba.ssd, mamba.ssd_stateful = original, original_stateful


@contextmanager
def capture_passes(common_anchor=None):
    """Observe base's sequential CPU-stored FP32 passes without another graph."""
    forward_backward, bf16_probe = base._forward_backward, base.bf16_cache_probe
    captured = {}

    def capture(model, x, y, backend, chunk_size, recorder):
        if backend in captured:
            raise RuntimeError("duplicate FP32 backend capture")
        result = forward_backward(model, x, y, backend, chunk_size, recorder)
        captured[backend] = result[:3]  # detached CPU logits, scalar loss, all gradients
        return result

    def probe(model, x, y, own_anchor, *args, **kwargs):
        # Treatment BF16 routes share the ORIGINAL reference FP32 anchor. Their
        # internal full-vs-cached checks still compare their own treatment outputs.
        anchor = common_anchor[0] if common_anchor is not None else own_anchor
        return bf16_probe(model, x, y, anchor, *args, **kwargs)

    try:
        base._forward_backward, base.bf16_cache_probe = capture, probe
        yield captured
    finally:
        base._forward_backward, base.bf16_cache_probe = forward_backward, bf16_probe


def _fp32_only(model, x, y, chunk_size):
    recorder = {"stage": "unexecuted", "events": {}}
    identity = base.weight_sha256(model)
    with base.preserve_model_execution(model):
        fp32, _ = base.fp32_backend_probe(model, x, y, chunk_size, recorder)
    if base.weight_sha256(model) != identity:
        raise RuntimeError("diagnostic changed model weights")
    events = [{**json.loads(identity), "calls": count} for identity, count in sorted(recorder["events"].items())]
    paths = {stage: sorted({event["path"] for event in events if event["stage"] == stage})
             for stage in sorted({event["stage"] for event in events})}
    paths.update(dict.fromkeys(BF16_STAGES))
    return {"weights_sha256": identity, "fp32_backend": fp32, "bf16_cached": None,
            "operator_observations": events, "actual_paths": paths, "passed": fp32["passed"]}


def compare_to_common_anchor(actual, expected, targets):
    logits, loss, gradients = actual
    anchor_logits, anchor_loss, anchor_gradients = expected
    checks = {"logits": base.compare(logits, anchor_logits, base.TOLERANCES["float32"]),
              "loss": base.compare(loss, anchor_loss, base.TOLERANCES["float32"]),
              "gradients": base.gradient_comparisons(gradients, anchor_gradients)}
    return {"scope": "separate comparison against original reference FP32 full forward/backward",
            "tolerance": dict(base.TOLERANCES["float32"]), **checks,
            "next_token_effects": base.token_effects(logits, anchor_logits, targets),
            "passed": checks["logits"]["passed"] and checks["loss"]["passed"] and checks["gradients"]["all_passed"]}


def _anchor_identity(anchor):
    logits, loss, gradients = anchor
    return {"treatment": "original", "backend": "reference", "logits_sha256": base.tensor_sha256(logits),
            "loss_sha256": base.tensor_sha256(loss.reshape(-1)), "parameter_tensors": len(gradients),
            "gradients": [{"parameter": name, "shape": list(value.shape), "dtype": str(value.dtype),
                           "sha256": base.tensor_sha256(value)} for name, value in gradients.items()]}


def _bindings():
    return {"reference_scan": mamba.ssd.__module__ + "." + mamba.ssd.__name__,
            "stateful_scan": mamba.ssd_stateful.__module__ + "." + mamba.ssd_stateful.__name__}


def study_model(model, x, y, *, prefix_length=128, chunk_size=128, include_bf16=False,
                tokenwise=True):
    """Same model/input: original and treatment graphs execute sequentially."""
    if (type(include_bf16) is not bool or type(tokenwise) is not bool
            or type(chunk_size) is not int or chunk_size <= 0 or type(prefix_length) is not int
            or x.ndim != 2 or x.shape[0] != 1 or x.shape != y.shape or x.shape[1] <= prefix_length
            or prefix_length <= 0):
        raise ValueError("positive batch-one input, matching targets and valid prefix/chunk/flags required")
    identity = base.weight_sha256(model)

    def execute():
        return (base.study_model(model, x, y, prefix_length=prefix_length, chunk_size=chunk_size, tokenwise=tokenwise)
                if include_bf16 else _fp32_only(model, x, y, chunk_size))

    with capture_passes() as captured:
        original = execute()
    if set(captured) != {"reference", "torch_chunked"}:
        raise RuntimeError("missing FP32 control capture")
    anchor = captured["reference"]
    anchor_identity = _anchor_identity(anchor)
    original["common_original_fp32_anchor"] = {"reference": None,
        "torch_chunked": compare_to_common_anchor(captured["torch_chunked"], anchor, y)}
    original["treatment_id"], original["scan_function_bindings"] = "original", _bindings()
    del captured
    with temporary_decay_treatment(), capture_passes(anchor) as captured:
        treatment = execute()
        treatment["scan_function_bindings"] = _bindings()
    if set(captured) != {"reference", "torch_chunked"}:
        raise RuntimeError("missing FP32 treatment capture")
    treatment["common_original_fp32_anchor"] = {backend: compare_to_common_anchor(values, anchor, y)
                                                for backend, values in captured.items()}
    treatment["treatment_id"] = TREATMENT
    for arm in (original, treatment):
        arm["include_bf16"] = include_bf16
        arm["unexecuted_stages"] = list(arm["bf16_cached"]["unexecuted_stages"]) if include_bf16 else list(BF16_STAGES)
        if include_bf16:
            arm["bf16_cached"]["fp32_anchor"]["anchor_identity"] = {key: anchor_identity[key]
                                                       for key in ("treatment", "backend", "logits_sha256")}
        arm["passed"] = arm["passed"] and all(check["passed"] for check in arm["common_original_fp32_anchor"].values() if check is not None)
        if arm["weights_sha256"] != identity:
            raise RuntimeError("treatment changed model weight identity")
    if base.weight_sha256(model) != identity:
        raise RuntimeError("treatment changed model weights")
    return {"weights_sha256": identity, "common_original_fp32_anchor": anchor_identity,
            "treatments": [original, treatment], "passed": all(arm["passed"] for arm in (original, treatment))}


def validate_declaration(path, *, baseline_report, baseline_declaration, isolation_report, device, include_bf16):
    planned = base.read_json(path)
    expected = {"schema": 1, "status_at_declaration": "planned_before_execution", "device": device,
                "seed": 2027, "length": 257, "prefix_length": 128, "chunk_size": 128,
                "start_token": EXPECTED_INPUT["start_token"], "batch_size": 1, "ratios": list(RATIOS),
                "tokens_sha256": EXPECTED_INPUT["tokens_sha256"], "targets_sha256": EXPECTED_INPUT["targets_sha256"],
                "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
                "tolerances": base.TOLERANCES, "treatment_id": TREATMENT, "include_bf16": include_bf16}
    if any(planned.get(key) != value for key, value in expected.items()):
        raise ValueError("decay declaration differs from the exact treatment/input protocol")
    for name, report in (("baseline", baseline_report), ("isolation", isolation_report)):
        if (planned.get(name + "_report") != base.public_reference(report)["path"]
                or planned.get(name + "_sha256") != base.file_sha256(report)):
            raise ValueError("declared " + name + " path/hash differs")
    if planned.get("baseline_declaration") != base.public_reference(baseline_declaration)["path"]:
        raise ValueError("declared baseline declaration path differs")
    return {**base.public_reference(path), "sha256": base.file_sha256(path)}


def run_study(device="cuda", *, include_bf16=False, tokenwise=True,
              declaration=ROOT / "docs/research/decay-precision-protocol-2026-10-04.json",
              baseline_declaration=ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json",
              baseline_report=ROOT / "docs/research/checks/trained-numerics-2026-10-04/natural-257-seed-2027.json",
              isolation_report=ROOT / "docs/research/checks/numerical-isolation-2026-10-04/natural-257-seed-2027.json",
              checkpoint_root=Path("checkpoints"), training_run_id="week3-700m-v1",
              data_dir=Path("data/openwebtext-5b"), tokenizer=Path("data/tokenizer/openwebtext.json")):
    if device not in ("cpu", "cuda") or type(include_bf16) is not bool or type(tokenwise) is not bool:
        raise ValueError("choose cpu/cuda and boolean BF16/tokenwise settings")
    _, baseline = _baseline_identities(baseline_declaration, baseline_report)
    planned = validate_declaration(declaration, baseline_report=baseline_report, baseline_declaration=baseline_declaration,
        isolation_report=isolation_report, device=device, include_bf16=include_bf16)
    if include_bf16 and base.read_json(declaration).get("full_tokenwise", True) != tokenwise:
        raise ValueError("declared BF16 tokenwise scope differs")
    evidence_files = {Path(path): base.file_sha256(path) for path in
                      (declaration, baseline_declaration, baseline_report, isolation_report)}
    isolated = base.read_json(isolation_report)
    if (isolated.get("kind") != "trained_checkpoint_operation_isolation" or isolated.get("execution_status") != "completed"
            or isolated.get("certified") is not False or isolated["protocol"]["input"] != EXPECTED_INPUT
            or isolated["protocol_sha256"] != base.canonical_sha256(isolated["protocol"])
            or isolated["protocol"]["baseline_report"]["sha256"] != base.file_sha256(baseline_report)):
        raise ValueError("isolation evidence differs from the exact baseline")
    checkpoints, data_identity, checkpoint_identity = base.audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, RATIOS)
    tokens, targets, input_identity = base.shared_windows(base.load_split(data_dir, "val"), 257, 1, 2027)[0]
    old_protocol = baseline["protocol"]
    if input_identity != EXPECTED_INPUT or data_identity != old_protocol["data"] or checkpoint_identity != old_protocol["checkpoints"]:
        raise ValueError("prepared window/data/checkpoints changed since the original diagnostic")
    sources = base.source_metadata()
    for name in ("scripts/study_decay_precision.py", "scripts/isolate_checkpoint_numerics.py"):
        sources[name] = base.file_sha256(ROOT / name)
    for name, previous in old_protocol["source_sha256"].items():
        raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if previous not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()}:
            raise ValueError("measured baseline source changed")
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA unavailable; no fallback permitted")
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    cases = []
    with torch.random.fork_rng(devices=devices), base.tf32_disabled():
        runtime = base.runtime_metadata(device)
        for ratio in RATIOS:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            model = base.load_variant_model(checkpoints[ratio], torch.device(device))
            try:
                result = study_model(model, tokens.to(device), targets.to(device), include_bf16=include_bf16, tokenwise=tokenwise)
                old = next(case for case in baseline["cases"] if case["ratio"] == ratio)
                if result["weights_sha256"] != old["weights_sha256"]:
                    raise ValueError("loaded weights differ from baseline")
                if device == "cuda":
                    torch.cuda.synchronize()
                reproduced = result["treatments"][0]["fp32_backend"]["logits"]["passed"] == old["fp32_backend"]["logits"]["passed"]
                cases.append({"ratio": ratio, **input_identity, **result, "passed": result["passed"] and reproduced,
                              "baseline_fp32_logits": old["fp32_backend"]["logits"],
                              "baseline_pass_flag_reproduced": reproduced,
                              "processing_seconds": time.perf_counter() - started,
                              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
            finally:
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        for checkpoint in checkpoint_identity:
            if base.file_sha256(checkpoints[checkpoint["ratio"]].best_path) != checkpoint["sha256"]:
                raise RuntimeError("historical checkpoint bytes changed during diagnostic")
        if any(base.file_sha256(path) != previous for path, previous in evidence_files.items()):
            raise RuntimeError("declared evidence bytes changed during diagnostic")
        if any(base.file_sha256(ROOT / name) != previous for name, previous in sources.items()):
            raise RuntimeError("measured source bytes changed during diagnostic")
    protocol = {"version": 1, "device": device, "ratios": list(RATIOS), "seed": 2027, "length": 257,
                "prefix_length": 128, "chunk_size": 128, "include_bf16": include_bf16, "full_tokenwise": tokenwise if include_bf16 else None,
                "input": input_identity, "checkpoints": checkpoint_identity, "data": data_identity,
                "treatment_id": TREATMENT, "treatment_policy": {
                    "dt_times_A": "original FP32 product", "cumsum_subtraction_exp": "FP64",
                    "decay_carry_end_coefficients": "cast to FP32 before unchanged contractions",
                    "contractions_and_autocast": "original policy preserved", "production": False},
                "declaration": planned, "baseline_declaration": {**base.public_reference(baseline_declaration), "sha256": base.file_sha256(baseline_declaration)},
                "baseline_report": {**base.public_reference(baseline_report), "sha256": base.file_sha256(baseline_report), "protocol_sha256": baseline["protocol_sha256"]},
                "isolation_report": {**base.public_reference(isolation_report), "sha256": base.file_sha256(isolation_report), "protocol_sha256": isolated["protocol_sha256"]},
                "tolerances": {key: dict(value) for key, value in base.TOLERANCES.items()},
                "runtime": runtime, "runtime_sha256": base.canonical_sha256(runtime), "source_sha256": sources,
                "optimizer_executed": False, "production_defaults_changed": False, "quality_equivalence_margin": None}
    return {"schema": 1, "kind": "trained_checkpoint_decay_precision", "certified": False,
            "status": "completed" if all(case["passed"] for case in cases) else "completed_with_parity_failures",
            "execution_status": "completed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol, "protocol_sha256": base.canonical_sha256(protocol), "cases": cases,
            "limits": ["Exact checkpoint-selection validation window; no independent quality or statistical replication.",
                       "Internal reference/chunked consistency and agreement with original FP32 anchor are separate gates.",
                       "All original tensor tolerances are retained; next-token probability changes are descriptive.",
                       "Temporary serial diagnostic overrides are restored; no production policy or optimizer is changed.",
                       "FP64 coefficients remain connected to autograd, then return to FP32 before original contractions.",
                       "BF16 routes are explicitly unexecuted when include_bf16 is false.",
                       "Batch-two, length-512, broader shapes and trained-cache qualification remain separate gates.",
                       "Processing time includes auditing and diagnostics; no throughput claim or backend certification."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--include-bf16", action="store_true")
    parser.add_argument("--skip-full-tokenwise", action="store_true")
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/decay-precision-protocol-2026-10-04.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new immutable decay artifact")
    try:
        report = run_study(args.device, include_bf16=args.include_bf16, tokenwise=not args.skip_full_tokenwise, declaration=args.declaration)
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        reason = f"{type(exc).__name__}: {exc}".replace(str(ROOT), ".").replace(str(Path.home()), "<home>")
        report = {"schema": 1, "kind": "trained_checkpoint_decay_precision", "certified": False,
                  "status": "incomplete", "execution_status": "incomplete", "reason": reason}
    base.write_report(args.output, report)
    print(json.dumps({"status": report["status"], "execution_status": report["execution_status"], "certified": False}))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
