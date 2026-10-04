"""Read-only FP32 operation isolation for the exact trained K2 failure and control.

Hooks only observe model execution. Frozen scan replays use identical operands;
an independent sequential FP64 recurrence is exploratory, not a production policy.
All original tolerances remain unchanged and failed endpoints stay visible.
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
from scripts.check_scan_backend import TOLERANCES  # noqa: E402
from scripts.study_checkpoint_numerics import (audit_inputs, canonical_sha256, file_sha256,
    load_split, load_variant_model, preserve_model_execution, public_reference, read_json,
    require_finite, runtime_metadata, shared_windows, source_metadata, tensor_sha256,
    tf32_disabled, weight_sha256, write_report)  # noqa: E402
from src.model.mamba2 import ssd, ssd_stateful  # noqa: E402
from src.model.scan_backend import BackendUnavailableError  # noqa: E402

RATIOS = ("1:15", "1:3")
EXPECTED_INPUT = {"window": 0, "start_token": 2342241, "length": 257,
                  "tokens_sha256": "f93bd06db8127edd86f15292d53df7c68b2333252ddc871fa44a0cf66c4d9818",
                  "targets_sha256": "1b6135f43e80589673fce4c2b3fcdd8074744ab60fb856d46637d1d9415502b9",
                  "target_policy": "y[i] = validation[start_token+i+1]"}
OPERANDS = ("x", "dt", "A", "B", "C", "D")


def coordinate(flat: int, shape: tuple) -> list[int]:
    result = []
    for size in reversed(shape):
        result.append(flat % size)
        flat //= size
    return list(reversed(result))


@torch.no_grad()
def detailed_comparison(actual, expected, *, coordinate_limit=8, comparison_dtype="float32") -> dict:
    if (not isinstance(coordinate_limit, int) or isinstance(coordinate_limit, bool)
            or not 1 <= coordinate_limit <= 32):
        raise ValueError("coordinate_limit must be an integer in 1..32")
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("comparison tensors must have identical, nonempty shapes")
    if comparison_dtype not in ("float32", "float64"):
        raise ValueError("comparison dtype must be float32 or float64")
    require_finite(actual, tuple(expected.shape), "actual comparison tensor")
    require_finite(expected, tuple(expected.shape), "reference comparison tensor")
    dtype = torch.float64 if comparison_dtype == "float64" else torch.float32
    a, b = actual.detach().cpu().to(dtype), expected.detach().cpu().to(dtype)
    error = (a - b).abs()
    tolerance = TOLERANCES["float32"]
    ratio = error / (tolerance["atol"] + tolerance["rtol"] * b.abs())
    require_finite(error, tuple(a.shape), "comparison absolute error")
    require_finite(ratio, tuple(a.shape), "comparison tolerance ratio")
    nonzero = torch.nonzero(error.flatten() != 0).flatten()
    violations = torch.nonzero(ratio.flatten() > 1).flatten()

    def sample(index):
        index = int(index)
        return {"coordinate": coordinate(index, tuple(a.shape)), "actual": a.flatten()[index].item(),
                "reference": b.flatten()[index].item(), "absolute_error": error.flatten()[index].item(),
                "tolerance_ratio": ratio.flatten()[index].item()}

    return {"finite": True, "passed": violations.numel() == 0, "shape": list(a.shape),
            "actual_dtype": str(actual.dtype), "reference_dtype": str(expected.dtype),
            "comparison_dtype": str(dtype), "tolerance": dict(tolerance), "elements": a.numel(),
            "nonzero_error_count": nonzero.numel(), "violation_count": violations.numel(),
            "max_absolute_error": error.max().item(), "max_tolerance_ratio": ratio.max().item(),
            "first_nonzero": sample(nonzero[0]) if nonzero.numel() else None,
            "first_violation": sample(violations[0]) if violations.numel() else None,
            "worst_absolute_error": sample(error.flatten().argmax()),
            "worst_tolerance_ratio": sample(ratio.flatten().argmax()),
            "violation_samples": [sample(index) for index in violations[:coordinate_limit]]}


@torch.no_grad()
def recurrence_fp64(operands: tuple, initial: torch.Tensor):
    """Independent elementwise recurrence, with no SSD/cumulative-sum/einsum reuse."""
    x, dt, A, B, C, D = (value.detach().double() for value in operands)
    if x.ndim != 4 or B.ndim != 3 or any(size <= 0 for size in x.shape) or B.shape[-1] <= 0:
        raise ValueError("oracle requires positive B,L,H,P,N dimensions")
    batch, length, heads, width = x.shape
    shapes = ((batch, length, heads), (heads,), (batch, length, B.shape[-1]), (batch, length, B.shape[-1]), (heads,))
    for name, value, shape in zip(OPERANDS[1:], (dt, A, B, C, D), shapes):
        require_finite(value, shape, "FP64 oracle operand " + name)
    require_finite(x, tuple(x.shape), "FP64 oracle x")
    require_finite(initial, (batch, heads, width, B.shape[-1]), "FP64 oracle initial state")
    memory, outputs = initial.detach().double().clone(), []
    for index in range(x.shape[1]):
        decay = torch.exp(dt[:, index] * A)[..., None, None]
        drive = (x[:, index] * dt[:, index, :, None])[..., None] * B[:, index, None, None, :]
        memory = decay * memory + drive
        output = (memory * C[:, index, None, None, :]).sum(-1) + x[:, index] * D[None, :, None]
        outputs.append(output)
    return torch.stack(outputs, dim=1), memory


@torch.no_grad()
def scan_sequence(operands: tuple, initial: torch.Tensor, chunk_size: int):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    outputs, memory = [], initial.detach().clone()
    x, dt, A, B, C, D = operands
    for start in range(0, x.shape[1], chunk_size):
        end = start + chunk_size
        output, memory = ssd_stateful(x[:, start:end], dt[:, start:end], A,
                                     B[:, start:end], C[:, start:end], D, memory)
        outputs.append(output)
    return torch.cat(outputs, dim=1), memory


@torch.no_grad()
def decay_precision_contrast(operands: tuple, *, multiply_fp64: bool):
    """Exploratory zero-state scan: higher-precision decay, original FP32 contractions."""
    x, dt, A, B, C, D = (value.detach().float() for value in operands)
    dtA = dt.double() * A.double() if multiply_fp64 else (dt * A).double()
    cumulative = dtA.cumsum(1).transpose(1, 2)
    log_decay = cumulative[..., :, None] - cumulative[..., None, :]
    length = x.shape[1]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
    decay = log_decay.masked_fill(~causal, float("-inf")).exp().float()
    cb = torch.einsum("bin,bjn->bij", C, B)
    output = torch.einsum("bhij,bjhp->bihp", cb[:, None] * decay, x * dt[..., None])
    return output + x * D[None, None, :, None]


class Trace:
    def __init__(self):
        self.stages, self.scans = [], {}

    def capture(self, layer, stage, value, axes):
        require_finite(value, tuple(value.shape), f"trace {stage}")
        self.stages.append({"layer": layer, "stage": stage, "axes": axes,
                            "value": value.detach().cpu().clone()})


class TraceScan:
    def __init__(self, backend, layer, trace):
        self.backend, self.layer, self.trace = backend, layer, trace
        self.name, self.chunk_size = backend.name, backend.chunk_size

    def scan(self, *args, **kwargs):
        # Freeze before a stateful caller can copy its newly computed memory back.
        operands = tuple(value.detach().cpu().clone() for value in args[:6])
        initial = None if args[6] is None else args[6].detach().cpu().clone()
        for name, value in zip(OPERANDS, operands):
            axes = ["batch", "token", "head", "head_feature"] if name == "x" else (
                ["batch", "token", "head"] if name == "dt" else ["head"] if name in ("A", "D")
                else ["batch", "token", "state_feature"])
            self.trace.capture(self.layer, "scan_input." + name, value, axes)
        result = self.backend.scan(*args, **kwargs)
        self.trace.capture(self.layer, "scan_output", result[0], ["batch", "token", "head", "head_feature"])
        self.trace.scans[self.layer] = {"operands": operands, "initial": initial,
            "output": result[0].detach().cpu().clone(),
            "final": None if result[1] is None else result[1].detach().cpu().clone(),
            "path": "torch.quadratic_ssd" if self.backend.name == "reference" and args[6] is None else "torch.chunked_ssd"}
        return result


@contextmanager
def trace_execution(model, trace: Trace, selected_layers=None):
    hooks, backends = [], []
    axes = ["batch", "token", "feature"]

    def output_hook(layer, stage, transform=None):
        def hook(_module, _inputs, output):
            trace.capture(layer, stage, transform(output) if transform else output, axes)
        return hook

    def input_hook(layer, stage):
        return lambda _module, inputs: trace.capture(layer, stage, inputs[0], axes)

    try:
        if selected_layers is None:
            hooks.append(model.embed.register_forward_hook(output_hook(None, "embedding")))
        for layer, block in enumerate(model.blocks):
            if selected_layers is not None and layer not in selected_layers:
                continue
            if selected_layers is None:
                hooks.append(block.register_forward_pre_hook(input_hook(layer, "block_input")))
                hooks.append(block.norm1.register_forward_hook(output_hook(layer, "norm1_output")))
            if not block.is_attn:
                mixer = block.mixer
                backends.append((mixer, mixer.scan_backend))
                mixer.scan_backend = TraceScan(mixer.scan_backend, layer, trace)
                if selected_layers is None:
                    hooks.append(mixer.in_proj.register_forward_hook(output_hook(layer, "in_projection")))
                    hooks.append(mixer.conv1d.register_forward_hook(output_hook(layer, "convolution",
                        lambda output: output[..., :trace.length].transpose(1, 2))))
                    hooks.append(mixer.norm.register_forward_pre_hook(input_hook(layer, "gated_norm_input")))
                    hooks.append(mixer.norm.register_forward_hook(output_hook(layer, "gated_norm_output")))
            elif selected_layers is None:
                hooks.append(block.mixer.qkv.register_forward_hook(output_hook(layer, "attention_qkv")))
            if selected_layers is None:
                hooks.append(block.mixer.register_forward_hook(output_hook(layer, "mixer_output")))
                if block.has_mlp:
                    hooks.append(block.norm2.register_forward_hook(output_hook(layer, "norm2_output")))
                    hooks.append(block.mlp.register_forward_hook(output_hook(layer, "mlp_output")))
                hooks.append(block.register_forward_hook(output_hook(layer, "block_output")))
        if selected_layers is None:
            hooks.append(model.norm_f.register_forward_hook(output_hook(None, "final_norm")))
            hooks.append(model.lm_head.register_forward_hook(output_hook(None, "logits")))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for mixer, backend in backends:
            mixer.scan_backend = backend


@torch.no_grad()
def model_trace(model, tokens, backend, chunk_size):
    metadata = model.configure_scan_backend(backend, chunk_size)
    trace = Trace()
    trace.length = tokens.shape[1]
    with trace_execution(model, trace), torch.autocast(tokens.device.type, enabled=False):
        logits, _ = model(tokens)
    require_finite(logits, (1, tokens.shape[1], model.cfg.vocab_size), "traced model logits")
    return trace, metadata


def compare_traces(actual: Trace, expected: Trace, coordinate_limit):
    if [(row["layer"], row["stage"], row["axes"]) for row in actual.stages] != [
            (row["layer"], row["stage"], row["axes"]) for row in expected.stages]:
        raise RuntimeError("model trace stage identities differ")
    return [{"layer": a["layer"], "stage": a["stage"], "axes": a["axes"],
             **detailed_comparison(a["value"], b["value"], coordinate_limit=coordinate_limit)}
            for a, b in zip(actual.stages, expected.stages)]


@torch.no_grad()
def real_prefix_scans(model, tokens, layers, prefix_length, chunk_size):
    model.configure_scan_backend("reference", chunk_size)
    with torch.autocast(tokens.device.type, enabled=False):
        _, prefix = model.prefill(tokens[:, :prefix_length], cache_dtype=torch.float32)
        snapshots = {layer: {"conv": prefix.layers[layer].conv.detach().cpu().clone(),
                             "ssm": prefix.layers[layer].ssm.detach().cpu().clone()} for layer in layers}
        continuation = prefix.clone()
        trace = Trace()
        with trace_execution(model, trace, set(layers)):
            model(tokens[:, prefix_length:], inference_state=continuation)
        for layer in layers:
            for name, value in snapshots[layer].items():
                if not torch.equal(value, getattr(prefix.layers[layer], name).cpu()):
                    raise RuntimeError("continuation mutated the original model prefix state")
            if not torch.equal(trace.scans[layer]["initial"], snapshots[layer]["ssm"]):
                raise RuntimeError("captured suffix state differs from real model prefix memory")
        if prefix.position != prefix_length or continuation.position != tokens.shape[1]:
            raise RuntimeError("real continuation positions differ")
    return trace.scans, snapshots


@torch.no_grad()
def frozen_scan_probe(scan, device, *, chunk_size, coordinate_limit, initial_kind):
    operands = tuple(value.detach().to(device) for value in scan["operands"])
    x, dt, A, B, C, D = operands
    initial = (torch.zeros(x.shape[0], x.shape[2], x.shape[3], B.shape[-1], device=device, dtype=torch.float32)
               if scan["initial"] is None else scan["initial"].detach().to(device))
    before = tensor_sha256(initial)
    with torch.autocast(x.device.type, enabled=False):
        oracle, oracle_state = recurrence_fp64(operands, initial)
        treatments = {}
        if scan["initial"] is None:
            treatments["quadratic"] = (ssd(*operands), None)
        treatments["stateful_one_shot"] = scan_sequence(operands, initial, x.shape[1])
        treatments["chunked128"] = scan_sequence(operands, initial, chunk_size)
        treatments["tokenwise"] = scan_sequence(operands, initial, 1)
        results = {name: {"output_vs_fp64": detailed_comparison(output, oracle,
                        coordinate_limit=coordinate_limit, comparison_dtype="float64"),
                          "state_vs_fp64": None if memory is None else detailed_comparison(memory, oracle_state,
                        coordinate_limit=coordinate_limit, comparison_dtype="float64")}
                   for name, (output, memory) in treatments.items()}
        anchor_name = "quadratic" if "quadratic" in treatments else "stateful_one_shot"
        pairs = {name + "_vs_" + anchor_name: detailed_comparison(output, treatments[anchor_name][0], coordinate_limit=coordinate_limit)
                 for name, (output, _) in treatments.items() if name != anchor_name}
        observed_name = "quadratic" if scan["initial"] is None else "chunked128"
        observed = detailed_comparison(treatments[observed_name][0], scan["output"].to(device), coordinate_limit=coordinate_limit)
        operations, contrasts = {}, {}
        if scan["initial"] is None:
            dtA32, dtA64 = dt.float() * A.float(), dt.double() * A.double()
            cum32, cum64 = dtA32.cumsum(1), dtA64.cumsum(1)
            cb32 = torch.einsum("bin,bjn->bij", C.float(), B.float())
            cb64 = (C.double()[:, :, None, :] * B.double()[:, None, :, :]).sum(-1)
            for name, actual, expected in (("dt_times_A", dtA32, dtA64), ("global_cumulative_log_decay", cum32, cum64), ("C_dot_B", cb32, cb64)):
                operations[name] = detailed_comparison(actual, expected, coordinate_limit=coordinate_limit, comparison_dtype="float64")
            for multiply64 in (False, True):
                name = "fp64_cumsum_decay" if not multiply64 else "fp64_product_cumsum_decay"
                output = decay_precision_contrast(operands, multiply_fp64=multiply64)
                contrasts[name] = {"exploratory": True, "production": False,
                    "policy": "FP64 cumulative decay coefficients, cast to FP32 before unchanged contractions; "
                              + ("FP64 dt*A" if multiply64 else "original FP32 dt*A"),
                    "output_vs_fp64": detailed_comparison(output, oracle, coordinate_limit=coordinate_limit, comparison_dtype="float64"),
                    "output_vs_quadratic": detailed_comparison(output, treatments["quadratic"][0], coordinate_limit=coordinate_limit)}
    if tensor_sha256(initial) != before:
        raise RuntimeError("frozen replay mutated its initial state")
    all_checks = [results[name]["output_vs_fp64"] for name in results]
    all_checks += [value["state_vs_fp64"] for value in results.values() if value["state_vs_fp64"] is not None]
    all_checks += [*pairs.values(), observed, *operations.values()]
    all_checks += [value[key] for value in contrasts.values() for key in ("output_vs_fp64", "output_vs_quadratic")]
    return {"scope": "same frozen reference operands; exploratory independent FP64 anchor",
            "observed_scan_path": scan.get("path"),
            "operand_sha256": {name: tensor_sha256(value) for name, value in zip(OPERANDS, operands)},
            "operand_shapes": {name: list(value.shape) for name, value in zip(OPERANDS, operands)},
            "initial_state": {"kind": initial_kind, "synthetic_random": False, "shape": list(initial.shape),
                              "sha256": before, "dtype": str(initial.dtype), "nonzero_elements": int(torch.count_nonzero(initial))},
            "oracle": {"method": "independent sequential elementwise recurrence", "dtype": "torch.float64",
                       "output_sha256": tensor_sha256(oracle), "final_state_sha256": tensor_sha256(oracle_state)},
            "treatments": results, "internal_pairs": pairs, "replay_vs_observed_scan": observed,
            "operation_probes": operations, "exploratory_decay_contrasts": contrasts,
            "all_comparisons_passed": all(check["passed"] for check in all_checks),
            "unexecuted": ["zero-state quadratic and decay contrasts"] if scan["initial"] is not None else []}


def _first(rows, field):
    row = next((row for row in rows if row[field] > 0), None)
    return None if row is None else {"layer": row["layer"], "stage": row["stage"], "axes": row["axes"],
        "nonzero_error_count": row["nonzero_error_count"], "violation_count": row["violation_count"],
        "first_nonzero": row["first_nonzero"], "first_violation": row["first_violation"]}


@torch.no_grad()
def isolate_model(model, tokens, *, prefix_length=128, chunk_size=128, coordinate_limit=8):
    identity = weight_sha256(model)
    with preserve_model_execution(model):
        reference, reference_backend = model_trace(model, tokens, "reference", chunk_size)
        candidate, candidate_backend = model_trace(model, tokens, "torch_chunked", chunk_size)
        rows = compare_traces(candidate, reference, coordinate_limit)
        nonzero, violation = _first(rows, "nonzero_error_count"), _first(rows, "violation_count")
        scans = [row for row in rows if row["stage"] == "scan_output"]
        first_scan = next((row["layer"] for row in scans if row["nonzero_error_count"]), scans[0]["layer"])
        layers = [first_scan]
        if violation:
            limit = violation["layer"] if violation["layer"] is not None else len(model.blocks) - 1
            upstream = max(layer for layer in reference.scans if layer <= limit)
            if upstream not in layers:
                layers.append(upstream)
        suffix_scans, prefix_states = real_prefix_scans(model, tokens, layers, prefix_length, chunk_size)
        replay, suggestions = [], []
        for layer in layers:
            full = frozen_scan_probe(reference.scans[layer], tokens.device, chunk_size=chunk_size,
                                      coordinate_limit=coordinate_limit, initial_kind="declared zero initial condition for stateless full scan")
            continued = frozen_scan_probe(suffix_scans[layer], tokens.device, chunk_size=chunk_size,
                                      coordinate_limit=coordinate_limit, initial_kind="actual reference-model prefill prefix memory")
            replay.append({"layer": layer, "selection": "first nonzero scan drift" if layer == first_scan else "at or upstream of first tolerance violation",
                           "operand_pair_checks": {name: detailed_comparison(a, b, coordinate_limit=coordinate_limit)
                               for name, a, b in zip(OPERANDS, candidate.scans[layer]["operands"], reference.scans[layer]["operands"])},
                           "full_zero_state": full, "real_prefix_continuation": continued,
                           "real_prefix": {"position": prefix_length, "origin": "reference backend model.prefill with FP32 cache",
                               "unchanged_after_cloned_continuation": True,
                               "conv_sha256": tensor_sha256(prefix_states[layer]["conv"]),
                               "ssm_sha256": tensor_sha256(prefix_states[layer]["ssm"]), "synthetic": False}})
            baseline_error = full["treatments"]["quadratic"]["output_vs_fp64"]["max_absolute_error"]
            for name, contrast in full["exploratory_decay_contrasts"].items():
                error = contrast["output_vs_fp64"]["max_absolute_error"]
                if error < baseline_error:
                    suggestions.append({"layer": layer, "candidate": name, "basis": "same-operand maximum scan error against FP64 decreases",
                        "baseline_max_absolute_error": baseline_error, "candidate_max_absolute_error": error,
                        "next_test": "separately named numerical treatment; compare full logits, gradients, actual states and other boundary shapes before any promotion",
                        "approved_for_production": False})
    if weight_sha256(model) != identity:
        raise RuntimeError("isolation changed model weights")
    endpoint = rows[-1]
    return {"weights_sha256": identity, "reference_backend": reference_backend, "candidate_backend": candidate_backend,
            "trace": rows, "first_nonzero_drift": nonzero, "first_tolerance_violation": violation,
            "logits": endpoint, "isolation_layers": layers, "frozen_replays": replay, "candidate_experiments": suggestions,
            "endpoint_passed": endpoint["passed"], "all_trace_checks_passed": all(row["passed"] for row in rows),
            "all_frozen_checks_passed": all(probe[scope]["all_comparisons_passed"] for probe in replay
                                             for scope in ("full_zero_state", "real_prefix_continuation"))}


def _baseline_identities(declaration, baseline_report):
    declaration_data, baseline = read_json(declaration), read_json(baseline_report)
    protocol = baseline["protocol"]
    if (baseline["protocol_sha256"] != canonical_sha256(protocol) or baseline["execution_status"] != "completed"
            or baseline["certified"] is not False or protocol["shared_inputs"] != [EXPECTED_INPUT]
            or protocol["seed"] != 2027 or protocol["length"] != 257 or protocol["prefix_length"] != 128
            or protocol["chunk_size"] != 128 or protocol["tolerances"] != TOLERANCES
            or declaration_data["tolerances"] != TOLERANCES):
        raise ValueError("baseline/declaration differs from the exact failing natural-window protocol")
    checkpoints = {item["ratio"]: item for item in protocol["checkpoints"]}
    declared = {item["ratio"]: item for item in declaration_data["checkpoints"]}
    for ratio in RATIOS:
        if any(checkpoints[ratio][key] != declared[ratio][key] for key in ("path", "sha256")):
            raise ValueError("baseline checkpoint differs from declaration")
    return declaration_data, baseline


def validate_isolation_declaration(path, *, baseline_report, declaration, device, coordinate_limit):
    planned = read_json(path)
    expected = {"schema": 1, "status_at_declaration": "planned_before_execution", "device": device,
                "seed": 2027, "length": 257, "prefix_length": 128, "chunk_size": 128,
                "start_token": 2342241, "batch_size": 1, "ratios": list(RATIOS),
                "tokens_sha256": EXPECTED_INPUT["tokens_sha256"], "targets_sha256": EXPECTED_INPUT["targets_sha256"],
                "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
                "coordinate_limit": coordinate_limit, "maximum_frozen_layers_per_model": 2}
    if any(planned.get(key) != value for key, value in expected.items()):
        raise ValueError("new isolation declaration differs from the exact execution protocol")
    declared_tolerances = planned.get("tolerances")
    declared_fp32 = planned.get("tolerance", (declared_tolerances or {}).get("float32"))
    if declared_fp32 != TOLERANCES["float32"] or (declared_tolerances is not None and declared_tolerances != TOLERANCES):
        raise ValueError("new isolation declaration changed original tolerances")
    if (planned.get("baseline_sha256") != file_sha256(baseline_report)
            or planned.get("baseline_report") != public_reference(baseline_report)["path"]
            or planned.get("baseline_declaration") != public_reference(declaration)["path"]):
        raise ValueError("new isolation declaration baseline hash/path differs")
    return {**public_reference(path), "sha256": file_sha256(path)}


def run_study(device="cuda", *, declaration=ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json",
              baseline_report=ROOT / "docs/research/checks/trained-numerics-2026-10-04/natural-257-seed-2027.json",
              isolation_declaration=ROOT / "docs/research/numerical-isolation-protocol-2026-10-04.json",
              checkpoint_root=Path("checkpoints"), training_run_id="week3-700m-v1",
              data_dir=Path("data/openwebtext-5b"), tokenizer=Path("data/tokenizer/openwebtext.json"),
              coordinate_limit=8):
    if device not in ("cpu", "cuda") or not isinstance(coordinate_limit, int) or isinstance(coordinate_limit, bool) or not 1 <= coordinate_limit <= 32:
        raise ValueError("choose cpu/cuda and coordinate_limit 1..32")
    declaration_data, baseline = _baseline_identities(Path(declaration), Path(baseline_report))
    planned = None
    if isolation_declaration is not None:
        planned = validate_isolation_declaration(isolation_declaration, baseline_report=baseline_report,
            declaration=declaration, device=device, coordinate_limit=coordinate_limit)
    checkpoints, data_identity, checkpoint_identity = audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, RATIOS)
    tokens, targets, input_identity = shared_windows(load_split(data_dir, "val"), 257, 1, 2027)[0]
    if input_identity != EXPECTED_INPUT:
        raise ValueError("prepared validation window differs from the exact baseline")
    original_protocol = baseline["protocol"]
    if data_identity != original_protocol["data"]:
        raise ValueError("prepared data identities changed since the original diagnostic")
    old_checkpoints = {item["ratio"]: item for item in original_protocol["checkpoints"]}
    if any(item != old_checkpoints[item["ratio"]] for item in checkpoint_identity):
        raise ValueError("registered checkpoint identities changed since the original diagnostic")
    sources = source_metadata()
    sources["scripts/isolate_checkpoint_numerics.py"] = file_sha256(Path(__file__))
    for name, previous in original_protocol["source_sha256"].items():
        raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if previous not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()}:
            raise ValueError("measured model/diagnostic source changed since baseline")
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA unavailable; no fallback permitted")
    cases = []
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), tf32_disabled():
        runtime = runtime_metadata(device)
        for ratio in RATIOS:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            model = load_variant_model(checkpoints[ratio], torch.device(device))
            try:
                result = isolate_model(model, tokens.to(device), coordinate_limit=coordinate_limit)
                baseline_case = next(case for case in baseline["cases"] if case["ratio"] == ratio)
                old_weights = baseline_case["weights_sha256"]
                if result["weights_sha256"] != old_weights:
                    raise ValueError("loaded weights differ from the failing baseline")
                if device == "cuda":
                    torch.cuda.synchronize()
                cases.append({"ratio": ratio, "role": "failing endpoint" if ratio == "1:15" else "passing control",
                              **input_identity, **result, "processing_seconds": time.perf_counter() - started,
                              "baseline_fp32_logits": baseline_case["fp32_backend"]["logits"],
                              "baseline_pass_flag_reproduced": result["endpoint_passed"] == baseline_case["fp32_backend"]["logits"]["passed"],
                              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
            finally:
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        for checkpoint in checkpoint_identity:
            if file_sha256(checkpoints[checkpoint["ratio"]].best_path) != checkpoint["sha256"]:
                raise RuntimeError("historical checkpoint bytes changed during isolation")
    protocol = {"version": 1, "device": device, "ratios": list(RATIOS), "seed": 2027, "length": 257,
                "prefix_length": 128, "chunk_size": 128, "coordinate_limit": coordinate_limit,
                "input": input_identity, "checkpoints": checkpoint_identity, "data": data_identity,
                "baseline_declaration": {**public_reference(declaration), "sha256": file_sha256(declaration)},
                "baseline_report": {**public_reference(baseline_report), "sha256": file_sha256(baseline_report),
                                    "protocol_sha256": baseline["protocol_sha256"]},
                "isolation_declaration": planned, "tolerances": {key: dict(value) for key, value in TOLERANCES.items()},
                "runtime": runtime, "runtime_sha256": canonical_sha256(runtime), "source_sha256": sources,
                "production_defaults_changed": False, "optimizer_executed": False}
    return {"schema": 1, "kind": "trained_checkpoint_operation_isolation", "certified": False,
            "status": "completed" if all(case["all_trace_checks_passed"] and case["all_frozen_checks_passed"] for case in cases) else "completed_with_parity_failures",
            "execution_status": "completed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol, "protocol_sha256": canonical_sha256(protocol), "cases": cases,
            "limits": ["Exact checkpoint-selection validation window; no independent quality evidence or statistical replication.",
                       "First nonzero drift and first fixed-tolerance violation are different observations.",
                       "Identical frozen reference operands isolate scan arithmetic; deeper model inputs can already contain upstream drift.",
                       "Real prefix probes use actual reference-model prefill memory and convolution context; no synthetic random memory is substituted.",
                       "FP64 recurrence and decay contrasts are exploratory anchors/treatments, not production policies or a widened tolerance.",
                       "No gradients, optimizer, training, BF16/RoPE attribution, speed comparison or backend certification in this isolation.",
                       "A smaller frozen-scan error suggests a follow-up experiment; it does not certify whole-model improvement."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json")
    parser.add_argument("--baseline-report", type=Path, default=ROOT / "docs/research/checks/trained-numerics-2026-10-04/natural-257-seed-2027.json")
    parser.add_argument("--isolation-declaration", type=Path, default=ROOT / "docs/research/numerical-isolation-protocol-2026-10-04.json")
    parser.add_argument("--coordinate-limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new immutable isolation artifact")
    try:
        report = run_study(args.device, declaration=args.declaration, baseline_report=args.baseline_report,
                           isolation_declaration=args.isolation_declaration, coordinate_limit=args.coordinate_limit)
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        reason = f"{type(exc).__name__}: {exc}".replace(str(ROOT), ".").replace(str(Path.home()), "<home>")
        report = {"schema": 1, "kind": "trained_checkpoint_operation_isolation", "certified": False,
                  "status": "incomplete", "execution_status": "incomplete", "reason": reason}
    write_report(args.output, report)
    print(json.dumps({"status": report["status"], "execution_status": report["execution_status"], "certified": False}))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
