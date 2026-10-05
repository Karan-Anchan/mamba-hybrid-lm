"""Declared forward-only RMS-coefficient × final-head precision diagnostic.

Historical tensors and default model code are immutable. Four temporary cells
keep the previously measured decay treatment fixed. Independent CPU FP64 norm
oracles and frozen inputs diagnose arithmetic without certifying production.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import scripts.study_checkpoint_numerics as base  # noqa: E402
import scripts.study_decay_precision as decay  # noqa: E402
import scripts.check_decay_breadth as breadth  # noqa: E402
import scripts.isolate_tokenwise_numerics as prior  # noqa: E402
from src.model.norm import RMSNorm  # noqa: E402
from src.model.scan_backend import BackendUnavailableError  # noqa: E402

RATIOS, CASES, CASE_IDS = prior.RATIOS, prior.CASES, prior.CASE_IDS
CELLS = ("N0H0", "N1H0", "N0H1", "N1H1")
PAIRINGS = ("full_vs_original", "full_vs_decay_baseline", "tokenwise_vs_own_full",
            "tokenwise_vs_original", "tokenwise_vs_decay_baseline")
DEFAULT_PRIOR_DECLARATION = ROOT / "docs/research/tokenwise-isolation-protocol-2026-10-04.json"
DEFAULT_PRIOR_REPORT = prior.DEFAULT_OUTPUT
DEFAULT_DECLARATION = ROOT / "docs/research/normalization-head-protocol-2026-10-05.json"
DEFAULT_OUTPUT = ROOT / "docs/research/checks/normalization-head-2026-10-05/natural-matched-seed-2027.json"
NORM_OUTPUT_CHECKS = ("original_full_vs_observed", "original_tokenwise_vs_full", "original_full_vs_fp64",
    "original_tokenwise_vs_fp64", "n1_full_vs_original_full", "n1_tokenwise_vs_n1_full", "n1_full_vs_fp64", "n1_tokenwise_vs_fp64")
NORM_STAT_CHECKS = ("squared_fp32_vs_fp64", "mean_square_original_full_vs_fp64", "mean_square_original_tokenwise_vs_fp64",
    "mean_square_n1_full_vs_fp64", "mean_square_n1_tokenwise_vs_fp64", "coefficient_original_full_vs_fp64",
    "coefficient_original_tokenwise_vs_fp64", "coefficient_n1_full_vs_fp64", "coefficient_n1_tokenwise_vs_fp64")
HEAD_CHECKS = ("h0_full_vs_observed", "h0_tokenwise_vs_h0_full", "h0_full_vs_fp64", "h0_tokenwise_vs_fp64",
               "h1_full_vs_h0_full", "h1_tokenwise_vs_h1_full", "h1_full_vs_fp64", "h1_tokenwise_vs_fp64")
REPORT_SCHEMA = {"version": 1, "passing_stage_rows": "complete names/shapes/dtypes/tolerances/counts/maxima and first_nonzero_coordinate; redundant numeric samples omitted",
    "failed_stage_rows": "complete detailed comparison including bounded first/worst/violation samples",
    "pair_first_markers": "complete first_nonzero and first_violation numeric samples",
    "coordinate_limit": 8, "serialized_activations": False, "frozen_norm_output_checks": list(NORM_OUTPUT_CHECKS),
    "frozen_norm_stat_checks": list(NORM_STAT_CHECKS), "frozen_head_checks": list(HEAD_CHECKS),
    "target_raw_bytes": 50 * 1024**2}


def norm_sites(model):
    sites = []
    for i, block in enumerate(model.blocks):
        sites.append((f"blocks.{i}.norm1", block.norm1, (i, "block_input"), (i, "norm1_output")))
        if not block.is_attn:
            sites.append((f"blocks.{i}.mixer.norm", block.mixer.norm, (i, "gated_norm_input"), (i, "gated_norm_output")))
        if block.has_mlp:
            sites.append((f"blocks.{i}.norm2", block.norm2, (i, "mixer_residual"), (i, "norm2_output")))
    sites.append(("norm_f", model.norm_f, (len(model.blocks) - 1, "block_output"), (None, "final_norm")))
    if any(not isinstance(module, RMSNorm) for _, module, _, _ in sites):
        raise ValueError("diagnostic requires audited RMSNorm modules")
    return sites


def norm_arithmetic(module, inputs, *, n1):
    """Exact proposed coefficient scope; keep all conversions autograd-connected."""
    if (not isinstance(inputs, torch.Tensor) or inputs.ndim < 1 or inputs.shape[-1] != module.weight.numel()
            or not inputs.is_floating_point() or module.weight.ndim != 1
            or not math.isfinite(module.eps) or module.eps <= 0):
        raise ValueError("valid floating RMS input/weight and positive finite epsilon required")
    dtype = inputs.dtype
    x = inputs.float()
    squared = x.pow(2)
    mean = (squared.double() if n1 else squared).mean(-1, keepdim=True)
    coefficient = torch.rsqrt(mean + module.eps)
    if n1:
        coefficient = coefficient.float()
    output = ((x * coefficient) * module.weight.float()).to(dtype)
    return output, squared, mean, coefficient


def n1_forward(module, inputs):
    return norm_arithmetic(module, inputs, n1=True)[0]


def h1_forward(module, inputs):
    if inputs.dtype != torch.float32 or module.weight.dtype != torch.float32:
        raise ValueError("the declared final-head treatment requires FP32 inputs and weights")
    return F.linear(inputs.double(), module.weight.double(), None if module.bias is None else module.bias.double()).float()


@contextmanager
def temporary_cell(model, cell):
    """Instance-only overrides; nested zero factors select historical formulas."""
    if cell not in CELLS:
        raise ValueError("unknown normalization/head cell")
    saved = []
    try:
        for _, module, _, _ in norm_sites(model):
            saved.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = MethodType(n1_forward if cell[1] == "1" else RMSNorm.forward, module)
        head = model.lm_head
        if not isinstance(head, torch.nn.Linear):
            raise ValueError("head treatment requires the audited linear lm_head")
        saved.append((head, "forward" in head.__dict__, head.__dict__.get("forward")))
        head.forward = MethodType(h1_forward if cell[3] == "1" else torch.nn.Linear.forward, head)
        yield
    finally:
        for module, existed, previous in reversed(saved):
            if existed:
                module.forward = previous
            else:
                module.__dict__.pop("forward", None)


def layout(value):
    return {"shape": list(value.shape), "stride": list(value.stride()), "storage_offset": value.storage_offset(), "dtype": str(value.dtype)}


def restore_layout(value, observed, device):
    if observed["shape"] != list(value.shape) or observed["dtype"] != str(value.dtype):
        raise ValueError("captured replay shape/dtype differs")
    stride, offset = observed["stride"], observed["storage_offset"]
    if len(stride) != value.ndim or any(type(item) is not int or item < 0 for item in [*stride, offset]):
        raise ValueError("captured replay layout is invalid")
    size = offset + 1 + sum((dim - 1) * step for dim, step in zip(value.shape, stride))
    storage = torch.empty(size, device=device, dtype=value.dtype)
    replay = storage.as_strided(value.shape, stride, offset)
    replay.copy_(value.to(device))
    return replay


@contextmanager
def capture_norm_layouts(model, observations):
    hooks = []
    try:
        for name, module, input_key, output_key in norm_sites(model):
            def pre(_module, inputs, name=name, input_key=input_key, output_key=output_key):
                if name in observations:
                    raise RuntimeError("duplicate full-route normalization observation")
                observations[name] = {"input_layout": layout(inputs[0]), "source_input_stage": list(input_key),
                                      "source_output_stage": list(output_key)}
            def post(_module, _inputs, output, name=name):
                observations[name]["output_layout"] = layout(output)
            hooks.append(module.register_forward_pre_hook(pre))
            hooks.append(module.register_forward_hook(post))
        def head_pre(_module, inputs):
            observations["lm_head"] = {"input_layout": layout(inputs[0]), "source_input_stage": [None, "lm_head.input"],
                                        "source_output_stage": [None, "lm_head.output"]}
        def head_post(_module, _inputs, output):
            observations["lm_head"]["output_layout"] = layout(output)
        hooks.append(model.lm_head.register_forward_pre_hook(head_pre))
        hooks.append(model.lm_head.register_forward_hook(head_post))
        yield
    finally:
        for hook in hooks:
            hook.remove()


def norm_oracle_fp64(inputs, weight, epsilon):
    """Independent NumPy FP64 feature reduction, never a Torch RMSNorm replay."""
    if (inputs.ndim != 3 or weight.ndim != 1 or inputs.shape[-1] != weight.numel()
            or min(inputs.shape) <= 0 or type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("oracle requires B,L,D inputs, D weights and positive finite epsilon")
    base.require_finite(inputs, tuple(inputs.shape), "norm oracle input")
    base.require_finite(weight, tuple(weight.shape), "norm oracle weight")
    x, w = inputs.detach().cpu().double().numpy(), weight.detach().cpu().double().numpy()
    squared = np.square(x)
    mean = np.sum(squared, axis=-1, keepdims=True, dtype=np.float64) / x.shape[-1]
    coefficient = np.reciprocal(np.sqrt(mean + epsilon))
    outputs = (x * coefficient) * w
    return tuple(torch.from_numpy(np.array(value, copy=True)) for value in (outputs, squared, mean, coefficient))


def head_oracle_fp64(inputs, weight, bias=None):
    """Detached CPU NumPy matrix product, independent of the H1 Torch operation."""
    if (inputs.ndim != 3 or weight.ndim != 2 or min(inputs.shape) <= 0 or min(weight.shape) <= 0
            or inputs.shape[-1] != weight.shape[-1] or (bias is not None and bias.shape != (weight.shape[0],))):
        raise ValueError("head oracle requires B,L,D inputs, V,D weight and optional V bias")
    for name, value in (("input", inputs), ("weight", weight), ("bias", bias)):
        if value is not None:
            base.require_finite(value, tuple(value.shape), "head oracle " + name)
    x, w = inputs.detach().cpu().double().numpy(), weight.detach().cpu().double().numpy()
    output = np.matmul(x, w.T)
    if bias is not None:
        output = output + bias.detach().cpu().double().numpy()
    result = torch.from_numpy(np.array(output, copy=True))
    base.require_finite(result, (*inputs.shape[:2], weight.shape[0]), "head FP64 oracle output")
    return result


def cmp(actual, expected, *, oracle=False):
    return prior.compare(actual, expected, coordinate_limit=8, oracle=oracle)


@torch.no_grad()
def frozen_norm_probe(inputs, observed_output, module, observation, check_limit):
    inputs = restore_layout(inputs, observation["input_layout"], module.weight.device)
    before = inputs.clone()
    check_limit()
    original = tuple(value.detach().cpu() for value in norm_arithmetic(module, inputs, n1=False))
    n1 = tuple(value.detach().cpu() for value in norm_arithmetic(module, inputs, n1=True))
    replay = {}
    for name, factor in (("original", False), ("n1", True)):
        parts = [[] for _ in range(4)]
        for i in range(inputs.shape[1]):
            check_limit()
            for bucket, value in zip(parts, norm_arithmetic(module, inputs[:, i:i + 1], n1=factor)):
                bucket.append(value)
        replay[name] = tuple(torch.cat(bucket, 1).detach().cpu() for bucket in parts)
    check_limit()
    oracle = norm_oracle_fp64(inputs, module.weight, module.eps)
    output_checks = dict(zip(NORM_OUTPUT_CHECKS, (
        cmp(original[0], observed_output), cmp(replay["original"][0], original[0]),
        cmp(original[0], oracle[0], oracle=True), cmp(replay["original"][0], oracle[0], oracle=True),
        cmp(n1[0], original[0]), cmp(replay["n1"][0], n1[0]),
        cmp(n1[0], oracle[0], oracle=True), cmp(replay["n1"][0], oracle[0], oracle=True))))
    stat_checks = dict(zip(NORM_STAT_CHECKS, (
        cmp(original[1], oracle[1], oracle=True), cmp(original[2], oracle[2], oracle=True),
        cmp(replay["original"][2], oracle[2], oracle=True), cmp(n1[2], oracle[2], oracle=True),
        cmp(replay["n1"][2], oracle[2], oracle=True), cmp(original[3], oracle[3], oracle=True),
        cmp(replay["original"][3], oracle[3], oracle=True), cmp(n1[3], oracle[3], oracle=True),
        cmp(replay["n1"][3], oracle[3], oracle=True))))
    if not torch.equal(inputs, before):
        raise RuntimeError("frozen norm replay changed its input")
    return {**observation, "input": prior.tensor_identity(inputs), "observed_output": prior.tensor_identity(observed_output),
            "weight": prior.tensor_identity(module.weight), "epsilon": module.eps, "replay_layout": layout(inputs),
            "tokenwise_shapes": [inputs.shape[0], 1, inputs.shape[2]], "identical_operands": True,
            "input_unchanged": True, "oracle": "independent detached CPU NumPy FP64 feature sum / reciprocal sqrt / weight multiplication",
            "output_checks": output_checks, "statistic_checks": stat_checks,
            "statistics": {name: {key: prior.tensor_identity(value) for key, value in zip(("squared", "mean_square", "coefficient"), values[1:])}
                           for name, values in (("original_full", original), ("original_tokenwise", replay["original"]),
                                                ("n1_full", n1), ("n1_tokenwise", replay["n1"]), ("fp64_oracle", oracle))},
            "passed": all(check["passed"] for check in [*output_checks.values(), *stat_checks.values()]), "exploratory": True}


@torch.no_grad()
def frozen_head_probe(inputs, observed_output, module, observation, check_limit):
    inputs = restore_layout(inputs, observation["input_layout"], module.weight.device)
    before = inputs.clone()
    if inputs.dtype != torch.float32 or module.weight.dtype != torch.float32:
        raise ValueError("frozen H0/H1 probes require declared FP32 operands")
    check_limit()
    h0 = F.linear(inputs, module.weight, module.bias).detach().cpu()
    h1 = h1_forward(module, inputs).detach().cpu()
    tokenwise = {}
    # Convert the frozen weight once; H1's mathematical operands stay identical.
    weight64 = module.weight.double()
    bias64 = None if module.bias is None else module.bias.double()
    for name in ("h0", "h1"):
        parts = []
        for i in range(inputs.shape[1]):
            check_limit()
            current = inputs[:, i:i + 1]
            value = F.linear(current, module.weight, module.bias) if name == "h0" else F.linear(current.double(), weight64, bias64).float()
            parts.append(value)
        tokenwise[name] = torch.cat(parts, 1).detach().cpu()
        del parts
    check_limit()
    oracle = head_oracle_fp64(inputs, module.weight, module.bias)
    check_limit()
    checks = dict(zip(HEAD_CHECKS, (cmp(h0, observed_output), cmp(tokenwise["h0"], h0),
        cmp(h0, oracle, oracle=True), cmp(tokenwise["h0"], oracle, oracle=True), cmp(h1, h0),
        cmp(tokenwise["h1"], h1), cmp(h1, oracle, oracle=True), cmp(tokenwise["h1"], oracle, oracle=True))))
    if not torch.equal(inputs, before):
        raise RuntimeError("frozen head replay changed input")
    return {**observation, "input": prior.tensor_identity(inputs), "observed_output": prior.tensor_identity(observed_output),
            "weight": prior.tensor_identity(module.weight), "bias": None if module.bias is None else prior.tensor_identity(module.bias),
            "replay_layout": layout(inputs), "tokenwise_shapes": [inputs.shape[0], 1, inputs.shape[2]],
            "identical_operands": True, "input_unchanged": True,
            "oracle": "independent detached CPU NumPy FP64 matrix multiplication and optional bias addition",
            "oracle_output": prior.tensor_identity(oracle), "checks": checks,
            "outputs": {name: prior.tensor_identity(value) for name, value in (("h0_full", h0), ("h0_tokenwise", tokenwise["h0"]),
                        ("h1_full", h1), ("h1_tokenwise", tokenwise["h1"]))},
            "passed": all(item["passed"] for item in checks.values()), "exploratory": True}


def runtime_policy():
    """Observe execution factors; only the separate TF32 scope changes flags."""
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def tied_weight_snapshot(model):
    head, embed = model.lm_head.weight, model.embed.weight
    return {"head_id": id(head), "embedding_id": id(embed), "head_ptr": head.data_ptr(), "embedding_ptr": embed.data_ptr(),
            "same_parameter": head is embed, "same_storage": head.data_ptr() == embed.data_ptr()}


def tied_weight_report(before, after):
    return {"before": {key: before[key] for key in ("same_parameter", "same_storage")},
            "after": {key: after[key] for key in ("same_parameter", "same_storage")},
            "parameter_identity_unchanged": all(before[key] == after[key] for key in ("head_id", "embedding_id")),
            "data_ptr_unchanged": all(before[key] == after[key] for key in ("head_ptr", "embedding_ptr")),
            "passed": before == after}


def compact_trace(actual, expected, check_limit):
    detailed = prior.stage_comparisons(actual, expected, 8, check_limit)
    rows = []
    samples = {"first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples"}
    for row in detailed["stages"]:
        if not row["passed"]:
            rows.append({**row, "sample_policy": "full_failed"})
        else:
            rows.append({**{key: value for key, value in row.items() if key not in samples}, "sample_policy": "passing_compact",
                "first_nonzero_coordinate": None if row["first_nonzero"] is None else row["first_nonzero"]["coordinate"]})
    return {**detailed, "stages": rows}


def flags(value):
    return prior.comparison_flags(value)


def effects(actual, expected, targets, position=0):
    values = base.token_effects(actual, expected, targets, position)
    return {**values, "batch_size": targets.shape[0], "positions_per_row": targets.shape[1],
            "flatten_order": "batch-major; position span describes each row", "targets_sha256": base.tensor_sha256(targets)}


@torch.no_grad()
def stateful_full(model, x, recorder, stage, check_limit):
    check_limit()
    model.configure_scan_backend("reference", 128)
    recorder["stage"] = stage
    state = model.init_inference_state(x.shape[0], device=x.device, cache_dtype=torch.float32)
    with base.observe_scans(model, recorder):
        logits, _ = model(x, inference_state=state)
    base.require_finite(logits, (*x.shape, model.cfg.vocab_size), stage)
    if state.position != x.shape[1]:
        raise RuntimeError("stateful full final position differs")
    return logits.detach().cpu(), base.state_snapshot(state)


@torch.no_grad()
def continuation_probe(model, x, y, prefix_length, anchors, memory_anchors, recorder, cell, check_limit):
    model.configure_scan_backend("reference", 128)
    recorder["stage"] = cell + ".prefix_prefill"
    check_limit()
    with base.observe_scans(model, recorder):
        prefix_logits, prefix = model.prefill(x[:, :prefix_length], cache_dtype=torch.float32)
    before = base.state_snapshot(prefix)
    prefix_checks = {name: cmp(prefix_logits, value[:, prefix_length - 1:prefix_length]) for name, value in anchors.items()}
    suffix = x[:, prefix_length:]
    logits, memories, positions = {}, {}, {}
    for name, size in (("one_shot", suffix.shape[1]), ("chunked128", 128), ("tokenwise", 1)):
        state, parts = prefix.clone(), []
        recorder["stage"] = cell + ".suffix_" + name
        with base.observe_scans(model, recorder):
            for start in range(0, suffix.shape[1], size):
                check_limit()
                output, _ = model(suffix[:, start:start + size], inference_state=state)
                parts.append(output.detach().cpu())
        logits[name] = torch.cat(parts, 1)
        base.require_finite(logits[name], (x.shape[0], suffix.shape[1], model.cfg.vocab_size), "suffix " + name)
        memories[name], positions[name] = base.state_snapshot(state), state.position
    after = base.state_snapshot(prefix)
    if prefix.position != prefix_length or any(not torch.equal(before[name], value) for name, value in after.items()) or any(p != x.shape[1] for p in positions.values()):
        raise RuntimeError("cloned continuation changed prefix or final positions")
    comparisons = {name: {anchor: cmp(value, reference[:, prefix_length:]) for anchor, reference in anchors.items()} for name, value in logits.items()}
    route_checks = {name + "_vs_one_shot": cmp(logits[name], logits["one_shot"]) for name in ("chunked128", "tokenwise")}
    retained = {name: {anchor: prior.state_comparison(value, reference, 8) for anchor, reference in memory_anchors.items()} for name, value in memories.items()}
    probability = {name: effects(value, anchors["original"][:, prefix_length:], y[:, prefix_length:], prefix_length) for name, value in logits.items()}
    report = {"prefix_position": prefix_length, "suffix_length": suffix.shape[1], "synthetic": False, "prefix_unchanged": True,
        "prefix_fields": breadth.state_identity(before), "prefix_logits": prior.tensor_identity(prefix_logits), "prefix_comparisons": prefix_checks,
        "suffix_logits": {name: prior.tensor_identity(value) for name, value in logits.items()}, "suffix_comparisons": comparisons,
        "route_comparisons": route_checks, "retained_state": retained, "final_positions": positions,
        "suffix_state_fields": {name: breadth.state_identity(value) for name, value in memories.items()},
        "chunked_schedule": [min(128, suffix.shape[1] - start) for start in range(0, suffix.shape[1], 128)],
        "next_token_effects_vs_original": probability}
    report["passed"] = all(flags(report))
    return report


def observed_events(recorder):
    return [{**json.loads(key), "calls": calls} for key, calls in sorted(recorder["events"].items())]


def prior_reproduction(current, previous):
    """Exact repeated N0H0 control, distinct from unchanged tolerance gates."""
    mapping = {"full_vs_original": "candidate_full_vs_original_stateless",
               "tokenwise_vs_own_full": "candidate_tokenwise_vs_candidate_full",
               "tokenwise_vs_original": "candidate_tokenwise_vs_original_stateless",
               "stateful_full_vs_own_full": "candidate_stateful_full_vs_candidate_stateless"}
    if previous is None:
        return {"executed": False, "reason": "standalone CPU probe has no historical CUDA case"}
    fields = ("shape", "passed", "tolerance", "nonzero_error_count", "violation_count", "max_absolute_error",
              "max_tolerance_ratio", "first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples")
    checks = {name: all(current[name][field] == previous["whole_model_comparisons"][old_name][field] for field in fields)
              for name, old_name in mapping.items()}
    return {"executed": True, "exact_fields": list(fields), "comparisons": checks, "exact_equal": all(checks.values()),
            "scope": "N0H0 repeated historical full/tokenwise/stateful endpoint statistics; separate from tolerance acceptance"}


@torch.no_grad()
def study_case(model, x, y, *, prefix_length, check_limit=lambda: None, progress=None, prior_case=None):
    if (x.ndim != 2 or x.shape != y.shape or x.dtype != torch.int64 or y.dtype != torch.int64 or min(x.shape) <= 0
            or type(prefix_length) is not int or not 0 < prefix_length < x.shape[1] or not torch.equal(x[:, 1:], y[:, :-1])):
        raise ValueError("paired shifted tokens and substantive prefix required")
    weights = base.weight_sha256(model)
    tie_before = tied_weight_snapshot(model)
    policy_before = runtime_policy()
    result = {"weights_sha256": weights, "execution_status": "incomplete", "cells": [], "failed_cell": None,
              "frozen_norms": [], "frozen_head": None, "backward_executed": False, "BF16_executed": False,
              "production_defaults_changed": False, "optimizer_executed": False}
    recorder = {"stage": "unexecuted", "events": {}}
    with base.preserve_model_execution(model), base.tf32_disabled(), torch.autocast(x.device.type, enabled=False):
        policy_active = runtime_policy()
        try:
            if progress is not None:
                progress.update(cell=None, stage="original anchors")
            original = prior.collect_route(model, x, y, check_limit=check_limit)
            check_limit()
            repeated, repeated_loss = model(x, y)
            base.require_finite(repeated_loss, (), "original loss")
            original_repeat = {"exact_equal": torch.equal(original["logits"], repeated.detach().cpu()), "comparison": cmp(repeated, original["logits"])}
            original_stateful, original_memory = stateful_full(model, x, recorder, "original.stateful_full", check_limit)
            result["original_anchor"] = {"logits": prior.tensor_identity(original["logits"]), "loss": prior.tensor_identity(repeated_loss),
                "stateful_logits": prior.tensor_identity(original_stateful), "state_fields": breadth.state_identity(original_memory),
                "stateful_vs_stateless": cmp(original_stateful, original["logits"]), "repeat": original_repeat,
                "actual_paths": {"scan": original["scan_events"], "attention": original["attention_events"]}}
            baseline = baseline_memory = None
            with decay.temporary_decay_treatment():
                for cell in CELLS:
                    result["failed_cell"] = cell
                    if progress is not None:
                        progress.update(cell=cell, stage="full/tokenwise/stateful/continuation")
                    with temporary_cell(model, cell):
                        observations = {}
                        if cell == "N0H0":
                            with capture_norm_layouts(model, observations):
                                full = prior.collect_route(model, x, y, check_limit=check_limit)
                            baseline = full
                        else:
                            full = prior.collect_route(model, x, y, check_limit=check_limit)
                        token = prior.collect_route(model, x, y, tokenwise=True, check_limit=check_limit)
                        own_stateful, own_memory = stateful_full(model, x, recorder, cell + ".stateful_full", check_limit)
                        if cell == "N0H0":
                            baseline_memory = own_memory
                            result["decay_baseline_anchor"] = {"logits": prior.tensor_identity(full["logits"]), "state_fields": breadth.state_identity(own_memory), "cell": cell}
                        check_limit()
                        model.configure_scan_backend("torch_chunked", 128)
                        recorder["stage"] = cell + ".full_chunked128"
                        with base.observe_scans(model, recorder):
                            chunked, chunked_loss = model(x, y)
                        base.require_finite(chunked, (*x.shape, model.cfg.vocab_size), "chunked logits")
                        base.require_finite(chunked_loss, (), "chunked loss")
                        model.configure_scan_backend("reference", 128)
                        endpoints = dict(zip(PAIRINGS, (cmp(full["logits"], original["logits"]), cmp(full["logits"], baseline["logits"]),
                            cmp(token["logits"], full["logits"]), cmp(token["logits"], original["logits"]), cmp(token["logits"], baseline["logits"]))))
                        endpoints.update(stateful_full_vs_own_full=cmp(own_stateful, full["logits"]),
                            stateful_full_vs_original=cmp(own_stateful, original["logits"]),
                            stateful_full_vs_decay_baseline=cmp(own_stateful, baseline["logits"]),
                            full_chunked_vs_own_full=cmp(chunked, full["logits"]), full_chunked_vs_original=cmp(chunked, original["logits"]),
                            full_chunked_vs_decay_baseline=cmp(chunked, baseline["logits"]))
                        if cell == "N0H0":
                            result["prior_reproduction"] = prior_reproduction(endpoints, prior_case)
                        traces = {name: compact_trace(actual, anchor, check_limit) for name, actual, anchor in (
                            ("full_vs_original", full, original), ("full_vs_decay_baseline", full, baseline),
                            ("tokenwise_vs_own_full", token, full), ("tokenwise_vs_original", token, original),
                            ("tokenwise_vs_decay_baseline", token, baseline))}
                        memory_anchors = {"own": own_memory, "original": original_memory, "decay_baseline": baseline_memory}
                        retained = {anchor: prior.state_comparison(token["state"], memory, 8) for anchor, memory in memory_anchors.items()}
                        stateful_retained = {anchor: prior.state_comparison(own_memory, memory, 8)
                                             for anchor, memory in memory_anchors.items() if anchor != "own"}
                        continuation = continuation_probe(model, x, y, prefix_length,
                            {"own": full["logits"], "original": original["logits"], "decay_baseline": baseline["logits"]},
                            memory_anchors, recorder, cell, check_limit)
                        report = {"cell_id": cell, "normalization_fp64_coefficients": cell[1] == "1", "head_fp64_matmul": cell[3] == "1",
                            "execution_status": "completed", "whole_model_comparisons": endpoints, "stage_comparisons": traces,
                            "retained_state": retained, "stateful_full_retained_state": stateful_retained,
                            "actual_prefix": continuation, "final_positions": {"tokenwise": token["position"], "stateful_full": x.shape[1]},
                            "logits": {"full": prior.tensor_identity(full["logits"]), "tokenwise": prior.tensor_identity(token["logits"]), "chunked128": prior.tensor_identity(chunked)},
                            "next_token_effects": {"full_vs_original": effects(full["logits"], original["logits"], y),
                                "tokenwise_vs_original": effects(token["logits"], original["logits"], y), "tokenwise_vs_own_full": effects(token["logits"], full["logits"], y)},
                            "actual_paths": {"full": {"scan": full["scan_events"], "attention": full["attention_events"]},
                                             "tokenwise": {"scan": token["scan_events"], "attention": token["attention_events"]}},
                            "backward_executed": False, "BF16_executed": False}
                        report["whole_model_passed"] = (all(check["passed"] for check in endpoints.values())
                            and all(check["passed"] for check in [*retained.values(), *stateful_retained.values()]) and continuation["passed"])
                        report["all_recorded_comparisons_passed"] = all(flags(report))
                        result["cells"].append(report)
                        # Freeze N0H0 operands once; later cells keep this common input anchor.
                        if cell == "N0H0":
                            if progress is not None:
                                progress.update(stage="frozen normalization/head")
                            for name, module, input_key, output_key in norm_sites(model):
                                check_limit()
                                probe = frozen_norm_probe(full["tensors"][input_key], full["tensors"][output_key], module, observations[name], check_limit)
                                result["frozen_norms"].append({"module": name, **probe})
                            result["frozen_head"] = frozen_head_probe(full["tensors"][(None, "lm_head.input")], full["logits"],
                                model.lm_head, observations["lm_head"], check_limit)
                        del token, own_stateful, own_memory, chunked, chunked_loss
                        if cell != "N0H0":
                            del full
            if len(result["frozen_norms"]) != len(norm_sites(model)) or result["frozen_head"] is None:
                raise RuntimeError("required frozen norm/head inventory incomplete")
            result["execution_status"], result["failed_cell"] = "completed", None
        except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError, MemoryError) as exc:
            result["reason"] = clean_reason(exc)
    policy_restored = runtime_policy()
    result["runtime_policy"] = {"before": policy_before, "active": policy_active, "restored": policy_restored,
                               "restoration_exact": policy_before == policy_restored}
    result["tied_weight_integrity"] = tied_weight_report(tie_before, tied_weight_snapshot(model))
    result["post_model_weights_unchanged"] = base.weight_sha256(model) == weights
    safeguards = []
    if not result["post_model_weights_unchanged"]:
        safeguards.append("model weight bytes changed")
    if not result["tied_weight_integrity"]["passed"]:
        safeguards.append("head/embedding parameter identity or data pointer changed")
    if not result["runtime_policy"]["restoration_exact"]:
        safeguards.append("runtime policy restoration failed")
    if safeguards:
        result["execution_status"] = "incomplete"
        result["reason"] = (result.get("reason", "") + "; " if result.get("reason") else "") + "; ".join(safeguards)
    result["operator_observations"] = observed_events(recorder)
    result["attribution_control_stable"] = (result.get("original_anchor", {}).get("repeat", {}).get("exact_equal", False)
        and result.get("prior_reproduction", {}).get("exact_equal", prior_case is None))
    result["all_recorded_comparisons_passed"] = result["execution_status"] == "completed" and result["attribution_control_stable"] and all(flags(result))
    return result


def clean_reason(exc):
    return f"{type(exc).__name__}: {exc}".replace(str(ROOT), ".").replace(str(Path.home()), "<home>")


def source_fingerprints():
    return {**prior.source_fingerprints(), "scripts/study_normalization_head_precision.py": base.file_sha256(Path(__file__))}


def declaration_template(*, device="cuda", prior_declaration=DEFAULT_PRIOR_DECLARATION, prior_report=DEFAULT_PRIOR_REPORT,
                         output=DEFAULT_OUTPUT):
    old = base.read_json(prior_report)["protocol"]
    observed = {key: value for key, value in runtime_policy().items() if "allow_tf32" not in key}
    return {"schema": 1, "date": "2026-10-05", "status_at_declaration": "planned_before_execution", "device": device,
            "seed": 2027, "ratios": list(RATIOS), "cases": list(CASES), "shared_inputs": old["shared_inputs"], "cells": list(CELLS),
            "decay_treatment": decay.TREATMENT, "normalization_treatment": "fp64_rms_reduction_coefficients",
            "normalization_scope": "all RMSNorms; FP32 input/square, FP64 mean+epsilon+rsqrt, FP32 coefficient then original FP32 input/weight products",
            "head_treatment": "fp64_final_head_matmul", "head_scope": "instance lm_head only; FP64 input/weight matrix calculation then FP32 output; tied parameter unchanged",
            "chunk_size": 128, "tolerances": base.TOLERANCES, "report_schema": REPORT_SCHEMA,
            "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
            "observed_inference_policy": observed,
            "historical_policy_coverage": "prior records deterministic_algorithms; other factors recorded before this execution, not retrospectively certified",
            "time_budget_seconds": 900, "minimum_large_case_free_bytes": breadth.MINIMUM_FREE_BYTES,
            "backward_executed": False, "BF16_executed": False, "production_defaults_changed": False, "optimizer_executed": False,
            "prior_declaration": {**base.public_reference(prior_declaration), "sha256": base.file_sha256(prior_declaration)},
            "prior_report": {**base.public_reference(prior_report), "sha256": base.file_sha256(prior_report)},
            "output": base.public_reference(output)["path"], "source_sha256": source_fingerprints(),
            "data": old["data"], "checkpoints": old["checkpoints"]}


def audit_declaration(path, *, device, prior_declaration, prior_report, output):
    planned = base.read_json(path)
    expected = declaration_template(device=device, prior_declaration=prior_declaration, prior_report=prior_report, output=output)
    if any(planned.get(key) != value for key, value in expected.items()):
        raise ValueError("declaration differs from exact normalization/head scope, output or identities")
    old, old_plan = base.read_json(prior_report), base.read_json(prior_declaration)
    if (old.get("kind") != "trained_checkpoint_tokenwise_isolation" or old.get("execution_status") != "completed"
            or old.get("certified") is not False or old["protocol_sha256"] != base.canonical_sha256(old["protocol"])
            or old["protocol"]["declaration"]["sha256"] != base.file_sha256(prior_declaration)
            or old["protocol"]["shared_inputs"] != old_plan["shared_inputs"] or old["protocol"]["tolerances"] != base.TOLERANCES
            or set(old["protocol"]["source_sha256"]) != set(prior.source_fingerprints())):
        raise ValueError("prior immutable tokenwise protocol/source scope differs")
    if planned["observed_inference_policy"]["deterministic_algorithms"] != old["protocol"]["runtime"]["precision_flags"]["deterministic_algorithms"]:
        raise ValueError("deterministic policy differs from the measured inference control")
    expected_cases = {(ratio, case_id) for ratio in RATIOS for case_id in CASE_IDS}
    recorded_cases = [(row["ratio"], row["case_id"]) for row in old["cases"]]
    if len(recorded_cases) != len(expected_cases) or set(recorded_cases) != expected_cases:
        raise ValueError("prior six-case coverage differs")
    for name, recorded in old["protocol"]["source_sha256"].items():
        normalized = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if recorded not in {hashlib.sha256(normalized).hexdigest(), hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest()}:
            raise ValueError("prior measured source fingerprint differs: " + name)
    return planned, old


def audit_final_identities(groups, check_limit):
    """Budgeted actual-byte audit; unexecuted groups remain null after a failure."""
    integrity = {name: None for name, _ in groups}
    files, reason = [], None
    for name, entries in groups:
        try:
            check_limit()
            matches = []
            for path, digest in entries:
                check_limit()
                actual = base.file_sha256(path)
                matches.append(actual == digest)
                files.append({"group": name, **base.public_reference(path), "expected_sha256": digest,
                              "actual_sha256": actual, "passed": actual == digest})
                check_limit()
            integrity[name] = all(matches)
        except (OSError, RuntimeError, MemoryError) as exc:
            reason = "final audit: " + clean_reason(exc)
            break
    if any(value is False for value in integrity.values()):
        reason = (reason + "; " if reason else "") + "post-execution identity audit failed"
    return integrity, files, reason


def run_study(device="cuda", *, declaration=DEFAULT_DECLARATION, output=DEFAULT_OUTPUT,
              prior_declaration=DEFAULT_PRIOR_DECLARATION, prior_report=DEFAULT_PRIOR_REPORT,
              checkpoint_root=Path("checkpoints"), training_run_id="week3-700m-v1",
              data_dir=Path("data/openwebtext-5b"), tokenizer=Path("data/tokenizer/openwebtext.json")):
    limit = breadth.DiagnosticLimit(900)
    if device not in ("cpu", "cuda") or Path(output).exists():
        raise ValueError("device must be cpu/cuda and declared output must not exist")
    limit.check()
    planned, previous = audit_declaration(declaration, device=device, prior_declaration=prior_declaration, prior_report=prior_report, output=output)
    limit.check()
    checkpoints, data_identity, checkpoint_identity = base.audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, RATIOS)
    limit.check()
    prepared = prior.prepare_inputs(base.load_split(data_dir, "val"))
    if ([row[2] for row in prepared] != planned["shared_inputs"] or data_identity != planned["data"] or checkpoint_identity != planned["checkpoints"]):
        raise ValueError("actual data/checkpoint/input identities differ from declaration")
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA unavailable; no fallback")
    sources = source_fingerprints()
    if sources != planned["source_sha256"]:
        raise ValueError("source bytes changed during declaration/input preflight")
    evidence = {Path(path): base.file_sha256(path) for path in (declaration, prior_declaration, prior_report)}
    rows, headroom, reason, active = [], [], None, None
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    policy_before = runtime_policy()
    with torch.random.fork_rng(devices=devices), base.tf32_disabled():
        runtime = base.runtime_metadata(device)
        policy_active = runtime_policy()
        runtime["inference_policy"] = policy_active
        try:
            for ratio in RATIOS:
                active = {"ratio": ratio, "case_id": None, "cell": None, "stage": "load model"}
                limit.check()
                model = base.load_variant_model(checkpoints[ratio], torch.device(device))
                try:
                    if base.weight_sha256(model) != next(row["weights_sha256"] for row in previous["cases"] if row["ratio"] == ratio):
                        raise ValueError("loaded weights differ from immutable prior evidence")
                    for x, y, identity in prepared:
                        active = {"ratio": ratio, "case_id": identity["case_id"], "cell": None, "stage": "case preflight"}
                        limit.check()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            if identity["batch_size"] == 2:
                                free, total = torch.cuda.mem_get_info()
                                headroom.append({"ratio": ratio, "case_id": identity["case_id"], "free_bytes": free, "total_bytes": total,
                                                 "minimum_free_bytes": breadth.MINIMUM_FREE_BYTES})
                                if free < breadth.MINIMUM_FREE_BYTES:
                                    raise RuntimeError("insufficient declared 8GiB large-case headroom; no fallback")
                        started = time.perf_counter()
                        prior_case = next(row for row in previous["cases"] if row["ratio"] == ratio and row["case_id"] == identity["case_id"])
                        row = study_case(model, x.to(device), y.to(device), prefix_length=identity["prefix_length"],
                                         check_limit=limit.check, progress=active, prior_case=prior_case)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        rows.append({"ratio": ratio, **identity, **row, "processing_seconds": time.perf_counter() - started,
                                     "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
                        print(json.dumps({"progress": "case_finished", "ratio": ratio, "case_id": identity["case_id"],
                                          "execution_status": row["execution_status"], "completed_cells": len(row["cells"]),
                                          "processing_seconds": rows[-1]["processing_seconds"]}), flush=True)
                        if row["execution_status"] != "completed":
                            raise RuntimeError(row.get("reason", "case incomplete"))
                finally:
                    del model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
            active = None
        except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError, MemoryError) as exc:
            reason = clean_reason(exc)
    policy_restored = runtime_policy()
    policy_report = {"before": policy_before, "active": policy_active, "restored": policy_restored,
                     "restoration_exact": policy_before == policy_restored}
    data_files = [(Path(data_dir) / "manifest.json", data_identity["manifest"]["sha256"]), (Path(tokenizer), data_identity["tokenizer"]["sha256"]),
                  *[(Path(data_dir) / Path(item["path"]).name, item["sha256"]) for item in data_identity["artifacts"].values()]]
    groups = (("checkpoints", [(checkpoints[item["ratio"]].best_path, item["sha256"]) for item in checkpoint_identity]),
                          ("training_manifests", [(checkpoints[item["ratio"]].run_dir / "manifest.json", item["training_manifest"]["sha256"]) for item in checkpoint_identity]),
                          ("prepared_data_and_tokenizer", data_files), ("prior_and_declaration", list(evidence.items())),
                          ("sources", [(ROOT / name, digest) for name, digest in sources.items()]))
    integrity, audit_files, audit_reason = audit_final_identities(groups, limit.check)
    if audit_reason:
        reason = (reason + "; " if reason else "") + audit_reason
    if not policy_report["restoration_exact"]:
        reason = (reason + "; " if reason else "") + "runtime policy restoration failed"
    try:
        limit.check()
    except RuntimeError as exc:
        reason = (reason + "; " if reason else "") + str(exc)
    if reason and active is None:
        active = {"ratio": None, "case_id": None, "cell": None, "stage": "final audit / allowance"}
    protocol = {**planned, "declaration": {**base.public_reference(declaration), "sha256": evidence[Path(declaration)]},
                "runtime": runtime, "runtime_sha256": base.canonical_sha256(runtime), "quality_equivalence_margin": None}
    return {"schema": 1, "kind": "trained_checkpoint_normalization_head_precision", "certified": False,
            "execution_status": "incomplete" if reason else "completed", "status": "incomplete" if reason else
                "completed" if all(row["all_recorded_comparisons_passed"] for row in rows) else "completed_with_parity_failures",
            "reason": reason, "failed_case": active if reason else None, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol, "protocol_sha256": base.canonical_sha256(protocol), "cases": rows,
            "large_case_headroom": headroom, "post_execution_integrity": integrity,
            "post_execution_audit_files": audit_files, "runtime_policy": policy_report,
            "execution_allowance": {**limit.metadata(), "scope": "runner entry through preflight/model work/final audits; immutable publication excluded"},
            "limits": ["Six fixed checkpoint-selection-pool cases, four scoped cells; no independent quality conclusion.",
                       "Decays fixed in all cells; normalization is a family-level factor and head changes only lm_head arithmetic.",
                       "Historical stateless scores, original stateful memory, decay-only baseline and each cell's own full route are separate anchors.",
                       "Frozen norm/head inputs are identical; local oracle agreement cannot qualify propagated whole-model outputs.",
                       "FP64 NumPy oracle is independent and exploratory; original FP32 acceptance thresholds remain fixed.",
                       "Passing stage numeric samples are compacted as declared; failed stages and pair-first markers retain bounded samples.",
                       "Real prefix states belong to each cell and are cloned; original prefix is checked unchanged.",
                       "Original gradient_check input identity is retained, but no backward, BF16, optimizer or production change executes.",
                       "The 900-second allowance is cooperative; in-flight GPU/NumPy operations and file hashes can overshoot before the next check. The 8GiB guard does not guarantee peak fit; incomplete routes supply no passes.",
                       "Any successful cell still needs expanded three-ratio/gradient/causality gates and signed numerical/recovery policies."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; use a new immutable declaration")
    if args.declaration.exists() and base.read_json(args.declaration).get("output") != base.public_reference(args.output)["path"]:
        parser.error("output differs from pre-execution declaration")
    try:
        report = run_study(args.device, declaration=args.declaration, output=args.output)
    except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError, MemoryError) as exc:
        report = {"schema": 1, "kind": "trained_checkpoint_normalization_head_precision", "certified": False,
                  "execution_status": "incomplete", "status": "incomplete", "reason": clean_reason(exc), "cases": []}
    base.write_report(args.output, report)
    print(json.dumps({"status": report["status"], "execution_status": report["execution_status"], "completed_cases":
        sum(row.get("execution_status") == "completed" for row in report["cases"]), "completed_cells": sum(len(row.get("cells", [])) for row in report["cases"])}))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
