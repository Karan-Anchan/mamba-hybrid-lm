"""Declared forward-only Mamba input-projection arithmetic isolation.

P0 repeats the measured decay + N1H0 diagnostic. P1 changes only Mamba
in_proj arithmetic to FP64, returning FP32 outputs. Historical sources and
weights stay unchanged. Frozen CPU NumPy references are exploratory, not a
production certificate or a reason to widen the original tolerances.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import scripts.study_normalization_head_precision as norm  # noqa: E402
from src.model.scan_backend import BackendUnavailableError  # noqa: E402

base, decay, breadth, prior = norm.base, norm.decay, norm.breadth, norm.prior
RATIOS, CASES, CASE_IDS = norm.RATIOS, norm.CASES, norm.CASE_IDS
CELLS = ("P0", "P1")
ANCHORS = ("own", "original", "decay_baseline", "projection_baseline")
PAIRINGS = (*norm.PAIRINGS, "full_vs_projection_baseline", "tokenwise_vs_projection_baseline")
ENDPOINTS = (*norm.PAIRINGS, "stateful_full_vs_own_full", "stateful_full_vs_original",
    "stateful_full_vs_decay_baseline", "full_chunked_vs_own_full", "full_chunked_vs_original",
    "full_chunked_vs_decay_baseline", "full_vs_projection_baseline", "tokenwise_vs_projection_baseline",
    "stateful_full_vs_projection_baseline", "full_chunked_vs_projection_baseline")
LOSS_ENDPOINTS = ("full_vs_original", "full_vs_decay_baseline", "full_vs_projection_baseline", "chunked_vs_own_full",
    "chunked_vs_original", "chunked_vs_decay_baseline", "chunked_vs_projection_baseline")
PROJECTION_CHECKS = ("observed_shape_replay_vs_observed", "p0_tokenwise_vs_p0_full", "p0_full_vs_fp64",
    "p0_tokenwise_vs_fp64", "p1_full_vs_p0_full", "p1_tokenwise_vs_p1_full", "p1_full_vs_fp64", "p1_tokenwise_vs_fp64")
DEFAULT_PRIOR_DECLARATION = norm.DEFAULT_DECLARATION
DEFAULT_PRIOR_REPORT = Path(str(norm.DEFAULT_OUTPUT) + ".gz")
DEFAULT_PRIOR_PUBLICATION = ROOT / "docs/research/normalization-head-raw-publication-2026-10-05.json"
DEFAULT_PRIOR_SUMMARY = ROOT / "docs/research/normalization-head-summary-2026-10-05.json"
DEFAULT_DECLARATION = ROOT / "docs/research/projection-precision-protocol-2026-10-06.json"
DEFAULT_OUTPUT = ROOT / "docs/research/checks/projection-precision-2026-10-06/natural-matched-seed-2027.json"
MAX_PRIOR_BYTES = 160 * 1024**2
REPORT_SCHEMA = {**norm.REPORT_SCHEMA, "frozen_norm_output_checks": [], "frozen_norm_stat_checks": [],
    "frozen_head_checks": [], "frozen_projection_checks": list(PROJECTION_CHECKS),
    "stage_pairings": list(PAIRINGS), "score_endpoints": list(ENDPOINTS), "anchors": list(ANCHORS),
    "loss_endpoints": list(LOSS_ENDPOINTS), "loss_policy": "actual model target cross-entropy on each GPU/CPU route; no cross-device recomputation",
    "frozen_full_sites": "all Mamba in_proj; identical P0 full inputs",
    "frozen_prefix_sites": "first and last Mamba in_proj; each cell's actual one_shot/chunked128/tokenwise suffix inputs",
    "prefix_layout_policy": "actual call layouts retained/reconstructed for observed replay; concatenated full/token contrasts include explicit shape/layout change",
    "prior_transport": "registered gzip or original JSON, exact original bytes verified before parsing"}


def projection_sites(model):
    sites = [(f"blocks.{i}.mixer.in_proj", block.mixer.in_proj, i) for i, block in enumerate(model.blocks) if not block.is_attn]
    if not sites or any(not isinstance(module, torch.nn.Linear) for _, module, _ in sites):
        raise ValueError("the diagnostic requires audited Mamba Linear input projections")
    return sites


def projection_arithmetic(module, inputs, *, p1):
    """FP64 multiply/add then FP32 output; preserve parameter connectivity."""
    if (not isinstance(module, torch.nn.Linear) or not isinstance(inputs, torch.Tensor) or inputs.ndim != 3
            or min(inputs.shape) <= 0 or inputs.shape[-1] != module.in_features
            or inputs.dtype != torch.float32 or module.weight.dtype != torch.float32
            or inputs.device != module.weight.device or (module.bias is not None and
                (module.bias.dtype != torch.float32 or module.bias.device != inputs.device))):
        raise ValueError("projection treatment requires nonempty B,L,D FP32 inputs and matching FP32 Linear operands")
    if not p1:
        return F.linear(inputs, module.weight, module.bias)
    return F.linear(inputs.double(), module.weight.double(), None if module.bias is None else module.bias.double()).float()


def p1_forward(module, inputs):
    return projection_arithmetic(module, inputs, p1=True)


@contextmanager
def temporary_projection(model, cell):
    """Instance-only methods; a nested P0 selects the original Linear formula."""
    if cell not in CELLS:
        raise ValueError("unknown projection cell")
    saved = []
    try:
        for _, module, _ in projection_sites(model):
            saved.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = MethodType(p1_forward if cell == "P1" else torch.nn.Linear.forward, module)
        yield
    finally:
        for module, existed, previous in reversed(saved):
            if existed:
                module.forward = previous
            else:
                module.__dict__.pop("forward", None)


@contextmanager
def preserve_rng(device):
    python, numpy = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)


def rng_identity(device):
    numpy = np.random.get_state()
    return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(numpy[0].encode() + numpy[1].tobytes() + repr(numpy[2:]).encode()).hexdigest(),
        "torch_cpu": base.tensor_sha256(torch.get_rng_state()),
        "torch_cuda": base.tensor_sha256(torch.cuda.get_rng_state(device)) if device.type == "cuda" else None}


@torch.no_grad()
def collect_full(model, x, y, check_limit):
    losses = []
    def capture(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
            raise RuntimeError("full route did not produce target loss")
        base.require_finite(output[1], (), "full route target loss")
        losses.append(output[1].detach().cpu())
    handle = model.register_forward_hook(capture)
    try:
        result = prior.collect_route(model, x, y, check_limit=check_limit)
    finally:
        handle.remove()
    if len(losses) != 1:
        raise RuntimeError("full route loss coverage differs")
    return {**result, "loss": losses[0]}


class ProjectionCapture:
    """Observe real call inputs/outputs without per-token CUDA synchronization."""
    def __init__(self, sites, position=0):
        self.sites, self.position, self.parts = sites, position, {}

    @contextmanager
    def scoped(self):
        handles, pending = [], {}
        try:
            for name, module, _ in self.sites:
                def before(_module, values, name=name):
                    if name in pending:
                        raise RuntimeError("nested projection capture is unsupported")
                    value = values[0]
                    pending[name] = (self.position, value.detach().clone(), norm.layout(value))
                def after(_module, _values, output, name=name):
                    position, value, observed_layout = pending.pop(name)
                    self.parts.setdefault(name, []).append((position, value, output.detach().clone(),
                        {"input_layout": observed_layout, "output_layout": norm.layout(output)}))
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def finish(self, length, *, offset=0):
        if set(self.parts) != {name for name, _, _ in self.sites}:
            raise RuntimeError("captured projection site inventory differs")
        result = {}
        for name, module, layer in self.sites:
            end, inputs, outputs, calls = offset, [], [], []
            for start, value, output, observation in self.parts[name]:
                if start != end or value.ndim != 3 or output.shape != (*value.shape[:2], module.out_features):
                    raise RuntimeError("captured projection token coverage/shape differs")
                end += value.shape[1]
                inputs.append(value)
                outputs.append(output)
                calls.append({"position": start, "length": value.shape[1], **observation})
            if end != offset + length:
                raise RuntimeError("captured projection length differs")
            values, observed = torch.cat(inputs, 1).cpu(), torch.cat(outputs, 1).cpu()
            base.require_finite(values, (values.shape[0], length, module.in_features), "captured projection input")
            base.require_finite(observed, (values.shape[0], length, module.out_features), "captured projection output")
            result[name] = {"layer": layer, "inputs": values, "observed": observed, "calls": calls,
                            "offset": offset, "length": length}
        return result


@torch.no_grad()
def frozen_projection_probe(inputs, observed, module, calls, *, active_cell="P0", offset=0, check_limit=lambda: None):
    """Same numeric operands; explicit original-call and shape/layout contrasts."""
    if active_cell not in CELLS:
        raise ValueError("unknown frozen projection policy")
    inputs, observed = inputs.detach().cpu(), observed.detach().cpu()
    base.require_finite(inputs, tuple(inputs.shape), "frozen projection input")
    base.require_finite(observed, (*inputs.shape[:2], module.out_features), "frozen projection output")
    before, weight_before = inputs.clone(), base.tensor_sha256(module.weight)
    bias_before = None if module.bias is None else base.tensor_sha256(module.bias)
    check_limit()
    replay_parts, end = [], offset
    for call in calls:
        check_limit()
        if call["position"] != end or type(call["length"]) is not int or call["length"] <= 0:
            raise ValueError("frozen observed-call positions differ")
        start, end = end - offset, end + call["length"]
        value = norm.restore_layout(inputs[:, start:end - offset], call["input_layout"], module.weight.device)
        replay_parts.append(projection_arithmetic(module, value, p1=active_cell == "P1").cpu())
    if end != offset + inputs.shape[1]:
        raise ValueError("frozen observed-call coverage differs")
    observed_replay = torch.cat(replay_parts, 1)
    # A single observed full call is reconstructed exactly; multiple prefix calls
    # are combined explicitly into a new contiguous exploratory replay layout.
    full = (norm.restore_layout(inputs, calls[0]["input_layout"], module.weight.device)
            if len(calls) == 1 else inputs.to(module.weight.device).contiguous())
    check_limit()
    p0, p1 = (projection_arithmetic(module, full, p1=factor).cpu() for factor in (False, True))
    token = {}
    for name, factor in (("p0", False), ("p1", True)):
        parts = []
        for position in range(full.shape[1]):
            check_limit()
            parts.append(projection_arithmetic(module, full[:, position:position + 1], p1=factor))
        token[name] = torch.cat(parts, 1).cpu()
        del parts
    check_limit()
    oracle = norm.head_oracle_fp64(inputs, module.weight, module.bias)
    check_limit()
    checks = dict(zip(PROJECTION_CHECKS, (norm.cmp(observed_replay, observed), norm.cmp(token["p0"], p0),
        norm.cmp(p0, oracle, oracle=True), norm.cmp(token["p0"], oracle, oracle=True), norm.cmp(p1, p0),
        norm.cmp(token["p1"], p1), norm.cmp(p1, oracle, oracle=True), norm.cmp(token["p1"], oracle, oracle=True))))
    if (not torch.equal(inputs, before) or base.tensor_sha256(module.weight) != weight_before
            or (None if module.bias is None else base.tensor_sha256(module.bias)) != bias_before):
        raise RuntimeError("frozen projection replay mutated operands")
    return {"active_cell": active_cell, "input": prior.tensor_identity(inputs), "observed_output": prior.tensor_identity(observed),
        "weight": prior.tensor_identity(module.weight), "bias": None if module.bias is None else prior.tensor_identity(module.bias),
        "actual_calls": calls, "position_offset": offset, "replay_layout": norm.layout(full),
        "layout_contrast": "captured single-call layout restored" if len(calls) == 1 else "combined contiguous full replay versus actual captured call layouts",
        "tokenwise_shape": [inputs.shape[0], 1, inputs.shape[2]], "identical_operands": True,
        "input_unchanged": True, "oracle": "independent detached CPU NumPy FP64 matrix multiplication and optional bias addition",
        "oracle_output": prior.tensor_identity(oracle), "checks": checks,
        "outputs": {name: prior.tensor_identity(value) for name, value in (("p0_full", p0), ("p0_tokenwise", token["p0"]),
            ("p1_full", p1), ("p1_tokenwise", token["p1"]))},
        "passed": all(check["passed"] for check in checks.values()), "exploratory": True}


@torch.no_grad()
def continuation_probe(model, x, y, prefix_length, anchors, memory_anchors, full, recorder, cell, check_limit, progress=None):
    """Clone this cell's real prefix; replay selected actual suffix operands."""
    sites = projection_sites(model)
    selected = [sites[0]] if len(sites) == 1 else [sites[0], sites[-1]]
    model.configure_scan_backend("reference", 128)
    recorder["stage"] = cell + ".prefix_prefill"
    check_limit()
    with base.observe_scans(model, recorder):
        prefix_logits, prefix = model.prefill(x[:, :prefix_length], cache_dtype=torch.float32)
    before = base.state_snapshot(prefix)
    prefix_checks = {name: norm.cmp(prefix_logits, value[:, prefix_length - 1:prefix_length]) for name, value in anchors.items()}
    suffix, logits, memories, positions, probes, frozen_inputs = x[:, prefix_length:], {}, {}, {}, [], {}
    for route, size in (("one_shot", suffix.shape[1]), ("chunked128", 128), ("tokenwise", 1)):
        if progress is not None:
            progress.update(stage=cell + ".actual_suffix_" + route)
        state, parts = prefix.clone(), []
        clone_identity = breadth.state_identity(base.state_snapshot(state))
        if clone_identity != breadth.state_identity(before) or state.position != prefix_length:
            raise RuntimeError("continuation clone differs from its real prefix")
        recorder["stage"] = cell + ".suffix_" + route
        capture = ProjectionCapture(selected, prefix_length)
        with capture.scoped(), base.observe_scans(model, recorder):
            for start in range(0, suffix.shape[1], size):
                check_limit()
                capture.position = prefix_length + start
                output, _ = model(suffix[:, start:start + size], inference_state=state)
                parts.append(output.detach().cpu())
        captured = capture.finish(suffix.shape[1], offset=prefix_length)
        logits[route] = torch.cat(parts, 1)
        base.require_finite(logits[route], (x.shape[0], suffix.shape[1], model.cfg.vocab_size), "suffix " + route)
        memories[route], positions[route] = base.state_snapshot(state), state.position
        frozen_inputs[route] = (captured, clone_identity)
        del capture, parts
    after = base.state_snapshot(prefix)
    if prefix.position != prefix_length or any(not torch.equal(before[name], value) for name, value in after.items()) or any(p != x.shape[1] for p in positions.values()):
        raise RuntimeError("cloned continuation changed prefix or final positions")
    comparisons = {name: {anchor: norm.cmp(value, reference[:, prefix_length:]) for anchor, reference in anchors.items()} for name, value in logits.items()}
    route_checks = {name + "_vs_one_shot": norm.cmp(logits[name], logits["one_shot"]) for name in ("chunked128", "tokenwise")}
    retained = {name: {anchor: prior.state_comparison(value, reference, 8) for anchor, reference in memory_anchors.items()} for name, value in memories.items()}
    report = {"prefix_position": prefix_length, "suffix_length": suffix.shape[1], "synthetic": False, "prefix_unchanged": True,
        "prefix_fields": breadth.state_identity(before), "prefix_logits": prior.tensor_identity(prefix_logits), "prefix_comparisons": prefix_checks,
        "suffix_logits": {name: prior.tensor_identity(value) for name, value in logits.items()}, "suffix_comparisons": comparisons,
        "route_comparisons": route_checks, "retained_state": retained, "final_positions": positions,
        "suffix_state_fields": {name: breadth.state_identity(value) for name, value in memories.items()},
        "chunked_schedule": [min(128, suffix.shape[1] - start) for start in range(0, suffix.shape[1], 128)],
        "next_token_effects_vs_original": {name: norm.effects(value, anchors["original"][:, prefix_length:], y[:, prefix_length:], prefix_length) for name, value in logits.items()},
        "projection_sites": [name for name, _, _ in selected], "frozen_projections": probes,
        "execution_status": "incomplete", "failed_projection": None}
    # Local arithmetic/upstream probes remain distinct from whole-route acceptance.
    report["measured_routes_passed"] = all(norm.flags({key: value for key, value in report.items() if key != "frozen_projections"}))
    try:
        # Save whole-route endpoints before optional exploratory arithmetic can
        # exhaust the allowance. Completed measurements survive a frozen failure.
        for route in ("one_shot", "chunked128", "tokenwise"):
            captured, clone_identity = frozen_inputs[route]
            for name, module, layer in selected:
                report["failed_projection"] = {"route": route, "module": name}
                if progress is not None:
                    progress.update(stage=cell + ".frozen_suffix_" + route + "." + name)
                check_limit()
                values = captured[name]
                probe = frozen_projection_probe(values["inputs"], values["observed"], module, values["calls"],
                    active_cell=cell, offset=prefix_length, check_limit=check_limit)
                probes.append({"module": name, "layer": layer, "route": route, "origin": "actual_same_cell_prefix_continuation",
                    "prefix_position": prefix_length, "prefix_fields": clone_identity, "final_position": positions[route],
                    "upstream_input_vs_own_full": norm.cmp(values["inputs"], full["tensors"][(layer, "in_projection.input")][:, prefix_length:]),
                    "observed_output_vs_own_full": norm.cmp(values["observed"], full["tensors"][(layer, "in_projection.output")][:, prefix_length:]),
                    "frozen_arithmetic": probe})
        report["execution_status"], report["failed_projection"] = "completed", None
    except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError, MemoryError) as exc:
        report["reason"] = norm.clean_reason(exc)
    report["passed"] = report["execution_status"] == "completed" and report["measured_routes_passed"]
    return report


def reproduction(current, previous, original_current=None, original_previous=None):
    """Exact prior N1H0 endpoints/intermediates/prefix negatives, not transitivity."""
    if previous is None:
        return {"executed": False, "reason": "standalone CPU fixture has no prior measured case"}
    checks = {}
    for name in ENDPOINTS[:11]:
        checks["score." + name] = current["whole_model_comparisons"][name] == previous["whole_model_comparisons"][name]
    for name in norm.PAIRINGS:
        checks["stage." + name] = current["stage_comparisons"][name] == previous["stage_comparisons"][name]
    for bank in ("retained_state", "stateful_full_retained_state"):
        checks[bank] = all(current[bank][name] == value for name, value in previous[bank].items())
    for key in ("logits", "next_token_effects", "final_positions"):
        checks[key] = current[key] == previous[key]
    if original_current is not None and original_previous is not None:
        for key in ("logits", "loss", "stateful_logits", "state_fields", "stateful_vs_stateless", "repeat", "actual_paths"):
            checks["original." + key] = original_current[key] == original_previous[key]
    old, new = previous["actual_prefix"], current["actual_prefix"]
    for key in ("prefix_fields", "prefix_logits", "suffix_logits", "route_comparisons", "final_positions", "suffix_state_fields", "chunked_schedule"):
        checks["prefix." + key] = new[key] == old[key]
    checks["prefix.prefix_comparisons"] = all(new["prefix_comparisons"][name] == value for name, value in old["prefix_comparisons"].items())
    for bank in ("suffix_comparisons", "retained_state"):
        checks["prefix." + bank] = all(new[bank][route][name] == value for route, values in old[bank].items() for name, value in values.items())
    checks["prefix.next_token_effects_vs_original"] = new["next_token_effects_vs_original"] == old["next_token_effects_vs_original"]
    return {"executed": True, "comparisons": checks, "exact_equal": all(checks.values()),
            "scope": "all prior N1H0 score/stage/state and genuine-prefix comparison fields repeated exactly; prior failures remain immutable"}


@torch.no_grad()
def study_case(model, x, y, *, prefix_length, check_limit=lambda: None, progress=None, prior_case=None, prior_original=None):
    if (x.ndim != 2 or x.shape != y.shape or x.dtype != torch.int64 or y.dtype != torch.int64 or min(x.shape) <= 0
            or type(prefix_length) is not int or not 0 < prefix_length < x.shape[1] or not torch.equal(x[:, 1:], y[:, :-1])):
        raise ValueError("paired shifted tokens and substantive prefix required")
    weights, tie_before, policy_before, rng_before = base.weight_sha256(model), norm.tied_weight_snapshot(model), norm.runtime_policy(), rng_identity(x.device)
    result = {"weights_sha256": weights, "execution_status": "incomplete", "cells": [], "failed_cell": None,
        "frozen_projections": [], "backward_executed": False, "BF16_executed": False,
        "production_defaults_changed": False, "optimizer_executed": False}
    recorder = {"stage": "unexecuted", "events": {}}
    with preserve_rng(x.device), base.preserve_model_execution(model), base.tf32_disabled(), torch.autocast(x.device.type, enabled=False):
        policy_active = norm.runtime_policy()
        try:
            if progress is not None:
                progress.update(cell=None, stage="original anchors")
            original = collect_full(model, x, y, check_limit)
            check_limit()
            repeated, repeated_loss = model(x, y)
            base.require_finite(repeated_loss, (), "original loss")
            original_stateful, original_memory = norm.stateful_full(model, x, recorder, "original.stateful_full", check_limit)
            result["original_anchor"] = {"logits": prior.tensor_identity(original["logits"]), "loss": prior.tensor_identity(repeated_loss),
                "stateful_logits": prior.tensor_identity(original_stateful), "state_fields": breadth.state_identity(original_memory),
                "stateful_vs_stateless": norm.cmp(original_stateful, original["logits"]),
                "repeat": {"exact_equal": torch.equal(original["logits"], repeated.detach().cpu()), "comparison": norm.cmp(repeated, original["logits"])},
                "actual_paths": {"scan": original["scan_events"], "attention": original["attention_events"]}}
            with decay.temporary_decay_treatment():
                common = collect_full(model, x, y, check_limit)
                _, common_memory = norm.stateful_full(model, x, recorder, "decay_baseline.stateful_full", check_limit)
                result["decay_baseline_anchor"] = {"logits": prior.tensor_identity(common["logits"]), "state_fields": breadth.state_identity(common_memory)}
                projection_baseline = projection_memory = None
                with norm.temporary_cell(model, "N1H0"):
                    for cell in CELLS:
                        result["failed_cell"] = cell
                        if progress is not None:
                            progress.update(cell=cell, stage="full/tokenwise/stateful/continuation")
                        with temporary_projection(model, cell):
                            capture = ProjectionCapture(projection_sites(model))
                            if cell == "P0":
                                with capture.scoped():
                                    full = collect_full(model, x, y, check_limit)
                                captured = capture.finish(x.shape[1])
                                projection_baseline = full
                            else:
                                full = collect_full(model, x, y, check_limit)
                            token = prior.collect_route(model, x, y, tokenwise=True, check_limit=check_limit)
                            own_stateful, own_memory = norm.stateful_full(model, x, recorder, cell + ".stateful_full", check_limit)
                            if cell == "P0":
                                projection_memory = own_memory
                                result["projection_baseline_anchor"] = {"logits": prior.tensor_identity(full["logits"]), "state_fields": breadth.state_identity(own_memory), "cell": cell}
                            check_limit()
                            model.configure_scan_backend("torch_chunked", 128)
                            recorder["stage"] = cell + ".full_chunked128"
                            with base.observe_scans(model, recorder):
                                chunked, chunked_loss = model(x, y)
                            base.require_finite(chunked, (*x.shape, model.cfg.vocab_size), "chunked logits")
                            base.require_finite(chunked_loss, (), "chunked loss")
                            model.configure_scan_backend("reference", 128)
                            anchors = {"own": full["logits"], "original": original["logits"], "decay_baseline": common["logits"], "projection_baseline": projection_baseline["logits"]}
                            memory_anchors = {"own": own_memory, "original": original_memory, "decay_baseline": common_memory, "projection_baseline": projection_memory}
                            endpoints = dict(zip(ENDPOINTS, (norm.cmp(full["logits"], original["logits"]), norm.cmp(full["logits"], common["logits"]),
                                norm.cmp(token["logits"], full["logits"]), norm.cmp(token["logits"], original["logits"]), norm.cmp(token["logits"], common["logits"]),
                                norm.cmp(own_stateful, full["logits"]), norm.cmp(own_stateful, original["logits"]), norm.cmp(own_stateful, common["logits"]),
                                norm.cmp(chunked, full["logits"]), norm.cmp(chunked, original["logits"]), norm.cmp(chunked, common["logits"]),
                                norm.cmp(full["logits"], projection_baseline["logits"]), norm.cmp(token["logits"], projection_baseline["logits"]),
                                norm.cmp(own_stateful, projection_baseline["logits"]), norm.cmp(chunked, projection_baseline["logits"]))))
                            traces = {name: norm.compact_trace(actual, anchor, check_limit) for name, actual, anchor in (
                                ("full_vs_original", full, original), ("full_vs_decay_baseline", full, common),
                                ("tokenwise_vs_own_full", token, full), ("tokenwise_vs_original", token, original),
                                ("tokenwise_vs_decay_baseline", token, common), ("full_vs_projection_baseline", full, projection_baseline),
                                ("tokenwise_vs_projection_baseline", token, projection_baseline))}
                            retained = {name: prior.state_comparison(token["state"], value, 8) for name, value in memory_anchors.items()}
                            stateful_retained = {name: prior.state_comparison(own_memory, value, 8) for name, value in memory_anchors.items() if name != "own"}
                            continuation = continuation_probe(model, x, y, prefix_length, anchors, memory_anchors, full, recorder, cell, check_limit, progress)
                            report = {"cell_id": cell, "input_projection_fp64_matmul": cell == "P1", "normalization_cell": "N1H0",
                                "projection_sites": [name for name, _, _ in projection_sites(model)], "execution_status": continuation["execution_status"],
                                "whole_model_comparisons": endpoints, "stage_comparisons": traces, "retained_state": retained,
                                "stateful_full_retained_state": stateful_retained, "actual_prefix": continuation,
                                "final_positions": {"tokenwise": token["position"], "stateful_full": x.shape[1]},
                                "logits": {"full": prior.tensor_identity(full["logits"]), "tokenwise": prior.tensor_identity(token["logits"]), "chunked128": prior.tensor_identity(chunked)},
                                "losses": {"full": prior.tensor_identity(full["loss"]), "chunked128": prior.tensor_identity(chunked_loss)},
                                "loss_comparisons": dict(zip(LOSS_ENDPOINTS, (norm.cmp(full["loss"], original["loss"]), norm.cmp(full["loss"], common["loss"]),
                                    norm.cmp(full["loss"], projection_baseline["loss"]), norm.cmp(chunked_loss, full["loss"]), norm.cmp(chunked_loss, original["loss"]),
                                    norm.cmp(chunked_loss, common["loss"]), norm.cmp(chunked_loss, projection_baseline["loss"])))),
                                "next_token_effects": {"full_vs_original": norm.effects(full["logits"], original["logits"], y),
                                    "tokenwise_vs_original": norm.effects(token["logits"], original["logits"], y), "tokenwise_vs_own_full": norm.effects(token["logits"], full["logits"], y)},
                                "actual_paths": {"full": {"scan": full["scan_events"], "attention": full["attention_events"]},
                                    "tokenwise": {"scan": token["scan_events"], "attention": token["attention_events"]}},
                                "backward_executed": False, "BF16_executed": False}
                            report["whole_model_passed"] = (report["execution_status"] == "completed" and all(value["passed"] for value in endpoints.values())
                                and all(value["passed"] for value in [*retained.values(), *stateful_retained.values()]) and continuation["passed"]
                                and all(value["passed"] for value in report["loss_comparisons"].values()))
                            report["all_recorded_comparisons_passed"] = report["execution_status"] == "completed" and all(norm.flags(report))
                            result["cells"].append(report)
                            if report["execution_status"] != "completed":
                                raise RuntimeError(continuation.get("reason", "frozen prefix projection incomplete"))
                            if cell == "P0":
                                result["prior_reproduction"] = reproduction(report, prior_case, result["original_anchor"], prior_original)
                                if progress is not None:
                                    progress.update(stage="frozen full input projections")
                                for name, module, layer in projection_sites(model):
                                    values = captured[name]
                                    check_limit()
                                    result["frozen_projections"].append({"module": name, "layer": layer,
                                        "origin": "identical_P0_stateless_full_operands", **frozen_projection_probe(values["inputs"],
                                            values["observed"], module, values["calls"], check_limit=check_limit)})
                                del captured, capture
                            del token, own_stateful, own_memory, chunked, chunked_loss
                            if cell != "P0":
                                del full
            if len(result["frozen_projections"]) != len(projection_sites(model)):
                raise RuntimeError("required frozen projection inventory incomplete")
            result["execution_status"], result["failed_cell"] = "completed", None
        except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError, MemoryError) as exc:
            result["reason"] = norm.clean_reason(exc)
    result["runtime_policy"] = {"before": policy_before, "active": policy_active, "restored": norm.runtime_policy(), "restoration_exact": policy_before == norm.runtime_policy()}
    result["rng_integrity"] = {"before": rng_before, "after": rng_identity(x.device), "restoration_exact": rng_before == rng_identity(x.device)}
    result["tied_weight_integrity"] = norm.tied_weight_report(tie_before, norm.tied_weight_snapshot(model))
    result["post_model_weights_unchanged"] = base.weight_sha256(model) == weights
    safeguards = []
    for passed, reason in ((result["post_model_weights_unchanged"], "model weight bytes changed"),
        (result["tied_weight_integrity"]["passed"], "head/embedding identity or storage changed"),
        (result["runtime_policy"]["restoration_exact"], "runtime policy restoration failed"),
        (result["rng_integrity"]["restoration_exact"], "RNG restoration failed")):
        if not passed:
            safeguards.append(reason)
    if safeguards:
        result["execution_status"] = "incomplete"
        result["reason"] = (result.get("reason", "") + "; " if result.get("reason") else "") + "; ".join(safeguards)
    result["operator_observations"] = norm.observed_events(recorder)
    result["attribution_control_stable"] = (result.get("original_anchor", {}).get("repeat", {}).get("exact_equal", False)
        and result.get("prior_reproduction", {}).get("exact_equal", prior_case is None))
    result["all_recorded_comparisons_passed"] = result["execution_status"] == "completed" and result["attribution_control_stable"] and all(norm.flags(result))
    return result


def source_fingerprints():
    return {**norm.source_fingerprints(), "scripts/study_projection_precision.py": base.file_sha256(Path(__file__))}


def read_prior_report(report_path, publication_path=DEFAULT_PRIOR_PUBLICATION):
    """Verify public transport and exact bounded original bytes before parsing."""
    manifest = base.read_json(publication_path)
    if (manifest.get("kind") != "lossless_normalization_head_raw_publication" or manifest.get("compression") != "gzip"
            or manifest.get("round_trip_exact") is not True or type(manifest.get("uncompressed_bytes")) is not int
            or not 0 < manifest["uncompressed_bytes"] <= MAX_PRIOR_BYTES):
        raise ValueError("invalid prior lossless transport manifest")
    path = Path(report_path)
    compressed = path.suffix == ".gz"
    expected_sha = manifest["sha256"] if compressed else manifest["uncompressed_sha256"]
    expected_bytes = manifest["bytes"] if compressed else manifest["uncompressed_bytes"]
    if path.stat().st_size != expected_bytes or base.file_sha256(path) != expected_sha:
        raise ValueError("prior transport bytes/hash differ")
    with (gzip.open(path, "rb") if compressed else path.open("rb")) as handle:
        payload = handle.read(manifest["uncompressed_bytes"] + 1)
    if len(payload) != manifest["uncompressed_bytes"] or hashlib.sha256(payload).hexdigest() != manifest["uncompressed_sha256"]:
        raise ValueError("prior original JSON bytes/hash differ")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate prior JSON key: " + key)
            result[key] = value
        return result
    report = json.loads(payload, object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite prior JSON constant: " + value)))
    del payload
    return report, {"publication": {**base.public_reference(publication_path), "sha256": base.file_sha256(publication_path)},
        "selected_representation": {**base.public_reference(path), "bytes": expected_bytes, "sha256": expected_sha, "compressed": compressed},
        "original": {"path": manifest["uncompressed_path"], "bytes": manifest["uncompressed_bytes"], "sha256": manifest["uncompressed_sha256"]}}


def declaration_template(*, device="cuda", prior_declaration=DEFAULT_PRIOR_DECLARATION, prior_report=DEFAULT_PRIOR_REPORT,
                         prior_publication=DEFAULT_PRIOR_PUBLICATION, prior_summary=DEFAULT_PRIOR_SUMMARY,
                         output=DEFAULT_OUTPUT, _previous=None):
    previous, transport = _previous or read_prior_report(prior_report, prior_publication)
    old = previous["protocol"]
    return {"schema": 1, "date": "2026-10-06", "status_at_declaration": "planned_before_execution", "device": device,
        "seed": 2027, "ratios": list(RATIOS), "cases": list(CASES), "shared_inputs": old["shared_inputs"], "cells": list(CELLS),
        "decay_treatment": decay.TREATMENT, "normalization_cell": "N1H0", "head_treatment": "historical FP32 lm_head",
        "projection_treatment": "fp64_mamba_input_projection_matmul",
        "projection_scope": "all Mamba in_proj instances only; FP64 F.linear inputs/weights/optional bias then FP32 output; original parameters unchanged",
        "projection_site_registry": {ratio: [f"blocks.{layer}.mixer.in_proj" for layer in range(16) if (layer + 1) % (16 if ratio == "1:15" else 4)] for ratio in RATIOS},
        "chunk_size": 128, "tolerances": base.TOLERANCES, "report_schema": REPORT_SCHEMA,
        "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
        "observed_inference_policy": {key: value for key, value in norm.runtime_policy().items() if "allow_tf32" not in key},
        "time_budget_seconds": 900, "minimum_large_case_free_bytes": breadth.MINIMUM_FREE_BYTES,
        "backward_executed": False, "BF16_executed": False, "production_defaults_changed": False, "optimizer_executed": False,
        "prior_declaration": {**base.public_reference(prior_declaration), "sha256": base.file_sha256(prior_declaration)},
        "prior_summary": {**base.public_reference(prior_summary), "sha256": base.file_sha256(prior_summary)},
        "prior_transport": transport, "prior_control_cell": "N1H0", "output": base.public_reference(output)["path"],
        "source_sha256": source_fingerprints(), "data": old["data"], "checkpoints": old["checkpoints"]}


def audit_declaration(path, *, device, prior_declaration, prior_report, prior_publication, prior_summary, output):
    planned = base.read_json(path)
    previous = read_prior_report(prior_report, prior_publication)
    expected = declaration_template(device=device, prior_declaration=prior_declaration, prior_report=prior_report,
        prior_publication=prior_publication, prior_summary=prior_summary, output=output, _previous=previous)
    if any(planned.get(key) != value for key, value in expected.items()):
        raise ValueError("declaration differs from exact projection scope, output or identities")
    old, _ = previous
    old_plan = base.read_json(prior_declaration)
    if (old.get("kind") != "trained_checkpoint_normalization_head_precision" or old.get("execution_status") != "completed"
            or old.get("certified") is not False or old.get("protocol_sha256") != base.canonical_sha256(old["protocol"])
            or old["protocol"]["declaration"]["sha256"] != base.file_sha256(prior_declaration)
            or old["protocol"]["shared_inputs"] != old_plan["shared_inputs"] or old["protocol"]["tolerances"] != base.TOLERANCES
            or set(old["protocol"]["source_sha256"]) != set(norm.source_fingerprints())
            or old["protocol"]["observed_inference_policy"] != planned["observed_inference_policy"]):
        raise ValueError("prior immutable normalization protocol/source/runtime scope differs")
    keys = [(row["ratio"], row["case_id"]) for row in old["cases"]]
    if len(keys) != 6 or set(keys) != {(ratio, case_id) for ratio in RATIOS for case_id in CASE_IDS}:
        raise ValueError("prior six-case coverage differs")
    for row in old["cases"]:
        if row["execution_status"] != "completed" or [cell["cell_id"] for cell in row["cells"]] != list(norm.CELLS) or not row["attribution_control_stable"]:
            raise ValueError("prior case/cell/control coverage differs")
    for name, digest in old["protocol"]["source_sha256"].items():
        normalized = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if digest not in {hashlib.sha256(normalized).hexdigest(), hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest()}:
            raise ValueError("prior measured source fingerprint differs: " + name)
    # Retain only the bound N1H0 rows required for repetition, not all four large cells.
    controls = [{"ratio": row["ratio"], "case_id": row["case_id"], "weights_sha256": row["weights_sha256"],
                 "original_anchor": row["original_anchor"],
                 "cell": next(cell for cell in row["cells"] if cell["cell_id"] == "N1H0")} for row in old["cases"]]
    return planned, controls


def run_study(device="cuda", *, declaration=DEFAULT_DECLARATION, output=DEFAULT_OUTPUT,
              prior_declaration=DEFAULT_PRIOR_DECLARATION, prior_report=DEFAULT_PRIOR_REPORT,
              prior_publication=DEFAULT_PRIOR_PUBLICATION, prior_summary=DEFAULT_PRIOR_SUMMARY,
              checkpoint_root=Path("checkpoints"), training_run_id="week3-700m-v1",
              data_dir=Path("data/openwebtext-5b"), tokenizer=Path("data/tokenizer/openwebtext.json")):
    limit = breadth.DiagnosticLimit(900)
    if device not in ("cpu", "cuda") or Path(output).exists():
        raise ValueError("device must be cpu/cuda and declared output must not exist")
    limit.check()
    planned, controls = audit_declaration(declaration, device=device, prior_declaration=prior_declaration,
        prior_report=prior_report, prior_publication=prior_publication, prior_summary=prior_summary, output=output)
    limit.check()
    checkpoints, data_identity, checkpoint_identity = base.audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, RATIOS)
    limit.check()
    prepared = prior.prepare_inputs(base.load_split(data_dir, "val"))
    if [row[2] for row in prepared] != planned["shared_inputs"] or data_identity != planned["data"] or checkpoint_identity != planned["checkpoints"]:
        raise ValueError("actual data/checkpoint/input identities differ from declaration")
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA unavailable; no fallback")
    sources = source_fingerprints()
    if sources != planned["source_sha256"]:
        raise ValueError("source bytes changed during declaration/input preflight")
    evidence = {Path(path): base.file_sha256(path) for path in (declaration, prior_declaration, prior_report, prior_publication, prior_summary)}
    rows, headroom, reason, active = [], [], None, None
    policy_before = norm.runtime_policy()
    with preserve_rng(torch.device(device)), base.tf32_disabled():
        runtime = base.runtime_metadata(device)
        policy_active = norm.runtime_policy()
        runtime["inference_policy"] = policy_active
        try:
            for ratio in RATIOS:
                active = {"ratio": ratio, "case_id": None, "cell": None, "stage": "load model"}
                limit.check()
                model = base.load_variant_model(checkpoints[ratio], torch.device(device))
                try:
                    if base.weight_sha256(model) != next(row["weights_sha256"] for row in controls if row["ratio"] == ratio):
                        raise ValueError("loaded weights differ from prior evidence")
                    if [name for name, _, _ in projection_sites(model)] != planned["projection_site_registry"][ratio]:
                        raise ValueError("loaded projection site registry differs")
                    for x, y, identity in prepared:
                        active = {"ratio": ratio, "case_id": identity["case_id"], "cell": None, "stage": "case preflight"}
                        limit.check()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            if identity["batch_size"] == 2:
                                free, total = torch.cuda.mem_get_info()
                                headroom.append({"ratio": ratio, "case_id": identity["case_id"], "free_bytes": free, "total_bytes": total, "minimum_free_bytes": breadth.MINIMUM_FREE_BYTES})
                                if free < breadth.MINIMUM_FREE_BYTES:
                                    raise RuntimeError("insufficient declared 8GiB large-case headroom; no fallback")
                        started = time.perf_counter()
                        control = next(row for row in controls if row["ratio"] == ratio and row["case_id"] == identity["case_id"])
                        row = study_case(model, x.to(device), y.to(device), prefix_length=identity["prefix_length"], check_limit=limit.check,
                            progress=active, prior_case=control["cell"], prior_original=control["original_anchor"])
                        if device == "cuda":
                            torch.cuda.synchronize()
                        rows.append({"ratio": ratio, **identity, **row, "processing_seconds": time.perf_counter() - started,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
                        print(json.dumps({"progress": "case_finished", "ratio": ratio, "case_id": identity["case_id"], "execution_status": row["execution_status"],
                            "completed_cells": len(row["cells"]), "processing_seconds": rows[-1]["processing_seconds"]}), flush=True)
                        if row["execution_status"] != "completed":
                            raise RuntimeError(row.get("reason", "case incomplete"))
                finally:
                    del model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
            active = None
        except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError, MemoryError) as exc:
            reason = norm.clean_reason(exc)
    policy_report = {"before": policy_before, "active": policy_active, "restored": norm.runtime_policy(), "restoration_exact": policy_before == norm.runtime_policy()}
    data_files = [(Path(data_dir) / "manifest.json", data_identity["manifest"]["sha256"]), (Path(tokenizer), data_identity["tokenizer"]["sha256"]),
        *[(Path(data_dir) / Path(item["path"]).name, item["sha256"]) for item in data_identity["artifacts"].values()]]
    groups = (("checkpoints", [(checkpoints[item["ratio"]].best_path, item["sha256"]) for item in checkpoint_identity]),
        ("training_manifests", [(checkpoints[item["ratio"]].run_dir / "manifest.json", item["training_manifest"]["sha256"]) for item in checkpoint_identity]),
        ("prepared_data_and_tokenizer", data_files), ("prior_and_declaration", list(evidence.items())), ("sources", [(ROOT / path, digest) for path, digest in sources.items()]))
    integrity, files, audit_reason = norm.audit_final_identities(groups, limit.check)
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
    return {"schema": 1, "kind": "trained_checkpoint_projection_precision", "certified": False,
        "execution_status": "incomplete" if reason else "completed", "status": "incomplete" if reason else
            "completed" if all(row["all_recorded_comparisons_passed"] for row in rows) else "completed_with_parity_failures",
        "reason": reason, "failed_case": active if reason else None, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "protocol_sha256": base.canonical_sha256(protocol), "cases": rows, "large_case_headroom": headroom,
        "post_execution_integrity": integrity, "post_execution_audit_files": files, "runtime_policy": policy_report,
        "execution_allowance": {**limit.metadata(), "scope": "runner entry through preflight/model work/final audits; immutable publication excluded"},
        "limits": ["Six checkpoint-selection-pool cases; no independent quality or training conclusion.",
            "P1 changes only the Mamba input-projection family; decay and N1H0 stay fixed. No single-layer causal attribution follows.",
            "Original stateless scores, actual original stateful memory, common decay, P0 projection baseline and own full anchors are distinct.",
            "Identical frozen operands and independently written CPU NumPy FP64 references isolate arithmetic; actual suffix upstream operands may differ.",
            "Original P0 negative endpoints/intermediates/prefixes must reproduce exactly before attribution; no tolerance widening.",
            "Passing-stage redundant samples compacted as declared; every failed stage and bounded sample retained.",
            "Genuine prefix clones retain complete state identity/positions; frozen projection replay does not reconstruct a recurrent scan.",
            "No backward, BF16, optimizer, fused backend or production change. Original gradient_check input identity retained but unexecuted.",
            "Cooperative900s allowance and8GiB free-memory guard do not guarantee fit or zero overshoot; incomplete routes supply no passes.",
            "A promising candidate still needs expanded three-ratio gradients/state/causality and separately signed numerical/recovery policies."]}


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
        report = {"schema": 1, "kind": "trained_checkpoint_projection_precision", "certified": False,
            "execution_status": "incomplete", "status": "incomplete", "reason": norm.clean_reason(exc), "cases": []}
    base.write_report(args.output, report)
    print(json.dumps({"status": report["status"], "execution_status": report["execution_status"],
        "completed_cases": sum(row.get("execution_status") == "completed" for row in report["cases"]), "completed_cells": sum(len(row.get("cells", [])) for row in report["cases"])}))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
