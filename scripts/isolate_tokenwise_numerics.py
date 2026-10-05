"""Bounded, read-only isolation of trained FP32 tokenwise numerical differences.

The existing cumulative-decay treatment is fixed. Hooks observe three complete
routes; frozen component replays distinguish shape effects from propagated input
differences. FP64 calculations are independent exploratory anchors, not policies.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import scripts.study_checkpoint_numerics as base  # noqa: E402
import scripts.check_decay_breadth as breadth  # noqa: E402
import scripts.study_decay_precision as decay  # noqa: E402
import src.model.mamba2 as mamba  # noqa: E402
from scripts.isolate_checkpoint_numerics import detailed_comparison, recurrence_fp64, OPERANDS  # noqa: E402
from src.model.scan_backend import BackendUnavailableError  # noqa: E402

RATIOS = ("1:15", "1:3")
CASE_IDS = ("b1-l128-p64", "b1-l129-p64", "b2-l512-p128")
CASES = tuple(case for case in breadth.CASES if breadth.case_id(case) in CASE_IDS)
MINIMUM_FREE_BYTES = breadth.MINIMUM_FREE_BYTES
ROUTES = ("original_full_reference", "candidate_full_reference", "candidate_tokenwise_reference")
DEFAULT_BREADTH_DECLARATION = ROOT / "docs/research/decay-breadth-protocol-2026-10-04.json"
DEFAULT_BREADTH_REPORT = ROOT / "docs/research/checks/decay-breadth-2026-10-04/natural-boundaries-seed-2027.json"
DEFAULT_OUTPUT = ROOT / "docs/research/checks/tokenwise-isolation-2026-10-04/natural-remaining-seed-2027.json"


def prepare_inputs(validation):
    """Use the immutable breadth sampling choices; no generator/global RNG changes."""
    return [row for row in breadth.prepare_inputs(validation) if row[2]["case_id"] in CASE_IDS]


def tensor_identity(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": base.tensor_sha256(value.reshape(-1))}


def comparison_flags(value):
    """Collect explicit comparison flags without turning unexecuted routes green."""
    if isinstance(value, dict):
        return ([value["passed"]] if "passed" in value else []) + [flag for item in value.values() for flag in comparison_flags(item)]
    if isinstance(value, list):
        return [flag for item in value for flag in comparison_flags(item)]
    return []


def compare(actual, expected, *, coordinate_limit=8, oracle=False):
    return detailed_comparison(actual, expected, coordinate_limit=coordinate_limit,
                               comparison_dtype="float64" if oracle else "float32")


class TokenTrace:
    """Detached observations, with an explicit token axis and constant coverage."""
    def __init__(self, *, selected_layers=None, include_global=True):
        self.parts, self.order, self.constants, self.scan_initials = {}, [], {}, {}
        self.selected_layers, self.include_global = selected_layers, include_global
        self.query_length, self.position, self.active = 0, 0, None
        self.scan_events, self.attention_events = [], []

    def capture(self, layer, stage, value, *, constant=False):
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            raise ValueError("trace requires nonempty tensor: " + stage)
        # CUDA copies stay on the device until the route finishes. Validating and
        # transferring once per aligned stage avoids a synchronization per token.
        frozen = value.detach().clone()
        key = (layer, stage)
        if constant:
            if key not in self.constants:
                self.constants[key] = []
                self.order.append(key)
            self.constants[key].append(frozen)
            return
        if frozen.ndim < 2 or frozen.shape[1] != self.query_length:
            raise ValueError("trace token axis/length differs: " + stage)
        if key not in self.parts:
            self.parts[key] = []
            self.order.append(key)
        self.parts[key].append((self.position, frozen))

    def tensors(self, length, *, offset=0):
        result = {}
        for key in self.order:
            if key in self.constants:
                copies = torch.stack(self.constants[key]).cpu()
                base.require_finite(copies, tuple(copies.shape), "trace constant " + key[1])
                if not torch.equal(copies, copies[0].expand_as(copies)):
                    raise RuntimeError("constant changed between trace calls: " + key[1])
                result[key] = copies[0].clone()
                continue
            position = offset
            values = []
            for start, value in self.parts[key]:
                if start != position:
                    raise RuntimeError("trace token coverage is missing, duplicated or out of order")
                position += value.shape[1]
                values.append(value)
            if position != offset + length:
                raise RuntimeError("trace token coverage differs from requested route")
            result[key] = torch.cat(values, 1).cpu()
            base.require_finite(result[key], tuple(result[key].shape), "trace " + key[1])
        return result


class TraceScan:
    def __init__(self, backend, layer, trace):
        self.backend, self.layer, self.trace = backend, layer, trace
        self.name, self.chunk_size = backend.name, backend.chunk_size

    def scan(self, *args, **kwargs):
        for name, value in zip(OPERANDS, args[:6]):
            self.trace.capture(self.layer, "scan_input." + name, value, constant=name in ("A", "D"))
        if self.layer not in self.trace.scan_initials:
            self.trace.scan_initials[self.layer] = None if args[6] is None else args[6].detach().cpu().clone()
        output, memory = self.backend.scan(*args, **kwargs)
        self.trace.capture(self.layer, "scan_output", output)
        self.trace.scan_events.append({"layer": self.layer, "position": self.trace.position,
            "length": args[0].shape[1], "stateful": args[6] is not None,
            "path": "torch.quadratic_ssd" if args[6] is None else "torch.chunked_ssd",
            "backend": self.name, "chunk_size": kwargs.get("inference_chunk_size", self.chunk_size)})
        return output, memory


@contextmanager
def trace_execution(model, trace):
    """Serial observational interceptors cover both module and functional paths."""
    hooks, backends = [], []
    conv1d, sdpa = F.conv1d, F.scaled_dot_product_attention

    def output_hook(layer, stage):
        return lambda _module, _inputs, output: trace.capture(layer, stage, output)

    def input_hook(layer, stage):
        return lambda _module, inputs: trace.capture(layer, stage, inputs[0])

    def linear_hooks(module, layer, stage):
        hooks.append(module.register_forward_pre_hook(input_hook(layer, stage + ".input")))
        hooks.append(module.register_forward_hook(output_hook(layer, stage + ".output")))

    def intercept_conv(inputs, weight, bias=None, *args, **kwargs):
        output = conv1d(inputs, weight, bias, *args, **kwargs)
        if trace.active is not None and trace.active[1] == "mamba":
            trace.capture(trace.active[0], "convolution_raw", output[..., :trace.query_length].transpose(1, 2))
        return output

    def intercept_attention(q, k, v, *args, **kwargs):
        output = sdpa(q, k, v, *args, **kwargs)
        if trace.active is not None and trace.active[1] == "attention":
            layer = trace.active[0]
            for name, value in (("query", q), ("key", k[..., -trace.query_length:, :]),
                                ("value", v[..., -trace.query_length:, :]), ("output", output)):
                # Head-first SDPA tensors become uniform (batch, token, head, feature).
                trace.capture(layer, "attention_" + name, value.transpose(1, 2))
            mask = kwargs.get("attn_mask")
            trace.attention_events.append({"layer": layer, "position": trace.position,
                "query_length": q.shape[2], "key_length": k.shape[2],
                "is_causal": kwargs.get("is_causal", False), "mask_present": mask is not None,
                "mask_shape": None if mask is None else list(mask.shape),
                "operand_layout": {name: {"stride": list(value.stride()), "storage_offset": value.storage_offset(),
                                            "dtype": str(value.dtype), "shape": list(value.shape)}
                                   for name, value in zip(("q", "k", "v"), (q, k, v))}})
        return output

    try:
        F.conv1d, F.scaled_dot_product_attention = intercept_conv, intercept_attention
        if trace.include_global:
            hooks.append(model.embed.register_forward_hook(output_hook(None, "embedding")))
        for layer, block in enumerate(model.blocks):
            if trace.selected_layers is not None and layer not in trace.selected_layers:
                continue
            hooks.append(block.register_forward_pre_hook(input_hook(layer, "block_input")))
            hooks.append(block.norm1.register_forward_hook(output_hook(layer, "norm1_output")))
            kind = "attention" if block.is_attn else "mamba"
            def activate(_module, _inputs, layer=layer, kind=kind):
                if trace.active is not None:
                    raise RuntimeError("nested mixer execution is unsupported by this serial diagnostic")
                trace.active = (layer, kind)
            def deactivate(_module, _inputs, _output):
                trace.active = None
            hooks.append(block.mixer.register_forward_pre_hook(activate))
            hooks.append(block.mixer.register_forward_hook(deactivate, always_call=True))
            if block.is_attn:
                linear_hooks(block.mixer.qkv, layer, "attention_qkv")
                linear_hooks(block.mixer.out, layer, "attention_projection")
            else:
                mixer = block.mixer
                backends.append((mixer, mixer.scan_backend))
                mixer.scan_backend = TraceScan(mixer.scan_backend, layer, trace)
                linear_hooks(mixer.in_proj, layer, "in_projection")
                hooks.append(mixer.norm.register_forward_pre_hook(input_hook(layer, "gated_norm_input")))
                hooks.append(mixer.norm.register_forward_hook(output_hook(layer, "gated_norm_output")))
                linear_hooks(mixer.out_proj, layer, "out_projection")
            hooks.append(block.mixer.register_forward_hook(output_hook(layer, "mixer_output")))
            if block.has_mlp:
                hooks.append(block.norm2.register_forward_pre_hook(input_hook(layer, "mixer_residual")))
                hooks.append(block.norm2.register_forward_hook(output_hook(layer, "norm2_output")))
                for name in ("gate", "up", "down"):
                    linear_hooks(getattr(block.mlp, name), layer, "mlp_" + name)
                hooks.append(block.mlp.register_forward_hook(output_hook(layer, "mlp_output")))
            hooks.append(block.register_forward_hook(output_hook(layer, "block_output")))
        if trace.include_global:
            hooks.append(model.norm_f.register_forward_hook(output_hook(None, "final_norm")))
            linear_hooks(model.lm_head, None, "lm_head")
        yield
    finally:
        trace.active = None
        F.conv1d, F.scaled_dot_product_attention = conv1d, sdpa
        for hook in hooks:
            hook.remove()
        for mixer, backend in backends:
            mixer.scan_backend = backend


@torch.no_grad()
def collect_route(model, x, y, *, tokenwise=False, check_limit=lambda: None):
    model.configure_scan_backend("reference", 128)
    trace = TokenTrace()
    state = model.init_inference_state(x.shape[0], device=x.device, cache_dtype=torch.float32) if tokenwise else None
    parts = []
    with trace_execution(model, trace), torch.autocast(x.device.type, enabled=False):
        for start in range(x.shape[1]) if tokenwise else (0,):
            check_limit()
            trace.position, trace.query_length = start, 1 if tokenwise else x.shape[1]
            output = model.decode(x[:, start:start + 1], state) if tokenwise else model(x, y)[0]
            parts.append(output.detach().cpu())
    logits = torch.cat(parts, 1)
    base.require_finite(logits, (*x.shape, model.cfg.vocab_size), "route logits")
    if state is not None and state.position != x.shape[1]:
        raise RuntimeError("tokenwise end position differs")
    return {"tensors": trace.tensors(x.shape[1]), "order": trace.order, "logits": logits,
            "state": None if state is None else base.state_snapshot(state),
            "position": None if state is None else state.position,
            "scan_events": trace.scan_events, "attention_events": trace.attention_events}


def stage_comparisons(actual, expected, coordinate_limit=8, check_limit=lambda: None):
    if actual["order"] != expected["order"] or set(actual["tensors"]) != set(expected["tensors"]):
        raise RuntimeError("route stage names/order differ")
    rows = []
    for layer, stage in expected["order"]:
        check_limit()
        rows.append({"layer": layer, "stage": stage,
            **compare(actual["tensors"][(layer, stage)], expected["tensors"][(layer, stage)],
                      coordinate_limit=coordinate_limit)})
    def first(key):
        row = next((item for item in rows if item[key] is not None), None)
        return None if row is None else {"layer": row["layer"], "stage": row["stage"], "sample": row[key]}
    return {"stages": rows, "first_nonzero": first("first_nonzero"), "first_violation": first("first_violation"),
            "passed": all(row["passed"] for row in rows)}


def select_mixers(stage_rows, model):
    """First mixer drift, first mixer threshold failure, last if no mixer failure."""
    picked = {}
    def add(layer, reason):
        if layer is not None:
            picked.setdefault(layer, []).append(reason)
    mixer_rows = [row for row in stage_rows if row["layer"] is not None
                  and not row["stage"].startswith(("mlp_", "norm2", "block_input", "block_output", "mixer_residual"))]
    drift = next((row for row in mixer_rows if row["nonzero_error_count"]), None)
    failure = next((row for row in mixer_rows if not row["passed"]), None)
    add(None if drift is None else drift["layer"], "first_mixer_nonzero_drift")
    add(None if failure is None else failure["layer"], "first_mixer_fixed_tolerance_violation")
    if failure is None:
        add(len(model.blocks) - 1, "last_mixer_no_intermediate_mixer_violation")
    return [{"layer": layer, "kind": "attention" if model.blocks[layer].is_attn else "mamba", "reasons": reasons}
            for layer, reasons in picked.items()][:3]


@torch.no_grad()
def state_comparison(actual, expected, coordinate_limit=8):
    if list(actual) != list(expected):
        raise RuntimeError("state fields differ")
    fields = [{"field": name, **compare(value, expected[name], coordinate_limit=coordinate_limit)}
              for name, value in actual.items() if value.numel()]
    return {"tolerance": dict(base.TOLERANCES["float32"]), "fields": fields,
            "passed": all(row["passed"] for row in fields)}


@torch.no_grad()
def linear_probe(inputs, module, observed, *, check_limit=lambda: None, coordinate_limit=8):
    """Identical operands, full vs per-token GEMM; FP64 is a frozen-input anchor."""
    inputs = inputs.detach().to(module.weight.device).float()
    weight, bias = module.weight.detach(), None if module.bias is None else module.bias.detach()
    check_limit()
    full = F.linear(inputs, weight, bias)
    parts = []
    for i in range(inputs.shape[1]):
        check_limit()
        parts.append(F.linear(inputs[:, i:i + 1], weight, bias))
    token = torch.cat(parts, 1)
    check_limit()
    oracle = F.linear(inputs.double(), weight.double(), None if bias is None else bias.double())
    return {"input": tensor_identity(inputs), "weight": tensor_identity(weight),
            "bias": None if bias is None else tensor_identity(bias), "identical_operands": True,
            "full_replay_vs_observed": compare(full, observed, coordinate_limit=coordinate_limit),
            "tokenwise_vs_full": compare(token, full, coordinate_limit=coordinate_limit),
            "full_vs_fp64": compare(full, oracle, coordinate_limit=coordinate_limit, oracle=True),
            "tokenwise_vs_fp64": compare(token, oracle, coordinate_limit=coordinate_limit, oracle=True)}


@torch.no_grad()
def convolution_oracle_fp64(inputs, weight, bias=None, initial_tail=None):
    """Independent causal depthwise sum; cross-correlation kernel order retained."""
    if inputs.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1 or weight.shape[0] != inputs.shape[2]:
        raise ValueError("depthwise convolution requires B,L,C inputs and C,1,K weights")
    b, length, channels = inputs.shape
    width = weight.shape[-1]
    if min(b, length, channels, width) <= 0:
        raise ValueError("convolution dimensions must be positive")
    tail = torch.zeros(b, channels, width - 1, device=inputs.device, dtype=torch.float64) if initial_tail is None else initial_tail.detach().double()
    base.require_finite(inputs, (b, length, channels), "convolution oracle input")
    base.require_finite(weight, (channels, 1, width), "convolution oracle weight")
    base.require_finite(tail, (b, channels, width - 1), "convolution oracle prefix")
    if bias is not None:
        base.require_finite(bias, (channels,), "convolution oracle bias")
    joined = torch.cat((tail, inputs.detach().double().transpose(1, 2)), 2)
    output = torch.zeros(b, length, channels, dtype=torch.float64, device=inputs.device)
    for k in range(width):
        output += joined[..., k:k + length].transpose(1, 2) * weight.detach().double()[:, 0, k]
    return output if bias is None else output + bias.detach().double()


@torch.no_grad()
def convolution_probe(inputs, module, observed, *, check_limit=lambda: None, coordinate_limit=8):
    inputs = inputs.detach().to(module.weight.device).float()
    b, length, channels = inputs.shape
    tail = module.kernel_size[0] - 1
    check_limit()
    full = F.conv1d(inputs.transpose(1, 2), module.weight, module.bias, padding=tail, groups=channels)[..., :length].transpose(1, 2)
    joined = F.pad(inputs.transpose(1, 2), (tail, 0))
    parts = []
    for i in range(length):
        check_limit()
        parts.append(F.conv1d(joined[..., i:i + tail + 1], module.weight, module.bias, groups=channels).transpose(1, 2))
    token = torch.cat(parts, 1)
    oracle = convolution_oracle_fp64(inputs, module.weight, module.bias)
    return {"input": tensor_identity(inputs), "weight": tensor_identity(module.weight), "identical_operands": True,
            "initial_condition": "zero causal convolution tail, same frozen full-route projected input",
            "full_replay_vs_observed": compare(full, observed, coordinate_limit=coordinate_limit),
            "tokenwise_vs_full": compare(token, full, coordinate_limit=coordinate_limit),
            "full_vs_fp64": compare(full, oracle, coordinate_limit=coordinate_limit, oracle=True),
            "tokenwise_vs_fp64": compare(token, oracle, coordinate_limit=coordinate_limit, oracle=True)}


@torch.no_grad()
def attention_oracle_fp64(q, k, v, *, prefix_length=0):
    """Explicit FP64 softmax on identical post-RoPE tensors and absolute positions."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError("attention expects compatible B,H,Q/K,D tensors")
    if type(prefix_length) is not int or prefix_length < 0 or k.shape[2] != prefix_length + q.shape[2]:
        raise ValueError("attention key length must include exactly the declared query prefix")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.numel() == 0:
            raise ValueError("attention tensors must be nonempty")
        base.require_finite(tensor, tuple(tensor.shape), "attention oracle " + name)
    q, k, v = (value.detach().double() for value in (q, k, v))
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    keys = torch.arange(k.shape[2], device=q.device)
    queries = prefix_length + torch.arange(q.shape[2], device=q.device)
    scores = scores.masked_fill(keys[None, :] > queries[:, None], float("-inf"))
    return scores.softmax(-1) @ v


@torch.no_grad()
def attention_probe(q, k, v, observed, *, device, coordinate_limit=8, check_limit=lambda: None, original_layout=None):
    values = [tensor.detach().to(device).transpose(1, 2) for tensor in (q, k, v)]
    if original_layout is not None:
        for index, name in enumerate(("q", "k", "v")):
            layout, value = original_layout[name], values[index]
            if layout["shape"] != list(value.shape) or layout["dtype"] != str(value.dtype):
                raise ValueError("frozen attention original layout shape/dtype differs")
            # Preserve even QKV's storage gaps/offsets so this is a shape replay,
            # rather than an unrecorded contiguity experiment.
            stride, offset = layout["stride"], layout["storage_offset"]
            if len(stride) != 4 or any(type(part) is not int or part < 0 for part in [*stride, offset]):
                raise ValueError("invalid attention operand strides/offset")
            size = offset + 1 + sum((dim - 1) * step for dim, step in zip(value.shape, stride))
            storage = torch.empty(size, dtype=value.dtype, device=device)
            replay = storage.as_strided(value.shape, stride, offset)
            replay.copy_(value)
            values[index] = replay
    q, k, v = values
    check_limit()
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    parts = []
    for index in range(q.shape[2]):
        check_limit()
        parts.append(F.scaled_dot_product_attention(q[:, :, index:index + 1], k[:, :, :index + 1], v[:, :, :index + 1],
                                                   is_causal=index == 0))
    token = torch.cat(parts, 2)
    check_limit()
    oracle = attention_oracle_fp64(q, k, v)
    return {"operands": {name: tensor_identity(value) for name, value in zip(("post_rope_q", "post_rope_k", "v"), (q, k, v))},
            "original_layout": original_layout,
            "replay_layout": {name: {"stride": list(value.stride()), "storage_offset": value.storage_offset(),
                                     "dtype": str(value.dtype), "shape": list(value.shape)}
                              for name, value in zip(("q", "k", "v"), (q, k, v))},
            "identical_operands": True, "mask": "query i sees key positions 0..i, token queries use the same sliced prefix",
            "full_replay_vs_observed": compare(full.transpose(1, 2), observed, coordinate_limit=coordinate_limit),
            "tokenwise_vs_full": compare(token, full, coordinate_limit=coordinate_limit),
            "full_vs_fp64": compare(full, oracle, coordinate_limit=coordinate_limit, oracle=True),
            "tokenwise_vs_fp64": compare(token, oracle, coordinate_limit=coordinate_limit, oracle=True)}


@torch.no_grad()
def frozen_scan_probe(operands, initial, *, device, coordinate_limit=8, check_limit=lambda: None, observed=None,
                      origin="exploratory_supplied_operands"):
    if origin not in ("exploratory_supplied_operands", "actual_model_stateless_operands_zero_state", "actual_model_prefix_continuation"):
        raise ValueError("unknown frozen scan operand/state origin")
    if origin == "actual_model_prefix_continuation" and initial is None:
        raise ValueError("actual prefix continuation requires observed memory")
    operands = tuple(value.detach().to(device) for value in operands)
    x, dt, A, B, C, D = operands
    real_initial = initial is not None
    initial = torch.zeros(x.shape[0], x.shape[2], x.shape[3], B.shape[-1], device=device) if initial is None else initial.detach().to(device).clone()
    before = initial.clone()
    check_limit()
    oracle_y, oracle_state = recurrence_fp64(operands, initial)
    routes = {}
    if not real_initial:
        for name, scan in (("original_quadratic", mamba.ssd), ("candidate_quadratic", decay.ssd_decay_fp64)):
            check_limit()
            value = scan(*operands)
            routes[name] = {"executed": True, "output_vs_fp64": compare(value, oracle_y, coordinate_limit=coordinate_limit, oracle=True),
                            "retained_state": None}
            if observed is not None and name == "candidate_quadratic":
                routes[name]["replay_vs_observed"] = compare(value, observed, coordinate_limit=coordinate_limit)
    else:
        routes.update({name: {"executed": False, "reason": "stateless quadratic scan cannot represent real nonzero prefix state"}
                       for name in ("original_quadratic", "candidate_quadratic")})
    for name, size in (("candidate_one_shot", x.shape[1]), ("candidate_chunked128", 128), ("candidate_tokenwise", 1)):
        memory, outputs = initial.clone(), []
        for start in range(0, x.shape[1], size):
            check_limit()
            out, memory = decay.ssd_stateful_decay_fp64(x[:, start:start + size], dt[:, start:start + size], A,
                                                      B[:, start:start + size], C[:, start:start + size], D, memory)
            outputs.append(out)
        routes[name] = {"executed": True, "output_vs_fp64": compare(torch.cat(outputs, 1), oracle_y, coordinate_limit=coordinate_limit, oracle=True),
                        "retained_state_vs_fp64": compare(memory, oracle_state, coordinate_limit=coordinate_limit, oracle=True)}
        if observed is not None and real_initial and name == "candidate_chunked128":
            routes[name]["replay_vs_observed"] = compare(torch.cat(outputs, 1), observed, coordinate_limit=coordinate_limit)
    if not torch.equal(initial, before):
        raise RuntimeError("frozen scan replay mutated its initial memory")
    return {"operands": {name: tensor_identity(value) for name, value in zip(OPERANDS, operands)},
            "initial": tensor_identity(initial), "initial_nonzero_elements": int(torch.count_nonzero(initial)),
            "initial_condition": "actual model prefix state" if origin == "actual_model_prefix_continuation" else
                                 "zero state on actual stateless projected operands" if origin == "actual_model_stateless_operands_zero_state" else
                                 "supplied exploratory state; not asserted to be a model cache",
            "origin": origin, "synthetic": origin == "exploratory_supplied_operands", "initial_unchanged": True,
            "oracle": "independent FP64 elementwise sequential recurrence",
            "routes": routes, "exploratory": True}


def frozen_mixer_probes(model, full, selections, *, coordinate_limit=8, check_limit=lambda: None):
    result, values = [], full["tensors"]
    device = model.embed.weight.device
    for selection in selections:
        check_limit()
        layer, mixer = selection["layer"], model.blocks[selection["layer"]].mixer
        row = {**selection, "linear": {}}
        names = ("attention_qkv", "attention_projection") if model.blocks[layer].is_attn else ("in_projection", "out_projection")
        modules = (mixer.qkv, mixer.out) if model.blocks[layer].is_attn else (mixer.in_proj, mixer.out_proj)
        if model.blocks[layer].has_mlp:
            names += tuple("mlp_" + name for name in ("gate", "up", "down"))
            modules += tuple(getattr(model.blocks[layer].mlp, name) for name in ("gate", "up", "down"))
        for name, module in zip(names, modules):
            row["linear"][name] = linear_probe(values[(layer, name + ".input")], module, values[(layer, name + ".output")],
                                              coordinate_limit=coordinate_limit, check_limit=check_limit)
        if model.blocks[layer].is_attn:
            original_layout = next(event["operand_layout"] for event in full["attention_events"] if event["layer"] == layer)
            row["attention"] = attention_probe(*(values[(layer, "attention_" + name)] for name in ("query", "key", "value")),
                values[(layer, "attention_output")], device=device, coordinate_limit=coordinate_limit, check_limit=check_limit,
                original_layout=original_layout)
        else:
            z, xBC, dt = values[(layer, "in_projection.output")].split([mixer.d_inner, mixer.conv_dim, mixer.nheads], -1)
            row["convolution"] = convolution_probe(xBC, mixer.conv1d, values[(layer, "convolution_raw")],
                                                   coordinate_limit=coordinate_limit, check_limit=check_limit)
            operands = tuple(values[(layer, "scan_input." + name)] for name in OPERANDS)
            row["scan"] = frozen_scan_probe(operands, None, device=device, observed=values[(layer, "scan_output")],
                                            coordinate_limit=coordinate_limit, check_limit=check_limit,
                                            origin="actual_model_stateless_operands_zero_state")
        result.append(row)
    return result


@torch.no_grad()
def actual_prefix_probes(model, x, prefix_length, selected, *, coordinate_limit=8, check_limit=lambda: None):
    check_limit()
    model.configure_scan_backend("reference", 128)
    _, prefix = model.prefill(x[:, :prefix_length], cache_dtype=torch.float32)
    before = base.state_snapshot(prefix)
    state = prefix.clone()
    trace = TokenTrace(selected_layers={item["layer"] for item in selected}, include_global=False)
    trace.query_length, trace.position = x.shape[1] - prefix_length, prefix_length
    with trace_execution(model, trace), torch.autocast(x.device.type, enabled=False):
        suffix = model(x[:, prefix_length:], inference_state=state)[0]
    tensors = trace.tensors(x.shape[1] - prefix_length, offset=prefix_length)
    result = []
    for selection in selected:
        if selection["kind"] != "mamba":
            continue
        layer = selection["layer"]
        initial = trace.scan_initials[layer]
        if initial is None or not torch.equal(initial, before[f"layer_{layer}.ssm"]):
            raise RuntimeError("observed continuation scan did not use its cloned real prefix memory")
        result.append({"layer": layer, "scan": frozen_scan_probe(tuple(tensors[(layer, "scan_input." + name)] for name in OPERANDS),
            initial, device=x.device, observed=tensors[(layer, "scan_output")], coordinate_limit=coordinate_limit, check_limit=check_limit,
            origin="actual_model_prefix_continuation")})
    after = base.state_snapshot(prefix)
    if prefix.position != prefix_length or state.position != x.shape[1] or any(not torch.equal(value, after[name]) for name, value in before.items()):
        raise RuntimeError("continuation changed the original prefix or final position")
    return {"synthetic": False, "prefix_position": prefix_length, "final_position": state.position,
            "suffix_length": x.shape[1] - prefix_length, "prefix_unchanged": True,
            "prefix_fields": breadth.state_identity(before), "scan_probes": result,
            "suffix_logits": tensor_identity(suffix), "retained_state": breadth.state_identity(base.state_snapshot(state))}


@torch.no_grad()
def study_case(model, x, y, *, prefix_length, coordinate_limit=8, check_limit=lambda: None):
    if (x.ndim != 2 or x.shape != y.shape or x.dtype != torch.int64 or y.dtype != torch.int64
            or min(x.shape) <= 0 or type(prefix_length) is not int or not 0 < prefix_length < x.shape[1]
            or not torch.equal(x[:, 1:], y[:, :-1])):
        raise ValueError("paired shifted int64 tokens and a substantive prefix are required")
    # Validate the coordinate bound before installing any hook or changing state.
    if type(coordinate_limit) is not int or not 1 <= coordinate_limit <= 32:
        raise ValueError("coordinate_limit must be in 1..32")
    weight_identity = base.weight_sha256(model)
    with base.preserve_model_execution(model), base.tf32_disabled(), torch.autocast(x.device.type, enabled=False):
        original = collect_route(model, x, y, check_limit=check_limit)
        check_limit()
        repeated, _ = model(x, y)
        repeat = {"exact_equal": torch.equal(original["logits"], repeated.detach().cpu()),
                  "comparison": compare(repeated, original["logits"], coordinate_limit=coordinate_limit)}
        original_state = model.init_inference_state(x.shape[0], device=x.device, cache_dtype=torch.float32)
        original_stateful, _ = model(x, inference_state=original_state)
        original_memory = base.state_snapshot(original_state)
        del original_state
        with decay.temporary_decay_treatment():
            candidate = collect_route(model, x, y, check_limit=check_limit)
            tokenwise = collect_route(model, x, y, tokenwise=True, check_limit=check_limit)
            state = model.init_inference_state(x.shape[0], device=x.device, cache_dtype=torch.float32)
            check_limit()
            candidate_stateful, _ = model(x, inference_state=state)
            candidate_memory = base.state_snapshot(state)
            own = stage_comparisons(tokenwise, candidate, coordinate_limit, check_limit)
            selected = select_mixers(own["stages"], model)
            # Frozen original scan replay must occur outside the temporary policy.
            prefix = actual_prefix_probes(model, x, prefix_length, selected, coordinate_limit=coordinate_limit, check_limit=check_limit)
        stages = {"candidate_full_vs_original_stateless": stage_comparisons(candidate, original, coordinate_limit, check_limit),
                  "candidate_tokenwise_vs_candidate_full": own,
                  "candidate_tokenwise_vs_original_stateless": stage_comparisons(tokenwise, original, coordinate_limit, check_limit)}
        endpoints = {"candidate_full_vs_original_stateless": compare(candidate["logits"], original["logits"], coordinate_limit=coordinate_limit),
                     "candidate_tokenwise_vs_candidate_full": compare(tokenwise["logits"], candidate["logits"], coordinate_limit=coordinate_limit),
                     "candidate_tokenwise_vs_original_stateless": compare(tokenwise["logits"], original["logits"], coordinate_limit=coordinate_limit),
                     "original_stateful_full_vs_original_stateless": compare(original_stateful, original["logits"], coordinate_limit=coordinate_limit),
                     "candidate_stateful_full_vs_candidate_stateless": compare(candidate_stateful, candidate["logits"], coordinate_limit=coordinate_limit)}
        retained = {"candidate_tokenwise_vs_candidate_stateful_full": state_comparison(tokenwise["state"], candidate_memory, coordinate_limit),
                   "candidate_tokenwise_vs_original_stateful_full": state_comparison(tokenwise["state"], original_memory, coordinate_limit)}
        probes = frozen_mixer_probes(model, candidate, selected, coordinate_limit=coordinate_limit, check_limit=check_limit)
        lm_head = linear_probe(candidate["tensors"][(None, "lm_head.input")], model.lm_head,
            candidate["tensors"][(None, "lm_head.output")], coordinate_limit=coordinate_limit, check_limit=check_limit)
        events = {name: {"scan": route["scan_events"], "attention": route["attention_events"]}
                  for name, route in zip(ROUTES, (original, candidate, tokenwise))}
    if base.weight_sha256(model) != weight_identity:
        raise RuntimeError("isolation changed checkpoint weights")
    report = {"weights_sha256": weight_identity, "whole_model_comparisons": endpoints, "stage_comparisons": stages,
            "retained_state": retained, "original_repeat": repeat, "selected_frozen_mixers": probes,
            "frozen_lm_head": lm_head, "actual_prefix": prefix, "actual_paths": events,
            "final_positions": {"candidate_tokenwise": tokenwise["position"], "candidate_stateful_full": state.position},
            "backward_executed": False, "BF16_executed": False,
            "attribution_control_stable": repeat["exact_equal"],
            "whole_model_passed": all(row["passed"] for row in endpoints.values()) and all(row["passed"] for row in retained.values()),
            "exploratory_components_are_not_whole_model_qualification": True}
    report["all_recorded_comparisons_passed"] = all(comparison_flags(report))
    return report


def declaration_template(*, device="cuda", breadth_declaration=DEFAULT_BREADTH_DECLARATION,
                         breadth_report=DEFAULT_BREADTH_REPORT, shared_inputs=None, output=DEFAULT_OUTPUT):
    """Parent can bind this read-only template and prepared identities before CUDA."""
    prior = base.read_json(breadth_report)["protocol"]
    return {"schema": 1, "status_at_declaration": "planned_before_execution", "device": device, "seed": 2027,
            "ratios": list(RATIOS), "cases": list(CASES), "routes": list(ROUTES), "chunk_size": 128,
            "treatment_id": decay.TREATMENT, "tolerances": base.TOLERANCES,
            "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
            "time_budget_seconds": 900, "minimum_large_case_free_bytes": MINIMUM_FREE_BYTES,
            "coordinate_limit": 8, "maximum_frozen_mixers": 3, "backward_executed": False, "BF16_executed": False,
            "breadth_declaration": {**base.public_reference(breadth_declaration), "sha256": base.file_sha256(breadth_declaration)},
            "breadth_report": {**base.public_reference(breadth_report), "sha256": base.file_sha256(breadth_report)},
            "output": base.public_reference(output)["path"], "source_sha256": source_fingerprints(),
            "data": prior["data"], "checkpoints": [next(item for item in prior["checkpoints"] if item["ratio"] == ratio) for ratio in RATIOS],
            "shared_inputs": shared_inputs}


def source_fingerprints():
    sources = base.source_metadata()
    for name in ("scripts/check_decay_breadth.py", "scripts/study_decay_precision.py",
                 "scripts/isolate_checkpoint_numerics.py", "scripts/isolate_tokenwise_numerics.py"):
        sources[name] = base.file_sha256(ROOT / name)
    return sources


def audit_declaration(declaration, breadth_declaration, breadth_report, device, output=DEFAULT_OUTPUT):
    raw, prior = base.read_json(breadth_report), base.read_json(breadth_declaration)
    planned = base.read_json(declaration)
    expected = declaration_template(device=device, breadth_declaration=breadth_declaration, breadth_report=breadth_report,
                                    shared_inputs=[identity for identity in raw["protocol"]["shared_inputs"] if identity["case_id"] in CASE_IDS], output=output)
    if any(planned.get(key) != value for key, value in expected.items()):
        raise ValueError("tokenwise isolation declaration differs from exact prior evidence and scope")
    if (raw.get("kind") != "trained_checkpoint_decay_breadth" or raw.get("execution_status") != "completed"
            or raw.get("certified") is not False or raw["protocol_sha256"] != base.canonical_sha256(raw["protocol"])
            or raw["protocol"]["declaration"]["sha256"] != base.file_sha256(breadth_declaration)
            or raw["protocol"]["tolerances"] != base.TOLERANCES or raw["protocol"]["shared_inputs"] != prior["shared_inputs"]):
        raise ValueError("prior breadth evidence identity differs")
    # Validate complete source coverage, not just whatever keys a caller supplies.
    required = set(source_fingerprints()) - {"scripts/isolate_tokenwise_numerics.py"}
    if set(raw["protocol"]["source_sha256"]) != required:
        raise ValueError("prior measured source set differs")
    for name, recorded in raw["protocol"]["source_sha256"].items():
        normalized = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if recorded not in {hashlib.sha256(normalized).hexdigest(), hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest()}:
            raise ValueError("prior measured source fingerprint differs: " + name)
    return planned, raw


def run_study(device="cuda", *, declaration=ROOT / "docs/research/tokenwise-isolation-protocol-2026-10-04.json",
              breadth_declaration=DEFAULT_BREADTH_DECLARATION, breadth_report=DEFAULT_BREADTH_REPORT,
              checkpoint_root=Path("checkpoints"), training_run_id="week3-700m-v1",
              data_dir=Path("data/openwebtext-5b"), tokenizer=Path("data/tokenizer/openwebtext.json"), output=DEFAULT_OUTPUT):
    limit = breadth.DiagnosticLimit(900)
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    limit.check()
    if Path(output).exists():
        raise FileExistsError("output already exists; choose a new immutable diagnostic declaration")
    planned, previous = audit_declaration(declaration, breadth_declaration, breadth_report, device, output)
    limit.check()
    checkpoints, data_identity, checkpoint_identity = base.audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, RATIOS)
    limit.check()
    prepared = prepare_inputs(base.load_split(data_dir, "val"))
    if ([row[2] for row in prepared] != planned["shared_inputs"] or data_identity != previous["protocol"]["data"]
            or checkpoint_identity != [next(item for item in previous["protocol"]["checkpoints"] if item["ratio"] == ratio) for ratio in RATIOS]):
        raise ValueError("prepared tokens/data/checkpoints differ from immutable breadth evidence")
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA unavailable; no fallback permitted")
    limit.check()
    sources = source_fingerprints()
    if "source_sha256" in planned and planned["source_sha256"] != sources:
        raise ValueError("declared source fingerprints differ")
    evidence = {Path(path): base.file_sha256(path) for path in (declaration, breadth_declaration, breadth_report)}
    rows, headroom, failure, active_case = [], [], None, None
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), base.tf32_disabled():
        runtime = base.runtime_metadata(device)
        try:
            for ratio in RATIOS:
                active_case = {"ratio": ratio, "case_id": None, "stage": "load model"}
                limit.check()
                model = base.load_variant_model(checkpoints[ratio], torch.device(device))
                try:
                    recorded = next(row for row in previous["cases"] if row["ratio"] == ratio)
                    if base.weight_sha256(model) != recorded["weights_sha256"]:
                        raise ValueError("loaded weights differ from breadth evidence")
                    for x, y, identity in prepared:
                        active_case = {"ratio": ratio, "case_id": identity["case_id"], "stage": "trace and frozen components"}
                        limit.check()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            if identity["batch_size"] == 2:
                                free, total = torch.cuda.mem_get_info()
                                headroom.append({"ratio": ratio, "case_id": identity["case_id"], "free_bytes": free,
                                                 "total_bytes": total, "minimum_free_bytes": MINIMUM_FREE_BYTES})
                                if free < MINIMUM_FREE_BYTES:
                                    raise RuntimeError("insufficient 8GiB large-case headroom; no fallback")
                        started = time.perf_counter()
                        result = study_case(model, x.to(device), y.to(device), prefix_length=identity["prefix_length"], check_limit=limit.check)
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
        except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError) as exc:
            failure = f"{type(exc).__name__}: {exc}".replace(str(ROOT), ".").replace(str(Path.home()), "<home>")
    integrity = {}
    data_files = [(Path(data_dir) / "manifest.json", data_identity["manifest"]["sha256"]),
                  (Path(tokenizer), data_identity["tokenizer"]["sha256"]),
                  *[(Path(data_dir) / Path(item["path"]).name, item["sha256"]) for item in data_identity["artifacts"].values()]]
    for name, entries in (("checkpoints", [(checkpoints[item["ratio"]].best_path, item["sha256"]) for item in checkpoint_identity]),
                          ("prepared_data_and_tokenizer", data_files),
                          ("declarations_and_prior_evidence", list(evidence.items())),
                          ("sources", [(ROOT / name, value) for name, value in sources.items()])):
        try:
            integrity[name] = all(base.file_sha256(path) == sha for path, sha in entries)
        except OSError:
            integrity[name] = False
    if not all(integrity.values()):
        failure = (failure + "; " if failure else "") + "post-execution identity check failed"
    try:
        limit.check()
    except RuntimeError as exc:
        failure = (failure + "; " if failure else "") + str(exc)
        if active_case is None:
            active_case = {"ratio": None, "case_id": None, "stage": "post-execution integrity / allowance"}
    protocol = {**planned, "declaration": {**base.public_reference(declaration), "sha256": evidence[Path(declaration)]},
                "data": data_identity, "checkpoints": checkpoint_identity, "source_sha256": sources,
                "runtime": runtime, "runtime_sha256": base.canonical_sha256(runtime),
                "production_defaults_changed": False, "optimizer_executed": False, "quality_equivalence_margin": None}
    return {"schema": 1, "kind": "trained_checkpoint_tokenwise_isolation", "certified": False,
            "execution_status": "incomplete" if failure else "completed",
            "status": "incomplete" if failure else "completed" if all(row["all_recorded_comparisons_passed"] and row["attribution_control_stable"] for row in rows) else "completed_with_parity_failures",
            "reason": failure, "failed_case": active_case if failure else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
            "protocol_sha256": base.canonical_sha256(protocol), "cases": rows,
            "execution_allowance": {**limit.metadata(), "scope": "from run_study entry, including evidence/data preflight and final integrity; excludes immutable report publication"},
            "large_case_headroom": headroom, "post_execution_integrity": integrity,
            "limits": ["Six declared trained model/shape cases; checkpoint-selection pool, not independent quality.",
                       "Same weights and paired inputs; no backward, optimizer, BF16 or new whole-model precision treatment.",
                       "Original stateless prediction, original stateful end memory and candidate own full route are separate anchors.",
                       "First nonzero rounding drift and first fixed-tolerance violation are recorded separately.",
                       "Frozen identical-input components omit upstream propagation and cannot qualify the whole model.",
                       "FP64 attention uses identical post-RoPE Q/K/V and each query's correct causal key prefix.",
                       "Actual prefix scan operands and memory come from a cloned real candidate prefill; quadratic nonzero-state routes are unexecuted.",
                       "Retained-state comparisons use unchanged FP32 tolerances; no tolerance or quality margin was widened.",
                       "The prior gradient_check input identity is retained, but backward_executed is false in every case.",
                       "Coordinate samples are bounded; no full activation tensor is serialized.",
                       "900 seconds is cooperative; the 8GiB free-headroom guard does not guarantee peak fit. Incomplete cases are not passes."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/tokenwise-isolation-protocol-2026-10-04.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new immutable isolation report")
    if args.declaration.exists():
        expected_output = base.read_json(args.declaration).get("output")
        if expected_output != base.public_reference(args.output)["path"]:
            parser.error("output differs from the pre-execution declaration")
    try:
        report = run_study(args.device, declaration=args.declaration, output=args.output)
    except (ValueError, RuntimeError, KeyError, OSError, FloatingPointError) as exc:
        reason = f"{type(exc).__name__}: {exc}".replace(str(ROOT), ".").replace(str(Path.home()), "<home>")
        report = {"schema": 1, "kind": "trained_checkpoint_tokenwise_isolation", "certified": False,
                  "execution_status": "incomplete", "status": "incomplete", "reason": reason, "cases": []}
    base.write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "execution_status", "certified")}
                     | {"completed_cases": len(report["cases"])}))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
