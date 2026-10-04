"""Bounded, same-weight precision diagnostics; no training or backend certification.

This study preserves the gate's tolerances and token/target RNG draw order. It
changes precision only on copied model instances, and reports failed comparisons
without converting them into a pass. Example:

    python scripts/study_scan_precision.py --device cuda --ratio 0:1 --seed 1337
        --output docs/research/checks/precision-pure-mamba-seed1337.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scripts.check_scan_backend import TOLERANCES, compare  # noqa: E402
from scripts.count_params import count  # noqa: E402
from src.model.config import ModelConfig  # noqa: E402
from src.model.lm import HybridLM  # noqa: E402
from src.model.mamba2 import ssd, ssd_stateful  # noqa: E402
from src.model.scan_backend import BackendUnavailableError, resolve_scan_backend  # noqa: E402

TREATMENTS = (
    {"id": "fp32_reference", "bf16": False, "scan_fp32": False, "projection_fp32": False},
    {"id": "bf16_original", "bf16": True, "scan_fp32": False, "projection_fp32": False},
    {"id": "bf16_scan_fp32", "bf16": True, "scan_fp32": True, "projection_fp32": False},
    {"id": "bf16_projection_fp32", "bf16": True, "scan_fp32": False, "projection_fp32": True},
    {"id": "bf16_scan_and_projection_fp32", "bf16": True, "scan_fp32": True, "projection_fp32": True},
)


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def weight_sha256(model: HybridLM) -> str:
    identity = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                 "sha256": tensor_sha256(value)} for name, value in model.state_dict().items()]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _precision(device: str, bf16: bool):
    return torch.autocast(device, dtype=torch.bfloat16) if bf16 else torch.autocast(device, enabled=False)


class ScanObserver:
    """Per-instance scan wrapper; no process-wide patch or production default change."""

    def __init__(self, backend, layer: int, recorder: dict, disable_autocast: bool):
        self.backend, self.layer = backend, layer
        self.recorder, self.disable_autocast = recorder, disable_autocast
        self.name, self.chunk_size = backend.name, backend.chunk_size

    def scan(self, *args, **kwargs):
        stage = self.recorder["stage"]
        device = args[0].device.type
        for key, path in (("reference_scan", "torch.quadratic_ssd"),
                          ("stateful_scan", "torch.chunked_ssd")):
            function = kwargs[key]

            def observed(*values, _function=function, _path=path):
                record = {"stage": stage, "layer": self.layer, "path": _path,
                          "autocast": torch.is_autocast_enabled(device),
                          "input_dtypes": [str(value.dtype) for value in values],
                          "length": values[0].shape[1]}
                result = _function(*values)
                tensors = result if isinstance(result, tuple) else (result,)
                record["output_dtypes"] = [str(value.dtype) for value in tensors]
                identity = json.dumps(record, sort_keys=True)
                self.recorder["events"][identity] = self.recorder["events"].get(identity, 0) + 1
                return result

            kwargs[key] = observed
        context = torch.autocast(device, enabled=False) if self.disable_autocast else nullcontext()
        with context:
            result = self.backend.scan(*args, **kwargs)
        if stage in ("full", "token_decode"):
            self.recorder["scan_outputs"].setdefault((self.layer, stage), []).append(result[0].detach().clone())
        if self.layer == self.recorder["first_mamba_layer"] and stage == "full" and "scan_inputs" not in self.recorder:
            self.recorder["scan_inputs"] = tuple(value.detach().clone() for value in args[:6])
        return result


@contextmanager
def experimental_precision(model: HybridLM, treatment: dict, recorder: dict):
    """Restore experimental wrappers and hooks even when a diagnostic raises."""
    restored, hooks = [], []
    first_mamba = next((block.mixer for block in model.blocks if not block.is_attn), None)
    try:
        for index, block in enumerate(model.blocks):
            if not block.is_attn:
                mixer, original = block.mixer, block.mixer.scan_backend
                restored.append((mixer, "scan_backend", original, False))
                mixer.scan_backend = ScanObserver(original, index, recorder, treatment["scan_fp32"])
        projections = [model.lm_head] + [block.mixer.out_proj for block in model.blocks if not block.is_attn]
        for module in projections:
            if treatment["projection_fp32"]:
                had_own_forward = "forward" in module.__dict__
                original = module.forward
                restored.append((module, "forward", original, not had_own_forward))

                def fp32_forward(value, _module=module, _forward=original):
                    with torch.autocast(value.device.type, enabled=False):
                        return _forward(value.to(_module.weight.dtype))

                module.forward = fp32_forward
        for name, module in (("mixer_projection", None if first_mamba is None else first_mamba.out_proj),
                             ("lm_projection", model.lm_head)):
            if module is None:
                continue

            def capture(_module, inputs, _name=name):
                stage = recorder["stage"]
                if stage in ("full", "token_decode"):
                    recorder["projection_inputs"].setdefault((_name, stage), []).append(inputs[0].detach().clone())

            hooks.append(module.register_forward_pre_hook(capture))
        for index, block in enumerate(model.blocks):
            if block.is_attn:
                continue

            def gate_input(_module, inputs, _index=index):
                if recorder["stage"] in ("full", "token_decode"):
                    recorder["layer_stages"].setdefault((_index, "gate_norm", recorder["stage"]), []).append(inputs[0].detach().clone())

            def mixer_output(_module, _inputs, output, _index=index):
                if recorder["stage"] in ("full", "token_decode"):
                    recorder["layer_stages"].setdefault((_index, "mixer_output", recorder["stage"]), []).append(output.detach().clone())

            hooks.append(block.mixer.out_proj.register_forward_pre_hook(gate_input))
            hooks.append(block.mixer.register_forward_hook(mixer_output))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for module, name, original, remove in reversed(restored):
            if remove:
                delattr(module, name)
            else:
                setattr(module, name, original)


def _memory(state) -> dict[str, torch.Tensor]:
    result = {}
    for layer, value in enumerate(state.layers):
        for field in (("conv", "ssm") if hasattr(value, "ssm") else ("key", "value")):
            tensor = getattr(value, field)
            if tensor is not None:
                result[f"layer_{layer}.{field}"] = tensor.detach().clone()
    return result


def compare_memory(actual: dict, expected: dict, tolerance: dict) -> dict:
    if actual.keys() != expected.keys():
        raise ValueError("retained-state field identities differ")
    fields = [{"field": name, **compare(value, expected[name], tolerance)}
              for name, value in actual.items() if value.numel()]
    return {"passed": all(field["passed"] for field in fields), "fields": fields}


def _assert_finite(value: torch.Tensor, shape: tuple, label: str) -> None:
    if value.shape != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite with shape {shape}; received {tuple(value.shape)}")


def _scan_sequence(operands: tuple, initial: torch.Tensor, chunk: int):
    x, dt, A, B, C, D = operands
    outputs, carried = [], initial
    for start in range(0, x.shape[1], chunk):
        stop = start + chunk
        output, carried = ssd_stateful(x[:, start:stop], dt[:, start:stop], A,
                                      B[:, start:stop], C[:, start:stop], D, carried)
        outputs.append(output)
    return torch.cat(outputs, dim=1), carried


def recurrence_oracle(operands: tuple, initial: torch.Tensor):
    """Independent sequential update using elementwise products/sums, not SSD einsums."""
    x, dt, A, B, C, D = (value.float() for value in operands)
    memory, outputs = initial.float(), []
    for index in range(x.shape[1]):
        decay = torch.exp(dt[:, index] * A)[..., None, None]
        drive = (x[:, index] * dt[:, index, :, None])[..., None] * B[:, index, None, None, :]
        memory = decay * memory + drive
        outputs.append((memory * C[:, index, None, None, :]).sum(-1) + x[:, index] * D[None, :, None])
    return torch.stack(outputs, dim=1), memory


def gradient_isolation(operands: tuple, initial: torch.Tensor, chunk_size: int) -> dict:
    """Direct scan derivatives and exact zero future gradients; no optimizer/update."""
    names = ("x", "dt", "A", "B", "C", "D", "initial_state")

    def leaves():
        return tuple(value.detach().clone().requires_grad_(True) for value in (*operands, initial))

    def objective(output, final):
        weights = torch.linspace(-0.5, 0.5, output.numel(), device=output.device).reshape(output.shape)
        return (output.float() * weights).mean() + 0.1 * final.float().mean()

    # Re-enable only this bounded derivative probe, even when its caller runs under no_grad.
    with torch.enable_grad(), torch.autocast(operands[0].device.type, enabled=False):
        reference = leaves()
        expected_output, expected_state = recurrence_oracle(reference[:6], reference[6])
        expected_gradients = torch.autograd.grad(objective(expected_output, expected_state), reference)
    cases = []
    for bf16 in (True, False):
        with torch.enable_grad(), _precision(operands[0].device.type, bf16):
            values = leaves()
            output, final = _scan_sequence(values[:6], values[6], chunk_size)
            gradients = torch.autograd.grad(objective(output, final), values)
            causality = None
            if output.shape[1] > 1:
                future = leaves()
                earlier, _ = _scan_sequence(future[:6], future[6], chunk_size)
                future_gradients = torch.autograd.grad(earlier[:, :-1].float().sum(), future)
                checks = [{"input": names[index], "finite": bool(torch.isfinite(future_gradients[index]).all()),
                           "max_future_absolute_gradient": future_gradients[index][:, -1].float().abs().max().item(),
                           "passed": bool(torch.isfinite(future_gradients[index]).all()
                                          and (future_gradients[index][:, -1] == 0).all())}
                          for index in (0, 1, 3, 4)]
                causality = {"passed": all(check["passed"] for check in checks), "inputs": checks}
        tolerance = TOLERANCES["bfloat16" if bf16 else "float32"]
        checks = {name: compare(actual, expected, tolerance)
                  for name, actual, expected in zip(names, gradients, expected_gradients)}
        cases.append({"autocast": bf16, "tolerance": dict(tolerance), "checks": checks,
                      "future_gradient_causality": causality,
                      "passed": all(check["passed"] for check in checks.values())
                                and (causality is None or causality["passed"])})
    return {"scope": "exploratory direct scan derivatives versus independent FP32 recurrence; final state contributes to loss",
            "cases": cases}


@torch.no_grad()
def scan_isolation(operands: tuple, chunk_size: int, seed: int) -> dict:
    """Same projected operands isolate scan arithmetic from every linear projection."""
    x, _, _, B, _, _ = operands
    generator = torch.Generator(device="cpu").manual_seed(seed + 104729)
    shape = (x.shape[0], x.shape[2], x.shape[3], B.shape[-1])
    nonzero = torch.randn(shape, generator=generator).to(x.device)
    with torch.autocast(x.device.type, enabled=False):
        oracle_zero, oracle_zero_state = recurrence_oracle(operands, torch.zeros_like(nonzero))
        oracle_nonzero, oracle_nonzero_state = recurrence_oracle(operands, nonzero)
    cases = []
    for bf16 in (True, False):
        tolerance = TOLERANCES["bfloat16" if bf16 else "float32"]
        with _precision(x.device.type, bf16):
            cb_dtype = str(torch.einsum("bin,bjn->bij", operands[4].float(), operands[3].float()).dtype)
            full = ssd(*operands)
            zero = torch.zeros_like(nonzero)
            one_shot, _ = _scan_sequence(operands, zero, x.shape[1])
            bounded, final_bounded = _scan_sequence(operands, zero, chunk_size)
            tokenwise, final_tokenwise = _scan_sequence(operands, zero, 1)
            continued, final_continued = _scan_sequence(operands, nonzero, x.shape[1])
            continued_chunks, final_continued_chunks = _scan_sequence(operands, nonzero, chunk_size)
            continued_tokens, final_continued_tokens = _scan_sequence(operands, nonzero, 1)
            causality = None
            if x.shape[1] > 1:
                changed = list(operands)
                changed[0] = x.clone()
                changed[0][:, -1] += 1
                future_changed = ssd(*changed)
                causality = compare(future_changed[:, :-1], full[:, :-1], tolerance)
        checks = {
            "zero_one_shot_vs_full": compare(one_shot, full, tolerance),
            "zero_bounded_vs_full": compare(bounded, full, tolerance),
            "zero_tokenwise_vs_full": compare(tokenwise, full, tolerance),
            "zero_final_state": compare(final_tokenwise, final_bounded, tolerance),
            "nonzero_bounded_output": compare(continued_chunks, continued, tolerance),
            "nonzero_tokenwise_output": compare(continued_tokens, continued, tolerance),
            "nonzero_bounded_state": compare(final_continued_chunks, final_continued, tolerance),
            "nonzero_tokenwise_state": compare(final_continued_tokens, final_continued, tolerance),
            "zero_output_vs_recurrence_oracle": compare(bounded, oracle_zero, tolerance),
            "zero_state_vs_recurrence_oracle": compare(final_bounded, oracle_zero_state, tolerance),
            "nonzero_output_vs_recurrence_oracle": compare(continued_chunks, oracle_nonzero, tolerance),
            "nonzero_state_vs_recurrence_oracle": compare(final_continued_chunks, oracle_nonzero_state, tolerance),
        }
        if causality is not None:
            checks["causality"] = causality
        cases.append({"autocast": bf16, "tolerance": dict(tolerance), "checks": checks,
                      "contraction_inputs_dtype": "torch.float32", "cb_contraction_output_dtype": cb_dtype,
                      "full_scan_output_dtype": str(full.dtype), "retained_state_dtype": str(final_bounded.dtype),
                      "passed": all(check["passed"] for check in checks.values())})
    return {"scope": "exploratory same-operand scan; nonzero continuation is synthetic",
            "operand_sha256": dict(zip(("x", "dt", "A", "B", "C", "D"), (tensor_sha256(value) for value in operands))),
            "initial_state_sha256": tensor_sha256(nonzero), "cases": cases,
            "direct_gradients": gradient_isolation(operands, nonzero, chunk_size)}


@torch.no_grad()
def projection_isolation(module, inputs: torch.Tensor, chunk_size: int) -> dict:
    """Same hidden vectors isolate precision changes caused by GEMM batch shape."""
    cases = []
    for bf16 in (True, False):
        with _precision(inputs.device.type, bf16):
            values = inputs if bf16 else inputs.to(module.weight.dtype)
            full = F.linear(values, module.weight, module.bias)
            bounded = torch.cat([F.linear(values[:, start:start + chunk_size], module.weight, module.bias)
                                 for start in range(0, values.shape[1], chunk_size)], dim=1)
            tokenwise = torch.cat([F.linear(values[:, index:index + 1], module.weight, module.bias)
                                   for index in range(values.shape[1])], dim=1)
        tolerance = TOLERANCES["bfloat16" if bf16 else "float32"]
        checks = {"bounded_vs_full": compare(bounded, full, tolerance),
                  "tokenwise_vs_full": compare(tokenwise, full, tolerance)}
        cases.append({"autocast": bf16, "tolerance": dict(tolerance), "checks": checks,
                      "passed": all(check["passed"] for check in checks.values())})
    return {"scope": "exploratory same-hidden-input projection; scan and weights fixed",
            "input_sha256": tensor_sha256(inputs), "cases": cases}


@torch.no_grad()
def _evaluate(model: HybridLM, tokens: torch.Tensor, treatment: dict, chunk_size: int) -> tuple[dict, dict]:
    recorder = {"stage": "full", "events": {}, "projection_inputs": {}, "scan_outputs": {}, "layer_stages": {},
                "first_mamba_layer": next(index for index, block in enumerate(model.blocks) if not block.is_attn)}
    device = tokens.device.type
    cache_dtype = torch.bfloat16 if treatment["bf16"] else torch.float32
    length, batch = tokens.shape[1], tokens.shape[0]
    logits, memories = {}, {}
    with experimental_precision(model, treatment, recorder), _precision(device, treatment["bf16"]):
        logits["full"], _ = model(tokens)
        recorder["stage"] = "bounded_prefill"
        logits["bounded_prefill"], state = model.prefill(tokens, cache_dtype=cache_dtype)
        memories["bounded_prefill"] = _memory(state)
        split = max(1, length - 3)
        recorder["stage"] = "gate_prefill"
        logits["gate_prefill"], state = model.prefill(tokens[:, :split], cache_dtype=cache_dtype)
        recorder["stage"] = "gate_decode"
        decoded = [model.decode(tokens[:, index:index + 1], state) for index in range(split, length)]
        if decoded:
            logits["gate_decode"] = torch.cat(decoded, dim=1)
        memories["gate_final"] = _memory(state)
        recorder["stage"] = "token_decode"
        state = model.init_inference_state(batch, device=tokens.device, cache_dtype=cache_dtype)
        logits["token_decode"] = torch.cat([model.decode(tokens[:, index:index + 1], state)
                                             for index in range(length)], dim=1)
        memories["token_decode"] = _memory(state)
        if state.position != length:
            raise ValueError("tokenwise final inference position differs from input length")
        recorder["stage"] = "causality"
        causality = None
        if length > 1:
            future_tokens = tokens.clone()
            future_tokens[:, -1] = (future_tokens[:, -1] + 1) % model.cfg.vocab_size
            future_logits, _ = model(future_tokens)
            causality = compare(future_logits[:, :-1], logits["full"][:, :-1],
                                TOLERANCES["bfloat16" if treatment["bf16"] else "float32"])
    for name, value in logits.items():
        expected_length = length if name in ("full", "token_decode") else length - split if name == "gate_decode" else 1
        _assert_finite(value, (batch, expected_length, model.cfg.vocab_size), f"{name} logits")
    for memory in memories.values():
        for name, value in memory.items():
            _assert_finite(value, tuple(value.shape), f"retained {name}")
    snapshot = {"logits": logits, "memories": memories, "split": split, "causality": causality}
    return snapshot, recorder


def _comparisons(snapshot: dict, tolerance: dict, anchor: dict | None) -> dict:
    logits, split = snapshot["logits"], snapshot["split"]
    internal = {
        "bounded_prefill_vs_full": compare(logits["bounded_prefill"], logits["full"][:, -1:], tolerance),
        "gate_prefill_vs_full": compare(logits["gate_prefill"], logits["full"][:, split - 1:split], tolerance),
        "token_decode_vs_full": compare(logits["token_decode"], logits["full"], tolerance),
    }
    if "gate_decode" in logits:
        internal["gate_decode_vs_full"] = compare(logits["gate_decode"], logits["full"][:, split:], tolerance)
    if snapshot["causality"] is not None:
        internal["causality"] = snapshot["causality"]
    memory = {name: compare_memory(values, snapshot["memories"]["bounded_prefill"], tolerance)
              for name, values in snapshot["memories"].items() if name != "bounded_prefill"}
    result = {"internal_consistency": internal, "internal_retained_state": memory,
              "internal_passed": all(check["passed"] for check in [*internal.values(), *memory.values()])}
    if anchor is not None:
        anchor_logits = {name: compare(value, anchor["logits"][name], tolerance) for name, value in logits.items()}
        anchor_memory = {name: compare_memory(values, anchor["memories"][name], tolerance)
                         for name, values in snapshot["memories"].items()}
        result["fp32_anchor"] = {"logits": anchor_logits, "retained_state": anchor_memory,
                                 "passed": all(check["passed"] for check in [*anchor_logits.values(), *anchor_memory.values()])}
    return result


def _source_metadata() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "src/model").glob("*.py")), Path(__file__), root / "scripts/check_scan_backend.py"]
    return {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _load_model(seed: int, ratio: str, device: str, checkpoint: Path | None):
    cfg = ModelConfig(ratio=ratio, d_model=64, n_layers=4, vocab_size=128, head_dim=16,
                      mamba_headdim=16, d_state=16, mlp_multiple_of=16)
    identity = None
    loaded = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        repository_root = Path(__file__).resolve().parents[1]
        try:
            checkpoint_reference = checkpoint.relative_to(repository_root).as_posix()
            reference_scope = "repository-relative"
        except ValueError:
            checkpoint_reference = checkpoint.name
            reference_scope = "external basename; content identified by SHA256"
        identity = {"path": checkpoint_reference, "path_scope": reference_scope,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "mode": "read-only; weights_only=True; no checkpoint writes"}
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("model_config"), dict):
            raise ValueError("checkpoint requires model_config and model tensors")
        cfg = ModelConfig(**loaded["model_config"])
    if cfg.n_mamba_layers == 0:
        raise ValueError("precision scan study requires at least one Mamba layer")
    if count(cfg)["total"] > 100_000_000:
        raise ValueError("precision diagnostics are bounded to models with at most 100M parameters")
    model = HybridLM(cfg)
    if loaded is not None:
        model.load_state_dict(loaded["model"], strict=True)
    if model.num_params() > 100_000_000:
        raise ValueError("precision diagnostics are bounded to models with at most 100M parameters")
    return model.to(device).eval(), identity


def run_study(device: str = "cpu", ratio: str = "0:1", seed: int = 1337,
              lengths: list[int] | None = None, chunk_size: int = 16, batch_size: int = 2,
              backend: str = "reference", checkpoint: Path | None = None) -> dict:
    lengths = [1, 15, 16, 17, 33] if lengths is None else lengths
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between zero and 2^32 - 1")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 4:
        raise ValueError("batch_size must be between one and four")
    if (not lengths or len(lengths) > 12 or len(set(lengths)) != len(lengths)
            or any(not isinstance(length, int) or isinstance(length, bool) or not 1 <= length <= 512 for length in lengths)
            or sum(lengths) * batch_size > 4096):
        raise ValueError("use unique lengths 1..512, at most twelve cases and 4096 batch-token positions")
    if backend not in ("reference", "torch_chunked"):
        raise ValueError("this isolation supports portable reference/torch_chunked only; it cannot certify fused scans")
    resolve_scan_backend(backend, chunk_size)
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA device is unavailable")
    sources = _source_metadata()
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        # Seed only the streams preserved by this context. CPU diagnostics must not change
        # a caller's CUDA RNG, and multi-GPU runs must not reseed unrelated devices.
        torch.random.default_generator.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        model, checkpoint_identity = _load_model(seed, ratio, device, checkpoint)
        weights = weight_sha256(model)
        inputs = []
        # Exactly the original gate order: initialize one model, draw tokens then targets for
        # each length. Deepcopy and all diagnostics below consume no input RNG stream.
        for length in lengths:
            tokens = torch.randint(0, model.cfg.vocab_size, (batch_size, length), device=device)
            targets = torch.randint(0, model.cfg.vocab_size, (batch_size, length), device=device)
            inputs.append((tokens, {"length": length, "tokens_sha256": tensor_sha256(tokens),
                                    "targets_sha256": tensor_sha256(targets)}))
        anchors, cases, probes = {}, [], []
        for treatment in TREATMENTS:
            candidate = deepcopy(model)
            selected_backend = "reference" if treatment["id"] == "fp32_reference" else backend
            metadata = candidate.configure_scan_backend(selected_backend, chunk_size)
            for tokens, input_identity in inputs:
                snapshot, recorder = _evaluate(candidate, tokens, treatment, chunk_size)
                tolerance = TOLERANCES["bfloat16" if treatment["bf16"] else "float32"]
                if treatment["id"] == "fp32_reference":
                    anchors[tokens.shape[1]] = snapshot
                checks = _comparisons(snapshot, tolerance, None if treatment["id"] == "fp32_reference" else anchors[tokens.shape[1]])
                hidden = {}
                for projection in ("mixer_projection", "lm_projection"):
                    full = torch.cat(recorder["projection_inputs"][(projection, "full")], dim=1)
                    recurrent = torch.cat(recorder["projection_inputs"][(projection, "token_decode")], dim=1)
                    hidden[projection] = compare(recurrent, full, tolerance)
                layer_checks = []
                for layer, block in enumerate(candidate.blocks):
                    if block.is_attn:
                        continue
                    for stage in ("scan_output", "gate_norm", "mixer_output"):
                        values = recorder["scan_outputs"] if stage == "scan_output" else recorder["layer_stages"]
                        full_key = (layer, "full") if stage == "scan_output" else (layer, stage, "full")
                        decode_key = (layer, "token_decode") if stage == "scan_output" else (layer, stage, "token_decode")
                        layer_checks.append({"layer": layer, "stage": stage,
                                             **compare(torch.cat(values[decode_key], dim=1), torch.cat(values[full_key], dim=1), tolerance)})
                events = [{**json.loads(identity), "calls": count} for identity, count in recorder["events"].items()]
                observed = {stage: sorted({event["path"] for event in events if event["stage"] == stage})
                            for stage in sorted({event["stage"] for event in events})}
                cases.append({**input_identity, "treatment": treatment["id"], "precision_policy": dict(treatment),
                              "weights_sha256": weights, "tolerance": dict(tolerance),
                              "backend": metadata, "observed_paths": observed, "operator_observations": events,
                              "comparisons": checks, "projection_input_consistency": hidden,
                              "layer_stage_consistency": layer_checks,
                              "first_failed_layer_stage": next(({"layer": row["layer"], "stage": row["stage"]}
                                                                 for row in layer_checks if not row["passed"]), None),
                              "passed": checks["internal_passed"] and checks.get("fp32_anchor", {}).get("passed", True)
                                        and all(row["passed"] for row in layer_checks)
                                        and all(row["passed"] for row in hidden.values())})
                if treatment["id"] == "bf16_original":
                    first = next(block.mixer for block in candidate.blocks if not block.is_attn)
                    probes.append({"length": tokens.shape[1], "scan": scan_isolation(recorder["scan_inputs"], chunk_size, seed),
                                   "projections": {
                                       "mixer_output": projection_isolation(first.out_proj, recorder["projection_inputs"][("mixer_projection", "full")][0], chunk_size),
                                       "lm_head": projection_isolation(candidate.lm_head, recorder["projection_inputs"][("lm_projection", "full")][0], chunk_size),
                                   }})
            if weight_sha256(candidate) != weights:
                raise RuntimeError("an experimental treatment changed model weights")
            del candidate
        if weight_sha256(model) != weights:
            raise RuntimeError("the study changed its source model weights")
    versions = {}
    for package in ("torch", "numpy", "mamba-ssm", "triton", "causal-conv1d"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {"version": 1, "seed": seed, "device": device, "model_config": asdict(model.cfg),
                "weights_sha256": weights, "checkpoint": checkpoint_identity, "inputs": [identity for _, identity in inputs],
                "chunk_size": chunk_size, "batch_size": batch_size, "backend": backend,
                "treatments": TREATMENTS, "tolerances": deepcopy(TOLERANCES), "source_sha256": sources,
                "input_policy": "original gate RNG order; target draws retained; no treatment consumes input RNG"}
    precision_flags = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    protocol["runtime"] = {"packages": versions, "python": platform.python_version(),
                           "torch_cuda_runtime": torch.version.cuda, "precision_flags": precision_flags,
                           "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                           "cuda_capability": list(torch.cuda.get_device_capability()) if device == "cuda" else None}
    finite_checks_pass = all(case["passed"] for case in cases)
    probe_checks_pass = all(probe_case["passed"] for probe in probes for group in [probe["scan"], probe["scan"]["direct_gradients"], *probe["projections"].values()]
                            for probe_case in group["cases"])
    return {"schema": 1, "status": "completed" if finite_checks_pass and probe_checks_pass else "completed_with_parity_failures",
            "execution_status": "completed", "certified": False,
            "scope": "bounded precision isolation; no training, quality claims or backend certification",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
            "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
            "packages": versions, "python": platform.python_version(), "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "precision_flags": precision_flags,
            "cases": cases, "exploratory_probes": probes,
            "limits": ["Common FP32-anchor differences and full/cached consistency are separate questions.",
                       "Disabled autocast is an experimental treatment; production defaults are preserved.",
                       "Projection replay fixes hidden values; it does not exclude amplification of earlier differences.",
                       "Direct frozen-operand scan gradients are exploratory; full-model/trained-checkpoint gradient parity remains untested.",
                       "No language-quality, speed or fused-backend certification.",
                       "Hybrid/pure models consume initialization RNG differently; shared seeds do not isolate an attention stabilization mechanism.",
                       "Zero-init model paths and synthetic nonzero scan probes do not certify all states or checkpoints."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ratio", default="0:1")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 15, 16, 17, 33])
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--backend", choices=("reference", "torch_chunked"), default="reference")
    parser.add_argument("--checkpoint", type=Path, help="optional historical checkpoint, read-only")
    parser.add_argument("--output", type=Path, help="new diagnostic artifact; existing paths are never overwritten")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new study artifact")
    try:
        report = run_study(args.device, args.ratio, args.seed, args.lengths, args.chunk_size,
                           args.batch_size, args.backend, args.checkpoint)
    except (RuntimeError, ValueError, KeyError) as exc:
        report = {"schema": 1, "status": "unavailable" if isinstance(exc, BackendUnavailableError) else "failed",
                  "execution_status": "incomplete", "certified": False, "reason": f"{type(exc).__name__}: {exc}",
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "arguments": {**vars(args),
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None, "output": str(args.output) if args.output else None}}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    print(encoded, end="")
    # Informative parity failures are completed diagnostics; unavailable/incomplete execution fails.
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
