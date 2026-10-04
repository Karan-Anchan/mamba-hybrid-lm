"""Bounded trained-weight numerical diagnostics on the checkpoint-selection val pool.

No optimizer is created. Original tensor tolerances are retained; next-token
probability changes are descriptive and have no invented quality-equivalence margin.
Example: python scripts/study_checkpoint_numerics.py --device cuda --output NEW.json
Use --length 257 for a real 128-token prefix followed by a 129-token continuation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from scripts.check_scan_backend import TOLERANCES, compare  # noqa: E402
from scripts.count_params import count  # noqa: E402
from src.data.dataset import load_split  # noqa: E402
from src.data.prepare_data import validate_prepared_dataset  # noqa: E402
from src.eval.suite import (WEEK3_RATIOS, discover_checkpoints, file_sha256,
                            load_variant_model, read_json)  # noqa: E402
from src.model.config import ModelConfig  # noqa: E402
from src.model.scan_backend import BackendUnavailableError, resolve_scan_backend  # noqa: E402

KIND = "trained_checkpoint_numerics"


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def public_reference(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        return {"path": path.relative_to(ROOT).as_posix(), "path_scope": "repository-relative"}
    except ValueError:
        return {"path": path.name, "path_scope": "external basename; content identified by SHA256"}


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def weight_sha256(model) -> str:
    return canonical_sha256({name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                   "sha256": tensor_sha256(value)}
                             for name, value in model.state_dict().items()})


def require_finite(value: torch.Tensor, shape: tuple, label: str) -> None:
    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite with shape {shape}; received {tuple(value.shape)}")


@contextmanager
def tf32_disabled():
    """Restore both backend flags even on failure; avoid mixing precision APIs."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class ObservedScan:
    """Record the actual local scan callbacks without changing their arithmetic."""

    def __init__(self, backend, layer: int, recorder: dict):
        self.backend, self.layer, self.recorder = backend, layer, recorder
        self.name, self.chunk_size = backend.name, backend.chunk_size

    def scan(self, *args, **kwargs):
        for key, path in (("reference_scan", "torch.quadratic_ssd"),
                          ("stateful_scan", "torch.chunked_ssd")):
            function = kwargs[key]

            def observed(*values, _function=function, _path=path):
                identity = json.dumps({"stage": self.recorder["stage"], "layer": self.layer,
                                       "path": _path, "length": values[0].shape[1],
                                       "autocast": torch.is_autocast_enabled(values[0].device.type),
                                       "input_dtypes": [str(value.dtype) for value in values]}, sort_keys=True)
                self.recorder["events"][identity] = self.recorder["events"].get(identity, 0) + 1
                return _function(*values)

            kwargs[key] = observed
        return self.backend.scan(*args, **kwargs)


@contextmanager
def observe_scans(model, recorder: dict):
    originals = []
    try:
        for layer, block in enumerate(model.blocks):
            if not block.is_attn:
                originals.append((block.mixer, block.mixer.scan_backend))
                block.mixer.scan_backend = ObservedScan(block.mixer.scan_backend, layer, recorder)
        yield
    finally:
        for mixer, backend in originals:
            mixer.scan_backend = backend


@contextmanager
def preserve_model_execution(model):
    modes = [(module, module.training) for module in model.modules()]
    backends = [(block.mixer, block.mixer.scan_backend) for block in model.blocks if not block.is_attn]
    model.eval()
    try:
        yield
    finally:
        model.zero_grad(set_to_none=True)
        for mixer, backend in backends:
            mixer.scan_backend = backend
        for module, training in modes:
            module.training = training


def shared_windows(data: np.ndarray, length: int, windows: int, seed: int) -> list[tuple]:
    """Choose starts once on a private CPU generator; y is strictly x shifted by one."""
    if len(data) <= length:
        raise ValueError("validation stream must contain length+1 tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(len(data) - length, (windows,), generator=generator).tolist()
    rows = []
    for index, start in enumerate(starts):
        values = torch.from_numpy(np.asarray(data[start:start + length + 1], dtype=np.int64).copy())
        x, y = values[:-1].unsqueeze(0).contiguous(), values[1:].unsqueeze(0).contiguous()
        rows.append((x, y, {"window": index, "start_token": start, "length": length,
                            "tokens_sha256": tensor_sha256(x), "targets_sha256": tensor_sha256(y),
                            "target_policy": "y[i] = validation[start_token+i+1]"}))
    return rows


def distribution(value: torch.Tensor) -> dict:
    value = value.detach().double().flatten().cpu()
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("descriptive values must be nonempty and finite")
    return {"mean": value.mean().item(), "minimum": value.min().item(), "maximum": value.max().item(),
            "mean_absolute": value.abs().mean().item(),
            "p95_absolute": torch.quantile(value.abs(), 0.95).item()}


def token_effects(actual: torch.Tensor, expected: torch.Tensor, targets: torch.Tensor,
                  position_start: int = 0) -> dict:
    """All metric arithmetic uses FP32 log_softmax on correctly shifted targets."""
    if actual.ndim != 3 or actual.shape != expected.shape or targets.shape != actual.shape[:2]:
        raise ValueError("logits/shifted-target shapes must agree")
    if targets.dtype != torch.int64 or bool((targets < 0).any() or (targets >= actual.shape[-1]).any()):
        raise ValueError("shifted targets must be int64 vocabulary IDs")
    require_finite(actual, tuple(actual.shape), "actual logits")
    require_finite(expected, tuple(actual.shape), "reference logits")
    actual, expected, targets = actual.detach().float().cpu(), expected.detach().float().cpu(), targets.cpu()
    actual_lp = torch.log_softmax(actual, -1).gather(-1, targets[..., None]).squeeze(-1)
    reference_lp = torch.log_softmax(expected, -1).gather(-1, targets[..., None]).squeeze(-1)
    actual_top, actual_ids = actual.topk(2, dim=-1)
    reference_top, reference_ids = expected.topk(2, dim=-1)
    agree = actual_ids[..., 0] == reference_ids[..., 0]
    delta = actual_lp - reference_lp
    return {"scope": "descriptive arithmetic effects; no quality-equivalence cutoff",
            "scored_tokens": targets.numel(), "position_start": position_start,
            "position_stop_exclusive": position_start + targets.shape[1],
            "metric_dtype": "torch.float32", "actual_mean_nll": -actual_lp.mean().item(),
            "reference_mean_nll": -reference_lp.mean().item(), "mean_nll_delta": -delta.mean().item(),
            "true_next_token_logprob_delta": distribution(delta),
            "greedy_agreement": agree.float().mean().item(), "greedy_disagreements": int((~agree).sum()),
            "actual_top1_top2_margin": distribution(actual_top[..., 0] - actual_top[..., 1]),
            "reference_top1_top2_margin": distribution(reference_top[..., 0] - reference_top[..., 1]),
            "per_token": {"target_ids": targets.flatten().tolist(),
                          "actual_logprob": actual_lp.flatten().tolist(),
                          "reference_logprob": reference_lp.flatten().tolist(),
                          "logprob_delta": delta.flatten().tolist(), "greedy_agreement": agree.flatten().tolist()}}


def state_snapshot(state) -> dict:
    result = {}
    for layer, value in enumerate(state.layers):
        for name in (("conv", "ssm") if hasattr(value, "ssm") else ("key", "value")):
            tensor = getattr(value, name)
            if tensor is not None:
                require_finite(tensor, tuple(tensor.shape), f"layer_{layer}.{name} state")
                result[f"layer_{layer}.{name}"] = tensor.detach().cpu().clone()
    return result


def state_comparison(actual: dict, expected: dict) -> dict:
    if actual.keys() != expected.keys():
        raise RuntimeError("retained-state names differ")
    fields = [{"field": name, "shape": list(value.shape),
               **compare(value, expected[name], TOLERANCES["bfloat16"])}
              for name, value in actual.items() if value.numel()]
    return {"passed": all(field["passed"] for field in fields), "fields": fields}


def _forward_backward(model, x, y, backend: str, chunk_size: int, recorder: dict):
    metadata = model.configure_scan_backend(backend, chunk_size)
    recorder["stage"] = f"fp32_{backend}_forward"
    model.zero_grad(set_to_none=True)
    try:
        with observe_scans(model, recorder), torch.autocast(x.device.type, enabled=False):
            logits, loss = model(x, y)
        require_finite(logits, (1, x.shape[1], model.cfg.vocab_size), f"{backend} FP32 logits")
        require_finite(loss, (), f"{backend} FP32 loss")
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None or parameter.grad.shape != parameter.shape:
                raise RuntimeError(f"missing or wrong-shaped parameter gradient: {name}")
            # Keep all gradients, including nonfinite ones, so every failure can be reported.
            gradients[name] = parameter.grad.detach().cpu().clone()
        return logits.detach().cpu(), loss.detach().cpu(), gradients, metadata
    finally:
        model.zero_grad(set_to_none=True)


def gradient_comparisons(actual: dict, expected: dict) -> dict:
    if list(actual) != list(expected):
        raise RuntimeError("trainable parameter-gradient names/order differ")
    checks = []
    for name, value in actual.items():
        reference = expected[name]
        result = compare(value, reference, TOLERANCES["float32"])
        norms = {"actual_l2": None, "reference_l2": None, "cosine_similarity": None}
        if result["finite"]:
            a, b = value.double().flatten(), reference.double().flatten()
            an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
            norms = {"actual_l2": an.item(), "reference_l2": bn.item(),
                     "cosine_similarity": (torch.dot(a / an, b / bn).clamp(-1, 1).item()
                                           if an.item() and bn.item() else None)}
        checks.append({"parameter": name, "shape": list(value.shape), **result, **norms})
    return {"scope": "full-model stateless next-token-loss backward; no optimizer/clip/update",
            "reference_storage": "CPU clones; reference graph freed before candidate forward",
            "tolerance": dict(TOLERANCES["float32"]), "parameter_tensors": len(checks),
            "all_passed": all(check["passed"] for check in checks),
            "failed_parameters": [check["parameter"] for check in checks if not check["passed"]],
            "checks": checks}


def fp32_backend_probe(model, x, y, chunk_size: int, recorder: dict):
    full, loss, reference_gradients, reference_backend = _forward_backward(model, x, y, "reference", chunk_size, recorder)
    candidate, candidate_loss, gradients, candidate_backend = _forward_backward(model, x, y, "torch_chunked", chunk_size, recorder)
    checks = {"logits": compare(candidate, full, TOLERANCES["float32"]),
              "loss": compare(candidate_loss, loss, TOLERANCES["float32"]),
              "gradients": gradient_comparisons(gradients, reference_gradients)}
    return {"reference_backend": reference_backend, "candidate_backend": candidate_backend,
            "observed_paths": {"prefill": None, "decode": None,
                               "backward": "autograd through each recorded forward scan; no separate kernel probe"},
            "tolerance": dict(TOLERANCES["float32"]), **checks,
            "next_token_effects": token_effects(candidate, full, y),
            "passed": checks["logits"]["passed"] and checks["loss"]["passed"] and checks["gradients"]["all_passed"]}, full


@torch.no_grad()
def bf16_cache_probe(model, x, y, fp32_anchor, prefix_length: int, chunk_size: int,
                     tokenwise: bool, recorder: dict) -> dict:
    backend = model.configure_scan_backend("reference", chunk_size)
    logits, states = {}, {}
    with observe_scans(model, recorder), torch.autocast(x.device.type, dtype=torch.bfloat16):
        recorder["stage"] = "bf16_full"
        logits["full"], _ = model(x)
        recorder["stage"] = "bf16_stateful_full"
        full_state = model.init_inference_state(1, device=x.device, cache_dtype=torch.bfloat16)
        logits["stateful_full"], _ = model(x, inference_state=full_state)
        states["stateful_full"] = state_snapshot(full_state)
        recorder["stage"] = "bf16_prefix_prefill"
        logits["prefix"], prefix = model.prefill(x[:, :prefix_length], cache_dtype=torch.bfloat16)
        prefix_snapshot = state_snapshot(prefix)
        suffix = x[:, prefix_length:]
        # Each schedule starts from a clone of the same naturally produced prefix state.
        recorder["stage"] = "bf16_suffix_one_shot"
        one_shot = prefix.clone()
        logits["suffix_one_shot"], _ = model(suffix, inference_state=one_shot)
        states["suffix_one_shot"] = state_snapshot(one_shot)
        recorder["stage"] = "bf16_suffix_segmented"
        segmented = prefix.clone()
        parts = [model(suffix[:, start:start + chunk_size], inference_state=segmented)[0]
                 for start in range(0, suffix.shape[1], chunk_size)]
        logits["suffix_segmented"] = torch.cat(parts, dim=1)
        states["suffix_segmented"] = state_snapshot(segmented)
        recorder["stage"] = "bf16_suffix_tokenwise"
        recurrent = prefix.clone()
        logits["suffix_tokenwise"] = torch.cat([model.decode(suffix[:, i:i + 1], recurrent)
                                                for i in range(suffix.shape[1])], dim=1)
        states["suffix_tokenwise"] = state_snapshot(recurrent)
        final_positions = {"stateful_full": full_state.position, "suffix_one_shot": one_shot.position,
                           "suffix_segmented": segmented.position, "suffix_tokenwise": recurrent.position}
        if tokenwise:
            recorder["stage"] = "bf16_full_tokenwise"
            recurrent_full = model.init_inference_state(1, device=x.device, cache_dtype=torch.bfloat16)
            logits["full_tokenwise"] = torch.cat([model.decode(x[:, i:i + 1], recurrent_full)
                                                  for i in range(x.shape[1])], dim=1)
            states["full_tokenwise"] = state_snapshot(recurrent_full)
            final_positions["full_tokenwise"] = recurrent_full.position
        unchanged = state_snapshot(prefix)
        if prefix.position != prefix_length or any(not torch.equal(value, unchanged[name])
                                                   for name, value in prefix_snapshot.items()):
            raise RuntimeError("a cloned continuation changed the original prefix state")
    if any(position != x.shape[1] for position in final_positions.values()):
        raise RuntimeError("continuation final position differs from input length")
    for name, value in logits.items():
        length = 1 if name == "prefix" else suffix.shape[1] if name.startswith("suffix_") else x.shape[1]
        require_finite(value, (1, length, model.cfg.vocab_size), f"{name} BF16 logits")
    logits = {name: value.cpu() for name, value in logits.items()}
    expected = {"stateful_full": logits["full"], "prefix": logits["full"][:, prefix_length - 1:prefix_length],
                **{name: logits["full"][:, prefix_length:] for name in logits if name.startswith("suffix_")}}
    if tokenwise:
        expected["full_tokenwise"] = logits["full"]
    internal = {name: compare(logits[name], value, TOLERANCES["bfloat16"]) for name, value in expected.items()}
    internal["suffix_tokenwise_vs_one_shot"] = compare(logits["suffix_tokenwise"], logits["suffix_one_shot"], TOLERANCES["bfloat16"])
    internal["suffix_segmented_vs_one_shot"] = compare(logits["suffix_segmented"], logits["suffix_one_shot"], TOLERANCES["bfloat16"])
    state_checks = {name: state_comparison(value, states["stateful_full"])
                    for name, value in states.items() if name != "stateful_full"}
    anchor = compare(logits["full"], fp32_anchor, TOLERANCES["bfloat16"])
    effects = {}
    for name, value in expected.items():
        start = prefix_length - 1 if name == "prefix" else prefix_length if name.startswith("suffix_") else 0
        targets = y[:, start:start + value.shape[1]]
        effects[name] = token_effects(logits[name], value, targets, start)
    effects["full_vs_fp32_anchor"] = token_effects(logits["full"], fp32_anchor, y)
    prefix_fields = [{"field": name, "shape": list(value.shape), "dtype": str(value.dtype),
                      "sha256": tensor_sha256(value), "nonzero_elements": int(torch.count_nonzero(value))}
                     for name, value in prefix_snapshot.items()]
    internal_passed = all(check["passed"] for check in [*internal.values(), *state_checks.values()])
    return {"backend": backend, "tolerance": dict(TOLERANCES["bfloat16"]), "cache_dtype": "torch.bfloat16",
            "internal_logits": internal, "retained_state": state_checks, "internal_passed": internal_passed,
            "fp32_anchor": {"scope": "separate cross-precision full-logit comparison", **anchor},
            "next_token_effects": effects, "real_prefix": {"position": prefix_length, "fields": prefix_fields,
                "unchanged_after_cloned_continuations": True, "synthetic": False},
            "suffix_length": suffix.shape[1], "one_shot_vs_tokenwise_degenerate": suffix.shape[1] == 1,
            "segmented_schedule": [min(chunk_size, suffix.shape[1] - start)
                                   for start in range(0, suffix.shape[1], chunk_size)],
            "final_positions": final_positions,
            "unexecuted_stages": [] if tokenwise else ["bf16_full_tokenwise"],
            "passed": internal_passed and anchor["passed"]}


def study_model(model, x, y, *, prefix_length: int, chunk_size: int, tokenwise: bool) -> dict:
    """One copied/loaded model, two sequential FP32 graphs and a BF16 inference probe."""
    recorder = {"stage": "unexecuted", "events": {}}
    identity = weight_sha256(model)
    with preserve_model_execution(model):
        fp32, anchor = fp32_backend_probe(model, x, y, chunk_size, recorder)
        bf16 = bf16_cache_probe(model, x, y, anchor, prefix_length, chunk_size, tokenwise, recorder)
    if weight_sha256(model) != identity:
        raise RuntimeError("diagnostic changed model weights")
    events = [{**json.loads(identity), "calls": calls} for identity, calls in sorted(recorder["events"].items())]
    paths = {stage: sorted({event["path"] for event in events if event["stage"] == stage})
             for stage in sorted({event["stage"] for event in events})}
    if not tokenwise:
        paths["bf16_full_tokenwise"] = None
    return {"weights_sha256": identity, "fp32_backend": fp32, "bf16_cached": bf16,
            "operator_observations": events, "actual_paths": paths,
            "passed": fp32["passed"] and bf16["passed"]}


def audit_inputs(data_dir: Path, tokenizer_path: Path, checkpoint_root: Path,
                 training_run_id: str, ratios: tuple):
    """Audit all data/config/checkpoint identities before any model allocation."""
    data_dir, tokenizer_path = Path(data_dir).resolve(), Path(tokenizer_path).resolve()
    manifest = validate_prepared_dataset(data_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    identity = manifest["tokenizer"]
    if (file_sha256(tokenizer_path) != identity["sha256"]
            or tokenizer.get_vocab_size() != identity["vocab_size"]
            or tokenizer.token_to_id(identity["eot_token"]) != identity["eot_id"]):
        raise RuntimeError("tokenizer does not match the prepared dataset identity")
    checkpoints = discover_checkpoints(Path(checkpoint_root), training_run_id, ratios)
    meta = read_json(data_dir / "meta.json")
    evidence = []
    for ratio in ratios:
        checkpoint = checkpoints[ratio]
        cfg = ModelConfig(**checkpoint.model_config)
        if count(cfg)["total"] > 100_000_000:
            raise ValueError("bounded diagnostics require checkpoints with at most 100M parameters")
        if cfg.vocab_size != identity["vocab_size"]:
            raise RuntimeError("checkpoint vocabulary differs from the prepared data")
        training = read_json(checkpoint.run_dir / "manifest.json")
        if asdict(ModelConfig(**training["model_config"])) != asdict(cfg):
            raise RuntimeError("checkpoint config differs from its training manifest")
        data = training.get("data", {})
        for label in ("train", "val", "meta"):
            artifact = manifest["outputs"][label]
            historical = data.get("files", {}).get(artifact["file"], {})
            if historical.get("sha256") != artifact["sha256"] or historical.get("bytes") != artifact["bytes"]:
                raise RuntimeError(f"checkpoint training-data identity differs for {label}")
        if data.get("meta") != meta:
            raise RuntimeError("checkpoint training metadata differs from the prepared data")
        evidence.append({"ratio": ratio, **public_reference(checkpoint.best_path),
                         "sha256": checkpoint.checkpoint_sha256, "training_signature": checkpoint.signature,
                         "step": checkpoint.step, "recorded_val_loss": checkpoint.val_loss,
                         "training_manifest": {**public_reference(checkpoint.run_dir / "manifest.json"),
                             "sha256": file_sha256(checkpoint.run_dir / "manifest.json")},
                         "model_config": asdict(cfg), "parameter_count": count(cfg)["total"],
                         "historical_training_data_match": True})
    data_evidence = {"audit": "full prepared dataset integrity, tokenizer and checkpoint training-data linkage verified",
                     "manifest": {**public_reference(data_dir / "manifest.json"), "sha256": file_sha256(data_dir / "manifest.json")},
                     "build_signature": manifest["build_signature"], "signature": manifest["signature"],
                     "artifacts": {label: {**public_reference(data_dir / item["file"]), "sha256": item["sha256"],
                                           "bytes": item["bytes"]} for label, item in manifest["outputs"].items()},
                     "tokenizer": {**public_reference(tokenizer_path), "sha256": identity["sha256"],
                                   "vocab_size": identity["vocab_size"], "eot_id": identity["eot_id"]},
                     "selection_scope": "checkpoint-selection validation pool; not an independent quality test"}
    return checkpoints, data_evidence, evidence


def source_metadata() -> dict:
    files = [*sorted((ROOT / "src/model").glob("*.py")), Path(__file__),
             ROOT / "scripts/check_scan_backend.py", ROOT / "scripts/count_params.py",
             ROOT / "src/eval/suite.py", ROOT / "src/data/dataset.py", ROOT / "src/data/prepare_data.py",
             ROOT / "src/data/train_tokenizer.py"]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in files}


def runtime_metadata(device: str) -> dict:
    packages = {}
    for package in ("torch", "numpy", "tokenizers", "mamba-ssm", "triton", "causal-conv1d"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    # Some Torch releases reject this getter after legacy backend-flag calls. It is
    # supplementary metadata, never a reason to abandon an otherwise valid study.
    try:
        matmul_precision = torch.get_float32_matmul_precision()
    except RuntimeError:
        matmul_precision = None
    return {"packages": packages, "python": platform.python_version(), "platform": platform.platform(),
            "torch_cuda_runtime": torch.version.cuda, "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "cuda_capability": list(torch.cuda.get_device_capability()) if device == "cuda" else None,
            "precision_flags": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                "float32_matmul_precision": matmul_precision,
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}


def run_study(device: str = "cuda", seed: int = 2027, length: int = 129,
              prefix_length: int = 128, windows: int = 1, chunk_size: int = 128,
              tokenwise: bool = True, checkpoint_root: Path = Path("checkpoints"),
              training_run_id: str = "week3-700m-v1", data_dir: Path = Path("data/openwebtext-5b"),
              tokenizer: Path = Path("data/tokenizer/openwebtext.json"),
              ratios: tuple = WEEK3_RATIOS) -> dict:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in 0..2^32-1")
    if (not isinstance(length, int) or isinstance(length, bool) or not 2 <= length <= 257
            or not isinstance(prefix_length, int) or isinstance(prefix_length, bool) or not 1 <= prefix_length < length):
        raise ValueError("length must be 2..257 and prefix_length must be 1..length-1")
    if not isinstance(windows, int) or isinstance(windows, bool) or not 1 <= windows <= 4 or windows * length > 1028:
        raise ValueError("use one to four windows and at most 1028 positions per checkpoint")
    if not isinstance(tokenwise, bool):
        raise ValueError("tokenwise must be boolean")
    if (not isinstance(training_run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", training_run_id)
            or training_run_id in (".", "..")):
        raise ValueError("training_run_id must be one nonempty directory name")
    ratios = tuple(ratios)
    if (not ratios or len(ratios) > 3 or any(not isinstance(ratio, str) for ratio in ratios)
            or len(set(ratios)) != len(ratios)):
        raise ValueError("select one to three distinct ratio names")
    resolve_scan_backend("reference", chunk_size)
    resolve_scan_backend("torch_chunked", chunk_size)
    checkpoints, data_evidence, checkpoint_evidence = audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, ratios)
    rows = shared_windows(load_split(data_dir, "val"), length, windows, seed)
    vocab_size = data_evidence["tokenizer"]["vocab_size"]
    if any(bool((x < 0).any() or (x >= vocab_size).any() or (y < 0).any() or (y >= vocab_size).any()) for x, y, _ in rows):
        raise ValueError("validation window contains IDs outside the common vocabulary")
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA is unavailable; CPU fallback is forbidden")
    sources = source_metadata()
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    cases = []
    with torch.random.fork_rng(devices=devices), tf32_disabled():
        runtime = runtime_metadata(device)
        for ratio in ratios:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model = load_variant_model(checkpoints[ratio], torch.device(device))
            try:
                for name, parameter in model.named_parameters():
                    require_finite(parameter, tuple(parameter.shape), f"loaded parameter {name}")
                initial_weights = weight_sha256(model)
                for x_cpu, y_cpu, identity in rows:
                    x, y = x_cpu.to(device), y_cpu.to(device)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = study_model(model, x, y, prefix_length=prefix_length,
                                         chunk_size=chunk_size, tokenwise=tokenwise)
                    if result["weights_sha256"] != initial_weights:
                        raise RuntimeError("checkpoint weights changed between windows")
                    if device == "cuda":
                        torch.cuda.synchronize()
                    cases.append({"ratio": ratio, **identity, **result,
                                  "processing_seconds": time.perf_counter() - start,
                                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
                    del x, y
            finally:
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        for ratio in ratios:
            if file_sha256(checkpoints[ratio].best_path) != checkpoints[ratio].checkpoint_sha256:
                raise RuntimeError("historical checkpoint bytes changed during the diagnostic")
    protocol = {"kind": KIND, "version": 1, "seed": seed, "device": device, "length": length,
                "prefix_length": prefix_length, "windows": windows, "batch_size": 1, "chunk_size": chunk_size,
                "full_tokenwise": tokenwise, "ratios": list(ratios), "training_run_id": training_run_id,
                "shared_inputs": [identity for _, _, identity in rows], "data": data_evidence,
                "checkpoints": checkpoint_evidence, "source_sha256": sources,
                "runtime": runtime, "runtime_sha256": canonical_sha256(runtime),
                "tolerances": {name: dict(value) for name, value in TOLERANCES.items()},
                "precision_policy": {"fp32_backend": "reference versus torch_chunked; autocast disabled",
                                     "bf16_cached": "reference backend; BF16 autocast; naturally produced cloned prefix states",
                                     "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
                "quality_equivalence_margin": None}
    return {"schema": 1, "kind": KIND, "status": "completed" if all(case["passed"] for case in cases)
            else "completed_with_parity_failures", "execution_status": "completed", "certified": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
            "protocol_sha256": canonical_sha256(protocol), "cases": cases,
            "limits": ["Validation data helped select these checkpoints; this is not an independent language-quality test.",
                       "Next-token probability/NLL changes and greedy agreement are descriptive; no quality-equivalence cutoff is applied.",
                       "The one-token default suffix makes one-shot/tokenwise continuation identical in call shape; use length257 for a longer suffix.",
                       "FP32 gradients compare only the two portable scan paths at the recorded shapes; BF16 full-model gradient parity is unexecuted.",
                       "Real prefix states cover only the recorded natural windows, not every reachable state or future context.",
                       "Processing time includes transfers, hashing and checks; it is not a throughput benchmark.",
                       "No optimizer, training, fused scan, production precision change or backend certification."]}


def write_report(path: Path, report: dict) -> None:
    """Publish a complete LF artifact atomically, without replacing any existing path."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("output already exists; choose a new diagnostic artifact")
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic no-replace publication on the same filesystem.
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--length", type=int, default=129)
    parser.add_argument("--prefix-length", type=int, default=128)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--skip-full-tokenwise", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--training-run-id", default="week3-700m-v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/openwebtext-5b"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/openwebtext.json"))
    parser.add_argument("--ratios", nargs="+", default=list(WEEK3_RATIOS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new diagnostic artifact")
    try:
        report = run_study(device=args.device, seed=args.seed, length=args.length,
                           prefix_length=args.prefix_length, windows=args.windows, chunk_size=args.chunk_size,
                           tokenwise=not args.skip_full_tokenwise, checkpoint_root=args.checkpoint_root,
                           training_run_id=args.training_run_id, data_dir=args.data_dir,
                           tokenizer=args.tokenizer, ratios=tuple(args.ratios))
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        for path in sorted([ROOT, Path.home(), args.checkpoint_root.resolve(), args.data_dir.resolve(),
                            args.tokenizer.resolve()], key=lambda value: len(str(value)), reverse=True):
            reason = reason.replace(str(path), public_reference(path)["path"])
        report = {"schema": 1, "kind": KIND, "status": "unavailable" if isinstance(exc, BackendUnavailableError) else "incomplete",
                  "execution_status": "incomplete", "certified": False, "reason": reason,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if args.output:
        write_report(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
