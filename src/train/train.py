"""Training loop for one hybrid variant.

The loop keeps each experiment under ``checkpoints/<run-id>/<variant>/``.  ``best.pt`` is the
lowest-validation-loss model for evaluation, while ``last.pt`` is the resumable training state.
Metrics and a provenance manifest live beside them so a run can be audited or recovered without
depending on an external tracking service.

    python -m src.train.train --model-config configs/ratio_1_7.yaml --run-id debug-001
    python -m src.train.train --model-config configs/ratio_1_3.yaml --wandb
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from src.data.dataset import get_batch, load_meta, load_split
from src.model.config import ModelConfig
from src.model.lm import HybridLM
from src.model.scan_backend import BackendUnavailableError, resolve_scan_backend

CHECKPOINT_SCHEMA = 1
MANIFEST_SCHEMA = 1
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STREAM_SEED_OFFSETS = {"train": 0, "eval_train": 1, "eval_val": 2}
ARTIFACT_FILES = {"best": "best.pt", "last": "last.pt", "metrics": "metrics.jsonl", "result": "result.json"}
_CHECKPOINT_JOURNAL = ".checkpoint-transaction.json"
_CHECKPOINT_STORE = ".checkpoint-transactions"
_TRANSACTION_FILES = {"best": "best.pt", "last": "last.pt", "metrics": "metrics.jsonl"}
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass
class TrainConfig:
    model_config: str = "configs/ratio_1_7.yaml"
    data_dir: str = "data/tinystories"
    # optim
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # schedule / length
    max_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 8
    grad_accum: int = 4
    block_size: int = 512
    # eval / logging / checkpoints
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    checkpoint_interval: int = 250
    ckpt_dir: str = "checkpoints"
    run_id: str = "default"
    resume: bool = True
    # misc
    seed: int = 1337
    # None preserves the historical seed policy; explicit streams support paired experiments.
    model_seed: int | None = None
    data_seed: int | None = None
    eval_seed: int | None = None
    device: str = "cuda"
    grad_checkpointing: bool = True
    scan_backend: str = "reference"
    scan_chunk_size: int = 128
    precision: str = "bfloat16"
    # Invocation wall allowance, not a different LR/token schedule. None keeps legacy identity.
    wall_time_limit_seconds: float | None = None
    wandb: bool = False
    wandb_project: str = "mamba-hybrid-lm"


class TrainingStopControl:
    """A shared invocation deadline/stop request; work finishes at a durable boundary."""

    def __init__(self, wall_time_limit_seconds: float | None = None, *, clock: Callable[[], float] = time.monotonic):
        _validate_wall_limit(wall_time_limit_seconds)
        self.wall_time_limit_seconds = wall_time_limit_seconds
        self.clock = clock
        self.started = clock()
        self.requested = threading.Event()
        self.request_reason = "requested"

    def request_stop(self, reason: str = "requested") -> None:
        if not self.requested.is_set():
            self.request_reason = reason
            self.requested.set()

    def reason(self) -> str | None:
        if self.requested.is_set():
            return self.request_reason
        if self.wall_time_limit_seconds is not None and self.clock() - self.started >= self.wall_time_limit_seconds:
            return "wall_time_limit"
        return None

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(0.0, self.clock() - self.started)
        return {
            "reason": self.reason(), "elapsed_seconds": elapsed,
            "wall_time_limit_seconds": self.wall_time_limit_seconds,
            "overshoot_seconds": max(0.0, elapsed - self.wall_time_limit_seconds)
            if self.wall_time_limit_seconds is not None else 0.0,
            "scope": "cooperative invocation wall time; in-flight work may exceed the allowance",
        }


_ACTIVE_STOP_CONTROL: ContextVar[TrainingStopControl | None] = ContextVar("training_stop_control", default=None)


def current_training_stop_control() -> TrainingStopControl | None:
    return _ACTIVE_STOP_CONTROL.get()


@contextmanager
def training_stop_scope(control: TrainingStopControl):
    """Share one deadline across sequential arms without changing the numerical config."""
    token = _ACTIVE_STOP_CONTROL.set(control)
    try:
        yield
    finally:
        _ACTIVE_STOP_CONTROL.reset(token)


@contextmanager
def cooperative_stop_signals(control: TrainingStopControl):
    """CLI signals request recovery-safe stopping; never promise to kill a GPU operation."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}
    try:
        for name in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is not None:
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda received, _frame: control.request_stop(f"signal:{signal.Signals(received).name}"))
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _validate_wall_limit(value: Any) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value <= 0):
        raise ValueError("wall_time_limit_seconds must be a finite positive number or None")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_run_id(run_id: str) -> str:
    """Reject path-like or ambiguous run identifiers before using one as a directory."""
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must start with an alphanumeric and contain only A-Z, a-z, 0-9, '.', '_' or '-'")
    if run_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"run_id is a reserved Windows filename: {run_id}")
    return run_id


def variant_slug(name: str) -> str:
    if not name or name in {".", ".."}:
        raise ValueError("model name must produce a non-empty variant directory")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not base or base.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"model name cannot be represented safely as a directory: {name!r}")
    # A digest prevents names such as "a:b" and "a/b" from collapsing to the same directory.
    return base if base == name else f"{base}-{hashlib.sha256(name.encode()).hexdigest()[:8]}"


def variant_run_dir(cfg: TrainConfig, model_name: str) -> Path:
    return Path(cfg.ckpt_dir) / validate_run_id(cfg.run_id) / variant_slug(model_name)


def load_model_config(path: str) -> ModelConfig:
    return ModelConfig(**yaml.safe_load(Path(path).read_text()))


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    # A zero-step warmup is useful in tiny CPU smoke tests and means "start at peak LR".
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    denom = max(1, cfg.max_steps - cfg.warmup_steps)
    frac = (step - cfg.warmup_steps) / denom
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (cfg.lr - cfg.min_lr)


def make_batch_generator(seed: int, stream: str) -> torch.Generator:
    """Create a CPU generator for one data stream, independent of model/global RNG use."""
    try:
        offset = _STREAM_SEED_OFFSETS[stream]
    except KeyError as exc:
        raise ValueError(f"unknown RNG stream: {stream}") from exc
    return torch.Generator(device="cpu").manual_seed(seed + offset)


def precision_policy(precision: str) -> dict[str, Any]:
    """Describe a numerical policy; selecting it does not certify its results."""
    if not isinstance(precision, str) or precision not in {"bfloat16", "float32"}:
        raise ValueError("precision must be 'bfloat16' or 'float32'")
    return {
        "precision": precision,
        "autocast_enabled": precision == "bfloat16",
        "autocast_dtype": "bfloat16" if precision == "bfloat16" else None,
        "tf32_policy": "legacy" if precision == "bfloat16" else "disabled",
    }


@contextmanager
def _autocast(cfg: TrainConfig):
    policy = precision_policy(cfg.precision)
    with _precision_runtime(cfg), torch.autocast(
            torch.device(cfg.device).type, dtype=torch.bfloat16,
            enabled=policy["autocast_enabled"]):
        yield


@contextmanager
def _precision_runtime(cfg: TrainConfig):
    """Scope the nondefault FP32 policy without leaking TF32 changes to the caller."""
    if cfg.precision == "bfloat16":
        yield
        return
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _precision_metadata(cfg: TrainConfig) -> dict[str, Any]:
    cuda = torch.device(cfg.device).type == "cuda"
    return {
        "policy": precision_policy(cfg.precision),
        "device_type": torch.device(cfg.device).type,
        "runtime_flags": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if cuda else None,
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if cuda else None,
        },
    }


def _validate_precision_metadata(value: dict[str, Any], cfg: TrainConfig) -> None:
    metadata = value.get("training_precision")
    # Missing precision means historical BF16. It cannot describe the new FP32 policy.
    if metadata is None and cfg.precision == "bfloat16":
        return
    expected = _precision_metadata(cfg)
    if expected["device_type"] == "cuda":
        # Default training sets matmul TF32 true after completed-run validation;
        # the explicit FP32 policy disables both flags before any model execution.
        expected["runtime_flags"]["cuda_matmul_allow_tf32"] = cfg.precision == "bfloat16"
        if cfg.precision == "float32":
            expected["runtime_flags"]["cudnn_allow_tf32"] = False
    if metadata != expected:
        raise RuntimeError("training precision metadata does not match this numerical policy")


@torch.no_grad()
def estimate_loss(model, splits, cfg: TrainConfig) -> dict[str, float]:
    """Evaluate the same fixed windows every time without advancing the training sampler."""
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    try:
        for name, data in splits.items():
            generator = make_batch_generator(
                cfg.seed if cfg.eval_seed is None else cfg.eval_seed, f"eval_{name}",
            )
            losses = torch.zeros(cfg.eval_iters)
            for i in range(cfg.eval_iters):
                x, y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device, generator=generator)
                with _autocast(cfg):
                    _, loss = model(x, y)
                batch_loss = loss.item()
                if not math.isfinite(batch_loss):
                    raise FloatingPointError(
                        f"non-finite evaluation loss for split {name!r}, batch {i + 1}/{cfg.eval_iters}"
                    )
                losses[i] = batch_loss
            mean_loss = losses.mean().item()
            if not math.isfinite(mean_loss):
                raise FloatingPointError(
                    f"non-finite aggregate evaluation loss for split {name!r}, "
                    f"after batch {cfg.eval_iters}/{cfg.eval_iters}"
                )
            out[name] = mean_loss
    finally:
        model.train(was_training)
    return out


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text artifact only after its complete temporary file reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> Any:
    """Read strict JSON; Python's default acceptance of NaN/Infinity is unsafe for metrics."""
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc


def atomic_torch_save(value: Any, path: Path) -> None:
    """Keep the previous checkpoint intact if serialization is interrupted or fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(value, temp)
        # Windows only permits FlushFileBuffers through a write-capable handle.
        with temp.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class MetricsWriter:
    """Append one fsynced JSON object per line so every completed event is locally durable."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def reconcile_metrics(path: Path, completed_steps: int, *, cfg: TrainConfig | None = None) -> None:
    """Drop partial/future metric rows that were written after the last durable checkpoint."""
    if not path.exists():
        return
    kept: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            if line_no != len(lines):
                raise RuntimeError(f"corrupt metrics row {line_no} in {path}")
            break
        if int(record["step"]) <= completed_steps:
            kept.append(json.dumps(record, sort_keys=True, allow_nan=False))
    contents = "".join(f"{line}\n" for line in kept)
    if cfg is not None:
        # A torn final row is only disposable when ALL required durable events remain.
        # Validate in memory before rewriting, preserving evidence on ordinary corruption.
        _validate_metrics(path, cfg, completed_steps=completed_steps, contents=contents)
    atomic_write_text(path, contents)


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _signature_payload(
    cfg: TrainConfig,
    mcfg: ModelConfig,
    data_provenance: dict[str, Any],
    code_provenance: dict[str, Any],
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    # Tracking preferences do not affect the numerical trajectory; every training-relevant field does.
    train_config = _trajectory_config_dict(asdict(cfg))
    return {
        "schema": MANIFEST_SCHEMA,
        "train_config": train_config,
        "model_config": asdict(mcfg),
        "data": data_provenance,
        "code": code_provenance,
        "runtime": runtime_provenance,
    }


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_provenance(data_dir: str, meta: dict[str, Any]) -> dict[str, Any]:
    root = Path(data_dir).resolve()
    files = {}
    for name in ("train.bin", "val.bin", "meta.json"):
        path = root / name
        stat = path.stat()
        files[name] = {
            "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": _file_sha256(path),
        }
    return {"root": str(root), "meta": meta, "files": files}


def _code_provenance() -> dict[str, Any]:
    """Fingerprint runtime code, while retaining the git commit/dirty state for scrutiny."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src").rglob("*.py"))
    paths.extend(path for path in (root / "scripts" / "run_sweep.py", root / "requirements.txt") if path.exists())
    files = {str(path.relative_to(root)).replace("\\", "/"): _file_sha256(path) for path in paths}
    return {"root": str(root), "fingerprint": _canonical_hash(files), "files": files, "git": _git_provenance()}


def _runtime_provenance() -> dict[str, Any]:
    packages = {}
    for package in (
        "einops", "datasets", "tokenizers", "transformers", "numpy", "wandb", "tqdm",
        "mamba-ssm", "triton", "causal-conv1d",
    ):
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "device_name": torch.cuda.get_device_name() if cuda_available else None,
        "device_capability": list(torch.cuda.get_device_capability()) if cuda_available else None,
        "packages": packages,
    }


def _new_manifest(
    cfg: TrainConfig,
    mcfg: ModelConfig,
    signature: str,
    data_provenance: dict[str, Any],
    code_provenance: dict[str, Any],
    runtime_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": cfg.run_id,
        "variant": mcfg.name,
        "ratio": mcfg.ratio,
        "status": "running",
        "signature": signature,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "command": sys.argv,
        "train_config": asdict(cfg),
        "model_config": asdict(mcfg),
        "data": data_provenance,
        "code": code_provenance,
        "runtime": runtime_provenance,
        "git": code_provenance["git"],
    }


def _capture_rng(train_generator: torch.Generator) -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        # Keep the payload inside torch.load(weights_only=True)'s tensor/primitive allowlist.
        "python": [python_state[0], list(python_state[1]), python_state[2]],
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_generator": train_generator.get_state(),
    }


def _restore_rng(state: dict[str, Any], train_generator: torch.Generator) -> None:
    python_state = state["python"]
    random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"], numpy_state["keys"].numpy(), numpy_state["position"],
        numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_generator"])


def _checkpoint_state(
    model: HybridLM,
    optim: torch.optim.Optimizer,
    cfg: TrainConfig,
    mcfg: ModelConfig,
    signature: str,
    completed_steps: int,
    tokens_seen: int,
    best_val: float,
    train_seconds: float,
    eval_seconds: float,
    peak_vram_mb: float,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "signature": signature,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "train_config": asdict(cfg),
        "model_config": asdict(mcfg),
        "completed_steps": completed_steps,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "peak_vram_mb": peak_vram_mb,
        "rng": _capture_rng(train_generator),
    }


def _load_resume_checkpoint(
    path: Path,
    cfg: TrainConfig,
    mcfg: ModelConfig,
    signature: str,
    model: HybridLM,
    optim: torch.optim.Optimizer,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    state = _safe_load_checkpoint(path, "last")
    _validate_last_state(state, cfg, mcfg, signature, require_complete=False)
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"], train_generator)
    return state


def _expected_tokens(cfg: TrainConfig, completed_steps: int) -> int:
    return completed_steps * cfg.batch_size * cfg.grad_accum * cfg.block_size


def _require_finite(value: Any, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{label} must be a finite number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise RuntimeError(f"{label} must be >= {minimum}")
    return numeric


def _trajectory_config_dict(value: dict[str, Any]) -> dict[str, Any]:
    filtered = dict(value)
    for key in ("resume", "wandb", "wandb_project"):
        filtered.pop(key, None)
    if filtered.get("precision", "bfloat16") == "bfloat16":
        filtered.pop("precision", None)
    if filtered.get("wall_time_limit_seconds") is None:
        filtered.pop("wall_time_limit_seconds", None)
    return filtered


def _safe_load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required {label} checkpoint is missing: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"required {label} checkpoint is unreadable: {path}") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"required {label} checkpoint is not a mapping: {path}")
    return state


def _validate_last_state(
    state: dict[str, Any], cfg: TrainConfig, mcfg: ModelConfig, signature: str, *, require_complete: bool,
) -> None:
    required = {
        "schema", "signature", "model", "optimizer", "train_config", "model_config",
        "completed_steps", "tokens_seen", "best_val_loss", "train_seconds", "eval_seconds",
        "peak_vram_mb", "rng",
    }
    missing = required.difference(state)
    if missing:
        raise RuntimeError(f"last checkpoint is missing keys: {sorted(missing)}")
    if state["schema"] != CHECKPOINT_SCHEMA or state["signature"] != signature:
        raise RuntimeError("last checkpoint signature/schema does not match this run")
    if state["model_config"] != asdict(mcfg):
        raise RuntimeError("last checkpoint model config does not match this variant")
    if _trajectory_config_dict(state["train_config"]) != _trajectory_config_dict(asdict(cfg)):
        raise RuntimeError("last checkpoint training config does not match this run")
    steps = state["completed_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= cfg.max_steps:
        raise RuntimeError("last checkpoint has an invalid completed_steps value")
    if require_complete and steps != cfg.max_steps:
        raise RuntimeError("completed result has an incomplete last checkpoint")
    if state["tokens_seen"] != _expected_tokens(cfg, steps):
        raise RuntimeError("last checkpoint token count does not match its completed steps")
    _require_finite(state["best_val_loss"], "last.best_val_loss", 0.0)
    _require_finite(state["train_seconds"], "last.train_seconds", 0.0)
    _require_finite(state["eval_seconds"], "last.eval_seconds", 0.0)
    _require_finite(state["peak_vram_mb"], "last.peak_vram_mb", 0.0)
    if not isinstance(state["model"], dict) or not isinstance(state["optimizer"], dict):
        raise RuntimeError("last checkpoint is missing model or optimizer state")
    if not isinstance(state["rng"], dict):
        raise RuntimeError("last checkpoint RNG state is invalid")


def _validate_best_state(
    state: dict[str, Any], mcfg: ModelConfig, signature: str, last_state: dict[str, Any], max_steps: int,
) -> None:
    required = {"schema", "signature", "model", "model_config", "step", "val_loss"}
    missing = required.difference(state)
    if missing:
        raise RuntimeError(f"best checkpoint is missing keys: {sorted(missing)}")
    if state["schema"] != CHECKPOINT_SCHEMA or state["signature"] != signature:
        raise RuntimeError("best checkpoint signature/schema does not match this run")
    if state["model_config"] != asdict(mcfg) or not isinstance(state["model"], dict):
        raise RuntimeError("best checkpoint model/config does not match this variant")
    last_step = int(last_state["completed_steps"])
    if (isinstance(state["step"], bool) or not isinstance(state["step"], int)
            or not 0 <= state["step"] <= min(max_steps, last_step)):
        raise RuntimeError("best checkpoint step is invalid")
    best_loss = _require_finite(state["val_loss"], "best.val_loss", 0.0)
    if not math.isclose(best_loss, float(last_state["best_val_loss"]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("best checkpoint and resumable checkpoint disagree on best validation loss")
    if "optimizer" in state or "rng" in state:
        raise RuntimeError("best checkpoint must remain distinct from resumable training state")


def _validate_metrics(path: Path, cfg: TrainConfig, *, completed_steps: int | None = None,
                      contents: str | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"required metrics artifact is missing: {path}")
    expected_events: list[tuple[str, int]] = [("eval", 0)]
    through = cfg.max_steps if completed_steps is None else completed_steps
    for step in range(1, through + 1):
        expected_events.append(("train", step))
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            expected_events.append(("eval", step))
    train_fields = {"event", "step", "tokens_seen", "loss", "grad_norm", "lr", "step_seconds", "tok_per_s"}
    eval_fields = {"event", "step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr"}
    seen_events = 0
    try:
        with (path.open("r", encoding="utf-8") if contents is None else io.StringIO(contents)) as handle:
            for line_no, line in enumerate(handle, start=1):
                record = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(record, dict) or record.get("event") not in {"train", "eval"}:
                    raise RuntimeError(f"invalid metrics event at line {line_no}")
                step = record.get("step")
                if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= cfg.max_steps:
                    raise RuntimeError(f"invalid metrics step at line {line_no}")
                if seen_events >= len(expected_events) or (record["event"], step) != expected_events[seen_events]:
                    raise RuntimeError(f"duplicate, missing, or out-of-order metrics event at line {line_no}")
                seen_events += 1
                required_fields = train_fields if record["event"] == "train" else eval_fields
                if set(record) != required_fields:
                    raise RuntimeError(f"metrics event has missing or unexpected fields at line {line_no}")
                expected_tokens = _expected_tokens(cfg, step)
                tokens_seen = record.get("tokens_seen")
                if isinstance(tokens_seen, bool) or not isinstance(tokens_seen, int) or tokens_seen != expected_tokens:
                    raise RuntimeError(f"invalid metrics token count at line {line_no}")
                for key, value in record.items():
                    if key not in {"event", "step", "tokens_seen"}:
                        minimum = 0.0 if key in {
                            "loss", "grad_norm", "lr", "step_seconds", "tok_per_s",
                            "train_loss", "val_loss", "val_ppl",
                        } else None
                        _require_finite(value, f"metrics[{line_no}].{key}", minimum)
                if record["event"] == "train":
                    expected_lr = cosine_lr(step - 1, cfg)
                else:
                    expected_lr = cosine_lr(0 if step == 0 else step - 1, cfg)
                    if record["val_ppl"] <= 0 or not math.isclose(
                        record["val_ppl"], math.exp(record["val_loss"]), rel_tol=1e-6, abs_tol=1e-6,
                    ):
                        raise RuntimeError(f"invalid evaluation perplexity at line {line_no}")
                if not math.isclose(record["lr"], expected_lr, rel_tol=0.0, abs_tol=1e-15):
                    raise RuntimeError(f"metrics LR does not match the applied schedule at line {line_no}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"metrics artifact is unreadable or non-finite: {path}") from exc
    if seen_events != len(expected_events):
        raise RuntimeError("metrics artifact does not describe a complete training trajectory")


def _transaction_path(run_dir: Path, directory: str, filename: str) -> Path:
    """Allow only our fixed private filenames, with no symlink/path escape during recovery."""
    if (not isinstance(directory, str)
            or re.fullmatch(r"\.checkpoint-transactions/tx-[0-9a-f]{32}", directory) is None
            or filename not in {f"{kind}-{name}" for kind in ("previous", "next") for name in _TRANSACTION_FILES.values()}):
        raise RuntimeError("unsafe checkpoint transaction path")
    root = run_dir.resolve()
    path = run_dir / directory / filename
    if any(part.is_symlink() for part in (run_dir / _CHECKPOINT_STORE, path.parent, path)):
        raise RuntimeError("checkpoint transaction cannot use symlinks")
    if not path.resolve().is_relative_to(root):
        raise RuntimeError("checkpoint transaction escapes its locked run directory")
    return path


def _atomic_snapshot(source: Path, destination: Path, *, hardlink: bool = False) -> None:
    """Immutable checkpoint links survive replacement; appendable metrics always use a copy."""
    if source.is_symlink() or destination.is_symlink():
        raise RuntimeError("checkpoint transaction cannot snapshot symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp")
    try:
        if temp.is_symlink():
            raise RuntimeError("checkpoint transaction temporary file cannot be a symlink")
        # A killed publisher may have left a hard link here. Never truncate that link:
        # it shares an inode with an immutable recovery snapshot.
        temp.unlink(missing_ok=True)
        if hardlink:
            try:
                os.link(source, temp)
            except OSError:
                shutil.copyfile(source, temp)
        else:
            shutil.copyfile(source, temp)
        with temp.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _metrics_prefix(path: Path, step: int | None) -> str:
    if step is None or not path.exists():
        return ""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line, parse_constant=_reject_json_constant)
        if record["step"] <= step:
            records.append(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return "".join(records)


def _clean_inert_transaction_staging(run_dir: Path) -> None:
    """Discard only allowlisted unpublished leftovers, after run provenance is verified."""
    if (run_dir / _CHECKPOINT_JOURNAL).exists():
        raise RuntimeError("pending checkpoint transaction must be recovered before staging another")
    store = run_dir / _CHECKPOINT_STORE
    if not store.exists():
        return
    if store.is_symlink() or not store.resolve().is_relative_to(run_dir.resolve()):
        raise RuntimeError("unsafe checkpoint transaction staging directory")
    names = {f"{kind}-{name}" for kind in ("previous", "next") for name in _TRANSACTION_FILES.values()}
    allowed = names | {f".{name}.tmp" for name in names}
    removable = []
    for folder in store.iterdir():
        if re.fullmatch(r"tx-[0-9a-f]{32}", folder.name) is None:
            continue
        _transaction_path(run_dir, f"{_CHECKPOINT_STORE}/{folder.name}", "next-last.pt")
        entries = list(folder.iterdir())
        if any(entry.name not in allowed for entry in entries):
            continue  # Never delete unknown files even in the private staging area.
        if any(entry.is_symlink() or not entry.is_file()
               or not entry.resolve().is_relative_to(run_dir.resolve()) for entry in entries):
            raise RuntimeError("unsafe unpublished checkpoint staging entry")
        removable.append((folder, entries))
    # Validate all candidate paths before touching any of them; do not use recursive deletion.
    for folder, entries in removable:
        for entry in entries:
            entry.unlink()
        folder.rmdir()
    try:
        store.rmdir()
    except OSError:
        pass


def _remove_transaction(run_dir: Path, directory: str) -> None:
    paths = [_transaction_path(run_dir, directory, f"{kind}-{name}")
             for kind in ("previous", "next") for name in _TRANSACTION_FILES.values()]
    # Remove the journal first: interruption during cleanup then leaves only inert private files.
    (run_dir / _CHECKPOINT_JOURNAL).unlink(missing_ok=True)
    for path in paths:
        path.unlink(missing_ok=True)
    folder = run_dir / directory
    try:
        folder.rmdir()
        folder.parent.rmdir()
    except OSError:
        pass  # An interrupted preparation can leave an unrelated inert staging directory.


def _validate_transaction(journal: dict, run_dir: Path, cfg: TrainConfig, mcfg: ModelConfig, signature: str) -> None:
    if (not isinstance(journal, dict) or journal.get("schema") != 1 or journal.get("signature") != signature
            or journal.get("phase") not in {"pending", "committed"}
            or journal.get("path_scope") != "run-directory-relative"):
        raise RuntimeError("checkpoint transaction signature/schema/phase differs from this run")
    directory = journal.get("directory")
    for kind in ("previous", "next"):
        entries = journal.get(kind)
        if not isinstance(entries, dict) or set(entries) != set(_TRANSACTION_FILES):
            raise RuntimeError("checkpoint transaction has an invalid artifact registry")
        step = journal.get(f"{kind}_step")
        if step is None and kind == "previous":
            if any(value is not None for value in entries.values()):
                raise RuntimeError("absent baseline transaction must have no previous artifacts")
            # Validate the directory even when all previous entries are absent.
            _transaction_path(run_dir, directory, "previous-last.pt")
            continue
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= cfg.max_steps:
            raise RuntimeError("checkpoint transaction has an invalid step")
        for label, entry in entries.items():
            expected = f"{kind}-{_TRANSACTION_FILES[label]}"
            if (not isinstance(entry, dict) or set(entry) != {"file", "sha256"} or entry["file"] != expected
                    or not isinstance(entry["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
                raise RuntimeError("checkpoint transaction has an invalid file/hash")
            path = _transaction_path(run_dir, directory, entry["file"])
            if not path.is_file() or _file_sha256(path) != entry["sha256"]:
                raise RuntimeError("checkpoint transaction snapshot is missing or failed its checksum")
        last = _safe_load_checkpoint(_transaction_path(run_dir, directory, entries["last"]["file"]), "transaction last")
        _validate_last_state(last, cfg, mcfg, signature, require_complete=False)
        if last["completed_steps"] != step:
            raise RuntimeError("checkpoint transaction step differs from saved progress")
        best = _safe_load_checkpoint(_transaction_path(run_dir, directory, entries["best"]["file"]), "transaction best")
        _validate_best_state(best, mcfg, signature, last, cfg.max_steps)
        _validate_metrics(_transaction_path(run_dir, directory, entries["metrics"]["file"]), cfg, completed_steps=step)
    if journal["previous_step"] is not None and journal["next_step"] < journal["previous_step"]:
        raise RuntimeError("checkpoint transaction would move progress backwards")
    if journal["previous_step"] is None and journal["next_step"] != 0:
        raise RuntimeError("absent checkpoint transaction baseline is only valid at step zero")
    for name in _TRANSACTION_FILES.values():
        public = run_dir / name
        if public.is_symlink() or not public.resolve().is_relative_to(run_dir.resolve()):
            raise RuntimeError("unsafe public checkpoint transaction destination")


def _recover_checkpoint_transaction(run_dir: Path, cfg: TrainConfig, mcfg: ModelConfig, signature: str) -> bool:
    """Recover only a verified in-flight publication, never unrelated artifact corruption."""
    path = run_dir / _CHECKPOINT_JOURNAL
    if not path.exists():
        return False
    if path.is_symlink():
        raise RuntimeError("checkpoint transaction journal cannot be a symlink")
    journal = read_json(path)
    _validate_transaction(journal, run_dir, cfg, mcfg, signature)
    selected = "next" if journal["phase"] == "committed" else "previous"
    # Snapshot files stay intact throughout recovery, so interruption here is repeatable.
    for label, name in _TRANSACTION_FILES.items():
        entry = journal[selected][label]
        if entry is None:
            (run_dir / name).unlink(missing_ok=True)
        else:
            _atomic_snapshot(_transaction_path(run_dir, journal["directory"], entry["file"]),
                             run_dir / name, hardlink=label != "metrics")
    _remove_transaction(run_dir, journal["directory"])
    return True


def _publish_checkpoint_transaction(run_dir: Path, cfg: TrainConfig, mcfg: ModelConfig, signature: str,
                                    state: dict, best: dict | None, evaluation: dict | None,
                                    previous_step: int | None) -> None:
    """A fsynced write-ahead journal makes best/latest/metrics one recoverable generation.

    This handles process interruption. Filesystem/power-loss durability across directory
    operations is platform-dependent and is not promised by this protocol.
    """
    _clean_inert_transaction_staging(run_dir)
    directory = f"{_CHECKPOINT_STORE}/tx-{uuid.uuid4().hex}"
    journal = {"schema": 1, "signature": signature, "phase": "pending", "path_scope": "run-directory-relative",
               "directory": directory, "previous_step": previous_step, "next_step": state["completed_steps"],
               "previous": {}, "next": {}}
    for label, name in _TRANSACTION_FILES.items():
        prior = _transaction_path(run_dir, directory, f"previous-{name}")
        future = _transaction_path(run_dir, directory, f"next-{name}")
        public = run_dir / name
        if previous_step is None:
            journal["previous"][label] = None
        else:
            if label == "metrics":
                atomic_write_text(prior, _metrics_prefix(public, previous_step))
            else:
                _atomic_snapshot(public, prior, hardlink=True)
            journal["previous"][label] = {"file": prior.name, "sha256": _file_sha256(prior)}
        if label == "last":
            atomic_torch_save(state, future)
        elif label == "best":
            if best is None:
                _atomic_snapshot(public, future, hardlink=True)
            else:
                atomic_torch_save(best, future)
        else:
            contents = public.read_text(encoding="utf-8") if public.exists() else ""
            if evaluation is not None:
                contents += json.dumps(evaluation, sort_keys=True, allow_nan=False) + "\n"
            atomic_write_text(future, contents)
        journal["next"][label] = {"file": future.name, "sha256": _file_sha256(future)}
    _validate_transaction(journal, run_dir, cfg, mcfg, signature)
    atomic_write_json(run_dir / _CHECKPOINT_JOURNAL, journal)
    for label, name in _TRANSACTION_FILES.items():
        _atomic_snapshot(_transaction_path(run_dir, directory, journal["next"][label]["file"]),
                         run_dir / name, hardlink=label != "metrics")
    journal["phase"] = "committed"
    atomic_write_json(run_dir / _CHECKPOINT_JOURNAL, journal)
    _remove_transaction(run_dir, directory)


def _validate_completed_run(
    cfg: TrainConfig,
    mcfg: ModelConfig,
    run_dir: Path,
    signature: str,
    data_provenance: dict[str, Any],
    code_provenance: dict[str, Any],
    runtime_provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise RuntimeError("completed manifest must be a JSON object")
    # Completed is irreversible. Validate its registered set before reading/recovering any member.
    if manifest.get("status") == "completed":
        for label, filename in ARTIFACT_FILES.items():
            path = run_dir / filename
            if not path.is_file():
                raise RuntimeError(f"required {label} artifact is missing: {path}")
    result = read_json(run_dir / "result.json")
    if not isinstance(result, dict):
        raise RuntimeError("completed result must be a JSON object")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("signature") != signature:
        raise RuntimeError("completed manifest signature/schema does not match current provenance")
    if manifest.get("run_id") != cfg.run_id or manifest.get("variant") != mcfg.name or manifest.get("ratio") != mcfg.ratio:
        raise RuntimeError("completed manifest identity does not match this run")
    if manifest.get("status") not in {"running", "completed"}:
        raise RuntimeError("completed manifest has an invalid status")
    if (manifest.get("data") != data_provenance or manifest.get("code") != code_provenance
            or manifest.get("runtime") != runtime_provenance):
        raise RuntimeError("completed manifest provenance does not match current data/code/runtime")
    if result.get("signature") != signature or result.get("run_id") != cfg.run_id:
        raise RuntimeError("completed result signature/run ID does not match this run")
    if result.get("name") != mcfg.name or result.get("ratio") != mcfg.ratio:
        raise RuntimeError("completed result variant identity does not match this run")
    _validate_precision_metadata(manifest, cfg)
    _validate_precision_metadata(result, cfg)
    expected_backend = {
        **resolve_scan_backend(cfg.scan_backend, cfg.scan_chunk_size).metadata(),
        "mamba_layers": mcfg.n_mamba_layers,
    }
    expected_paths = {
        "training": expected_backend["paths"]["training"] if cfg.max_steps and mcfg.n_mamba_layers else None,
        "prefill": None, "decode": None,
    }
    if manifest.get("scan_backend") != expected_backend or result.get("scan_backend") != expected_backend:
        raise RuntimeError("completed scan backend metadata does not match the requested execution path")
    if result.get("observed_paths") != expected_paths:
        raise RuntimeError("completed observed scan paths do not match this training workload")
    if (manifest.get("status") == "completed" and manifest.get("observed_paths") != expected_paths
            or manifest.get("status") == "running" and manifest.get("observed_paths") not in (
                expected_paths, {"training": None, "prefill": None, "decode": None},
            )):
        raise RuntimeError("manifest observed scan paths do not match this training workload")
    expected_tokens = _expected_tokens(cfg, cfg.max_steps)
    if result.get("completed_steps") != cfg.max_steps or result.get("tokens_seen") != expected_tokens:
        raise RuntimeError("completed result step/token counts do not match the requested budget")
    if manifest.get("status") == "completed":
        expected_artifacts = ARTIFACT_FILES
        if (manifest.get("completed_steps") != cfg.max_steps or manifest.get("tokens_seen") != expected_tokens
                or manifest.get("artifacts") != expected_artifacts):
            raise RuntimeError("completed manifest counters/artifact registry are inconsistent")
        expected_hashes = manifest.get("artifact_sha256")
        if not isinstance(expected_hashes, dict) or expected_hashes.keys() != expected_artifacts.keys():
            raise RuntimeError("completed manifest is missing artifact checksums")
        for label, filename in expected_artifacts.items():
            path = run_dir / filename
            if not path.is_file():
                raise RuntimeError(f"required {label} artifact is missing: {path}")
            if _file_sha256(path) != expected_hashes[label]:
                raise RuntimeError(f"completed {label} artifact failed its checksum")
    for key, minimum in {
        "params_m": 0.0, "best_val_loss": 0.0, "best_val_ppl": 0.0,
        "avg_tok_per_s": 0.0, "peak_vram_mb": 0.0,
    }.items():
        _require_finite(result.get(key), f"result.{key}", minimum)
    if result["best_val_ppl"] <= 0:
        raise RuntimeError("completed result has a non-positive perplexity")
    if result.get("n_attention") != mcfg.n_attention_layers or result.get("n_mamba") != mcfg.n_mamba_layers:
        raise RuntimeError("completed result layer counts do not match the model config")

    last_state = _safe_load_checkpoint(run_dir / "last.pt", "last")
    _validate_last_state(last_state, cfg, mcfg, signature, require_complete=True)
    best_state = _safe_load_checkpoint(run_dir / "best.pt", "best")
    _validate_best_state(best_state, mcfg, signature, last_state, cfg.max_steps)
    if result["best_val_loss"] != round(float(last_state["best_val_loss"]), 4):
        raise RuntimeError("completed result best loss does not match the last checkpoint")
    if result["best_val_ppl"] != round(math.exp(float(last_state["best_val_loss"])), 2):
        raise RuntimeError("completed result perplexity does not match the last checkpoint")
    if result["peak_vram_mb"] != round(float(last_state["peak_vram_mb"])):
        raise RuntimeError("completed result peak VRAM does not match the last checkpoint")
    _validate_metrics(run_dir / "metrics.jsonl", cfg)
    return manifest, result


class RunLock:
    """Hold a non-blocking OS file lock for one variant run to prevent concurrent writers."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"run is already active: {self.path.parent}") from exc
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.handle is None:
            return
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _validate_intervals(cfg: TrainConfig) -> None:
    precision_policy(cfg.precision)
    _validate_wall_limit(cfg.wall_time_limit_seconds)
    for name in (
        "eval_interval", "eval_iters", "log_interval", "checkpoint_interval",
        "batch_size", "grad_accum", "block_size",
    ):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("max_steps", "warmup_steps"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("lr", "min_lr", "weight_decay", "grad_clip"):
        value = getattr(cfg, name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")
    for name in ("beta1", "beta2"):
        value = getattr(cfg, name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0 <= value < 1):
            raise ValueError(f"{name} must be a finite number in [0, 1)")
    for name in ("seed", "model_seed", "data_seed", "eval_seed"):
        value = getattr(cfg, name)
        if value is None and name != "seed":
            continue
        if (
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32
        ):
            raise ValueError(f"{name} must be an integer between 0 and 2^32 - 1")


def run(cfg: TrainConfig, *, stop_control: TrainingStopControl | None = None) -> dict[str, Any]:
    """Train, resume, or skip one variant and return its completed headline metrics."""
    _validate_intervals(cfg)
    control = stop_control or _ACTIVE_STOP_CONTROL.get() or TrainingStopControl(cfg.wall_time_limit_seconds)
    if control.wall_time_limit_seconds != cfg.wall_time_limit_seconds:
        raise ValueError("shared stop controller wall policy differs from the signed training config")
    # Resolve optional engines before creating a run directory or allocating model weights. An
    # unavailable fused request fails explicitly and cannot be relabeled as a portable run.
    resolve_scan_backend(cfg.scan_backend, cfg.scan_chunk_size)
    if cfg.scan_backend == "fused_mamba" and torch.device(cfg.device).type != "cuda":
        raise BackendUnavailableError("fused_mamba training requires a CUDA device; CPU fallback is forbidden")
    mcfg = load_model_config(cfg.model_config)
    run_dir = variant_run_dir(cfg, mcfg.name)
    with RunLock(run_dir / ".run.lock"):
        with _precision_runtime(cfg):
            return _run_locked(cfg, mcfg, run_dir, control)


def _run_locked(cfg: TrainConfig, mcfg: ModelConfig, run_dir: Path, control: TrainingStopControl) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    result_path = run_dir / "result.json"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.jsonl"
    meta = load_meta(cfg.data_dir)
    if meta["vocab_size"] != mcfg.vocab_size:
        raise ValueError("tokenizer/model vocab mismatch")
    data_provenance = _data_provenance(cfg.data_dir, meta)
    code_provenance = _code_provenance()
    runtime_provenance = _runtime_provenance()
    signature = _canonical_hash(_signature_payload(
        cfg, mcfg, data_provenance, code_provenance, runtime_provenance,
    ))

    manifest_on_disk = read_json(manifest_path) if manifest_path.exists() else None
    if manifest_on_disk is not None and not isinstance(manifest_on_disk, dict):
        raise RuntimeError(f"run manifest must be a JSON object: {manifest_path}")
    # Check provenance BEFORE any recovery writes. A journal never authorizes another run/code.
    if manifest_on_disk is not None:
        if (manifest_on_disk.get("signature") != signature or manifest_on_disk.get("data") != data_provenance
                or manifest_on_disk.get("code") != code_provenance or manifest_on_disk.get("runtime") != runtime_provenance):
            raise RuntimeError(f"run configuration differs from existing manifest (provenance/signature mismatch) in {run_dir}")
        _validate_precision_metadata(manifest_on_disk, cfg)
    if cfg.resume and (run_dir / _CHECKPOINT_JOURNAL).exists():
        if manifest_on_disk is None or manifest_on_disk.get("status") == "completed":
            raise RuntimeError("checkpoint transaction cannot repair an absent or completed manifest")
        _recover_checkpoint_transaction(run_dir, cfg, mcfg, signature)
    if manifest_on_disk is not None and manifest_on_disk.get("status") == "completed":
        if not cfg.resume:
            raise FileExistsError(f"refusing to overwrite completed run directory: {run_dir}")
        manifest, result = _validate_completed_run(
            cfg, mcfg, run_dir, signature, data_provenance, code_provenance, runtime_provenance,
        )
        print(f"skip completed: {cfg.run_id}/{mcfg.name}")
        return result

    if result_path.exists() and cfg.resume:
        manifest, result = _validate_completed_run(
            cfg, mcfg, run_dir, signature, data_provenance, code_provenance, runtime_provenance,
        )
        # The result is written before the manifest is marked complete. If the process stopped in
        # that narrow window, the fully validated artifact set can finish the manifest transaction.
        if manifest.get("status") != "completed":
            manifest.update({
                "status": "completed", "updated_at": utc_now(), "completed_at": utc_now(),
                "completed_steps": result["completed_steps"], "tokens_seen": result["tokens_seen"],
                "observed_paths": result["observed_paths"],
                "artifacts": ARTIFACT_FILES,
                "artifact_sha256": {
                    label: _file_sha256(run_dir / filename) for label, filename in ARTIFACT_FILES.items()
                },
            })
            atomic_write_json(manifest_path, manifest)
        print(f"skip completed: {cfg.run_id}/{mcfg.name}")
        return result
    existing_artifacts = [path for path in run_dir.iterdir() if path.name != ".run.lock"]
    if existing_artifacts and not cfg.resume:
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")

    model_seed = cfg.seed if cfg.model_seed is None else cfg.model_seed
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if cfg.precision == "bfloat16":
        torch.backends.cuda.matmul.allow_tf32 = True

    splits = {s: load_split(cfg.data_dir, s) for s in ("train", "val")}

    if manifest_on_disk is not None:
        manifest = manifest_on_disk
        if (manifest.get("signature") != signature or manifest.get("data") != data_provenance
                or manifest.get("code") != code_provenance or manifest.get("runtime") != runtime_provenance):
            raise RuntimeError(f"run configuration differs from existing manifest in {run_dir}")
        _validate_precision_metadata(manifest, cfg)
    else:
        manifest = _new_manifest(
            cfg, mcfg, signature, data_provenance, code_provenance, runtime_provenance,
        )

    model = HybridLM(mcfg).to(cfg.device)
    backend_metadata = model.configure_scan_backend(cfg.scan_backend, cfg.scan_chunk_size)
    if manifest_on_disk is None:
        manifest["scan_backend"] = backend_metadata
        manifest["training_precision"] = _precision_metadata(cfg)
        # Finalization records paths that this workload actually exercised. This trainer never
        # runs prefill/decode, so those remain null rather than presenting them as measured paths.
        manifest["observed_paths"] = {"training": None, "prefill": None, "decode": None}
        atomic_write_json(manifest_path, manifest)
    elif manifest.get("scan_backend") != backend_metadata:
        raise RuntimeError(f"scan backend metadata differs from existing manifest in {run_dir}")
    model.grad_checkpointing = cfg.grad_checkpointing
    print(f"{mcfg.name}: {model.num_params()/1e6:.2f}M params  "
          f"(eff. batch {cfg.batch_size*cfg.grad_accum} x {cfg.block_size} tokens)")

    fused = torch.device(cfg.device).type == "cuda" and torch.cuda.is_available()
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(float(cfg.beta1), float(cfg.beta2)),
        weight_decay=cfg.weight_decay, fused=fused,
    )
    train_generator = make_batch_generator(cfg.seed if cfg.data_seed is None else cfg.data_seed, "train")

    completed_steps = 0
    tokens_seen = 0
    best_val = float("inf")
    train_seconds = 0.0
    eval_seconds = 0.0
    prior_peak_vram_mb = 0.0
    if last_path.exists() and cfg.resume:
        state = _load_resume_checkpoint(last_path, cfg, mcfg, signature, model, optim, train_generator)
        if not metrics_path.is_file():
            raise RuntimeError(f"cannot resume without metrics artifact: {metrics_path}")
        best_state = _safe_load_checkpoint(best_path, "best")
        _validate_best_state(best_state, mcfg, signature, state, cfg.max_steps)
        completed_steps = int(state["completed_steps"])
        tokens_seen = int(state["tokens_seen"])
        best_val = float(state["best_val_loss"])
        train_seconds = float(state["train_seconds"])
        eval_seconds = float(state["eval_seconds"])
        prior_peak_vram_mb = float(state["peak_vram_mb"])
        reconcile_metrics(metrics_path, completed_steps, cfg=cfg)
        print(f"resume: {cfg.run_id}/{mcfg.name} from {completed_steps}/{cfg.max_steps} steps")
    elif metrics_path.exists() and metrics_path.stat().st_size > 0:
        raise RuntimeError(f"cannot resume progress metrics without last checkpoint: {metrics_path}")
    durable_step = completed_steps if last_path.exists() else None
    pending_best = None
    pending_evaluation = None
    if manifest.get("status") == "stopped":
        manifest.update({"status": "running", "updated_at": utc_now(), "resumed_at": utc_now()})
        atomic_write_json(manifest_path, manifest)

    wb = None
    if cfg.wandb:
        import wandb
        wb = wandb.init(
            project=cfg.wandb_project, name=f"{cfg.run_id}-{variant_slug(mcfg.name)}",
            config={**asdict(cfg), **asdict(mcfg)}, resume="allow",
        )

    if torch.device(cfg.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics = MetricsWriter(metrics_path)

    def observed_peak_vram_mb() -> float:
        current = torch.cuda.max_memory_allocated() / 1e6 if torch.device(cfg.device).type == "cuda" else 0.0
        return max(prior_peak_vram_mb, current)

    def save_last() -> None:
        nonlocal durable_step, pending_best, pending_evaluation
        state = _checkpoint_state(
            model, optim, cfg, mcfg, signature, completed_steps, tokens_seen, best_val,
            train_seconds, eval_seconds, observed_peak_vram_mb(), train_generator,
        )
        _publish_checkpoint_transaction(run_dir, cfg, mcfg, signature, state,
                                        pending_best, pending_evaluation, durable_step)
        durable_step = completed_steps
        pending_best = None
        pending_evaluation = None

    def evaluate(step: int, applied_lr: float) -> None:
        nonlocal best_val, eval_seconds, pending_best, pending_evaluation
        e0 = time.perf_counter()
        losses = estimate_loss(model, splits, cfg)
        try:
            ppl = math.exp(losses["val"])
        except OverflowError as error:
            raise FloatingPointError(
                f"validation perplexity overflow at completed step {step}; "
                f"validation loss {losses['val']!r}. Evaluation metrics and best checkpoint were not published."
            ) from error
        if not math.isfinite(ppl):
            raise FloatingPointError(
                f"non-finite validation perplexity at completed step {step}; "
                f"validation loss {losses['val']!r}. Evaluation metrics and best checkpoint were not published."
            )
        eval_seconds += time.perf_counter() - e0
        record = {
            "event": "eval", "step": step, "tokens_seen": tokens_seen,
            "train_loss": losses["train"], "val_loss": losses["val"], "val_ppl": ppl, "lr": applied_lr,
        }
        # Publish this event and any new best only together with the matching resumable state.
        pending_evaluation = record
        print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
              f"| ppl {ppl:.1f} | lr {applied_lr:.2e}")
        if wb:
            wb.log({"val/loss": losses["val"], "val/ppl": ppl,
                    "train/eval_loss": losses["train"]}, step=step)
        if losses["val"] < best_val:
            best_val = losses["val"]
            pending_best = {
                "schema": CHECKPOINT_SCHEMA, "model": model.state_dict(),
                "model_config": asdict(mcfg), "step": step, "val_loss": best_val,
                "signature": signature,
            }

    # A new run gets a baseline at step zero. A resumed run already has the evaluation/checkpoint
    # associated with its completed step, so it continues directly with the next optimizer update.
    if completed_steps == 0 and not last_path.exists():
        evaluate(0, cosine_lr(0, cfg))
        save_last()

    device_type = torch.device(cfg.device).type
    while completed_steps < cfg.max_steps:
        if control.reason() is not None:
            if durable_step != completed_steps:
                save_last()
            stopped = {
                "status": "stopped", "certified": False, "name": mcfg.name, "ratio": mcfg.ratio,
                "run_id": cfg.run_id, "signature": signature, "completed_steps": completed_steps,
                "tokens_seen": tokens_seen, "checkpoint_step": durable_step,
                "observed_paths": {
                    "training": backend_metadata["paths"]["training"] if completed_steps and mcfg.n_mamba_layers else None,
                    "prefill": None, "decode": None,
                },
                "stop": {**control.snapshot(), "boundary": "complete checkpoint transaction"},
            }
            manifest.update({"status": "stopped", "updated_at": utc_now(),
                             "completed_steps": completed_steps, "tokens_seen": tokens_seen,
                             "observed_paths": stopped["observed_paths"],
                             "stop": stopped["stop"]})
            atomic_write_json(manifest_path, manifest)
            if wb:
                wb.finish()
            print(f"stopped durably: {cfg.run_id}/{mcfg.name} at {completed_steps}/{cfg.max_steps} steps")
            del model, optim
            if device_type == "cuda":
                torch.cuda.empty_cache()
            return stopped
        step_index = completed_steps
        lr = cosine_lr(step_index, cfg)

        step_start = time.perf_counter()
        loss_sum = torch.zeros((), device=cfg.device)
        for microbatch_index in range(cfg.grad_accum):
            x, y = get_batch(
                splits["train"], cfg.block_size, cfg.batch_size, cfg.device,
                generator=train_generator,
            )
            with _autocast(cfg):
                _, loss = model(x, y)
            if not torch.isfinite(loss.detach()).all().item():
                optim.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite training loss at optimizer step {step_index + 1}, "
                    f"microbatch {microbatch_index + 1}/{cfg.grad_accum}; update skipped. "
                    f"Resume from the last durable checkpoint: {last_path}"
                )
            (loss / cfg.grad_accum).backward()
            loss_sum += loss.detach()
        train_loss = (loss_sum / cfg.grad_accum).item()
        if not math.isfinite(train_loss):
            optim.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite accumulated training loss at optimizer step {step_index + 1}, "
                f"after microbatch {cfg.grad_accum}/{cfg.grad_accum}; update skipped. "
                f"Resume from the last durable checkpoint: {last_path}"
            )
        try:
            # Reject the norm before clipping can turn an invalid gradient into another invalid
            # value. This also detects finite individual gradients whose combined norm overflows.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip, error_if_nonfinite=True,
            )
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            optim.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite training gradient norm at optimizer step {step_index + 1}, "
                f"after microbatch {cfg.grad_accum}/{cfg.grad_accum}; update skipped. "
                f"Resume from the last durable checkpoint: {last_path}"
            ) from error
        # The learning rate changes only once the update passes its numerical checks. Rejected
        # updates therefore leave both model weights and optimizer state at their previous values.
        for group in optim.param_groups:
            group["lr"] = lr
        optim.step()
        optim.zero_grad(set_to_none=True)
        if device_type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start
        train_seconds += elapsed
        completed_steps += 1
        step_tokens = cfg.batch_size * cfg.grad_accum * cfg.block_size
        tokens_seen += step_tokens
        step_tps = step_tokens / elapsed
        metrics.append({
            "event": "train", "step": completed_steps, "tokens_seen": tokens_seen,
            "loss": train_loss, "grad_norm": float(grad_norm), "lr": lr,
            "step_seconds": elapsed, "tok_per_s": step_tps,
        })
        if (completed_steps - 1) % cfg.log_interval == 0:
            print(f"  step {completed_steps:5d} | loss {train_loss:.3f} | "
                  f"gnorm {float(grad_norm):.2f} | {step_tps:.0f} tok/s")
            if wb:
                wb.log({"train/loss": train_loss, "train/grad_norm": float(grad_norm),
                        "lr": lr, "throughput/tok_per_s": step_tps}, step=completed_steps)

        do_eval = completed_steps % cfg.eval_interval == 0 or completed_steps == cfg.max_steps
        if do_eval:
            evaluate(completed_steps, lr)
        if do_eval or completed_steps % cfg.checkpoint_interval == 0:
            save_last()

    _validate_metrics(metrics_path, cfg)
    peak_vram = observed_peak_vram_mb()
    avg_tps = tokens_seen / train_seconds if train_seconds else 0.0
    result = {
        "name": mcfg.name, "ratio": mcfg.ratio,
        "params_m": round(model.num_params() / 1e6, 2),
        "n_attention": mcfg.n_attention_layers, "n_mamba": mcfg.n_mamba_layers,
        "best_val_loss": round(best_val, 4), "best_val_ppl": round(math.exp(best_val), 2),
        "avg_tok_per_s": round(avg_tps), "peak_vram_mb": round(peak_vram),
        "tokens_seen": tokens_seen, "completed_steps": completed_steps,
        "run_id": cfg.run_id, "signature": signature,
        "scan_backend": backend_metadata,
        "training_precision": _precision_metadata(cfg),
        "observed_paths": {
            "training": backend_metadata["paths"]["training"] if completed_steps and mcfg.n_mamba_layers else None,
            "prefill": None, "decode": None,
        },
    }
    atomic_write_json(result_path, result)
    manifest.update({
        "status": "completed", "updated_at": utc_now(), "completed_at": utc_now(),
        "completed_steps": completed_steps, "tokens_seen": tokens_seen,
        "observed_paths": result["observed_paths"],
        "artifacts": ARTIFACT_FILES,
        "artifact_sha256": {
            label: _file_sha256(run_dir / filename) for label, filename in ARTIFACT_FILES.items()
        },
    })
    if cfg.wall_time_limit_seconds is not None:
        manifest["invocation_wall_time"] = control.snapshot()
    atomic_write_json(manifest_path, manifest)
    print(f"done. {mcfg.name}: best val ppl {result['best_val_ppl']} | "
          f"{result['avg_tok_per_s']} tok/s | {result['peak_vram_mb']} MB peak")
    if wb:
        wb.finish()
    del model, optim
    if device_type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for field, default in asdict(TrainConfig()).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(arg, action=argparse.BooleanOptionalAction, default=default, dest=field)
        else:
            value_type = (int if field in {"model_seed", "data_seed", "eval_seed"}
                          else float if field == "wall_time_limit_seconds" else type(default))
            choices = ("bfloat16", "float32") if field == "precision" else None
            ap.add_argument(arg, type=value_type, default=default, dest=field, choices=choices)
    cfg = TrainConfig(**vars(ap.parse_args()))
    control = TrainingStopControl(cfg.wall_time_limit_seconds)
    with cooperative_stop_signals(control), training_stop_scope(control):
        run(cfg)


if __name__ == "__main__":
    main()
