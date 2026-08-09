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
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.dataset import get_batch, load_meta, load_split
from src.model.config import ModelConfig
from src.model.lm import HybridLM

CHECKPOINT_SCHEMA = 1
MANIFEST_SCHEMA = 1
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STREAM_SEED_OFFSETS = {"train": 0, "eval_train": 1, "eval_val": 2}
ARTIFACT_FILES = {"best": "best.pt", "last": "last.pt", "metrics": "metrics.jsonl", "result": "result.json"}
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
    device: str = "cuda"
    grad_checkpointing: bool = True
    wandb: bool = False
    wandb_project: str = "mamba-hybrid-lm"


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


@torch.no_grad()
def estimate_loss(model, splits, cfg: TrainConfig) -> dict[str, float]:
    """Evaluate the same fixed windows every time without advancing the training sampler."""
    model.eval()
    out: dict[str, float] = {}
    device_type = torch.device(cfg.device).type
    for name, data in splits.items():
        generator = make_batch_generator(cfg.seed, f"eval_{name}")
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device, generator=generator)
            with torch.autocast(device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
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


def reconcile_metrics(path: Path, completed_steps: int) -> None:
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
    atomic_write_text(path, "".join(f"{line}\n" for line in kept))


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
    train_config = asdict(cfg)
    for key in ("resume", "wandb", "wandb_project"):
        train_config.pop(key)
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
    for package in ("einops", "datasets", "tokenizers", "transformers", "numpy", "wandb", "tqdm"):
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


def _validate_metrics(path: Path, cfg: TrainConfig) -> None:
    if not path.is_file():
        raise RuntimeError(f"required metrics artifact is missing: {path}")
    expected_events: list[tuple[str, int]] = [("eval", 0)]
    for step in range(1, cfg.max_steps + 1):
        expected_events.append(("train", step))
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            expected_events.append(("eval", step))
    train_fields = {"event", "step", "tokens_seen", "loss", "grad_norm", "lr", "step_seconds", "tok_per_s"}
    eval_fields = {"event", "step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "lr"}
    seen_events = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
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
    for name in (
        "eval_interval", "eval_iters", "log_interval", "checkpoint_interval",
        "batch_size", "grad_accum", "block_size",
    ):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.max_steps < 0 or cfg.warmup_steps < 0:
        raise ValueError("max_steps and warmup_steps must be non-negative")
    for name in ("lr", "min_lr", "weight_decay", "grad_clip"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def run(cfg: TrainConfig) -> dict[str, Any]:
    """Train, resume, or skip one variant and return its completed headline metrics."""
    _validate_intervals(cfg)
    mcfg = load_model_config(cfg.model_config)
    run_dir = variant_run_dir(cfg, mcfg.name)
    with RunLock(run_dir / ".run.lock"):
        return _run_locked(cfg, mcfg, run_dir)


def _run_locked(cfg: TrainConfig, mcfg: ModelConfig, run_dir: Path) -> dict[str, Any]:
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

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    splits = {s: load_split(cfg.data_dir, s) for s in ("train", "val")}

    if manifest_on_disk is not None:
        manifest = manifest_on_disk
        if (manifest.get("signature") != signature or manifest.get("data") != data_provenance
                or manifest.get("code") != code_provenance or manifest.get("runtime") != runtime_provenance):
            raise RuntimeError(f"run configuration differs from existing manifest in {run_dir}")
    else:
        manifest = _new_manifest(
            cfg, mcfg, signature, data_provenance, code_provenance, runtime_provenance,
        )
        atomic_write_json(manifest_path, manifest)

    model = HybridLM(mcfg).to(cfg.device)
    model.grad_checkpointing = cfg.grad_checkpointing
    print(f"{mcfg.name}: {model.num_params()/1e6:.2f}M params  "
          f"(eff. batch {cfg.batch_size*cfg.grad_accum} x {cfg.block_size} tokens)")

    fused = torch.device(cfg.device).type == "cuda" and torch.cuda.is_available()
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay, fused=fused,
    )
    train_generator = make_batch_generator(cfg.seed, "train")

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
        reconcile_metrics(metrics_path, completed_steps)
        print(f"resume: {cfg.run_id}/{mcfg.name} from {completed_steps}/{cfg.max_steps} steps")
    elif metrics_path.exists() and metrics_path.stat().st_size > 0:
        raise RuntimeError(f"cannot resume progress metrics without last checkpoint: {metrics_path}")

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
        state = _checkpoint_state(
            model, optim, cfg, mcfg, signature, completed_steps, tokens_seen, best_val,
            train_seconds, eval_seconds, observed_peak_vram_mb(), train_generator,
        )
        atomic_torch_save(state, last_path)

    def evaluate(step: int, applied_lr: float) -> None:
        nonlocal best_val, eval_seconds
        e0 = time.perf_counter()
        losses = estimate_loss(model, splits, cfg)
        ppl = math.exp(losses["val"])
        eval_seconds += time.perf_counter() - e0
        record = {
            "event": "eval", "step": step, "tokens_seen": tokens_seen,
            "train_loss": losses["train"], "val_loss": losses["val"], "val_ppl": ppl, "lr": applied_lr,
        }
        metrics.append(record)
        print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
              f"| ppl {ppl:.1f} | lr {applied_lr:.2e}")
        if wb:
            wb.log({"val/loss": losses["val"], "val/ppl": ppl,
                    "train/eval_loss": losses["train"]}, step=step)
        if losses["val"] < best_val:
            best_val = losses["val"]
            atomic_torch_save({
                "schema": CHECKPOINT_SCHEMA, "model": model.state_dict(),
                "model_config": asdict(mcfg), "step": step, "val_loss": best_val,
                "signature": signature,
            }, best_path)

    # A new run gets a baseline at step zero. A resumed run already has the evaluation/checkpoint
    # associated with its completed step, so it continues directly with the next optimizer update.
    if completed_steps == 0 and not last_path.exists():
        evaluate(0, cosine_lr(0, cfg))
        save_last()

    device_type = torch.device(cfg.device).type
    while completed_steps < cfg.max_steps:
        step_index = completed_steps
        lr = cosine_lr(step_index, cfg)
        for group in optim.param_groups:
            group["lr"] = lr

        step_start = time.perf_counter()
        loss_sum = torch.zeros((), device=cfg.device)
        for _ in range(cfg.grad_accum):
            x, y = get_batch(
                splits["train"], cfg.block_size, cfg.batch_size, cfg.device,
                generator=train_generator,
            )
            with torch.autocast(device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
        train_loss = (loss_sum / cfg.grad_accum).item()
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
    }
    atomic_write_json(result_path, result)
    manifest.update({
        "status": "completed", "updated_at": utc_now(), "completed_at": utc_now(),
        "completed_steps": completed_steps, "tokens_seen": tokens_seen,
        "artifacts": ARTIFACT_FILES,
        "artifact_sha256": {
            label: _file_sha256(run_dir / filename) for label, filename in ARTIFACT_FILES.items()
        },
    })
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
            ap.add_argument(arg, type=type(default), default=default, dest=field)
    run(TrainConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
