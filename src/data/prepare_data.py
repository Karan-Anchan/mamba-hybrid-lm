"""Build verified uint16 token memmaps through a run-specific staging directory.

The authoritative OpenWebText path is token-targeted: validation consumes the deterministic head of
one pinned streaming traversal, then training continues from the next document until both targets are
met. Whole documents (plus EOT) are retained, so exact counts may exceed a target by the final document.

    python -m src.data.prepare_data --dataset openwebtext \
      --tokenizer data/tokenizer/openwebtext.json --out-dir data/openwebtext-5b \
      --run-id owt-5b-20260809 --revision 79d93d786212f7344586290adb811d4ae6a1762c \
      --train-tokens 5000000000 --val-tokens 10000000

The older document-count flags remain available for small debug datasets. Existing outputs are never
replaced. A failed stream leaves its partial stage in place when the filesystem permits; updating the
interrupted manifest is best effort. Replay-based resume is deliberately refused because a remote
streaming source cannot prove an identical cursor cheaply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

import numpy as np
from tokenizers import Tokenizer

from src.data.train_tokenizer import EOT

SCHEMA_VERSION = 3
OWT_SOURCE = "Skylion007/openwebtext"
OWT_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
TRAVERSAL = "sequential streaming order; no shuffle; validation then training"
DEFAULT_TRAIN_DOCS = 50_000
DEFAULT_VAL_DOCS = 22_000
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_FILES = {"train": "train.bin", "val": "val.bin", "meta": "meta.json"}
_BUILD_IDENTITY_KEYS = ("schema", "config", "source", "tokenizer", "tool", "runtime", "traversal")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class PrepareConfig:
    dataset: str
    tokenizer: str
    out_dir: str
    run_id: str
    source: str
    revision: str
    split: str = "train"
    source_config: str | None = None
    text_field: str = "text"
    train_tokens: int | None = None
    val_tokens: int | None = None
    train_docs: int | None = None
    val_docs: int | None = None
    progress_docs: int = 1_000
    reuse_existing: bool = False

    @property
    def mode(self) -> str:
        return "tokens" if self.train_tokens is not None else "docs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_run_id(run_id: str) -> str:
    if (not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."} or run_id.endswith(".")
            or run_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED):
        raise ValueError("run_id must use only alphanumerics, '.', '_' and '-', without path syntax")
    return run_id


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _tool_provenance() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "git": _git_provenance()}


def _runtime_provenance() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "tokenizers", "datasets"):
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _validate_config(cfg: PrepareConfig) -> None:
    validate_run_id(cfg.run_id)
    if cfg.dataset not in {"tinystories", "openwebtext"}:
        raise ValueError(f"unsupported dataset: {cfg.dataset}")
    if cfg.dataset == "openwebtext" and not _FULL_COMMIT_RE.fullmatch(cfg.revision):
        raise ValueError("OpenWebText requires an immutable 40-character hexadecimal commit revision")
    token_mode = cfg.train_tokens is not None or cfg.val_tokens is not None
    doc_mode = cfg.train_docs is not None or cfg.val_docs is not None
    if token_mode and doc_mode:
        raise ValueError("token targets and document targets are mutually exclusive")
    if token_mode and (cfg.train_tokens is None or cfg.val_tokens is None):
        raise ValueError("both --train-tokens and --val-tokens are required")
    if doc_mode and (cfg.train_docs is None or cfg.val_docs is None):
        raise ValueError("both --train-docs and --val-docs are required")
    targets = (cfg.train_tokens, cfg.val_tokens) if token_mode else (cfg.train_docs, cfg.val_docs)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in targets):
        raise ValueError("all selected targets must be positive integers")
    if cfg.progress_docs <= 0:
        raise ValueError("progress_docs must be positive")
    out = Path(cfg.out_dir).resolve()
    if out == out.parent:
        raise ValueError("out_dir cannot be a filesystem root")
    if out.name.endswith(".") or out.name.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError("out_dir has an unsafe or reserved final path component")


class BuildLock:
    """Serialize builders targeting the same final directory."""

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
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another preparation process is active for {self.path.parent}") from exc
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


def _tokenizer_identity(path: Path, tok: Tokenizer, raw: bytes) -> dict[str, Any]:
    eot_id = tok.token_to_id(EOT)
    vocab_size = tok.get_vocab_size()
    vocab_ids = set(tok.get_vocab().values())
    if eot_id is None:
        raise ValueError(f"tokenizer does not define required token {EOT!r}")
    if (not 0 < vocab_size <= 65_536 or vocab_ids != set(range(vocab_size))
            or not 0 <= eot_id < vocab_size):
        raise ValueError("tokenizer vocabulary must use dense IDs that fit uint16")
    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "vocab_size": vocab_size, "eot_token": EOT, "eot_id": eot_id,
    }


def _load_tokenizer(path: Path) -> tuple[Tokenizer, dict[str, Any]]:
    """Load from the same bytes that are hashed, so encoding and provenance cannot diverge."""
    try:
        resolved = path.resolve(strict=True)
        raw = resolved.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"tokenizer file not found or unreadable: {path}") from exc
    try:
        tok = Tokenizer.from_str(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid tokenizer artifact: {resolved}") from exc
    identity = _tokenizer_identity(resolved, tok, raw)
    return tok, identity


def _build_identity(
    cfg: PrepareConfig,
    tokenizer_identity: dict[str, Any],
    tool: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    config = asdict(cfg)
    config.pop("reuse_existing")
    identity = {
        "schema": SCHEMA_VERSION,
        "config": config,
        "source": {
            "repository": cfg.source,
            "revision": cfg.revision,
            "split": cfg.split,
            "config": cfg.source_config,
            "text_field": cfg.text_field,
        },
        "tokenizer": tokenizer_identity,
        "tool": tool,
        "runtime": runtime,
        "traversal": TRAVERSAL,
    }
    return identity, canonical_hash(identity)


def hf_iterator_factory(cfg: PrepareConfig) -> Callable[[str], Iterator[str]]:
    def iterator(split: str) -> Iterator[str]:
        from datasets import load_dataset

        kwargs: dict[str, Any] = {
            "split": split,
            "streaming": True,
            "revision": cfg.revision,
        }
        if cfg.source_config is None:
            dataset = load_dataset(cfg.source, **kwargs)
        else:
            dataset = load_dataset(cfg.source, cfg.source_config, **kwargs)
        for example in dataset:
            yield example[cfg.text_field]

    return iterator


def _target_reached(cfg: PrepareConfig, partition: str, docs: int, tokens: int) -> bool:
    target = getattr(cfg, f"{partition}_{cfg.mode}")
    current = tokens if cfg.mode == "tokens" else docs
    return current >= target


def _write_partition(
    cfg: PrepareConfig,
    partition: str,
    source_split: str,
    source_start: int,
    iterator: Iterator[str],
    tok: Tokenizer,
    eot_id: int,
    path: Path,
    progress: dict[str, dict[str, int]],
    progress_callback: Callable[[], None],
) -> dict[str, Any]:
    docs = 0
    tokens = 0
    with path.open("xb") as handle:
        while not _target_reached(cfg, partition, docs, tokens):
            try:
                text = next(iterator)
            except StopIteration as exc:
                target = getattr(cfg, f"{partition}_{cfg.mode}")
                raise RuntimeError(
                    f"source exhausted before {partition} reached {target:,} {cfg.mode}; "
                    f"got docs={docs:,}, tokens={tokens:,}"
                ) from exc
            ids = list(tok.encode(text).ids)
            ids.append(eot_id)
            if any(isinstance(token_id, bool) or not isinstance(token_id, int)
                   or not 0 <= token_id <= 65_535 for token_id in ids):
                raise ValueError("tokenizer emitted an ID outside uint16")
            encoded = np.asarray(ids, dtype="<u2")
            handle.write(encoded.tobytes())
            docs += 1
            tokens += len(ids)
            progress[partition] = {"docs": docs, "tokens": tokens}
            if docs % cfg.progress_docs == 0:
                handle.flush()
                os.fsync(handle.fileno())
                progress_callback()
        handle.flush()
        os.fsync(handle.fileno())
    progress_callback()
    return {
        "split": source_split,
        "source_doc_start": source_start,
        "source_doc_end": source_start + docs - 1,
        "docs": docs,
        "tokens": tokens,
        "target": getattr(cfg, f"{partition}_{cfg.mode}"),
        "target_unit": cfg.mode,
        "overshoot_tokens": tokens - cfg.__getattribute__(f"{partition}_tokens") if cfg.mode == "tokens" else None,
    }


def _artifact_record(path: Path, tokens: int | None = None, docs: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if tokens is not None:
        record.update({"tokens": tokens, "docs": docs})
    return record


def _manifest_identity(
    build_identity: dict[str, Any],
    build_signature: str,
    selection: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "build": build_identity,
        "build_signature": build_signature,
        "selection": selection,
        "outputs": outputs,
    }


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(file_stat, "st_file_attributes", 0) & flag)


def _regular_child(root: Path, basename: str) -> Path:
    """Return an exact, direct, non-link regular child without following an escaping reparse point."""
    if Path(basename).name != basename or basename in {"", ".", ".."}:
        raise RuntimeError(f"artifact name is not a basename: {basename!r}")
    try:
        root_stat = os.lstat(root)
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode) or _is_reparse_point(root_stat):
            raise RuntimeError(f"dataset path must be a real directory, not a link or reparse point: {root}")
        resolved_root = root.resolve(strict=True)
        artifact = root / basename
        artifact_stat = os.lstat(artifact)
    except OSError as exc:
        raise RuntimeError(f"required prepared artifact is missing or unreadable: {root / basename}") from exc
    if (not stat.S_ISREG(artifact_stat.st_mode) or stat.S_ISLNK(artifact_stat.st_mode)
            or _is_reparse_point(artifact_stat)):
        raise RuntimeError(f"prepared artifact must be a regular file, not a link or reparse point: {artifact}")
    try:
        resolved_artifact = artifact.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"prepared artifact cannot be resolved safely: {artifact}") from exc
    if resolved_artifact.parent != resolved_root:
        raise RuntimeError(f"prepared artifact escapes the dataset directory: {artifact}")
    return artifact


def _stored_build_identity(manifest: dict[str, Any]) -> tuple[dict[str, Any], PrepareConfig]:
    config = manifest.get("config")
    expected_config_keys = set(PrepareConfig.__dataclass_fields__) - {"reuse_existing"}
    if not isinstance(config, dict) or set(config) != expected_config_keys:
        raise RuntimeError("manifest build config has an invalid field set")
    try:
        cfg = PrepareConfig(**config, reuse_existing=False)
        _validate_config(cfg)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("manifest build config is invalid") from exc

    expected_source = {
        "repository": cfg.source,
        "revision": cfg.revision,
        "split": cfg.split,
        "config": cfg.source_config,
        "text_field": cfg.text_field,
    }
    source = manifest.get("source")
    if source != expected_source:
        raise RuntimeError("manifest source identity disagrees with its build config")

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or set(tokenizer) != {
        "path", "sha256", "vocab_size", "eot_token", "eot_id",
    }:
        raise RuntimeError("manifest tokenizer identity is invalid")
    if (not isinstance(tokenizer["path"], str) or not isinstance(tokenizer["sha256"], str)
            or not _SHA256_RE.fullmatch(tokenizer["sha256"])
            or isinstance(tokenizer["vocab_size"], bool) or not isinstance(tokenizer["vocab_size"], int)
            or not 0 < tokenizer["vocab_size"] <= 65_536 or tokenizer["eot_token"] != EOT
            or isinstance(tokenizer["eot_id"], bool) or not isinstance(tokenizer["eot_id"], int)
            or not 0 <= tokenizer["eot_id"] < tokenizer["vocab_size"]):
        raise RuntimeError("manifest tokenizer properties are invalid")

    tool = manifest.get("tool")
    if (not isinstance(tool, dict) or set(tool) != {"path", "sha256", "git"}
            or not isinstance(tool["path"], str) or not isinstance(tool["sha256"], str)
            or not _SHA256_RE.fullmatch(tool["sha256"])
            or not isinstance(tool["git"], dict)):
        raise RuntimeError("manifest tool provenance is invalid")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("manifest runtime provenance is invalid")
    if manifest.get("traversal") != TRAVERSAL:
        raise RuntimeError("manifest traversal policy is invalid")

    identity = {key: manifest.get(key) for key in _BUILD_IDENTITY_KEYS}
    return identity, cfg


def validate_prepared_dataset(
    path: Path,
    expected_build_identity: dict[str, Any] | None = None,
    expected_signature: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    manifest_path = _regular_child(path, "manifest.json")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_VERSION or manifest.get("status") != "completed":
        raise RuntimeError(f"dataset manifest is not completed: {manifest_path}")
    build_identity, cfg = _stored_build_identity(manifest)
    recomputed_build_signature = canonical_hash(build_identity)
    if manifest.get("build_signature") != recomputed_build_signature:
        raise RuntimeError("manifest build signature does not match its canonical build identity")
    if expected_signature is not None and recomputed_build_signature != expected_signature:
        raise RuntimeError(f"prepared dataset is incompatible with this request: {path}")
    if expected_build_identity is not None and build_identity != expected_build_identity:
        raise RuntimeError(f"prepared dataset build identity is incompatible with this request: {path}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"train", "val", "meta"}:
        raise RuntimeError(f"dataset output registry is invalid: {manifest_path}")
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or set(selection) != {"shared_source_stream", "train", "val"}:
        raise RuntimeError("dataset selection registry is invalid")
    recomputed_manifest_signature = canonical_hash(
        _manifest_identity(build_identity, recomputed_build_signature, selection, outputs)
    )
    if manifest.get("signature") != recomputed_manifest_signature:
        raise RuntimeError("manifest signature does not match its canonical contents")

    artifacts: dict[str, Path] = {}
    for name, record in outputs.items():
        expected_keys = {"file", "bytes", "sha256", "tokens", "docs"} if name in {"train", "val"} else {
            "file", "bytes", "sha256",
        }
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise RuntimeError(f"{name} artifact record is invalid")
        if record["file"] != _OUTPUT_FILES[name]:
            raise RuntimeError(f"{name} artifact must use exact filename {_OUTPUT_FILES[name]!r}")
        if (isinstance(record["bytes"], bool) or not isinstance(record["bytes"], int)
                or record["bytes"] <= 0 or not isinstance(record["sha256"], str)
                or not _SHA256_RE.fullmatch(record["sha256"])):
            raise RuntimeError(f"{name} artifact size or checksum record is invalid")
        artifact = _regular_child(path, record["file"])
        artifacts[name] = artifact
        artifact_size = artifact.stat().st_size
        if artifact_size != record["bytes"] or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"prepared artifact failed integrity check: {artifact}")
        if name in {"train", "val"}:
            tokens = record["tokens"]
            docs = record["docs"]
            if (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0
                    or isinstance(docs, bool) or not isinstance(docs, int) or docs <= 0
                    or artifact_size != tokens * 2):
                raise RuntimeError(f"prepared token/doc count or byte size mismatch: {artifact}")

    meta = read_json(artifacts["meta"])
    if not isinstance(meta, dict):
        raise RuntimeError("meta.json must be an object")
    if (meta.get("train_tokens") != outputs["train"]["tokens"]
            or meta.get("val_tokens") != outputs["val"]["tokens"]
            or meta.get("train_docs") != outputs["train"]["docs"]
            or meta.get("val_docs") != outputs["val"]["docs"]):
        raise RuntimeError("meta.json counts disagree with the manifest")
    if meta.get("build_signature") != recomputed_build_signature:
        raise RuntimeError("meta.json build signature disagrees with the manifest")
    tokenizer = manifest.get("tokenizer", {})
    source = manifest.get("source", {})
    if (meta.get("dataset") != cfg.dataset or meta.get("dtype") != "uint16-le"
            or meta.get("vocab_size") != tokenizer.get("vocab_size")
            or meta.get("eot_id") != tokenizer.get("eot_id") or meta.get("source") != source.get("repository")
            or meta.get("revision") != source.get("revision") or meta.get("split") != source.get("split")):
        raise RuntimeError("meta.json dtype/tokenizer/source identity disagrees with the manifest")
    config = manifest["config"]
    mode = "tokens" if config.get("train_tokens") is not None else "docs"
    for partition in ("train", "val"):
        selected = selection[partition]
        if not isinstance(selected, dict) or set(selected) != {
            "split", "source_doc_start", "source_doc_end", "docs", "tokens", "target",
            "target_unit", "overshoot_tokens",
        }:
            raise RuntimeError(f"{partition} selection record is invalid")
        if (selected.get("tokens") != outputs[partition]["tokens"]
                or selected.get("docs") != outputs[partition]["docs"]):
            raise RuntimeError(f"{partition} selection counts disagree with outputs")
        target = config.get(f"{partition}_{mode}")
        current = selected.get("tokens" if mode == "tokens" else "docs")
        if (isinstance(target, bool) or not isinstance(target, int)
                or selected.get("target") != target or selected.get("target_unit") != mode
                or isinstance(current, bool) or not isinstance(current, int) or current < target):
            raise RuntimeError(f"{partition} did not meet its declared {mode} target")
        start = selected.get("source_doc_start")
        end = selected.get("source_doc_end")
        if (isinstance(start, bool) or not isinstance(start, int) or start < 0
                or isinstance(end, bool) or not isinstance(end, int)
                or end - start + 1 != selected.get("docs")):
            raise RuntimeError(f"{partition} source range does not match its document count")
        expected_overshoot = selected["tokens"] - target if mode == "tokens" else None
        if selected.get("overshoot_tokens") != expected_overshoot:
            raise RuntimeError(f"{partition} overshoot record is invalid")
        expected_split = cfg.split if partition == "train" or cfg.dataset == "openwebtext" else "validation"
        if selected.get("split") != expected_split:
            raise RuntimeError(f"{partition} selection uses an unexpected source split")
    if not isinstance(selection["shared_source_stream"], bool):
        raise RuntimeError("shared-source-stream flag must be boolean")
    if selection["shared_source_stream"] != (cfg.dataset == "openwebtext"):
        raise RuntimeError("shared-source-stream policy disagrees with the dataset protocol")
    if selection["shared_source_stream"]:
        val = selection["val"]
        train = selection["train"]
        if val["source_doc_start"] != 0 or train["source_doc_start"] != val["source_doc_end"] + 1:
            raise RuntimeError("validation and training source ranges must be contiguous and non-overlapping")
    elif selection["val"]["source_doc_start"] != 0 or selection["train"]["source_doc_start"] != 0:
        raise RuntimeError("independent source splits must each start at document zero")
    return manifest


def _disk_preflight(cfg: PrepareConfig, parent: Path) -> dict[str, int | None]:
    """Fail before streaming when current free space cannot hold even the token target bytes."""
    usage = shutil.disk_usage(parent)
    minimum = None
    if cfg.mode == "tokens":
        minimum = (cfg.train_tokens + cfg.val_tokens) * np.dtype("<u2").itemsize
        if usage.free < minimum:
            raise OSError(
                f"insufficient free space under {parent}: {usage.free:,} bytes available, "
                f"at least {minimum:,} bytes required before whole-document overshoot and metadata"
            )
    return {"free_bytes_at_start": usage.free, "minimum_binary_bytes": minimum}


def prepare_dataset(
    cfg: PrepareConfig,
    iterator_factory: Callable[[str], Iterator[str]] | None = None,
) -> Path:
    """Build, verify, and atomically promote one immutable prepared dataset."""
    _validate_config(cfg)
    out_dir = Path(cfg.out_dir).resolve()
    tokenizer_path = Path(cfg.tokenizer).resolve()
    tok, tokenizer_identity = _load_tokenizer(tokenizer_path)
    tool = _tool_provenance()
    runtime = _runtime_provenance()
    identity, build_signature = _build_identity(cfg, tokenizer_identity, tool, runtime)
    stage_dir = out_dir.parent / f".{out_dir.name}.staging-{cfg.run_id}"
    lock_path = out_dir.parent / f".{out_dir.name}.prepare.lock"
    factory = iterator_factory or hf_iterator_factory(cfg)

    with BuildLock(lock_path):
        if out_dir.exists():
            if not cfg.reuse_existing:
                raise FileExistsError(f"refusing to overwrite existing prepared dataset: {out_dir}")
            validate_prepared_dataset(
                out_dir,
                expected_build_identity=identity,
                expected_signature=build_signature,
            )
            print(f"reuse verified dataset: {out_dir}")
            return out_dir
        if stage_dir.exists():
            state_path = stage_dir / "stage_manifest.json"
            status = read_json(state_path).get("status") if state_path.exists() else "unknown"
            raise RuntimeError(
                f"preserved {status} stage exists: {stage_dir}. Streaming cursor resume is not "
                "supported; inspect/remove it explicitly or choose a new run-id."
            )

        disk_preflight = _disk_preflight(cfg, out_dir.parent)
        stage_dir.mkdir(parents=False)
        progress = {"val": {"docs": 0, "tokens": 0}, "train": {"docs": 0, "tokens": 0}}
        stage_state = {
            **identity,
            "build_signature": build_signature,
            "status": "building",
            "created_at": utc_now(),
            "stage_dir": str(stage_dir),
            "intended_output": str(out_dir),
            "disk_preflight": disk_preflight,
            "progress": progress,
        }

        def write_progress() -> None:
            stage_state["updated_at"] = utc_now()
            stage_state["progress"] = progress
            atomic_write_json(stage_dir / "stage_manifest.json", stage_state)

        write_progress()
        try:
            eot_id = tokenizer_identity["eot_id"]
            if cfg.dataset == "openwebtext":
                shared_iterator = iter(factory(cfg.split))
                val_stats = _write_partition(
                    cfg, "val", cfg.split, 0, shared_iterator, tok, eot_id,
                    stage_dir / "val.bin", progress, write_progress,
                )
                train_stats = _write_partition(
                    cfg, "train", cfg.split, val_stats["docs"], shared_iterator, tok, eot_id,
                    stage_dir / "train.bin", progress, write_progress,
                )
                shared_source_stream = True
            else:
                val_stats = _write_partition(
                    cfg, "val", "validation", 0, iter(factory("validation")), tok, eot_id,
                    stage_dir / "val.bin", progress, write_progress,
                )
                train_stats = _write_partition(
                    cfg, "train", cfg.split, 0, iter(factory(cfg.split)), tok, eot_id,
                    stage_dir / "train.bin", progress, write_progress,
                )
                shared_source_stream = False

            meta = {
                "dataset": cfg.dataset,
                "source": cfg.source,
                "revision": cfg.revision,
                "split": cfg.split,
                "vocab_size": tokenizer_identity["vocab_size"],
                "dtype": "uint16-le",
                "eot_id": eot_id,
                "train_tokens": train_stats["tokens"],
                "val_tokens": val_stats["tokens"],
                "train_docs": train_stats["docs"],
                "val_docs": val_stats["docs"],
                "build_signature": build_signature,
            }
            atomic_write_json(stage_dir / "meta.json", meta)
            outputs = {
                "train": _artifact_record(stage_dir / "train.bin", train_stats["tokens"], train_stats["docs"]),
                "val": _artifact_record(stage_dir / "val.bin", val_stats["tokens"], val_stats["docs"]),
                "meta": _artifact_record(stage_dir / "meta.json"),
            }
            manifest_core = {
                **identity,
                "build_signature": build_signature,
                "status": "completed",
                "created_at": stage_state["created_at"],
                "completed_at": utc_now(),
                "selection": {
                    "shared_source_stream": shared_source_stream,
                    "val": val_stats,
                    "train": train_stats,
                },
                "outputs": outputs,
            }
            manifest = {
                **manifest_core,
                "signature": canonical_hash(
                    _manifest_identity(identity, build_signature, manifest_core["selection"], outputs)
                ),
            }
            atomic_write_json(stage_dir / "manifest.json", manifest)
            validate_prepared_dataset(
                stage_dir,
                expected_build_identity=identity,
                expected_signature=build_signature,
            )
            stage_state.update({"status": "verified", "updated_at": utc_now(), "progress": progress})
            atomic_write_json(stage_dir / "stage_manifest.json", stage_state)
            if out_dir.exists():
                raise FileExistsError(f"output appeared during preparation; stage preserved: {out_dir}")
            stage_dir.rename(out_dir)
            validate_prepared_dataset(
                out_dir,
                expected_build_identity=identity,
                expected_signature=build_signature,
            )
            print(
                f"done. train={train_stats['tokens']:,} tokens/{train_stats['docs']:,} docs  "
                f"val={val_stats['tokens']:,} tokens/{val_stats['docs']:,} docs  -> {out_dir}"
            )
            return out_dir
        except BaseException as exc:
            if stage_dir.exists():
                stage_state.update({
                    "status": "interrupted",
                    "updated_at": utc_now(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "progress": progress,
                })
                try:
                    atomic_write_json(stage_dir / "stage_manifest.json", stage_state)
                except Exception as state_exc:
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            "The partial stage was left in place, but its interrupted manifest could not "
                            f"be updated: {state_exc}"
                        )
            raise


def _config_from_args(args: argparse.Namespace) -> PrepareConfig:
    token_mode = args.train_tokens is not None or args.val_tokens is not None
    if token_mode and (args.train_docs is not None or args.val_docs is not None):
        raise ValueError("token targets and document targets are mutually exclusive")
    if token_mode:
        train_docs = val_docs = None
    else:
        train_docs = args.train_docs if args.train_docs is not None else DEFAULT_TRAIN_DOCS
        val_docs = args.val_docs if args.val_docs is not None else DEFAULT_VAL_DOCS
    source = args.source or (OWT_SOURCE if args.dataset == "openwebtext" else "roneneldan/TinyStories")
    revision = args.revision or (OWT_REVISION if args.dataset == "openwebtext" else "main")
    out_dir = args.out_dir or str(Path("data") / args.dataset)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("prepare-%Y%m%d-%H%M%S")
    return PrepareConfig(
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        out_dir=out_dir,
        run_id=run_id,
        source=source,
        revision=revision,
        split=args.split,
        source_config=args.source_config,
        text_field=args.text_field,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        train_docs=train_docs,
        val_docs=val_docs,
        progress_docs=args.progress_docs,
        reuse_existing=args.reuse_existing,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--validate-only", metavar="DATA_DIR",
        help="verify an existing prepared dataset and exit without loading a tokenizer or source",
    )
    ap.add_argument("--dataset", default="tinystories", choices=["tinystories", "openwebtext"])
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--out-dir", help="immutable final dataset directory (default: data/<dataset>)")
    ap.add_argument("--run-id", help="stage/provenance namespace; timestamped when omitted")
    ap.add_argument("--source", help="Hugging Face dataset repository override")
    ap.add_argument("--revision", help=f"immutable source revision (OWT default: {OWT_REVISION})")
    ap.add_argument("--split", default="train")
    ap.add_argument("--source-config")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--train-tokens", type=int)
    ap.add_argument("--val-tokens", type=int)
    ap.add_argument("--train-docs", type=int, help=f"legacy document target (default: {DEFAULT_TRAIN_DOCS:,})")
    ap.add_argument("--val-docs", type=int, help=f"legacy document target (default: {DEFAULT_VAL_DOCS:,})")
    ap.add_argument("--progress-docs", type=int, default=1_000)
    ap.add_argument("--reuse-existing", action="store_true", help="validate and reuse only an exactly compatible output")
    args = ap.parse_args()

    if args.validate_only:
        manifest = validate_prepared_dataset(Path(args.validate_only))
        print(
            f"verified prepared dataset: {Path(args.validate_only).resolve()} "
            f"(signature {manifest['signature']})"
        )
        return
    cfg = _config_from_args(args)
    prepare_dataset(cfg)


if __name__ == "__main__":
    main()
