"""Declared FP32 reference-only CUDA stop/restart smoke, never a research comparison.

An uninterrupted four-update run is compared with the same trajectory stopped
after its complete step-two checkpoint publication, then resumed. No historical
checkpoint is trained or overwritten. Exact recovery is an operational endpoint,
not a forward/backend parity or language-quality certificate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import src.train.train as trainer  # noqa: E402
from src.data.prepare_data import validate_prepared_dataset  # noqa: E402

KIND = "cuda_checkpoint_recovery_operational_smoke"
STOP_REASON = "diagnostic:after-complete-checkpoint-step-2"
WRAPPER_ID = "post-successful _publish_checkpoint_transaction request_stop; restored in finally"
WALL_SECONDS = 300
MIN_FREE_CUDA_BYTES = 8 * 1024**3
TRAIN_SETTINGS = {
    "device": "cuda", "precision": "float32", "scan_backend": "reference", "scan_chunk_size": 128,
    "seed": 1337, "model_seed": 1337, "data_seed": 1337, "eval_seed": 1337,
    "max_steps": 4, "block_size": 512, "batch_size": 1, "grad_accum": 1, "warmup_steps": 1,
    "eval_interval": 2, "eval_iters": 1, "checkpoint_interval": 1, "log_interval": 1,
    "lr": 1e-3, "min_lr": 1e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
    "grad_clip": 1.0, "grad_checkpointing": True, "wandb": False, "resume": True,
    "wall_time_limit_seconds": float(WALL_SECONDS),
}
CHECKPOINT_ROOT = "checkpoints/operational-smoke-2026-10-04"
CONTROL_ID = "cuda-recovery-control-v1"
RESUME_ID = "cuda-recovery-resume-v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    return trainer._file_sha256(Path(path))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_declaration(path):
    def pairs(items):
        require(len(items) == len({key for key, _ in items}), "duplicate declaration keys")
        return dict(items)

    def invalid(value):
        raise ValueError(f"nonfinite declaration constant: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "declaration must be an object")
    return value


def public_path(value, root=ROOT):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value
            and not PurePosixPath(value).is_absolute() and PurePosixPath(value).as_posix() == value
            and all(part not in {".", ".."} for part in PurePosixPath(value).parts), "unsafe public relative path")
    root = Path(root).resolve()
    path = root / value
    require(path.resolve().is_relative_to(root), "public path escapes repository")
    current = path
    while current != root:
        require(not current.is_symlink(), "public path cannot traverse symlinks")
        current = current.parent
    return path


def source_registry(root=ROOT):
    root = Path(root).resolve()
    paths = sorted((root / "src").rglob("*.py"))
    paths += [root / "scripts/run_sweep.py", root / "requirements.txt",
              root / "scripts/check_cuda_checkpoint_recovery.py"]
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def sanitize(message, root=ROOT):
    for path, replacement in ((Path(root).resolve(), "[repository]"), (Path.home(), "[home]")):
        for spelling in (str(path), path.as_posix()):
            message = message.replace(spelling, replacement)
    return message


def public_metadata(value, root=ROOT):
    """Display metadata may redact local paths; the original declaration hashes remain authoritative."""
    if isinstance(value, str):
        return sanitize(value, root)
    if isinstance(value, dict):
        return {sanitize(key, root): public_metadata(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item, root) for item in value]
    return value


def historical_preservation(histories, root=ROOT):
    """A missing/unreadable evidence file must retain a failed report, never obscure the original error."""
    checks = []
    for row in histories:
        check = {"path": row["path"], "expected_sha256": row["sha256"], "actual_sha256": None,
                 "bytes_unchanged": False}
        try:
            check["actual_sha256"] = file_sha256(public_path(row["path"], root))
            check["bytes_unchanged"] = check["actual_sha256"] == row["sha256"]
        except (RuntimeError, ValueError, OSError) as error:
            check["reason"] = sanitize(f"{type(error).__name__}: {error}", root)
        checks.append(check)
    return {"all_bytes_unchanged": all(row["bytes_unchanged"] for row in checks), "checks": checks}


def process_io():
    """OS process transfer counters include cached I/O; they are not physical-disk traffic."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("read_operations", "write_operations", "other_operations", "read_bytes", "write_bytes", "other_bytes")]
        counters = Counters()
        function = ctypes.windll.kernel32.GetProcessIoCounters
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters)]
        function.restype = wintypes.BOOL
        if function(wintypes.HANDLE(-1), ctypes.byref(counters)):
            return {"available": True, "source": "Windows GetProcessIoCounters",
                    "counters": {name: getattr(counters, name) for name, _ in Counters._fields_}}
    else:
        try:
            counters = {name: int(value) for name, value in
                        (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}
            return {"available": True, "source": "Linux /proc/self/io", "counters": counters}
        except (OSError, ValueError):
            pass
    return {"available": False, "source": None, "counters": None}


def io_delta(before, after):
    if not before["available"] or not after["available"] or before["source"] != after["source"]:
        return {"available": False, "source": None, "counters": None}
    return {"available": True, "source": after["source"],
            "counters": {name: value - before["counters"][name] for name, value in after["counters"].items()},
            "scope": "OS process transfer counters, including cached I/O and checks; not physical-disk bytes"}


def runtime_metadata():
    packages = {}
    for name in ("torch", "numpy", "tokenizers", "mamba-ssm", "triton", "causal-conv1d"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "training_policy": trainer.precision_policy("float32"),
            "ambient_flags_outside_training": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}


def cuda_guard(started):
    require(time.monotonic() - started < WALL_SECONDS, "workflow wall allowance exhausted")
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback is forbidden for this declaration")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    require(free >= MIN_FREE_CUDA_BYTES, "CUDA headroom below declared 8 GiB before next phase")
    return {"free_bytes": free, "total_bytes": total}


@contextmanager
def request_stop_after_publication(control, run_id):
    """Observe only a successful publisher return; never alter training calculations."""
    original = trainer._publish_checkpoint_transaction
    observation = {"wrapper_id": WRAPPER_ID, "stop_reason": STOP_REASON,
                   "original_binding": original.__module__ + "." + original.__qualname__,
                   "successful_steps": [], "stop_requests": 0, "restored": False}

    def observe(run_dir, cfg, mcfg, signature, state, best, evaluation, previous_step):
        original(run_dir, cfg, mcfg, signature, state, best, evaluation, previous_step)
        if cfg.run_id == run_id:
            step = state["completed_steps"]
            observation["successful_steps"].append(step)
            if step == 2:
                control.request_stop(STOP_REASON)
                observation["stop_requests"] += 1

    try:
        trainer._publish_checkpoint_transaction = observe
        yield observation
    finally:
        trainer._publish_checkpoint_transaction = original
        observation["restored"] = trainer._publish_checkpoint_transaction is original


def exact_comparison(left, right):
    differences = []
    tensor_checks = 0

    def walk(a, b, name):
        nonlocal tensor_checks
        if isinstance(a, torch.Tensor):
            tensor_checks += 1
            equal = isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
            if not equal:
                differences.append(name)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                differences.append(name + ".keys")
                return
            for key in a:
                walk(a[key], b[key], name + "." + str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                differences.append(name + ".sequence")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{name}.{index}")
        elif type(a) is not type(b) or a != b:
            differences.append(name)

    walk(left, right, "value")
    return {"exact_equal": not differences, "tensor_checks": tensor_checks,
            "difference_count": len(differences), "first_differences": differences[:20]}


def semantic_metrics(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [{key: value for key, value in row.items() if key not in {"step_seconds", "tok_per_s"}} for row in rows]


def compare_trajectories(control_dir, resumed_dir):
    control = torch.load(control_dir / "last.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=True)
    checks = {name: exact_comparison(control[name], resumed[name]) for name in
              ("model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss")}
    checks["train_batch_generator"] = exact_comparison(control["rng"]["train_generator"], resumed["rng"]["train_generator"])
    checks["semantic_metrics"] = exact_comparison(semantic_metrics(control_dir / "metrics.jsonl"),
                                                  semantic_metrics(resumed_dir / "metrics.jsonl"))
    return {"passed": all(row["exact_equal"] for row in checks.values()), "checks": checks,
            "timing_excluded": ["train_seconds", "eval_seconds", "peak_vram_mb", "step_seconds", "tok_per_s"],
            "scope": "same reference backend/numerical policy stop-restart identity; no backend or quality comparison"}


def execute_workflow(cfg, *, control_run_id=CONTROL_ID, interrupted_run_id=RESUME_ID, started=None,
                     guard=lambda: None, evidence_check=lambda: None, progress=None):
    """The orchestrator is also exercised on a real tiny CPU fixture; public mode is CUDA only."""
    started = time.monotonic() if started is None else started
    phases = [] if progress is None else progress.setdefault("phases", [])
    interrupted_cfg = trainer.TrainConfig(**{**vars(cfg), "run_id": interrupted_run_id})
    control_cfg = trainer.TrainConfig(**{**vars(cfg), "run_id": control_run_id})
    model_name = trainer.load_model_config(cfg.model_config).name
    dirs = {name: trainer.variant_run_dir(value, model_name) for name, value in
            (("control", control_cfg), ("resumed", interrupted_cfg))}
    require(control_run_id != interrupted_run_id, "control/resumed namespaces must differ")
    require(all(not (Path(cfg.ckpt_dir) / trainer.validate_run_id(name)).exists()
                for name in (control_run_id, interrupted_run_id)), "operational namespace already exists; no overwrite")

    def call(name, configuration, request_stop=False):
        evidence_check()
        headroom = guard()
        control = trainer.TrainingStopControl(configuration.wall_time_limit_seconds)
        control.started = started  # All calls retain the actual common workflow deadline.
        before_io, before_time = process_io(), time.monotonic()
        observation = None
        result = None
        try:
            if request_stop:
                with request_stop_after_publication(control, interrupted_run_id) as observation:
                    result = trainer.run(configuration, stop_control=control)
            else:
                result = trainer.run(configuration, stop_control=control)
        finally:
            after_time = time.monotonic()
            phase = {"phase": name, "wall_seconds": after_time - before_time, "io": io_delta(before_io, process_io()),
                     "headroom_before": headroom, "status": "incomplete" if result is None else result.get("status", "completed"),
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated() if configuration.device == "cuda" else None}
            if result is not None:
                phase.update({key: result.get(key) for key in ("completed_steps", "tokens_seen", "stop", "observed_paths")})
            if observation is not None:
                phase["wrapper"] = observation
            phases.append(phase)
        require(after_time - started < configuration.wall_time_limit_seconds, "workflow wall allowance exceeded during in-flight work")
        evidence_check()
        if request_stop:
            require(result.get("status") == "stopped" and result["completed_steps"] == 2
                    and result["stop"]["reason"] == STOP_REASON and observation["stop_requests"] == 1
                    and observation["restored"], "expected post-publication step-two stop did not occur")
            require(not (dirs["resumed"] / "result.json").exists(), "stopped arm cannot publish a completed result")
            phase["last_sha256_at_stop"] = file_sha256(dirs["resumed"] / "last.pt")
            phase["boundary_artifact_scope"] = "last.pt fingerprint at stop; superseded by successful resume"
        else:
            require(result.get("status", "completed") == "completed" and result["completed_steps"] == cfg.max_steps,
                    "unexpected incomplete/budget-stopped operational arm")

    call("uninterrupted", control_cfg)
    call("requested_stop", interrupted_cfg, request_stop=True)
    call("resumed", interrupted_cfg)
    comparison = compare_trajectories(dirs["control"], dirs["resumed"])
    return {"phases": phases, "recovery_identity": comparison, "directories": dirs}


def validate_declaration(declaration, output, root=ROOT):
    require(type(declaration.get("schema")) is int and declaration["schema"] == 1
            and type(declaration.get("version")) is int and declaration["version"] == 1
            and declaration.get("status_at_declaration") == "planned_before_execution"
            and declaration.get("kind") == KIND, "unexpected operational declaration identity")
    require(declaration.get("train_config") == TRAIN_SETTINGS, "changed operational train settings")
    for key, expected in TRAIN_SETTINGS.items():
        actual = declaration["train_config"][key]
        valid_type = (type(actual) is bool if type(expected) is bool else type(actual) is int if type(expected) is int
                      else type(actual) in (int, float) and math.isfinite(actual) if type(expected) is float
                      else type(actual) is type(expected))
        require(valid_type, "malformed operational train settings")
    require(declaration.get("guards") == {"wall_seconds": WALL_SECONDS, "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES}, "changed safety guards")
    require(declaration.get("stop") == {"step": 2, "reason": STOP_REASON, "wrapper": WRAPPER_ID}, "changed stop injection")
    require(declaration.get("checkpoint_root") == CHECKPOINT_ROOT and declaration.get("control_run_id") == CONTROL_ID
            and declaration.get("interrupted_run_id") == RESUME_ID, "unexpected operational namespaces")
    require(Path(output).resolve() == public_path(declaration["output"], root).resolve(), "output differs from declaration")
    require(not Path(output).exists(), "output already exists; no overwrite")
    checkpoint_root = public_path(declaration["checkpoint_root"], root)
    require(all(not (checkpoint_root / name).exists() for name in (CONTROL_ID, RESUME_ID)), "run namespace already exists; no overwrite")
    require(declaration["source_sha256"] == source_registry(root), "source fingerprints differ from declaration")
    model = public_path(declaration["model_config"]["path"], root)
    require(file_sha256(model) == declaration["model_config"]["sha256"], "model config identity differs")
    cfg = trainer.load_model_config(str(model))
    require((cfg.ratio, cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.d_state, cfg.head_dim,
             cfg.mamba_headdim, cfg.n_attention_layers, cfg.n_mamba_layers) == ("1:15", 448, 16, 16000, 128, 64, 64, 1, 15),
            "operational model must use full registered ratio1:15 geometry")
    require((cfg.expand, cfg.d_conv, cfg.n_groups, cfg.mlp_ratio, cfg.mlp_multiple_of,
             cfg.tie_embeddings, cfg.attn_bias, cfg.mlp_bias, cfg.mlp_on_every_layer)
            == (2, 4, 1, 2.6667, 64, True, False, False, True), "changed full model geometry")
    histories = declaration["historical_checkpoints"]
    require([row["ratio"] for row in histories] == ["1:3", "1:7", "1:15"], "historical preservation registry differs")
    for row in histories:
        require(row["path"].startswith("checkpoints/week3-700m-v1/") and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "invalid historical evidence")
        public_path(row["path"], root)
    return model, checkpoint_root


def run_declared(declaration_path, output, *, root=ROOT):
    root = Path(root).resolve()
    started, io_before = time.monotonic(), process_io()
    declaration = read_declaration(declaration_path)
    model, checkpoint_root = validate_declaration(declaration, output, root)
    declaration_sha = file_sha256(declaration_path)
    data = declaration["data"]
    data_dir = public_path(data["directory"], root)
    manifest = validate_prepared_dataset(data_dir)
    require(manifest["signature"] == data["signature"] and file_sha256(data_dir / "manifest.json") == data["manifest_sha256"],
            "prepared corpus identity differs")
    require(manifest["tokenizer"]["vocab_size"] == 16000
            and all(manifest["outputs"][name]["tokens"] > TRAIN_SETTINGS["block_size"] + 1 for name in ("train", "val")), "prepared corpus does not fit workload")
    evidence = {declaration["model_config"]["path"]: declaration["model_config"]["sha256"],
                f"{data['directory']}/manifest.json": data["manifest_sha256"]}
    evidence.update({f"{data['directory']}/{row['file']}": row["sha256"] for row in manifest["outputs"].values()})
    evidence.update({row["path"]: row["sha256"] for row in declaration["historical_checkpoints"]})
    display_protocol = public_metadata(declaration, root)

    def verify_evidence():
        require(file_sha256(declaration_path) == declaration_sha, "declaration changed during workflow")
        require(source_registry(root) == declaration["source_sha256"], "source files changed during workflow")
        for name, digest in evidence.items():
            require(file_sha256(public_path(name, root)) == digest, "input/historical evidence bytes changed")

    report = {"schema": 1, "kind": KIND, "date": declaration["date"], "certified": False,
              "declaration_sha256": declaration_sha, "declaration_canonical_sha256": canonical_hash(declaration),
              "protocol": display_protocol, "protocol_display_sanitized": display_protocol != declaration,
              "declaration_hash_scope": "original declaration bytes/canonical contents before public display redaction",
              "execution_status": "incomplete", "status": "incomplete",
              "research_pilot_executed": False, "backend_parity_certified": False,
              "workflow_resource_scope": "preflight, training, restart, comparison and evidence checks; excludes final report publication",
              "limits": ["Operational stop/restart check on one new initialization, one portable backend and one shape.",
                         "No architecture/quality comparison, fused execution, forward parity approval or research-pilot result.",
                         "Same LR/token schedule across stop/resume; timing and I/O include artifact validation and publication.",
                         "Process-interruption recovery is separate from power-loss guarantees.",
                         "Cooperative deadline permits in-flight overshoot, which fails this smoke and is retained."]}
    try:
        verify_evidence()
        headroom = cuda_guard(started)
        report["runtime"] = runtime_metadata()
        report["initial_cuda_headroom"] = headroom
        configuration = trainer.TrainConfig(**TRAIN_SETTINGS, model_config=str(model), data_dir=str(data_dir),
                                            ckpt_dir=str(checkpoint_root), run_id=CONTROL_ID)
        result = execute_workflow(configuration, started=started, guard=lambda: cuda_guard(started),
                                  evidence_check=verify_evidence, progress=report)
        report.update({key: value for key, value in result.items() if key != "directories"})
        report["artifacts"] = {name: {label: {"path": (path / filename).relative_to(root).as_posix(),
                                               "sha256": file_sha256(path / filename)}
                                      for label, filename in {**trainer.ARTIFACT_FILES, "manifest": "manifest.json"}.items()}
                               for name, path in result["directories"].items()}
        report["training_provenance"] = {}
        for name, path in result["directories"].items():
            manifest = trainer.read_json(path / "manifest.json")
            report["training_provenance"][name] = {
                "signature": manifest["signature"], "source_fingerprint": manifest["code"]["fingerprint"],
                "git": manifest["git"],
            }
        verify_evidence()
        require(time.monotonic() - started < WALL_SECONDS, "workflow wall allowance exceeded during checks")
        report.update({"execution_status": "completed", "status": "completed" if result["recovery_identity"]["passed"]
                       else "completed_with_recovery_failures", "historical_bytes_unchanged": True})
    except (RuntimeError, ValueError, OSError) as error:
        report["reason"] = sanitize(f"{type(error).__name__}: {error}", root)
    report["historical_preservation"] = historical_preservation(declaration["historical_checkpoints"], root)
    report["historical_bytes_unchanged"] = report["historical_preservation"]["all_bytes_unchanged"]
    if not report["historical_bytes_unchanged"]:
        report["execution_status"] = report["status"] = "incomplete"
        report.setdefault("reason", "historical evidence changed or could not be verified")
    report["workflow_wall_seconds"] = time.monotonic() - started
    report["wall_overshoot_seconds"] = max(0.0, report["workflow_wall_seconds"] - WALL_SECONDS)
    if report["workflow_wall_seconds"] >= WALL_SECONDS:
        report["execution_status"] = report["status"] = "incomplete"
        report.setdefault("reason", "workflow wall allowance exceeded during final evidence checks")
    report["workflow_io"] = io_delta(io_before, process_io())
    return report


def write_report(path, report):
    path = Path(path)
    require(not path.exists(), "output already exists; no overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_declared(args.declaration, args.output)
        write_report(args.output, report)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        parser.error(sanitize(f"invalid operational smoke: {error}"))
    print(json.dumps({"status": report["status"], "execution_status": report["execution_status"], "certified": False,
                      "workflow_wall_seconds": report["workflow_wall_seconds"]}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
