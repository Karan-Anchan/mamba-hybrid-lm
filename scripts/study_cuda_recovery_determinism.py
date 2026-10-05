"""Declared reference/FP32 repeatability and restart study, never a quality/backend certificate.

The parent imports only the standard library. Each policy receives a fresh Python
process with its declared cuBLAS environment set before PyTorch imports. Within a
policy, two four-update controls and a stopped/resumed trajectory use identical
seeds. Exact equality remains the endpoint; difference magnitudes are descriptive.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
KIND = "cuda_recovery_determinism_study"
WALL_SECONDS = 900
MIN_FREE_CUDA_BYTES = 8 * 1024**3
ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
CHECKPOINT_ROOT = "checkpoints/operational-determinism-2026-10-04"
POLICIES = [
    {"name": "legacy", "deterministic_algorithms": False, "warn_only": False,
     "cudnn_deterministic": False, "cudnn_benchmark": False},
    {"name": "strict", "deterministic_algorithms": True, "warn_only": False,
     "cudnn_deterministic": True, "cudnn_benchmark": False},
]
RUN_IDS = {name: {arm: f"cuda-recovery-{name}-{suffix}-v1" for arm, suffix in
                 (("control_a", "control-a"), ("control_b", "control-b"), ("resumed", "resume"))}
           for name in ("legacy", "strict")}
TRAIN_SETTINGS = {
    "device": "cuda", "precision": "float32", "scan_backend": "reference", "scan_chunk_size": 128,
    "seed": 1337, "model_seed": 1337, "data_seed": 1337, "eval_seed": 1337,
    "max_steps": 4, "block_size": 512, "batch_size": 1, "grad_accum": 1, "warmup_steps": 1,
    "eval_interval": 2, "eval_iters": 1, "checkpoint_interval": 1, "log_interval": 1,
    "lr": 1e-3, "min_lr": 1e-4, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95,
    "grad_clip": 1.0, "grad_checkpointing": True, "wandb": False, "resume": True,
    "wall_time_limit_seconds": float(WALL_SECONDS),
}
STOP_REASON = "diagnostic:after-complete-checkpoint-step-2"
WRAPPER_ID = "post-successful _publish_checkpoint_transaction request_stop; restored in finally"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        require(len(items) == len({key for key, _ in items}), "duplicate declaration/report keys")
        return dict(items)

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(result, dict), "declaration/report must be an object")
    return result


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
    paths += [root / name for name in ("scripts/run_sweep.py", "requirements.txt",
                                      "scripts/check_cuda_checkpoint_recovery.py",
                                      "scripts/study_cuda_recovery_determinism.py")]
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def sanitize(message, root=ROOT):
    for path, replacement in ((Path(root).resolve(), "[repository]"), (Path.home(), "[home]")):
        for spelling in (str(path), path.as_posix()):
            message = message.replace(spelling, replacement)
    return message


def helpers():
    # Worker-only in production: the parent never imports PyTorch or initializes CUDA.
    sys.path.insert(0, str(ROOT))
    import scripts.check_cuda_checkpoint_recovery as smoke
    return smoke


def worker_output_paths(declaration, root=ROOT):
    output = public_path(declaration["output"], root)
    return {name: output.with_name(f"{name}-worker.json") for name in ("legacy", "strict")}


def declared_evidence(declaration, root=ROOT):
    """Canonical prepared outputs are bound by their declared manifest, including full token files."""
    data = declaration["data"]
    manifest = read_json(public_path(f"{data['directory']}/manifest.json", root))
    evidence = {declaration[label]["path"]: declaration[label]["sha256"] for label in ("model_config", "prior_smoke")}
    evidence[f"{data['directory']}/manifest.json"] = data["manifest_sha256"]
    evidence.update({f"{data['directory']}/{row['file']}": row["sha256"] for row in manifest["outputs"].values()})
    evidence.update({row["path"]: row["sha256"] for row in declaration["historical_checkpoints"]})
    return evidence


def validate_worker_result(worker, name, declaration, declaration_sha, exit_code, *, expected_started=None):
    require(type(worker.get("schema")) is int and worker["schema"] == 1
            and worker.get("kind") == KIND and worker.get("policy") == name
            and worker.get("declaration_sha256") == declaration_sha and worker.get("source_sha256") == declaration["source_sha256"]
            and worker.get("certified") is False and worker.get("process_environment") == ENVIRONMENT, "worker report identity differs")
    if expected_started is not None:
        require(worker.get("common_workflow_started_monotonic") == expected_started, "worker shared monotonic origin differs")
    require(worker.get("execution_status") in {"completed", "incomplete"}, "invalid worker execution status")
    require(worker.get("research_pilot_executed") is False and worker.get("backend_parity_certified") is False, "worker scope differs")
    if worker["execution_status"] == "incomplete":
        require(worker.get("status") == "incomplete" and exit_code == 2, "incomplete worker status/exit differs")
        return
    comparisons = worker.get("comparisons")
    require(isinstance(comparisons, dict) and set(comparisons) == {"control_a_vs_control_b", "control_a_vs_resumed"}, "worker comparisons missing")
    expected_checks = {"model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss", "train_batch_generator", "semantic_metrics"}
    passes = []
    for row in comparisons.values():
        require(set(row.get("checks", {})) == expected_checks
                and all(type(check.get("exact_equal")) is bool for check in row["checks"].values()), "worker exact checks missing/malformed")
        passed = all(check["exact_equal"] for check in row["checks"].values())
        require(row.get("passed") is passed and row.get("tolerances_relaxed") is False, "worker comparison conclusion differs from checks")
        passes.append(passed)
    expected_status = "completed" if all(passes) else "completed_with_identity_failures"
    require(worker.get("status") == expected_status and exit_code == (0 if all(passes) else 2), "worker completed status/exit differs from exact checks")
    policy = next(row for row in POLICIES if row["name"] == name)
    flags = worker.get("policy_flags", {})
    require(flags.get("requested") == policy and flags.get("restored") is True
            and all(flags.get("active", {}).get(key) is policy[key] for key in
                    ("deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark")), "effective/restored worker policy differs")
    require(worker.get("historical_bytes_unchanged") is True, "worker historical preservation failed")
    require(worker.get("torch_imported_at_worker_entry") is False, "worker did not start before PyTorch import")
    provenance = worker.get("training_provenance", {})
    expected_source = canonical_hash({path: digest for path, digest in declaration["source_sha256"].items()
                                      if path.startswith("src/") or path in {"scripts/run_sweep.py", "requirements.txt"}})
    require(set(provenance) == {"control_a", "control_b", "resumed"}, "completed training provenance missing")
    for row in provenance.values():
        precision = row.get("training_precision", {})
        require(row.get("source_fingerprint") == expected_source and precision.get("device_type") == "cuda"
                and precision.get("policy") == {"precision": "float32", "autocast_enabled": False, "autocast_dtype": None, "tf32_policy": "disabled"}
                and precision.get("policy", {}).get("autocast_enabled") is False
                and precision.get("runtime_flags", {}).get("cuda_matmul_allow_tf32") is False
                and precision.get("runtime_flags", {}).get("cudnn_allow_tf32") is False, "completed training FP32/source identity differs")


def validate_declaration(declaration, output, root=ROOT, *, worker_policy=None):
    require(type(declaration.get("schema")) is int and declaration["schema"] == 1
            and type(declaration.get("version")) is int and declaration["version"] == 1
            and declaration.get("kind") == KIND and declaration.get("status_at_declaration") == "planned_before_execution",
            "unexpected determinism declaration identity")
    require(declaration.get("train_config") == TRAIN_SETTINGS, "changed training declaration")
    for key, expected in TRAIN_SETTINGS.items():
        actual = declaration["train_config"][key]
        valid_type = (type(actual) is bool if type(expected) is bool else type(actual) is int if type(expected) is int
                      else type(actual) in (int, float) and math.isfinite(actual) if type(expected) is float
                      else type(actual) is type(expected))
        require(valid_type, "malformed training declaration")
    require(declaration.get("guards") == {"wall_seconds": WALL_SECONDS, "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES}, "changed guards")
    require(declaration.get("process_environment") == ENVIRONMENT, "changed cuBLAS environment")
    require(declaration.get("policies") == POLICIES, "changed deterministic policies")
    require(all(type(row[key]) is bool for row in declaration["policies"] for key in
                ("deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark")), "malformed deterministic policies")
    require(declaration.get("stop") == {"step": 2, "reason": STOP_REASON, "wrapper": WRAPPER_ID}, "changed stop injection")
    require(declaration.get("checkpoint_root") == CHECKPOINT_ROOT and declaration.get("run_ids") == RUN_IDS, "changed namespaces")
    require(Path(output).resolve() == public_path(declaration["output"], root).resolve(), "output differs from declaration")
    require(not Path(output).exists(), "output exists; no overwrite")
    names = (worker_policy,) if worker_policy is not None else ("legacy", "strict")
    require(len({Path(output).resolve(), *(path.resolve() for path in worker_output_paths(declaration, root).values())}) == 3,
            "parent/worker report paths must differ")
    require(all(not worker_output_paths(declaration, root)[name].exists() for name in names), "worker report exists; no overwrite")
    checkpoint_root = public_path(declaration["checkpoint_root"], root)
    require(all(not (checkpoint_root / run_id).exists() for name in names for run_id in RUN_IDS[name].values()),
            "run namespace exists; no overwrite")
    require(declaration.get("source_sha256") == source_registry(root), "source fingerprints differ")
    for label in ("model_config", "prior_smoke"):
        record = declaration[label]
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
                and file_sha256(public_path(record["path"], root)) == record["sha256"], f"{label} identity differs")
    data = declaration["data"]
    data_dir = public_path(data["directory"], root)
    require(re.fullmatch(r"[0-9a-f]{64}", data["signature"]) is not None
            and file_sha256(data_dir / "manifest.json") == data["manifest_sha256"], "data manifest identity differs")
    histories = declaration["historical_checkpoints"]
    require([row["ratio"] for row in histories] == ["1:3", "1:7", "1:15"], "historical registry differs")
    for row in histories:
        require(row["path"].startswith("checkpoints/week3-700m-v1/")
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "invalid historical evidence")
        require(file_sha256(public_path(row["path"], root)) == row["sha256"], "historical evidence differs")
    # Canonical encoding also rejects overflowed JSON numbers before any artifact/GPU work.
    canonical_hash(declaration)
    return checkpoint_root, data_dir


def policy_flags(torch):
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory}


@contextmanager
def deterministic_policy(policy, *, torch_module=None):
    torch = helpers().torch if torch_module is None else torch_module
    before = policy_flags(torch)
    observation = {"before": before, "requested": policy, "restored": False}
    try:
        if policy["name"] == "legacy":
            require(all(before[key] == policy[key] for key in
                        ("deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark")),
                    "legacy actual defaults differ from declared policy")
        else:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        observation["active"] = policy_flags(torch)
        yield observation
    finally:
        torch.use_deterministic_algorithms(before["deterministic_algorithms"], warn_only=before["warn_only"])
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.utils.deterministic.fill_uninitialized_memory = before["fill_uninitialized_memory"]
        observation["after"] = policy_flags(torch)
        observation["restored"] = observation["after"] == before


def descriptive_magnitudes(left, right):
    """Bounded-memory numeric/layout description; none of these values is an acceptance tolerance."""
    torch = helpers().torch
    summary = {"scope": "descriptive numeric differences; exact equality alone decides the check",
               "different_numeric_elements": 0, "nonfinite_pairs": 0, "max_absolute_difference": 0.0,
               "max_symmetric_relative_difference": 0.0, "relative_denominator": "max(abs(left),abs(right),1e-12)",
               "checkpoint_layout_differences": 0, "layout_examples": []}

    def numbers(a, b):
        a, b = a.double().reshape(-1), b.double().reshape(-1)
        # Inputs here are at most 262144 elements: avoid a full model-sized double copy.
        finite = torch.isfinite(a) & torch.isfinite(b)
        summary["nonfinite_pairs"] += int((~finite).sum().item())
        summary["different_numeric_elements"] += int((a != b).sum().item())
        if finite.any():
            a, b = a[finite], b[finite]
            diff = (a - b).abs()
            summary["max_absolute_difference"] = max(summary["max_absolute_difference"], diff.max().item())
            denominator = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
            summary["max_symmetric_relative_difference"] = max(summary["max_symmetric_relative_difference"], (diff / denominator).max().item())

    def walk(a, b, name):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype:
            if a.stride() != b.stride():
                summary["checkpoint_layout_differences"] += 1
                if len(summary["layout_examples"]) < 5:
                    summary["layout_examples"].append({"path": name, "shape": list(a.shape), "dtype": str(a.dtype),
                                                       "left_stride": list(a.stride()), "right_stride": list(b.stride())})
            if not torch.equal(a, b):
                flat_a, flat_b = a.reshape(-1), b.reshape(-1)
                for offset in range(0, a.numel(), 262144):
                    numbers(flat_a[offset:offset + 262144], flat_b[offset:offset + 262144])
        elif isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() & b.keys():
                walk(a[key], b[key], f"{name}.{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{name}.{index}")
        elif type(a) in (int, float) and type(b) in (int, float) and a != b:
            numbers(torch.tensor([a], dtype=torch.float64), torch.tensor([b], dtype=torch.float64))

    walk(left, right, "value")
    return summary


def compare_trajectories(left_dir, right_dir):
    smoke = helpers()
    result = smoke.compare_trajectories(left_dir, right_dir)
    left = smoke.torch.load(Path(left_dir) / "last.pt", map_location="cpu", weights_only=True)
    right = smoke.torch.load(Path(right_dir) / "last.pt", map_location="cpu", weights_only=True)
    values = {key: (left[key], right[key]) for key in
              ("model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss")}
    values["train_batch_generator"] = (left["rng"]["train_generator"], right["rng"]["train_generator"])
    values["semantic_metrics"] = (smoke.semantic_metrics(Path(left_dir) / "metrics.jsonl"),
                                  smoke.semantic_metrics(Path(right_dir) / "metrics.jsonl"))
    for key, pair in values.items():
        result["checks"][key]["descriptive_magnitudes"] = descriptive_magnitudes(*pair)
    result["tolerances_relaxed"] = False
    result["scope"] = "exact same-policy trajectory identity; descriptive magnitudes do not convert failures to passes"
    return result


def execute_policy(cfg, policy, *, started=None, guard=lambda: None, evidence_check=lambda: None, progress=None):
    """Real tiny CPU tests exercise this same orchestration; public workers remain CUDA-only."""
    smoke = helpers()
    trainer = smoke.trainer
    started = time.monotonic() if started is None else started
    progress = {} if progress is None else progress
    phases = progress.setdefault("phases", [])
    ids = RUN_IDS[policy["name"]]
    configurations = {arm: trainer.TrainConfig(**{**vars(cfg), "run_id": run_id}) for arm, run_id in ids.items()}
    require(all(not (Path(cfg.ckpt_dir) / run_id).exists() for run_id in ids.values()), "run namespace exists; no overwrite")
    model_name = trainer.load_model_config(cfg.model_config).name
    dirs = {arm: trainer.variant_run_dir(configuration, model_name) for arm, configuration in configurations.items()}

    def call(phase_name, arm, requested_stop=False):
        evidence_check()
        headroom = guard()
        configuration = configurations[arm]
        control = trainer.TrainingStopControl(configuration.wall_time_limit_seconds)
        control.started = started
        before_io, before_time = smoke.process_io(), time.monotonic()
        result, wrapper = None, None
        try:
            with trainer.cooperative_stop_signals(control):
                if requested_stop:
                    with smoke.request_stop_after_publication(control, ids["resumed"]) as wrapper:
                        result = trainer.run(configuration, stop_control=control)
                else:
                    result = trainer.run(configuration, stop_control=control)
        finally:
            phase = {"phase": phase_name, "arm": arm, "wall_seconds": time.monotonic() - before_time,
                     "io": smoke.io_delta(before_io, smoke.process_io()), "headroom_before": headroom,
                     "status": "incomplete" if result is None else result.get("status", "completed"),
                     "peak_allocated_bytes": smoke.torch.cuda.max_memory_allocated() if configuration.device == "cuda" else None}
            if result is not None:
                phase.update({key: result.get(key) for key in ("completed_steps", "tokens_seen", "stop", "observed_paths")})
            if wrapper is not None:
                phase["wrapper"] = wrapper
            phases.append(phase)
        require(time.monotonic() - started < configuration.wall_time_limit_seconds, "shared workflow allowance exceeded during in-flight work")
        evidence_check()
        if requested_stop:
            require(result.get("status") == "stopped" and result["completed_steps"] == 2
                    and result["stop"]["reason"] == STOP_REASON and wrapper["stop_requests"] == 1 and wrapper["restored"],
                    "expected complete-step-two stop did not occur")
            require(not (dirs[arm] / "result.json").exists(), "stopped arm published a completed result")
            phase["last_sha256_at_stop"] = file_sha256(dirs[arm] / "last.pt")
            phase["boundary_artifact_scope"] = "last.pt at stop; superseded by successful resume"
        else:
            require(result.get("status", "completed") == "completed" and result["completed_steps"] == cfg.max_steps,
                    "unexpected incomplete/budget-stopped arm")

    with deterministic_policy(policy) as observation:
        progress["policy_flags"] = observation
        call("control_a", "control_a")
        call("control_b", "control_b")
        progress["comparisons"] = {"control_a_vs_control_b": compare_trajectories(dirs["control_a"], dirs["control_b"])}
        call("requested_stop", "resumed", requested_stop=True)
        call("resumed", "resumed")
        progress["comparisons"]["control_a_vs_resumed"] = compare_trajectories(dirs["control_a"], dirs["resumed"])
    require(observation["restored"], "deterministic policy flags were not restored")
    return dirs


def cuda_guard(started):
    smoke = helpers()
    require(time.monotonic() - started < WALL_SECONDS, "shared workflow allowance exhausted")
    require(smoke.torch.cuda.is_available(), "CUDA unavailable; CPU fallback forbidden")
    smoke.torch.cuda.empty_cache()
    smoke.torch.cuda.synchronize()
    free, total = smoke.torch.cuda.mem_get_info()
    require(free >= MIN_FREE_CUDA_BYTES, "CUDA headroom below 8 GiB before next phase")
    return {"free_bytes": free, "total_bytes": total}


def run_worker(declaration_path, *, policy_name, started, root=ROOT):
    require(type(started) in (int, float) and math.isfinite(started) and 0 <= started <= time.monotonic(),
            "worker needs an actual shared monotonic origin, never a future timer reset")
    torch_imported_at_entry = "torch" in sys.modules
    require({key: os.environ.get(key) for key in ENVIRONMENT} == ENVIRONMENT, "worker cuBLAS environment differs")
    declaration = read_json(declaration_path)
    output = public_path(declaration["output"], root)
    checkpoint_root, data_dir = validate_declaration(declaration, output, root, worker_policy=policy_name)
    smoke = helpers()
    report = {"schema": 1, "kind": KIND, "policy": policy_name, "certified": False,
              "execution_status": "incomplete", "status": "incomplete", "process_environment": dict(ENVIRONMENT),
              "declaration_sha256": file_sha256(declaration_path), "source_sha256": declaration["source_sha256"],
              "comparison_endpoint": "exact equality in all eight categories; no tolerance acceptance",
              "research_pilot_executed": False, "backend_parity_certified": False}
    report["torch_imported_at_worker_entry"] = torch_imported_at_entry
    report["common_workflow_started_monotonic"] = started
    before_io, before_time = smoke.process_io(), time.monotonic()
    dirs = {}
    try:
        model = public_path(declaration["model_config"]["path"], root)
        mcfg = smoke.trainer.load_model_config(str(model))
        require((mcfg.ratio, mcfg.d_model, mcfg.n_layers, mcfg.vocab_size, mcfg.d_state, mcfg.head_dim,
                 mcfg.mamba_headdim, mcfg.n_attention_layers, mcfg.n_mamba_layers)
                == ("1:15", 448, 16, 16000, 128, 64, 64, 1, 15), "full registered ratio1:15 geometry required")
        require((mcfg.expand, mcfg.d_conv, mcfg.n_groups, mcfg.mlp_ratio, mcfg.mlp_multiple_of,
                 mcfg.tie_embeddings, mcfg.attn_bias, mcfg.mlp_bias, mcfg.mlp_on_every_layer)
                == (2, 4, 1, 2.6667, 64, True, False, False, True), "changed model geometry")
        manifest = smoke.validate_prepared_dataset(data_dir)
        require(manifest["signature"] == declaration["data"]["signature"] and manifest["tokenizer"]["vocab_size"] == 16000
                and all(manifest["outputs"][name]["tokens"] > 513 for name in ("train", "val")), "corpus identity/workload differs")
        evidence = {declaration["model_config"]["path"]: declaration["model_config"]["sha256"],
                    declaration["prior_smoke"]["path"]: declaration["prior_smoke"]["sha256"],
                    f"{declaration['data']['directory']}/manifest.json": declaration["data"]["manifest_sha256"]}
        evidence.update({f"{declaration['data']['directory']}/{row['file']}": row["sha256"] for row in manifest["outputs"].values()})
        evidence.update({row["path"]: row["sha256"] for row in declaration["historical_checkpoints"]})

        def verify():
            require(file_sha256(declaration_path) == report["declaration_sha256"], "declaration changed during workflow")
            require(source_registry(root) == declaration["source_sha256"], "source files changed during workflow")
            for name, digest in evidence.items():
                require(file_sha256(public_path(name, root)) == digest, "input/historical evidence changed")

        verify()
        report["initial_cuda_headroom"] = cuda_guard(started)
        report["runtime_ambient_before_policy"] = smoke.runtime_metadata()
        policy = next(row for row in POLICIES if row["name"] == policy_name)
        configuration = smoke.trainer.TrainConfig(**TRAIN_SETTINGS, model_config=str(model), data_dir=str(data_dir),
                                                 ckpt_dir=str(checkpoint_root), run_id=RUN_IDS[policy_name]["control_a"])
        dirs = {arm: smoke.trainer.variant_run_dir(smoke.trainer.TrainConfig(**{**vars(configuration), "run_id": run_id}), mcfg.name)
                for arm, run_id in RUN_IDS[policy_name].items()}
        dirs = execute_policy(configuration, policy, started=started, guard=lambda: cuda_guard(started), evidence_check=verify, progress=report)
        verify()
        report["execution_status"] = "completed"
        report["status"] = "completed" if all(row["passed"] for row in report["comparisons"].values()) else "completed_with_identity_failures"
    except (RuntimeError, ValueError, FloatingPointError, OSError) as error:
        report["reason"] = sanitize(f"{type(error).__name__}: {error}", root)
    report["artifacts"] = {}
    report["training_provenance"] = {}
    report["artifact_capture_scope"] = "files present at worker end, including incomplete trajectories; existence does not imply completion"
    for arm, path in dirs.items():
        try:
            report["artifacts"][arm] = {label: {"path": (path / filename).relative_to(root).as_posix(),
                                               "sha256": file_sha256(path / filename)}
                                        for label, filename in {**smoke.trainer.ARTIFACT_FILES, "manifest": "manifest.json"}.items()
                                        if (path / filename).is_file()}
            if (path / "manifest.json").is_file():
                manifest = smoke.trainer.read_json(path / "manifest.json")
                report["training_provenance"][arm] = {"signature": manifest["signature"],
                                                       "source_fingerprint": manifest["code"]["fingerprint"], "git": manifest["git"],
                                                       "training_precision": manifest.get("training_precision")}
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            report.setdefault("artifact_capture_errors", []).append({"arm": arm, "reason": sanitize(f"{type(error).__name__}: {error}", root)})
            report["execution_status"] = report["status"] = "incomplete"
    report["historical_preservation"] = smoke.historical_preservation(declaration["historical_checkpoints"], root)
    report["historical_bytes_unchanged"] = report["historical_preservation"]["all_bytes_unchanged"]
    report["worker_wall_seconds"] = time.monotonic() - before_time
    report["shared_workflow_elapsed_seconds"] = time.monotonic() - started
    report["wall_overshoot_seconds"] = max(0.0, report["shared_workflow_elapsed_seconds"] - WALL_SECONDS)
    report["worker_io"] = smoke.io_delta(before_io, smoke.process_io())
    report["worker_resource_scope"] = "worker after declaration/source/history preflight: corpus validation, CUDA checks, training, comparisons and final evidence; excludes import/startup and report publication"
    report["cuda_peak_scope"] = "cumulative worker CUDA allocation high-water mark as of each phase, not an exclusive phase peak"
    if not report["historical_bytes_unchanged"] or report["shared_workflow_elapsed_seconds"] >= WALL_SECONDS:
        report["execution_status"] = report["status"] = "incomplete"
        report.setdefault("reason", "historical preservation or shared wall allowance failed")
    return smoke.public_metadata(report, root)


def run_declared(declaration_path, output, *, root=ROOT, launch=subprocess.run):
    """Serial fresh processes share one real monotonic deadline; there is no hard-kill promise."""
    started = time.monotonic()
    declaration = read_json(declaration_path)
    validate_declaration(declaration, output, root)
    declaration_sha = file_sha256(declaration_path)
    report = {"schema": 1, "kind": KIND, "date": declaration["date"], "certified": False,
              "declaration_sha256": declaration_sha, "declaration_canonical_sha256": canonical_hash(declaration),
              "execution_status": "incomplete", "status": "incomplete", "policies": [],
              "research_pilot_executed": False, "backend_parity_certified": False,
              "process_environment": dict(ENVIRONMENT), "source_sha256": declaration["source_sha256"],
              "common_workflow_started_monotonic": started,
              "resource_scope": "shared parent/worker workflow wall time; per-worker OS process I/O excludes parent and final report publication",
              "limits": ["One initialization, one full model shape and one reference FP32 policy; no quality or backend parity certificate.",
                         "Both policies receive the same cuBLAS environment; legacy is not a byte-for-byte replay of the older ambient process.",
                         "Exact endpoint retained; numeric magnitudes are descriptive and never relax a tolerance.",
                         "Checkpoint layouts describe serialized CPU tensors, not live gradient/workspace layout or kernel identity.",
                         "Deterministic control can fail on unsupported operations; this is retained as incomplete.",
                         "Cooperative stop checks allow in-flight overshoot; there is no hard wall-time guarantee.",
                         "Run-to-run controls precede causal attribution; strict success would not prove which operator caused earlier drift."]}
    outputs = worker_output_paths(declaration, root)
    evidence = declared_evidence(declaration, root)
    worker_registry = {}
    for name in ("legacy", "strict"):
        if time.monotonic() - started >= WALL_SECONDS:
            report["policies"].append({"policy": name, "execution_status": "not_started", "status": "incomplete",
                                       "reason": "shared workflow allowance exhausted before worker start"})
            continue
        try:
            require(file_sha256(declaration_path) == declaration_sha and source_registry(root) == declaration["source_sha256"], "declaration/source changed")
            command = [sys.executable, str(Path(root) / "scripts/study_cuda_recovery_determinism.py"),
                       "--declaration", str(Path(declaration_path).resolve()), "--worker", name, "--started", repr(started)]
            completed = launch(command, cwd=root, env={**os.environ, **ENVIRONMENT}, capture_output=True, text=True)
            require(outputs[name].is_file(), f"{name} worker did not preserve a report (exit {completed.returncode})")
            worker = read_json(outputs[name])
            worker_registry[name] = {"path": outputs[name].relative_to(root).as_posix(), "sha256": file_sha256(outputs[name])}
            validate_worker_result(worker, name, declaration, declaration_sha, completed.returncode, expected_started=started)
            report["policies"].append({**worker, "worker_report": worker_registry[name],
                                        "worker_exit_code": completed.returncode})
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            report["policies"].append({"policy": name, "execution_status": "incomplete", "status": "incomplete",
                                       "reason": sanitize(f"{type(error).__name__}: {error}", root),
                                       "worker_report": worker_registry.get(name)})
    try:
        require(file_sha256(declaration_path) == declaration_sha and source_registry(root) == declaration["source_sha256"], "final declaration/source audit failed")
        for path, digest in {**evidence, **{row["path"]: row["sha256"] for row in worker_registry.values()}}.items():
            require(file_sha256(public_path(path, root)) == digest, "final input/worker evidence audit failed")
        report["final_evidence_audit"] = {"passed": True, "source_sha256": declaration["source_sha256"], "inputs_sha256": evidence,
                                            "worker_reports": worker_registry}
    except (RuntimeError, ValueError, OSError) as error:
        report["final_evidence_audit"] = {"passed": False, "reason": sanitize(f"{type(error).__name__}: {error}", root)}
    report["historical_preservation"] = {
        "verified_in_final_audit": report["final_evidence_audit"]["passed"],
        "all_bytes_unchanged": True if report["final_evidence_audit"]["passed"] else None,
        "scope": "historical checkpoint bytes included in full final input audit; null means the audit did not complete",
    }
    all_completed = all(row["execution_status"] == "completed" for row in report["policies"])
    report["workflow_wall_seconds"] = time.monotonic() - started
    report["wall_overshoot_seconds"] = max(0.0, report["workflow_wall_seconds"] - WALL_SECONDS)
    if all_completed and report["workflow_wall_seconds"] < WALL_SECONDS and report["final_evidence_audit"]["passed"]:
        report["execution_status"] = "completed"
        report["status"] = "completed" if all(row["status"] == "completed" for row in report["policies"]) else "completed_with_identity_failures"
    return report


def write_report(path, report):
    """Exclusive, flushed publication without importing PyTorch in the parent."""
    path = Path(path)
    require(not path.exists(), "report exists; no overwrite")
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("legacy", "strict"), help=argparse.SUPPRESS)
    parser.add_argument("--started", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            require(args.started is not None and math.isfinite(args.started), "worker needs common monotonic start")
            require("torch" not in sys.modules, "worker must set cuBLAS environment before PyTorch import")
            declaration = read_json(args.declaration)
            report = run_worker(args.declaration, policy_name=args.worker, started=args.started)
            output = worker_output_paths(declaration)[args.worker]
        else:
            require(args.output is not None, "--output is required for parent")
            report = run_declared(args.declaration, args.output)
            output = args.output
        write_report(output, report)
    except (RuntimeError, ValueError, FloatingPointError, KeyError, OSError) as error:
        parser.error(sanitize(f"invalid determinism study: {error}"))
    print(json.dumps({"status": report["status"], "execution_status": report["execution_status"], "certified": False}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
