"""Strict, standard-library export of separate repeatability and recovery gates.

Public hashes bind the producer's private binary audit. This exporter never loads
checkpoint tensors, runs training, imports PyTorch or turns a missing check into
a pass. Exact equality remains the endpoint; magnitudes are descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_checkpoint_numerics as base  # noqa: E402
from scripts.summarize_decay_precision import same  # noqa: E402

KIND = "cuda_recovery_determinism_study"
ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
WALL_SECONDS = 900
MIN_FREE = 8 * 1024**3
CHECKPOINT_ROOT = "checkpoints/operational-determinism-2026-10-04"
POLICIES = [{"name": name, "deterministic_algorithms": strict, "warn_only": False,
             "cudnn_deterministic": strict, "cudnn_benchmark": False} for name, strict in (("legacy", False), ("strict", True))]
RUN_IDS = {name: {arm: f"cuda-recovery-{name}-{suffix}-v1" for arm, suffix in
                 (("control_a", "control-a"), ("control_b", "control-b"), ("resumed", "resume"))}
           for name in ("legacy", "strict")}
TRAIN_SETTINGS = {"device": "cuda", "precision": "float32", "scan_backend": "reference", "scan_chunk_size": 128,
    "seed": 1337, "model_seed": 1337, "data_seed": 1337, "eval_seed": 1337,
    "max_steps": 4, "block_size": 512, "batch_size": 1, "grad_accum": 1, "warmup_steps": 1,
    "eval_interval": 2, "eval_iters": 1, "checkpoint_interval": 1, "log_interval": 1,
    "lr": .001, "min_lr": .0001, "weight_decay": .1, "beta1": .9, "beta2": .95,
    "grad_clip": 1., "grad_checkpointing": True, "wandb": False, "resume": True, "wall_time_limit_seconds": 900.}
STOP_REASON = "diagnostic:after-complete-checkpoint-step-2"
WRAPPER_ID = "post-successful _publish_checkpoint_transaction request_stop; restored in finally"
SOURCES = {"src/train/train.py", "src/serve/__init__.py", "src/serve/registry.py", "src/serve/app.py",
    "src/model/scan_backend.py", "src/model/norm.py", "src/model/mlp.py", "src/model/mamba2.py", "src/model/lm.py",
    "src/model/inference.py", "src/model/config.py", "src/model/block.py", "src/model/attention.py", "src/generation.py",
    "src/eval/week5_report.py", "src/eval/suite.py", "src/eval/report.py", "src/data/train_tokenizer.py",
    "src/data/prepare_data.py", "src/data/dataset.py", "scripts/run_sweep.py", "requirements.txt",
    "scripts/check_cuda_checkpoint_recovery.py", "scripts/study_cuda_recovery_determinism.py"}
CATEGORIES = {"model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss", "train_batch_generator", "semantic_metrics"}
COMPARISONS = ["control_a_vs_control_b", "control_a_vs_resumed"]
PHASES = [("control_a", "control_a"), ("control_b", "control_b"), ("requested_stop", "resumed"), ("resumed", "resumed")]
PRECISION = {"policy": {"precision": "float32", "autocast_enabled": False, "autocast_dtype": None, "tf32_policy": "disabled"},
             "device_type": "cuda", "runtime_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}}
ARTIFACTS = {"best": "best.pt", "last": "last.pt", "metrics": "metrics.jsonl", "result": "result.json", "manifest": "manifest.json"}
FLAG_KEYS = {"deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "fill_uninitialized_memory"}


def path(value, root):
    relative = base.relative_path(value)
    root = Path(root).resolve()
    result = root / relative
    base.require(result.resolve().is_relative_to(root), "public path escapes root")
    current = result
    while current != root:
        base.require(not current.is_symlink(), "public path traverses symlink")
        current = current.parent
    return result


def sources(registry, source_root):
    same(sorted(registry), sorted(SOURCES), "source coverage differs")
    for name, recorded in registry.items():
        base.digest(recorded, "source " + name)
        raw = path(name, source_root).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        base.require(recorded in {hashlib.sha256(item).hexdigest() for item in (raw, lf, lf.replace(b"\n", b"\r\n"))},
                     "stale measured source: " + name)


def declaration_contract(d, source_root):
    for name, expected in {"schema": 1, "version": 1, "kind": KIND, "status_at_declaration": "planned_before_execution",
                           "train_config": TRAIN_SETTINGS, "guards": {"wall_seconds": WALL_SECONDS, "min_free_cuda_bytes": MIN_FREE},
                           "process_environment": ENVIRONMENT, "policies": POLICIES, "checkpoint_root": CHECKPOINT_ROOT,
                           "run_ids": RUN_IDS, "stop": {"step": 2, "reason": STOP_REASON, "wrapper": WRAPPER_ID}}.items():
        same(d[name], expected, "declaration differs: " + name)
    base.require(isinstance(d["date"], str) and d["date"], "declaration date missing")
    sources(d["source_sha256"], source_root)
    same(d["model_config"]["path"], "configs/ratio_1_15.yaml", "model configuration path differs")
    for key in ("model_config", "prior_smoke"):
        base.digest(d[key]["sha256"], key)
    histories = d["historical_checkpoints"]
    same([row["ratio"] for row in histories], base.RATIOS, "historical checkpoint matrix differs")
    for row in histories:
        base.require(row["path"].startswith("checkpoints/week3-700m-v1/") and row["path"].endswith("/best.pt"), "historical namespace differs")
        base.relative_path(row["path"])
        base.digest(row["sha256"], "historical checkpoint")
    base.relative_path(d["data"]["directory"])
    base.digest(d["data"]["signature"], "data signature")
    base.digest(d["data"]["manifest_sha256"], "data manifest")
    base.relative_path(d["output"])


def descriptive(value):
    same(sorted(value), sorted(["scope", "different_numeric_elements", "nonfinite_pairs", "max_absolute_difference",
        "max_symmetric_relative_difference", "relative_denominator", "checkpoint_layout_differences", "layout_examples"]), "descriptive fields differ")
    same(value["scope"], "descriptive numeric differences; exact equality alone decides the check", "magnitude scope differs")
    same(value["relative_denominator"], "max(abs(left),abs(right),1e-12)", "relative denominator differs")
    for key in ("different_numeric_elements", "nonfinite_pairs", "checkpoint_layout_differences"):
        base.integer(value[key], key)
    for key in ("max_absolute_difference", "max_symmetric_relative_difference"):
        base.number(value[key], key, 0)
    base.require((value["max_absolute_difference"] == 0) == (value["max_symmetric_relative_difference"] == 0), "magnitude maxima inconsistent")
    base.require(value["different_numeric_elements"] > 0 or value["max_absolute_difference"] == 0, "nonzero error without unequal numeric elements")
    examples = value["layout_examples"]
    base.require(isinstance(examples, list) and len(examples) == min(5, value["checkpoint_layout_differences"]), "layout example coverage differs")
    for item in examples:
        same(sorted(item), sorted(["path", "shape", "dtype", "left_stride", "right_stride"]), "layout fields differ")
        base.require(isinstance(item["path"], str) and item["path"].startswith("value.") and isinstance(item["dtype"], str)
                     and item["dtype"].startswith("torch."), "layout path/dtype differs")
        base.require(isinstance(item["shape"], list), "layout shape missing")
        for size in item["shape"]:
            base.integer(size, "layout shape")
        for name in ("left_stride", "right_stride"):
            base.require(isinstance(item[name], list) and len(item[name]) == len(item["shape"]), "layout rank differs")
            for stride in item[name]:
                base.integer(stride, "layout stride")
        base.require(item["left_stride"] != item["right_stride"], "equal layout in difference example")


def comparison(value):
    same(sorted(value), sorted(["passed", "checks", "timing_excluded", "scope", "tolerances_relaxed"]), "comparison fields differ")
    same(sorted(value["checks"]), sorted(CATEGORIES), "missing/extra exact categories")
    passes = []
    for name, check in value["checks"].items():
        same(sorted(check), sorted(["exact_equal", "tensor_checks", "difference_count", "first_differences", "descriptive_magnitudes"]), "exact category fields differ")
        base.require(type(check["exact_equal"]) is bool, "exact flag must be boolean")
        base.integer(check["tensor_checks"], "exact tensor count")
        base.integer(check["difference_count"], "exact difference count")
        first = check["first_differences"]
        base.require(isinstance(first, list) and len(first) == min(20, check["difference_count"])
                     and all(isinstance(item, str) and item.startswith("value") for item in first), "exact difference examples differ")
        base.flag(check["exact_equal"], check["difference_count"] == 0, "category exact gate")
        descriptive(check["descriptive_magnitudes"])
        if check["exact_equal"]:
            base.require(check["descriptive_magnitudes"]["different_numeric_elements"] == 0, "exact category has unequal numeric values")
        passes.append(check["exact_equal"])
    base.flag(value["passed"], all(passes), "comparison aggregate")
    base.require(value["tolerances_relaxed"] is False, "exact endpoint relaxed")
    same(value["scope"], "exact same-policy trajectory identity; descriptive magnitudes do not convert failures to passes", "comparison scope differs")
    same(value["timing_excluded"], ["train_seconds", "eval_seconds", "peak_vram_mb", "step_seconds", "tok_per_s"], "timing exclusions differ")


def io(value):
    base.require(type(value["available"]) is bool, "I/O availability flag malformed")
    if not value["available"]:
        base.require(value["source"] is None and value["counters"] is None, "unavailable I/O has measured counters")
    else:
        base.require(value["source"] in ("Windows GetProcessIoCounters", "Linux /proc/self/io")
                     and isinstance(value["counters"], dict), "I/O source differs")
        for key, count in value["counters"].items():
            base.require(isinstance(key, str), "I/O counter name malformed")
            base.integer(count, "I/O counter")
        same(value["scope"], "OS process transfer counters, including cached I/O and checks; not physical-disk bytes", "I/O scope differs")


def headroom(value):
    for key in ("free_bytes", "total_bytes"):
        base.integer(value[key], "CUDA " + key, 1)
    base.require(MIN_FREE <= value["free_bytes"] <= value["total_bytes"], "CUDA headroom guard did not hold")


def policy_observation(value, policy, completed):
    same(value["requested"], policy, "requested deterministic policy differs")
    for name in ("before", "active", "after"):
        if name not in value:
            base.require(not completed, "completed policy observation missing flags")
            continue
        same(sorted(value[name]), sorted(FLAG_KEYS), "runtime flag coverage differs")
        base.require(all(type(flag) is bool for flag in value[name].values()), "runtime flags malformed")
    if "active" in value:
        for key in ("deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark"):
            base.flag(value["active"][key], policy[key], "effective deterministic policy")
    if "after" in value:
        base.flag(value["restored"], value["after"] == value["before"], "policy restoration")
    base.require(not completed or value["restored"] is True, "completed policy not restored")


def phases_contract(phases, completed):
    base.require(isinstance(phases, list) and len(phases) <= len(PHASES), "phase coverage differs")
    same([(phase["phase"], phase["arm"]) for phase in phases], PHASES[:len(phases)], "missing/duplicate/unordered phases")
    base.require(not completed or len(phases) == len(PHASES), "completed worker missing phases")
    for phase in phases:
        base.number(phase["wall_seconds"], "phase wall seconds", 0)
        base.integer(phase["peak_allocated_bytes"], "cumulative allocation peak")
        headroom(phase["headroom_before"])
        io(phase["io"])
        base.require(phase["status"] in ("completed", "stopped", "incomplete"), "phase status unknown")
        if phase["status"] != "incomplete":
            base.integer(phase["completed_steps"], "phase completed steps")
            base.integer(phase["tokens_seen"], "phase token positions")
            base.require(phase["tokens_seen"] == phase["completed_steps"] * 512, "phase sampled-token counter differs")
            same(phase["observed_paths"], {"training": "torch.quadratic_ssd", "prefill": None, "decode": None}, "actual reference training route differs")
        if completed:
            expected = 2 if phase["phase"] == "requested_stop" else 4
            same(phase["completed_steps"], expected, "completed phase update count differs")
            same(phase["status"], "stopped" if expected == 2 else "completed", "completed phase status differs")
        if phase["phase"] == "requested_stop" and "wrapper" in phase:
            wrapper = phase["wrapper"]
            same(wrapper["wrapper_id"], WRAPPER_ID, "stop wrapper identity differs")
            same(wrapper["stop_reason"], STOP_REASON, "stop reason differs")
            same(wrapper["original_binding"], "src.train.train._publish_checkpoint_transaction", "stop publisher binding differs")
            base.require(type(wrapper["restored"]) is bool, "wrapper restoration malformed")
            base.require(isinstance(wrapper["successful_steps"], list), "published boundary steps missing")
            for step in wrapper["successful_steps"]:
                base.integer(step, "published boundary step")
            base.require(wrapper["successful_steps"] == sorted(set(wrapper["successful_steps"])), "published boundary steps duplicated or unordered")
            base.integer(wrapper["stop_requests"], "stop requests")
            if completed:
                base.require(wrapper["restored"] is True and wrapper["stop_requests"] == 1 and 2 in wrapper["successful_steps"], "complete checkpoint stop not established")
                same(phase["stop"]["reason"], STOP_REASON, "phase stop reason differs")
                base.digest(phase["last_sha256_at_stop"], "stop checkpoint")
        base.require(not completed or phase["phase"] != "requested_stop" or "wrapper" in phase, "completed stop wrapper missing")


def preservation(value, declaration):
    checks = value["checks"]
    same([row["path"] for row in checks], [row["path"] for row in declaration["historical_checkpoints"]], "preservation checkpoint matrix differs")
    for check, expected in zip(checks, declaration["historical_checkpoints"]):
        same(check["expected_sha256"], expected["sha256"], "preservation expected digest differs")
        if check["actual_sha256"] is not None:
            base.digest(check["actual_sha256"], "preservation actual digest")
        base.flag(check["bytes_unchanged"], check["actual_sha256"] == expected["sha256"], "historical preservation")
    base.flag(value["all_bytes_unchanged"], all(check["bytes_unchanged"] for check in checks), "preservation aggregate")


def worker_contract(worker, policy, declaration, declaration_hash):
    for key, expected in {"schema": 1, "kind": KIND, "policy": policy["name"], "certified": False,
                           "declaration_sha256": declaration_hash, "source_sha256": declaration["source_sha256"],
                           "process_environment": ENVIRONMENT, "research_pilot_executed": False, "backend_parity_certified": False}.items():
        same(worker[key], expected, "worker identity differs: " + key)
    completed = worker["execution_status"] == "completed"
    base.require(worker["torch_imported_at_worker_entry"] is False, "worker did not precede PyTorch import")
    same(worker["comparison_endpoint"], "exact equality in all eight categories; no tolerance acceptance", "worker exact endpoint differs")
    base.require(worker["execution_status"] in ("completed", "incomplete"), "worker execution status unknown")
    comparisons = worker.get("comparisons", {})
    base.require(isinstance(comparisons, dict) and list(comparisons) in ([], COMPARISONS[:1], COMPARISONS), "comparison order/coverage differs")
    for value in comparisons.values():
        comparison(value)
    base.require(not completed or set(comparisons) == set(COMPARISONS), "completed worker missing exact comparison")
    expected_status = "incomplete" if not completed else "completed" if all(value["passed"] for value in comparisons.values()) else "completed_with_identity_failures"
    same(worker["status"], expected_status, "worker status hides failed/missing checks")
    base.require(completed or isinstance(worker.get("reason"), str) and worker["reason"], "incomplete worker reason missing")
    phases_contract(worker.get("phases", []), completed)
    phases = worker.get("phases", [])
    for name, required_phases in zip(COMPARISONS, (2, 4)):
        if name in comparisons:
            base.require(len(phases) >= required_phases, "comparison precedes its required training phases")
            phases_contract(phases[:required_phases], required_phases == 4)
            for phase in phases[:required_phases]:
                expected = 2 if phase["phase"] == "requested_stop" else 4
                same(phase["completed_steps"], expected, "recorded comparison has unfinished trajectory")
                same(phase["status"], "stopped" if expected == 2 else "completed", "recorded comparison has unfinished phase")
    if "policy_flags" in worker:
        policy_observation(worker["policy_flags"], policy, completed)
    else:
        base.require(not completed, "completed worker policy observation missing")
    if "runtime_ambient_before_policy" in worker:
        runtime = worker["runtime_ambient_before_policy"]
        base.require(isinstance(runtime["gpu"], str) and runtime["gpu"] and isinstance(runtime["cuda_runtime"], str)
                     and runtime["cuda_runtime"], "CUDA runtime identity missing")
        same(runtime["training_policy"], PRECISION["policy"], "ambient training policy differs")
        same(sorted(runtime["packages"]), sorted(["torch", "numpy", "tokenizers", "mamba-ssm", "triton", "causal-conv1d"]), "runtime package coverage differs")
        for name in ("torch", "numpy", "tokenizers"):
            base.require(isinstance(runtime["packages"][name], str) and runtime["packages"][name], "required runtime package version missing")
        for version in runtime["packages"].values():
            base.require(version is None or isinstance(version, str) and version, "runtime package version malformed")
        base.require(isinstance(runtime["python"], str) and runtime["python"] and isinstance(runtime["platform"], str) and runtime["platform"], "Python/platform identity missing")
        same(len(runtime["capability"]), 2, "CUDA capability rank differs")
        for item in runtime["capability"]:
            base.integer(item, "CUDA capability")
        if "policy_flags" in worker:
            before = worker["policy_flags"]["before"]
            same(runtime["deterministic_algorithms"], before["deterministic_algorithms"], "ambient-before-policy deterministic flag differs")
            same(runtime["ambient_flags_outside_training"], {key: before[key] for key in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32")}, "ambient-before-policy TF32 flags differ")
    else:
        base.require(not completed, "completed worker runtime missing")
    provenances = worker.get("training_provenance", {})
    base.require(set(provenances) <= set(RUN_IDS[policy["name"]]), "unknown training arm")
    base.require(not completed or set(provenances) == set(RUN_IDS[policy["name"]]), "completed worker missing training provenance")
    for value in provenances.values():
        base.digest(value["signature"], "training signature")
        base.digest(value["source_fingerprint"], "training source fingerprint")
        expected_source = base.canonical_sha256({name: digest for name, digest in declaration["source_sha256"].items()
                                               if name.startswith("src/") or name in {"scripts/run_sweep.py", "requirements.txt"}})
        same(value["source_fingerprint"], expected_source, "training source fingerprint differs from declaration")
        same(value["training_precision"], PRECISION, "actual FP32 training precision differs")
    artifacts = worker.get("artifacts", {})
    base.require(set(artifacts) <= set(RUN_IDS[policy["name"]]), "unknown artifact arm")
    base.require(not completed or set(artifacts) == set(RUN_IDS[policy["name"]]), "completed artifact arms missing")
    for arm, registry in artifacts.items():
        base.require(set(registry) <= set(ARTIFACTS), "unknown checkpoint artifact")
        base.require(not completed or set(registry) == set(ARTIFACTS), "completed artifacts missing")
        parents = []
        for label, record in registry.items():
            relative = base.relative_path(record["path"])
            base.require(relative.startswith(CHECKPOINT_ROOT + "/" + RUN_IDS[policy["name"]][arm] + "/")
                         and Path(relative).name == ARTIFACTS[label], "artifact namespace/filename differs")
            base.digest(record["sha256"], "training artifact")
            parents.append(Path(relative).parent.as_posix())
        base.require(len(set(parents)) <= 1, "arm artifacts use different run directories")
    if "historical_preservation" in worker:
        preservation(worker["historical_preservation"], declaration)
        base.flag(worker["historical_bytes_unchanged"], worker["historical_preservation"]["all_bytes_unchanged"], "worker preservation flag")
    base.require(not completed or worker.get("historical_bytes_unchanged") is True, "completed worker historical bytes unstable")
    base.require(not completed or "historical_preservation" in worker, "completed worker preservation audit missing")
    for key in ("worker_wall_seconds", "shared_workflow_elapsed_seconds", "wall_overshoot_seconds"):
        if key in worker:
            base.number(worker[key], key, 0)
        else:
            base.require(not completed, "completed resource identity missing")
    if "shared_workflow_elapsed_seconds" in worker:
        base.close_double(worker["wall_overshoot_seconds"], max(0., worker["shared_workflow_elapsed_seconds"] - WALL_SECONDS), "worker wall overshoot")
        base.require(not completed or worker["shared_workflow_elapsed_seconds"] < WALL_SECONDS, "completed worker exceeded common allowance")
    if "worker_io" in worker:
        io(worker["worker_io"])
    if completed:
        same(worker["cuda_peak_scope"], "cumulative worker CUDA allocation high-water mark as of each phase, not an exclusive phase peak", "CUDA high-water scope differs")
        same(worker["worker_resource_scope"], "worker after declaration/source/history preflight: corpus validation, CUDA checks, training, comparisons and final evidence; excludes import/startup and report publication", "worker resource scope differs")
        base.require("worker_io" in worker, "completed worker I/O observation missing")
    return completed


def compact_comparison(value):
    if value is None:
        return None
    return {**value, "category_counts": {"passed": sum(check["exact_equal"] for check in value["checks"].values()), "total": len(value["checks"])},
            "failed_categories": [name for name, check in value["checks"].items() if not check["exact_equal"]]}


def build_summary(declaration_path, raw_path=None, root=ROOT, source_root=ROOT):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    declaration, declaration_hash = base.read_json(declaration_path)
    declaration_contract(declaration, source_root)
    declared_raw = path(declaration["output"], root).resolve()
    raw_path = declared_raw if raw_path is None else Path(raw_path).resolve()
    base.require(raw_path == declared_raw, "raw path differs from declaration")
    report, raw_hash = base.read_json(raw_path)
    base.number(report["common_workflow_started_monotonic"], "shared monotonic origin", 0)
    for key, expected in {"schema": 1, "kind": KIND, "date": declaration["date"], "certified": False,
                           "declaration_sha256": declaration_hash, "declaration_canonical_sha256": base.canonical_sha256(declaration),
                           "source_sha256": declaration["source_sha256"], "process_environment": ENVIRONMENT,
                           "research_pilot_executed": False, "backend_parity_certified": False}.items():
        same(report[key], expected, "parent report identity differs: " + key)
    prior, prior_hash = base.read_json(path(declaration["prior_smoke"]["path"], root))
    same(prior_hash, declaration["prior_smoke"]["sha256"], "prior smoke bytes differ")
    base.require(prior["kind"] == "cuda_checkpoint_recovery_operational_smoke" and prior["certified"] is False, "prior smoke identity differs")
    for key in ("model_config", "data", "historical_checkpoints"):
        same(declaration[key], prior["protocol"][key], "prior immutable input registry differs: " + key)
    same(base.file_sha256(path(declaration["model_config"]["path"], source_root)), declaration["model_config"]["sha256"], "model configuration bytes differ")
    manifest, manifest_hash = base.read_json(path(declaration["data"]["directory"] + "/manifest.json", source_root))
    same(manifest_hash, declaration["data"]["manifest_sha256"], "data manifest bytes differ")
    same(manifest["signature"], declaration["data"]["signature"], "data signature differs")
    same(sorted(manifest["outputs"]), ["meta", "train", "val"], "prepared manifest output coverage differs")
    inputs = {declaration[key]["path"]: declaration[key]["sha256"] for key in ("model_config", "prior_smoke")}
    inputs[declaration["data"]["directory"] + "/manifest.json"] = manifest_hash
    inputs.update({declaration["data"]["directory"] + "/" + base.relative_path(value["file"]): value["sha256"] for value in manifest["outputs"].values()})
    inputs.update({row["path"]: row["sha256"] for row in declaration["historical_checkpoints"]})
    for digest in inputs.values():
        base.digest(digest, "input audit digest")
    base.require(isinstance(report["limits"], list) and all(isinstance(item, str) and item for item in report["limits"]), "report limits malformed")
    policies = report["policies"]
    same([row["policy"] for row in policies], [policy["name"] for policy in POLICIES], "missing/duplicate/unordered policy rows")
    rows, bindings = [], {}
    for entry, policy in zip(policies, POLICIES):
        executed = entry["execution_status"] in ("completed", "incomplete") and "schema" in entry
        binding = entry.get("worker_report")
        if binding is not None:
            expected_path = declared_raw.with_name(policy["name"] + "-worker.json").relative_to(root).as_posix()
            same(binding["path"], expected_path, "worker report path differs")
            raw_worker, worker_hash = base.read_json(path(binding["path"], root))
            same(binding["sha256"], worker_hash, "worker report bytes differ")
            bindings[policy["name"]] = binding
        if executed:
            base.require(binding is not None, "executed worker has no immutable report")
            same({key: value for key, value in entry.items() if key not in ("worker_report", "worker_exit_code")}, raw_worker, "parent worker payload differs from raw worker")
            completed = worker_contract(raw_worker, policy, declaration, declaration_hash)
            same(raw_worker["common_workflow_started_monotonic"], report["common_workflow_started_monotonic"], "worker uses a different workflow clock origin")
            expected_exit = 0 if completed and raw_worker["status"] == "completed" else 2
            same(entry["worker_exit_code"], expected_exit, "worker exit code differs from exact gate")
        else:
            base.require(entry["execution_status"] in ("incomplete", "not_started") and entry["status"] == "incomplete"
                         and isinstance(entry.get("reason"), str) and entry["reason"], "incomplete policy reason/status differs")
        comparisons = entry.get("comparisons", {}) if executed else {}
        row = {"policy": policy["name"], "execution_status": entry["execution_status"], "status": entry["status"], "reason": entry.get("reason"),
               "worker_exit_code": entry.get("worker_exit_code"), "worker_report": binding,
               "runtime_ambient_before_policy": entry.get("runtime_ambient_before_policy") if executed else None,
               "policy_flags": entry.get("policy_flags") if executed else None,
               "comparisons": {name: compact_comparison(comparisons.get(name)) for name in COMPARISONS},
               "phases": [{key: phase.get(key) for key in ("phase", "arm", "status", "completed_steps", "tokens_seen", "wall_seconds", "observed_paths", "wrapper")}
                          | {"peak_allocated_mib": phase["peak_allocated_bytes"] / 1048576} for phase in entry.get("phases", [])] if executed else [],
               "training_precision": {arm: value["training_precision"] for arm, value in entry.get("training_provenance", {}).items()} if executed else {},
               "historical_bytes_unchanged": entry.get("historical_bytes_unchanged") if executed else None,
               "worker_wall_seconds": entry.get("worker_wall_seconds") if executed else None,
               "shared_workflow_elapsed_seconds": entry.get("shared_workflow_elapsed_seconds") if executed else None,
               "worker_resource_scope": entry.get("worker_resource_scope") if executed else None,
               "allocation_peak_scope": "cumulative worker high-water mark as of each phase; not an exclusive phase peak"}
        rows.append(row)
    audit = report["final_evidence_audit"]
    base.require(type(audit["passed"]) is bool, "final evidence flag malformed")
    if audit["passed"]:
        same(audit["source_sha256"], declaration["source_sha256"], "final source audit differs")
        same(audit["inputs_sha256"], inputs, "final input audit registry differs")
        same(audit["worker_reports"], bindings, "final worker audit registry differs")
    else:
        base.require(isinstance(audit.get("reason"), str) and audit["reason"], "failed final audit reason missing")
    same(report["historical_preservation"]["verified_in_final_audit"], audit["passed"], "parent historical-audit flag differs")
    same(report["historical_preservation"]["all_bytes_unchanged"], True if audit["passed"] else None, "unverified parent historical bytes claimed unchanged")
    base.number(report["workflow_wall_seconds"], "shared workflow wall", 0)
    base.close_double(report["wall_overshoot_seconds"], max(0., report["workflow_wall_seconds"] - WALL_SECONDS), "shared wall overshoot")
    for row in rows:
        if row["shared_workflow_elapsed_seconds"] is not None:
            base.require(row["shared_workflow_elapsed_seconds"] <= report["workflow_wall_seconds"], "worker elapsed time exceeds parent workflow")
    can_complete = all(row["execution_status"] == "completed" for row in rows) and audit["passed"] and report["workflow_wall_seconds"] < WALL_SECONDS
    same(report["execution_status"], "completed" if can_complete else "incomplete", "parent completion gate differs")
    expected_status = "incomplete" if not can_complete else "completed" if all(row["status"] == "completed" for row in rows) else "completed_with_identity_failures"
    same(report["status"], expected_status, "parent exact conclusion differs")
    all_comparisons = [value for row in rows for value in row["comparisons"].values() if value is not None]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"], "execution_status": report["execution_status"],
            "certified": False, "research_pilot_executed": False, "backend_parity_certified": False,
            "workflow_wall_seconds": report["workflow_wall_seconds"], "wall_overshoot_seconds": report["wall_overshoot_seconds"],
            "common_workflow_started_monotonic": report["common_workflow_started_monotonic"],
            "final_evidence_audit_passed": audit["passed"], "final_evidence_audit": audit,
            "declaration": {"path": declaration_path.relative_to(root).as_posix(), "sha256": declaration_hash, "canonical_sha256": base.canonical_sha256(declaration)},
            "raw_report": {"path": declaration["output"], "sha256": raw_hash}, "process_environment": ENVIRONMENT,
            "source_sha256": declaration["source_sha256"], "model_config": declaration["model_config"], "data": declaration["data"], "prior_smoke": declaration["prior_smoke"],
            "policies": rows, "coverage": {"expected_policies": 2, "completed_policies": sum(row["execution_status"] == "completed" for row in rows),
                "expected_comparisons": 4, "recorded_comparisons": len(all_comparisons), "passed_comparisons": sum(value["passed"] for value in all_comparisons),
                "expected_categories": 32, "recorded_categories": sum(value["category_counts"]["total"] for value in all_comparisons),
                "passed_categories": sum(value["category_counts"]["passed"] for value in all_comparisons)},
            "limits": [*report["limits"],
                       "Repeat-control and restart comparisons are separate gates; repeat drift limits attribution of restart drift.",
                       "Missing comparisons are null and contribute no passing categories. Partial worker findings do not complete the study.",
                       "Exact flags/counts and public identities are validated; checkpoint tensor equality and magnitudes cannot be recomputed without private tensors.",
                       "Data/model seeds and verified artifact registries describe the declared sampling recipe; actual initial weights and batch token hashes were not separately retained.",
                       "Magnitude statistics inspect unequal values; a zero nonfinite count is not a certificate that every checkpoint value is finite."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        base.write_summary(args.output, build_summary(args.declaration, args.raw, args.root, args.source_root))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Validated recovery determinism summary written; certified=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
