"""Public-export contracts exercised with synthetic JSON, without CUDA or tensors."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import summarize_cuda_recovery_determinism as export

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return export.base.file_sha256(path)


def magnitude(failed=False):
    return {"scope": "descriptive numeric differences; exact equality alone decides the check",
            "different_numeric_elements": int(failed), "nonfinite_pairs": 0,
            "max_absolute_difference": .0001 if failed else 0.,
            "max_symmetric_relative_difference": .0002 if failed else 0.,
            "relative_denominator": "max(abs(left),abs(right),1e-12)",
            "checkpoint_layout_differences": 0, "layout_examples": []}


def comparison(failed=()):
    return {"checks": {name: {"exact_equal": name not in failed, "tensor_checks": 1,
                              "difference_count": int(name in failed),
                              "first_differences": ["value.weight"] if name in failed else [],
                              "descriptive_magnitudes": magnitude(name in failed)}
                       for name in sorted(export.CATEGORIES)},
            "passed": not failed, "tolerances_relaxed": False,
            "scope": "exact same-policy trajectory identity; descriptive magnitudes do not convert failures to passes",
            "timing_excluded": ["train_seconds", "eval_seconds", "peak_vram_mb", "step_seconds", "tok_per_s"]}


def unavailable_io():
    return {"available": False, "source": None, "counters": None}


def worker(policy, declaration, declaration_hash):
    before = {key: False for key in export.FLAG_KEYS}
    before["fill_uninitialized_memory"] = True
    active = {**before, **{key: policy[key] for key in
                          ("deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark")}}
    phases = []
    for name, arm in export.PHASES:
        stop = name == "requested_stop"
        phase = {"phase": name, "arm": arm, "wall_seconds": 1., "io": unavailable_io(),
                 "headroom_before": {"free_bytes": export.MIN_FREE, "total_bytes": 12 * 1024**3},
                 "peak_allocated_bytes": 1024**2, "status": "stopped" if stop else "completed",
                 "completed_steps": 2 if stop else 4, "tokens_seen": (2 if stop else 4) * 512,
                 "observed_paths": {"training": "torch.quadratic_ssd", "prefill": None, "decode": None}}
        if stop:
            phase.update({"wrapper": {"wrapper_id": export.WRAPPER_ID, "stop_reason": export.STOP_REASON,
                                      "original_binding": "src.train.train._publish_checkpoint_transaction",
                                      "restored": True, "successful_steps": [0, 1, 2], "stop_requests": 1},
                          "stop": {"reason": export.STOP_REASON}, "last_sha256_at_stop": "1" * 64})
        phases.append(phase)
    fingerprint = export.base.canonical_sha256({name: digest for name, digest in declaration["source_sha256"].items()
                                               if name.startswith("src/") or name in {"scripts/run_sweep.py", "requirements.txt"}})
    preservation = {"checks": [{"path": row["path"], "expected_sha256": row["sha256"],
                                "actual_sha256": row["sha256"], "bytes_unchanged": True}
                               for row in declaration["historical_checkpoints"]], "all_bytes_unchanged": True}
    return {"schema": 1, "kind": export.KIND, "policy": policy["name"], "certified": False,
            "declaration_sha256": declaration_hash, "source_sha256": declaration["source_sha256"],
            "common_workflow_started_monotonic": 1000., "process_environment": export.ENVIRONMENT,
            "research_pilot_executed": False, "backend_parity_certified": False,
            "comparison_endpoint": "exact equality in all eight categories; no tolerance acceptance",
            "torch_imported_at_worker_entry": False, "execution_status": "completed", "status": "completed",
            "comparisons": {name: comparison() for name in export.COMPARISONS}, "phases": phases,
            "policy_flags": {"requested": policy, "before": before, "active": active, "after": before, "restored": True},
            "runtime_ambient_before_policy": {"gpu": "synthetic CUDA identity for unit test", "cuda_runtime": "12.fixture",
                "capability": [12, 0], "python": "3.fixture", "platform": "synthetic",
                "packages": {"torch": "fixture", "numpy": "fixture", "tokenizers": "fixture",
                             "mamba-ssm": None, "triton": None, "causal-conv1d": None},
                "training_policy": deepcopy(export.PRECISION["policy"]), "deterministic_algorithms": False,
                "ambient_flags_outside_training": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}},
            "training_provenance": {arm: {"signature": "2" * 64, "source_fingerprint": fingerprint,
                                          "git": {"fixture": True}, "training_precision": deepcopy(export.PRECISION)}
                                    for arm in export.RUN_IDS[policy["name"]]},
            "artifacts": {arm: {label: {"path": f"{export.CHECKPOINT_ROOT}/{run_id}/synthetic-model/{filename}", "sha256": "3" * 64}
                                for label, filename in export.ARTIFACTS.items()}
                          for arm, run_id in export.RUN_IDS[policy["name"]].items()},
            "historical_preservation": preservation, "historical_bytes_unchanged": True,
            "worker_wall_seconds": 5., "shared_workflow_elapsed_seconds": 10. if policy["name"] == "legacy" else 20.,
            "wall_overshoot_seconds": 0., "worker_io": unavailable_io(),
            "worker_resource_scope": "worker after declaration/source/history preflight: corpus validation, CUDA checks, training, comparisons and final evidence; excludes import/startup and report publication",
            "cuda_peak_scope": "cumulative worker CUDA allocation high-water mark as of each phase, not an exclusive phase peak"}


@pytest.fixture
def evidence(tmp_path):
    root, source_root = tmp_path / "public", tmp_path / "source"
    declaration = json.loads((ROOT / "docs/research/cuda-recovery-determinism-protocol-2026-10-04.json").read_text())
    for name in export.SOURCES | {declaration["model_config"]["path"]}:
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    declaration["source_sha256"] = {name: export.base.file_sha256(source_root / name) for name in sorted(export.SOURCES)}
    manifest = {"signature": declaration["data"]["signature"],
                "outputs": {name: {"file": filename, "sha256": "4" * 64}
                            for name, filename in (("train", "train.bin"), ("val", "val.bin"), ("meta", "meta.json"))}}
    declaration["data"]["manifest_sha256"] = write(source_root / declaration["data"]["directory"] / "manifest.json", manifest)
    prior = {"kind": "cuda_checkpoint_recovery_operational_smoke", "certified": False,
             "protocol": {name: deepcopy(declaration[name]) for name in ("model_config", "data", "historical_checkpoints")}}
    declaration["prior_smoke"]["sha256"] = write(root / declaration["prior_smoke"]["path"], prior)
    declaration_path = root / "docs/research/cuda-recovery-determinism-protocol-2026-10-04.json"
    declaration_hash = write(declaration_path, declaration)
    inputs = {declaration[name]["path"]: declaration[name]["sha256"] for name in ("model_config", "prior_smoke")}
    inputs[declaration["data"]["directory"] + "/manifest.json"] = declaration["data"]["manifest_sha256"]
    inputs.update({declaration["data"]["directory"] + "/" + row["file"]: row["sha256"] for row in manifest["outputs"].values()})
    inputs.update({row["path"]: row["sha256"] for row in declaration["historical_checkpoints"]})
    report = {"schema": 1, "kind": export.KIND, "date": declaration["date"], "certified": False,
              "declaration_sha256": declaration_hash, "declaration_canonical_sha256": export.base.canonical_sha256(declaration),
              "source_sha256": deepcopy(declaration["source_sha256"]), "process_environment": deepcopy(export.ENVIRONMENT),
              "research_pilot_executed": False, "backend_parity_certified": False,
              "common_workflow_started_monotonic": 1000., "policies": [], "workflow_wall_seconds": 25., "wall_overshoot_seconds": 0.,
              "execution_status": "completed", "status": "completed", "limits": ["Synthetic unit-test evidence only."],
              "final_evidence_audit": {"passed": True, "source_sha256": deepcopy(declaration["source_sha256"]),
                                       "inputs_sha256": inputs, "worker_reports": {}},
              "historical_preservation": {"verified_in_final_audit": True, "all_bytes_unchanged": True}}
    raw_path = root / declaration["output"]
    for policy in export.POLICIES:
        raw_worker = worker(deepcopy(policy), declaration, declaration_hash)
        worker_path = raw_path.with_name(policy["name"] + "-worker.json")
        binding = {"path": worker_path.relative_to(root).as_posix(), "sha256": write(worker_path, raw_worker)}
        report["policies"].append({**raw_worker, "worker_report": binding, "worker_exit_code": 0})
        report["final_evidence_audit"]["worker_reports"][policy["name"]] = binding
    write(raw_path, report)
    return root, source_root, declaration_path, raw_path


def build(evidence):
    root, source_root, declaration, raw = evidence
    return export.build_summary(declaration, raw, root, source_root)


def alter_parent(evidence, change):
    raw = evidence[3]
    report = json.loads(raw.read_text())
    change(report)
    write(raw, report)


def alter_worker(evidence, change, index=0):
    root, _, _, raw = evidence
    report = json.loads(raw.read_text())
    entry = report["policies"][index]
    worker_path = root / entry["worker_report"]["path"]
    value = json.loads(worker_path.read_text())
    change(value)
    binding = {"path": entry["worker_report"]["path"], "sha256": write(worker_path, value)}
    report["policies"][index] = {**value, "worker_report": binding,
                                "worker_exit_code": 0 if value["status"] == "completed" else 2}
    report["final_evidence_audit"]["worker_reports"][entry["policy"]] = binding
    report["execution_status"] = "completed" if all(row["execution_status"] == "completed" for row in report["policies"]) else "incomplete"
    report["status"] = "incomplete" if report["execution_status"] != "completed" else "completed" if all(row["status"] == "completed" for row in report["policies"]) else "completed_with_identity_failures"
    write(raw, report)


def test_complete_summary_has_separate_32_exact_checks_and_no_private_binary_loading(evidence):
    result = build(evidence)
    assert result["coverage"] == {"expected_policies": 2, "completed_policies": 2, "expected_comparisons": 4,
                                   "recorded_comparisons": 4, "passed_comparisons": 4,
                                   "expected_categories": 32, "recorded_categories": 32, "passed_categories": 32}
    assert result["certified"] is False and result["research_pilot_executed"] is False
    assert result["backend_parity_certified"] is False
    assert result["policies"][0]["phases"][2]["wrapper"]["successful_steps"] == [0, 1, 2]
    assert result["policies"][0]["phases"][0]["peak_allocated_mib"] == 1
    assert not list(evidence[0].rglob("*.pt")) and not list(evidence[1].rglob("*.bin"))
    assert str(evidence[0]) not in json.dumps(result)


def test_repeat_drift_is_retained_separately_from_restart_drift(evidence):
    def fail(value):
        value["comparisons"]["control_a_vs_control_b"] = comparison(("model", "optimizer"))
        value["comparisons"]["control_a_vs_resumed"] = comparison(("model",))
        value["status"] = "completed_with_identity_failures"
    alter_worker(evidence, fail)
    result = build(evidence)
    assert result["execution_status"] == "completed"
    assert result["status"] == "completed_with_identity_failures"
    assert result["coverage"]["passed_categories"] == 29
    assert result["policies"][0]["comparisons"]["control_a_vs_control_b"]["failed_categories"] == ["model", "optimizer"]
    assert result["policies"][0]["comparisons"]["control_a_vs_resumed"]["checks"]["model"]["descriptive_magnitudes"]["max_absolute_difference"] == .0001


def test_partial_worker_preserves_completed_controls_without_inventing_restart(evidence):
    def partial(value):
        value["execution_status"] = value["status"] = "incomplete"
        value["reason"] = "Synthetic unsupported operation after controls"
        value["phases"] = value["phases"][:2]
        value["comparisons"].pop("control_a_vs_resumed")
        value["artifacts"].pop("resumed")
        value["training_provenance"].pop("resumed")
    alter_worker(evidence, partial)
    result = build(evidence)
    assert result["execution_status"] == "incomplete" and result["coverage"]["recorded_categories"] == 24
    assert result["policies"][0]["comparisons"]["control_a_vs_resumed"] is None
    assert result["policies"][0]["comparisons"]["control_a_vs_control_b"]["category_counts"] == {"passed": 8, "total": 8}


def test_not_started_policy_records_no_comparisons(evidence):
    def stop(report):
        report["policies"][1] = {"policy": "strict", "execution_status": "not_started", "status": "incomplete",
                                 "reason": "Synthetic common allowance exhausted"}
        report["final_evidence_audit"]["worker_reports"].pop("strict")
        report["workflow_wall_seconds"] = 901.
        report["wall_overshoot_seconds"] = 1.
        report["execution_status"] = report["status"] = "incomplete"
    alter_parent(evidence, stop)
    result = build(evidence)
    assert result["coverage"]["recorded_categories"] == 16
    assert result["policies"][1]["comparisons"] == dict.fromkeys(export.COMPARISONS)


@pytest.mark.parametrize("change", [
    lambda value: value["checks"].pop("rng"),
    lambda value: value["checks"]["model"].__setitem__("exact_equal", 1),
    lambda value: value["checks"]["model"].__setitem__("difference_count", True),
    lambda value: value["checks"]["model"].__setitem__("difference_count", 1),
    lambda value: value["checks"]["model"]["descriptive_magnitudes"].__setitem__("different_numeric_elements", 1),
    lambda value: value["checks"]["model"]["descriptive_magnitudes"].__setitem__("max_absolute_difference", .1),
    lambda value: value.__setitem__("passed", False),
    lambda value: value.__setitem__("tolerances_relaxed", True),
    lambda value: value["timing_excluded"].pop(),
    lambda value: value.__setitem__("quality_equivalent", True),
    lambda value: value["checks"]["model"].__setitem__("certified", True),
])
def test_exact_metrics_rejected_after_worker_hash_is_resigned(evidence, change):
    alter_worker(evidence, lambda value: change(value["comparisons"][export.COMPARISONS[0]]))
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda value: value.__setitem__("torch_imported_at_worker_entry", True),
    lambda value: value.__setitem__("comparison_endpoint", "close enough"),
    lambda value: value.__setitem__("common_workflow_started_monotonic", 1001.),
    lambda value: value["process_environment"].__setitem__("CUBLAS_WORKSPACE_CONFIG", ":16:8"),
    lambda value: value["policy_flags"]["active"].__setitem__("warn_only", True),
    lambda value: value["policy_flags"].__setitem__("restored", False),
    lambda value: value["runtime_ambient_before_policy"]["packages"].pop("torch"),
    lambda value: value["runtime_ambient_before_policy"].__setitem__("deterministic_algorithms", True),
    lambda value: value["training_provenance"]["resumed"]["training_precision"]["runtime_flags"].__setitem__("cuda_matmul_allow_tf32", True),
    lambda value: value["training_provenance"]["resumed"].__setitem__("source_fingerprint", "0" * 64),
    lambda value: value["artifacts"]["control_a"].pop("last"),
    lambda value: value["artifacts"]["control_a"]["last"].__setitem__("path", "../last.pt"),
    lambda value: value["phases"][0].__setitem__("completed_steps", 3),
    lambda value: value["phases"][0]["headroom_before"].__setitem__("free_bytes", 1),
    lambda value: value["phases"][2]["wrapper"].__setitem__("restored", False),
    lambda value: value["phases"][2]["wrapper"].__setitem__("successful_steps", [0, 2, 2]),
    lambda value: value["phases"][2]["wrapper"].__setitem__("stop_requests", 0),
    lambda value: value.__setitem__("shared_workflow_elapsed_seconds", 900.),
    lambda value: value["historical_preservation"]["checks"][0].__setitem__("actual_sha256", "0" * 64),
    lambda value: value.pop("historical_preservation"),
])
def test_worker_policy_inputs_phases_and_provenance_rejected_after_resigning(evidence, change):
    alter_worker(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


def test_partial_report_cannot_attach_finished_comparison_to_unfinished_phase(evidence):
    def partial(value):
        value["execution_status"] = value["status"] = "incomplete"
        value["reason"] = "Synthetic failure"
        value["phases"] = value["phases"][:2]
    alter_worker(evidence, partial)
    with pytest.raises(ValueError, match="required training phases"):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda value: value["policies"].pop(),
    lambda value: value["policies"].reverse(),
    lambda value: value["policies"][0].__setitem__("worker_exit_code", 2),
    lambda value: value["policies"][0]["worker_report"].__setitem__("sha256", "0" * 64),
    lambda value: value["policies"][0]["worker_report"].__setitem__("path", "../legacy-worker.json"),
    lambda value: value["policies"][0].__setitem__("certified", True),
    lambda value: value["final_evidence_audit"]["inputs_sha256"].pop("data/openwebtext-5b/train.bin"),
    lambda value: value["final_evidence_audit"]["source_sha256"].pop("src/model/lm.py"),
    lambda value: value.__setitem__("status", "completed_with_identity_failures"),
    lambda value: value.__setitem__("research_pilot_executed", True),
    lambda value: value.__setitem__("declaration_sha256", "0" * 64),
])
def test_parent_exact_bindings_and_claims_rejected(evidence, change):
    alter_parent(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


def test_failed_final_audit_is_incomplete_not_historical_preservation_success(evidence):
    def failure(value):
        value["final_evidence_audit"] = {"passed": False, "reason": "Synthetic audit failure"}
        value["historical_preservation"] = {"verified_in_final_audit": False, "all_bytes_unchanged": None}
        value["execution_status"] = value["status"] = "incomplete"
    alter_parent(evidence, failure)
    result = build(evidence)
    assert not result["final_evidence_audit_passed"] and result["status"] == "incomplete"


def test_changed_source_rejected_but_line_endings_are_portable(evidence):
    source = evidence[1] / "scripts/study_cuda_recovery_determinism.py"
    original = source.read_bytes()
    source.write_bytes(original.replace(b"\r\n", b"\n"))
    assert build(evidence)["status"] == "completed"
    source.write_bytes(original + b"\n# synthetic source change\n")
    with pytest.raises(ValueError, match="stale measured source"):
        build(evidence)


def test_explicit_raw_path_must_match_predeclared_output(evidence):
    with pytest.raises(ValueError, match="raw path differs"):
        export.build_summary(evidence[2], evidence[3].with_name("alternate.json"), evidence[0], evidence[1])


def test_atomic_summary_rejects_overwrite(evidence, tmp_path):
    destination = tmp_path / "summary.json"
    export.base.write_summary(destination, build(evidence))
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        export.base.write_summary(destination, {"certified": True})
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(".summary.json.*"))


@pytest.mark.parametrize("text", ['{"schema":1,"schema":2}', '{"value":NaN}', '{"value":1e999}'])
def test_json_reader_fails_closed(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        export.base.read_json(path)


def test_import_is_standard_library_only():
    result = subprocess.run([sys.executable, "-c",
                             "import sys; from scripts import summarize_cuda_recovery_determinism; "
                             "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
