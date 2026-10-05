"""CPU-only deterministic policy, exact-control and parent/worker evidence contracts."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

import scripts.study_cuda_recovery_determinism as study
import scripts.check_cuda_checkpoint_recovery as smoke
import src.train.train as trainer
from test_training_reliability import _tiny_train_config
from test_cuda_checkpoint_recovery_smoke import declared_fixture as smoke_fixture


@pytest.fixture
def neutral_policy():
    before = study.policy_flags(torch)
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    yield
    torch.use_deterministic_algorithms(before["deterministic_algorithms"], warn_only=before["warn_only"])
    torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
    torch.utils.deterministic.fill_uninitialized_memory = before["fill_uninitialized_memory"]


@pytest.fixture
def tiny_configuration(tmp_path):
    return replace(_tiny_train_config(tmp_path), max_steps=4, warmup_steps=1, eval_interval=2,
                   checkpoint_interval=1, precision="float32", wall_time_limit_seconds=900,
                   seed=1337, model_seed=1337, data_seed=1337, eval_seed=1337)


@pytest.mark.parametrize("policy", study.POLICIES, ids=lambda row: row["name"])
def test_real_cpu_repeats_and_recovery_all_eight_categories_match_and_flags_restore(tiny_configuration, neutral_policy, policy):
    before = study.policy_flags(torch)
    original = trainer._publish_checkpoint_transaction
    progress = {}
    guards = []
    dirs = study.execute_policy(tiny_configuration, policy, guard=lambda: guards.append(True), progress=progress)
    assert len(guards) == 4 and len(dirs) == 3
    assert [row["completed_steps"] for row in progress["phases"]] == [4, 4, 2, 4]
    assert progress["phases"][2]["stop"]["reason"] == study.STOP_REASON
    assert progress["phases"][2]["wrapper"]["successful_steps"] == [0, 1, 2]
    assert trainer._publish_checkpoint_transaction is original
    assert study.policy_flags(torch) == before
    assert progress["policy_flags"]["restored"]
    assert progress["policy_flags"]["active"]["deterministic_algorithms"] is policy["deterministic_algorithms"]
    assert set(progress["comparisons"]) == {"control_a_vs_control_b", "control_a_vs_resumed"}
    for row in progress["comparisons"].values():
        assert row["passed"] and len(row["checks"]) == 8 and not row["tolerances_relaxed"]
        assert all(check["exact_equal"] for check in row["checks"].values())


def test_strict_scope_restores_unusual_prior_flags_on_exception(neutral_policy):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    torch.utils.deterministic.fill_uninitialized_memory = False
    before = study.policy_flags(torch)
    with pytest.raises(FloatingPointError, match="injected"):
        with study.deterministic_policy(study.POLICIES[1]) as observation:
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
            raise FloatingPointError("injected")
    assert observation["restored"] and study.policy_flags(torch) == before


def test_legacy_refuses_undeclared_ambient_flags_without_training(neutral_policy):
    torch.backends.cudnn.benchmark = True
    before = study.policy_flags(torch)
    with pytest.raises(RuntimeError, match="actual defaults"):
        with study.deterministic_policy(study.POLICIES[0]):
            pytest.fail("changed defaults cannot enter legacy policy")
    assert study.policy_flags(torch) == before


def test_completed_repeat_comparison_survives_later_nonfinite_failure(tiny_configuration, neutral_policy, monkeypatch):
    original = trainer.run
    calls = []

    def fail_later(cfg, **kwargs):
        calls.append(cfg.run_id)
        if len(calls) == 3:
            raise FloatingPointError("injected nonfinite loss")
        return original(cfg, **kwargs)

    monkeypatch.setattr(trainer, "run", fail_later)
    progress = {}
    before = study.policy_flags(torch)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        study.execute_policy(tiny_configuration, study.POLICIES[1], progress=progress)
    assert progress["comparisons"]["control_a_vs_control_b"]["passed"]
    assert "control_a_vs_resumed" not in progress["comparisons"]
    assert progress["phases"][-1]["status"] == "incomplete"
    assert progress["phases"][-1]["wrapper"]["restored"]
    assert progress["policy_flags"]["restored"] and study.policy_flags(torch) == before


def test_shared_policy_deadline_does_not_reset(tiny_configuration, neutral_policy, monkeypatch):
    started = time.monotonic() - 901

    def expired(cfg, *, stop_control):
        assert stop_control.started == started and stop_control.reason() == "wall_time_limit"
        return {"status": "stopped", "completed_steps": 0, "tokens_seen": 0, "stop": stop_control.snapshot()}

    monkeypatch.setattr(trainer, "run", expired)
    progress = {}
    with pytest.raises(RuntimeError, match="shared workflow allowance"):
        study.execute_policy(tiny_configuration, study.POLICIES[1], started=started, progress=progress)
    assert len(progress["phases"]) == 1 and progress["policy_flags"]["restored"]


def test_magnitudes_describe_values_and_layout_without_relaxing_equality():
    left = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "metric": 2.0}
    right = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t().contiguous().t(), "metric": 2.5}
    right["weight"][1, 1] = 4.25
    result = study.descriptive_magnitudes(left, right)
    assert result["different_numeric_elements"] == 2
    assert result["max_absolute_difference"] == 0.5
    assert result["max_symmetric_relative_difference"] == pytest.approx(0.2)
    assert result["checkpoint_layout_differences"] == 1
    assert not smoke.exact_comparison(left, right)["exact_equal"]
    nonfinite = study.descriptive_magnitudes(torch.tensor([float("nan")]), torch.tensor([1.0]))
    assert nonfinite["nonfinite_pairs"] == 1
    assert json.dumps(nonfinite, allow_nan=False)


def test_importing_parent_runner_does_not_import_torch():
    completed = subprocess.run([sys.executable, "-c", "import sys; import scripts.study_cuda_recovery_determinism; print('torch' in sys.modules)"],
                               cwd=study.ROOT, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


@pytest.fixture
def declared_fixture(smoke_fixture, monkeypatch):
    root, path, declaration, _output = smoke_fixture
    declaration.update({"kind": study.KIND, "train_config": dict(study.TRAIN_SETTINGS),
                        "guards": {"wall_seconds": 900, "min_free_cuda_bytes": 8 * 1024**3},
                        "process_environment": dict(study.ENVIRONMENT), "policies": copy.deepcopy(study.POLICIES),
                        "checkpoint_root": study.CHECKPOINT_ROOT, "run_ids": copy.deepcopy(study.RUN_IDS),
                        "output": "docs/research/checks/determinism/report.json"})
    declaration.pop("control_run_id")
    declaration.pop("interrupted_run_id")
    prior = root / "docs/research/checks/prior/report.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({"kind": smoke.KIND, "certified": False, "status": "completed_with_recovery_failures"}))
    declaration["prior_smoke"] = {"path": prior.relative_to(root).as_posix(), "sha256": study.file_sha256(prior)}
    path.write_text(json.dumps(declaration))
    monkeypatch.setattr(study, "source_registry", lambda _root=study.ROOT: dict(declaration["source_sha256"]))
    return root, path, declaration, root / declaration["output"]


def valid_worker(name, declaration, declaration_sha, *, failed=False):
    checks = {key: {"exact_equal": True} for key in
              ("model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss", "train_batch_generator", "semantic_metrics")}
    if failed:
        checks["model"]["exact_equal"] = False
    comparison = {"passed": not failed, "checks": checks, "tolerances_relaxed": False}
    policy = next(row for row in study.POLICIES if row["name"] == name)
    source = study.canonical_hash({path: digest for path, digest in declaration["source_sha256"].items()
                                   if path.startswith("src/") or path in {"scripts/run_sweep.py", "requirements.txt"}})
    precision = {"device_type": "cuda", "policy": {"precision": "float32", "autocast_enabled": False,
                                                   "autocast_dtype": None, "tf32_policy": "disabled"},
                 "runtime_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}}
    return {"schema": 1, "kind": study.KIND, "policy": name, "certified": False,
            "execution_status": "completed", "status": "completed_with_identity_failures" if failed else "completed",
            "declaration_sha256": declaration_sha, "source_sha256": declaration["source_sha256"],
            "process_environment": dict(study.ENVIRONMENT), "research_pilot_executed": False, "backend_parity_certified": False,
            "comparisons": {"control_a_vs_control_b": copy.deepcopy(comparison), "control_a_vs_resumed": copy.deepcopy(comparison)},
            "policy_flags": {"requested": policy, "active": dict(policy), "restored": True}, "historical_bytes_unchanged": True,
            "torch_imported_at_worker_entry": False,
            "training_provenance": {arm: {"source_fingerprint": source, "training_precision": copy.deepcopy(precision)}
                                     for arm in ("control_a", "control_b", "resumed")}}


@pytest.mark.parametrize("mutation", ["missing_check", "false_conclusion", "wrong_exit", "bad_flags", "false_certified", "wrong_environment", "wrong_tf32", "import_order"])
def test_parent_rejects_false_worker_success(declared_fixture, mutation):
    _root, path, declaration, _output = declared_fixture
    worker = valid_worker("strict", declaration, study.file_sha256(path))
    exit_code = 0
    if mutation == "missing_check":
        del worker["comparisons"]["control_a_vs_resumed"]["checks"]["rng"]
    elif mutation == "false_conclusion":
        worker["comparisons"]["control_a_vs_control_b"]["checks"]["model"]["exact_equal"] = False
    elif mutation == "wrong_exit":
        exit_code = 3
    elif mutation == "bad_flags":
        worker["policy_flags"]["active"]["deterministic_algorithms"] = False
    elif mutation == "false_certified":
        worker["certified"] = 0
    elif mutation == "wrong_environment":
        worker["process_environment"] = {}
    elif mutation == "wrong_tf32":
        worker["training_provenance"]["resumed"]["training_precision"]["runtime_flags"]["cudnn_allow_tf32"] = True
    else:
        worker["torch_imported_at_worker_entry"] = True
    with pytest.raises(RuntimeError):
        study.validate_worker_result(worker, "strict", declaration, study.file_sha256(path), exit_code)


def test_parent_sets_environment_before_each_fresh_worker_and_retains_exact_failures(declared_fixture):
    root, path, declaration, output = declared_fixture
    starts, names = [], []

    def launch(command, **kwargs):
        name = command[command.index("--worker") + 1]
        names.append(name)
        starts.append(command[command.index("--started") + 1])
        assert kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert kwargs["cwd"] == root
        worker = valid_worker(name, declaration, study.file_sha256(path), failed=name == "legacy")
        worker["common_workflow_started_monotonic"] = float(starts[-1])
        study.write_report(study.worker_output_paths(declaration, root)[name], worker)
        return subprocess.CompletedProcess(command, 2 if name == "legacy" else 0)

    result = study.run_declared(path, output, root=root, launch=launch)
    assert names == ["legacy", "strict"] and starts[0] == starts[1]
    assert result["execution_status"] == "completed" and result["status"] == "completed_with_identity_failures"
    assert result["final_evidence_audit"]["passed"]
    assert not result["certified"] and not result["research_pilot_executed"]


def test_parent_final_audit_detects_changed_prior_or_worker_report(declared_fixture):
    root, path, declaration, output = declared_fixture

    def launch(command, **_kwargs):
        name = command[command.index("--worker") + 1]
        worker = valid_worker(name, declaration, study.file_sha256(path))
        worker["common_workflow_started_monotonic"] = float(command[command.index("--started") + 1])
        study.write_report(study.worker_output_paths(declaration, root)[name], worker)
        if name == "strict":
            (root / declaration["prior_smoke"]["path"]).write_bytes(b"changed evidence in temporary CPU fixture")
        return subprocess.CompletedProcess(command, 0)

    result = study.run_declared(path, output, root=root, launch=launch)
    assert result["status"] == result["execution_status"] == "incomplete"
    assert not result["final_evidence_audit"]["passed"]


@pytest.mark.parametrize("mutation", ["bool_budget", "policy", "environment", "source", "namespace", "worker_output"])
def test_invalid_declarations_or_existing_namespaces_never_launch(declared_fixture, mutation):
    root, path, declaration, output = declared_fixture
    if mutation == "bool_budget":
        declaration["train_config"]["max_steps"] = True
    elif mutation == "policy":
        declaration["policies"][1]["warn_only"] = True
    elif mutation == "environment":
        declaration["process_environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    elif mutation == "source":
        declaration["source_sha256"] = {}
        # Freeze a distinct actual producer identity, rather than a self-referential fixture hash.
    elif mutation == "namespace":
        (root / study.CHECKPOINT_ROOT / study.RUN_IDS["strict"]["control_b"]).mkdir(parents=True)
    else:
        worker = study.worker_output_paths(declaration, root)["legacy"]
        worker.parent.mkdir(parents=True)
        worker.write_bytes(b"retain prior worker")
    path.write_text(json.dumps(declaration))
    if mutation == "source":
        # The fixture normally resolves to the declared registry; use an immutable actual registry here.
        original_registry = {"src/train/train.py": "b" * 64}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(study, "source_registry", lambda _root=study.ROOT: original_registry)
            with pytest.raises(RuntimeError):
                study.run_declared(path, output, root=root, launch=lambda *_a, **_k: pytest.fail("cannot launch"))
    else:
        with pytest.raises(RuntimeError):
            study.run_declared(path, output, root=root, launch=lambda *_a, **_k: pytest.fail("cannot launch"))


def test_worker_retains_nonfinite_failure_and_historical_preservation(declared_fixture, monkeypatch):
    root, path, _declaration, _output = declared_fixture
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(study, "cuda_guard", lambda _started: {"free_bytes": 9 * 1024**3})
    monkeypatch.setattr(smoke, "runtime_metadata", lambda: {"deterministic_algorithms": False})

    def failure(*_args, **kwargs):
        kwargs["progress"]["phases"] = [{"status": "incomplete", "phase": "control_a"}]
        raise FloatingPointError("injected nonfinite loss")

    monkeypatch.setattr(study, "execute_policy", failure)
    result = study.run_worker(path, policy_name="strict", started=time.monotonic(), root=root)
    assert result["execution_status"] == result["status"] == "incomplete"
    assert "FloatingPointError" in result["reason"] and result["historical_bytes_unchanged"]
    assert result["phases"][0]["status"] == "incomplete" and not result["certified"]


def test_worker_rejects_future_shared_origin_before_any_cuda_or_training(declared_fixture, monkeypatch):
    root, path, _declaration, _output = declared_fixture
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(study, "cuda_guard", lambda *_args: pytest.fail("future timer cannot reach CUDA"))
    with pytest.raises(RuntimeError, match="future timer reset"):
        study.run_worker(path, policy_name="strict", started=time.monotonic() + 900, root=root)


def test_exclusive_report_no_overwrite_and_no_temporary_leak(tmp_path):
    output = tmp_path / "report.json"
    study.write_report(output, {"status": "incomplete"})
    before = output.read_bytes()
    with pytest.raises(RuntimeError, match="no overwrite"):
        study.write_report(output, {"status": "completed"})
    assert output.read_bytes() == before and not list(tmp_path.glob("*.tmp"))
