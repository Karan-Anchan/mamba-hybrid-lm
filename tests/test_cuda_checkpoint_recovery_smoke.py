"""CPU-only orchestration, strict declaration and diagnostic-wrapper contracts."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import time

import pytest
import torch

import scripts.check_cuda_checkpoint_recovery as smoke
import src.train.train as trainer
from test_training_reliability import _tiny_train_config


@pytest.fixture
def cpu_workflow(tmp_path):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=4, warmup_steps=1, eval_interval=2,
                  checkpoint_interval=1, precision="float32", wall_time_limit_seconds=300,
                  model_seed=1337, data_seed=1337, eval_seed=1337)
    return cfg


def test_real_cpu_control_stop_resume_preserves_exact_trajectory_and_restores_wrapper(cpu_workflow):
    original = trainer._publish_checkpoint_transaction
    guards = []
    result = smoke.execute_workflow(cpu_workflow, guard=lambda: guards.append(True))
    assert len(guards) == 3
    assert trainer._publish_checkpoint_transaction is original
    assert [phase["completed_steps"] for phase in result["phases"]] == [4, 2, 4]
    assert [phase["status"] for phase in result["phases"]] == ["completed", "stopped", "completed"]
    stopped = result["phases"][1]
    assert stopped["stop"]["reason"] == smoke.STOP_REASON
    assert stopped["wrapper"]["successful_steps"] == [0, 1, 2]
    assert stopped["wrapper"]["stop_requests"] == 1 and stopped["wrapper"]["restored"]
    assert result["recovery_identity"]["passed"]
    assert all(check["exact_equal"] for check in result["recovery_identity"]["checks"].values())
    assert "step_seconds" in result["recovery_identity"]["timing_excluded"]
    for directory in result["directories"].values():
        assert trainer.read_json(directory / "manifest.json")["status"] == "completed"


def test_wrapper_never_requests_stop_when_publisher_fails_and_always_restores(monkeypatch):
    control = trainer.TrainingStopControl(300)

    def failing_publisher(*_args, **_kwargs):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(trainer, "_publish_checkpoint_transaction", failing_publisher)
    with pytest.raises(RuntimeError, match="publication failed"):
        with smoke.request_stop_after_publication(control, "selected") as observation:
            trainer._publish_checkpoint_transaction(None, trainer.TrainConfig(run_id="selected"), None,
                                                  None, {"completed_steps": 2}, None, None, 1)
    assert not control.requested.is_set()
    assert observation["stop_requests"] == 0 and observation["restored"]
    assert trainer._publish_checkpoint_transaction is failing_publisher


def test_wrapper_is_exactly_scoped_to_selected_run_and_step(monkeypatch):
    control = trainer.TrainingStopControl(300)
    monkeypatch.setattr(trainer, "_publish_checkpoint_transaction", lambda *_args: None)
    with smoke.request_stop_after_publication(control, "selected") as observation:
        for run_id, step in (("other", 2), ("selected", 1)):
            trainer._publish_checkpoint_transaction(None, trainer.TrainConfig(run_id=run_id), None,
                                                  None, {"completed_steps": step}, None, None, 0)
        assert not control.requested.is_set()
        trainer._publish_checkpoint_transaction(None, trainer.TrainConfig(run_id="selected"), None,
                                              None, {"completed_steps": 2}, None, None, 1)
    assert control.reason() == smoke.STOP_REASON and observation["successful_steps"] == [1, 2]


def test_shared_deadline_cannot_reset_between_operational_calls(cpu_workflow, monkeypatch):
    def budget_stop(cfg, *, stop_control):
        assert stop_control.started == common_start
        assert stop_control.reason() == "wall_time_limit"
        return {"status": "stopped", "completed_steps": 0, "tokens_seen": 0,
                "stop": stop_control.snapshot()}

    common_start = time.monotonic() - 301
    monkeypatch.setattr(trainer, "run", budget_stop)
    progress = {}
    with pytest.raises(RuntimeError, match="wall allowance"):
        smoke.execute_workflow(cpu_workflow, started=common_start, progress=progress)
    assert progress["phases"][0]["status"] == "stopped"


def test_incomplete_phase_resources_survive_exception(cpu_workflow, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(trainer, "run", fail)
    progress = {}
    with pytest.raises(RuntimeError, match="worker failure"):
        smoke.execute_workflow(cpu_workflow, progress=progress)
    phase = progress["phases"][0]
    assert phase["status"] == "incomplete" and phase["wall_seconds"] >= 0
    assert "io" in phase


@pytest.mark.parametrize("group", ["model", "optimizer", "rng", "tokens_seen", "metrics"])
def test_exact_comparison_detects_weight_optimizer_rng_counter_or_metric_change(tmp_path, group):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    state = {"model": {"weight": torch.tensor([1.0])}, "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.3])}}},
             "rng": {"train_generator": torch.tensor([1, 2], dtype=torch.uint8)},
             "completed_steps": 4, "tokens_seen": 2048, "best_val_loss": 4.0}
    changed = copy.deepcopy(state)
    if group == "model":
        changed[group]["weight"][0] += 0.1
    elif group == "optimizer":
        changed[group]["state"][0]["exp_avg"][0] += 0.1
    elif group == "rng":
        changed[group]["train_generator"][0] += 1
    elif group == "tokens_seen":
        changed[group] += 1
    torch.save(state, first / "last.pt")
    torch.save(changed, second / "last.pt")
    for path, loss in ((first, 4.0), (second, 4.1 if group == "metrics" else 4.0)):
        (path / "metrics.jsonl").write_text(json.dumps({"event": "train", "step": 1, "loss": loss,
                                                       "step_seconds": 8 if path == first else 9}) + "\n")
    result = smoke.compare_trajectories(first, second)
    assert not result["passed"]
    key = "semantic_metrics" if group == "metrics" else group
    assert not result["checks"][key]["exact_equal"]


@pytest.fixture
def declared_fixture(tmp_path, monkeypatch):
    root = tmp_path
    model = root / "configs/ratio_1_15.yaml"
    model.parent.mkdir()
    model.write_bytes((smoke.ROOT / "configs/ratio_1_15.yaml").read_bytes())
    data = root / "data/openwebtext-5b"
    data.mkdir(parents=True)
    records = {}
    for name, filename in (("train", "train.bin"), ("val", "val.bin"), ("meta", "meta.json")):
        path = data / filename
        path.write_bytes(name.encode())
        records[name] = {"file": filename, "sha256": smoke.file_sha256(path), "bytes": path.stat().st_size,
                         **({"tokens": 1024} if name != "meta" else {})}
    manifest = {"signature": "a" * 64, "tokenizer": {"vocab_size": 16000}, "outputs": records}
    (data / "manifest.json").write_text(json.dumps(manifest))
    histories = []
    for ratio in ("1:3", "1:7", "1:15"):
        relative = f"checkpoints/week3-700m-v1/hybrid-{ratio.replace(':', '-')}/best.pt"
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(f"unchanged historical {ratio}".encode())
        histories.append({"ratio": ratio, "path": relative, "sha256": smoke.file_sha256(path)})
    sources = {"src/train/train.py": "b" * 64}
    monkeypatch.setattr(smoke, "source_registry", lambda _root=smoke.ROOT: dict(sources))
    monkeypatch.setattr(smoke, "validate_prepared_dataset", lambda _path: manifest)
    declaration = {"schema": 1, "version": 1, "kind": smoke.KIND, "date": "2026-10-04",
                   "status_at_declaration": "planned_before_execution", "train_config": dict(smoke.TRAIN_SETTINGS),
                   "guards": {"wall_seconds": 300, "min_free_cuda_bytes": 8 * 1024**3},
                   "stop": {"step": 2, "reason": smoke.STOP_REASON, "wrapper": smoke.WRAPPER_ID},
                   "checkpoint_root": smoke.CHECKPOINT_ROOT, "control_run_id": smoke.CONTROL_ID,
                   "interrupted_run_id": smoke.RESUME_ID, "model_config": {"path": model.relative_to(root).as_posix(),
                                                                        "sha256": smoke.file_sha256(model)},
                   "data": {"directory": "data/openwebtext-5b", "signature": manifest["signature"],
                            "manifest_sha256": smoke.file_sha256(data / "manifest.json")},
                   "source_sha256": sources, "historical_checkpoints": histories,
                   "output": "docs/research/checks/cuda-recovery/report.json"}
    declaration_path = root / "declaration.json"
    declaration_path.write_text(json.dumps(declaration))
    return root, declaration_path, declaration, root / declaration["output"]


@pytest.mark.parametrize("mutation", ["precision", "bool_interval", "source", "config_hash", "path", "output", "namespace"])
def test_malformed_declaration_or_existing_paths_fail_before_cuda_or_training(declared_fixture, monkeypatch, mutation):
    root, path, declaration, output = declared_fixture
    if mutation == "precision":
        declaration["train_config"]["precision"] = "bfloat16"
    elif mutation == "bool_interval":
        declaration["train_config"]["checkpoint_interval"] = True
    elif mutation == "source":
        declaration["source_sha256"] = {}
    elif mutation == "config_hash":
        declaration["model_config"]["sha256"] = "0" * 64
    elif mutation == "path":
        declaration["model_config"]["path"] = "../outside.yaml"
    elif mutation == "output":
        output.parent.mkdir(parents=True)
        output.write_bytes(b"retain previous report")
    else:
        (root / smoke.CHECKPOINT_ROOT / smoke.CONTROL_ID).mkdir(parents=True)
    path.write_text(json.dumps(declaration))
    monkeypatch.setattr(smoke, "cuda_guard", lambda *_args: pytest.fail("invalid declaration cannot reach CUDA"))
    monkeypatch.setattr(smoke, "execute_workflow", lambda *_args, **_kwargs: pytest.fail("invalid declaration cannot train"))
    with pytest.raises(RuntimeError):
        smoke.run_declared(path, output, root=root)
    if mutation == "output":
        assert output.read_bytes() == b"retain previous report"


def test_unavailable_cuda_records_uncertified_incomplete_without_training(declared_fixture, monkeypatch):
    root, path, declaration, output = declared_fixture
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(smoke, "execute_workflow", lambda *_args, **_kwargs: pytest.fail("CUDA failure cannot train"))
    result = smoke.run_declared(path, output, root=root)
    assert result["status"] == result["execution_status"] == "incomplete"
    assert not result["certified"] and not result["research_pilot_executed"]
    assert result["historical_bytes_unchanged"]
    assert not (root / smoke.CHECKPOINT_ROOT).exists()
    assert not output.exists()


def test_cuda_headroom_guard_rejects_before_training(monkeypatch):
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(smoke.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(smoke.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(smoke.torch.cuda, "mem_get_info", lambda: (7 * 1024**3, 12 * 1024**3))
    with pytest.raises(RuntimeError, match="headroom"):
        smoke.cuda_guard(time.monotonic())


def test_evidence_change_is_retained_as_incomplete_without_false_certificate(declared_fixture, monkeypatch):
    root, path, declaration, output = declared_fixture
    monkeypatch.setattr(smoke, "cuda_guard", lambda _started: {"free_bytes": 9 * 1024**3})
    monkeypatch.setattr(smoke, "runtime_metadata", lambda: {"scope": "CPU mocked control contract"})

    def mutate_evidence(_cfg, **kwargs):
        history = root / declaration["historical_checkpoints"][0]["path"]
        history.write_bytes(b"failure injected into temporary CPU fixture")
        kwargs["evidence_check"]()

    monkeypatch.setattr(smoke, "execute_workflow", mutate_evidence)
    result = smoke.run_declared(path, output, root=root)
    assert result["status"] == "incomplete" and not result["historical_bytes_unchanged"]
    assert not result["certified"] and "evidence bytes changed" in result["reason"]
    assert str(root) not in json.dumps(result)


def test_missing_historical_evidence_retains_original_failure_and_public_report(declared_fixture, monkeypatch):
    root, path, declaration, output = declared_fixture
    monkeypatch.setattr(smoke, "cuda_guard", lambda _started: {"free_bytes": 9 * 1024**3})
    monkeypatch.setattr(smoke, "runtime_metadata", lambda: {"scope": "CPU mocked control contract"})

    def remove_evidence(_cfg, **kwargs):
        (root / declaration["historical_checkpoints"][0]["path"]).unlink()
        kwargs["evidence_check"]()

    monkeypatch.setattr(smoke, "execute_workflow", remove_evidence)
    result = smoke.run_declared(path, output, root=root)
    assert result["status"] == "incomplete" and not result["historical_bytes_unchanged"]
    assert "FileNotFoundError" in result["reason"]
    check = result["historical_preservation"]["checks"][0]
    assert not check["bytes_unchanged"] and check["actual_sha256"] is None
    assert "FileNotFoundError" in check["reason"]
    assert str(root) not in json.dumps(result)
    smoke.write_report(output, result)
    assert json.loads(output.read_text())["status"] == "incomplete"


def test_public_protocol_redacts_private_paths_without_rewriting_original_declaration(declared_fixture, monkeypatch):
    root, path, declaration, output = declared_fixture
    declaration["notes"] = {"local_file": str(root / "local-notes.txt"), "home_file": str(Path.home() / "private.txt")}
    path.write_text(json.dumps(declaration))
    original = path.read_bytes()
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: False)
    result = smoke.run_declared(path, output, root=root)
    assert result["protocol_display_sanitized"]
    assert result["declaration_canonical_sha256"] == smoke.canonical_hash(declaration)
    assert result["declaration_sha256"] == smoke.file_sha256(path)
    assert str(root) not in json.dumps(result) and str(Path.home()) not in json.dumps(result)
    assert path.read_bytes() == original


def test_report_publication_is_exclusive_and_never_overwrites(tmp_path):
    output = tmp_path / "report.json"
    smoke.write_report(output, {"status": "incomplete", "certified": False})
    original = output.read_bytes()
    with pytest.raises(RuntimeError, match="already exists"):
        smoke.write_report(output, {"status": "completed"})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
