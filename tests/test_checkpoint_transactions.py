"""Real CPU trajectories interrupted around each recoverable checkpoint publication."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

import scripts.run_sweep as sweep
import src.train.train as trainer
from test_training_reliability import _assert_nested_equal, _tiny_train_config


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _metrics(path):
    # Timing is invocation dependent; losses, update schedule and counts must be identical.
    return [{key: value for key, value in json.loads(line).items()
             if key not in {"step_seconds", "tok_per_s"}}
            for line in path.read_text(encoding="utf-8").splitlines()]


def _inject_crash(monkeypatch, point, step):
    armed = {"value": True}

    def crash():
        armed["value"] = False
        raise RuntimeError("injected publication interruption")

    if point == "evaluation":
        original = trainer.estimate_loss
        calls = {"value": 0}

        def evaluate(*args, **kwargs):
            current = calls["value"]
            calls["value"] += 1
            if armed["value"] and current == step:
                crash()
            return original(*args, **kwargs)

        monkeypatch.setattr(trainer, "estimate_loss", evaluate)
        return
    name, when = point.rsplit("_", 1)
    if name in {"stage_best", "stage_last"}:
        original = trainer.atomic_torch_save

        def save(value, path):
            relevant = (path.name == f"next-{name[6:]}.pt"
                        and value.get("completed_steps", value.get("step")) == step)
            if relevant and when == "before" and armed["value"]:
                crash()
            original(value, path)
            if relevant and when == "after" and armed["value"]:
                crash()

        monkeypatch.setattr(trainer, "atomic_torch_save", save)
    elif name == "stage_metrics":
        original = trainer.atomic_write_text

        def write(path, contents):
            relevant = path.name == "next-metrics.jsonl" and json.loads(contents.splitlines()[-1])["step"] == step
            if relevant and when == "before" and armed["value"]:
                crash()
            original(path, contents)
            if relevant and when == "after" and armed["value"]:
                crash()

        monkeypatch.setattr(trainer, "atomic_write_text", write)
    elif name in {"pending", "commit"}:
        original = trainer.atomic_write_json

        def journal(path, value):
            relevant = (path.name == trainer._CHECKPOINT_JOURNAL and value["next_step"] == step
                        and value["phase"] == ("pending" if name == "pending" else "committed"))
            if relevant and when == "before" and armed["value"]:
                crash()
            original(path, value)
            if relevant and when == "after" and armed["value"]:
                crash()

        monkeypatch.setattr(trainer, "atomic_write_json", journal)
    elif name.startswith("publish_"):
        original = trainer._atomic_snapshot
        filename = trainer._TRANSACTION_FILES[name[8:]]

        def snapshot(source, destination, **kwargs):
            journal_path = destination.parent / trainer._CHECKPOINT_JOURNAL
            relevant = (destination.name == filename and journal_path.exists()
                        and trainer.read_json(journal_path)["next_step"] == step)
            if relevant and when == "before" and armed["value"]:
                crash()
            original(source, destination, **kwargs)
            if relevant and when == "after" and armed["value"]:
                crash()

        monkeypatch.setattr(trainer, "_atomic_snapshot", snapshot)
    elif name == "cleanup":
        original = trainer._remove_transaction

        def cleanup(run_dir, directory):
            relevant = trainer.read_json(run_dir / trainer._CHECKPOINT_JOURNAL)["next_step"] == step
            if relevant and when == "before" and armed["value"]:
                crash()
            original(run_dir, directory)
            if relevant and when == "after" and armed["value"]:
                crash()

        monkeypatch.setattr(trainer, "_remove_transaction", cleanup)
    else:
        raise AssertionError(point)


_POINTS = ["evaluation", *[f"{name}_{when}" for name in
    ("stage_best", "stage_last", "stage_metrics", "pending", "publish_best", "publish_last",
     "publish_metrics", "commit", "cleanup") for when in ("before", "after")]]


@pytest.mark.parametrize("step", [0, 1], ids=["initial-baseline", "new-best"])
@pytest.mark.parametrize("point", _POINTS)
def test_publication_interruption_resumes_exact_model_optimizer_rng_and_metrics(tmp_path, monkeypatch, point, step):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=3)
    baseline_cfg = replace(cfg, run_id="uninterrupted")
    trainer.run(baseline_cfg)
    baseline = trainer.variant_run_dir(baseline_cfg, "tiny-1:3")
    _inject_crash(monkeypatch, point, step)
    with pytest.raises(RuntimeError, match="injected publication"):
        trainer.run(cfg)
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    assert not (run_dir / "result.json").exists()
    assert trainer.read_json(run_dir / "manifest.json")["status"] != "completed"
    journal_path = run_dir / trainer._CHECKPOINT_JOURNAL
    if journal_path.exists():
        journal = trainer.read_json(journal_path)
        signature = trainer.read_json(run_dir / "manifest.json")["signature"]
        with trainer.RunLock(run_dir / ".run.lock"):
            assert trainer._recover_checkpoint_transaction(run_dir, cfg, trainer.load_model_config(cfg.model_config), signature)
        selected = journal["next_step"] if journal["phase"] == "committed" else journal["previous_step"]
        if selected is None:
            assert all(not (run_dir / name).exists() for name in trainer._TRANSACTION_FILES.values())
        else:
            restored = _load(run_dir / "last.pt")
            assert restored["completed_steps"] == selected
            assert _load(run_dir / "best.pt")["step"] <= selected
            trainer._validate_metrics(run_dir / "metrics.jsonl", cfg, completed_steps=selected)
    trainer.run(cfg)
    reference, actual = _load(baseline / "last.pt"), _load(run_dir / "last.pt")
    for key in ("model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss"):
        _assert_nested_equal(reference[key], actual[key])
    for key in ("model", "step", "val_loss"):
        _assert_nested_equal(_load(baseline / "best.pt")[key], _load(run_dir / "best.pt")[key])
    assert _metrics(baseline / "metrics.jsonl") == _metrics(run_dir / "metrics.jsonl")
    assert not journal_path.exists()
    assert not (run_dir / trainer._CHECKPOINT_STORE).exists()


def _pending_transaction(tmp_path, monkeypatch):
    cfg = _tiny_train_config(tmp_path)
    _inject_crash(monkeypatch, "publish_last_after", 1)
    with pytest.raises(RuntimeError, match="injected publication"):
        trainer.run(cfg)
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    return cfg, run_dir


@pytest.mark.parametrize("mutation", ["signature", "directory", "filename", "hash", "missing", "snapshot", "step", "baseline"])
def test_bad_recovery_metadata_rejects_before_any_artifact_mutation(tmp_path, monkeypatch, mutation):
    cfg, run_dir = _pending_transaction(tmp_path, monkeypatch)
    path = run_dir / trainer._CHECKPOINT_JOURNAL
    journal = trainer.read_json(path)
    if mutation == "signature":
        journal["signature"] = "0" * 64
    elif mutation == "directory":
        journal["directory"] = "../outside"
    elif mutation == "filename":
        journal["previous"]["best"]["file"] = "../../best.pt"
    elif mutation == "hash":
        journal["previous"]["last"]["sha256"] = "0" * 64
    elif mutation == "step":
        journal["previous_step"] = True
    elif mutation == "baseline":
        journal["previous_step"] = None
        journal["previous"] = dict.fromkeys(trainer._TRANSACTION_FILES)
    else:
        snapshot = run_dir / journal["directory"] / journal["next"]["best"]["file"]
        if mutation == "missing":
            snapshot.unlink()
        else:
            snapshot.write_bytes(b"corrupt snapshot")
    trainer.atomic_write_json(path, journal)
    before = {p.relative_to(run_dir): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="transaction"):
        trainer.run(cfg)
    assert {p.relative_to(run_dir): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == before


def test_interrupted_recovery_preserves_snapshot_links_and_can_repeat(tmp_path, monkeypatch):
    cfg, run_dir = _pending_transaction(tmp_path, monkeypatch)
    journal_path = run_dir / trainer._CHECKPOINT_JOURNAL
    journal = trainer.read_json(journal_path)
    signature = trainer.read_json(run_dir / "manifest.json")["signature"]
    model = trainer.load_model_config(cfg.model_config)
    original = trainer._atomic_snapshot
    once = {"value": True}

    def interrupt_recovery(source, destination, **kwargs):
        original(source, destination, **kwargs)
        if destination == run_dir / "best.pt" and once["value"]:
            once["value"] = False
            # Model a kill after creation of a temporary hard link, before its rename.
            os.link(run_dir / journal["directory"] / "next-best.pt", run_dir / ".best.pt.tmp")
            raise RuntimeError("interrupted recovery")

    monkeypatch.setattr(trainer, "_atomic_snapshot", interrupt_recovery)
    with trainer.RunLock(run_dir / ".run.lock"):
        with pytest.raises(RuntimeError, match="interrupted recovery"):
            trainer._recover_checkpoint_transaction(run_dir, cfg, model, signature)
        trainer._recover_checkpoint_transaction(run_dir, cfg, model, signature)
    assert _load(run_dir / "last.pt")["completed_steps"] == 0
    assert _load(run_dir / "best.pt")["step"] == 0
    trainer.run(cfg)
    assert not journal_path.exists()


def test_real_process_exit_after_best_replace_recovers_old_generation(tmp_path):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=3)
    baseline_cfg = replace(cfg, run_id="process-control")
    trainer.run(baseline_cfg)
    baseline = trainer.variant_run_dir(baseline_cfg, "tiny-1:3")
    code = """
import json, os, sys
import src.train.train as t
cfg = t.TrainConfig(**json.loads(sys.argv[1]))
original = t._atomic_snapshot
def exit_between_best_and_last(source, destination, **kwargs):
    original(source, destination, **kwargs)
    journal = destination.parent / t._CHECKPOINT_JOURNAL
    if destination.name == 'best.pt' and journal.exists() and t.read_json(journal)['next_step'] == 1:
        os._exit(71)
t._atomic_snapshot = exit_between_best_and_last
t.run(cfg)
"""
    child = subprocess.run([sys.executable, "-c", code, json.dumps(asdict(cfg))],
                           capture_output=True, text=True, timeout=60,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert child.returncode == 71, child.stdout + child.stderr
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    assert _load(run_dir / "best.pt")["step"] == 1
    assert _load(run_dir / "last.pt")["completed_steps"] == 0
    assert trainer.read_json(run_dir / trainer._CHECKPOINT_JOURNAL)["phase"] == "pending"
    trainer.run(cfg)
    for key in ("model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss"):
        _assert_nested_equal(_load(baseline / "last.pt")[key], _load(run_dir / "last.pt")[key])
    assert _metrics(baseline / "metrics.jsonl") == _metrics(run_dir / "metrics.jsonl")


def test_checkpoint_transactions_work_without_hardlink_support(tmp_path, monkeypatch):
    cfg = _tiny_train_config(tmp_path)
    baseline_cfg = replace(cfg, run_id="link-control")
    trainer.run(baseline_cfg)

    def no_links(*_args, **_kwargs):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(trainer.os, "link", no_links)
    trainer.run(cfg)
    baseline, actual = (trainer.variant_run_dir(item, "tiny-1:3") for item in (baseline_cfg, cfg))
    for key in ("model", "optimizer", "rng"):
        _assert_nested_equal(_load(baseline / "last.pt")[key], _load(actual / "last.pt")[key])
    assert not (actual / trainer._CHECKPOINT_STORE).exists()


def test_budget_stop_saves_unscheduled_progress_and_exact_resume(tmp_path, monkeypatch):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=4, eval_interval=2,
                  checkpoint_interval=3, wall_time_limit_seconds=5.0)
    baseline_cfg = replace(cfg, run_id="complete-control")
    trainer.run(baseline_cfg)
    baseline = trainer.variant_run_dir(baseline_cfg, "tiny-1:3")
    clock = {"value": 0.0}
    control = trainer.TrainingStopControl(5.0, clock=lambda: clock["value"])
    original_step, original_publish = torch.optim.AdamW.step, trainer._publish_checkpoint_transaction

    def expire_after_update(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        clock["value"] = 7.0
        return result

    def finish_save(*args, **kwargs):
        original_publish(*args, **kwargs)
        if args[4]["completed_steps"]:
            clock["value"] = 9.0

    monkeypatch.setattr(torch.optim.AdamW, "step", expire_after_update)
    monkeypatch.setattr(trainer, "_publish_checkpoint_transaction", finish_save)
    stopped = trainer.run(cfg, stop_control=control)
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    assert stopped["status"] == "stopped" and stopped["certified"] is False
    assert stopped["completed_steps"] == stopped["checkpoint_step"] == 1
    assert stopped["tokens_seen"] == 4
    assert stopped["stop"]["reason"] == "wall_time_limit"
    assert stopped["stop"]["overshoot_seconds"] == 4.0
    assert trainer.read_json(run_dir / "manifest.json")["status"] == "stopped"
    assert not (run_dir / "result.json").exists()
    trainer._validate_metrics(run_dir / "metrics.jsonl", cfg, completed_steps=1)
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    trainer.run(cfg)  # same signed wall policy, fresh invocation allowance
    for key in ("model", "optimizer", "rng", "completed_steps", "tokens_seen", "best_val_loss"):
        _assert_nested_equal(_load(baseline / "last.pt")[key], _load(run_dir / "last.pt")[key])
    assert _metrics(baseline / "metrics.jsonl") == _metrics(run_dir / "metrics.jsonl")


def test_stop_request_during_baseline_preserves_complete_zero_step_restart(tmp_path, monkeypatch):
    cfg = _tiny_train_config(tmp_path)
    control = trainer.TrainingStopControl()
    original = trainer.estimate_loss

    def request_during_eval(*args, **kwargs):
        losses = original(*args, **kwargs)
        control.request_stop("requested:test")
        return losses

    monkeypatch.setattr(trainer, "estimate_loss", request_during_eval)
    stopped = trainer.run(cfg, stop_control=control)
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    assert stopped["completed_steps"] == 0 and stopped["checkpoint_step"] == 0
    assert _load(run_dir / "last.pt")["completed_steps"] == 0
    assert _metrics(run_dir / "metrics.jsonl")[0]["event"] == "eval"
    monkeypatch.setattr(trainer, "estimate_loss", original)
    assert trainer.run(cfg)["completed_steps"] == cfg.max_steps


def test_durable_metrics_corruption_is_rejected_before_resumed_update(tmp_path, monkeypatch):
    cfg = _tiny_train_config(tmp_path)
    control = trainer.TrainingStopControl()
    original_step = torch.optim.AdamW.step

    def stop_after_first_update(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        control.request_stop()
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", stop_after_first_update)
    assert trainer.run(cfg, stop_control=control)["completed_steps"] == 1
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    path = run_dir / "metrics.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join([rows[0], rows[1], rows[1], rows[2]]), encoding="utf-8", newline="\n")
    before = {name: (run_dir / name).read_bytes() for name in trainer._TRANSACTION_FILES.values()}
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *_args, **_kwargs: pytest.fail("bad durable metrics cannot update"))
    with pytest.raises(RuntimeError, match="duplicate, missing, or out-of-order"):
        trainer.run(cfg)
    assert {name: (run_dir / name).read_bytes() for name in trainer._TRANSACTION_FILES.values()} == before


@pytest.mark.parametrize("tail_scope", ["durable", "undurable"])
def test_torn_metrics_tail_is_preserved_unless_durable_prefix_is_complete(tmp_path, monkeypatch, tail_scope):
    cfg = _tiny_train_config(tmp_path)
    control = trainer.TrainingStopControl()
    original_step = torch.optim.AdamW.step

    def stop_after_first_update(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        control.request_stop()
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", stop_after_first_update)
    assert trainer.run(cfg, stop_control=control)["completed_steps"] == 1
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    metrics = run_dir / "metrics.jsonl"
    original_metrics = metrics.read_bytes()
    fragment = b'{"event":"train","step":2,'
    if tail_scope == "durable":
        rows = original_metrics.splitlines(keepends=True)
        metrics.write_bytes(b"".join(rows[:-1]) + b'{"event":"eval","step":1,')
    else:
        metrics.write_bytes(original_metrics + fragment)
    before = {name: (run_dir / name).read_bytes() for name in trainer._TRANSACTION_FILES.values()}
    if tail_scope == "durable":
        monkeypatch.setattr(torch.optim.AdamW, "step", lambda *_args, **_kwargs: pytest.fail("missing durable metrics cannot update"))
        with pytest.raises(RuntimeError, match="complete training trajectory"):
            trainer.run(cfg)
        assert {name: (run_dir / name).read_bytes() for name in trainer._TRANSACTION_FILES.values()} == before
    else:
        trainer.reconcile_metrics(metrics, 1, cfg=cfg)
        assert metrics.read_bytes() == original_metrics
        assert (run_dir / "last.pt").read_bytes() == before["last.pt"]
        monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
        assert trainer.run(cfg)["completed_steps"] == cfg.max_steps


@pytest.mark.parametrize("invalid", [0, -1, True, "30", float("nan"), float("inf")])
def test_invalid_wall_policy_has_no_checkpoint_namespace(tmp_path, invalid):
    cfg = trainer.TrainConfig(ckpt_dir=str(tmp_path / "checkpoints"), wall_time_limit_seconds=invalid)
    with pytest.raises(ValueError, match="wall_time_limit_seconds"):
        trainer.run(cfg)
    assert not Path(cfg.ckpt_dir).exists()


def test_wall_policy_identity_defaults_and_mismatched_resume(tmp_path):
    cfg = _tiny_train_config(tmp_path)
    legacy = asdict(cfg)
    legacy.pop("wall_time_limit_seconds")
    assert trainer._trajectory_config_dict(legacy) == trainer._trajectory_config_dict(asdict(cfg))
    limited = replace(cfg, wall_time_limit_seconds=5)
    assert trainer._trajectory_config_dict(asdict(limited)) != trainer._trajectory_config_dict(legacy)
    control = trainer.TrainingStopControl(5)
    control.request_stop()
    trainer.run(limited, stop_control=control)
    run_dir = trainer.variant_run_dir(cfg, "tiny-1:3")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    with pytest.raises(RuntimeError, match="signature"):
        trainer.run(cfg)
    assert {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()} == before


def test_cooperative_signal_requests_stop_and_restores_previous_handlers():
    control = trainer.TrainingStopControl()
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match="unrelated"):
        with trainer.cooperative_stop_signals(control):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            assert control.reason() == "signal:SIGINT"
            raise RuntimeError("unrelated")
    assert {signum: signal.getsignal(signum) for signum in previous} == previous


def test_sweep_uses_one_deadline_and_never_exports_partial_completed_table(tmp_path, monkeypatch):
    identity = {"signature": "test-data", "tokenizer": {"vocab_size": 16000},
                "outputs": {name: {"tokens": 1024} for name in ("train", "val")}}
    monkeypatch.setattr(sweep, "validate_prepared_dataset", lambda _path: identity)
    controls = []

    def stopped_arm(cfg):
        control = trainer.current_training_stop_control()
        controls.append(control)
        assert control.wall_time_limit_seconds == cfg.wall_time_limit_seconds == 30
        control.request_stop("requested:pilot-margin")
        return {"status": "stopped", "certified": False, "completed_steps": 1,
                "tokens_seen": 16, "stop": control.snapshot()}

    monkeypatch.setattr(sweep, "run", stopped_arm)
    args = ["--data-dir", str(tmp_path / "data"), "--out", str(tmp_path / "results"),
            "--checkpoint-root", str(tmp_path / "checkpoints"), "--run-id", "stopped-sweep",
            "--configs", "configs/attention_only.yaml", "configs/mamba_only.yaml",
            "--max-steps", "4", "--warmup-steps", "0", "--wall-time-limit-seconds", "30"]
    sweep.main(args)
    root = tmp_path / "results" / "stopped-sweep"
    assert len(controls) == 1
    manifest = trainer.read_json(root / "sweep_manifest.json")
    assert manifest["status"] == "stopped" and manifest["completed_arms"] == []
    assert manifest["stopped_arm"]["status"] == "stopped"
    assert not (root / "sweep_results.json").exists()
    assert not (root / "sweep_table.md").exists()
    assert trainer.current_training_stop_control() is None


def test_sweep_default_wall_policy_keeps_legacy_signature(tmp_path):
    import argparse
    args = sweep.argument_parser().parse_args(["--data-dir", str(tmp_path)])
    legacy = argparse.Namespace(**{key: value for key, value in vars(args).items() if key != "wall_time_limit_seconds"})
    assert sweep._sweep_signature(args, "same", "data", [], {}) == sweep._sweep_signature(legacy, "same", "data", [], {})
    args.wall_time_limit_seconds = 30
    assert sweep._sweep_signature(args, "same", "data", [], {}) != sweep._sweep_signature(legacy, "same", "data", [], {})
