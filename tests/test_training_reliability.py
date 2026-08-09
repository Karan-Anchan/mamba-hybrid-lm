"""Fast checks for run isolation, deterministic sampling, and recoverable training artifacts."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import src.train.train as train_module
from scripts.run_sweep import (
    steps_for_tokens,
    sweep_output_dir,
    tokens_for_steps,
    warmup_steps_for_fraction,
)
from src.data.dataset import get_batch
from src.train.train import (
    MetricsWriter,
    RunLock,
    TrainConfig,
    atomic_write_json,
    atomic_torch_save,
    estimate_loss,
    make_batch_generator,
    reconcile_metrics,
    run,
    validate_run_id,
    variant_slug,
    variant_run_dir,
)


def _write_tiny_fixture(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    data_dir.mkdir()
    ramp = (np.arange(512, dtype=np.uint16) % 64).astype(np.uint16)
    ramp.tofile(data_dir / "train.bin")
    ramp[::-1].tofile(data_dir / "val.bin")
    (data_dir / "meta.json").write_text(json.dumps({"vocab_size": 64}), encoding="utf-8")

    model_config = root / "tiny.yaml"
    model_config.write_text(
        "\n".join([
            'name: "tiny-1:3"', 'ratio: "1:3"', "vocab_size: 64", "d_model: 16",
            "n_layers: 1", "head_dim: 8", "expand: 2", "d_state: 4",
            "mamba_headdim: 8", "d_conv: 2", "n_groups: 1", "mlp_ratio: 2.0",
            "mlp_multiple_of: 8",
        ]) + "\n",
        encoding="utf-8",
    )
    return data_dir, model_config


def _tiny_train_config(tmp_path: Path) -> TrainConfig:
    data_dir, model_config = _write_tiny_fixture(tmp_path)
    return TrainConfig(
        model_config=str(model_config), data_dir=str(data_dir), ckpt_dir=str(tmp_path / "checkpoints"),
        run_id="resume-test", device="cpu", max_steps=2, warmup_steps=0,
        batch_size=1, grad_accum=1, block_size=4, eval_interval=1, eval_iters=1,
        log_interval=1, checkpoint_interval=1, grad_checkpointing=False,
    )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_run_namespaces_preserve_previous_artifacts(tmp_path):
    preview = TrainConfig(ckpt_dir=str(tmp_path / "checkpoints"), run_id="preview")
    authoritative = TrainConfig(ckpt_dir=str(tmp_path / "checkpoints"), run_id="week3-700m")
    preview_dir = variant_run_dir(preview, "hybrid-1:7")
    full_dir = variant_run_dir(authoritative, "hybrid-1:7")
    preview_dir.mkdir(parents=True)
    marker = preview_dir / "best.pt"
    marker.write_bytes(b"preview")

    full_dir.mkdir(parents=True)
    assert preview_dir != full_dir
    assert marker.read_bytes() == b"preview"
    assert sweep_output_dir(tmp_path / "results", "preview") != sweep_output_dir(
        tmp_path / "results", "week3-700m"
    )
    with pytest.raises(ValueError):
        validate_run_id("../escape")
    with pytest.raises(ValueError):
        validate_run_id("CON")
    assert variant_slug("a:b") != variant_slug("a/b")
    with pytest.raises(ValueError):
        variant_slug("..")

    lock_path = tmp_path / "locked" / ".run.lock"
    with RunLock(lock_path):
        with pytest.raises(RuntimeError, match="already active"):
            with RunLock(lock_path):
                pass


def test_data_generators_ignore_model_rng_and_evaluation_is_fixed():
    data = np.arange(512, dtype=np.uint16)
    torch.manual_seed(11)
    first = make_batch_generator(123, "train")
    x1, _ = get_batch(data, 8, 4, device="cpu", generator=first)
    torch.rand(1000)  # stand in for a model consuming a different amount of initialization RNG
    second = make_batch_generator(123, "train")
    x2, _ = get_batch(data, 8, 4, device="cpu", generator=second)
    assert torch.equal(x1, x2)

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.windows = []

        def forward(self, x, targets):
            self.windows.append(x.detach().clone())
            return None, x.float().mean()

    cfg = TrainConfig(device="cpu", block_size=8, batch_size=2, eval_iters=2, seed=77)
    model = RecordingModel()
    splits = {"train": data, "val": data[::-1].copy()}
    train_generator = make_batch_generator(cfg.seed, "train")
    train_state = train_generator.get_state().clone()
    global_state = torch.get_rng_state().clone()
    first_losses = estimate_loss(model, splits, cfg)
    assert torch.equal(train_generator.get_state(), train_state)
    assert torch.equal(torch.get_rng_state(), global_state)
    first_windows = [window.clone() for window in model.windows]
    model.windows.clear()
    second_losses = estimate_loss(model, splits, cfg)
    assert first_losses == second_losses
    assert all(torch.equal(a, b) for a, b in zip(first_windows, model.windows))


def test_atomic_checkpoint_and_metrics_recovery(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"known-good")

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(train_module.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        atomic_torch_save({"new": True}, checkpoint)
    assert checkpoint.read_bytes() == b"known-good"
    assert not (tmp_path / ".last.pt.tmp").exists()

    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"known-good"}\n', encoding="utf-8")
    real_replace = train_module.os.replace

    def fail_summary_replace(source, destination):
        if Path(destination) == summary:
            raise OSError("simulated replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(train_module.os, "replace", fail_summary_replace)
    with pytest.raises(OSError, match="replace"):
        atomic_write_json(summary, {"status": "new"})
    assert json.loads(summary.read_text(encoding="utf-8")) == {"status": "known-good"}
    assert not (tmp_path / ".summary.json.tmp").exists()
    with pytest.raises(ValueError, match="Out of range float"):
        atomic_write_json(summary, {"metric": float("nan")})
    assert json.loads(summary.read_text(encoding="utf-8")) == {"status": "known-good"}

    metrics_path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(metrics_path)
    writer.append({"event": "train", "step": 1})
    writer.append({"event": "train", "step": 2})
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial":')
    reconcile_metrics(metrics_path, completed_steps=1)
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"event": "train", "step": 1}]


def test_interrupted_run_resumes_then_completed_run_skips(tmp_path, monkeypatch):
    cfg = _tiny_train_config(tmp_path)
    baseline_cfg = replace(cfg, run_id="baseline")
    run(baseline_cfg)
    baseline_dir = variant_run_dir(baseline_cfg, "tiny-1:3")
    baseline_last = torch.load(baseline_dir / "last.pt", map_location="cpu", weights_only=True)

    real_append = train_module.MetricsWriter.append
    fail_once = {"value": True}

    def interrupt_during_second_step(self, record):
        if record["event"] == "train" and record["step"] == 2 and fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("simulated interruption")
        return real_append(self, record)

    monkeypatch.setattr(train_module.MetricsWriter, "append", interrupt_during_second_step)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(cfg)

    run_dir = variant_run_dir(cfg, "tiny-1:3")
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert not (run_dir / "result.json").exists()
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["completed_steps"] == 1
    assert checkpoint["tokens_seen"] == 4
    assert {"model", "optimizer", "train_config", "model_config", "best_val_loss", "rng"} <= checkpoint.keys()
    checkpoint["peak_vram_mb"] = 123.4  # stand in for a peak recorded before a GPU interruption
    torch.save(checkpoint, run_dir / "last.pt")

    monkeypatch.setattr(train_module.MetricsWriter, "append", real_append)

    last_path = run_dir / "last.pt"
    last_bytes = last_path.read_bytes()
    last_path.unlink()
    with pytest.raises(RuntimeError, match="progress metrics without last checkpoint"):
        run(cfg)
    assert not (run_dir / "result.json").exists()
    last_path.write_bytes(last_bytes)

    train_path = Path(cfg.data_dir) / "train.bin"
    original_data = train_path.read_bytes()
    original_stat = train_path.stat()
    mutated = bytearray(original_data)
    mutated[0] ^= 1
    train_path.write_bytes(mutated)
    with pytest.raises(RuntimeError, match="configuration differs"):
        run(cfg)
    train_path.write_bytes(original_data)
    os.utime(train_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    resumed = run(cfg)
    assert resumed["completed_steps"] == 2
    assert resumed["tokens_seen"] == 8
    assert resumed["peak_vram_mb"] == 123
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "completed"
    resumed_last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    _assert_nested_equal(baseline_last["model"], resumed_last["model"])
    _assert_nested_equal(baseline_last["optimizer"], resumed_last["optimizer"])
    _assert_nested_equal(baseline_last["rng"], resumed_last["rng"])
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert "optimizer" not in best and "rng" not in best
    assert "optimizer" in resumed_last and "rng" in resumed_last
    metric_rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    train_step_one = next(row for row in metric_rows if row["event"] == "train" and row["step"] == 1)
    eval_step_one = next(row for row in metric_rows if row["event"] == "eval" and row["step"] == 1)
    assert eval_step_one["lr"] == train_step_one["lr"]

    def should_not_load_data(*_args, **_kwargs):
        raise AssertionError("completed variant should be skipped before loading data or building a model")

    monkeypatch.setattr(train_module, "load_split", should_not_load_data)
    skipped = run(cfg)
    assert skipped == resumed

    with pytest.raises(RuntimeError, match="provenance|signature|configuration"):
        run(replace(cfg, lr=cfg.lr * 2))


def test_completed_skip_rejects_changed_data_code_and_required_artifacts(tmp_path, monkeypatch):
    cfg = _tiny_train_config(tmp_path)
    completed = run(cfg)
    run_dir = variant_run_dir(cfg, "tiny-1:3")

    result_path = run_dir / "result.json"
    original_result = result_path.read_bytes()
    result_path.unlink()
    with pytest.raises(RuntimeError, match="required result artifact is missing"):
        run(cfg)
    assert not result_path.exists()
    result_path.write_bytes(original_result)

    last_path = run_dir / "last.pt"
    original_last = last_path.read_bytes()
    tampered_last = torch.load(last_path, map_location="cpu", weights_only=True)
    first_tensor = next(iter(tampered_last["model"].values()))
    first_tensor.view(-1)[0] += 1
    torch.save(tampered_last, last_path)
    tampered_bytes = last_path.read_bytes()
    result_path.unlink()
    with pytest.raises(RuntimeError, match="required result artifact is missing"):
        run(cfg)
    assert not result_path.exists()
    assert last_path.read_bytes() == tampered_bytes
    last_path.write_bytes(original_last)
    result_path.write_bytes(original_result)

    for name in ("best.pt", "last.pt", "metrics.jsonl"):
        path = run_dir / name
        original = path.read_bytes()
        path.unlink()
        with pytest.raises(RuntimeError, match="missing"):
            run(cfg)
        path.write_bytes(original)

    last_path = run_dir / "last.pt"
    original_last = last_path.read_bytes()
    last_path.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="unreadable|checksum"):
        run(cfg)
    last_path.write_bytes(original_last)

    metrics_path = run_dir / "metrics.jsonl"
    original_metrics = metrics_path.read_bytes()
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
    with pytest.raises(RuntimeError, match="metrics"):
        run(cfg)
    metrics_path.write_bytes(original_metrics)

    manifest_path = run_dir / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    rows = metrics_path.read_text(encoding="utf-8").splitlines()
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(rows[-1] + "\n")
    manifest = json.loads(original_manifest)
    manifest["artifact_sha256"]["metrics"] = train_module._file_sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate, missing, or out-of-order"):
        run(cfg)
    metrics_path.write_bytes(original_metrics)
    manifest_path.write_bytes(original_manifest)

    result_path = run_dir / "result.json"
    original_result = result_path.read_bytes()
    invalid_result = dict(completed)
    invalid_result["best_val_loss"] = float("nan")
    result_path.write_text(json.dumps(invalid_result) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        run(cfg)
    result_path.write_bytes(original_result)

    real_code_provenance = train_module._code_provenance
    monkeypatch.setattr(
        train_module, "_code_provenance",
        lambda: {**real_code_provenance(), "fingerprint": "changed"},
    )
    with pytest.raises(RuntimeError, match="provenance|signature"):
        run(cfg)
    monkeypatch.setattr(train_module, "_code_provenance", real_code_provenance)

    real_runtime_provenance = train_module._runtime_provenance
    monkeypatch.setattr(
        train_module, "_runtime_provenance",
        lambda: {**real_runtime_provenance(), "torch": "changed"},
    )
    with pytest.raises(RuntimeError, match="provenance|signature"):
        run(cfg)
    monkeypatch.setattr(train_module, "_runtime_provenance", real_runtime_provenance)

    train_path = Path(cfg.data_dir) / "train.bin"
    mutated = bytearray(train_path.read_bytes())
    mutated[-1] ^= 1
    train_path.write_bytes(mutated)
    with pytest.raises(RuntimeError, match="provenance|signature"):
        run(cfg)


def test_authoritative_token_budget_arithmetic():
    assert tokens_for_steps(8000, 8, 4, 512) == 131_072_000
    assert steps_for_tokens(700_000_000, 8, 4, 512) == 42_725
    assert tokens_for_steps(42_725, 8, 4, 512) == 700_006_400
    assert warmup_steps_for_fraction(42_725, 0.02) == 855
    assert steps_for_tokens(800_000_000, 8, 4, 512) == 48_829
