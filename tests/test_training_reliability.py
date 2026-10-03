"""Fast checks for run isolation, deterministic sampling, and recoverable training artifacts."""

import copy
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
from src.model.scan_backend import BackendUnavailableError, ScanBackend
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


@pytest.mark.parametrize("stage, invalid", [
    ("loss", float("nan")), ("loss", float("inf")),
    ("gradient", float("nan")), ("gradient", float("inf")),
    ("accumulated", 3e38),
])
def test_nonfinite_update_preserves_weights_optimizer_and_durable_progress(tmp_path, monkeypatch, stage, invalid):
    cfg = replace(_tiny_train_config(tmp_path), grad_accum=2)
    real_model = train_module.HybridLM
    real_optimizer = train_module.torch.optim.AdamW
    captured = {}
    training_microbatches = 0
    update_calls = 0
    invalid_loss_backwards = []

    def make_model(*args, **kwargs):
        model = real_model(*args, **kwargs)
        captured["model"] = model
        real_forward = model.forward

        def forward(x, targets):
            nonlocal training_microbatches
            logits, loss = real_forward(x, targets)
            if model.training:
                training_microbatches += 1
                if stage == "accumulated" and training_microbatches >= 3:
                    loss = loss * 0 + invalid  # finite microbatch losses, overflowing float32 sum
                if training_microbatches == 4:  # second microbatch of the second optimizer step
                    run_dir = variant_run_dir(cfg, "tiny-1:3")
                    captured["last_bytes"] = (run_dir / "last.pt").read_bytes()
                    captured["best_bytes"] = (run_dir / "best.pt").read_bytes()
                    captured["metrics_bytes"] = (run_dir / "metrics.jsonl").read_bytes()
                    captured["optimizer_before_failure"] = copy.deepcopy(captured["optimizer"].state_dict())
                    if stage == "loss":
                        loss = loss * invalid
                        loss.register_hook(lambda gradient: invalid_loss_backwards.append(gradient))
                    elif stage == "gradient":
                        loss.register_hook(lambda gradient: torch.full_like(gradient, invalid))
            return logits, loss

        model.forward = forward
        return model

    def make_optimizer(*args, **kwargs):
        optimizer = real_optimizer(*args, **kwargs)
        captured["optimizer"] = optimizer
        real_step = optimizer.step

        def step(*step_args, **step_kwargs):
            nonlocal update_calls
            update_calls += 1
            return real_step(*step_args, **step_kwargs)

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(train_module, "HybridLM", make_model)
    monkeypatch.setattr(train_module.torch.optim, "AdamW", make_optimizer)
    expected = "accumulated training loss" if stage == "accumulated" else f"training {stage}"
    with pytest.raises(FloatingPointError, match=rf"non-finite {expected}.*step 2.*microbatch 2/2"):
        run(cfg)

    run_dir = variant_run_dir(cfg, "tiny-1:3")
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    assert update_calls == 1
    assert training_microbatches == 4
    assert invalid_loss_backwards == []  # a rejected loss never enters backward
    assert last["completed_steps"] == 1
    assert last["tokens_seen"] == 8
    assert (run_dir / "last.pt").read_bytes() == captured["last_bytes"]
    assert (run_dir / "best.pt").read_bytes() == captured["best_bytes"]
    assert (run_dir / "metrics.jsonl").read_bytes() == captured["metrics_bytes"]
    assert not (run_dir / "result.json").exists()
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] != "completed"
    _assert_nested_equal(last["model"], captured["model"].state_dict())
    _assert_nested_equal(last["optimizer"], captured["optimizer"].state_dict())
    _assert_nested_equal(captured["optimizer_before_failure"], captured["optimizer"].state_dict())
    assert all(parameter.grad is None for parameter in captured["model"].parameters())

    # The valid checkpoint remains usable; the failed in-memory update cannot contaminate resume.
    monkeypatch.setattr(train_module, "HybridLM", real_model)
    monkeypatch.setattr(train_module.torch.optim, "AdamW", real_optimizer)
    assert run(cfg)["completed_steps"] == 2


def test_explicit_seed_streams_separate_initialization_training_and_evaluation(tmp_path, monkeypatch):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=1, model_seed=11, data_seed=22, eval_seed=33)
    real_batch = train_module.get_batch
    observed = []

    def record_batch(*args, **kwargs):
        x, y = real_batch(*args, **kwargs)
        observed.append((kwargs["generator"].initial_seed(), x.clone()))
        return x, y

    monkeypatch.setattr(train_module, "get_batch", record_batch)
    outcomes = {}
    for label, overrides in (
        ("first", {}), ("new-model", {"model_seed": 44}),
        ("new-data", {"data_seed": 55}), ("new-eval", {"eval_seed": 66}),
    ):
        observed.clear()
        current = replace(cfg, run_id=label, **overrides)
        run(current)
        run_dir = variant_run_dir(current, "tiny-1:3")
        outcomes[label] = {
            "model": torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)["model"],
            "train": [x for seed, x in observed if seed == current.data_seed],
            "eval": [x for seed, x in observed if seed in (current.eval_seed + 1, current.eval_seed + 2)],
        }

    assert torch.equal(outcomes["first"]["train"][0], outcomes["new-model"]["train"][0])
    assert all(torch.equal(a, b) for a, b in zip(outcomes["first"]["eval"], outcomes["new-model"]["eval"]))
    assert any(not torch.equal(outcomes["first"]["model"][name], outcomes["new-model"]["model"][name])
               for name in outcomes["first"]["model"])
    assert not torch.equal(outcomes["first"]["train"][0], outcomes["new-data"]["train"][0])
    assert all(torch.equal(a, b) for a, b in zip(outcomes["first"]["eval"], outcomes["new-data"]["eval"]))
    assert torch.equal(outcomes["first"]["train"][0], outcomes["new-eval"]["train"][0])
    assert any(not torch.equal(a, b) for a, b in zip(outcomes["first"]["eval"], outcomes["new-eval"]["eval"]))


@pytest.mark.parametrize("field", ["seed", "model_seed", "data_seed", "eval_seed"])
@pytest.mark.parametrize("invalid", [-1, 2**32, 1.5, True])
def test_seed_streams_reject_invalid_values_before_creating_artifacts(tmp_path, field, invalid):
    cfg = TrainConfig(ckpt_dir=str(tmp_path / "checkpoints"), **{field: invalid})
    with pytest.raises(ValueError, match=rf"{field} must be an integer"):
        run(cfg)
    assert not (tmp_path / "checkpoints").exists()


def test_unspecified_seed_streams_match_the_explicit_legacy_policy(tmp_path):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=1, run_id="default-streams")
    explicit_cfg = replace(
        cfg, run_id="explicit-streams", model_seed=cfg.seed, data_seed=cfg.seed, eval_seed=cfg.seed,
    )
    run(cfg)
    run(explicit_cfg)
    checkpoints = [torch.load(
        variant_run_dir(current, "tiny-1:3") / "last.pt", map_location="cpu", weights_only=True,
    ) for current in (cfg, explicit_cfg)]
    for key in ("model", "optimizer", "rng"):
        _assert_nested_equal(checkpoints[0][key], checkpoints[1][key])


def test_optional_seed_cli_arguments_are_integers(monkeypatch):
    captured = []
    monkeypatch.setattr(train_module, "run", lambda cfg: captured.append(cfg))
    monkeypatch.setattr(train_module.sys, "argv", [
        "train", "--model-seed", "0", "--data-seed", "17", "--eval-seed", "23",
    ])
    train_module.main()
    assert (captured[0].model_seed, captured[0].data_seed, captured[0].eval_seed) == (0, 17, 23)


def test_base_seed_cannot_be_unspecified(tmp_path):
    with pytest.raises(ValueError, match="seed must be an integer"):
        run(TrainConfig(seed=None, ckpt_dir=str(tmp_path / "checkpoints")))
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("backend, expected_path", [
    ("reference", "torch.quadratic_ssd"), ("torch_chunked", "torch.chunked_ssd"),
])
def test_training_records_requested_resolved_and_executed_scan_paths(tmp_path, monkeypatch, backend, expected_path):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=1, scan_backend=backend, scan_chunk_size=3)
    calls = []
    real_scan = ScanBackend.scan

    def record_scan(self, *args, **kwargs):
        calls.append((self.name, torch.is_grad_enabled()))
        return real_scan(self, *args, **kwargs)

    monkeypatch.setattr(ScanBackend, "scan", record_scan)
    result = run(cfg)
    run_dir = variant_run_dir(cfg, "tiny-1:3")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (backend, True) in calls  # the selected engine actually participated in a training graph
    assert result["scan_backend"]["requested"] == backend
    assert result["scan_backend"]["resolved"] == backend
    assert result["scan_backend"]["fused"] is False
    assert result["scan_backend"]["chunk_size"] == 3
    assert result["observed_paths"] == {"training": expected_path, "prefill": None, "decode": None}
    assert manifest["scan_backend"] == result["scan_backend"]
    assert manifest["observed_paths"] == result["observed_paths"]
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    assert "scan_backend" not in checkpoint["model_config"]
    assert all("scan_backend" not in name for name in checkpoint["model"])
    assert run(cfg) == result


@pytest.mark.parametrize("overrides, message", [
    ({"scan_backend": "pretend_cuda"}, "unknown scan backend"),
    ({"scan_chunk_size": 0}, "positive integer"),
    ({"scan_chunk_size": -1}, "positive integer"),
    ({"scan_chunk_size": 2.5}, "positive integer"),
    ({"scan_chunk_size": True}, "positive integer"),
])
def test_invalid_scan_selection_fails_before_creating_a_run(tmp_path, overrides, message):
    cfg = TrainConfig(ckpt_dir=str(tmp_path / "checkpoints"), **overrides)
    with pytest.raises(ValueError, match=message):
        run(cfg)
    assert not (tmp_path / "checkpoints").exists()


def test_unavailable_fused_selection_fails_without_fallback_or_run_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.model.scan_backend.platform.system", lambda: "Windows")
    cfg = TrainConfig(ckpt_dir=str(tmp_path / "checkpoints"), scan_backend="fused_mamba")
    with pytest.raises(BackendUnavailableError, match="Linux/CUDA environment"):
        run(cfg)
    assert not (tmp_path / "checkpoints").exists()


def test_result_before_manifest_finalization_can_recover_scan_execution_metadata(tmp_path, monkeypatch):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=1)
    real_write = train_module.atomic_write_json

    def interrupt_manifest_finalization(path, value):
        if path.name == "manifest.json" and value.get("status") == "completed":
            raise RuntimeError("simulated finalization interruption")
        return real_write(path, value)

    monkeypatch.setattr(train_module, "atomic_write_json", interrupt_manifest_finalization)
    with pytest.raises(RuntimeError, match="finalization interruption"):
        run(cfg)
    run_dir = variant_run_dir(cfg, "tiny-1:3")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["observed_paths"]["training"] is None
    assert (run_dir / "result.json").is_file()

    monkeypatch.setattr(train_module, "atomic_write_json", real_write)
    result = run(cfg)
    recovered = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert recovered["status"] == "completed"
    assert recovered["observed_paths"] == result["observed_paths"]


def test_all_attention_training_never_claims_an_executed_scan(tmp_path, monkeypatch):
    cfg = replace(_tiny_train_config(tmp_path), max_steps=1, scan_backend="torch_chunked")
    config_path = Path(cfg.model_config)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace('name: "tiny-1:3"', 'name: "tiny-attention"')
        .replace('ratio: "1:3"', 'ratio: "1:0"'),
        encoding="utf-8",
    )

    def no_scan(*_args, **_kwargs):
        raise AssertionError("an all-attention model cannot execute a Mamba scan")

    monkeypatch.setattr(ScanBackend, "scan", no_scan)
    result = run(cfg)
    manifest = json.loads((variant_run_dir(cfg, "tiny-attention") / "manifest.json").read_text(encoding="utf-8"))
    assert result["n_mamba"] == 0
    assert result["scan_backend"]["mamba_layers"] == 0
    assert result["observed_paths"] == {"training": None, "prefill": None, "decode": None}
    assert manifest["observed_paths"] == result["observed_paths"]
    assert run(cfg) == result
