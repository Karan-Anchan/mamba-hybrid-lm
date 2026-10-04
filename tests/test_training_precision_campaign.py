"""Precision policy propagation and campaign identity, without GPU/model execution."""

import argparse
import json

import pytest

import scripts.run_sweep as sweep
from src.train.train import load_model_config


def _arguments(tmp_path, *extra):
    return [
        "--data-dir", str(tmp_path / "data"), "--out", str(tmp_path / "results"),
        "--checkpoint-root", str(tmp_path / "checkpoints"), "--run-id", "precision-campaign",
        "--configs", "configs/attention_only.yaml", "configs/mamba_only.yaml",
        "--model-seeds", "11", "--data-seed", "22", "--eval-seed", "33",
        "--max-steps", "3", "--warmup-steps", "0", "--block-size", "4",
        "--batch-size", "2", "--grad-accum", "2", *extra,
    ]


def _data_identity():
    # Data integrity itself is covered by prepared-data/campaign tests and the real
    # pilot dry-run. This fixture isolates policy propagation through the runner.
    return {"signature": "fixture-data", "tokenizer": {"vocab_size": 16000},
            "outputs": {split: {"tokens": 1024} for split in ("train", "val")}}


def _fake_completed(cfg):
    model = load_model_config(cfg.model_config)
    return {"name": model.name, "ratio": model.ratio, "params_m": 1.0,
            "n_attention": model.n_attention_layers, "best_val_ppl": 12.0,
            "avg_tok_per_s": 100, "peak_vram_mb": 10, "tokens_seen": 48,
            "run_id": cfg.run_id}


def test_fp32_dry_run_declares_policy_and_keeps_every_path_unexecuted(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "validate_prepared_dataset", lambda _path: _data_identity())
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("dry-run cannot train"))
    sweep.main(_arguments(tmp_path, "--precision", "float32", "--dry-run"))
    report = json.loads(capsys.readouterr().out)
    assert report["tokens_total"] == 96
    assert report["precision_policy"] == {
        "precision": "float32", "autocast_enabled": False,
        "autocast_dtype": None, "tf32_policy": "disabled",
    }
    assert all(arm["precision"] == "float32" for arm in report["matrix"])
    assert report["scan_backend_plan"]["execution_status"] == "unexecuted"
    assert report["scan_backend_plan"]["observed_paths"] == dict.fromkeys(("training", "prefill", "decode"))
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_fp32_reaches_each_arm_and_cannot_be_replaced_by_default_in_same_namespace(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(sweep, "validate_prepared_dataset", lambda _path: _data_identity())
    monkeypatch.setattr(sweep, "run", lambda cfg: observed.append(cfg) or _fake_completed(cfg))
    sweep.main(_arguments(tmp_path, "--precision", "float32"))
    assert len(observed) == 2
    assert all(cfg.precision == "float32" and cfg.data_seed == 22 and cfg.eval_seed == 33 for cfg in observed)
    root = tmp_path / "results" / "precision-campaign"
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("changed precision cannot train"))
    with pytest.raises(RuntimeError, match="settings differ"):
        sweep.main(_arguments(tmp_path))
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_default_sweep_signature_matches_namespace_without_precision_field(tmp_path):
    args = sweep.argument_parser().parse_args(_arguments(tmp_path))
    matrix = sweep.prepare_matrix(args, args.run_id, _data_identity())
    assert all("precision" not in arm for arm in matrix)
    legacy = argparse.Namespace(**{key: value for key, value in vars(args).items() if key != "precision"})
    assert sweep._sweep_signature(args, args.run_id, "fixture-data", matrix, {}) == sweep._sweep_signature(
        legacy, args.run_id, "fixture-data", matrix, {},
    )
    changed = sweep.argument_parser().parse_args(_arguments(tmp_path, "--precision", "float32"))
    changed_matrix = sweep.prepare_matrix(changed, changed.run_id, _data_identity())
    assert sweep._sweep_signature(args, args.run_id, "fixture-data", matrix, {}) != sweep._sweep_signature(
        changed, changed.run_id, "fixture-data", changed_matrix, {},
    )


def test_bad_precision_cli_fails_before_reading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "validate_prepared_dataset", lambda _path: pytest.fail("invalid policy cannot read data"))
    with pytest.raises(SystemExit):
        sweep.main(_arguments(tmp_path, "--precision", "float16"))
    assert not (tmp_path / "results").exists()
