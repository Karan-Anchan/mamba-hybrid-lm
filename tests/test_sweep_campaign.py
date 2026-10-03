"""Verify sweep isolation and read-only campaign preparation without running training."""

import json
from dataclasses import asdict

import pytest
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers

import scripts.run_sweep as sweep
from src.data.prepare_data import PrepareConfig, prepare_dataset
from src.data.train_tokenizer import EOT
from src.model.config import ModelConfig
from src.model.scan_backend import BackendUnavailableError
from src.train.train import RunLock, load_model_config


@pytest.fixture
def campaign(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab = {f"v{i}": i for i in range(15999)} | {EOT: 15999}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="v0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_path))
    prepared = prepare_dataset(PrepareConfig(
        dataset="openwebtext", tokenizer=str(tokenizer_path), out_dir=str(tmp_path / "data"),
        run_id="fixture", source="local/campaign", revision="0" * 40,
        train_tokens=24, val_tokens=12,
    ), iterator_factory=lambda _split: iter(["v0 v1 v2"] * 40))
    paths = []
    for name, ratio in (("arm-a", "1:1"), ("arm-b", "0:1")):
        cfg = ModelConfig(name=name, ratio=ratio, d_model=16, n_layers=2, vocab_size=16000,
                          head_dim=8, mamba_headdim=8, d_state=4, d_conv=2, mlp_multiple_of=8)
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")
        paths.append(path)
    return {
        "data": prepared, "configs": paths,
        "out": tmp_path / "results", "checkpoints": tmp_path / "checkpoints",
    }


def command(campaign, *extra, use_configs=True):
    args = [
        "--data-dir", str(campaign["data"]), "--out", str(campaign["out"]),
        "--checkpoint-root", str(campaign["checkpoints"]), "--run-id", "campaign",
        "--max-steps", "3", "--warmup-steps", "0", "--batch-size", "2",
        "--grad-accum", "2", "--block-size", "4",
    ]
    if use_configs:
        args.extend(["--configs", *(str(path) for path in campaign["configs"])])
    return [*args, *extra]


def fake_completed(cfg):
    model = load_model_config(cfg.model_config)
    return {
        "name": model.name, "ratio": model.ratio, "params_m": 1.0,
        "n_attention": model.n_attention_layers, "best_val_ppl": 12.0,
        "avg_tok_per_s": 100, "peak_vram_mb": 10,
        "tokens_seen": cfg.max_steps * cfg.batch_size * cfg.grad_accum * cfg.block_size,
        "run_id": cfg.run_id,
    }


def test_dry_run_verifies_real_data_and_configs_but_creates_no_outputs(campaign, monkeypatch, capsys):
    capsys.readouterr()
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("dry-run cannot start training"))
    sweep.main(command(campaign, "--model-seeds", "11", "22", "--data-seed", "55", "--eval-seed", "66", "--dry-run"))
    planned = json.loads(capsys.readouterr().out)
    assert planned["status"] == "dry_run" and planned["budget_kind"] == "token_positions"
    assert planned["tokens_per_step"] == 16 and planned["tokens_per_arm"] == 48
    assert planned["tokens_total"] == 192
    assert [(arm["name"], arm["model_seed"], arm["run_id"]) for arm in planned["matrix"]] == [
        ("arm-a", 11, "campaign-seed-11"), ("arm-b", 11, "campaign-seed-11"),
        ("arm-a", 22, "campaign-seed-22"), ("arm-b", 22, "campaign-seed-22"),
    ]
    assert all(arm["data_seed"] == 55 and arm["eval_seed"] == 66 for arm in planned["matrix"])
    assert all(arm["config_sha256"] and arm["model_config"]["layer_types"] for arm in planned["matrix"])
    assert not campaign["out"].exists() and not campaign["checkpoints"].exists()


def test_chunked_dry_run_reports_only_intended_paths_and_remains_unexecuted(campaign, monkeypatch, capsys):
    capsys.readouterr()
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("dry-run cannot execute a scan"))
    sweep.main(command(campaign, "--scan-backend", "torch_chunked", "--scan-chunk-size", "16", "--dry-run"))
    report = json.loads(capsys.readouterr().out)
    backend = report["scan_backend_plan"]
    assert backend["requested"] == backend["resolved"] == "torch_chunked"
    assert backend["chunk_size"] == 16 and backend["execution_status"] == "unexecuted"
    assert backend["paths"] == dict.fromkeys(("training", "prefill", "decode"), "torch.chunked_ssd")
    assert backend["observed_paths"] == dict.fromkeys(("training", "prefill", "decode"))
    assert not campaign["out"].exists() and not campaign["checkpoints"].exists()


def test_unavailable_fused_dry_run_fails_before_data_reads_without_fallback(campaign, monkeypatch):
    import src.model.scan_backend as scan_module

    monkeypatch.setattr(scan_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sweep, "validate_prepared_dataset", lambda _path: pytest.fail("backend resolution must fail first"))
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("unavailable fused cannot train"))
    with pytest.raises(BackendUnavailableError, match="supported Linux/CUDA environment"):
        sweep.main(command(campaign, "--scan-backend", "fused_mamba", "--dry-run"))
    assert not campaign["out"].exists() and not campaign["checkpoints"].exists()


def test_selected_backend_is_passed_to_each_arm_and_recorded_in_identity(campaign, monkeypatch):
    observed = []
    monkeypatch.setattr(sweep, "run", lambda cfg: observed.append(cfg) or fake_completed(cfg))
    sweep.main(command(campaign, "--scan-backend", "torch_chunked", "--scan-chunk-size", "16"))
    assert all(cfg.scan_backend == "torch_chunked" and cfg.scan_chunk_size == 16 for cfg in observed)
    manifest_path = campaign["out"] / "campaign" / "sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scan_backend_plan"]["requested"] == "torch_chunked"
    assert manifest["scan_backend_plan"]["execution_status"] == "unexecuted"


@pytest.mark.parametrize("option, value", [("--scan-backend", "torch_chunked"), ("--scan-chunk-size", "16")])
def test_changed_backend_or_chunk_cannot_overwrite_existing_namespace(campaign, monkeypatch, option, value):
    monkeypatch.setattr(sweep, "run", fake_completed)
    sweep.main(command(campaign))
    root = campaign["out"] / "campaign"
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("backend protocol changes cannot train in an old namespace"))
    with pytest.raises(RuntimeError, match="settings differ"):
        sweep.main(command(campaign, option, value))
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_explicit_initialization_seeds_use_common_data_seeds_and_durable_matrix(campaign, monkeypatch):
    observed = []

    def record_run(cfg):
        observed.append(cfg)
        return fake_completed(cfg)

    monkeypatch.setattr(sweep, "run", record_run)
    sweep.main(command(campaign, "--model-seeds", "7", "9"))
    assert len(observed) == 4
    assert [cfg.model_seed for cfg in observed] == [7, 7, 9, 9]
    assert [cfg.run_id for cfg in observed] == ["campaign-seed-7"] * 2 + ["campaign-seed-9"] * 2
    assert all(cfg.seed == 1337 and cfg.data_seed == 1337 and cfg.eval_seed == 1337 for cfg in observed)
    root = campaign["out"] / "campaign"
    manifest = json.loads((root / "sweep_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((root / "sweep_results.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["tokens_total"] == 192
    assert [result["sweep_arm"] for result in results] == manifest["matrix"]
    assert [result["model_seed"] for result in results] == [7, 7, 9, 9]
    assert manifest["completed_arms"] == [arm["arm_id"] for arm in manifest["matrix"]]
    assert "model_seed" in (root / "sweep_table.md").read_text(encoding="utf-8")


def test_default_sweep_retains_three_historical_configs_and_legacy_seed_policy(campaign, monkeypatch):
    observed = []
    monkeypatch.setattr(sweep, "run", lambda cfg: observed.append(cfg) or fake_completed(cfg))
    sweep.main(command(campaign, use_configs=False))
    assert [cfg.model_config for cfg in observed] == [
        "configs/ratio_1_3.yaml", "configs/ratio_1_7.yaml", "configs/ratio_1_15.yaml",
    ]
    assert all(cfg.run_id == "campaign" and cfg.seed == 1337 for cfg in observed)
    assert all(cfg.model_seed is None and cfg.data_seed is None and cfg.eval_seed is None for cfg in observed)
    table = (campaign["out"] / "campaign" / "sweep_table.md").read_text(encoding="utf-8")
    assert table.startswith("| ratio | params_m |")


def test_custom_arms_report_effective_seeds_without_changing_legacy_train_policy(campaign, monkeypatch):
    observed = []
    monkeypatch.setattr(sweep, "run", lambda cfg: observed.append(cfg) or fake_completed(cfg))
    sweep.main(command(campaign))
    results = json.loads((campaign["out"] / "campaign" / "sweep_results.json").read_text(encoding="utf-8"))
    assert all(cfg.model_seed is None and cfg.run_id == "campaign" for cfg in observed)
    assert all(result["model_seed"] == result["data_seed"] == result["eval_seed"] == 1337 for result in results)
    table = (campaign["out"] / "campaign" / "sweep_table.md").read_text(encoding="utf-8")
    assert "arm-a" in table and "model_seed" in table


@pytest.mark.parametrize("option, value", [
    ("--max-steps", "0"), ("--max-steps", "-1"), ("--block-size", "0"),
    ("--batch-size", "0"), ("--grad-accum", "-1"), ("--eval-interval", "0"),
    ("--eval-iters", "0"), ("--log-interval", "0"), ("--checkpoint-interval", "0"),
    ("--warmup-steps", "4"), ("--warmup-steps", "-1"), ("--lr", "nan"),
    ("--lr", "0"), ("--data-seed", "-1"), ("--eval-seed", str(2**32)),
    ("--model-seeds", "-1"),
    ("--scan-chunk-size", "0"), ("--scan-chunk-size", "-1"),
])
def test_invalid_settings_fail_before_data_reads_or_output_creation(campaign, monkeypatch, option, value):
    monkeypatch.setattr(sweep, "validate_prepared_dataset", lambda _path: pytest.fail("settings must be validated first"))
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("invalid settings cannot train"))
    with pytest.raises(ValueError):
        sweep.main(command(campaign, option, value))
    assert not campaign["out"].exists() and not campaign["checkpoints"].exists()


def test_duplicate_seeds_fail_before_outputs(campaign, monkeypatch):
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("duplicate arms cannot train"))
    with pytest.raises(ValueError, match="duplicate model seeds"):
        sweep.main(command(campaign, "--model-seeds", "7", "7"))
    assert not campaign["out"].exists()


def test_every_model_is_validated_before_the_first_arm_runs(campaign, monkeypatch):
    second = campaign["configs"][1]
    bad = yaml.safe_load(second.read_text(encoding="utf-8"))
    bad["d_conv"] = 0
    second.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("a later malformed config cannot follow a trained arm"))
    with pytest.raises(ValueError, match="d_conv"):
        sweep.main(command(campaign))
    assert not campaign["out"].exists() and not campaign["checkpoints"].exists()


@pytest.mark.parametrize("kind, message", [
    ("name", "duplicate model name"), ("path", "duplicate config path"),
    ("case_name", "checkpoint directory on Windows"),
    ("vocab", "vocab mismatch"), ("window", "too short"),
])
def test_invalid_matrix_rejects_before_any_outputs(campaign, monkeypatch, kind, message):
    extra = []
    if kind == "path":
        campaign["configs"][1] = campaign["configs"][0]
    elif kind == "window":
        extra = ["--block-size", "100"]
    else:
        second = campaign["configs"][1]
        bad = yaml.safe_load(second.read_text(encoding="utf-8"))
        if kind in ("name", "case_name"):
            bad["name"] = "arm-a" if kind == "name" else "ARM-A"
        else:
            bad["vocab_size"] = 32
        second.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("invalid matrix cannot train"))
    with pytest.raises(ValueError, match=message):
        sweep.main(command(campaign, *extra))
    assert not campaign["out"].exists() and not campaign["checkpoints"].exists()


def test_dry_run_corrupt_data_is_rejected_without_outputs(campaign, monkeypatch):
    val_path = campaign["data"] / "val.bin"
    contents = bytearray(val_path.read_bytes())
    contents[0] ^= 1
    val_path.write_bytes(contents)
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("corrupt data cannot train"))
    with pytest.raises(RuntimeError, match="integrity"):
        sweep.main(command(campaign, "--dry-run"))
    assert not campaign["out"].exists()


def test_changed_config_invalidates_resume_identity_before_training(campaign, monkeypatch):
    monkeypatch.setattr(sweep, "run", fake_completed)
    sweep.main(command(campaign, "--model-seeds", "7"))
    result_path = campaign["out"] / "campaign" / "sweep_results.json"
    before = result_path.read_bytes()
    second = campaign["configs"][1]
    second.write_text(second.read_text(encoding="utf-8") + "\n# changed config identity\n", encoding="utf-8")
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("changed identity must fail before training"))
    with pytest.raises(RuntimeError, match="settings differ"):
        sweep.main(command(campaign, "--model-seeds", "7"))
    assert result_path.read_bytes() == before


def test_changed_model_seed_invalidates_resume_identity(campaign, monkeypatch):
    monkeypatch.setattr(sweep, "run", fake_completed)
    sweep.main(command(campaign, "--model-seeds", "7"))
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("changed seed cannot overwrite campaign"))
    with pytest.raises(RuntimeError, match="settings differ"):
        sweep.main(command(campaign, "--model-seeds", "9"))


def test_tampered_recorded_matrix_fails_resume_before_training(campaign, monkeypatch):
    monkeypatch.setattr(sweep, "run", fake_completed)
    sweep.main(command(campaign, "--model-seeds", "7"))
    manifest_path = campaign["out"] / "campaign" / "sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matrix"][0]["model_seed"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("tampered matrix must not train"))
    with pytest.raises(RuntimeError, match="signed identity"):
        sweep.main(command(campaign, "--model-seeds", "7"))


def test_active_sweep_lock_prevents_aggregate_writers(campaign, monkeypatch):
    monkeypatch.setattr(sweep, "run", lambda _cfg: pytest.fail("active sweep must not start another arm"))
    with RunLock(campaign["out"] / "campaign" / ".sweep.lock"):
        with pytest.raises(RuntimeError, match="already active"):
            sweep.main(command(campaign))
    assert not (campaign["out"] / "campaign" / "sweep_manifest.json").exists()


def test_pure_endpoint_configs_share_historical_geometry_but_change_layer_types():
    historical = asdict(load_model_config("configs/ratio_1_3.yaml"))
    for identity in ("name", "ratio", "layer_types"):
        historical.pop(identity)
    for path, ratio, layer in (("configs/attention_only.yaml", "1:0", "attention"),
                               ("configs/mamba_only.yaml", "0:1", "mamba")):
        cfg = load_model_config(path)
        assert cfg.ratio == ratio and cfg.layer_types == [layer] * 16
        endpoint = asdict(cfg)
        for identity in ("name", "ratio", "layer_types"):
            endpoint.pop(identity)
        assert endpoint == historical
