"""Natural-token diagnostics must preserve evidence, targets and execution settings."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

import scripts.study_checkpoint_numerics as study
from scripts.check_scan_backend import TOLERANCES
from src.data.prepare_data import PrepareConfig, prepare_dataset, validate_prepared_dataset
from src.data.train_tokenizer import EOT
from src.model.config import ModelConfig
from src.model.lm import HybridLM


@pytest.fixture(autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(ratio="1:3"):
    return HybridLM(ModelConfig(ratio=ratio, vocab_size=32, d_model=16, n_layers=4,
                                head_dim=8, mamba_headdim=8, d_state=4, mlp_multiple_of=8)).eval()


@pytest.fixture
def registered_fixture(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab = {f"v{i}": i for i in range(30)} | {"[UNK]": 30, EOT: 31}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_path))
    data_dir = tmp_path / "data"
    cfg = PrepareConfig(dataset="openwebtext", tokenizer=str(tokenizer_path), out_dir=str(data_dir),
                        run_id="numerics-fixture", source="local/openwebtext",
                        revision="0123456789abcdef0123456789abcdef01234567",
                        train_tokens=30, val_tokens=80, progress_docs=100)
    documents = [" ".join(f"v{(i+j)%30}" for j in range(8)) for i in range(50)]
    prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    data_manifest = validate_prepared_dataset(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    root = tmp_path / "checkpoints"
    for ratio in ("1:3", "0:1"):
        model = tiny_model(ratio)
        run_dir = root / "registered-v1" / ratio.replace(":", "-")
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "best.pt"
        signature = f"training-signature-{ratio}"
        torch.save({"model_config": asdict(model.cfg), "model": model.state_dict(),
                    "signature": signature, "step": 7, "val_loss": 2.5}, checkpoint)
        manifest = {"status": "completed", "ratio": ratio, "model_config": asdict(model.cfg),
                    "artifacts": {"best": "best.pt"}, "artifact_sha256": {"best": study.file_sha256(checkpoint)},
                    "signature": signature, "data": {"meta": meta,
                        "files": {item["file"]: {"sha256": item["sha256"], "bytes": item["bytes"]}
                                  for item in data_manifest["outputs"].values()}}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"checkpoint_root": root, "training_run_id": "registered-v1", "data_dir": data_dir,
            "tokenizer": tokenizer_path, "ratios": ("1:3", "0:1")}


def test_shared_windows_preserve_shifted_targets_private_rng_and_last_valid_window():
    data = np.arange(130, dtype="<u2")
    before = torch.get_rng_state().clone()
    rows = study.shared_windows(data, 129, 1, 2027)
    x, y, identity = rows[0]
    assert identity["start_token"] == 0
    assert x.tolist() == [list(range(129))] and y.tolist() == [list(range(1, 130))]
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(before, torch.get_rng_state())
    assert study.shared_windows(data, 129, 1, 2027)[0][2] == identity
    assert identity["tokens_sha256"] != identity["targets_sha256"]
    with pytest.raises(ValueError, match="length\\+1"):
        study.shared_windows(data, 130, 1, 2027)


def test_token_metrics_score_true_next_token_and_use_common_fp32_arithmetic():
    targets = torch.tensor([[1, 2, 3]])
    reference = torch.zeros(1, 3, 4)
    actual = torch.zeros_like(reference)
    actual.scatter_(-1, targets[..., None], 4)
    result = study.token_effects(actual.bfloat16(), reference, targets, position_start=128)
    expected_lp = torch.log_softmax(actual, -1).gather(-1, targets[..., None]).flatten()
    assert result["position_start"] == 128 and result["position_stop_exclusive"] == 131
    assert result["actual_mean_nll"] == pytest.approx(-expected_lp.mean().item())
    assert result["mean_nll_delta"] < 0
    assert result["per_token"]["actual_logprob"] == pytest.approx(expected_lp.tolist())
    # Scoring the input token itself would produce a very different, incorrect result.
    wrong = study.token_effects(actual, reference, torch.tensor([[0, 1, 2]]))
    assert wrong["actual_mean_nll"] > result["actual_mean_nll"] + 3
    assert "passed" not in result and "no quality-equivalence cutoff" in result["scope"]


def test_real_prefix_states_are_cloned_and_long_suffix_schedules_are_recorded():
    model = tiny_model()
    x, y, _ = study.shared_windows(np.arange(20, dtype="<u2"), 9, 1, 2027)[0]
    with torch.no_grad():
        anchor, _ = model(x)
    recorder = {"stage": "unexecuted", "events": {}}
    result = study.bf16_cache_probe(model, x, y, anchor, 4, 4, True, recorder)
    assert result["real_prefix"]["position"] == 4
    assert not result["real_prefix"]["synthetic"]
    assert result["real_prefix"]["unchanged_after_cloned_continuations"]
    assert all(field["nonzero_elements"] > 0 for field in result["real_prefix"]["fields"] if field["field"].endswith("ssm"))
    assert result["segmented_schedule"] == [4, 1]
    assert not result["one_shot_vs_tokenwise_degenerate"]
    assert set(result["final_positions"].values()) == {9}
    assert result["next_token_effects"]["prefix"]["per_token"]["target_ids"] == y[:, 3:4].flatten().tolist()
    assert result["next_token_effects"]["suffix_one_shot"]["per_token"]["target_ids"] == y[:, 4:].flatten().tolist()
    assert "suffix_tokenwise_vs_one_shot" in result["internal_logits"]
    assert result["retained_state"] and recorder["events"]


def test_nonfinite_gradients_are_all_retained_and_weights_are_not_updated():
    model = tiny_model()
    original = study.weight_sha256(model)
    hook = model.embed.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    recorder = {"stage": "unexecuted", "events": {}}
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    try:
        result, _ = study.fp32_backend_probe(model, x, y, 2, recorder)
    finally:
        hook.remove()
    assert not result["passed"]
    assert "embed.weight" in result["gradients"]["failed_parameters"]
    check = next(check for check in result["gradients"]["checks"] if check["parameter"] == "embed.weight")
    assert not check["finite"] and check["cosine_similarity"] is None
    assert study.weight_sha256(model) == original
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_comparisons_keep_every_failure_and_reject_name_shape_mismatch():
    reference = {"first": torch.ones(2), "second": torch.ones(2), "zero": torch.zeros(2)}
    candidate = {"first": torch.zeros(2), "second": torch.full((2,), float("nan")), "zero": torch.zeros(2)}
    report = study.gradient_comparisons(candidate, reference)
    assert report["failed_parameters"] == ["first", "second"]
    assert report["checks"][2]["passed"] and report["checks"][2]["cosine_similarity"] is None
    with pytest.raises(RuntimeError, match="names/order"):
        study.gradient_comparisons({"other": torch.ones(2)}, reference)
    with pytest.raises(ValueError, match="identical shapes"):
        study.gradient_comparisons({"a": torch.ones(3)}, {"a": torch.ones(2)})


def test_execution_wrappers_and_mixed_modes_restore_after_injected_exception(monkeypatch):
    model = tiny_model().train()
    model.blocks[0].eval()
    modes = [module.training for module in model.modules()]
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    before = study.weight_sha256(model)

    def injected(*_args, **_kwargs):
        assert any(isinstance(block.mixer.scan_backend, study.ObservedScan) for block in model.blocks if not block.is_attn)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(model, "forward", injected)
    with pytest.raises(RuntimeError, match="injected"):
        study.study_model(model, torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]),
                          prefix_length=2, chunk_size=2, tokenwise=True)
    assert [module.training for module in model.modules()] == modes
    assert all(block.mixer.scan_backend is original
               for block, original in zip([block for block in model.blocks if not block.is_attn], backends))
    assert study.weight_sha256(model) == before


def test_tf32_flags_restore_in_nested_scope_and_on_exception():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError, match="injected"):
            with study.tf32_disabled():
                assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                with study.tf32_disabled():
                    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                    raise RuntimeError("injected")
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def test_unsupported_matmul_precision_metadata_is_not_execution_blocker(monkeypatch):
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: (_ for _ in ()).throw(RuntimeError("mixed APIs")))
    assert study.runtime_metadata("cpu")["precision_flags"]["float32_matmul_precision"] is None


def test_end_to_end_registered_inputs_hashes_gradients_and_rng_are_reproducible(registered_fixture):
    checkpoints = sorted(registered_fixture["checkpoint_root"].rglob("best.pt"))
    hashes = [study.file_sha256(path) for path in checkpoints]
    before = torch.get_rng_state().clone()
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    kwargs = {**registered_fixture, "device": "cpu", "length": 5, "prefix_length": 4, "chunk_size": 4}
    first = study.run_study(**kwargs)
    second = study.run_study(**kwargs)
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert torch.equal(before, torch.get_rng_state())
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == flags
    assert [study.file_sha256(path) for path in checkpoints] == hashes
    assert first["certified"] is False and first["execution_status"] == "completed"
    assert len(first["cases"]) == 2
    inputs = first["protocol"]["shared_inputs"][0]
    for row in first["cases"]:
        assert row["tokens_sha256"] == inputs["tokens_sha256"] and row["targets_sha256"] == inputs["targets_sha256"]
        gradient_names = [check["parameter"] for check in row["fp32_backend"]["gradients"]["checks"]]
        assert len(gradient_names) == len(set(gradient_names))
        assert "embed.weight" in gradient_names and "lm_head.weight" not in gradient_names
        assert row["fp32_backend"]["tolerance"] == TOLERANCES["float32"]
        assert row["bf16_cached"]["tolerance"] == TOLERANCES["bfloat16"]
        assert row["bf16_cached"]["one_shot_vs_tokenwise_degenerate"]
        assert row["actual_paths"]["fp32_reference_forward"] == ["torch.quadratic_ssd"]
        assert row["actual_paths"]["fp32_torch_chunked_forward"] == ["torch.chunked_ssd"]
        assert row["actual_paths"]["bf16_full_tokenwise"] == ["torch.chunked_ssd"]
    assert first["status"] == ("completed" if all(row["passed"] for row in first["cases"]) else "completed_with_parity_failures")
    assert first["protocol"]["quality_equivalence_margin"] is None
    assert first["protocol"]["source_sha256"]["scripts/study_checkpoint_numerics.py"]
    assert first["protocol"]["runtime"]["precision_flags"]["cuda_matmul_allow_tf32"] is False
    assert first["protocol"]["runtime"]["precision_flags"]["cudnn_allow_tf32"] is False
    assert first["protocol_sha256"] == study.canonical_sha256(first["protocol"])
    assert str(registered_fixture["data_dir"].parent) not in json.dumps(first)


def test_skipped_full_tokenwise_stage_is_null_not_reported_as_executed(registered_fixture):
    report = study.run_study(**{**registered_fixture, "device": "cpu", "length": 3, "prefix_length": 2,
                               "chunk_size": 2, "tokenwise": False, "ratios": ("1:3",)})
    row = report["cases"][0]
    assert row["actual_paths"]["bf16_full_tokenwise"] is None
    assert row["bf16_cached"]["unexecuted_stages"] == ["bf16_full_tokenwise"]
    assert "full_tokenwise" not in row["bf16_cached"]["internal_logits"]


@pytest.mark.parametrize("settings", [{"device": "gpu"}, {"seed": True}, {"length": 258},
    {"prefix_length": 129}, {"windows": 5}, {"chunk_size": 0}, {"tokenwise": 1},
    {"training_run_id": ".."}, {"ratios": ("1:3", "1:3")}, {"ratios": (["1:3"],)}])
def test_invalid_protocol_fails_before_data_audit_or_model_work(settings, monkeypatch):
    monkeypatch.setattr(study, "audit_inputs", lambda *_: pytest.fail("invalid protocol cannot audit/allocate"))
    with pytest.raises(ValueError):
        study.run_study(**settings)


@pytest.mark.parametrize("tamper", ["validation", "tokenizer", "historical_data"])
def test_tampered_input_identities_fail_before_any_model_is_loaded(registered_fixture, monkeypatch, tamper):
    if tamper == "validation":
        with (registered_fixture["data_dir"] / "val.bin").open("r+b") as handle:
            handle.write(b"\xff\xff")
    elif tamper == "tokenizer":
        with registered_fixture["tokenizer"].open("a", encoding="utf-8") as handle:
            handle.write("\n")
    else:
        manifest_path = next(registered_fixture["checkpoint_root"].rglob("manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["data"]["files"]["val.bin"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(study, "load_variant_model", lambda *_: pytest.fail("failed audit cannot load model"))
    with pytest.raises(RuntimeError):
        study.run_study(**{**registered_fixture, "device": "cpu", "length": 3, "prefix_length": 2})


def test_cli_refuses_existing_evidence_before_diagnostics(tmp_path, monkeypatch):
    output = tmp_path / "historical.json"
    output.write_text('{"historical":true}', encoding="utf-8")
    monkeypatch.setattr(study, "run_study", lambda **_: pytest.fail("cannot run when output exists"))
    with pytest.raises(SystemExit) as stopped:
        study.main(["--output", str(output)])
    assert stopped.value.code == 2
    assert output.read_text(encoding="utf-8") == '{"historical":true}'


def test_atomic_publication_preserves_lf_and_rejects_racing_existing_evidence(tmp_path, monkeypatch):
    output = tmp_path / "new.json"
    study.write_report(output, {"complete": True})
    assert b"\r\n" not in output.read_bytes()
    with pytest.raises(FileExistsError):
        study.write_report(output, {"changed": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"complete": True}
    raced = tmp_path / "raced.json"
    original_link = study.os.link

    def race(source, destination):
        Path(destination).write_text('{"historical":true}', encoding="utf-8")
        original_link(source, destination)

    monkeypatch.setattr(study.os, "link", race)
    with pytest.raises(FileExistsError):
        study.write_report(raced, {"complete": True})
    assert json.loads(raced.read_text(encoding="utf-8")) == {"historical": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_nonfinite_report_is_rejected_before_output_directory_creation(tmp_path):
    output = tmp_path / "absent" / "new.json"
    with pytest.raises(ValueError):
        study.write_report(output, {"invalid": float("nan")})
    assert not output.parent.exists()
