"""RoPE treatments retain the original control, fixed evidence and execution state."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

import scripts.study_rope_precision as study
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


def tiny_model():
    return HybridLM(ModelConfig(ratio="1:3", vocab_size=32, d_model=16, n_layers=4,
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
                        run_id="rope-fixture", source="local/openwebtext",
                        revision="0123456789abcdef0123456789abcdef01234567",
                        train_tokens=30, val_tokens=60, progress_docs=100)
    documents = [" ".join(f"v{(i+j)%30}" for j in range(8)) for i in range(50)]
    prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    manifest = validate_prepared_dataset(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    root = tmp_path / "checkpoints"
    run_dir = root / "registered-v1" / "hybrid"
    run_dir.mkdir(parents=True)
    model = tiny_model()
    checkpoint = run_dir / "best.pt"
    torch.save({"model_config": asdict(model.cfg), "model": model.state_dict(),
                "signature": "rope-fixture-training", "step": 7, "val_loss": 2.5}, checkpoint)
    training = {"status": "completed", "ratio": "1:3", "model_config": asdict(model.cfg),
                "artifacts": {"best": "best.pt"}, "artifact_sha256": {"best": study.base.file_sha256(checkpoint)},
                "signature": "rope-fixture-training", "data": {"meta": meta,
                    "files": {item["file"]: {"sha256": item["sha256"], "bytes": item["bytes"]}
                              for item in manifest["outputs"].values()}}}
    (run_dir / "manifest.json").write_text(json.dumps(training), encoding="utf-8")
    return {"checkpoint_root": root, "training_run_id": "registered-v1", "data_dir": data_dir,
            "tokenizer": tokenizer_path, "ratios": ("1:3",), "device": "cpu",
            "length": 5, "prefix_length": 2, "chunk_size": 2}


def test_actual_operand_replay_keeps_original_full_fp32_and_cached_bf16_distinct():
    q = (torch.arange(48).reshape(1, 2, 3, 8) / 37).bfloat16()
    k = q.flip(-1).contiguous()
    angles = torch.arange(24).reshape(3, 8).float() * .271
    cos, sin = angles.cos(), angles.sin()
    before = [study.base.tensor_sha256(value) for value in (q, k, cos, sin)]
    report = study.frozen_rope_probe(q, k, cos, sin)
    original, fp32, bf16 = report["policies"]
    assert original["post_rotation"]["full"]["q"]["dtype"] == "torch.float32"
    assert original["post_rotation"]["cached"]["q"]["dtype"] == "torch.bfloat16"
    assert not original["post_rotation"]["bitwise_equal"]
    assert original["post_rotation"]["comparisons"]["q"]["max_absolute_error"] > 0
    assert original["post_cast"]["comparison_dtype"] == "torch.bfloat16"
    assert "original stateless model has no added explicit cast" in original["post_cast"]["scope"]
    assert not original["post_cast"]["bitwise_equal"]
    assert fp32["post_rotation"]["bitwise_equal"] and fp32["post_cast"]["bitwise_equal"]
    assert bf16["post_rotation"]["bitwise_equal"] and bf16["post_cast"]["bitwise_equal"]
    assert fp32["post_rotation"]["full"]["q"]["sha256"] == original["post_rotation"]["full"]["q"]["sha256"]
    assert bf16["post_rotation"]["full"]["q"]["sha256"] == original["post_rotation"]["cached"]["q"]["sha256"]
    assert [study.base.tensor_sha256(value) for value in (q, k, cos, sin)] == before
    with pytest.raises(ValueError, match="BF16"):
        study.frozen_rope_probe(q.float(), k.float(), cos, sin)


def test_original_full_model_control_matches_unpatched_probe_and_retains_real_memory():
    model = tiny_model()
    x, y, _ = study.base.shared_windows(np.arange(20, dtype="<u2"), 5, 1, 2027)[0]
    model.configure_scan_backend("reference", 2)
    with torch.no_grad():
        anchor, _ = model(x)
    recorder = {"stage": "unexecuted", "events": {}}
    original = study.base.bf16_cache_probe(model, x, y, anchor.cpu(), 2, 2, True, recorder)
    weights = study.base.weight_sha256(model)
    report = study.study_model(model, x, y, prefix_length=2, chunk_size=2)
    assert study.base.canonical_sha256(report["policies"][0]["bf16_cached"]) == study.base.canonical_sha256(original)
    assert [row["policy"] for row in report["policies"]] == list(study.POLICIES)
    assert len(report["frozen_probes"]) == 1
    for policy in report["policies"]:
        check = policy["bf16_cached"]
        assert check["real_prefix"]["position"] == 2 and not check["real_prefix"]["synthetic"]
        assert check["real_prefix"]["unchanged_after_cloned_continuations"]
        assert check["segmented_schedule"] == [2, 1]
        assert len(check["internal_logits"]) == 8 and len(check["retained_state"]) == 4
        assert check["tolerance"] == TOLERANCES["bfloat16"]
        assert check["fp32_anchor"]["scope"] == "separate cross-precision full-logit comparison"
        assert check["next_token_effects"]["full_vs_fp32_anchor"]["scored_tokens"] == 5
        assert "passed" not in check["next_token_effects"]["full_vs_fp32_anchor"]
        assert policy["attention_observations"] and policy["scan_observations"]
    assert study.base.weight_sha256(model) == weights
    assert all(parameter.grad is None for parameter in model.parameters())
    assert report["full_model_passed"] == all(row["bf16_cached"]["passed"] for row in report["policies"])


@pytest.mark.parametrize("policy", list(study.POLICIES))
def test_temporary_forward_hooks_modes_and_scan_backends_restore_after_failure(policy, monkeypatch):
    model = tiny_model().train()
    model.blocks[0].eval()
    attention = next(block.mixer for block in model.blocks if block.is_attn)
    original = attention.forward
    modes = [module.training for module in model.modules()]
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    weights = study.base.weight_sha256(model)
    recorder = {"stage": "bf16_full", "frozen": {}, "attention_events": {}}
    monkeypatch.setattr(attention.out, "forward", lambda *_: (_ for _ in ()).throw(RuntimeError("injected after rotation")))
    with pytest.raises(RuntimeError, match="injected"):
        with study.base.preserve_model_execution(model), study.rope_policy(model, policy, recorder), torch.autocast("cpu", dtype=torch.bfloat16):
            model(torch.tensor([[1, 2, 3]]))
    assert attention.forward == original and "forward" not in attention.__dict__
    assert not attention.qkv._forward_hooks
    assert [module.training for module in model.modules()] == modes
    assert [block.mixer.scan_backend for block in model.blocks if not block.is_attn] == backends
    assert study.base.weight_sha256(model) == weights


def test_existing_instance_forward_is_restored_and_frozen_only_never_claims_full_route_execution():
    model = tiny_model()
    attention = next(block.mixer for block in model.blocks if block.is_attn)
    existing = attention.forward
    custom = lambda *args, **kwargs: existing(*args, **kwargs)
    attention.forward = custom
    recorder = {"stage": "unexecuted", "frozen": {}, "attention_events": {}}
    with study.rope_policy(model, "shared_fp32", recorder):
        assert attention.forward is not custom
    assert attention.forward is custom
    result = study.study_model(model, torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]),
                               prefix_length=2, chunk_size=2, full_model=False)
    assert result["full_model_passed"] is None and result["frozen_probes"]
    assert all(row["bf16_cached"] is None for row in result["policies"])
    assert [row["execution_status"] for row in result["policies"]] == ["completed", "unexecuted", "unexecuted"]
    assert attention.forward is custom


def test_registered_inputs_source_identity_rng_and_checkpoint_bytes_remain_fixed(registered_fixture):
    checkpoint = next(registered_fixture["checkpoint_root"].rglob("best.pt"))
    original_hash = study.base.file_sha256(checkpoint)
    before = torch.get_rng_state().clone()
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    report = study.run_study(**registered_fixture)
    assert report["certified"] is False and report["execution_status"] == "completed"
    assert len(report["cases"]) == 1
    assert report["protocol_sha256"] == study.base.canonical_sha256(report["protocol"])
    assert report["protocol"]["source_sha256"]["scripts/study_rope_precision.py"]
    assert report["protocol"]["quality_equivalence_margin"] is None
    assert report["cases"][0]["tokens_sha256"] == report["protocol"]["shared_inputs"][0]["tokens_sha256"]
    assert report["protocol"]["runtime"]["precision_flags"]["cuda_matmul_allow_tf32"] is False
    assert report["protocol"]["runtime"]["precision_flags"]["cudnn_allow_tf32"] is False
    assert torch.equal(before, torch.get_rng_state())
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == flags
    assert study.base.file_sha256(checkpoint) == original_hash
    assert str(registered_fixture["data_dir"].parent) not in json.dumps(report)


def test_declared_window_and_baseline_hash_bind_before_model_allocation(registered_fixture, tmp_path, monkeypatch):
    baseline = study.base.run_study(**registered_fixture)
    baseline_path = tmp_path / "baseline.json"
    study.base.write_report(baseline_path, baseline)
    inputs = baseline["protocol"]["shared_inputs"][0]
    declaration = {"schema": 1, "status_at_declaration": "planned_before_execution",
        "policies": study.POLICIES, "tolerances": TOLERANCES, "batch_size": 1,
        "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
        "baseline_report": str(baseline_path), "baseline_sha256": study.base.file_sha256(baseline_path),
        **{name: registered_fixture[name] for name in ("device", "length", "prefix_length", "chunk_size")},
        "seed": 2027, "ratios": ["1:3"], **{name: inputs[name] for name in ("start_token", "tokens_sha256", "targets_sha256")}}
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    report = study.run_study(**registered_fixture, protocol=declaration_path)
    assert report["protocol"]["declaration"]["sha256"] == study.base.file_sha256(declaration_path)
    assert report["protocol"]["baseline_report"]["sha256"] == study.base.file_sha256(baseline_path)
    assert report["cases"][0]["baseline_control_comparison"]["exact_fields_equal"]
    monkeypatch.setattr(study.base, "load_variant_model", lambda *_: pytest.fail("bad identity cannot allocate a model"))
    declaration["tokens_sha256"] = "0" * 64
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match="window differs"):
        study.run_study(**registered_fixture, protocol=declaration_path)
    declaration["tokens_sha256"] = inputs["tokens_sha256"]
    declaration["baseline_sha256"] = "0" * 64
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline report hash"):
        study.run_study(**registered_fixture, protocol=declaration_path)


@pytest.mark.parametrize("settings", [{"device": "gpu"}, {"seed": True}, {"length": 258}, {"prefix_length": 257},
    {"windows": 5}, {"chunk_size": 0}, {"tokenwise": 1}, {"full_model": 0}, {"ratios": ("1:3", "1:3")},
    {"ratios": (["1:3"],)}, {"training_run_id": "../bad"}])
def test_invalid_protocol_fails_before_data_audit(settings, monkeypatch):
    monkeypatch.setattr(study.base, "audit_inputs", lambda *_: pytest.fail("invalid settings cannot audit or allocate"))
    with pytest.raises(ValueError):
        study.run_study(**settings)


def test_cli_preserves_atomic_output_refuses_overwrite_and_records_incomplete(monkeypatch, tmp_path, capsys):
    report = {"schema": 1, "execution_status": "completed", "certified": False}
    monkeypatch.setattr(study, "run_study", lambda **_: report)
    output = tmp_path / "report.json"
    assert study.main(["--output", str(output)]) == 0
    first = output.read_bytes()
    assert json.loads(first) == report and b"\r\n" not in first
    with pytest.raises(SystemExit):
        study.main(["--output", str(output)])
    assert output.read_bytes() == first
    monkeypatch.setattr(study, "run_study", lambda **_: (_ for _ in ()).throw(RuntimeError("injected")))
    failed = tmp_path / "failed.json"
    assert study.main(["--output", str(failed)]) == 2
    failure = json.loads(failed.read_text(encoding="utf-8"))
    assert failure["execution_status"] == "incomplete" and failure["certified"] is False
    assert not list(tmp_path.glob("*.tmp"))
    capsys.readouterr()
