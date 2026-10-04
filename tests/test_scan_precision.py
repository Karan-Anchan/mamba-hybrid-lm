"""Bounded diagnostics must isolate treatments and preserve the model they inspect."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest
import torch

import scripts.study_scan_precision as study
from scripts.check_scan_backend import TOLERANCES
from src.model.config import ModelConfig
from src.model.lm import HybridLM


@pytest.fixture(autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return HybridLM(ModelConfig(ratio="0:1", vocab_size=32, d_model=16, n_layers=2,
                                head_dim=8, mamba_headdim=8, d_state=4, mlp_multiple_of=8)).eval()


def recorder():
    return {"stage": "full", "events": {}, "projection_inputs": {}, "scan_outputs": {},
            "layer_stages": {}, "first_mamba_layer": 0}


def test_experimental_wrappers_restore_on_runtime_exception():
    model = tiny_model()
    backends = [block.mixer.scan_backend for block in model.blocks]
    weights = study.weight_sha256(model)
    with pytest.raises(RuntimeError, match="injected"):
        with study.experimental_precision(model, study.TREATMENTS[-1], recorder()):
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                model(torch.tensor([[1, 2, 3]]))
            assert "forward" in model.lm_head.__dict__
            raise RuntimeError("injected diagnostic failure")
    assert study.weight_sha256(model) == weights
    assert "forward" not in model.lm_head.__dict__
    for block, original in zip(model.blocks, backends):
        assert block.mixer.scan_backend is original
        assert "forward" not in block.mixer.out_proj.__dict__
        assert not block.mixer._forward_hooks
        assert not block.mixer.out_proj._forward_pre_hooks
    assert not model.lm_head._forward_pre_hooks


def test_nonfinite_logits_fail_and_restore_wrappers(monkeypatch):
    model = tiny_model()
    originals = [block.mixer.scan_backend for block in model.blocks]

    def nonfinite(value):
        return torch.full((*value.shape[:-1], model.cfg.vocab_size), float("nan"))

    monkeypatch.setattr(model.lm_head, "forward", nonfinite)
    with pytest.raises(ValueError, match="finite"):
        study._evaluate(model, torch.tensor([[1, 2, 3]]), study.TREATMENTS[-1], 2)
    assert model.lm_head.forward is nonfinite
    assert all(block.mixer.scan_backend is original for block, original in zip(model.blocks, originals))
    assert not model.lm_head._forward_pre_hooks


def test_independent_recurrence_has_known_nonzero_memory_solution():
    x = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)
    dt = torch.ones(1, 3, 1)
    A = torch.tensor([-torch.log(torch.tensor(2.0))])
    B = C = torch.ones(1, 3, 1)
    D = torch.zeros(1)
    output, final = study.recurrence_oracle((x, dt, A, B, C, D), torch.full((1, 1, 1, 1), 2.0))
    torch.testing.assert_close(output.flatten(), torch.tensor([2.0, 3.0, 4.5]))
    torch.testing.assert_close(final.flatten(), torch.tensor([4.5]))


def test_scan_probe_has_independent_anchor_direct_gradients_and_future_causality():
    generator = torch.Generator().manual_seed(7)
    operands = (
        torch.randn(1, 5, 2, 2, generator=generator),
        torch.rand(1, 5, 2, generator=generator) * 0.1,
        -torch.ones(2),
        torch.randn(1, 5, 3, generator=generator),
        torch.randn(1, 5, 3, generator=generator),
        torch.ones(2),
    )
    report = study.scan_isolation(operands, 2, 7)
    assert set(report["operand_sha256"]) == {"x", "dt", "A", "B", "C", "D"}
    precise = next(case for case in report["cases"] if not case["autocast"])
    assert precise["passed"]
    gradients = report["direct_gradients"]["cases"]
    assert all(set(case["checks"]) == {"x", "dt", "A", "B", "C", "D", "initial_state"} for case in gradients)
    assert all(case["future_gradient_causality"]["passed"] for case in gradients)
    assert next(case for case in gradients if not case["autocast"])["passed"]


def test_reproducible_inputs_weights_protocol_and_rng_without_certification():
    tolerances_before = deepcopy(TOLERANCES)
    torch.manual_seed(314)
    rng_before = torch.get_rng_state().clone()
    first = study.run_study(lengths=[1, 3], chunk_size=2, batch_size=1, seed=1337)
    assert torch.equal(torch.get_rng_state(), rng_before)
    second = study.run_study(lengths=[1, 3], chunk_size=2, batch_size=1, seed=1337)
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert first["protocol"]["inputs"] == second["protocol"]["inputs"]
    assert first["certified"] is False and first["execution_status"] == "completed"
    assert first["status"] in ("completed", "completed_with_parity_failures")
    assert TOLERANCES == tolerances_before
    assert len(first["cases"]) == 10
    assert all(case["weights_sha256"] == first["protocol"]["weights_sha256"] for case in first["cases"])
    assert all(case["tolerance"] == TOLERANCES["bfloat16"] for case in first["cases"] if case["treatment"].startswith("bf16"))
    assert all("fp32_anchor" in case["comparisons"] for case in first["cases"] if case["treatment"].startswith("bf16"))
    for case in first["cases"]:
        assert case["observed_paths"]["full"] == ["torch.quadratic_ssd"]
        assert case["observed_paths"]["token_decode"] == ["torch.chunked_ssd"]
        assert case["operator_observations"] and case["layer_stage_consistency"]
    changed = study.run_study(lengths=[1], chunk_size=2, batch_size=1, seed=1338)
    assert changed["protocol"]["inputs"][0]["tokens_sha256"] != first["protocol"]["inputs"][0]["tokens_sha256"]


def test_inputs_preserve_original_gate_rng_draw_order():
    lengths = [1, 3]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1337)
        cfg = ModelConfig(ratio="0:1", d_model=64, n_layers=4, vocab_size=128, head_dim=16,
                          mamba_headdim=16, d_state=16, mlp_multiple_of=16)
        HybridLM(cfg)
        expected = []
        for length in lengths:
            expected.append((study.tensor_sha256(torch.randint(0, 128, (1, length))),
                             study.tensor_sha256(torch.randint(0, 128, (1, length)))))
    report = study.run_study(lengths=lengths, batch_size=1, chunk_size=2)
    assert [(item["tokens_sha256"], item["targets_sha256"]) for item in report["protocol"]["inputs"]] == expected


def test_historical_checkpoint_is_read_only_and_configuration_controls_the_study(tmp_path):
    model = tiny_model()
    checkpoint = tmp_path / "historical.pt"
    torch.save({"model_config": asdict(model.cfg), "model": model.state_dict()}, checkpoint)
    before = checkpoint.read_bytes()
    report = study.run_study(lengths=[1], batch_size=1, chunk_size=2, checkpoint=checkpoint)
    assert checkpoint.read_bytes() == before
    assert report["protocol"]["checkpoint"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert report["protocol"]["checkpoint"]["path"] == checkpoint.name
    assert report["protocol"]["checkpoint"]["path_scope"] == "external basename; content identified by SHA256"
    assert report["protocol"]["model_config"]["d_model"] == 16
    assert report["protocol"]["weights_sha256"] == study.weight_sha256(model)


@pytest.mark.parametrize("settings", [{"lengths": []}, {"lengths": [1, 1]}, {"lengths": [513]},
                                      {"batch_size": 5}, {"chunk_size": 0}, {"backend": "fused_mamba"}])
def test_out_of_bounds_studies_fail_before_model_construction(settings, monkeypatch):
    monkeypatch.setattr(study, "_load_model", lambda *_: pytest.fail("invalid studies cannot construct models"))
    with pytest.raises(ValueError):
        study.run_study(**settings)


def test_cli_never_overwrites_existing_evidence(tmp_path, monkeypatch):
    output = tmp_path / "historical.json"
    output.write_text('{"historical":true}', encoding="utf-8")
    monkeypatch.setattr(study, "run_study", lambda *_: pytest.fail("existing output must fail before diagnostics"))
    with pytest.raises(SystemExit) as stopped:
        study.main(["--output", str(output)])
    assert stopped.value.code == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"historical": True}
