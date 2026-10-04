"""Breadth probes retain real FP32 states, pairing and every negative result."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.check_decay_breadth as study
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
    return HybridLM(ModelConfig(ratio="1:3", vocab_size=32, d_model=16, n_layers=4,
                               head_dim=8, mamba_headdim=8, d_state=4, mlp_multiple_of=8)).eval()


def tokens(batch=1, length=9):
    values = torch.arange(batch * (length + 1)).reshape(batch, length + 1) % 32
    return values[:, :-1].contiguous(), values[:, 1:].contiguous()


def test_declared_cases_have_substantive_prefix_suffix_and_gradient_scope():
    assert [(item["batch_size"], item["length"], item["prefix_length"], item["gradient_check"]) for item in study.CASES] == [
        (1, 127, 63, False), (1, 128, 64, False), (1, 129, 64, False), (1, 257, 128, False), (2, 512, 128, True)]
    assert all(item["length"] - item["prefix_length"] > 1 for item in study.CASES)
    assert study.MINIMUM_FREE_BYTES == 8 * 1024**3


def test_input_preparation_freezes_natural_paired_batches_without_global_rng_consumption():
    data = np.arange(8192, dtype=np.uint16) % 32
    before = torch.get_rng_state().clone()
    first, second = study.prepare_inputs(data), study.prepare_inputs(data)
    assert torch.equal(before, torch.get_rng_state())
    assert [row[2] for row in first] == [row[2] for row in second]
    for x, y, identity in first:
        assert x.shape == (identity["batch_size"], identity["length"])
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert len(identity["windows"]) == identity["batch_size"]
        assert study.base.tensor_sha256(x) == identity["tokens_sha256"]
        assert study.base.tensor_sha256(y) == identity["targets_sha256"]
        for index, window in enumerate(identity["windows"]):
            start = window["start_token"]
            assert x[index].tolist() == data[start:start + identity["length"]].astype(np.int64).tolist()
            assert y[index].tolist() == data[start + 1:start + identity["length"] + 1].astype(np.int64).tolist()


def test_retained_memory_uses_fp32_tolerance_and_keeps_failure_that_bf16_would_accept():
    result = study.state_comparison({"layer_0.ssm": torch.full((2,), 4e-5)}, {"layer_0.ssm": torch.zeros(2)})
    assert result["tolerance"] == TOLERANCES["float32"]
    assert not result["passed"] and not result["fields"][0]["passed"]
    assert study.base.compare(torch.full((2,), 4e-5), torch.zeros(2), TOLERANCES["bfloat16"])["passed"]


@pytest.mark.parametrize("batch,gradients", [(1, False), (2, True)])
def test_original_and_treatment_full_cached_real_state_and_all_gradient_reports_are_distinct(batch, gradients):
    model, (x, y) = tiny_model().train(), tokens(batch)
    model.blocks[1].eval()
    modes = [module.training for module in model.modules()]
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    identity, tolerance = study.base.weight_sha256(model), deepcopy(TOLERANCES)
    with study.base.tf32_disabled():
        report = study.study_case(model, x, y, prefix_length=3, chunk_size=2, gradient_check=gradients)
    assert study.base.weight_sha256(model) == identity and TOLERANCES == tolerance
    assert [module.training for module in model.modules()] == modes
    assert all(block.mixer.scan_backend is backend for block, backend in
               zip([block for block in model.blocks if not block.is_attn], backends))
    assert all(parameter.grad is None for parameter in model.parameters())
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert [arm["treatment_id"] for arm in report["treatments"]] == ["original", study.decay.TREATMENT]
    for arm in report["treatments"]:
        cached = arm["fp32_cached"]
        assert cached["cache_dtype"] == "torch.float32" and cached["tolerance"] == TOLERANCES["float32"]
        assert cached["real_prefix"]["position"] == 3 and not cached["real_prefix"]["synthetic"]
        assert cached["real_prefix"]["unchanged_after_cloned_continuations"]
        assert all(item["nonzero_elements"] > 0 for item in cached["real_prefix"]["fields"] if item["field"].endswith("ssm"))
        assert cached["suffix_length"] == 6 and cached["chunked_schedule"] == [2, 2, 2]
        assert not cached["one_shot_vs_tokenwise_degenerate"]
        assert set(cached["final_positions"].values()) == {9}
        assert len(cached["internal_logits"]) == 9 and len(cached["original_fp32_anchor_logits"]) == 7
        assert len(cached["retained_state"]) == 5
        assert "all parameter gradients" in arm["unexecuted"] if not gradients else "all parameter gradients" not in arm["unexecuted"]
        if gradients:
            checks = arm["fp32_backend"]["gradients"]["checks"]
            assert [check["parameter"] for check in checks] == names
            assert len({check["parameter"] for check in checks}) == len(names)
            assert "lm_head.weight" not in names
        else:
            assert arm["fp32_backend"]["gradients"] is None
        for check in arm["common_original_fp32_anchor"].values():
            if check is not None:
                assert (check["gradients"] is not None) == gradients
                assert check["next_token_effects"]["scored_tokens"] == batch * 9
        assert arm["actual_paths"]["fp32_reference_full"] == ["torch.quadratic_ssd"]
        assert arm["actual_paths"]["fp32_full_tokenwise"] == ["torch.chunked_ssd"]
    assert report["treatments"][0]["fp32_cached"]["original_reference_stateful_anchor"] is None
    assert len(report["treatments"][1]["fp32_cached"]["original_reference_stateful_anchor"]) == 6


def test_experimental_override_and_model_state_restore_when_treatment_fails(monkeypatch):
    model = tiny_model().train()
    modes = [module.training for module in model.modules()]
    originals = study.decay.mamba.ssd, study.decay.mamba.ssd_stateful
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]

    def fail(*_args):
        raise RuntimeError("treatment failure")

    monkeypatch.setattr(study.decay, "ssd_decay_fp64", fail)
    with pytest.raises(RuntimeError, match="treatment failure"):
        study.study_case(model, *tokens(length=5), prefix_length=3, chunk_size=2)
    assert (study.decay.mamba.ssd, study.decay.mamba.ssd_stateful) == originals
    assert [module.training for module in model.modules()] == modes
    assert all(block.mixer.scan_backend is backend for block, backend in
               zip([block for block in model.blocks if not block.is_attn], backends))


def test_deadline_is_cooperative_and_retains_overshoot_without_reduced_case_retry():
    clock = {"value": 0.}
    limit = study.DiagnosticLimit(900, clock=lambda: clock["value"])
    limit.check()
    clock["value"] = 901.5
    with pytest.raises(RuntimeError, match="no reduced-shape fallback"):
        limit.check()
    assert limit.metadata()["overshoot_seconds"] == 1.5


@pytest.mark.parametrize("invalid", [0, -1, True, float("nan"), float("inf"), 901])
def test_invalid_allowance_fails_before_dataset_audit(invalid, monkeypatch):
    monkeypatch.setattr(study.base, "audit_inputs", lambda *_: pytest.fail("invalid allowance cannot audit"))
    with pytest.raises(ValueError):
        study.run_study(time_budget_seconds=invalid)


@pytest.mark.parametrize("field", ["cases", "time_budget_seconds", "minimum_large_case_free_bytes", "decay_sha256", "baseline_sha256", "tolerances"])
def test_mutated_declaration_rejects_before_dataset_or_model_work(tmp_path, monkeypatch, field):
    planned = json.loads((study.ROOT / "docs/research/decay-breadth-protocol-2026-10-04.json").read_text(encoding="utf-8"))
    planned[field] = {} if field in ("cases", "tolerances") else 1 if field.endswith("seconds") or field.endswith("bytes") else "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(planned), encoding="utf-8")
    monkeypatch.setattr(study.base, "audit_inputs", lambda *_: pytest.fail("invalid declaration cannot audit"))
    with pytest.raises(ValueError):
        study.run_study(declaration=path)


def test_incomplete_execution_preserves_completed_negative_cases_and_no_retry(tmp_path, monkeypatch):
    declaration = study.base.read_json(study.ROOT / "docs/research/decay-breadth-protocol-2026-10-04.json")
    baseline = study.base.read_json(study.ROOT / declaration["baseline_report"])
    prepared = study.prepare_inputs(np.arange(8192, dtype=np.uint16) % 32)
    prepared[3][2]["windows"] = [study.EXPECTED_INPUT]
    declaration["shared_inputs"] = [row[2] for row in prepared]
    declaration["device"] = "cpu"
    path = tmp_path / "cpu-declaration.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"readonly fixture")
    monkeypatch.setattr(study.base, "audit_inputs", lambda *_args: (
        dict.fromkeys(study.decay.RATIOS, SimpleNamespace(best_path=checkpoint)), baseline["protocol"]["data"], baseline["protocol"]["checkpoints"]))
    monkeypatch.setattr(study.base, "load_split", lambda *_args: np.zeros(1))
    monkeypatch.setattr(study, "prepare_inputs", lambda _data: prepared)
    monkeypatch.setattr(study.base, "load_variant_model", lambda *_args: tiny_model())
    weights = baseline["cases"][0]["weights_sha256"]
    monkeypatch.setattr(study.base, "weight_sha256", lambda _model: weights)
    calls = []

    def bounded_failure(_model, _x, _y, **_settings):
        calls.append(True)
        if len(calls) == 2:
            raise torch.OutOfMemoryError("injected OOM")
        return {"passed": False, "weights_sha256": weights, "treatments": []}

    monkeypatch.setattr(study, "study_case", bounded_failure)
    report = study.run_study("cpu", declaration=path)
    assert report["status"] == report["execution_status"] == "incomplete"
    assert len(calls) == 2 and len(report["cases"]) == 1 and not report["cases"][0]["passed"]
    assert "OOM" in report["reason"] and report["failed_case"]["case_id"] == prepared[1][2]["case_id"]
    assert report["protocol"]["shared_inputs"] == declaration["shared_inputs"]
    assert report["post_execution_integrity"]["measured_sources"] is True
    assert report["post_execution_integrity"]["declaration_and_prior_evidence"] is True
    assert report["post_execution_integrity"]["checkpoints"] is False  # intentional fake bytes
    assert checkpoint.read_bytes() == b"readonly fixture"


def test_existing_output_refused_before_execution_and_stdout_is_compact(tmp_path, monkeypatch, capsys):
    output = tmp_path / "old.json"
    output.write_text('{"old":true}', encoding="utf-8")
    monkeypatch.setattr(study, "run_study", lambda *_args, **_kwargs: pytest.fail("cannot overwrite evidence"))
    with pytest.raises(SystemExit) as stopped:
        study.main(["--output", str(output)])
    assert stopped.value.code == 2 and output.read_text(encoding="utf-8") == '{"old":true}'
    new = tmp_path / "new.json"
    report = {"status": "incomplete", "execution_status": "incomplete", "certified": False, "cases": [], "large_private_payload": [1] * 1000}
    monkeypatch.setattr(study, "run_study", lambda *_args, **_kwargs: report)
    assert study.main(["--output", str(new)]) == 2
    stdout = capsys.readouterr().out
    assert len(stdout) < 200 and "large_private_payload" not in stdout
    assert b"\r\n" not in new.read_bytes()
