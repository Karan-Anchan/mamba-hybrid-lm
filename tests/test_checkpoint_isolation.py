"""Operation isolation observes immutable weights and retains fixed-threshold drift."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

import scripts.isolate_checkpoint_numerics as isolation
from scripts.check_scan_backend import TOLERANCES
from src.eval.suite import VariantCheckpoint
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


def operands():
    x = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1)
    dt = torch.ones(1, 3, 1)
    return (x, dt, torch.zeros(1), torch.ones(1, 3, 1), torch.ones(1, 3, 1), torch.zeros(1))


def test_independent_fp64_oracle_has_known_nonzero_memory_solution_and_is_detached():
    values = tuple(value.clone().requires_grad_() for value in operands())
    initial = torch.full((1, 1, 1, 1), 2., requires_grad=True)
    before = initial.detach().clone()
    output, final = isolation.recurrence_fp64(values, initial)
    assert output.dtype == final.dtype == torch.float64
    assert not output.requires_grad and output.grad_fn is None and not final.requires_grad
    torch.testing.assert_close(output.flatten(), torch.tensor([3., 5., 8.], dtype=torch.float64))
    assert final.item() == 8.
    assert torch.equal(initial.detach(), before) and initial.grad is None
    assert all(value.grad is None for value in values)


def test_exact_coordinate_counts_distinguish_nonzero_drift_from_violations():
    actual = torch.tensor([[[0., 1e-5, 4e-5], [6e-5, 0., 0.]]])
    report = isolation.detailed_comparison(actual, torch.zeros_like(actual), coordinate_limit=1)
    assert report["elements"] == 6 and report["nonzero_error_count"] == 3 and report["violation_count"] == 2
    assert report["first_nonzero"]["coordinate"] == [0, 0, 1]
    assert report["first_violation"]["coordinate"] == [0, 0, 2]
    assert report["worst_tolerance_ratio"]["coordinate"] == [0, 1, 0]
    assert len(report["violation_samples"]) == 1
    assert report["tolerance"] == TOLERANCES["float32"] and not report["passed"]
    harmless = isolation.detailed_comparison(actual / 10, torch.zeros_like(actual))
    assert harmless["nonzero_error_count"] == 3 and harmless["violation_count"] == 0 and harmless["passed"]


def test_fp64_anchor_comparison_does_not_round_reference_back_to_fp32():
    actual, expected = torch.tensor([1. + 1e-10], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    precise = isolation.detailed_comparison(actual, expected, comparison_dtype="float64")
    assert precise["nonzero_error_count"] == 1 and precise["max_absolute_error"] > 0
    assert isolation.detailed_comparison(actual, expected)["nonzero_error_count"] == 0


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "coordinate_limit"])
def test_invalid_comparisons_fail_explicitly(kind):
    with pytest.raises(ValueError):
        if kind == "shape":
            isolation.detailed_comparison(torch.ones(2), torch.ones(3))
        elif kind == "nonfinite":
            isolation.detailed_comparison(torch.tensor([float("nan")]), torch.ones(1))
        else:
            isolation.detailed_comparison(torch.ones(1), torch.ones(1), coordinate_limit=0)


def test_trace_captures_are_detached_and_do_not_alias_model_inputs():
    trace = isolation.Trace()
    value = torch.tensor([[[1., 2.]]], requires_grad=True)
    trace.capture(0, "block_input", value, ["batch", "token", "feature"])
    saved = trace.stages[0]["value"]
    assert saved.grad_fn is None and not saved.requires_grad
    with torch.no_grad():
        value.add_(10)
    assert saved.tolist() == [[[1., 2.]]]


def test_full_model_trace_and_real_prefix_replays_leave_weights_modes_and_hooks_unchanged():
    model = tiny_model().train()
    model.blocks[1].eval()
    modes = [module.training for module in model.modules()]
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    weights = isolation.weight_sha256(model)
    tolerance = deepcopy(TOLERANCES)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with isolation.tf32_disabled():
        report = isolation.isolate_model(model, tokens, prefix_length=3, chunk_size=2, coordinate_limit=2)
    assert isolation.weight_sha256(model) == weights and TOLERANCES == tolerance
    assert [module.training for module in model.modules()] == modes
    assert all(block.mixer.scan_backend is backend for block, backend in
               zip([block for block in model.blocks if not block.is_attn], backends))
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    assert report["trace"][-1]["stage"] == "logits" and len(report["isolation_layers"]) <= 2
    for replay in report["frozen_replays"]:
        assert replay["real_prefix"]["position"] == 3 and not replay["real_prefix"]["synthetic"]
        assert replay["real_prefix"]["unchanged_after_cloned_continuation"]
        suffix = replay["real_prefix_continuation"]
        assert suffix["initial_state"]["sha256"] == replay["real_prefix"]["ssm_sha256"]
        assert suffix["initial_state"]["nonzero_elements"] > 0
        assert "quadratic" not in suffix["treatments"] and suffix["unexecuted"]
        assert replay["full_zero_state"]["initial_state"]["nonzero_elements"] == 0
        assert len(replay["full_zero_state"]["exploratory_decay_contrasts"]) == 2


def test_trace_wrappers_restore_after_injected_execution_error(monkeypatch):
    model = tiny_model()
    originals = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]

    def fail(_value):
        raise RuntimeError("injected")
    monkeypatch.setattr(model.lm_head, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        isolation.isolate_model(model, torch.tensor([[1, 2, 3]]), prefix_length=2, chunk_size=2)
    assert all(block.mixer.scan_backend is original for block, original in
               zip([block for block in model.blocks if not block.is_attn], originals))
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_frozen_routes_have_same_operands_and_independent_oracle_without_initial_mutation():
    values, initial = operands(), torch.full((1, 1, 1, 1), 2.)
    observed, final = isolation.scan_sequence(values, initial, 2)
    before = initial.clone()
    scan = {"operands": values, "initial": initial, "output": observed, "final": final}
    report = isolation.frozen_scan_probe(scan, "cpu", chunk_size=2, coordinate_limit=2, initial_kind="known nonzero test memory")
    assert torch.equal(initial, before)
    assert report["oracle"]["dtype"] == "torch.float64"
    assert set(report["operand_sha256"]) == set(isolation.OPERANDS)
    assert report["initial_state"]["kind"] == "known nonzero test memory"
    assert set(report["treatments"]) == {"stateful_one_shot", "chunked128", "tokenwise"}
    assert all(value["output_vs_fp64"]["passed"] and value["state_vs_fp64"]["passed"] for value in report["treatments"].values())
    assert report["replay_vs_observed_scan"]["max_absolute_error"] == 0


def test_saved_checkpoint_loading_and_isolation_are_read_only(tmp_path):
    model = tiny_model()
    path = tmp_path / "best.pt"
    torch.save({"model_config": asdict(model.cfg), "model": model.state_dict()}, path)
    raw = path.read_bytes()
    checkpoint = VariantCheckpoint("1:3", tmp_path, path, hashlib.sha256(raw).hexdigest(), "signature", 3, 2., asdict(model.cfg))
    loaded = isolation.load_variant_model(checkpoint, torch.device("cpu"))
    with isolation.tf32_disabled():
        report = isolation.isolate_model(loaded, torch.tensor([[1, 2, 3]]), prefix_length=2, chunk_size=2)
    assert path.read_bytes() == raw
    assert report["weights_sha256"] == isolation.weight_sha256(model)


@pytest.mark.parametrize("field", ["baseline_sha256", "tokens_sha256", "targets_sha256", "coordinate_limit"])
def test_new_declaration_must_bind_exact_baseline_and_input_hashes(tmp_path, field):
    original = isolation.ROOT / "docs/research/numerical-isolation-protocol-2026-10-04.json"
    declaration = json.loads(original.read_text(encoding="utf-8"))
    declaration[field] = 1 if field == "coordinate_limit" else "0" * 64
    path = tmp_path / "changed-declaration.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")
    with pytest.raises(ValueError):
        isolation.validate_isolation_declaration(path,
            baseline_report=isolation.ROOT / declaration["baseline_report"],
            declaration=isolation.ROOT / declaration["baseline_declaration"], device="cuda", coordinate_limit=8)


def test_baseline_identity_validation_is_read_only_and_accepts_exact_record():
    declaration = isolation.ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json"
    baseline = isolation.ROOT / "docs/research/checks/trained-numerics-2026-10-04/natural-257-seed-2027.json"
    before = baseline.read_bytes()
    _, record = isolation._baseline_identities(declaration, baseline)
    assert record["protocol"]["shared_inputs"] == [isolation.EXPECTED_INPUT]
    assert baseline.read_bytes() == before


def test_exact_new_declaration_is_accepted_with_unchanged_full_tolerance_registry():
    declaration = isolation.ROOT / "docs/research/numerical-isolation-protocol-2026-10-04.json"
    metadata = isolation.validate_isolation_declaration(declaration,
        baseline_report=isolation.ROOT / "docs/research/checks/trained-numerics-2026-10-04/natural-257-seed-2027.json",
        declaration=isolation.ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json",
        device="cuda", coordinate_limit=8)
    assert metadata["sha256"] == hashlib.sha256(declaration.read_bytes()).hexdigest()


def test_existing_output_is_rejected_before_any_model_work(tmp_path, monkeypatch):
    output = tmp_path / "historical.json"
    output.write_text('{"historical":true}', encoding="utf-8")
    monkeypatch.setattr(isolation, "run_study", lambda *_args, **_kwargs: pytest.fail("no model work allowed"))
    with pytest.raises(SystemExit) as stopped:
        isolation.main(["--output", str(output)])
    assert stopped.value.code == 2 and output.read_text(encoding="utf-8") == '{"historical":true}'
