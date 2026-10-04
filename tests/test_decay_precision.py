"""Temporary decay precision remains connected, reversible and separately gated."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

import scripts.study_decay_precision as study
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


def values():
    x = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1)
    return (x, torch.ones(1, 3, 1), torch.tensor([-torch.log(torch.tensor(2.)).item()]),
            torch.ones(1, 3, 1), torch.ones(1, 3, 1), torch.zeros(1))


def test_independent_half_decay_nonzero_state_solution_and_state_are_fp32():
    operands = values()
    initial = torch.full((1, 1, 1, 1), 2.)
    before = initial.clone()
    output, state = study.ssd_stateful_decay_fp64(*operands, initial)
    torch.testing.assert_close(output.flatten(), torch.tensor([2., 3., 4.5]))
    torch.testing.assert_close(state.flatten(), torch.tensor([4.5]))
    zero_output = study.ssd_decay_fp64(*operands)
    torch.testing.assert_close(zero_output.flatten(), torch.tensor([1., 2.5, 4.25]))
    assert output.dtype == state.dtype == zero_output.dtype == torch.float32
    assert torch.equal(initial, before)


def test_treatment_preserves_fp32_product_before_fp64_accumulation():
    generator = torch.Generator().manual_seed(41)
    dt = torch.rand(1, 17, 2, generator=generator)
    A = -torch.rand(2, generator=generator)
    local, carry, end = study._decay_coefficients(dt, A)
    cumulative = (dt * A).double().cumsum(1).transpose(1, 2)
    assert torch.equal(carry, cumulative.exp().float())
    assert torch.equal(end, (cumulative[..., -1, None] - cumulative).exp().float())
    causal = torch.tril(torch.ones(17, 17, dtype=torch.bool))
    expected = (cumulative[..., :, None] - cumulative[..., None, :]).masked_fill(~causal, float("-inf")).exp().float()
    assert torch.equal(local, expected)
    # This input distinguishes the declared treatment from also promoting dt*A.
    alternative = (dt.double() * A.double()).cumsum(1).transpose(1, 2).exp().float()
    assert not torch.equal(carry, alternative)


def test_every_scan_operand_and_nonzero_carried_memory_remain_connected_to_gradients():
    generator = torch.Generator().manual_seed(4)
    operands = (torch.randn(1, 5, 2, 3, generator=generator),
                torch.rand(1, 5, 2, generator=generator) + .1,
                -torch.rand(2, generator=generator),
                torch.randn(1, 5, 4, generator=generator),
                torch.randn(1, 5, 4, generator=generator),
                torch.randn(2, generator=generator))
    initial = torch.randn(1, 2, 3, 4, generator=generator)

    def differentiate(function):
        args = tuple(value.clone().requires_grad_() for value in (*operands, initial))
        output, state = function(*args)
        loss = output.square().mean() + .1 * state.square().mean()
        gradients = torch.autograd.grad(loss, args)
        return output, state, gradients

    actual_output, actual_state, actual_gradients = differentiate(study.ssd_stateful_decay_fp64)
    reference_output, reference_state, reference_gradients = differentiate(study.mamba.ssd_stateful)
    for actual, reference in zip((actual_output, actual_state, *actual_gradients),
                                 (reference_output, reference_state, *reference_gradients)):
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, reference, **TOLERANCES["float32"])


def test_chunk_continuation_keeps_autograd_across_the_real_initial_memory():
    operands = tuple(value.clone().requires_grad_() for value in values())
    initial = torch.full((1, 1, 1, 1), 2., requires_grad=True)
    first = tuple(value[:, :2] if value.ndim >= 3 else value for value in operands)
    second = tuple(value[:, 2:] if value.ndim >= 3 else value for value in operands)
    output1, memory = study.ssd_stateful_decay_fp64(*first, initial)
    output2, final = study.ssd_stateful_decay_fp64(*second, memory)
    torch.testing.assert_close(torch.cat([output1, output2], 1).flatten(), torch.tensor([2., 3., 4.5]))
    final.sum().backward()
    assert initial.grad.item() == pytest.approx(.125)
    assert operands[0].grad.flatten().tolist() == pytest.approx([.25, .5, 1.])


def test_module_and_capture_overrides_restore_even_with_nested_exception():
    original, stateful = study.mamba.ssd, study.mamba.ssd_stateful
    forward_backward, bf16 = study.base._forward_backward, study.base.bf16_cache_probe
    with pytest.raises(RuntimeError, match="injected"):
        with study.temporary_decay_treatment(), study.capture_passes():
            assert study.mamba.ssd is study.ssd_decay_fp64
            assert study.mamba.ssd_stateful is study.ssd_stateful_decay_fp64
            assert study.base._forward_backward is not forward_backward
            with study.temporary_decay_treatment():
                assert study.mamba.ssd is study.ssd_decay_fp64
            raise RuntimeError("injected")
    assert study.mamba.ssd is original and study.mamba.ssd_stateful is stateful
    assert study.base._forward_backward is forward_backward and study.base.bf16_cache_probe is bf16


def test_whole_model_treatment_comparisons_use_fixed_original_anchor_without_bf16_or_weight_changes():
    model = tiny_model().train()
    model.blocks[0].eval()
    modes = [module.training for module in model.modules()]
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    original = study.base.weight_sha256(model)
    tolerances = deepcopy(TOLERANCES)
    x, y = torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[2, 3, 4, 5, 6]])
    with study.base.tf32_disabled():
        report = study.study_model(model, x, y, prefix_length=3, chunk_size=2)
    assert [module.training for module in model.modules()] == modes
    assert all(block.mixer.scan_backend is backend for block, backend in
               zip([block for block in model.blocks if not block.is_attn], backends))
    assert study.base.weight_sha256(model) == original and TOLERANCES == tolerances
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [arm["treatment_id"] for arm in report["treatments"]] == ["original", study.TREATMENT]
    anchor = report["common_original_fp32_anchor"]
    expected_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert [entry["parameter"] for entry in anchor["gradients"]] == expected_names
    assert anchor["parameter_tensors"] == len(expected_names)
    assert "lm_head.weight" not in expected_names  # tied weights appear once
    for arm in report["treatments"]:
        assert arm["bf16_cached"] is None and not arm["include_bf16"]
        assert all(arm["actual_paths"][stage] is None for stage in study.BF16_STAGES)
        assert arm["unexecuted_stages"] == list(study.BF16_STAGES)
        assert arm["fp32_backend"]["tolerance"] == TOLERANCES["float32"]
        for comparison in arm["common_original_fp32_anchor"].values():
            if comparison is not None:
                assert comparison["tolerance"] == TOLERANCES["float32"]
                assert comparison["gradients"]["parameter_tensors"] == len(expected_names)
                assert comparison["next_token_effects"]["per_token"]["target_ids"] == y.flatten().tolist()
                assert "passed" not in comparison["next_token_effects"]
    assert report["treatments"][0]["common_original_fp32_anchor"]["reference"] is None
    assert report["treatments"][1]["scan_function_bindings"]["reference_scan"].endswith("ssd_decay_fp64")


def test_optional_bf16_retains_common_original_anchor_and_genuine_prefix_states():
    model = tiny_model()
    with study.base.tf32_disabled():
        report = study.study_model(model, torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[2, 3, 4, 5, 6]]),
                                   prefix_length=3, chunk_size=2, include_bf16=True, tokenwise=False)
    for arm in report["treatments"]:
        bf16 = arm["bf16_cached"]
        assert bf16["fp32_anchor"]["anchor_identity"]["logits_sha256"] == report["common_original_fp32_anchor"]["logits_sha256"]
        assert bf16["real_prefix"]["position"] == 3 and not bf16["real_prefix"]["synthetic"]
        assert bf16["real_prefix"]["unchanged_after_cloned_continuations"]
        assert bf16["unexecuted_stages"] == ["bf16_full_tokenwise"]
        assert arm["unexecuted_stages"] == ["bf16_full_tokenwise"]
        assert arm["actual_paths"]["bf16_full_tokenwise"] is None
        assert bf16["tolerance"] == TOLERANCES["bfloat16"]


def test_failure_during_treatment_restores_module_bindings_base_helpers_modes_and_backends(monkeypatch):
    model = tiny_model().train()
    modes = [module.training for module in model.modules()]
    originals = study.mamba.ssd, study.mamba.ssd_stateful, study.base._forward_backward, study.base.bf16_cache_probe
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    original_treatment = study.ssd_decay_fp64

    def fail(*_args):
        assert study.mamba.ssd is fail
        raise RuntimeError("treatment failure")

    monkeypatch.setattr(study, "ssd_decay_fp64", fail)
    with pytest.raises(RuntimeError, match="treatment failure"):
        study.study_model(model, torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]), prefix_length=2, chunk_size=2)
    assert (study.mamba.ssd, study.mamba.ssd_stateful, study.base._forward_backward, study.base.bf16_cache_probe) == originals
    assert [module.training for module in model.modules()] == modes
    assert all(block.mixer.scan_backend is backend for block, backend in
               zip([block for block in model.blocks if not block.is_attn], backends))
    assert original_treatment is not study.ssd_decay_fp64


def test_all_gradient_failures_against_common_anchor_remain_visible():
    logits, loss = torch.zeros(1, 2, 4), torch.tensor(1.)
    reference = {"first": torch.ones(2), "second": torch.ones(3)}
    actual = {"first": torch.zeros(2), "second": torch.full((3,), float("nan"))}
    result = study.compare_to_common_anchor((logits, loss, actual), (logits, loss, reference), torch.tensor([[1, 2]]))
    assert result["logits"]["passed"] and result["loss"]["passed"] and not result["passed"]
    assert result["gradients"]["failed_parameters"] == ["first", "second"]
    assert result["gradients"]["checks"][1]["max_absolute_error"] is None


def test_declared_public_baseline_isolation_and_input_identities_validate_read_only():
    path = study.ROOT / "docs/research/decay-precision-protocol-2026-10-04.json"
    before = path.read_bytes()
    planned = json.loads(before)
    result = study.validate_declaration(path, baseline_report=study.ROOT / planned["baseline_report"],
        baseline_declaration=study.ROOT / planned["baseline_declaration"], isolation_report=study.ROOT / planned["isolation_report"],
        device="cuda", include_bf16=False)
    assert result["sha256"] == hashlib.sha256(before).hexdigest() and path.read_bytes() == before


@pytest.mark.parametrize("field", ["tokens_sha256", "targets_sha256", "baseline_sha256", "isolation_sha256", "tolerances", "treatment_id", "include_bf16"])
def test_mutated_declaration_fails_before_any_model_or_data_work(tmp_path, monkeypatch, field):
    planned = json.loads((study.ROOT / "docs/research/decay-precision-protocol-2026-10-04.json").read_text(encoding="utf-8"))
    planned[field] = True if field == "include_bf16" else {} if field == "tolerances" else "0" * 64
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(planned), encoding="utf-8")
    monkeypatch.setattr(study.base, "audit_inputs", lambda *_: pytest.fail("cannot audit invalid protocol"))
    with pytest.raises(ValueError):
        study.run_study(declaration=path)


def test_existing_output_is_rejected_before_diagnostic_and_immutable_bytes_remain(tmp_path, monkeypatch):
    output = tmp_path / "historical.json"
    output.write_text('{"historical":true}', encoding="utf-8")
    monkeypatch.setattr(study, "run_study", lambda *_args, **_kwargs: pytest.fail("existing evidence blocks execution"))
    with pytest.raises(SystemExit) as stopped:
        study.main(["--output", str(output)])
    assert stopped.value.code == 2 and output.read_text(encoding="utf-8") == '{"historical":true}'
