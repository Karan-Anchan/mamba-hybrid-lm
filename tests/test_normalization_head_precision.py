"""Portable tests for scoped numerical factors, independent oracles and evidence."""
import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import scripts.study_normalization_head_precision as study
from src.model.config import ModelConfig
from src.model.lm import HybridLM
from src.model.norm import RMSNorm


@pytest.fixture(autouse=True)
def modest_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_model(ratio="1:3", layers=4):
    torch.manual_seed(19)
    return HybridLM(ModelConfig(ratio=ratio, vocab_size=32, d_model=16, n_layers=layers,
                               head_dim=8, mamba_headdim=8, d_state=4, mlp_multiple_of=8)).eval()


def tokens(batch=1, length=9):
    window = torch.arange(batch * (length + 1)).reshape(batch, length + 1) % 32
    return window[:, :-1].contiguous(), window[:, 1:].contiguous()


@pytest.mark.parametrize("ratio,count", [("1:15", 48), ("1:3", 45)])
def test_every_declared_normalization_site_including_gated_and_final(ratio, count):
    model = tiny_model(ratio, 16)
    sites = study.norm_sites(model)
    assert len(sites) == count
    assert len({name for name, *_ in sites}) == count
    assert sites[-1][0] == "norm_f"
    assert sum("mixer.norm" in name for name, *_ in sites) == sum(not block.is_attn for block in model.blocks)
    assert all(module is dict(model.named_modules())[name] for name, module, *_ in sites)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_n0_historical_arithmetic_and_n1_exact_coefficient_scope(dtype):
    module = RMSNorm(3, eps=.25)
    module.weight.data.copy_(torch.tensor([.5, -2., 3.]))
    x = torch.tensor([[[1.234567, 3.1415927, 11.111111]]], dtype=dtype)
    out0, squared, mean0, coeff0 = study.norm_arithmetic(module, x, n1=False)
    out1, squared1, mean1, coeff1 = study.norm_arithmetic(module, x, n1=True)
    assert torch.equal(out0, RMSNorm.forward(module, x))
    assert torch.equal(squared, x.float().square()) and torch.equal(squared1, squared)
    assert torch.equal(mean1, squared.double().mean(-1, keepdim=True))
    assert mean0.dtype == coeff0.dtype == coeff1.dtype == torch.float32 and mean1.dtype == torch.float64
    expected = ((x.float() * torch.rsqrt(squared.double().mean(-1, keepdim=True) + module.eps).float()) * module.weight.float()).to(dtype)
    assert torch.equal(out1, expected)
    if dtype == torch.float32:
        assert not torch.equal(squared.double(), x.float().double().square())


def test_coefficient_and_head_treatments_keep_gradients_and_tied_weight():
    model = tiny_model()
    x = torch.randn(2, 3, 16, requires_grad=True)
    before = study.tied_weight_snapshot(model)
    with study.temporary_cell(model, "N1H1"):
        output = model.lm_head(model.norm_f(x))
        output.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert model.norm_f.weight.grad is not None and torch.isfinite(model.norm_f.weight.grad).all()
    assert model.embed.weight.grad is model.lm_head.weight.grad and model.embed.weight.grad.abs().sum() > 0
    assert study.tied_weight_report(before, study.tied_weight_snapshot(model))["passed"]


def test_nested_override_exception_restores_instances_parameters_and_source_functions():
    model = tiny_model()
    model.norm_f.forward = lambda x: x + 123
    previous = model.norm_f.forward
    weights = study.base.weight_sha256(model)
    snapshots = study.tied_weight_snapshot(model)
    historical_norm, historical_linear = RMSNorm.forward, torch.nn.Linear.forward
    with pytest.raises(RuntimeError, match="injected"):
        with study.temporary_cell(model, "N1H1"):
            outer_norm, outer_head = model.norm_f.forward, model.lm_head.forward
            with study.temporary_cell(model, "N0H0"):
                assert model.norm_f.forward.__func__ is historical_norm
                assert model.lm_head.forward.__func__ is historical_linear
            assert model.norm_f.forward is outer_norm and model.lm_head.forward is outer_head
            raise RuntimeError("injected failure")
    assert model.norm_f.forward is previous and "forward" not in model.lm_head.__dict__
    assert RMSNorm.forward is historical_norm and torch.nn.Linear.forward is historical_linear
    assert study.base.weight_sha256(model) == weights
    assert snapshots == study.tied_weight_snapshot(model)


def test_override_partial_installation_failure_removes_earlier_norm_hooks():
    model = tiny_model()
    model.lm_head = torch.nn.Identity()
    with pytest.raises(ValueError, match="linear"):
        with study.temporary_cell(model, "N1H1"):
            pytest.fail("invalid head must reject")
    assert all("forward" not in module.__dict__ for _, module, *_ in study.norm_sites(model))
    with pytest.raises(ValueError, match="unknown"):
        with study.temporary_cell(model, "N2H1"):
            pass


def test_independent_numpy_norm_and_head_oracles_known_solution_no_grad(monkeypatch):
    x = torch.tensor([[[3., 4.]]], requires_grad=True)
    weight = torch.tensor([2., -1.], requires_grad=True)
    monkeypatch.setattr(RMSNorm, "forward", lambda *a: pytest.fail("shared norm implementation"))
    monkeypatch.setattr(F, "linear", lambda *a: pytest.fail("shared linear implementation"))
    out, squared, mean, coefficient = study.norm_oracle_fp64(x, weight, .25)
    np.testing.assert_allclose(out.numpy(), [[[6 / np.sqrt(12.75), -4 / np.sqrt(12.75)]]], rtol=1e-15)
    assert squared.tolist() == [[[9., 16.]]] and mean.item() == 12.5
    assert coefficient.item() == 1 / np.sqrt(12.75)
    head_weight = torch.tensor([[2., -1.], [1., 3.]], requires_grad=True)
    bias = torch.tensor([.5, -2.], requires_grad=True)
    head = study.head_oracle_fp64(x, head_weight, bias)
    assert head.tolist() == [[[2.5, 13.]]]
    assert all(not value.requires_grad and value.device.type == "cpu" and value.dtype == torch.float64 for value in (out, squared, mean, coefficient, head))
    assert x.grad is weight.grad is head_weight.grad is bias.grad is None


@pytest.mark.parametrize("which", ["norm_input", "norm_weight", "head_input", "head_weight", "head_bias"])
def test_oracles_reject_nonfinite_operands(which):
    x, weight, bias = torch.ones(1, 2, 3), torch.ones(4, 3), torch.ones(4)
    if which == "norm_input":
        x[0, 0, 0] = float("nan")
    elif which == "norm_weight":
        weight[0, 0] = float("inf")
    elif which == "head_input":
        x[0, 0, 0] = float("inf")
    elif which == "head_weight":
        weight[0, 0] = float("nan")
    else:
        bias[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        if which.startswith("norm"):
            study.norm_oracle_fp64(x, weight[0], 1e-5)
        else:
            study.head_oracle_fp64(x, weight, bias)


def test_frozen_norm_layout_all_sites_stats_dtypes_and_immutable_inputs():
    module = RMSNorm(4)
    backing = torch.randn(2, 7, 8)
    inputs = backing[..., 1::2]
    before = backing.clone()
    observed = module(inputs)
    observation = {"input_layout": study.layout(inputs), "output_layout": study.layout(observed)}
    report = study.frozen_norm_probe(inputs.detach().cpu(), observed, module, observation, lambda: None)
    assert report["replay_layout"] == observation["input_layout"] and report["replay_layout"]["storage_offset"] == 1
    assert set(report["output_checks"]) == set(study.NORM_OUTPUT_CHECKS)
    assert set(report["statistic_checks"]) == set(study.NORM_STAT_CHECKS)
    assert report["output_checks"]["original_full_vs_observed"]["nonzero_error_count"] == 0
    for name in ("n1_full", "n1_tokenwise"):
        assert report["statistics"][name]["mean_square"]["dtype"] == "torch.float64"
        assert report["statistics"][name]["coefficient"]["dtype"] == "torch.float32"
        assert report["statistics"][name]["mean_square"]["shape"] == [2, 7, 1]
    assert torch.equal(backing, before) and report["input_unchanged"]
    json.dumps(report, allow_nan=False)


def test_frozen_head_independent_h0_h1_shape_contrasts_and_no_mutation():
    module = torch.nn.Linear(4, 11)
    inputs = torch.randn(2, 7, 8)[..., 1::2].requires_grad_()
    observed = module(inputs)
    before = inputs.detach().clone()
    report = study.frozen_head_probe(inputs, observed, module, {"input_layout": study.layout(inputs), "output_layout": study.layout(observed)}, lambda: None)
    assert set(report["checks"]) == set(study.HEAD_CHECKS)
    assert report["checks"]["h0_full_vs_observed"]["nonzero_error_count"] == 0
    assert report["checks"]["h1_full_vs_fp64"]["reference_dtype"] == "torch.float64"
    assert report["checks"]["h1_tokenwise_vs_h1_full"]["shape"] == [2, 7, 11]
    assert report["oracle_output"]["dtype"] == "torch.float64" and "NumPy" in report["oracle"]
    assert report["replay_layout"] == report["input_layout"] and torch.equal(inputs.detach(), before)
    assert inputs.grad is None and module.weight.grad is None and module.bias.grad is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("batch,length,prefix", [(1, 9, 4), (2, 7, 3)])
def test_complete_four_cell_case_real_prefix_full_axes_and_controls_restore(batch, length, prefix):
    model, (x, y) = tiny_model(), tokens(batch, length)
    model.train()
    model.blocks[1].eval()
    model.configure_scan_backend("torch_chunked", 2)
    modes = [module.training for module in model.modules()]
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    functions = F.conv1d, F.scaled_dot_product_attention
    before = study.base.weight_sha256(model)
    report = study.study_case(model, x, y, prefix_length=prefix)
    assert report["execution_status"] == "completed", report.get("reason")
    assert [row["cell_id"] for row in report["cells"]] == list(study.CELLS)
    assert len(report["frozen_norms"]) == len(study.norm_sites(model))
    assert report["frozen_head"]["input"]["shape"] == [batch, length, 16]
    assert report["original_anchor"]["repeat"]["exact_equal"] and report["attribution_control_stable"]
    for row in report["cells"]:
        assert set(row["stage_comparisons"]) == set(study.PAIRINGS)
        assert len(row["whole_model_comparisons"]) == 11
        assert set(row["retained_state"]) == {"own", "original", "decay_baseline"}
        assert set(row["stateful_full_retained_state"]) == {"original", "decay_baseline"}
        prefix_report = row["actual_prefix"]
        assert prefix_report["prefix_position"] == prefix and prefix_report["prefix_unchanged"]
        assert prefix_report["suffix_length"] == length - prefix and not prefix_report["synthetic"]
        assert set(prefix_report["suffix_comparisons"]) == {"one_shot", "chunked128", "tokenwise"}
        assert all(position == length for position in prefix_report["final_positions"].values())
        assert row["next_token_effects"]["full_vs_original"]["scored_tokens"] == batch * length
        assert not row["backward_executed"] and not row["BF16_executed"]
        assert row["whole_model_comparisons"]["full_vs_original"]["shape"] == [batch, length, 32]
    assert [module.training for module in model.modules()] == modes
    assert [block.mixer.scan_backend for block in model.blocks if not block.is_attn] == backends
    assert (F.conv1d, F.scaled_dot_product_attention) == functions
    assert before == study.base.weight_sha256(model) and report["post_model_weights_unchanged"]
    assert report["tied_weight_integrity"]["passed"] and report["runtime_policy"]["restoration_exact"]
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", [RuntimeError("allowance injected"), MemoryError("CPU allocation injected")])
def test_frozen_failure_preserves_completed_cell_and_restores_all_scopes(monkeypatch, failure):
    model, (x, y) = tiny_model(), tokens()
    policy = study.runtime_policy()
    weights = study.base.weight_sha256(model)
    monkeypatch.setattr(study, "frozen_norm_probe", lambda *a: (_ for _ in ()).throw(failure))
    report = study.study_case(model, x, y, prefix_length=4)
    assert report["execution_status"] == "incomplete" and report["failed_cell"] == "N0H0"
    assert len(report["cells"]) == 1 and report["cells"][0]["execution_status"] == "completed"
    assert not report["all_recorded_comparisons_passed"] and report["frozen_head"] is None
    assert type(failure).__name__ in report["reason"]
    assert study.runtime_policy() == policy and study.base.weight_sha256(model) == weights
    assert all("forward" not in module.__dict__ for _, module, *_ in study.norm_sites(model))
    assert "forward" not in model.lm_head.__dict__


def test_no_control_stability_promotion_when_repeat_differs(monkeypatch):
    original = study.prior.collect_route
    calls = 0
    def altered(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            result["logits"] = result["logits"] + 1e-6
        return result
    monkeypatch.setattr(study.prior, "collect_route", altered)
    model, (x, y) = tiny_model(), tokens()
    report = study.study_case(model, x, y, prefix_length=4)
    assert report["execution_status"] == "completed"
    assert not report["attribution_control_stable"] and not report["all_recorded_comparisons_passed"]
    assert report["original_anchor"]["repeat"]["comparison"]["passed"]


def test_exact_prior_control_preserves_failure_and_detects_small_changed_statistics():
    values = torch.tensor([[[1., -1.]]])
    current = {name: study.cmp(values + .01, values) for name in ("full_vs_original", "tokenwise_vs_own_full", "tokenwise_vs_original", "stateful_full_vs_own_full")}
    old_names = ("candidate_full_vs_original_stateless", "candidate_tokenwise_vs_candidate_full", "candidate_tokenwise_vs_original_stateless", "candidate_stateful_full_vs_candidate_stateless")
    old = {"whole_model_comparisons": dict(zip(old_names, copy.deepcopy(list(current.values()))))}
    report = study.prior_reproduction(current, old)
    assert report["exact_equal"] and not current["tokenwise_vs_original"]["passed"]
    old["whole_model_comparisons"][old_names[0]]["max_absolute_error"] += 1e-12
    assert not study.prior_reproduction(current, old)["exact_equal"]


def test_stage_compaction_preserves_failed_samples_and_pair_markers():
    zeros = torch.zeros(1, 2, 3)
    expected = {"order": [(0, "norm1_output"), (0, "mixer_output")], "tensors": {(0, "norm1_output"): zeros, (0, "mixer_output"): zeros}}
    actual = {"order": expected["order"], "tensors": {(0, "norm1_output"): zeros + 1e-6, (0, "mixer_output"): zeros + .01}}
    detailed = study.prior.stage_comparisons(actual, expected)
    compact = study.compact_trace(actual, expected, lambda: None)
    assert compact["first_nonzero"] == detailed["first_nonzero"] and compact["first_violation"] == detailed["first_violation"]
    assert compact["stages"][0]["sample_policy"] == "passing_compact" and "worst_tolerance_ratio" not in compact["stages"][0]
    assert compact["stages"][0]["first_nonzero_coordinate"] == [0, 0, 0]
    for key, value in detailed["stages"][1].items():
        assert compact["stages"][1][key] == value


def test_layout_hooks_removed_when_capture_raises():
    model = tiny_model()
    with pytest.raises(RuntimeError, match="injected"):
        with study.capture_norm_layouts(model, {}):
            raise RuntimeError("injected")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_runner_allowance_starts_before_declaration_audit(monkeypatch, tmp_path):
    ticks = [0.]
    original_limit = study.breadth.DiagnosticLimit
    monkeypatch.setattr(study.breadth, "DiagnosticLimit", lambda seconds: original_limit(seconds, clock=lambda: ticks[0]))
    def audit(*args, **kwargs):
        ticks[0] = 901.
        return {}, {}
    monkeypatch.setattr(study, "audit_declaration", audit)
    monkeypatch.setattr(study.base, "audit_inputs", lambda *a: pytest.fail("expired preflight proceeded to corpus"))
    with pytest.raises(RuntimeError, match="allowance"):
        study.run_study("cpu", output=tmp_path / "new.json")
    assert not (tmp_path / "new.json").exists()


def test_main_memory_failure_immutable_partial_and_no_overwrite(monkeypatch, tmp_path, capsys):
    path = tmp_path / "incomplete.json"
    monkeypatch.setattr(study, "run_study", lambda *a, **k: (_ for _ in ()).throw(MemoryError("private corpus allocation")))
    assert study.main(["--device", "cpu", "--declaration", str(tmp_path / "missing.json"), "--output", str(path)]) == 2
    report = json.loads(path.read_text())
    assert report["execution_status"] == "incomplete" and report["cases"] == [] and not report["certified"]
    assert "MemoryError" in report["reason"]
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        study.main(["--device", "cpu", "--output", str(path)])
    assert path.read_bytes() == before
    assert "completed_cases" in capsys.readouterr().out


def test_main_declared_output_mismatch_rejected_before_execution(monkeypatch, tmp_path):
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"output": "different-output.json"}))
    monkeypatch.setattr(study, "run_study", lambda *a, **k: pytest.fail("mismatch reached measurement"))
    with pytest.raises(SystemExit):
        study.main(["--device", "cpu", "--declaration", str(declaration), "--output", str(tmp_path / "new.json")])
    assert not (tmp_path / "new.json").exists()


@pytest.fixture
def declaration_files(tmp_path):
    # The public immutable report supplies metadata only. No ignored corpus or
    # checkpoint file is loaded, and the local fixture retains a tiny case list.
    historical = study.base.read_json(study.DEFAULT_PRIOR_REPORT)
    report = {key: historical[key] for key in ("kind", "execution_status", "certified", "protocol", "protocol_sha256")}
    report["cases"] = [{"ratio": row["ratio"], "case_id": row["case_id"]} for row in historical["cases"]]
    report_path, declaration_path, output = tmp_path / "prior.json", tmp_path / "declaration.json", tmp_path / "future.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    declaration = study.declaration_template(device="cpu", prior_report=report_path, output=output)
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    kwargs = {"device": "cpu", "prior_declaration": study.DEFAULT_PRIOR_DECLARATION, "prior_report": report_path, "output": output}
    return declaration_path, kwargs, declaration, report_path


def test_declaration_audit_public_only_validates_exact_source_input_runtime_scope(declaration_files):
    path, kwargs, expected, _ = declaration_files
    planned, previous = study.audit_declaration(path, **kwargs)
    assert planned == expected and len(previous["cases"]) == 6
    assert set(planned["report_schema"]["frozen_head_checks"]) == set(study.HEAD_CHECKS)
    assert planned["observed_inference_policy"]["deterministic_algorithms"] is False


@pytest.mark.parametrize("field", ["cells", "source_sha256", "shared_inputs", "tolerances", "observed_inference_policy", "output", "prior_report"])
def test_preflight_declared_scope_tampering_rejected_without_private_data(declaration_files, field):
    path, kwargs, planned, _ = declaration_files
    planned[field] = {} if isinstance(planned[field], dict) else [] if isinstance(planned[field], list) else "altered-output"
    path.write_text(json.dumps(planned), encoding="utf-8")
    with pytest.raises(ValueError, match="declaration differs"):
        study.audit_declaration(path, **kwargs)


def test_prior_matrix_duplicate_rejected_even_when_parent_hashes_recomputed(declaration_files):
    path, kwargs, _, report_path = declaration_files
    previous = study.base.read_json(report_path)
    previous["cases"][1] = copy.deepcopy(previous["cases"][0])
    report_path.write_text(json.dumps(previous), encoding="utf-8")
    path.write_text(json.dumps(study.declaration_template(**{key: kwargs[key] for key in ("device", "prior_declaration", "prior_report", "output")})), encoding="utf-8")
    with pytest.raises(ValueError, match="six-case coverage"):
        study.audit_declaration(path, **kwargs)


def test_final_actual_byte_integrity_detects_modified_data_and_preserves_all_checks(tmp_path):
    data, source = tmp_path / "val.bin", tmp_path / "source.py"
    data.write_bytes(b"immutable natural tokens")
    source.write_bytes(b"source")
    expected = study.base.file_sha256(data)
    data.write_bytes(b"modified natural tokens")
    calls = []
    integrity, files, reason = study.audit_final_identities((("data", [(data, expected)]), ("sources", [(source, study.base.file_sha256(source))])), lambda: calls.append(True))
    assert integrity == {"data": False, "sources": True} and "identity audit failed" in reason
    assert len(files) == 2 and not files[0]["passed"] and files[1]["passed"] and len(calls) == 6
    assert all("<home>" not in row["path"] and row["path"] == ("val.bin" if row["group"] == "data" else "source.py") for row in files)


def test_final_file_hash_inflight_overshoot_stops_next_file_and_unexecuted_groups(monkeypatch, tmp_path):
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    digest = study.base.file_sha256(first)
    ticks, reads = [0.], []
    limit = study.breadth.DiagnosticLimit(900, clock=lambda: ticks[0])
    original = study.base.file_sha256
    def slow_hash(path):
        reads.append(path)
        ticks[0] = 901.
        return original(path)
    monkeypatch.setattr(study.base, "file_sha256", slow_hash)
    integrity, files, reason = study.audit_final_identities((("first", [(first, digest), (second, digest)]), ("sources", [(second, digest)])), limit.check)
    assert reads == [first] and len(files) == 1 and files[0]["passed"]
    assert integrity == {"first": None, "sources": None} and "allowance" in reason
