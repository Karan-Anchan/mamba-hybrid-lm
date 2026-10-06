"""CPU contracts for isolated projection factors, real states and evidence chains."""
import copy
import gzip
import hashlib
import json
from types import MethodType

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import scripts.study_projection_precision as study
from src.model.config import ModelConfig
from src.model.lm import HybridLM


@pytest.fixture(autouse=True)
def modest_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny_model():
    torch.manual_seed(31)
    return HybridLM(ModelConfig(ratio="1:3", n_layers=4, vocab_size=32, d_model=16,
        head_dim=8, mamba_headdim=8, d_state=4, mlp_multiple_of=8)).eval()


def tokens(batch=1, length=9):
    window = torch.arange(batch * (length + 1)).reshape(batch, length + 1) % 32
    return window[:, :-1].contiguous(), window[:, 1:].contiguous()


@pytest.mark.parametrize("bias", [False, True])
def test_projection_scope_connects_input_weight_bias_gradients_and_preserves_values(bias):
    module = torch.nn.Linear(4, 7, bias=bias)
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    weight = module.weight.detach().clone()
    expected = F.linear(inputs.double(), module.weight.double(), None if not bias else module.bias.double()).float()
    actual = study.projection_arithmetic(module, inputs, p1=True)
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    actual.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
    if bias:
        assert module.bias.grad is not None and torch.isfinite(module.bias.grad).all()
    assert torch.equal(module.weight, weight)
    assert torch.equal(study.projection_arithmetic(module, inputs, p1=False), module(inputs))


@pytest.mark.parametrize("inputs", [torch.zeros(2, 3, 4, dtype=torch.float64), torch.zeros(2, 3, 5), torch.zeros(2, 0, 4), torch.zeros(3, 4)])
def test_projection_rejects_unregistered_shape_or_dtype(inputs):
    with pytest.raises(ValueError, match="FP32"):
        study.projection_arithmetic(torch.nn.Linear(4, 7), inputs, p1=True)


def test_instance_only_nested_zero_and_exception_restoration_keep_ties_and_other_linears():
    model = tiny_model()
    sites = study.projection_sites(model)
    first = sites[0][1]
    custom = MethodType(lambda self, value: F.linear(value, self.weight, self.bias), first)
    first.forward = custom
    before = study.base.weight_sha256(model)
    ties = study.norm.tied_weight_snapshot(model)
    other = model.blocks[3].mixer.qkv.forward.__func__
    with pytest.raises(RuntimeError, match="injected"):
        with study.temporary_projection(model, "P1"):
            assert all(module.forward.__func__ is study.p1_forward for _, module, _ in sites)
            assert model.blocks[3].mixer.qkv.forward.__func__ is other
            with study.temporary_projection(model, "P0"):
                assert all(module.forward.__func__ is torch.nn.Linear.forward for _, module, _ in sites)
            assert first.forward.__func__ is study.p1_forward
            raise RuntimeError("injected")
    assert first.forward is custom
    assert all("forward" not in module.__dict__ for _, module, _ in sites[1:])
    assert study.base.weight_sha256(model) == before and study.norm.tied_weight_snapshot(model) == ties
    with pytest.raises(ValueError):
        with study.temporary_projection(model, "unknown"):
            pass


def test_numpy_oracle_is_independent_of_torch_linear_and_preserves_cancellation(monkeypatch):
    inputs = torch.tensor([[[16777216., 1., -16777216., 2.]]])
    weights = torch.tensor([[1., 1., 1., 1.], [0., 3., 0., -2.]])
    bias = torch.tensor([.5, 2.])
    monkeypatch.setattr(F, "linear", lambda *a: pytest.fail("NumPy oracle called Torch F.linear"))
    result = study.norm.head_oracle_fp64(inputs, weights, bias)
    assert result.dtype == torch.float64
    assert torch.equal(result, torch.tensor([[[3.5, 1.]]], dtype=torch.float64))


def test_frozen_noncontiguous_full_call_reconstructs_layout_and_detaches_outputs():
    module = torch.nn.Linear(4, 11)
    inputs = torch.randn(2, 7, 8)[..., 1::2].requires_grad_()
    observed = module(inputs)
    calls = [{"position": 0, "length": 7, "input_layout": study.norm.layout(inputs), "output_layout": study.norm.layout(observed)}]
    before = inputs.detach().clone()
    report = study.frozen_projection_probe(inputs, observed, module, calls)
    assert report["replay_layout"] == calls[0]["input_layout"]
    assert report["checks"]["observed_shape_replay_vs_observed"]["nonzero_error_count"] == 0
    assert report["oracle_output"]["dtype"] == "torch.float64"
    assert set(report["checks"]) == set(study.PROJECTION_CHECKS)
    assert torch.equal(inputs.detach(), before) and inputs.grad is None and module.weight.grad is None
    json.dumps(report, allow_nan=False)


def test_actual_multi_call_projection_has_explicit_layout_contrast_and_exact_replay():
    module = torch.nn.Linear(4, 6)
    values = torch.randn(2, 7, 4)
    capture = study.ProjectionCapture([("test.in_proj", module, 0)], position=128)
    outputs = []
    with capture.scoped():
        for start, length in ((0, 3), (3, 4)):
            capture.position = 128 + start
            outputs.append(module(values[:, start:start + length]))
    captured = capture.finish(7, offset=128)["test.in_proj"]
    report = study.frozen_projection_probe(captured["inputs"], captured["observed"], module, captured["calls"], offset=128)
    assert report["position_offset"] == 128 and [call["position"] for call in report["actual_calls"]] == [128, 131]
    assert "combined contiguous" in report["layout_contrast"]
    assert report["checks"]["observed_shape_replay_vs_observed"]["nonzero_error_count"] == 0
    assert torch.equal(captured["observed"], torch.cat(outputs, 1))
    assert not module._forward_hooks and not module._forward_pre_hooks


def test_missing_projection_positions_nonfinite_oracle_and_hook_errors_fail_cleanly():
    module = torch.nn.Linear(4, 6)
    capture = study.ProjectionCapture([("test", module, 0)], position=1)
    with capture.scoped():
        module(torch.ones(1, 2, 4))
    with pytest.raises(RuntimeError, match="coverage"):
        capture.finish(2)
    with pytest.raises(ValueError, match="finite"):
        study.norm.head_oracle_fp64(torch.full((1, 1, 4), float("nan")), module.weight, module.bias)
    with pytest.raises(RuntimeError, match="injected"):
        with capture.scoped():
            raise RuntimeError("injected")
    assert not module._forward_hooks and not module._forward_pre_hooks


@pytest.mark.parametrize("batch,length,prefix", [(1, 9, 4), (2, 7, 3)])
def test_complete_case_all_direct_anchors_true_prefix_capture_and_restoration(batch, length, prefix):
    model, (x, y) = tiny_model(), tokens(batch, length)
    model.train()
    model.blocks[1].eval()
    model.configure_scan_backend("torch_chunked", 2)
    modes = [module.training for module in model.modules()]
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    original_functions = F.conv1d, F.scaled_dot_product_attention
    weights, rng, ties = study.base.weight_sha256(model), study.rng_identity(x.device), study.norm.tied_weight_snapshot(model)
    report = study.study_case(model, x, y, prefix_length=prefix)
    assert report["execution_status"] == "completed", report.get("reason")
    assert [row["cell_id"] for row in report["cells"]] == list(study.CELLS)
    assert len(report["frozen_projections"]) == len(study.projection_sites(model))
    assert report["original_anchor"]["repeat"]["exact_equal"] and report["attribution_control_stable"]
    for row in report["cells"]:
        assert tuple(row["whole_model_comparisons"]) == study.ENDPOINTS
        assert tuple(row["loss_comparisons"]) == study.LOSS_ENDPOINTS
        assert tuple(row["stage_comparisons"]) == study.PAIRINGS
        assert tuple(row["retained_state"]) == study.ANCHORS
        assert set(row["stateful_full_retained_state"]) == set(study.ANCHORS) - {"own"}
        continuation = row["actual_prefix"]
        assert continuation["execution_status"] == "completed" and continuation["prefix_unchanged"] and not continuation["synthetic"]
        assert all(end == length for end in continuation["final_positions"].values())
        assert len(continuation["frozen_projections"]) == 6
        for route in ("one_shot", "chunked128", "tokenwise"):
            assert tuple(continuation["suffix_comparisons"][route]) == study.ANCHORS
            assert tuple(continuation["retained_state"][route]) == study.ANCHORS
        for probe in continuation["frozen_projections"]:
            assert probe["prefix_fields"] == continuation["prefix_fields"]
            assert probe["prefix_position"] == prefix and probe["final_position"] == length
            assert probe["frozen_arithmetic"]["position_offset"] == prefix
            assert probe["upstream_input_vs_own_full"]["shape"] == [batch, length - prefix, 16]
        assert row["next_token_effects"]["full_vs_original"]["scored_tokens"] == batch * length
    assert [module.training for module in model.modules()] == modes
    assert [block.mixer.scan_backend for block in model.blocks if not block.is_attn] == backends
    assert (F.conv1d, F.scaled_dot_product_attention) == original_functions
    assert study.base.weight_sha256(model) == weights and study.rng_identity(x.device) == rng and study.norm.tied_weight_snapshot(model) == ties
    assert report["post_model_weights_unchanged"] and report["runtime_policy"]["restoration_exact"] and report["rng_integrity"]["restoration_exact"]
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())
    json.dumps(report, allow_nan=False)


def test_real_chunk_boundary_and_global207_not_a_synthetic_state():
    model, (x, y) = tiny_model(), tokens(2, 257)
    report = study.study_case(model, x, y, prefix_length=128)
    assert report["execution_status"] == "completed", report.get("reason")
    for cell in report["cells"]:
        prefix = cell["actual_prefix"]
        assert prefix["chunked_schedule"] == [128, 1]
        assert prefix["next_token_effects_vs_original"]["chunked128"]["scored_tokens"] == 2 * 129
        for probe in prefix["frozen_projections"]:
            calls = probe["frozen_arithmetic"]["actual_calls"]
            assert calls[0]["position"] == 128 and sum(call["length"] for call in calls) == 129
            if probe["route"] == "tokenwise":
                assert calls[79]["position"] == 207
            if probe["route"] == "chunked128":
                assert [call["position"] for call in calls] == [128, 256]


def test_exact_prior_reproduction_covers_negative_intermediate_and_prefix_statistics():
    model, (x, y) = tiny_model(), tokens()
    current = study.study_case(model, x, y, prefix_length=4)["cells"][0]
    previous = copy.deepcopy(current)
    assert study.reproduction(current, previous)["exact_equal"]
    previous["actual_prefix"]["suffix_comparisons"]["chunked128"]["decay_baseline"]["max_tolerance_ratio"] += 1e-12
    repeated = study.reproduction(current, previous)
    assert not repeated["exact_equal"] and not repeated["comparisons"]["prefix.suffix_comparisons"]
    previous = copy.deepcopy(current)
    previous["stage_comparisons"]["tokenwise_vs_original"]["stages"][-1]["max_absolute_error"] += 1e-12
    assert not study.reproduction(current, previous)["exact_equal"]
    previous = copy.deepcopy(current)
    previous["logits"]["tokenwise"]["sha256"] = "0" * 64
    assert not study.reproduction(current, previous)["exact_equal"]
    previous = copy.deepcopy(current)
    previous["next_token_effects"]["tokenwise_vs_own_full"]["mean_nll_delta"] += 1e-12
    assert not study.reproduction(current, previous)["exact_equal"]


def test_repeat_exactness_is_a_separate_prerequisite_even_inside_original_tolerance(monkeypatch):
    original, calls = study.prior.collect_route, [0]
    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 1:
            result["logits"] = result["logits"] + 1e-6
        return result
    monkeypatch.setattr(study.prior, "collect_route", altered)
    model, (x, y) = tiny_model(), tokens()
    report = study.study_case(model, x, y, prefix_length=4)
    assert report["execution_status"] == "completed"
    assert report["original_anchor"]["repeat"]["comparison"]["passed"]
    assert not report["attribution_control_stable"] and not report["all_recorded_comparisons_passed"]


@pytest.mark.parametrize("failure", [RuntimeError("allowance injected"), MemoryError("CPU allocation injected")])
def test_prefix_frozen_failure_retains_measured_endpoints_and_no_completed_pass(monkeypatch, failure):
    model, (x, y) = tiny_model(), tokens()
    before = study.base.weight_sha256(model)
    monkeypatch.setattr(study, "frozen_projection_probe", lambda *a, **k: (_ for _ in ()).throw(failure))
    report = study.study_case(model, x, y, prefix_length=4)
    assert report["execution_status"] == "incomplete" and report["failed_cell"] == "P0"
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert cell["execution_status"] == "incomplete" and not cell["whole_model_passed"] and not cell["all_recorded_comparisons_passed"]
    assert len(cell["whole_model_comparisons"]) == 15 and len(cell["stage_comparisons"]) == 7
    assert len(cell["actual_prefix"]["suffix_comparisons"]) == 3 and not cell["actual_prefix"]["passed"]
    assert cell["actual_prefix"]["failed_projection"] == {"route": "one_shot", "module": "blocks.0.mixer.in_proj"}
    assert before == study.base.weight_sha256(model)
    assert all("forward" not in module.__dict__ for module in model.modules())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_actual_model_loss_capture_is_one_call_and_hook_restores_on_failure():
    model, (x, y) = tiny_model(), tokens()
    result = study.collect_full(model, x, y, lambda: None)
    assert torch.equal(result["loss"], model(x, y)[1])
    with pytest.raises(RuntimeError, match="injected"):
        study.collect_full(model, x, y, lambda: (_ for _ in ()).throw(RuntimeError("injected")))
    assert not model._forward_hooks


def prior_fixture(tmp_path):
    # Public-only semantic fixture: no model checkpoint or prepared corpus needed.
    protocol = study.base.read_json(study.DEFAULT_PRIOR_DECLARATION)
    protocol.update(declaration={"sha256": study.base.file_sha256(study.DEFAULT_PRIOR_DECLARATION)})
    rows = [{"ratio": ratio, "case_id": case, "weights_sha256": "1" * 64, "execution_status": "completed", "attribution_control_stable": True,
             "original_anchor": {},
             "cells": [{"cell_id": cell} for cell in study.norm.CELLS]} for ratio in study.RATIOS for case in study.CASE_IDS]
    report = {"kind": "trained_checkpoint_normalization_head_precision", "execution_status": "completed", "certified": False,
              "protocol": protocol, "protocol_sha256": study.base.canonical_sha256(protocol), "cases": rows}
    payload = json.dumps(report, allow_nan=False).encode()
    compressed = gzip.compress(payload, mtime=0)
    raw, zipped, publication, summary = (tmp_path / name for name in ("prior.json", "prior.json.gz", "publication.json", "summary.json"))
    raw.write_bytes(payload)
    zipped.write_bytes(compressed)
    publication.write_text(json.dumps({"schema": 1, "kind": "lossless_normalization_head_raw_publication", "compression": "gzip", "round_trip_exact": True,
        "path": zipped.name, "bytes": len(compressed), "sha256": hashlib.sha256(compressed).hexdigest(), "uncompressed_path": raw.name,
        "uncompressed_bytes": len(payload), "uncompressed_sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8")
    summary.write_text("{}", encoding="utf-8")
    return raw, zipped, publication, summary


def test_exact_public_gzip_and_original_representations_without_private_data(tmp_path):
    raw, zipped, publication, _ = prior_fixture(tmp_path)
    first, first_identity = study.read_prior_report(zipped, publication)
    second, second_identity = study.read_prior_report(raw, publication)
    assert first == second and first_identity["original"] == second_identity["original"]
    assert first_identity["selected_representation"]["compressed"] and not second_identity["selected_representation"]["compressed"]
    assert first_identity["selected_representation"]["sha256"] == study.base.file_sha256(zipped)
    assert first_identity["original"]["sha256"] == study.base.file_sha256(raw)


@pytest.mark.parametrize("change", ["hash", "original_hash", "bytes", "compression", "round_trip_exact", "bomb"])
def test_gzip_transport_tampering_or_oversize_rejected(tmp_path, change):
    _, zipped, publication, _ = prior_fixture(tmp_path)
    manifest = study.base.read_json(publication)
    if change == "hash":
        manifest["sha256"] = "0" * 64
    elif change == "original_hash":
        manifest["uncompressed_sha256"] = "0" * 64
    elif change == "bytes":
        manifest["uncompressed_bytes"] -= 1
    elif change == "bomb":
        manifest["uncompressed_bytes"] = study.MAX_PRIOR_BYTES + 1
    else:
        manifest[change] = "wrong" if change == "compression" else False
    publication.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        study.read_prior_report(zipped, publication)


@pytest.fixture
def declaration_files(tmp_path):
    _, zipped, publication, summary = prior_fixture(tmp_path)
    output = tmp_path / "future.json"
    kwargs = {"device": "cpu", "prior_declaration": study.DEFAULT_PRIOR_DECLARATION, "prior_report": zipped,
              "prior_publication": publication, "prior_summary": summary, "output": output}
    planned = study.declaration_template(**kwargs)
    path = tmp_path / "declared.json"
    path.write_text(json.dumps(planned), encoding="utf-8")
    return path, kwargs, planned


def test_read_only_declaration_exact_scope_registry_sources_and_transport(declaration_files):
    path, kwargs, expected = declaration_files
    planned, controls = study.audit_declaration(path, **kwargs)
    assert planned == expected and len(controls) == 6
    assert list(planned["projection_site_registry"]) == list(study.RATIOS)
    assert [len(value) for value in planned["projection_site_registry"].values()] == [15, 12]
    assert planned["report_schema"]["stage_pairings"] == list(study.PAIRINGS)
    assert "scripts/study_projection_precision.py" in planned["source_sha256"]


@pytest.mark.parametrize("field", ["cells", "source_sha256", "projection_site_registry", "shared_inputs", "tolerances", "observed_inference_policy", "output", "prior_transport"])
def test_declared_scope_mutations_reject_before_allocations(declaration_files, field):
    path, kwargs, planned = declaration_files
    planned[field] = {} if isinstance(planned[field], dict) else [] if isinstance(planned[field], list) else "altered"
    path.write_text(json.dumps(planned), encoding="utf-8")
    with pytest.raises(ValueError, match="declaration differs"):
        study.audit_declaration(path, **kwargs)


def test_allowance_starts_before_preflight_and_never_reaches_private_data(monkeypatch, tmp_path):
    ticks = [0.]
    original = study.breadth.DiagnosticLimit
    monkeypatch.setattr(study.breadth, "DiagnosticLimit", lambda seconds: original(seconds, clock=lambda: ticks[0]))
    def slow_preflight(*a, **k):
        ticks[0] = 901.
        return {}, []
    monkeypatch.setattr(study, "audit_declaration", slow_preflight)
    monkeypatch.setattr(study.base, "audit_inputs", lambda *a: pytest.fail("expired preflight reached private data"))
    with pytest.raises(RuntimeError, match="allowance"):
        study.run_study("cpu", output=tmp_path / "new.json")
    assert not (tmp_path / "new.json").exists()


def test_output_binding_failure_and_immutable_exception_report(monkeypatch, tmp_path, capsys):
    output, declaration = tmp_path / "incomplete.json", tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"output": "wrong.json"}))
    monkeypatch.setattr(study, "run_study", lambda *a, **k: pytest.fail("mismatched output reached execution"))
    with pytest.raises(SystemExit):
        study.main(["--device", "cpu", "--declaration", str(declaration), "--output", str(output)])
    assert not output.exists()
    monkeypatch.setattr(study, "run_study", lambda *a, **k: (_ for _ in ()).throw(MemoryError("allocation failed")))
    assert study.main(["--device", "cpu", "--declaration", str(tmp_path / "missing.json"), "--output", str(output)]) == 2
    report = study.base.read_json(output)
    assert report["status"] == "incomplete" and report["cases"] == [] and not report["certified"]
    assert b"\r\n" not in output.read_bytes()
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        study.main(["--device", "cpu", "--output", str(output)])
    assert output.read_bytes() == before and "completed_cells" in capsys.readouterr().out
