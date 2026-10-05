"""Portable CPU checks for observational isolation, independent oracles and scope."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import scripts.isolate_tokenwise_numerics as isolation
import scripts.study_decay_precision as decay
from src.model.config import ModelConfig
from src.model.lm import HybridLM


@pytest.fixture(autouse=True)
def modest_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    torch.manual_seed(19)
    return HybridLM(ModelConfig(ratio="1:3", vocab_size=32, d_model=16, n_layers=4,
                               head_dim=8, mamba_headdim=8, d_state=4, mlp_multiple_of=8)).eval()


def tokens():
    return torch.arange(9).reshape(1, -1), torch.arange(1, 10).reshape(1, -1)


def test_time_axis_concatenation_constants_coverage_and_detachment():
    trace = isolation.TokenTrace()
    value = torch.tensor([[[1., 2.]]], requires_grad=True)
    trace.query_length = 1
    trace.capture(0, "attention_query", value)
    trace.capture(0, "A", torch.tensor([2.]), constant=True)
    trace.position = 1
    trace.capture(0, "attention_query", value + 1)
    trace.capture(0, "A", torch.tensor([2.]), constant=True)
    with torch.no_grad():
        value.add_(10)
    joined = trace.tensors(2)
    assert joined[(0, "attention_query")].tolist() == [[[1., 2.], [2., 3.]]]
    assert not joined[(0, "attention_query")].requires_grad
    assert joined[(0, "A")].shape == (1,)
    with pytest.raises(RuntimeError, match="coverage"):
        trace.tensors(3)
    trace.capture(0, "A", torch.tensor([3.]), constant=True)
    with pytest.raises(RuntimeError, match="constant changed"):
        trace.tensors(2)


def test_deferred_nonfinite_trace_rejected_at_route_finalization():
    trace = isolation.TokenTrace()
    trace.query_length = 1
    trace.capture(0, "stage", torch.tensor([[[float("nan")]]]))
    with pytest.raises(ValueError, match="finite"):
        trace.tensors(1)


@pytest.mark.parametrize("start", [0, 2])
def test_duplicate_or_missing_token_coverage_rejected(start):
    trace = isolation.TokenTrace()
    trace.query_length = 1
    trace.capture(0, "stage", torch.zeros(1, 1, 2))
    trace.position = start
    trace.capture(0, "stage", torch.zeros(1, 1, 2))
    with pytest.raises(RuntimeError, match="coverage"):
        trace.tensors(2)


def test_functional_interceptors_backends_and_hooks_restore_on_failure():
    model = tiny_model()
    conv, attention = F.conv1d, F.scaled_dot_product_attention
    backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    trace = isolation.TokenTrace()
    with pytest.raises(RuntimeError, match="injected"):
        with isolation.trace_execution(model, trace):
            assert F.conv1d is not conv and F.scaled_dot_product_attention is not attention
            raise RuntimeError("injected exception")
    assert F.conv1d is conv and F.scaled_dot_product_attention is attention
    assert [block.mixer.scan_backend for block in model.blocks if not block.is_attn] == backends
    assert trace.active is None
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_full_token_trace_aligns_current_keys_conv_crop_and_context_lengths():
    model, (x, y) = tiny_model(), tokens()
    with isolation.base.preserve_model_execution(model), decay.temporary_decay_treatment():
        full = isolation.collect_route(model, x, y)
        token = isolation.collect_route(model, x, y, tokenwise=True)
    assert full["order"] == token["order"]
    for key, value in full["tensors"].items():
        assert value.shape == token["tensors"][key].shape
        assert not value.requires_grad and value.device.type == "cpu"
    assert full["tensors"][(3, "attention_query")].shape == (1, 9, 2, 8)
    assert full["tensors"][(0, "convolution_raw")].shape[1] == 9
    assert [row["key_length"] for row in token["attention_events"]] == list(range(1, 10))
    assert [row["position"] for row in token["attention_events"]] == list(range(9))
    assert token["attention_events"][0]["is_causal"]
    assert all(not row["is_causal"] and not row["mask_present"] for row in token["attention_events"][1:])
    assert len(full["scan_events"]) == 3 and len(token["scan_events"]) == 27
    assert {row["path"] for row in full["scan_events"]} == {"torch.quadratic_ssd"}
    assert {row["path"] for row in token["scan_events"]} == {"torch.chunked_ssd"}


def test_independent_depthwise_oracle_kernel_order_real_tail_and_no_grad():
    x = torch.tensor([1., 2., 3.]).reshape(1, 3, 1).requires_grad_()
    weight = torch.tensor([1., 10., 100.]).reshape(1, 1, 3).requires_grad_()
    before = x.detach().clone()
    zero = isolation.convolution_oracle_fp64(x, weight)
    actual = isolation.convolution_oracle_fp64(x, weight, initial_tail=torch.tensor([[[4., 5.]]]))
    torch.testing.assert_close(zero.flatten(), torch.tensor([100., 210., 321.], dtype=torch.float64))
    torch.testing.assert_close(actual.flatten(), torch.tensor([154., 215., 321.], dtype=torch.float64))
    assert not actual.requires_grad and x.grad is None and weight.grad is None
    assert torch.equal(x.detach(), before)


def test_convolution_probe_replays_exact_projected_operand():
    module = torch.nn.Conv1d(2, 2, 3, groups=2, padding=2)
    x = torch.randn(2, 7, 2)
    observed = module(x.transpose(1, 2))[..., :7].transpose(1, 2)
    report = isolation.convolution_probe(x, module, observed)
    assert report["full_replay_vs_observed"]["nonzero_error_count"] == 0
    assert report["tokenwise_vs_full"]["passed"] and report["full_vs_fp64"]["passed"]
    assert report["input"]["sha256"] == isolation.base.tensor_sha256(x.reshape(-1))


def test_attention_oracle_correct_causal_prefix_and_future_independence():
    q = torch.zeros(1, 1, 3, 2, requires_grad=True)
    k = torch.ones_like(q)
    v = torch.tensor([1., 3., 6.]).reshape(1, 1, 3, 1).expand(1, 1, 3, 2)
    original = isolation.attention_oracle_fp64(q, k, v)
    torch.testing.assert_close(original[0, 0, :, 0], torch.tensor([1., 2., 10/3], dtype=torch.float64))
    future = v.clone()
    future[:, :, 2] = 100
    perturbed = isolation.attention_oracle_fp64(q, k, future)
    assert torch.equal(original[:, :, :2], perturbed[:, :, :2])
    assert not torch.equal(original[:, :, 2], perturbed[:, :, 2])
    suffix = isolation.attention_oracle_fp64(q[:, :, 1:], k, v, prefix_length=1)
    assert torch.equal(suffix, original[:, :, 1:]) and not suffix.requires_grad
    with pytest.raises(ValueError, match="prefix"):
        isolation.attention_oracle_fp64(q[:, :, 1:], k, v, prefix_length=0)


def test_attention_frozen_replay_preserves_post_rope_and_masks():
    generator = torch.Generator().manual_seed(23)
    q, k, v = [torch.randn(2, 7, 2, 4, generator=generator) for _ in range(3)]
    observed = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
    report = isolation.attention_probe(q, k, v, observed, device="cpu")
    assert report["full_replay_vs_observed"]["nonzero_error_count"] == 0
    assert report["tokenwise_vs_full"]["passed"] and report["tokenwise_vs_fp64"]["passed"]
    assert report["operands"]["post_rope_q"]["shape"] == [2, 2, 7, 4]


def test_frozen_attention_retains_original_strides_and_storage_offsets():
    backing = torch.randn(2, 7, 24)
    q, k, v = [value.view(2, 7, 2, 4).transpose(1, 2) for value in backing.split(8, -1)]
    layout = {name: {"shape": list(value.shape), "stride": list(value.stride()), "storage_offset": value.storage_offset(), "dtype": str(value.dtype)}
              for name, value in zip(("q", "k", "v"), (q, k, v))}
    observed = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    report = isolation.attention_probe(*(value.transpose(1, 2) for value in (q, k, v)), observed, device="cpu", original_layout=layout)
    assert report["original_layout"] == report["replay_layout"] == layout
    assert report["full_replay_vs_observed"]["nonzero_error_count"] == 0


def test_frozen_scan_nonzero_real_state_independent_solution_and_no_quadratic_claim():
    x = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1)
    operands = (x, torch.ones(1, 3, 1), torch.zeros(1), torch.ones(1, 3, 1), torch.ones(1, 3, 1), torch.zeros(1))
    initial = torch.full((1, 1, 1, 1), 2., requires_grad=True)
    before = initial.detach().clone()
    report = isolation.frozen_scan_probe(operands, initial, device="cpu")
    assert report["initial_condition"] == "supplied exploratory state; not asserted to be a model cache" and report["synthetic"]
    assert not report["routes"]["candidate_quadratic"]["executed"]
    for name in ("candidate_one_shot", "candidate_chunked128", "candidate_tokenwise"):
        assert report["routes"][name]["output_vs_fp64"]["nonzero_error_count"] == 0
        assert report["routes"][name]["retained_state_vs_fp64"]["nonzero_error_count"] == 0
    assert initial.grad is None and torch.equal(initial.detach(), before)


def test_linear_replay_same_operands_and_weight_unchanged():
    module = torch.nn.Linear(5, 9)
    x = torch.randn(2, 7, 5, requires_grad=True)
    weight, original = module.weight.detach().clone(), x.detach().clone()
    report = isolation.linear_probe(x, module, module(x))
    assert report["identical_operands"] and report["full_replay_vs_observed"]["nonzero_error_count"] == 0
    assert report["tokenwise_vs_full"]["passed"] and report["full_vs_fp64"]["passed"]
    assert torch.equal(weight, module.weight) and torch.equal(original, x) and x.grad is None


def test_mixer_selection_distinguishes_harmless_drift_threshold_and_last():
    model = tiny_model()
    rows = [{"layer": 0, "stage": "scan_output", "nonzero_error_count": 1, "passed": True},
            {"layer": 2, "stage": "mlp_down.output", "nonzero_error_count": 1, "passed": False}]
    selected = isolation.select_mixers(rows, model)
    assert [row["layer"] for row in selected] == [0, 3]
    rows.append({"layer": 1, "stage": "out_projection.output", "nonzero_error_count": 1, "passed": False})
    selected = isolation.select_mixers(rows, model)
    assert [row["layer"] for row in selected] == [0, 1]


def test_complete_tiny_case_real_prefix_anchors_weights_modes_and_hooks_unchanged():
    model, (x, y) = tiny_model(), tokens()
    model.train()
    model.blocks[1].eval()
    model.configure_scan_backend("torch_chunked", 2)
    original_modes = [module.training for module in model.modules()]
    original_backends = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    weights = isolation.base.weight_sha256(model)
    conv, sdpa = F.conv1d, F.scaled_dot_product_attention
    report = isolation.study_case(model, x, y, prefix_length=4)
    assert isolation.base.weight_sha256(model) == weights
    assert [module.training for module in model.modules()] == original_modes
    assert [block.mixer.scan_backend for block in model.blocks if not block.is_attn] == original_backends
    assert F.conv1d is conv and F.scaled_dot_product_attention is sdpa
    assert set(report["stage_comparisons"]) == {"candidate_full_vs_original_stateless", "candidate_tokenwise_vs_candidate_full", "candidate_tokenwise_vs_original_stateless"}
    assert report["actual_prefix"]["prefix_position"] == 4 and report["actual_prefix"]["prefix_unchanged"]
    assert report["actual_prefix"]["scan_probes"] and report["actual_prefix"]["final_position"] == 9
    assert report["whole_model_passed"] and report["original_repeat"]["exact_equal"]
    assert not report["backward_executed"] and not report["BF16_executed"]
    assert len(report["selected_frozen_mixers"]) <= 3
    assert all(parameter.grad is None for parameter in model.parameters())
    json.dumps(report, allow_nan=False)


def test_study_case_failure_restores_mixed_modes_backend_flags_and_overrides():
    model, (x, y) = tiny_model(), tokens()
    model.train()
    model.blocks[-1].eval()
    modes = [module.training for module in model.modules()]
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    bindings = (F.conv1d, F.scaled_dot_product_attention, isolation.mamba.ssd, isolation.mamba.ssd_stateful)
    calls = 0
    def injected():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected budget exhaustion")
    with pytest.raises(RuntimeError, match="injected"):
        isolation.study_case(model, x, y, prefix_length=4, check_limit=injected)
    assert [module.training for module in model.modules()] == modes
    assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    assert bindings == (F.conv1d, F.scaled_dot_product_attention, isolation.mamba.ssd, isolation.mamba.ssd_stateful)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_prepared_inputs_reproducible_paired_and_does_not_change_rng():
    validation = np.arange(10000, dtype=np.uint16) % 32
    before = torch.random.get_rng_state().clone()
    one, two = isolation.prepare_inputs(validation), isolation.prepare_inputs(validation)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert [row[2]["case_id"] for row in one] == list(isolation.CASE_IDS)
    assert [row[2] for row in one] == [row[2] for row in two]
    assert one[-1][2]["gradient_check"] is True  # Immutable input identity, not execution.
    assert all(torch.equal(x[:, 1:], y[:, :-1]) for x, y, _ in one)


def test_report_publication_is_immutable_and_lf(tmp_path):
    path = tmp_path / "report.json"
    isolation.base.write_report(path, {"status": "incomplete", "cases": []})
    original = path.read_bytes()
    assert b"\r\n" not in original
    with pytest.raises(FileExistsError):
        isolation.base.write_report(path, {"status": "completed"})
    assert path.read_bytes() == original


def test_racing_report_publication_cannot_replace_other_artifact(tmp_path, monkeypatch):
    path = tmp_path / "race.json"
    original_link = isolation.base.os.link
    def racing_link(source, destination):
        destination.write_bytes(b"other publisher's immutable evidence\n")
        return original_link(source, destination)
    monkeypatch.setattr(isolation.base.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        isolation.base.write_report(path, {"status": "completed"})
    assert path.read_bytes() == b"other publisher's immutable evidence\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_output_binding_rejects_before_execution_or_directory_creation(tmp_path, monkeypatch):
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"output": "declared.json"}), encoding="utf-8")
    output = tmp_path / "new-directory" / "wrong.json"
    monkeypatch.setattr(isolation, "run_study", lambda *args, **kwargs: pytest.fail("execution must not begin"))
    with pytest.raises(SystemExit) as exc:
        isolation.main(["--device", "cpu", "--declaration", str(declaration), "--output", str(output)])
    assert exc.value.code == 2 and not output.parent.exists()


@pytest.fixture
def declared_public_evidence(tmp_path):
    # These small public reports are committed. No private checkpoints/corpus used.
    raw_path = tmp_path / "breadth.json"
    raw = isolation.base.read_json(isolation.DEFAULT_BREADTH_REPORT)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    inputs = [row for row in raw["protocol"]["shared_inputs"] if row["case_id"] in isolation.CASE_IDS]
    planned = isolation.declaration_template(breadth_report=raw_path, shared_inputs=inputs)
    path = tmp_path / "declared.json"
    path.write_text(json.dumps(planned), encoding="utf-8")
    return path, raw_path, planned, raw


def test_exact_declaration_and_measured_source_scope_audited(declared_public_evidence):
    path, raw_path, _, _ = declared_public_evidence
    planned, raw = isolation.audit_declaration(path, isolation.DEFAULT_BREADTH_DECLARATION, raw_path, "cuda")
    assert len(planned["shared_inputs"]) == 3 and raw["execution_status"] == "completed"


@pytest.mark.parametrize("mutation", ["raw_hash", "inputs", "tolerance", "output", "source_empty", "source_missing", "source_stale"])
def test_evidence_tampering_rejected(declared_public_evidence, mutation):
    path, raw_path, planned, raw = declared_public_evidence
    if mutation == "raw_hash":
        planned["breadth_report"]["sha256"] = "0" * 64
    elif mutation == "inputs":
        planned["shared_inputs"][0]["tokens_sha256"] = "0" * 64
    elif mutation == "tolerance":
        planned["tolerances"]["float32"]["atol"] *= 2
    elif mutation == "output":
        planned["output"] = "wrong.json"
    else:
        sources = raw["protocol"]["source_sha256"]
        if mutation == "source_empty":
            sources.clear()
        elif mutation == "source_missing":
            sources.pop("scripts/study_decay_precision.py")
        else:
            sources["scripts/study_decay_precision.py"] = "0" * 64
        raw["protocol_sha256"] = isolation.base.canonical_sha256(raw["protocol"])
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        planned["breadth_report"]["sha256"] = isolation.base.file_sha256(raw_path)
    path.write_text(json.dumps(planned), encoding="utf-8")
    with pytest.raises(ValueError):
        isolation.audit_declaration(path, isolation.DEFAULT_BREADTH_DECLARATION, raw_path, "cuda")
