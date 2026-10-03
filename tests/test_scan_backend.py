"""Same-weight numerical/gradient contracts; mocks do not certify a CUDA kernel."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from src.model.config import ModelConfig
from src.model.lm import HybridLM
from src.model.mamba2 import Mamba2Mixer, ssd, ssd_stateful
from src.model.scan_backend import BackendUnavailableError, ScanBackend, resolve_scan_backend


def tiny_config():
    return ModelConfig(ratio="1:3", d_model=32, n_layers=4, vocab_size=64, head_dim=8,
                       mamba_headdim=8, d_state=8, mlp_multiple_of=8)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 17, 31])
def test_bounded_training_matches_outputs_and_every_parameter_gradient(length):
    torch.manual_seed(42)
    reference = Mamba2Mixer(tiny_config())
    bounded = deepcopy(reference)
    bounded.scan_backend = resolve_scan_backend("torch_chunked", 8)
    u = torch.randn(2, length, 32, requires_grad=True)
    v = u.detach().clone().requires_grad_()
    target = torch.randn(2, length, 32)
    a, b = reference(u), bounded(v)
    torch.testing.assert_close(b, a, atol=3e-5, rtol=3e-4)
    ((a - target).square().mean()).backward()
    ((b - target).square().mean()).backward()
    torch.testing.assert_close(v.grad, u.grad, atol=3e-5, rtol=3e-4)
    for (name, p), (other_name, q) in zip(reference.named_parameters(), bounded.named_parameters()):
        assert name == other_name
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(q.grad, p.grad, atol=3e-5, rtol=3e-4, msg=name)


def test_nonzero_initial_memory_and_its_gradient_cross_chunk_boundaries():
    torch.manual_seed(43)
    x = torch.randn(2, 19, 3, 4, requires_grad=True)
    dt = torch.rand(2, 19, 3, requires_grad=True) * 0.1
    A = -torch.arange(1, 4).float()
    B, C = torch.randn(2, 19, 5), torch.randn(2, 19, 5)
    D = torch.ones(3)
    initial = torch.randn(2, 3, 4, 5, requires_grad=True)
    full, full_state = ssd_stateful(x, dt, A, B, C, D, initial)
    bounded, next_state = resolve_scan_backend("torch_chunked", 8).scan(
        x, dt, A, B, C, D, initial, reference_scan=ssd, stateful_scan=ssd_stateful)
    torch.testing.assert_close(bounded, full, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(next_state, full_state, atol=3e-5, rtol=3e-4)
    full_grad = torch.autograd.grad(full.square().mean() + full_state.square().mean(),
                                    (x, initial), retain_graph=True)
    bounded_grad = torch.autograd.grad(bounded.square().mean() + next_state.square().mean(),
                                       (x, initial))
    for a, b in zip(full_grad, bounded_grad):
        torch.testing.assert_close(b, a, atol=3e-5, rtol=3e-4)


def test_backend_selection_keeps_checkpoint_fields_and_prefill_decode_semantics():
    torch.manual_seed(44)
    model = HybridLM(tiny_config()).eval()
    params = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config = asdict(model.cfg)
    tokens = torch.randint(0, 64, (1, 21))
    with torch.no_grad():
        reference, _ = model(tokens)
        metadata = model.configure_scan_backend("torch_chunked", 8)
        bounded, _ = model(tokens)
        cached, state = model.prefill(tokens[:, :18])
        decoded = model.decode(tokens[:, 18:], state)
    torch.testing.assert_close(bounded, reference, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(cached, reference[:, 17:18], atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(decoded, reference[:, -1:], atol=3e-5, rtol=3e-4)
    assert state.position == 21
    assert config == asdict(model.cfg)
    assert metadata["paths"]["training"] == "torch.chunked_ssd"
    assert metadata["fused"] is False
    assert params.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, params[name]), name


def test_unavailable_fused_selection_is_atomic(monkeypatch):
    import src.model.scan_backend as module
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    model = HybridLM(tiny_config())
    model.configure_scan_backend("torch_chunked", 8)
    previous = [block.mixer.scan_backend for block in model.blocks if not block.is_attn]
    with pytest.raises(BackendUnavailableError, match="supported Linux"):
        model.configure_scan_backend("fused_mamba")
    assert previous == [block.mixer.scan_backend for block in model.blocks if not block.is_attn]


@pytest.mark.parametrize("name,chunk", [("typo", 8), ("reference", 0),
                                       ("torch_chunked", True), ("fused_mamba", 7)])
def test_invalid_backend_settings_are_explicit(name, chunk):
    with pytest.raises(ValueError):
        resolve_scan_backend(name, chunk)


def test_fused_adapter_preserves_local_rules_and_requires_fp32_memory(monkeypatch):
    """Only tests adapter arguments. No CUDA kernel is executed by this test."""
    import src.model.scan_backend as module
    calls = []

    def operator(x, dt, A, B, C, chunk_size, *, D, z, dt_bias, initial_states,
                 dt_softplus, return_final_states, state_dtype):
        calls.append(locals())
        return torch.zeros_like(x), torch.zeros_like(initial_states)

    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.importlib, "import_module", lambda _: type("Module", (),
                        {"mamba_chunk_scan_combined": staticmethod(operator)})())
    backend = resolve_scan_backend("fused_mamba", 8)
    assert backend.operator is operator
    # The CPU rejection must happen before calling even an available operator.
    x = torch.randn(1, 3, 2, 4)
    with pytest.raises(BackendUnavailableError, match="CPU fallback is forbidden"):
        backend.scan(x, torch.ones(1, 3, 2), -torch.ones(2), torch.ones(1, 3, 5),
                     torch.ones(1, 3, 5), torch.ones(2), torch.zeros(1, 2, 4, 5),
                     reference_scan=ssd, stateful_scan=ssd_stateful)
    assert calls == []


def test_incompatible_fused_api_is_rejected(monkeypatch):
    import src.model.scan_backend as module
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.importlib, "import_module", lambda _: type("Module", (),
                        {"mamba_chunk_scan_combined": staticmethod(lambda x: x)})())
    with pytest.raises(BackendUnavailableError, match="lacks required API"):
        resolve_scan_backend("fused_mamba", 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA adapter arguments need CUDA tensors")
def test_fused_adapter_arguments_on_cuda_are_only_a_mock_contract():
    calls = []

    def operator(x, dt, A, B, C, chunk_size, **kwargs):
        calls.append((x, dt, A, B, C, chunk_size, kwargs))
        return torch.zeros_like(x), torch.zeros_like(kwargs["initial_states"])

    backend = ScanBackend("fused_mamba", 8, operator)
    x = torch.randn(1, 3, 2, 4, device="cuda", dtype=torch.bfloat16)
    dt = torch.ones(1, 3, 2, device="cuda", dtype=torch.bfloat16) * 0.1
    B = torch.ones(1, 3, 5, device="cuda", dtype=torch.bfloat16)
    state = torch.ones(1, 2, 4, 5, device="cuda", dtype=torch.float32)
    y, final = backend.scan(x, dt, -torch.ones(2, device="cuda"), B, B,
                           torch.ones(2, device="cuda"), state,
                           reference_scan=ssd, stateful_scan=ssd_stateful)
    args = calls[0]
    assert args[3].shape == (1, 3, 1, 5) and args[4].shape == (1, 3, 1, 5)
    assert torch.equal(args[1], dt)  # dt has not received a second softplus.
    kwargs = args[-1]
    assert kwargs["dt_softplus"] is False and kwargs["dt_bias"] is None
    assert kwargs["z"] is None and kwargs["state_dtype"] is torch.float32
    assert kwargs["initial_states"] is state and kwargs["return_final_states"] is True
    assert y.dtype == final.dtype == torch.float32


@pytest.mark.parametrize("backend", ["reference", "torch_chunked"])
def test_explicit_portable_inference_chunk_override_is_effective(monkeypatch, backend):
    import src.model.mamba2 as module
    sizes = []
    original = module.ssd_stateful
    def record(x, *args):
        sizes.append(x.shape[1])
        return original(x, *args)
    monkeypatch.setattr(module, "ssd_stateful", record)
    mixer = Mamba2Mixer(tiny_config())
    mixer.scan_backend = resolve_scan_backend(backend, 8)
    with torch.no_grad():
        mixer(torch.randn(1, 10, 32), mixer.init_state(1, "cpu", torch.float32), chunk_size=3)
    assert sizes == [3, 3, 3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA mock output validation")
@pytest.mark.parametrize("malformed", ["flat", "cpu_output", "integer", "half_state", "cpu_state", "tuple"])
def test_malformed_fused_returns_leave_mixer_memory_unchanged(malformed):
    def operator(x, *args, **kwargs):
        y = torch.zeros_like(x)
        state = torch.zeros_like(kwargs["initial_states"])
        if malformed == "flat":
            y = y.flatten()
        elif malformed == "cpu_output":
            y = y.cpu()
        elif malformed == "integer":
            y = y.long()
        elif malformed == "half_state":
            state = state.half()
        elif malformed == "cpu_state":
            state = state.cpu()
        elif malformed == "tuple":
            return (y,)
        return y, state
    mixer = Mamba2Mixer(tiny_config()).cuda()
    mixer.scan_backend = ScanBackend("fused_mamba", 8, operator)
    state = mixer.init_state(1, "cuda", torch.float32)
    state.conv.fill_(0.2)
    state.ssm.fill_(0.3)
    before = state.clone()
    with torch.no_grad(), pytest.raises(BackendUnavailableError, match="incompatible|did not return"):
        mixer(torch.randn(1, 3, 32, device="cuda"), state)
    assert torch.equal(state.conv, before.conv) and torch.equal(state.ssm, before.ssm)
