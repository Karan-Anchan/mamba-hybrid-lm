"""Checks that recurrent inference is the same model, not an approximate second path."""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.inference import AttentionCache, Mamba2State
from src.model.lm import HybridLM
from src.model.mamba2 import Mamba2Mixer


def tiny_config(ratio: str = "1:3", n_layers: int = 4) -> ModelConfig:
    return ModelConfig(
        ratio=ratio,
        d_model=128,
        n_layers=n_layers,
        vocab_size=256,
        head_dim=32,
        mamba_headdim=32,
        d_state=32,
    )


def test_stateful_prefill_matches_parallel_last_logit():
    torch.manual_seed(11)
    model = HybridLM(tiny_config()).eval()
    tokens = torch.randint(0, 256, (2, 19))

    with torch.no_grad():
        parallel, _ = model(tokens)
        cached, state = model.prefill(tokens)

    assert state.position == tokens.shape[1]
    assert torch.allclose(cached, parallel[:, -1:], atol=2e-4, rtol=2e-4)


def test_token_decode_matches_every_parallel_position():
    torch.manual_seed(17)
    model = HybridLM(tiny_config()).eval()
    tokens = torch.randint(0, 256, (1, 13))

    with torch.no_grad():
        parallel, _ = model(tokens)
        state = model.init_inference_state(1)
        incremental = []
        for position in range(tokens.shape[1]):
            incremental.append(model.decode(tokens[:, position:position + 1], state))

    decoded = torch.cat(incremental, dim=1)
    assert state.position == tokens.shape[1]
    assert torch.allclose(decoded, parallel, atol=3e-4, rtol=3e-4)


def test_cached_multi_token_chunks_remain_causal():
    torch.manual_seed(23)
    model = HybridLM(tiny_config()).eval()
    tokens = torch.randint(0, 256, (1, 17))

    with torch.no_grad():
        parallel, _ = model(tokens)
        state = model.init_inference_state(1)
        chunks = []
        start = 0
        for size in (5, 7, 5):
            logits, _ = model(tokens[:, start:start + size], inference_state=state)
            chunks.append(logits)
            start += size

    assert torch.allclose(torch.cat(chunks, dim=1), parallel, atol=3e-4, rtol=3e-4)


def test_state_reports_logical_bytes_by_mixer_type():
    torch.manual_seed(29)
    cfg = tiny_config()
    model = HybridLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 10))

    with torch.no_grad():
        _, state = model.prefill(tokens)

    breakdown = state.byte_breakdown()
    expected_kv = 2 * cfg.n_attention_layers * cfg.n_heads * 10 * cfg.head_dim * 4
    expected_conv = (
        cfg.n_mamba_layers
        * (cfg.d_inner + 2 * cfg.d_state)
        * (cfg.d_conv - 1)
        * 4
    )
    expected_ssm = (
        cfg.n_mamba_layers
        * cfg.n_mamba_heads
        * cfg.mamba_headdim
        * cfg.d_state
        * 4
    )

    assert breakdown == {
        "attention_kv": expected_kv,
        "mamba_conv": expected_conv,
        "mamba_ssm": expected_ssm,
    }
    assert state.byte_count() == sum(breakdown.values())


def test_cloned_state_can_branch_without_mutating_the_prompt_state():
    torch.manual_seed(30)
    model = HybridLM(tiny_config()).eval()
    tokens = torch.randint(0, 256, (1, 9))

    with torch.no_grad():
        logits, state = model.prefill(tokens)
        branch = state.clone()
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        model.decode(next_token, branch)

    assert state.position == 9
    assert branch.position == 10
    assert state.layers[0].conv.data_ptr() != branch.layers[0].conv.data_ptr()


def test_state_rejects_batch_and_context_mismatch():
    model = HybridLM(tiny_config()).eval()
    state = model.init_inference_state(1)
    state.max_seq_len = 4

    with pytest.raises(ValueError, match="batch size"):
        model(torch.ones(2, 1, dtype=torch.long), inference_state=state)
    with pytest.raises(ValueError, match="configured limit"):
        model(torch.ones(1, 5, dtype=torch.long), inference_state=state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the bf16 cache check")
def test_cuda_bf16_prefill_matches_parallel():
    torch.manual_seed(31)
    model = HybridLM(tiny_config()).cuda().eval()
    tokens = torch.randint(0, 256, (1, 37), device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        parallel, _ = model(tokens)
        cached, state = model.prefill(tokens, cache_dtype=torch.bfloat16)

    torch.testing.assert_close(cached.float(), parallel[:, -1:].float(), atol=0.05, rtol=0.02)
    assert state.byte_breakdown()["attention_kv"] > 0
    attention_states = [layer for layer in state.layers if isinstance(layer, AttentionCache)]
    assert all(layer.key.dtype == torch.bfloat16 for layer in attention_states)
    assert all(layer.value.dtype == torch.bfloat16 for layer in attention_states)


def tiny_mixer(d_conv: int = 4) -> Mamba2Mixer:
    return Mamba2Mixer(ModelConfig(
        ratio="0:1", d_model=16, n_layers=1, vocab_size=32, head_dim=8,
        mamba_headdim=8, d_state=4, d_conv=d_conv, mlp_multiple_of=8,
    )).eval()


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5, "4"])
def test_invalid_chunk_cannot_change_retained_memory(chunk_size):
    mixer = tiny_mixer()
    state = mixer.init_state(1, "cpu", torch.float32)
    state.conv.normal_()
    state.ssm.normal_()
    before = state.clone()
    with torch.no_grad(), pytest.raises(ValueError, match="chunk size"):
        mixer(torch.randn(1, 5, 16), state, chunk_size=chunk_size)
    assert torch.equal(state.conv, before.conv)
    assert torch.equal(state.ssm, before.ssm)


@pytest.mark.parametrize("kind, message", [
    ("conv_shape", "shape"), ("ssm_shape", "shape"),
    ("conv_dtype", "convolution state"), ("ssm_dtype", "SSM state"),
])
def test_invalid_state_fails_before_projection_or_memory_update(kind, message):
    mixer = tiny_mixer()
    state = mixer.init_state(1, "cpu", torch.float32)
    state.conv.normal_()
    state.ssm.normal_()
    if kind == "conv_shape":
        state.conv = state.conv[:, :-1]
    elif kind == "ssm_shape":
        state.ssm = state.ssm[..., :-1]
    elif kind == "conv_dtype":
        state.conv = state.conv.to(torch.float64)
    else:
        state.ssm = state.ssm.to(torch.bfloat16)
    before = state.clone()
    projected = []
    mixer.in_proj.register_forward_pre_hook(lambda *_: projected.append(True))
    with torch.no_grad(), pytest.raises(ValueError, match=message):
        mixer(torch.randn(1, 5, 16), state)
    assert not projected
    assert torch.equal(state.conv, before.conv)
    assert torch.equal(state.ssm, before.ssm)


def test_state_device_mismatch_is_rejected_before_using_or_mutating_memory():
    mixer = tiny_mixer()
    state = mixer.init_state(1, "cpu", torch.float32)
    state.conv.normal_()
    before_conv = state.conv.clone()
    # A meta tensor has shape/dtype but no values; use it to exercise a second device on CPU CI.
    state.ssm = torch.empty_like(state.ssm, device="meta")
    wrong_device_tensor = state.ssm
    with pytest.raises(ValueError, match="state device"):
        mixer(torch.randn(1, 5, 16), state)
    assert torch.equal(state.conv, before_conv)
    assert state.ssm is wrong_device_tensor


@pytest.mark.parametrize("input_shape, input_dtype, message", [
    ((1, 16), torch.float32, "shape"),
    ((0, 5, 16), torch.float32, "positive"),
    ((1, 0, 16), torch.float32, "positive"),
    ((1, 5, 15), torch.float32, "d_model"),
    ((1, 5, 16), torch.int64, "floating dtype"),
    ((1, 5, 16), torch.float64, "dtype must match"),
])
def test_invalid_input_cannot_change_retained_memory(input_shape, input_dtype, message):
    mixer = tiny_mixer()
    state = mixer.init_state(1, "cpu", torch.float32)
    state.conv.normal_()
    state.ssm.normal_()
    before = state.clone()
    with pytest.raises(ValueError, match=message):
        mixer(torch.zeros(input_shape, dtype=input_dtype), state)
    assert torch.equal(state.conv, before.conv)
    assert torch.equal(state.ssm, before.ssm)


def test_input_device_mismatch_cannot_change_retained_memory():
    mixer = tiny_mixer()
    state = mixer.init_state(1, "cpu", torch.float32)
    before = state.clone()
    with pytest.raises(ValueError, match="input device"):
        mixer(torch.empty(1, 5, 16, device="meta"), state)
    assert torch.equal(state.conv, before.conv)
    assert torch.equal(state.ssm, before.ssm)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_initial_state_requires_positive_integer_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch size"):
        tiny_mixer().init_state(batch_size, "cpu", torch.float32)


def test_initial_state_rejects_nonfloating_memory_and_invalid_device():
    mixer = tiny_mixer()
    with pytest.raises(ValueError, match="dtype"):
        mixer.init_state(1, "cpu", torch.int64)
    with pytest.raises(ValueError, match="device"):
        mixer.init_state(1, "not-a-device", torch.float32)


def test_mixer_failure_preserves_both_memories(monkeypatch):
    mixer = tiny_mixer()
    state = mixer.init_state(1, "cpu", torch.float32)
    state.conv.normal_()
    state.ssm.normal_()
    before = state.clone()

    def fail_projection(_input):
        raise RuntimeError("injected output projection failure")

    monkeypatch.setattr(mixer.out_proj, "forward", fail_projection)
    with torch.no_grad(), pytest.raises(RuntimeError, match="injected output"):
        mixer(torch.randn(1, 5, 16), state)
    assert torch.equal(state.conv, before.conv)
    assert torch.equal(state.ssm, before.ssm)


@pytest.mark.parametrize("d_conv", [1, 4])
@pytest.mark.parametrize("nonzero_memory", [False, True])
def test_convolution_boundaries_and_final_state_match_chunked_and_tokenwise(d_conv, nonzero_memory):
    torch.manual_seed(41)
    mixer = tiny_mixer(d_conv)
    inputs = torch.randn(2, 17, 16)
    initial = mixer.init_state(2, "cpu", torch.float32)
    if nonzero_memory:
        initial.conv.normal_()
        initial.ssm.normal_()
    one_shot_state, chunked_state, token_state = (initial.clone() for _ in range(3))
    with torch.no_grad():
        one_shot = mixer(inputs, one_shot_state, chunk_size=3)
        chunks = []
        start = 0
        for size in (1, 3, 8, 5):
            chunks.append(mixer(inputs[:, start:start + size], chunked_state, chunk_size=2))
            start += size
        tokenwise = torch.cat([
            mixer(inputs[:, index:index + 1], token_state)
            for index in range(inputs.shape[1])
        ], dim=1)
        if not nonzero_memory:
            torch.testing.assert_close(mixer(inputs), one_shot, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(torch.cat(chunks, dim=1), one_shot, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(tokenwise, one_shot, atol=3e-4, rtol=3e-4)
    for state in (chunked_state, token_state):
        # Different projection batch shapes can round their GEMMs slightly differently.
        torch.testing.assert_close(state.conv, one_shot_state.conv, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(state.ssm, one_shot_state.ssm, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("ratio", ["1:0", "0:1"])
def test_pure_control_models_preserve_stateful_parity(ratio):
    torch.manual_seed(43)
    model = HybridLM(tiny_config(ratio=ratio, n_layers=2)).eval()
    tokens = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        parallel, _ = model(tokens)
        cached, state = model.prefill(tokens)
    torch.testing.assert_close(cached, parallel[:, -1:], atol=3e-4, rtol=3e-4)
    assert state.position == tokens.shape[1]


def test_mixer_rejects_grouped_bc_if_a_config_is_mutated_after_validation():
    cfg = tiny_config()
    cfg.n_groups = 2
    with pytest.raises(ValueError, match="grouped Mamba"):
        Mamba2Mixer(cfg)


def test_mixer_rejects_non_mamba_state_and_non_tensor_memories():
    mixer = tiny_mixer()
    inputs = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match="Mamba2State"):
        mixer(inputs, AttentionCache())
    with pytest.raises(ValueError, match="tensors"):
        mixer(inputs, Mamba2State(conv=None, ssm=None))
