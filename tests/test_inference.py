"""Checks that recurrent inference is the same model, not an approximate second path."""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.inference import AttentionCache
from src.model.lm import HybridLM


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
