"""Sampling and recurrent-generation contracts."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from src.generation import SamplingSettings, generate_text
from src.model.config import ModelConfig
from src.model.lm import HybridLM


def tiny_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel(
        {"<|endoftext|>": 0, "<unk>": 1, "small": 2, "models": 3, "can": 4,
         "learn": 5, "patterns": 6, ".": 7},
        unk_token="<unk>",
    ))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def tiny_model() -> HybridLM:
    torch.manual_seed(41)
    config = ModelConfig(
        name="tiny", ratio="1:3", vocab_size=8, d_model=32, n_layers=4,
        head_dim=16, mamba_headdim=16, d_state=8, mlp_multiple_of=16,
    )
    return HybridLM(config).eval()


def test_sampling_settings_reject_out_of_range_values():
    with pytest.raises(ValueError, match="temperature"):
        SamplingSettings(temperature=0.0).validate()
    with pytest.raises(ValueError, match="top_k"):
        SamplingSettings(top_k=101).validate()
    with pytest.raises(ValueError, match="max_new_tokens"):
        SamplingSettings(max_new_tokens=513).validate()


def test_generate_text_is_seeded_and_reports_the_recurrent_state():
    model = tiny_model()
    tokenizer = tiny_tokenizer()
    settings = SamplingSettings(temperature=0.8, top_k=4, max_new_tokens=5, seed=17)
    events = []

    first = generate_text(
        model, tokenizer, "small models", settings, device=torch.device("cpu"),
        ratio="1:3", checkpoint_sha256="abc", on_token=events.append,
    )
    second = generate_text(
        model, tokenizer, "small models", settings, device=torch.device("cpu"),
        ratio="1:3", checkpoint_sha256="abc",
    )

    assert first.token_ids == second.token_ids
    assert first.metrics.prompt_tokens == 2
    assert first.metrics.generated_tokens == len(first.token_ids)
    assert first.metrics.logical_state_mib > 0
    assert first.metrics.peak_vram_mib is None
    assert len(events) == first.metrics.generated_tokens
    assert events[-1].completion == first.completion


def test_generate_text_rejects_empty_and_overlong_prompts():
    model = tiny_model()
    tokenizer = tiny_tokenizer()
    with pytest.raises(ValueError, match="visible text"):
        generate_text(
            model, tokenizer, "   ", SamplingSettings(), device=torch.device("cpu"),
            ratio="1:3", checkpoint_sha256="abc",
        )
    with pytest.raises(ValueError, match="limit is 1"):
        generate_text(
            model, tokenizer, "small models", SamplingSettings(max_prompt_tokens=1),
            device=torch.device("cpu"), ratio="1:3", checkpoint_sha256="abc",
        )
