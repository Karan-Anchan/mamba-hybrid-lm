"""Sampling utilities shared by the benchmark and the FastAPI service."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import time
from typing import Callable

import torch
from tokenizers import Tokenizer

from src.model.lm import HybridLM


MIB = 1024 * 1024


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 80
    seed: int = 1337
    max_prompt_tokens: int = 1024

    def validate(self) -> None:
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.1 and 2.0")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if not 1 <= self.max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed must be between 0 and 2^31 - 1")
        if self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")


@dataclass(frozen=True)
class TokenEvent:
    index: int
    token_id: int
    text: str
    completion: str
    elapsed_seconds: float


@dataclass(frozen=True)
class GenerationMetrics:
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    tokens_per_second: float
    decode_tokens_per_second: float
    time_to_first_token_seconds: float
    logical_state_mib: float
    peak_vram_mib: float | None
    current_vram_mib: float | None
    device: str
    stop_reason: str


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    completion: str
    text: str
    token_ids: tuple[int, ...]
    ratio: str
    checkpoint_sha256: str
    settings: SamplingSettings
    metrics: GenerationMetrics

    def to_dict(self) -> dict:
        return asdict(self)


def sample_token(logits: torch.Tensor, settings: SamplingSettings, generator: torch.Generator) -> torch.Tensor:
    """Sample one token after temperature scaling and top-k filtering."""
    scores = logits[:, -1, :].float() / settings.temperature
    k = min(settings.top_k, scores.shape[-1])
    cutoff = torch.topk(scores, k=k, dim=-1).values[:, -1:]
    scores = scores.masked_fill(scores < cutoff, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _precision_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def generate_text(
    model: HybridLM,
    tokenizer: Tokenizer,
    prompt: str,
    settings: SamplingSettings,
    *,
    device: torch.device,
    ratio: str,
    checkpoint_sha256: str,
    on_token: Callable[[TokenEvent], None] | None = None,
) -> GenerationResult:
    """Generate from one prompt with the model's recurrent inference state."""
    settings.validate()
    if not prompt.strip():
        raise ValueError("prompt must contain visible text")

    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("the tokenizer produced an empty prompt")
    if len(prompt_ids) > settings.max_prompt_tokens:
        raise ValueError(
            f"prompt contains {len(prompt_ids)} tokens; the limit is {settings.max_prompt_tokens}"
        )
    if len(prompt_ids) + settings.max_new_tokens > model.rope.cos.shape[0]:
        raise ValueError("prompt and completion exceed the model context limit")

    model.eval()
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    stop_id = tokenizer.token_to_id("<|endoftext|>")
    completion_ids: list[int] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    cache_dtype = torch.bfloat16 if device.type == "cuda" else None
    with _precision_context(device):
        logits, state = model.prefill(tokens, cache_dtype=cache_dtype)
    _sync(device)
    prefill_finished = time.perf_counter()
    first_token_finished = prefill_finished
    stop_reason = "length"

    with _precision_context(device):
        for index in range(settings.max_new_tokens):
            next_token = sample_token(logits, settings, generator)
            token_id = int(next_token.item())
            completion_ids.append(token_id)
            _sync(device)
            emitted_at = time.perf_counter()
            if index == 0:
                first_token_finished = emitted_at
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            if on_token is not None:
                on_token(TokenEvent(
                    index=index,
                    token_id=token_id,
                    text=tokenizer.decode([token_id], skip_special_tokens=True),
                    completion=completion,
                    elapsed_seconds=emitted_at - started,
                ))
            if stop_id is not None and token_id == stop_id:
                stop_reason = "end_of_text"
                break
            if index + 1 < settings.max_new_tokens:
                logits = model.decode(next_token, state)

    _sync(device)
    finished = time.perf_counter()
    generated_tokens = len(completion_ids)
    prefill_seconds = prefill_finished - started
    total_seconds = finished - started
    decode_seconds = max(finished - prefill_finished, 0.0)
    state_mib = state.byte_count() / MIB
    peak_vram = None
    current_vram = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / MIB
        current_vram = torch.cuda.memory_allocated(device) / MIB

    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    metrics = GenerationMetrics(
        prompt_tokens=len(prompt_ids),
        generated_tokens=generated_tokens,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        total_seconds=total_seconds,
        tokens_per_second=generated_tokens / total_seconds if total_seconds else 0.0,
        decode_tokens_per_second=(generated_tokens - 1) / decode_seconds
        if decode_seconds and generated_tokens > 1 else 0.0,
        time_to_first_token_seconds=first_token_finished - started,
        logical_state_mib=state_mib,
        peak_vram_mib=peak_vram,
        current_vram_mib=current_vram,
        device=str(device),
        stop_reason=stop_reason,
    )
    return GenerationResult(
        prompt=prompt,
        completion=completion,
        text=prompt + completion,
        token_ids=tuple(completion_ids),
        ratio=ratio,
        checkpoint_sha256=checkpoint_sha256,
        settings=settings,
        metrics=metrics,
    )
