"""Verified checkpoint registry and one-model-at-a-time runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tokenizers import Tokenizer

from src.eval.suite import VariantCheckpoint, discover_checkpoints, load_variant_model
from src.generation import GenerationResult, SamplingSettings, TokenEvent, generate_text
from src.model.config import ModelConfig


RATIOS = ("1:3", "1:7", "1:15")


def parameter_count(config_data: dict) -> int:
    """Count one tied-embedding model without allocating its tensors."""
    config = ModelConfig(**config_data)
    total = config.vocab_size * config.d_model + config.d_model
    for layer_type in config.layer_types:
        total += 2 * config.d_model
        if layer_type == "attention":
            total += 4 * config.d_model**2
            if config.attn_bias:
                total += 4 * config.d_model
        else:
            group_state = config.n_groups * config.d_state
            conv_dim = config.d_inner + 2 * group_state
            total += config.d_model * (
                2 * config.d_inner + 2 * group_state + config.n_mamba_heads
            )
            total += conv_dim * config.d_conv + conv_dim
            total += 3 * config.n_mamba_heads + config.d_inner
            total += config.d_inner * config.d_model
        if config.mlp_on_every_layer:
            total += 3 * config.d_model * config.mlp_hidden
            if config.mlp_bias:
                total += 2 * config.mlp_hidden + config.d_model
    return total


@dataclass(frozen=True)
class RuntimeSettings:
    checkpoint_root: Path
    tokenizer_path: Path
    training_run_id: str = "week3-700m-v1"
    allowed_ratios: tuple[str, ...] = RATIOS
    default_ratio: str = "1:3"
    device: str = "auto"
    eager_load: bool = True

    def resolved_device(self) -> torch.device:
        requested = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if requested == "auto":
            requested = "cpu"
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device

    def validate(self) -> None:
        if not self.tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer is missing: {self.tokenizer_path}")
        if not self.allowed_ratios:
            raise ValueError("at least one ratio must be allowed")
        if len(set(self.allowed_ratios)) != len(self.allowed_ratios):
            raise ValueError("allowed ratios must be unique")
        if not set(self.allowed_ratios).issubset(RATIOS):
            raise ValueError(f"allowed ratios must be selected from {RATIOS}")
        if self.default_ratio not in self.allowed_ratios:
            raise ValueError("default ratio must be in allowed_ratios")


class DemoRuntime:
    """Own the tokenizer and one verified model, swapping ratios only under the API lock."""

    def __init__(self, settings: RuntimeSettings):
        settings.validate()
        self.settings = settings
        self.device = settings.resolved_device()
        self.tokenizer = Tokenizer.from_file(str(settings.tokenizer_path))
        self.checkpoints = discover_checkpoints(
            settings.checkpoint_root,
            settings.training_run_id,
            settings.allowed_ratios,
        )
        self.model = None
        self.loaded_ratio: str | None = None
        if settings.eager_load:
            self._load(settings.default_ratio)

    def _load(self, ratio: str) -> VariantCheckpoint:
        if ratio not in self.checkpoints:
            raise ValueError(f"ratio {ratio!r} is not available")
        checkpoint = self.checkpoints[ratio]
        if self.loaded_ratio == ratio and self.model is not None:
            return checkpoint
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model = load_variant_model(checkpoint, self.device)
        self.loaded_ratio = ratio
        return checkpoint

    def health(self) -> dict:
        return {
            "status": "ready",
            "mode": "cuda" if self.device.type == "cuda" else "cpu",
            "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "loaded_ratio": self.loaded_ratio,
            "available_ratios": list(self.settings.allowed_ratios),
            "training_run_id": self.settings.training_run_id,
        }

    def models(self) -> list[dict]:
        records = []
        for ratio in self.settings.allowed_ratios:
            checkpoint = self.checkpoints[ratio]
            config = checkpoint.model_config
            records.append({
                "ratio": ratio,
                "name": config["name"],
                "parameters": parameter_count(config),
                "attention_layers": config["layer_types"].count("attention"),
                "mamba_layers": config["layer_types"].count("mamba"),
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "best_step": checkpoint.step,
                "validation_loss": checkpoint.val_loss,
                "loaded": ratio == self.loaded_ratio,
            })
        return records

    def generate(
        self,
        prompt: str,
        ratio: str,
        sampling: SamplingSettings,
        on_token: Callable[[TokenEvent], None] | None = None,
    ) -> GenerationResult:
        try:
            checkpoint = self._load(ratio)
            if self.model is None:
                raise RuntimeError("model failed to load")
            return generate_text(
                self.model,
                self.tokenizer,
                prompt,
                sampling,
                device=self.device,
                ratio=ratio,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                on_token=on_token,
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "The GPU ran out of memory. Try a shorter prompt or fewer new tokens."
            ) from error
