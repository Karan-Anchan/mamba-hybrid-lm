"""Typed inference state shared by the attention, Mamba, block, and LM layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def tensor_bytes(value: torch.Tensor | None) -> int:
    """Return logical tensor storage, independent of allocator rounding or reserved memory."""
    return 0 if value is None else value.numel() * value.element_size()


@dataclass
class AttentionCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    @property
    def length(self) -> int:
        if (self.key is None) != (self.value is None):
            raise RuntimeError("attention cache must contain both key and value tensors")
        if self.key is None:
            return 0
        if self.key.shape != self.value.shape:
            raise RuntimeError("attention key/value cache shapes do not match")
        return self.key.shape[2]

    def byte_count(self) -> int:
        return tensor_bytes(self.key) + tensor_bytes(self.value)

    def clone(self) -> "AttentionCache":
        return AttentionCache(
            key=None if self.key is None else self.key.clone(),
            value=None if self.value is None else self.value.clone(),
        )


@dataclass
class Mamba2State:
    conv: torch.Tensor
    ssm: torch.Tensor

    def byte_breakdown(self) -> dict[str, int]:
        return {"mamba_conv": tensor_bytes(self.conv), "mamba_ssm": tensor_bytes(self.ssm)}

    def clone(self) -> "Mamba2State":
        return Mamba2State(conv=self.conv.clone(), ssm=self.ssm.clone())


LayerInferenceState = AttentionCache | Mamba2State


@dataclass
class HybridInferenceState:
    layers: list[LayerInferenceState]
    batch_size: int
    max_seq_len: int
    position: int = 0

    def validate_step(self, token_count: int, batch_size: int) -> None:
        if token_count <= 0:
            raise ValueError("stateful inference needs at least one token")
        if batch_size != self.batch_size:
            raise ValueError(
                f"inference-state batch size is {self.batch_size}, received {batch_size}"
            )
        if self.position + token_count > self.max_seq_len:
            raise ValueError(
                f"context would reach {self.position + token_count} tokens; "
                f"the configured limit is {self.max_seq_len}"
            )
        for layer in self.layers:
            if isinstance(layer, AttentionCache) and layer.length != self.position:
                raise RuntimeError(
                    f"attention cache has {layer.length} tokens at model position {self.position}"
                )

    def advance(self, token_count: int) -> None:
        self.position += token_count

    def byte_breakdown(self) -> dict[str, int]:
        totals = {"attention_kv": 0, "mamba_conv": 0, "mamba_ssm": 0}
        for layer in self.layers:
            if isinstance(layer, AttentionCache):
                totals["attention_kv"] += layer.byte_count()
            else:
                for name, value in layer.byte_breakdown().items():
                    totals[name] += value
        return totals

    def byte_count(self) -> int:
        return sum(self.byte_breakdown().values())

    def clone(self) -> "HybridInferenceState":
        return HybridInferenceState(
            layers=[layer.clone() for layer in self.layers],
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            position=self.position,
        )
