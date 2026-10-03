"""Explicit scan engines. Runtime choices never become checkpoint parameter fields.

The portable engines use the project's existing SSD equations. The optional fused
engine replaces only that scan, preserving local projection, convolution, gating
and normalization. Unavailable or incompatible fused kernels raise; no fallback
may turn a requested fused measurement into a portable measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import platform
from typing import Callable

import torch


class BackendUnavailableError(RuntimeError):
    """A requested execution engine cannot satisfy its declared contract."""


@dataclass(frozen=True)
class ScanBackend:
    name: str
    chunk_size: int = 128
    operator: Callable | None = field(default=None, repr=False, compare=False)

    def metadata(self) -> dict:
        if self.name == "reference":
            paths = {"training": "torch.quadratic_ssd", "prefill": "torch.chunked_ssd",
                     "decode": "torch.chunked_ssd"}
        elif self.name == "torch_chunked":
            paths = dict.fromkeys(("training", "prefill", "decode"), "torch.chunked_ssd")
        else:
            paths = dict.fromkeys(("training", "prefill", "decode"),
                                  "mamba_ssm.mamba_chunk_scan_combined")
        return {"requested": self.name, "resolved": self.name, "chunk_size": self.chunk_size,
                "paths": paths, "fused": self.name == "fused_mamba",
                "certification": "requires a recorded numerical and gradient parity run"}

    def scan(self, x, dt, A, B, C, D, initial_state, *, reference_scan,
             stateful_scan, inference_chunk_size: int = 128):
        """Return outputs and final memory, keeping state connected to autograd."""
        if self.name == "reference" and initial_state is None:
            return reference_scan(x, dt, A, B, C, D), None
        if self.name == "fused_mamba":
            if x.device.type != "cuda":
                raise BackendUnavailableError("fused_mamba requires CUDA tensors; CPU fallback is forbidden")
            if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise BackendUnavailableError("fused_mamba supports float16, bfloat16 or float32 scan inputs")
            if self.operator is None:
                raise BackendUnavailableError("fused_mamba has no resolved operator")
            # B/C have one group locally. dt already includes softplus and dt_bias;
            # repeating either transformation changes the checkpoint's mathematics.
            result = self.operator(
                x.contiguous(), dt.contiguous(), A.float(),
                B.unsqueeze(2).contiguous(), C.unsqueeze(2).contiguous(),
                self.chunk_size, D=D.float(), z=None, dt_bias=None,
                initial_states=initial_state, dt_softplus=False,
                return_final_states=initial_state is not None, state_dtype=torch.float32,
            )
            if initial_state is None:
                if (not isinstance(result, torch.Tensor) or result.shape != x.shape
                        or result.device != x.device or result.dtype not in
                        (torch.float16, torch.bfloat16, torch.float32)):
                    raise BackendUnavailableError("fused_mamba returned incompatible scan outputs")
                return result.float(), None
            if not isinstance(result, tuple) or len(result) != 2:
                raise BackendUnavailableError("fused_mamba did not return outputs and final memory")
            y, final_state = result
            if (not isinstance(y, torch.Tensor) or y.shape != x.shape or y.device != x.device
                    or y.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
                raise BackendUnavailableError("fused_mamba returned incompatible scan outputs")
            if (not isinstance(final_state, torch.Tensor) or final_state.shape != initial_state.shape
                    or final_state.dtype != torch.float32 or final_state.device != initial_state.device):
                raise BackendUnavailableError("fused_mamba returned an incompatible recurrent state")
            return y.float(), final_state

        chunk = inference_chunk_size if initial_state is not None else self.chunk_size
        carried = initial_state
        if carried is None:
            carried = torch.zeros(x.shape[0], x.shape[2], x.shape[3], B.shape[-1],
                                  device=x.device, dtype=torch.float32)
        outputs = []
        for start in range(0, x.shape[1], chunk):
            stop = min(start + chunk, x.shape[1])
            y, carried = stateful_scan(x[:, start:stop], dt[:, start:stop], A,
                                      B[:, start:stop], C[:, start:stop], D, carried)
            outputs.append(y)
        return torch.cat(outputs, dim=1), carried


def resolve_scan_backend(name: str, chunk_size: int = 128) -> ScanBackend:
    """Resolve once, before mutating a model or allocating inference memory."""
    if name not in ("reference", "torch_chunked", "fused_mamba"):
        raise ValueError(f"unknown scan backend: {name!r}")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("scan chunk_size must be a positive integer")
    if name != "fused_mamba":
        return ScanBackend(name, chunk_size)
    if chunk_size & (chunk_size - 1):
        raise ValueError("fused_mamba chunk_size must be a power of two")
    if platform.system() != "Linux":
        raise BackendUnavailableError(
            "fused_mamba requires the supported Linux/CUDA environment; this system is "
            f"{platform.system()}. Use reference or torch_chunked explicitly."
        )
    if not torch.cuda.is_available():
        raise BackendUnavailableError("fused_mamba requires an available CUDA device")
    try:
        module = importlib.import_module("mamba_ssm.ops.triton.ssd_combined")
        operator = module.mamba_chunk_scan_combined
        parameters = inspect.signature(operator).parameters
    except Exception as exc:
        raise BackendUnavailableError(f"cannot load the official Mamba scan: {exc}") from exc
    required = {"D", "z", "dt_bias", "initial_states", "dt_softplus",
                "return_final_states", "state_dtype"}
    if not required.issubset(parameters):
        raise BackendUnavailableError(
            "installed Mamba scan lacks required API fields: "
            + ", ".join(sorted(required - parameters.keys()))
        )
    return ScanBackend(name, chunk_size, operator)
