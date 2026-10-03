"""Run a bounded, same-weight backend gate; never train a research campaign.

Example: python scripts/check_scan_backend.py --backend torch_chunked --device cuda
    --dtype bfloat16 --output results/backend-gates/chunked-cuda-bf16.json

An unavailable engine exits 2 and records 'unavailable', not success. This tiny
model gate is useful before larger checkpoint/data comparisons, but cannot
certify equivalence for every shape or establish a language-quality result.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from src.model.config import ModelConfig  # noqa: E402
from src.model.lm import HybridLM  # noqa: E402
from src.model.scan_backend import BackendUnavailableError  # noqa: E402

# Declared before execution, applied to outputs, gradients and cached logits.
TOLERANCES = {"float32": {"atol": 3e-5, "rtol": 3e-4},
              "bfloat16": {"atol": 2e-3, "rtol": 2e-2}}


def compare(actual: torch.Tensor, expected: torch.Tensor, tolerance: dict) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"parity tensors must have identical shapes: {actual.shape} versus {expected.shape}")
    actual, expected = actual.detach().float(), expected.detach().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"finite": False, "passed": False, "max_absolute_error": None,
                "max_tolerance_ratio": None}
    error = (actual - expected).abs()
    ratio = error / (tolerance["atol"] + tolerance["rtol"] * expected.abs())
    return {"finite": True, "passed": bool((ratio <= 1).all()),
            "max_absolute_error": error.max().item(), "max_tolerance_ratio": ratio.max().item()}


def run_gate(backend: str, device: str, dtype: str, chunk_size: int,
             lengths: list[int], seed: int, ratio: str = "1:3") -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA device is unavailable")
    cfg = ModelConfig(ratio=ratio, d_model=64, n_layers=4, vocab_size=128, head_dim=16,
                      mamba_headdim=16, d_state=16, mlp_multiple_of=16)
    if cfg.n_mamba_layers == 0:
        raise ValueError("a scan gate requires at least one Mamba layer")
    tolerance = TOLERANCES[dtype]
    torch.manual_seed(seed)
    reference = HybridLM(cfg).to(device)
    reference_metadata = reference.configure_scan_backend("reference", chunk_size)
    candidate = deepcopy(reference)
    metadata = candidate.configure_scan_backend(backend, chunk_size)
    def precision():
        return (torch.autocast(device_type=device, dtype=torch.bfloat16)
                if dtype == "bfloat16" else nullcontext())
    cases = []
    for length in lengths:
        tokens = torch.randint(0, cfg.vocab_size, (2, length), device=device)
        targets = torch.randint(0, cfg.vocab_size, (2, length), device=device)
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        with precision():
            a, a_loss = reference(tokens, targets)
            b, b_loss = candidate(tokens, targets)
        a_loss.backward()
        b_loss.backward()
        gradients = []
        for (name, p), (other, q) in zip(reference.named_parameters(), candidate.named_parameters()):
            if name != other or p.grad is None or q.grad is None:
                raise RuntimeError(f"missing or mismatched parameter gradient: {name}")
            gradients.append({"parameter": name, **compare(q.grad, p.grad, tolerance)})
        with torch.no_grad(), precision():
            # A prompt ending before a chunk boundary plus a separate decode call.
            split = max(1, length - 3)
            cache_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
            cached, state = candidate.prefill(tokens[:, :split], cache_dtype=cache_dtype)
            _, reference_state = reference.prefill(tokens[:, :split], cache_dtype=cache_dtype)
            prefill = compare(cached, a[:, split - 1:split], tolerance)
            decoded = None
            if split < length:
                decoded_positions = []
                for position in range(split, length):
                    decoded_positions.append(compare(candidate.decode(tokens[:, position:position + 1], state),
                                                     a[:, position:position + 1], tolerance))
                    reference.decode(tokens[:, position:position + 1], reference_state)
                decoded = {"passed": all(p["passed"] for p in decoded_positions),
                           "tokenwise_positions": decoded_positions}
            state_checks = []
            for layer, (actual_state, expected_state) in enumerate(zip(state.layers, reference_state.layers)):
                fields = ("conv", "ssm") if hasattr(actual_state, "ssm") else ("key", "value")
                for field in fields:
                    state_checks.append({"layer": layer, "field": field,
                                         **compare(getattr(actual_state, field),
                                                   getattr(expected_state, field), tolerance)})
        worst = max(gradients, key=lambda g: g["max_tolerance_ratio"]
                    if g["max_tolerance_ratio"] is not None else float("inf"))
        case = {"length": length, "logits": compare(b, a, tolerance),
                "loss": compare(b_loss, a_loss, tolerance),
                "gradient_tensors": len(gradients), "worst_gradient": worst,
                "all_gradients_passed": all(g["passed"] for g in gradients),
                "prefill_logits": prefill, "decode_logits": decoded,
                "retained_state": state_checks,
                "final_position": state.position}
        case["passed"] = (case["logits"]["passed"] and case["loss"]["passed"]
                          and case["all_gradients_passed"] and prefill["passed"]
                          and all(s["passed"] for s in state_checks)
                          and (decoded is None or decoded["passed"]) and state.position == length)
        cases.append(case)
    model_dir = Path(__file__).resolve().parents[1] / "src/model"
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(model_dir.glob("*.py"))}
    source_hashes["scripts/check_scan_backend.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    versions = {}
    for package in ("mamba-ssm", "triton", "causal-conv1d"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"schema": 1, "scope": "tiny-model same-weight parity; no research training",
            "status": "passed" if all(c["passed"] for c in cases) else "failed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "model_config": asdict(cfg),
            "device": device, "dtype": dtype, "tolerances": tolerance,
            "backend": metadata, "observed_paths": metadata["paths"],
            "reference_backend": reference_metadata,
            "source_sha256": source_hashes, "python": platform.python_version(),
            "optional_packages": versions,
            "precision_flags": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "cases": cases,
            "limits": ["Small random model and listed lengths only.",
                       "Not a speed benchmark, checkpoint certification, or quality comparison.",
                       "A portable backend pass does not certify fused_mamba."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "torch_chunked", "fused_mamba"),
                        default="torch_chunked")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=tuple(TOLERANCES), default="float32")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 15, 16, 17, 33])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ratio", default="1:3", help="also test pure controls with 0:1 or 1:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(length <= 0 for length in args.lengths):
        parser.error("all lengths must be positive")
    try:
        report = run_gate(args.backend, args.device, args.dtype, args.chunk_size,
                          args.lengths, args.seed, args.ratio)
    except BackendUnavailableError as exc:
        report = {"schema": 1, "status": "unavailable", "backend": args.backend,
                  "reason": str(exc), "device": args.device, "dtype": args.dtype,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "observed_paths": None, "certified": False}
    except (RuntimeError, ValueError) as exc:
        report = {"schema": 1, "status": "failed", "backend": args.backend,
                  "reason": f"{type(exc).__name__}: {exc}", "device": args.device,
                  "dtype": args.dtype, "ratio": args.ratio,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "observed_paths": None, "certified": False}
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
