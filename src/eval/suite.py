"""Measured Week 4 evaluation primitives for the three certified hybrid checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tokenizers import Tokenizer

from src.data.dataset import get_batch, load_split
from src.model.config import ModelConfig
from src.model.lm import HybridLM


WEEK3_RATIOS = ("1:3", "1:7", "1:15")
NEEDLE_CODES = (
    "orchid", "lantern", "cobalt", "saffron", "maple",
    "quartz", "raven", "willow", "ember", "harbor",
    "violet", "cedar", "silver", "falcon", "meadow",
)
EVALUATION_SOURCE_FILES = (
    "scripts/run_week4_eval.py",
    "src/eval/report.py",
    "src/eval/suite.py",
    "src/model/attention.py",
    "src/model/block.py",
    "src/model/inference.py",
    "src/model/lm.py",
    "src/model/mamba2.py",
)


@dataclass(frozen=True)
class Week4Protocol:
    context_lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    decode_tokens: int = 32
    throughput_repeats: int = 3
    validation_block_size: int = 512
    validation_batch_size: int = 8
    validation_iters: int = 50
    needle_depths: tuple[float, ...] = (0.1, 0.5, 0.9)
    needle_decode_tokens: int = 8
    seed: int = 1337

    def validate(self, max_seq_len: int = 16384) -> None:
        if not self.context_lengths or tuple(sorted(set(self.context_lengths))) != self.context_lengths:
            raise ValueError("context lengths must be unique and increasing")
        if any(length <= 0 for length in self.context_lengths):
            raise ValueError("context lengths must be positive")
        if self.context_lengths[-1] + max(self.decode_tokens, self.needle_decode_tokens) > max_seq_len:
            raise ValueError("context plus decode exceeds the model's RoPE table")
        if self.decode_tokens <= 0 or self.needle_decode_tokens <= 0:
            raise ValueError("decode lengths must be positive")
        if self.throughput_repeats <= 0 or self.validation_iters <= 0:
            raise ValueError("repeat and validation counts must be positive")
        if self.validation_block_size <= 0 or self.validation_batch_size <= 0:
            raise ValueError("validation geometry must be positive")
        if any(not 0.0 <= depth <= 1.0 for depth in self.needle_depths):
            raise ValueError("needle depths must stay between zero and one")
        if len(self.context_lengths) * len(self.needle_depths) > len(NEEDLE_CODES):
            raise ValueError("the protocol needs more pre-registered needle codes")

    @classmethod
    def smoke(cls) -> "Week4Protocol":
        return cls(
            context_lengths=(128, 512),
            decode_tokens=4,
            throughput_repeats=1,
            validation_batch_size=2,
            validation_iters=2,
            needle_depths=(0.5,),
            needle_decode_tokens=4,
        )


@dataclass(frozen=True)
class VariantCheckpoint:
    ratio: str
    run_dir: Path
    best_path: Path
    checkpoint_sha256: str
    signature: str
    step: int
    val_loss: float
    model_config: dict[str, Any]

    def evidence(self, root: Path) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "path": self.best_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": self.checkpoint_sha256,
            "training_signature": self.signature,
            "step": self.step,
            "recorded_val_loss": self.val_loss,
        }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoints(
    checkpoint_root: Path, training_run_id: str, ratios: Iterable[str] = WEEK3_RATIOS
) -> dict[str, VariantCheckpoint]:
    run_root = checkpoint_root / training_run_id
    if not run_root.is_dir():
        raise FileNotFoundError(f"training run directory is missing: {run_root}")

    wanted = tuple(ratios)
    found: dict[str, VariantCheckpoint] = {}
    for manifest_path in sorted(run_root.glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        ratio = manifest.get("ratio")
        if ratio not in wanted:
            continue
        if ratio in found:
            raise RuntimeError(f"more than one completed checkpoint claims ratio {ratio}")
        if manifest.get("status") != "completed":
            raise RuntimeError(f"checkpoint manifest for {ratio} is not completed")
        artifacts = manifest.get("artifacts")
        hashes = manifest.get("artifact_sha256")
        if not isinstance(artifacts, dict) or not isinstance(hashes, dict):
            raise RuntimeError(f"checkpoint manifest for {ratio} has no artifact registry")
        best_name = artifacts.get("best")
        expected_hash = hashes.get("best")
        if not isinstance(best_name, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"checkpoint manifest for {ratio} has no registered best model")
        best_path = manifest_path.parent / best_name
        if not best_path.is_file() or file_sha256(best_path) != expected_hash:
            raise RuntimeError(f"best checkpoint for {ratio} failed its registered checksum")

        state = torch.load(best_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
            raise RuntimeError(f"best checkpoint for {ratio} has no model state")
        if state.get("signature") != manifest.get("signature"):
            raise RuntimeError(f"best checkpoint for {ratio} disagrees with its manifest signature")
        model_config = state.get("model_config")
        if not isinstance(model_config, dict) or model_config.get("ratio") != ratio:
            raise RuntimeError(f"best checkpoint for {ratio} has the wrong model config")
        step, val_loss = state.get("step"), state.get("val_loss")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise RuntimeError(f"best checkpoint for {ratio} has an invalid step")
        if not isinstance(val_loss, (int, float)) or not math.isfinite(float(val_loss)):
            raise RuntimeError(f"best checkpoint for {ratio} has an invalid validation loss")
        found[ratio] = VariantCheckpoint(
            ratio=ratio,
            run_dir=manifest_path.parent,
            best_path=best_path,
            checkpoint_sha256=expected_hash,
            signature=str(manifest["signature"]),
            step=step,
            val_loss=float(val_loss),
            model_config=model_config,
        )

    missing = set(wanted).difference(found)
    if missing:
        raise RuntimeError(f"completed checkpoints are missing for ratios: {sorted(missing)}")
    return found


def load_variant_model(checkpoint: VariantCheckpoint, device: torch.device) -> HybridLM:
    state = torch.load(checkpoint.best_path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint.model_config)
    model = HybridLM(config)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def _precision_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_allocated(device: torch.device) -> int:
    return torch.cuda.memory_allocated(device) if device.type == "cuda" else 0


@torch.no_grad()
def validation_perplexity(
    model: HybridLM,
    validation_data: np.ndarray,
    protocol: Week4Protocol,
    device: torch.device,
) -> dict[str, float | int]:
    generator = torch.Generator(device="cpu").manual_seed(protocol.seed + 2)
    losses = []
    for _ in range(protocol.validation_iters):
        x, y = get_batch(
            validation_data,
            protocol.validation_block_size,
            protocol.validation_batch_size,
            str(device),
            generator=generator,
        )
        with _precision_context(device):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    mean_loss = statistics.fmean(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "batches": protocol.validation_iters,
        "batch_size": protocol.validation_batch_size,
        "block_size": protocol.validation_block_size,
        "token_positions": (
            protocol.validation_iters
            * protocol.validation_batch_size
            * protocol.validation_block_size
        ),
    }


def validation_prompt(
    validation_data: np.ndarray, context_length: int, seed: int, index: int = 0
) -> torch.Tensor:
    available = len(validation_data) - context_length
    if available <= 0:
        raise ValueError("validation split is shorter than the requested prompt")
    start = (seed * 104729 + context_length * 8191 + index * 65537) % available
    values = np.array(validation_data[start:start + context_length], dtype=np.int64, copy=True)
    return torch.from_numpy(values)[None, :]


@torch.no_grad()
def _timed_inference_once(
    model: HybridLM,
    prompt: torch.Tensor,
    decode_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline = _cuda_allocated(device)
    cache_dtype = torch.bfloat16 if device.type == "cuda" else model.embed.weight.dtype

    _sync(device)
    prefill_start = time.perf_counter()
    with _precision_context(device):
        logits, state = model.prefill(prompt, cache_dtype=cache_dtype)
    _sync(device)
    prefill_seconds = time.perf_counter() - prefill_start
    allocated_after_prefill = _cuda_allocated(device)
    peak_after_prefill = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    state_breakdown = state.byte_breakdown()

    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    _sync(device)
    decode_start = time.perf_counter()
    with _precision_context(device):
        for _ in range(decode_tokens):
            logits = model.decode(next_token, state)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    _sync(device)
    decode_seconds = time.perf_counter() - decode_start

    result = {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "state_breakdown": state_breakdown,
        "allocated_delta_bytes": max(0, allocated_after_prefill - baseline),
        "peak_delta_bytes": max(0, peak_after_prefill - baseline),
    }
    del logits, next_token, state
    return result


@torch.no_grad()
def benchmark_inference(
    model: HybridLM,
    validation_data: np.ndarray,
    protocol: Week4Protocol,
    device: torch.device,
) -> list[dict[str, Any]]:
    warmup_prompt = validation_prompt(validation_data, min(128, protocol.context_lengths[0]), protocol.seed)
    warmup_prompt = warmup_prompt.to(device)
    _timed_inference_once(model, warmup_prompt, min(2, protocol.decode_tokens), device)

    rows = []
    for context_length in protocol.context_lengths:
        prompt = validation_prompt(validation_data, context_length, protocol.seed).to(device)
        repeats = [
            _timed_inference_once(model, prompt, protocol.decode_tokens, device)
            for _ in range(protocol.throughput_repeats)
        ]
        prefill_seconds = statistics.median(row["prefill_seconds"] for row in repeats)
        decode_seconds = statistics.median(row["decode_seconds"] for row in repeats)
        logical_state = repeats[0]["state_breakdown"]
        rows.append({
            "context_length": context_length,
            "decode_tokens": protocol.decode_tokens,
            "repeats": protocol.throughput_repeats,
            "prefill_seconds_median": prefill_seconds,
            "prefill_tokens_per_second": context_length / prefill_seconds,
            "decode_seconds_median": decode_seconds,
            "decode_tokens_per_second": protocol.decode_tokens / decode_seconds,
            "logical_state_bytes": sum(logical_state.values()),
            "attention_kv_bytes": logical_state["attention_kv"],
            "mamba_conv_bytes": logical_state["mamba_conv"],
            "mamba_ssm_bytes": logical_state["mamba_ssm"],
            "allocated_delta_bytes_median": int(statistics.median(
                row["allocated_delta_bytes"] for row in repeats
            )),
            "peak_delta_bytes_median": int(statistics.median(
                row["peak_delta_bytes"] for row in repeats
            )),
            "raw_timings": [{
                "prefill_seconds": row["prefill_seconds"],
                "decode_seconds": row["decode_seconds"],
            } for row in repeats],
        })
        del prompt
    return rows


FILLER_TEXT = (
    "Research note: the measurement was repeated under the same controlled conditions. "
    "The observer recorded the sample, checked the timestamp, and continued to the next entry. "
    "No conclusion was changed while the document was assembled.\n"
)


def build_needle_prompt(
    tokenizer: Tokenizer, context_length: int, depth: float, code: str
) -> tuple[list[int], list[int], float]:
    if not 0.0 <= depth <= 1.0:
        raise ValueError("needle depth must be between zero and one")
    lead = tokenizer.encode(
        "This document contains research records. Read every record before answering.\n"
    ).ids
    needle = tokenizer.encode(f"Important record: The access code is {code}.\n").ids
    query = tokenizer.encode(
        "Question: What is the access code? Answer: The access code is"
    ).ids
    target = tokenizer.encode(f" {code}").ids
    filler_budget = context_length - len(lead) - len(needle) - len(query)
    if filler_budget < 0:
        raise ValueError("context is too short for the fixed needle template")

    filler_unit = tokenizer.encode(FILLER_TEXT).ids
    filler = (filler_unit * math.ceil(max(1, filler_budget) / len(filler_unit)))[:filler_budget]
    split = int(round(depth * filler_budget))
    prompt = lead + filler[:split] + needle + filler[split:] + query
    if len(prompt) != context_length:
        raise RuntimeError("needle prompt builder did not preserve the requested token length")
    actual_depth = (len(lead) + split) / max(1, context_length - len(query))
    return prompt, target, actual_depth


def normalize_answer(text: str) -> str:
    match = re.search(r"[a-z0-9]+", text.lower())
    return "" if match is None else match.group(0)


@torch.no_grad()
def evaluate_needle_retrieval(
    model: HybridLM,
    tokenizer: Tokenizer,
    protocol: Week4Protocol,
    device: torch.device,
) -> dict[str, Any]:
    trials = []
    code_index = 0
    cache_dtype = torch.bfloat16 if device.type == "cuda" else model.embed.weight.dtype
    for context_length in protocol.context_lengths:
        for depth in protocol.needle_depths:
            code = NEEDLE_CODES[code_index]
            code_index += 1
            prompt_ids, target_ids, actual_depth = build_needle_prompt(
                tokenizer, context_length, depth, code
            )
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
            with _precision_context(device):
                prompt_logits, generation_state = model.prefill(
                    prompt, cache_dtype=cache_dtype
                )
            teacher_state = generation_state.clone()

            generated = []
            next_token = prompt_logits[:, -1].argmax(dim=-1, keepdim=True)
            for step in range(protocol.needle_decode_tokens):
                generated.append(int(next_token.item()))
                if step + 1 < protocol.needle_decode_tokens:
                    with _precision_context(device):
                        next_logits = model.decode(next_token, generation_state)
                    next_token = next_logits[:, -1].argmax(dim=-1, keepdim=True)

            token_nll = []
            logits = prompt_logits[:, -1].float()
            for target_index, target_id in enumerate(target_ids):
                token_nll.append(float(-torch.log_softmax(logits, dim=-1)[0, target_id].item()))
                if target_index + 1 < len(target_ids):
                    teacher_token = torch.tensor([[target_id]], dtype=torch.long, device=device)
                    with _precision_context(device):
                        logits = model.decode(teacher_token, teacher_state)[:, -1].float()

            generated_text = tokenizer.decode(generated)
            normalized = normalize_answer(generated_text)
            trials.append({
                "context_length": context_length,
                "requested_depth": depth,
                "actual_depth": actual_depth,
                "code": code,
                "generated_token_ids": generated,
                "generated_text": generated_text,
                "normalized_answer": normalized,
                "exact_match": normalized == code,
                "target_token_nll": statistics.fmean(token_nll),
            })
            del prompt, prompt_logits, generation_state, teacher_state, next_token, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    matches = sum(trial["exact_match"] for trial in trials)
    by_context = []
    for context_length in protocol.context_lengths:
        context_trials = [row for row in trials if row["context_length"] == context_length]
        by_context.append({
            "context_length": context_length,
            "matches": sum(row["exact_match"] for row in context_trials),
            "trials": len(context_trials),
            "accuracy": statistics.fmean(float(row["exact_match"]) for row in context_trials),
            "mean_target_token_nll": statistics.fmean(
                row["target_token_nll"] for row in context_trials
            ),
        })
    return {
        "matches": matches,
        "trials": len(trials),
        "accuracy": matches / len(trials),
        "by_context": by_context,
        "trial_records": trials,
    }


def git_provenance(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, encoding="utf-8"
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def runtime_provenance(device: torch.device) -> dict[str, Any]:
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        runtime["gpu"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        }
    return runtime


def evaluation_source_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for relative in EVALUATION_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"evaluation source file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def evaluate_variant(
    checkpoint: VariantCheckpoint,
    validation_data: np.ndarray,
    tokenizer: Tokenizer,
    protocol: Week4Protocol,
    device: torch.device,
) -> dict[str, Any]:
    model = load_variant_model(checkpoint, device)
    started = time.perf_counter()
    quality = validation_perplexity(model, validation_data, protocol, device)
    performance = benchmark_inference(model, validation_data, protocol, device)
    retrieval = evaluate_needle_retrieval(model, tokenizer, protocol, device)
    result = {
        "name": model.cfg.name,
        "ratio": model.cfg.ratio,
        "parameters": model.num_params(),
        "attention_layers": model.cfg.n_attention_layers,
        "mamba_layers": model.cfg.n_mamba_layers,
        "checkpoint": checkpoint.evidence(Path.cwd()),
        "validation": quality,
        "inference": performance,
        "needle_retrieval": retrieval,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
