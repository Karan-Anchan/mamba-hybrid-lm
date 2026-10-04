"""Temporary RoPE-only BF16 diagnostics; no production model changes or certification.

Replay actual pre-rotation operands with the original full/cached table policies,
then test explicitly matched table policies on the same learned weights and tokens.
The separate FP32 scan discrepancy is outside this experiment's explanation scope.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import scripts.study_checkpoint_numerics as base  # noqa: E402
from scripts.check_scan_backend import TOLERANCES, compare  # noqa: E402
from src.model.attention import apply_rope  # noqa: E402
from src.model.scan_backend import BackendUnavailableError, resolve_scan_backend  # noqa: E402

KIND = "rope_precision_isolation"
POLICIES = {
    "original": "Only cached attention rounds FP32 rotation tables to the query dtype",
    "shared_fp32": "Both routes rotate in FP32, then explicitly cast query/key back to the original dtype",
    "shared_bf16": "Both routes round rotation tables to BF16 before rotation",
}


def tensor_identity(value: torch.Tensor) -> dict:
    base.require_finite(value, tuple(value.shape), "frozen RoPE operand")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": base.tensor_sha256(value)}


def rotate_operands(q, k, cos, sin, policy: str, cached: bool, *, final_cast: bool = True):
    """Return the specified treatment, not an inferred SDPA kernel's internals."""
    if policy not in POLICIES:
        raise ValueError("unknown RoPE policy")
    dtype = q.dtype
    if policy == "original":
        if cached:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return apply_rope(q, k, cos, sin)
    if policy == "shared_fp32":
        rotated = apply_rope(q.float(), k.float(), cos.float(), sin.float())
        return tuple(value.to(dtype) for value in rotated) if final_cast else rotated
    return apply_rope(q, k, cos.bfloat16(), sin.bfloat16())


@torch.no_grad()
def frozen_rope_probe(q, k, cos, sin) -> dict:
    """Same BF16 q/k and FP32 tables; full versus cached is the only replay factor."""
    if (q.ndim != 4 or q.shape != k.shape or q.dtype != torch.bfloat16 or k.dtype != q.dtype
            or q.shape[-1] % 2 or cos.shape != sin.shape
            or tuple(cos.shape) != tuple(q.shape[-2:]) or cos.dtype != torch.float32
            or sin.dtype != torch.float32 or any(value.device != q.device for value in (k, cos, sin))):
        raise ValueError("frozen probe needs equal BF16 (batch,heads,tokens,even features) q/k and matching FP32 tables")
    operands = {name: tensor_identity(value) for name, value in (("q", q), ("k", k), ("cos", cos), ("sin", sin))}
    rows = []
    with torch.autocast(q.device.type, enabled=False):
        for policy in POLICIES:
            full = rotate_operands(q, k, cos, sin, policy, False, final_cast=False)
            cached = rotate_operands(q, k, cos, sin, policy, True, final_cast=False)
            full_cast = tuple(value.to(q.dtype) for value in full)
            cached_cast = tuple(value.to(q.dtype) for value in cached)
            rows.append({"policy": policy,
                "post_rotation": {"full": {name: tensor_identity(value) for name, value in zip(("q", "k"), full)},
                                  "cached": {name: tensor_identity(value) for name, value in zip(("q", "k"), cached)},
                                  "comparisons": {name: compare(b, a, TOLERANCES["bfloat16"]) for name, a, b in zip(("q", "k"), full, cached)},
                                  "bitwise_equal": all(torch.equal(a, b) for a, b in zip(full, cached))},
                "post_cast": {"comparison_dtype": str(q.dtype),
                              "scope": "explicit replay cast; the original stateless model has no added explicit cast",
                              "comparisons": {name: compare(b, a, TOLERANCES["bfloat16"]) for name, a, b in zip(("q", "k"), full_cast, cached_cast)},
                              "bitwise_equal": all(torch.equal(a, b) for a, b in zip(full_cast, cached_cast))}})
    return {"scope": "actual frozen pre-RoPE operands; no recomputation of upstream layers",
            "operands": operands, "policies": rows,
            "limit": "Matched frozen routes agree by construction. This cannot approve full-model routes, gradients, quality or the FP32 scan gate."}


def _attention_forward(module, x, cos, sin, cache, policy: str):
    """Mirror local attention with only the declared rotation treatment changed."""
    batch, length, _ = x.shape
    q, k, v = module.qkv(x).split(x.shape[-1], dim=-1)
    q, k, v = [value.view(batch, length, module.nheads, module.hd).transpose(1, 2) for value in (q, k, v)]
    q, k = rotate_operands(q, k, cos, sin, policy, cache is not None)
    cached = cache.length if cache is not None else 0
    if cached:
        k, v = torch.cat([cache.key, k], 2), torch.cat([cache.value, v], 2)
    if cache is not None:
        cache.key, cache.value = k, v
    mask = None
    if cached and length > 1:
        queries = cached + torch.arange(length, device=x.device)
        keys = torch.arange(cached + length, device=x.device)
        mask = keys[None, :] <= queries[:, None]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=cached == 0)
    return module.out(y.transpose(1, 2).reshape(batch, length, -1))


@contextmanager
def rope_policy(model, policy: str, recorder: dict):
    """Patch attention instances only and restore forwards/hooks even on failure."""
    if policy not in POLICIES:
        raise ValueError("unknown RoPE policy")
    restored, hooks = [], []
    try:
        for layer, block in enumerate(model.blocks):
            if not block.is_attn:
                continue
            module = block.mixer
            original, had_own = module.forward, "forward" in module.__dict__
            context = {}
            restored.append((module, original, had_own))

            def capture(_module, _inputs, output, _context=context, _layer=layer):
                if policy != "original" or recorder["stage"] != "bf16_full" or _layer in recorder["frozen"]:
                    return
                x, cos, sin = _context["inputs"]
                batch, length, features = x.shape
                heads, width = _context["geometry"]
                q, k, _v = output.split(features, -1)
                q, k = [value.view(batch, length, heads, width).transpose(1, 2).detach() for value in (q, k)]
                recorder["frozen"][_layer] = frozen_rope_probe(q, k, cos.detach(), sin.detach())

            hooks.append(module.qkv.register_forward_hook(capture))

            def forward(x, cos, sin, cache=None, _module=module, _original=original, _context=context, _layer=layer):
                _context["inputs"], _context["geometry"] = (x, cos, sin), (_module.nheads, _module.hd)
                identity = json.dumps({"layer": _layer, "stage": recorder["stage"], "policy": policy,
                                       "cached": cache is not None, "length": x.shape[1],
                                       "tables_dtype_before_treatment": str(cos.dtype)}, sort_keys=True)
                recorder["attention_events"][identity] = recorder["attention_events"].get(identity, 0) + 1
                return _original(x, cos, sin, cache) if policy == "original" else _attention_forward(_module, x, cos, sin, cache, policy)

            module.forward = forward
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for module, original, had_own in reversed(restored):
            if had_own:
                module.forward = original
            else:
                delattr(module, "forward")


@torch.no_grad()
def study_model(model, x, y, *, prefix_length: int, chunk_size: int, tokenwise: bool = True,
                full_model: bool = True) -> dict:
    if not any(block.is_attn for block in model.blocks):
        raise ValueError("RoPE isolation requires an attention layer")
    initial_weights = base.weight_sha256(model)
    policies, frozen = [], []
    with base.preserve_model_execution(model):
        model.configure_scan_backend("reference", chunk_size)
        with torch.autocast(x.device.type, enabled=False):
            anchor, _ = model(x)
        base.require_finite(anchor, (x.shape[0], x.shape[1], model.cfg.vocab_size), "unchanged FP32 anchor")
        anchor = anchor.detach().cpu()
        for policy in POLICIES:
            recorder = {"stage": "unexecuted", "events": {}, "frozen": {}, "attention_events": {}}
            if x.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            with rope_policy(model, policy, recorder):
                if full_model:
                    result = base.bf16_cache_probe(model, x, y, anchor, prefix_length, chunk_size, tokenwise, recorder)
                elif policy == "original":
                    recorder["stage"] = "bf16_full"
                    with torch.autocast(x.device.type, dtype=torch.bfloat16):
                        model(x)
                    result = None
                else:
                    result = None
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            if policy == "original":
                frozen = [{"layer": layer, **probe} for layer, probe in sorted(recorder["frozen"].items())]
            policies.append({"policy": policy, "description": POLICIES[policy], "bf16_cached": result,
                "execution_status": "completed" if full_model or policy == "original" else "unexecuted",
                "attention_observations": [{**json.loads(key), "calls": calls} for key, calls in sorted(recorder["attention_events"].items())],
                "scan_observations": [{**json.loads(key), "calls": calls} for key, calls in sorted(recorder["events"].items())],
                "processing_seconds": time.perf_counter() - start,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if x.device.type == "cuda" else None})
        if base.weight_sha256(model) != initial_weights:
            raise RuntimeError("RoPE treatment changed learned weights")
    if len(frozen) != sum(block.is_attn for block in model.blocks):
        raise RuntimeError("missing actual-operand frozen probe for an attention layer")
    return {"weights_sha256": initial_weights,
            "fp32_anchor": {"scope": "unchanged original stateless reference, autocast disabled",
                            "logits_sha256": base.tensor_sha256(anchor)},
            "frozen_probes": frozen, "policies": policies,
            "full_model_passed": all(row["bf16_cached"]["passed"] for row in policies) if full_model else None}


def _bound_evidence(protocol_path: Path | None, baseline_path: Path | None, settings: dict):
    declaration, baseline, identities = None, None, {}
    if protocol_path is not None:
        protocol_path = Path(protocol_path)
        declaration = base.read_json(protocol_path)
        if (declaration.get("schema") != 1 or declaration.get("status_at_declaration") != "planned_before_execution"
                or list(declaration.get("policies", {})) != list(POLICIES)
                or declaration.get("tolerances") != TOLERANCES):
            raise ValueError("declaration has changed policies, status or tolerances")
        for name in ("device", "seed", "length", "prefix_length", "chunk_size", "ratios"):
            expected = list(settings[name]) if name == "ratios" else settings[name]
            if declaration.get(name) != expected:
                raise ValueError(f"declaration differs for {name}")
        if settings["windows"] != 1 or declaration.get("batch_size") != 1:
            raise ValueError("bound declaration requires one window and batch size one")
        if declaration.get("precision_flags") != {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}:
            raise ValueError("declaration differs for TF32 policy")
        identities["declaration"] = {**base.public_reference(protocol_path), "sha256": base.file_sha256(protocol_path)}
        baseline_path = baseline_path or ROOT / declaration["baseline_report"]
    if baseline_path is not None:
        baseline_path = Path(baseline_path)
        digest = base.file_sha256(baseline_path)
        if declaration is not None and digest != declaration["baseline_sha256"]:
            raise ValueError("baseline report hash differs from declaration")
        baseline = base.read_json(baseline_path)
        if baseline.get("kind") != base.KIND or baseline.get("execution_status") != "completed" or baseline.get("certified") is not False:
            raise ValueError("baseline must be a completed uncertified natural-window diagnostic")
        identities["baseline_report"] = {**base.public_reference(baseline_path), "sha256": digest}
    return declaration, baseline, identities


def run_study(device: str = "cpu", seed: int = 2027, length: int = 257,
              prefix_length: int = 128, windows: int = 1, chunk_size: int = 128,
              tokenwise: bool = True, full_model: bool = True,
              checkpoint_root: Path = Path("checkpoints"), training_run_id: str = "week3-700m-v1",
              data_dir: Path = Path("data/openwebtext-5b"), tokenizer: Path = Path("data/tokenizer/openwebtext.json"),
              ratios: tuple = base.WEEK3_RATIOS, protocol: Path | None = None,
              baseline_report: Path | None = None) -> dict:
    if device not in ("cpu", "cuda") or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("device must be cpu/cuda and seed an integer in 0..2**32-1")
    if type(length) is not int or not 2 <= length <= 257 or type(prefix_length) is not int or not 1 <= prefix_length < length:
        raise ValueError("length must be 2..257 and prefix_length inside that sequence")
    if type(windows) is not int or not 1 <= windows <= 4 or windows * length > 1028:
        raise ValueError("use 1..4 windows and at most 1028 tokens per checkpoint")
    if type(tokenwise) is not bool or type(full_model) is not bool:
        raise ValueError("tokenwise and full_model must be boolean")
    if not isinstance(training_run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", training_run_id):
        raise ValueError("training_run_id must be one directory name")
    ratios = tuple(ratios)
    if not ratios or len(ratios) > 3 or any(not isinstance(ratio, str) for ratio in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("select one to three distinct ratio strings")
    resolve_scan_backend("reference", chunk_size)
    settings = {"device": device, "seed": seed, "length": length, "prefix_length": prefix_length,
                "chunk_size": chunk_size, "windows": windows, "ratios": ratios}
    declaration, baseline, bindings = _bound_evidence(protocol, baseline_report, settings)
    checkpoints, data_evidence, checkpoint_evidence = base.audit_inputs(data_dir, tokenizer, checkpoint_root, training_run_id, ratios)
    rows = base.shared_windows(base.load_split(data_dir, "val"), length, windows, seed)
    for _, _, identity in rows:
        if declaration is not None and any(identity[name] != declaration[name] for name in ("start_token", "tokens_sha256", "targets_sha256")):
            raise ValueError("natural window differs from declared start/token/target hashes")
    if baseline is not None:
        for name in ("device", "seed", "length", "prefix_length", "chunk_size"):
            if baseline["protocol"][name] != settings[name]:
                raise ValueError(f"baseline differs for {name}")
        if baseline["protocol"]["shared_inputs"] != [identity for _, _, identity in rows]:
            raise ValueError("baseline input identities differ")
        for checkpoint in checkpoint_evidence:
            if not any(saved["ratio"] == checkpoint["ratio"] and saved["sha256"] == checkpoint["sha256"]
                       for saved in baseline["protocol"]["checkpoints"]):
                raise ValueError("baseline checkpoint identities differ")
    vocab_size = data_evidence["tokenizer"]["vocab_size"]
    if any(bool((value < 0).any() or (value >= vocab_size).any()) for x, y, _ in rows for value in (x, y)):
        raise ValueError("window has IDs outside the audited vocabulary")
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA unavailable; no CPU fallback")
    cases = []
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), base.tf32_disabled():
        runtime = base.runtime_metadata(device)
        for ratio in ratios:
            model = base.load_variant_model(checkpoints[ratio], torch.device(device))
            try:
                for name, parameter in model.named_parameters():
                    base.require_finite(parameter, tuple(parameter.shape), f"loaded parameter {name}")
                for x_cpu, y_cpu, identity in rows:
                    result = study_model(model, x_cpu.to(device), y_cpu.to(device), prefix_length=prefix_length,
                                         chunk_size=chunk_size, tokenwise=tokenwise, full_model=full_model)
                    if baseline is not None and full_model:
                        prior = next(row for row in baseline["cases"] if row["ratio"] == ratio and row["window"] == identity["window"])
                        original = result["policies"][0]["bf16_cached"]
                        keys = ("internal_logits", "retained_state", "fp32_anchor", "next_token_effects")
                        result["baseline_control_comparison"] = {
                            "scope": "exact saved numerical fields; equality is descriptive, not certification",
                            "original_fields_sha256": base.canonical_sha256({key: original[key] for key in keys}),
                            "baseline_fields_sha256": base.canonical_sha256({key: prior["bf16_cached"][key] for key in keys}),
                        }
                        check = result["baseline_control_comparison"]
                        check["exact_fields_equal"] = check["original_fields_sha256"] == check["baseline_fields_sha256"]
                    cases.append({"ratio": ratio, **identity, **result})
            finally:
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
    for ratio in ratios:
        if base.file_sha256(checkpoints[ratio].best_path) != checkpoints[ratio].checkpoint_sha256:
            raise RuntimeError("historical checkpoint bytes changed")
    protocol_record = {"kind": KIND, "version": 1, **settings, "ratios": list(ratios), "batch_size": 1,
        "full_model": full_model, "full_tokenwise": tokenwise, "training_run_id": training_run_id,
        "policies": POLICIES, "shared_inputs": [identity for _, _, identity in rows],
        "data": data_evidence, "checkpoints": checkpoint_evidence, **bindings,
        "source_sha256": {**base.source_metadata(), "scripts/study_rope_precision.py": base.file_sha256(Path(__file__))},
        "runtime": runtime, "runtime_sha256": base.canonical_sha256(runtime),
        "tolerances": TOLERANCES, "quality_equivalence_margin": None,
        "resource_scope": "per-policy inference, operand replay, hashing and checks; not throughput"}
    failures = full_model and any(not row["full_model_passed"] for row in cases)
    return {"schema": 1, "kind": KIND, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_with_parity_failures" if failures else "completed", "execution_status": "completed",
        "certified": False, "protocol": protocol_record, "protocol_sha256": base.canonical_sha256(protocol_record), "cases": cases,
        "limits": ["Temporary instance-scoped treatments; no production model/default or learned weight change.",
                   "Shared FP32 includes an explicit final cast: it is not every detail of historical stateless arithmetic.",
                   "Frozen matched-policy agreement is by construction and does not approve full-model endpoints.",
                   "BF16 RoPE inconsistency cannot explain the separate FP32 scan disagreement.",
                   "No optimizer, training, parameter-gradient test, fused kernel or backend certification.",
                   "Checkpoint-selection validation pool; descriptive NLL/greedy effects, no independent quality claim.",
                   "Original tolerances and all endpoint failures retained; diagnostic time/memory cannot rank throughput."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--length", type=int, default=257)
    parser.add_argument("--prefix-length", type=int, default=128)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--skip-full-tokenwise", action="store_true")
    parser.add_argument("--frozen-only", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--training-run-id", default="week3-700m-v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/openwebtext-5b"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/openwebtext.json"))
    parser.add_argument("--ratios", nargs="+", default=list(base.WEEK3_RATIOS))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new diagnostic artifact")
    try:
        report = run_study(device=args.device, seed=args.seed, length=args.length, prefix_length=args.prefix_length,
                           windows=args.windows, chunk_size=args.chunk_size, tokenwise=not args.skip_full_tokenwise,
                           full_model=not args.frozen_only, checkpoint_root=args.checkpoint_root,
                           training_run_id=args.training_run_id, data_dir=args.data_dir, tokenizer=args.tokenizer,
                           ratios=tuple(args.ratios), protocol=args.protocol, baseline_report=args.baseline_report)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        reason = f"{type(error).__name__}: {error}"
        for path in sorted((ROOT, Path.home(), args.checkpoint_root.resolve(), args.data_dir.resolve(), args.tokenizer.resolve()), key=lambda value: len(str(value)), reverse=True):
            reason = reason.replace(str(path), base.public_reference(path)["path"])
        report = {"schema": 1, "kind": KIND, "status": "unavailable" if isinstance(error, BackendUnavailableError) else "incomplete",
                  "execution_status": "incomplete", "certified": False, "reason": reason,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if args.output:
        base.write_report(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
