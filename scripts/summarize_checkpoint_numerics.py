"""Strict, read-only compact export of declared trained-checkpoint diagnostics.

Only standard-library code runs here. Stored tolerance flags, full parameter
coverage, real continuation shapes and descriptive per-token statistics are
validated before a new summary is published. Failed checks remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
KIND = "trained_checkpoint_numerics"
TOLERANCES = {"float32": {"atol": 3e-5, "rtol": 3e-4}, "bfloat16": {"atol": 2e-3, "rtol": 2e-2}}
RATIOS = ["1:3", "1:7", "1:15"]
SOURCES = {"src/model/attention.py", "src/model/block.py", "src/model/config.py", "src/model/inference.py",
           "src/model/lm.py", "src/model/mamba2.py", "src/model/mlp.py", "src/model/norm.py", "src/model/scan_backend.py",
           "scripts/study_checkpoint_numerics.py", "scripts/check_scan_backend.py", "scripts/count_params.py",
           "src/eval/suite.py", "src/data/dataset.py", "src/data/prepare_data.py", "src/data/train_tokenizer.py"}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        require(len(items) == len({key for key, _ in items}), "duplicate JSON keys")
        return dict(items)
    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def finite_tree(value):
        if isinstance(value, dict):
            for item in value.values():
                finite_tree(item)
        elif isinstance(value, list):
            for item in value:
                finite_tree(item)
        elif isinstance(value, float):
            require(math.isfinite(value), "nonfinite JSON number")
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(value, dict), "JSON root must be an object")
    finite_tree(value)
    return value, hashlib.sha256(raw).hexdigest()


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"invalid SHA256: {label}")


def relative_path(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value
            and not PurePosixPath(value).is_absolute() and all(part not in (".", "..") for part in PurePosixPath(value).parts),
            "unsafe/nonrelative provenance path")
    return value


def number(value, label, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value) and (minimum is None or value >= minimum), f"invalid finite number: {label}")


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid integer: {label}")


def flag(value, expected, label):
    require(type(value) is bool and value == expected, f"inconsistent passed flag: {label}")


def metric(value, label):
    require(isinstance(value, dict) and type(value.get("finite")) is bool, f"invalid finite flag: {label}")
    if value["finite"]:
        number(value.get("max_absolute_error"), label + ".error", 0)
        number(value.get("max_tolerance_ratio"), label + ".ratio", 0)
        require((value["max_absolute_error"] == 0) == (value["max_tolerance_ratio"] == 0), f"inconsistent zero errors: {label}")
        passed = value["max_tolerance_ratio"] <= 1
    else:
        require(value.get("max_absolute_error") is None and value.get("max_tolerance_ratio") is None, f"nonfinite errors must be null: {label}")
        passed = False
    flag(value.get("passed"), passed, label)
    return passed


def parameter_inventory(cfg):
    d, inner, state = cfg["d_model"], cfg["d_model"] * cfg["expand"], cfg["d_state"]
    heads, conv = inner // cfg["mamba_headdim"], inner + 2 * state
    hidden = ((int(cfg["mlp_ratio"] * d) + cfg["mlp_multiple_of"] - 1) // cfg["mlp_multiple_of"]) * cfg["mlp_multiple_of"]
    shapes = {"embed.weight": [cfg["vocab_size"], d]}
    for layer, kind in enumerate(cfg["layer_types"]):
        prefix = f"blocks.{layer}."
        shapes[prefix + "norm1.weight"] = [d]
        if kind == "attention":
            shapes[prefix + "mixer.qkv.weight"] = [3 * d, d]
            shapes[prefix + "mixer.out.weight"] = [d, d]
            if cfg["attn_bias"]:
                shapes[prefix + "mixer.qkv.bias"] = [3 * d]
                shapes[prefix + "mixer.out.bias"] = [d]
        else:
            for name in ("dt_bias", "A_log", "D"):
                shapes[prefix + "mixer." + name] = [heads]
            shapes[prefix + "mixer.in_proj.weight"] = [2 * inner + 2 * state + heads, d]
            shapes[prefix + "mixer.conv1d.weight"] = [conv, 1, cfg["d_conv"]]
            shapes[prefix + "mixer.conv1d.bias"] = [conv]
            shapes[prefix + "mixer.norm.weight"] = [inner]
            shapes[prefix + "mixer.out_proj.weight"] = [d, inner]
        if cfg["mlp_on_every_layer"]:
            shapes[prefix + "norm2.weight"] = [d]
            for name, shape in (("gate", [hidden, d]), ("up", [hidden, d]), ("down", [d, hidden])):
                shapes[prefix + f"mlp.{name}.weight"] = shape
                if cfg["mlp_bias"]:
                    shapes[prefix + f"mlp.{name}.bias"] = [shape[0]]
    shapes["norm_f.weight"] = [d]
    if not cfg["tie_embeddings"]:
        shapes["lm_head.weight"] = [cfg["vocab_size"], d]
    return shapes


def state_inventory(cfg, length):
    d, inner = cfg["d_model"], cfg["d_model"] * cfg["expand"]
    fields = {}
    for layer, kind in enumerate(cfg["layer_types"]):
        if kind == "attention":
            for name in ("key", "value"):
                fields[f"layer_{layer}.{name}"] = ([1, d // cfg["head_dim"], length, cfg["head_dim"]], "torch.bfloat16")
        else:
            fields[f"layer_{layer}.conv"] = ([1, inner + 2 * cfg["d_state"], cfg["d_conv"] - 1], "torch.bfloat16")
            fields[f"layer_{layer}.ssm"] = ([1, inner // cfg["mamba_headdim"], cfg["mamba_headdim"], cfg["d_state"]], "torch.float32")
    return fields


def validate_state(value, inventory, label):
    require(isinstance(value, dict) and isinstance(value.get("fields"), list), f"missing state comparisons: {label}")
    names = [item["field"] for item in value["fields"]]
    require(len(names) == len(set(names)) and set(names) == set(inventory), f"missing/duplicate state fields: {label}")
    passes = []
    for item in value["fields"]:
        require(item["shape"] == inventory[item["field"]][0], f"state shape differs: {label}")
        passes.append(metric(item, label + "." + item["field"]))
    flag(value.get("passed"), all(passes), label)
    return all(passes)


def close_double(actual, expected, label):
    number(actual, label)
    require(math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), f"descriptive statistic inconsistent: {label}")


def close_fp32_mean(actual, values, label):
    # Serialization checks allow only FP32 reduction rounding, never an extra
    # equivalence margin on model outputs or on language quality.
    number(actual, label)
    expected = math.fsum(values) / len(values)
    bound = 4 * 2**-23 * max(math.fsum(abs(value) for value in values) / len(values), 1e-12)
    require(abs(actual - expected) <= bound, f"FP32 descriptive mean inconsistent: {label}")


def validate_distribution(value, values, label):
    require(set(value) == {"mean", "minimum", "maximum", "mean_absolute", "p95_absolute"}, f"distribution fields differ: {label}")
    absolute = sorted(abs(item) for item in values)
    rank = 0.95 * (len(values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    expected = {"mean": math.fsum(values) / len(values), "minimum": min(values), "maximum": max(values),
                "mean_absolute": math.fsum(absolute) / len(values),
                "p95_absolute": absolute[lo] + (rank - lo) * (absolute[hi] - absolute[lo])}
    for name, item in expected.items():
        close_double(value[name], item, label + "." + name)


def validate_effect(value, targets, start, label):
    require(value["scope"] == "descriptive arithmetic effects; no quality-equivalence cutoff"
            and value["metric_dtype"] == "torch.float32" and "passed" not in value, f"probability effect cannot claim equivalence: {label}")
    require(value["scored_tokens"] == len(targets) and value["position_start"] == start
            and value["position_stop_exclusive"] == start + len(targets), f"probability positions/counts differ: {label}")
    per = value["per_token"]
    require(set(per) == {"target_ids", "actual_logprob", "reference_logprob", "logprob_delta", "greedy_agreement"}
            and per["target_ids"] == targets, f"shifted target identities differ: {label}")
    require(all(isinstance(items, list) and len(items) == len(targets) for items in per.values()), f"per-token counts differ: {label}")
    for name in ("actual_logprob", "reference_logprob", "logprob_delta"):
        for item in per[name]:
            number(item, label + "." + name)
            if name != "logprob_delta":
                require(item <= 0, f"logprob must be nonpositive: {label}")
    differences = [struct.unpack("<f", struct.pack("<f", actual - reference))[0]
                   for actual, reference in zip(per["actual_logprob"], per["reference_logprob"])]
    require(per["logprob_delta"] == differences, f"per-token logprob delta inconsistent: {label}")
    validate_distribution(value["true_next_token_logprob_delta"], differences, label + ".delta")
    close_fp32_mean(value["actual_mean_nll"], [-item for item in per["actual_logprob"]], label + ".actual_nll")
    close_fp32_mean(value["reference_mean_nll"], [-item for item in per["reference_logprob"]], label + ".reference_nll")
    close_fp32_mean(value["mean_nll_delta"], [-item for item in differences], label + ".nll_delta")
    require(all(type(item) is bool for item in per["greedy_agreement"]), f"invalid greedy flags: {label}")
    disagreement = sum(not item for item in per["greedy_agreement"])
    require(type(value["greedy_disagreements"]) is int and value["greedy_disagreements"] == disagreement, f"greedy count inconsistent: {label}")
    close_fp32_mean(value["greedy_agreement"], [float(item) for item in per["greedy_agreement"]], label + ".greedy_agreement")
    for name in ("actual_top1_top2_margin", "reference_top1_top2_margin"):
        stats = value[name]
        require(set(stats) == {"mean", "minimum", "maximum", "mean_absolute", "p95_absolute"}, f"margin fields differ: {label}")
        for item in stats.values():
            number(item, label + "." + name, 0)
        require(stats["minimum"] <= stats["mean"] <= stats["maximum"]
                and stats["minimum"] <= stats["p95_absolute"] <= stats["maximum"], f"margin range inconsistent: {label}")
        close_double(stats["mean"], stats["mean_absolute"], label + ".margin_mean_absolute")


def validate_backend(value, name, cfg, chunk):
    path = "torch.quadratic_ssd" if name == "reference" else "torch.chunked_ssd"
    require(value["requested"] == value["resolved"] == name and value["chunk_size"] == chunk
            and value["mamba_layers"] == cfg["layer_types"].count("mamba") and value["fused"] is False,
            "backend identity differs")
    require(value["paths"] == {"training": path, "prefill": "torch.chunked_ssd", "decode": "torch.chunked_ssd"}, "intended scan paths differ")


def validate_operators(case, cfg, length, prefix, chunk):
    full_chunks = Counter(min(chunk, length - start) for start in range(0, length, chunk))
    suffix = length - prefix
    suffix_chunks = Counter(min(chunk, suffix - start) for start in range(0, suffix, chunk))
    stages = {"fp32_reference_forward": Counter({length: 1}), "fp32_torch_chunked_forward": full_chunks,
              "bf16_full": Counter({length: 1}), "bf16_stateful_full": full_chunks,
              "bf16_prefix_prefill": Counter({prefix: 1}), "bf16_suffix_one_shot": suffix_chunks,
              "bf16_suffix_segmented": suffix_chunks, "bf16_suffix_tokenwise": Counter({1: suffix}),
              "bf16_full_tokenwise": Counter({1: length})}
    mamba = [layer for layer, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]
    seen = {}
    expected_paths = {}
    for event in case["operator_observations"]:
        stage, layer = event["stage"], event["layer"]
        require(stage in stages and type(layer) is int and layer in mamba, "unexpected operator stage/layer")
        path = "torch.quadratic_ssd" if stage in ("fp32_reference_forward", "bf16_full") else "torch.chunked_ssd"
        require(event["path"] == path and event["autocast"] is stage.startswith("bf16_"), "actual operator path/precision differs")
        dtypes = (["torch.bfloat16", "torch.float32", "torch.float32", "torch.bfloat16", "torch.bfloat16", "torch.float32"]
                  if stage.startswith("bf16_") else ["torch.float32"] * 6)
        if path == "torch.chunked_ssd":
            dtypes += ["torch.float32"]
        require(event["input_dtypes"] == dtypes, "operator input dtypes differ")
        integer(event["calls"], "operator calls", 1)
        integer(event["length"], "operator length", 1)
        key = (stage, layer, event["length"])
        require(key not in seen, "duplicate operator observation")
        seen[key] = event["calls"]
        expected_paths[stage] = [path]
    expected = {(stage, layer, size): count for stage, parts in stages.items() for layer in mamba for size, count in parts.items()}
    require(seen == expected and case["actual_paths"] == expected_paths, "missing/incorrect actual operator calls or paths")


def validate_case(case, protocol, checkpoint, targets):
    cfg = checkpoint["model_config"]
    length, prefix, chunk = protocol["length"], protocol["prefix_length"], protocol["chunk_size"]
    input_row = protocol["shared_inputs"][0]
    require(case["ratio"] == checkpoint["ratio"] and all(case[key] == input_row[key]
            for key in ("window", "start_token", "length", "tokens_sha256", "targets_sha256", "target_policy")), "case shared input identity differs")
    digest(case["weights_sha256"], "weights")
    number(case["processing_seconds"], "processing seconds", 0)
    integer(case["cuda_peak_allocated_bytes"], "CUDA peak", 1)
    fp32, bf16 = case["fp32_backend"], case["bf16_cached"]
    require(fp32["tolerance"] == TOLERANCES["float32"] and bf16["tolerance"] == TOLERANCES["bfloat16"], "case tolerances changed")
    validate_backend(fp32["reference_backend"], "reference", cfg, chunk)
    validate_backend(fp32["candidate_backend"], "torch_chunked", cfg, chunk)
    validate_backend(bf16["backend"], "reference", cfg, chunk)
    require(fp32["observed_paths"] == {"prefill": None, "decode": None,
             "backward": "autograd through each recorded forward scan; no separate kernel probe"}, "FP32 unexecuted-path labels differ")
    logits_pass, loss_pass = metric(fp32["logits"], "FP32 logits"), metric(fp32["loss"], "FP32 loss")
    gradients = fp32["gradients"]
    require(gradients["tolerance"] == TOLERANCES["float32"], "gradient tolerance changed")
    inventory = parameter_inventory(cfg)
    checks = gradients["checks"]
    names = [check["parameter"] for check in checks]
    require(len(names) == len(set(names)) and set(names) == set(inventory)
            and type(gradients["parameter_tensors"]) is int and gradients["parameter_tensors"] == len(inventory), "missing/duplicate parameter gradient checks")
    passes = []
    for check in checks:
        require(check["shape"] == inventory[check["parameter"]], "parameter gradient shape differs")
        passes.append(metric(check, "gradient." + check["parameter"]))
        if check["finite"]:
            number(check["actual_l2"], "gradient norm", 0)
            number(check["reference_l2"], "gradient reference norm", 0)
            if check["actual_l2"] and check["reference_l2"]:
                number(check["cosine_similarity"], "gradient cosine")
                require(-1 <= check["cosine_similarity"] <= 1, "invalid gradient cosine")
            else:
                require(check["cosine_similarity"] is None, "zero gradient cosine must be null")
        else:
            require(all(check[name] is None for name in ("actual_l2", "reference_l2", "cosine_similarity")), "nonfinite gradient statistics must be null")
    require(gradients["failed_parameters"] == [check["parameter"] for check in checks if not check["passed"]], "failed gradient names differ")
    flag(gradients["all_passed"], all(passes), "all gradients")
    flag(fp32["passed"], logits_pass and loss_pass and all(passes), "FP32 aggregate")
    validate_effect(fp32["next_token_effects"], targets, 0, "FP32 backend effects")
    internal_keys = {"stateful_full", "prefix", "suffix_one_shot", "suffix_segmented", "suffix_tokenwise", "full_tokenwise",
                     "suffix_tokenwise_vs_one_shot", "suffix_segmented_vs_one_shot"}
    require(set(bf16["internal_logits"]) == internal_keys, "missing/extra BF16 logits comparisons")
    internal_passes = [metric(value, "BF16." + name) for name, value in bf16["internal_logits"].items()]
    state_keys = {"suffix_one_shot", "suffix_segmented", "suffix_tokenwise", "full_tokenwise"}
    require(set(bf16["retained_state"]) == state_keys, "missing/extra BF16 state comparisons")
    states = [validate_state(value, state_inventory(cfg, length), "BF16 state." + name) for name, value in bf16["retained_state"].items()]
    flag(bf16["internal_passed"], all(internal_passes) and all(states), "BF16 internal")
    anchor = metric(bf16["fp32_anchor"], "BF16 FP32 anchor")
    flag(bf16["passed"], bf16["internal_passed"] and anchor, "BF16 aggregate")
    require(bf16["cache_dtype"] == "torch.bfloat16" and bf16["suffix_length"] == length - prefix
            and bf16["one_shot_vs_tokenwise_degenerate"] is False
            and bf16["segmented_schedule"] == [128, 1] and bf16["unexecuted_stages"] == [], "real continuation schedule differs")
    require(bf16["final_positions"] == {name: length for name in ("stateful_full", *sorted(state_keys))}, "final continuation positions differ")
    real = bf16["real_prefix"]
    require(real["position"] == prefix and real["synthetic"] is False
            and real["unchanged_after_cloned_continuations"] is True, "real prefix clone facts differ")
    inventory_state = state_inventory(cfg, prefix)
    names = [field["field"] for field in real["fields"]]
    require(len(names) == len(set(names)) and set(names) == set(inventory_state), "missing/duplicate real prefix fields")
    for field in real["fields"]:
        shape, dtype = inventory_state[field["field"]]
        require(field["shape"] == shape and field["dtype"] == dtype, "real prefix shape/dtype differs")
        digest(field["sha256"], "real prefix field")
        integer(field["nonzero_elements"], "real prefix nonzero count")
        require(field["nonzero_elements"] <= math.prod(shape), "invalid real prefix nonzero count")
        if field["field"].endswith(".ssm"):
            require(field["nonzero_elements"] > 0, "real prefix SSM memory is zero-only")
    effects = bf16["next_token_effects"]
    require(set(effects) == {"stateful_full", "prefix", "suffix_one_shot", "suffix_segmented", "suffix_tokenwise", "full_tokenwise", "full_vs_fp32_anchor"}, "missing probability effects")
    for name, effect in effects.items():
        start = prefix - 1 if name == "prefix" else prefix if name.startswith("suffix_") else 0
        stop = prefix if name == "prefix" else length
        validate_effect(effect, targets[start:stop], start, "BF16 effects." + name)
    full_bf16_lp = effects["full_vs_fp32_anchor"]["per_token"]["actual_logprob"]
    require(effects["full_vs_fp32_anchor"]["per_token"]["reference_logprob"]
            == fp32["next_token_effects"]["per_token"]["reference_logprob"], "common FP32 anchor logprob identities differ")
    for name, effect in effects.items():
        if name == "full_vs_fp32_anchor":
            continue
        start, stop = effect["position_start"], effect["position_stop_exclusive"]
        require(effect["per_token"]["reference_logprob"] == full_bf16_lp[start:stop], "internal BF16 reference logprob identities differ")
    validate_operators(case, cfg, length, prefix, chunk)
    flag(case["passed"], fp32["passed"] and bf16["passed"], "case aggregate")


def validate_report(report, declaration, source_root):
    require(declaration["schema"] == 1 and declaration["status_at_declaration"] == "planned_before_execution", "invalid declaration")
    require([item["ratio"] for item in declaration["checkpoints"]] == RATIOS, "declaration checkpoint matrix differs")
    require(report["schema"] == 1 and report["kind"] == KIND and report["certified"] is False
            and report["execution_status"] == "completed", "report incomplete or incorrectly certified")
    protocol = report["protocol"]
    require(report["protocol_sha256"] == canonical_sha256(protocol), "protocol hash differs")
    require(protocol["kind"] == KIND and protocol["version"] == 1 and protocol["ratios"] == RATIOS, "protocol ratio matrix differs")
    expected = {"device": "cuda", "seed": 2027, "length": 257, "prefix_length": 128,
                "windows": 1, "batch_size": 1, "chunk_size": 128, "full_tokenwise": True,
                "training_run_id": "week3-700m-v1"}
    require(all(protocol[key] == value and declaration[key] == value for key, value in expected.items()), "protocol differs from declared workload")
    require(protocol["tolerances"] == declaration["tolerances"] == TOLERANCES, "original tolerances changed")
    require(protocol["quality_equivalence_margin"] is None and declaration["quality_equivalence_margin"] is None, "quality margin must remain null")
    policy = protocol["precision_policy"]
    require(policy == {"fp32_backend": "reference versus torch_chunked; autocast disabled",
        "bf16_cached": "reference backend; BF16 autocast; naturally produced cloned prefix states",
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}, "precision policy differs")
    runtime = protocol["runtime"]
    require(protocol["runtime_sha256"] == canonical_sha256(runtime), "runtime hash differs")
    require(runtime["device"] == "cuda" and isinstance(runtime["gpu"], str) and runtime["gpu"], "runtime CUDA identity missing")
    require(set(runtime["packages"]) == {"torch", "numpy", "tokenizers", "mamba-ssm", "triton", "causal-conv1d"}, "runtime package coverage differs")
    for name in ("torch", "numpy", "tokenizers"):
        require(isinstance(runtime["packages"][name], str) and runtime["packages"][name], "required package version missing")
    require(all(runtime["precision_flags"][key] is value and declaration["precision_flags"][key] is value
                for key, value in {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}.items()), "TF32 flags differ")
    source_root = Path(source_root).resolve()
    require(set(protocol["source_sha256"]) == SOURCES, "source coverage differs")
    for path, value in protocol["source_sha256"].items():
        digest(value, "source")
        raw = (source_root / relative_path(path)).read_bytes()
        # Git may materialize the same text as LF or CRLF on another host. Keep
        # the recorded byte hash intact and accept only those two text versions.
        lf = raw.replace(b"\r\n", b"\n")
        hashes = {hashlib.sha256(data).hexdigest() for data in (raw, lf, lf.replace(b"\n", b"\r\n"))}
        require(value in hashes, "recorded source hash differs from source checkout")
    data = protocol["data"]
    require(data["selection_scope"] == "checkpoint-selection validation pool; not an independent quality test", "validation scope differs")
    data_manifest_path = source_root / relative_path(data["manifest"]["path"])
    data_manifest, manifest_hash = read_json(data_manifest_path)
    require(data["manifest"]["sha256"] == manifest_hash and data["signature"] == data_manifest["signature"]
            and data["build_signature"] == data_manifest["build_signature"], "data manifest identity differs")
    require(set(data["artifacts"]) == {"train", "val", "meta"}, "data artifact coverage differs")
    for name, item in data["artifacts"].items():
        digest(item["sha256"], "data artifact")
        expected_item = data_manifest["outputs"][name]
        require(item["sha256"] == expected_item["sha256"] and item["bytes"] == expected_item["bytes"], "data registry differs")
        require((source_root / relative_path(item["path"])).stat().st_size == item["bytes"], "data artifact byte count differs")
    tokenizer = data["tokenizer"]
    require(tokenizer["sha256"] == data_manifest["tokenizer"]["sha256"]
            and tokenizer["vocab_size"] == data_manifest["tokenizer"]["vocab_size"]
            and tokenizer["eot_id"] == data_manifest["tokenizer"]["eot_id"]
            and file_sha256(source_root / relative_path(tokenizer["path"])) == tokenizer["sha256"], "tokenizer identity differs")
    require(len(protocol["shared_inputs"]) == 1, "shared input coverage differs")
    shared = protocol["shared_inputs"][0]
    require(shared["window"] == 0 and shared["length"] == 257 and shared["target_policy"] == "y[i] = validation[start_token+i+1]", "shared input protocol differs")
    integer(shared["start_token"], "shared start")
    for key in ("tokens_sha256", "targets_sha256"):
        digest(shared[key], "shared input")
    with (source_root / relative_path(data["artifacts"]["val"]["path"])).open("rb") as handle:
        handle.seek(shared["start_token"] * 2)
        raw = handle.read((shared["length"] + 1) * 2)
    require(len(raw) == (shared["length"] + 1) * 2, "shared window exceeds validation data")
    values = list(struct.unpack("<" + "H" * (shared["length"] + 1), raw))
    require(max(values) < tokenizer["vocab_size"], "shared window vocabulary differs")
    for key, items in (("tokens_sha256", values[:-1]), ("targets_sha256", values[1:])):
        require(hashlib.sha256(struct.pack("<" + "q" * len(items), *items)).hexdigest() == shared[key], "shared window byte/hash identity differs")
    require([item["ratio"] for item in protocol["checkpoints"]] == RATIOS
            and [case["ratio"] for case in report["cases"]] == RATIOS, "missing/duplicate report checkpoint cases")
    for checkpoint, expected_checkpoint, case in zip(protocol["checkpoints"], declaration["checkpoints"], report["cases"]):
        require(all(checkpoint[key] == expected_checkpoint[key] for key in ("ratio", "path", "sha256")), "checkpoint differs from pre-run declaration")
        digest(checkpoint["sha256"], "checkpoint")
        digest(checkpoint["training_signature"], "training signature")
        relative_path(checkpoint["path"])
        manifest, sha = read_json(source_root / relative_path(checkpoint["training_manifest"]["path"]))
        require(sha == checkpoint["training_manifest"]["sha256"] and manifest["signature"] == checkpoint["training_signature"]
                and manifest["status"] == "completed" and manifest["ratio"] == checkpoint["ratio"]
                and manifest["artifact_sha256"]["best"] == checkpoint["sha256"]
                and manifest["model_config"] == checkpoint["model_config"], "checkpoint registry/config identity differs")
        cfg = checkpoint["model_config"]
        require(cfg["ratio"] == checkpoint["ratio"] and cfg["n_groups"] == 1 and cfg["vocab_size"] == tokenizer["vocab_size"], "checkpoint geometry differs")
        require(checkpoint["parameter_count"] == sum(math.prod(shape) for shape in parameter_inventory(cfg).values()), "parameter count differs")
        require(checkpoint["historical_training_data_match"] is True, "historical data linkage missing")
        for name, item in data["artifacts"].items():
            historical = manifest["data"]["files"][data_manifest["outputs"][name]["file"]]
            require(historical["sha256"] == item["sha256"] and historical["bytes"] == item["bytes"], "historical training data linkage differs")
        validate_case(case, protocol, checkpoint, values[1:])
    status = "completed" if all(case["passed"] for case in report["cases"]) else "completed_with_parity_failures"
    require(report["status"] == status, "overall report status hides failed checks")


def count_checks(values):
    values = list(values)
    return {"passed": sum(value["passed"] for value in values), "total": len(values)}


def compact_effect(value):
    return {key: item for key, item in value.items() if key != "per_token"}


def build_summary(declaration_path: Path, raw_path: Path | None = None, root: Path = ROOT,
                  source_root: Path = ROOT) -> dict:
    declaration, declaration_hash = read_json(declaration_path)
    root = Path(root).resolve()
    expected_path = relative_path(declaration["output"])
    raw_path = root / expected_path if raw_path is None else Path(raw_path).resolve()
    require(raw_path == root / expected_path, "raw report path differs from declaration")
    report, raw_hash = read_json(raw_path)
    validate_report(report, declaration, source_root)
    rows = []
    for case in report["cases"]:
        fp32, bf16 = case["fp32_backend"], case["bf16_cached"]
        gradients = fp32["gradients"]
        worst = max(gradients["checks"], key=lambda item: item["max_tolerance_ratio"] if item["max_tolerance_ratio"] is not None else math.inf)
        state_fields = [field for group in bf16["retained_state"].values() for field in group["fields"]]
        rows.append({"ratio": case["ratio"], "window": case["window"], "passed": case["passed"],
                     "weights_sha256": case["weights_sha256"],
                     "fp32": {"passed": fp32["passed"], "logits": fp32["logits"], "loss": fp32["loss"],
                         "gradients": {**count_checks(gradients["checks"]), "all_passed": gradients["all_passed"],
                             "failed_parameters": gradients["failed_parameters"], "worst_gradient": worst},
                         "next_token_effects": compact_effect(fp32["next_token_effects"])},
                     "bf16": {"passed": bf16["passed"], "internal_passed": bf16["internal_passed"],
                         "internal": {**count_checks(bf16["internal_logits"].values()),
                             "failed_checks": [name for name, value in bf16["internal_logits"].items() if not value["passed"]],
                             "checks": bf16["internal_logits"]},
                         "retained_state": {**count_checks(bf16["retained_state"].values()),
                             "fields": count_checks(state_fields),
                             "failed_fields": [{"comparison": name, "field": field["field"], **{key: field[key] for key in
                                 ("finite", "max_absolute_error", "max_tolerance_ratio")}}
                                 for name, group in bf16["retained_state"].items() for field in group["fields"] if not field["passed"]]},
                         "fp32_anchor": bf16["fp32_anchor"], "real_prefix": bf16["real_prefix"],
                         "suffix_length": bf16["suffix_length"], "segmented_schedule": bf16["segmented_schedule"],
                         "final_positions": bf16["final_positions"],
                         "next_token_effects": {name: compact_effect(value) for name, value in bf16["next_token_effects"].items()}},
                     "actual_paths": case["actual_paths"], "processing_seconds": case["processing_seconds"],
                     "cuda_peak_allocated_bytes": case["cuda_peak_allocated_bytes"]})
    try:
        declaration_reference = Path(declaration_path).resolve().relative_to(root).as_posix()
    except ValueError:
        declaration_reference = Path(declaration_path).name
    protocol = report["protocol"]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"],
            "execution_status": "completed", "certified": False, "scope": declaration["scope"],
            "declaration": {"path": declaration_reference, "sha256": declaration_hash, "canonical_sha256": canonical_sha256(declaration)},
            "raw_report": {"path": expected_path, "sha256": raw_hash, "canonical_sha256": canonical_sha256(report),
                           "cases": len(report["cases"]), "protocol_sha256": report["protocol_sha256"]},
            "protocol_sha256": report["protocol_sha256"], "shared_inputs": protocol["shared_inputs"],
            "data": protocol["data"], "checkpoints": protocol["checkpoints"], "source_sha256": protocol["source_sha256"],
            "runtime": protocol["runtime"], "runtime_sha256": protocol["runtime_sha256"],
            "tolerances": TOLERANCES, "quality_equivalence_margin": None, "rows": rows,
            "limits": list(dict.fromkeys([*declaration["limits"], *report["limits"],
                "Exporter rechecks source/registry identities and the natural window; binary checkpoint/training-stream hashes remain bound to the pre-run declaration and recorded integrity audit.",
                "Source verification recognizes LF/CRLF versions of the same text while preserving recorded byte hashes.",
                "Descriptive-statistic validation allows FP32 reduction rounding only; tensor tolerances and quality conclusions are unchanged."]))}


def write_summary(path: Path, summary: dict):
    path = Path(path)
    require(not path.exists(), "summary output already exists")
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new summary artifact")
    try:
        summary = build_summary(args.declaration, args.raw)
        write_summary(args.output, summary)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(f"invalid checkpoint numerics evidence: {exc}")
    print(json.dumps({"status": summary["status"], "rows": len(summary["rows"]),
                      "raw_sha256": summary["raw_report"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
