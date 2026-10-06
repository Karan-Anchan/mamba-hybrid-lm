"""Independent standard-library export of the declared projection isolation.

Measured producer modules, Torch, CUDA and private checkpoint/corpus binaries
are never loaded. Historical report bytes remain immutable; public gzip is a
lossless transport whose decompressed bytes must match the declared original.
"""
from __future__ import annotations

import gzip
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_normalization_head_precision as norm  # noqa: E402

base = norm.base
same = norm.same
isolation = norm.isolation
KIND = "trained_checkpoint_projection_precision"
RATIOS, CASES, CASE_IDS = norm.RATIOS, norm.CASES, norm.CASE_IDS
CELLS = ["P0", "P1"]
ANCHORS = ["own", "original", "decay_baseline", "projection_baseline"]
PAIRINGS = [*norm.PAIRINGS, "full_vs_projection_baseline", "tokenwise_vs_projection_baseline"]
ENDPOINTS = [*norm.ENDPOINTS, "full_vs_projection_baseline", "tokenwise_vs_projection_baseline", "stateful_full_vs_projection_baseline", "full_chunked_vs_projection_baseline"]
LOSS_ENDPOINTS = ["full_vs_original", "full_vs_decay_baseline", "full_vs_projection_baseline", "chunked_vs_own_full", "chunked_vs_original", "chunked_vs_decay_baseline", "chunked_vs_projection_baseline"]
PROJECTION_CHECKS = ["observed_shape_replay_vs_observed", "p0_tokenwise_vs_p0_full", "p0_full_vs_fp64", "p0_tokenwise_vs_fp64", "p1_full_vs_p0_full", "p1_tokenwise_vs_p1_full", "p1_full_vs_fp64", "p1_tokenwise_vs_fp64"]
SOURCES = norm.SOURCES | {"scripts/study_projection_precision.py"}
MAX_PUBLIC_JSON_BYTES = 256 * 1024**2
RAW_PUBLICATION = "docs/research/projection-precision-raw-publication-2026-10-06.json"
RAW_TRANSPORT_SCOPE = "Complete original raw trace; lossless transport only. No scientific record or declared comparison changed."
TRANSPORT_KEYS = {"schema", "kind", "compression", "path", "bytes", "sha256", "uncompressed_path", "uncompressed_bytes",
                  "uncompressed_sha256", "round_trip_exact", "scope"}


def repository_path(root, relative):
    """Keep every public identity canonical and inside the declared repository."""
    root = Path(root).resolve()
    path = (root / norm.relative(relative)).resolve()
    base.require(path.is_relative_to(root), "public evidence path resolves outside repository")
    return path


def parse_public_json(raw):
    def pairs(items):
        base.require(len(items) == len({key for key, _value in items}), "duplicate JSON keys")
        return dict(items)
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def finite_tree(value):
        if isinstance(value, dict):
            for item in value.values():
                finite_tree(item)
        elif isinstance(value, list):
            for item in value:
                finite_tree(item)
        elif isinstance(value, float):
            base.require(math.isfinite(value), "nonfinite JSON number")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    base.require(isinstance(value, dict), "public JSON root must be an object")
    finite_tree(value)
    return value


def read_public_json(record, *, root=ROOT, maximum_bytes=MAX_PUBLIC_JSON_BYTES):
    """Validate exact public JSON bytes, without line-ending normalization."""
    norm.reference(record)
    path = repository_path(root, record["path"])
    base.integer(maximum_bytes, "public JSON byte bound", 1)
    base.require(path.stat().st_size <= maximum_bytes, "public JSON exceeds byte bound")
    raw = path.read_bytes()
    same(hashlib.sha256(raw).hexdigest(), record["sha256"], "public JSON immutable bytes differ")
    return parse_public_json(raw)


def read_lossless_prior(record, transport_record, *, root=ROOT, maximum_bytes=MAX_PUBLIC_JSON_BYTES,
                        transport_kind="lossless_normalization_head_raw_publication",
                        transport_scope="Lossless transport of the original immutable report; no JSON reserialization or evidence removal."):
    """Check the registered gzip and independently hash its exact decoded bytes.

    The original JSON need not be materialized in a public clone. If it exists,
    check that its byte identity also agrees. No fallback normalizes/reserializes
    JSON or writes a decompressed copy into the repository.
    """
    norm.reference(record)
    transport = read_public_json(transport_record, root=root, maximum_bytes=64 * 1024)
    norm.closed(transport, TRANSPORT_KEYS, "historical gzip transport")
    same(transport["schema"], 1, "historical transport schema differs")
    same(transport["kind"], transport_kind, "historical transport kind differs")
    same(transport["compression"], "gzip", "historical transport compression differs")
    same(transport["scope"], transport_scope, "historical transport scope differs")
    base.require(transport["round_trip_exact"] is True, "historical transport does not declare exact bytes")
    same(transport["uncompressed_path"], record["path"], "historical transport original path differs")
    same(transport["uncompressed_sha256"], record["sha256"], "historical transport original hash differs")
    same(transport["path"], record["path"] + ".gz", "historical gzip path differs")
    for key in ("bytes", "uncompressed_bytes"):
        base.integer(transport[key], "transport " + key, 1)
    base.digest(transport["sha256"], "compressed historical bytes")
    base.integer(maximum_bytes, "decompression byte bound", 1)
    base.require(transport["uncompressed_bytes"] <= maximum_bytes, "declared decompression exceeds byte bound")
    archive = repository_path(root, transport["path"])
    same(archive.stat().st_size, transport["bytes"], "compressed byte count differs")
    same(base.file_sha256(archive), transport["sha256"], "compressed historical byte hash differs")
    parts, decoded_hash, size = [], hashlib.sha256(), 0
    try:
        with gzip.open(archive, "rb") as handle:
            while True:
                # Read only one byte beyond the declared size; a gzip bomb or
                # incorrect manifest cannot allocate an unbounded report.
                remaining = transport["uncompressed_bytes"] - size
                part = handle.read(min(1024**2, remaining + 1))
                if not part:
                    break
                size += len(part)
                base.require(size <= transport["uncompressed_bytes"], "decoded bytes exceed exact declared size")
                decoded_hash.update(part)
                parts.append(part)
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError("invalid historical gzip bytes") from exc
    same(size, transport["uncompressed_bytes"], "decompressed byte count differs")
    same(decoded_hash.hexdigest(), record["sha256"], "decompressed historical byte hash differs")
    original = repository_path(root, record["path"])
    if original.exists():
        same(original.stat().st_size, size, "local original JSON byte count differs")
        same(base.file_sha256(original), record["sha256"], "local original immutable JSON bytes differ")
    return parse_public_json(b"".join(parts))


def read_raw_report(relative, *, root=ROOT, publication_path=None):
    """A public gzip preserves the declared original path and byte fingerprint.

    The transport is publication metadata, not another experimental factor.
    Scientific summaries retain the original JSON identity and do not change
    when a lossless companion is published later.
    """
    root = Path(root).resolve()
    original = repository_path(root, relative)
    explicit = publication_path is not None
    publication_path = repository_path(root, RAW_PUBLICATION) if publication_path is None else Path(publication_path).resolve()
    base.require(publication_path.is_relative_to(root), "raw publication manifest outside repository")
    if publication_path.exists():
        base.require(publication_path.stat().st_size <= 64 * 1024, "raw publication manifest exceeds byte bound")
        manifest, manifest_hash = base.read_json(publication_path)
        norm.closed(manifest, TRANSPORT_KEYS, "projection raw publication")
        same(manifest["uncompressed_path"], relative, "raw transport changes declared original path")
        record = {"path": relative, "path_scope": "repository-relative", "sha256": manifest["uncompressed_sha256"]}
        transport_record = {"path": publication_path.relative_to(root).as_posix(), "path_scope": "repository-relative", "sha256": manifest_hash}
        value = read_lossless_prior(record, transport_record, root=root, transport_kind="lossless_projection_precision_raw_publication", transport_scope=RAW_TRANSPORT_SCOPE)
        return value, record["sha256"]
    base.require(not explicit, "explicit raw publication manifest missing")
    base.require(original.stat().st_size <= MAX_PUBLIC_JSON_BYTES, "raw report exceeds public JSON byte bound")
    return base.read_json(original)


def projection_inventory(cfg):
    return [(f"blocks.{layer}.mixer.in_proj", layer) for layer, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]


def projection_width(cfg):
    inner = cfg["d_model"] * cfg["expand"]
    return 2 * inner + 2 * cfg.get("n_groups", 1) * cfg["d_state"] + inner // cfg["mamba_headdim"]


def projection_probe(value, cfg, batch, length, *, active_cell="P0", offset=0, schedule=None):
    norm.closed(value, ["active_cell", "input", "observed_output", "weight", "bias", "actual_calls", "position_offset", "replay_layout", "layout_contrast", "tokenwise_shape", "identical_operands", "input_unchanged", "oracle", "oracle_output", "checks", "outputs", "passed", "exploratory"], "frozen projection")
    shape, out = [batch, length, cfg["d_model"]], [batch, length, projection_width(cfg)]
    same(value["active_cell"], active_cell, "frozen projection policy differs")
    same(value["position_offset"], offset, "frozen projection offset differs")
    for key in ("identical_operands", "input_unchanged", "exploratory"):
        base.require(value[key] is True, "frozen projection control missing: " + key)
    isolation.identities(value["input"], shape)
    isolation.identities(value["observed_output"], out)
    isolation.identities(value["weight"], [out[-1], shape[-1]])
    # Registered historical Mamba projections have bias=False.
    base.require(value["bias"] is None, "unexpected input-projection bias")
    isolation.identities(value["oracle_output"], out, "torch.float64")
    same(value["oracle"], "independent detached CPU NumPy FP64 matrix multiplication and optional bias addition", "projection oracle is not independent CPU NumPy FP64")
    same(value["tokenwise_shape"], [batch, 1, cfg["d_model"]], "frozen token shape differs")
    calls = value["actual_calls"]
    base.require(isinstance(calls, list) and bool(calls), "projection actual calls missing")
    if schedule is not None:
        same([call["length"] for call in calls], schedule, "actual projection call partition differs")
    position = offset
    for call in calls:
        norm.closed(call, ["position", "length", "input_layout", "output_layout"], "projection actual call")
        same(call["position"], position, "actual projection call position differs")
        base.integer(call["length"], "projection call span", 1)
        position += call["length"]
        isolation.operand_layout(call["input_layout"], [batch, call["length"], shape[-1]])
        isolation.operand_layout(call["output_layout"], [batch, call["length"], out[-1]])
    same(position, offset + length, "actual projection calls do not cover operands")
    isolation.operand_layout(value["replay_layout"], shape)
    if len(calls) == 1:
        same(value["replay_layout"], calls[0]["input_layout"], "single-call projection layout was changed")
        same(value["layout_contrast"], "captured single-call layout restored", "single-call replay scope differs")
    else:
        same(value["replay_layout"], {"shape": shape, "stride": [length * shape[-1], shape[-1], 1], "storage_offset": 0, "dtype": "torch.float32"}, "combined replay is not declared contiguous layout")
        same(value["layout_contrast"], "combined contiguous full replay versus actual captured call layouts", "combined replay scope differs")
    same(list(value["checks"]), PROJECTION_CHECKS, "eight frozen projection leaves differ")
    for name, leaf in value["checks"].items():
        norm.detailed(leaf, out, reference="torch.float64" if name.endswith("vs_fp64") else "torch.float32")
    same(list(value["outputs"]), ["p0_full", "p0_tokenwise", "p1_full", "p1_tokenwise"], "projection output identity coverage differs")
    for output in value["outputs"].values():
        isolation.identities(output, out)
    base.flag(value["passed"], all(leaf["passed"] for leaf in value["checks"].values()), "frozen projection aggregate")


def prefix_leaves(value, *, frozen=True):
    rows = norm.prefix_leaves(value)
    if frozen:
        for probe in value["frozen_projections"]:
            rows += [probe["upstream_input_vs_own_full"], probe["observed_output_vs_own_full"], *probe["frozen_arithmetic"]["checks"].values()]
    return rows


def prefix_probe(value, cfg, batch, length, prefix, targets, cell_id):
    required = {"prefix_position", "suffix_length", "synthetic", "prefix_unchanged", "prefix_fields", "prefix_logits", "prefix_comparisons", "suffix_logits", "suffix_comparisons", "route_comparisons", "retained_state", "final_positions", "suffix_state_fields", "chunked_schedule", "next_token_effects_vs_original", "projection_sites", "frozen_projections", "execution_status", "failed_projection", "measured_routes_passed", "passed"}
    base.require(isinstance(value, dict) and required <= set(value) <= required | {"reason"}, "projection prefix schema differs")
    same((value["prefix_position"], value["suffix_length"]), (prefix, length - prefix), "genuine prefix positions differ")
    base.require(value["synthetic"] is False and value["prefix_unchanged"] is True, "synthetic or mutated prefix")
    isolation.state_identities(value["prefix_fields"], cfg, batch, prefix)
    isolation.identities(value["prefix_logits"], [batch, 1, cfg["vocab_size"]])
    same(list(value["prefix_comparisons"]), ANCHORS, "four genuine prefix anchors differ")
    for leaf in value["prefix_comparisons"].values():
        norm.detailed(leaf, [batch, 1, cfg["vocab_size"]])
    for key in ("suffix_logits", "suffix_comparisons", "retained_state", "final_positions", "suffix_state_fields", "next_token_effects_vs_original"):
        same(list(value[key]), norm.SUFFIX_ROUTES, "suffix route coverage differs: " + key)
    suffix = length - prefix
    for route in norm.SUFFIX_ROUTES:
        isolation.identities(value["suffix_logits"][route], [batch, suffix, cfg["vocab_size"]])
        same(list(value["suffix_comparisons"][route]), ANCHORS, "four suffix score anchors differ")
        for leaf in value["suffix_comparisons"][route].values():
            norm.detailed(leaf, [batch, suffix, cfg["vocab_size"]])
        same(list(value["retained_state"][route]), ANCHORS, "four suffix memory anchors differ")
        for bundle in value["retained_state"][route].values():
            norm.state_bundle(bundle, cfg, batch, length)
        same(value["final_positions"][route], length, "suffix final position differs")
        isolation.state_identities(value["suffix_state_fields"][route], cfg, batch, length)
        norm.effect(value["next_token_effects_vs_original"][route], batch, suffix, prefix, norm.suffix_targets(targets, batch, length, prefix))
    same(list(value["route_comparisons"]), ["chunked128_vs_one_shot", "tokenwise_vs_one_shot"], "suffix route comparisons differ")
    for leaf in value["route_comparisons"].values():
        norm.detailed(leaf, [batch, suffix, cfg["vocab_size"]])
    schedules = {"one_shot": [suffix], "chunked128": [min(128, suffix - start) for start in range(0, suffix, 128)], "tokenwise": [1] * suffix}
    same(value["chunked_schedule"], schedules["chunked128"], "suffix chunk boundaries differ")
    sites = projection_inventory(cfg)
    selected = sites[:1] if len(sites) == 1 else [sites[0], sites[-1]]
    same(value["projection_sites"], [name for name, _layer in selected], "genuine suffix projection inventory differs")
    expected = [(route, name, layer) for route in norm.SUFFIX_ROUTES for name, layer in selected]
    probes = value["frozen_projections"]
    base.require(isinstance(probes, list) and len(probes) <= len(expected), "extra prefix projection")
    same([(row["route"], row["module"], row["layer"]) for row in probes], expected[:len(probes)], "prefix frozen projection order/coverage differs")
    for row, (route, name, layer) in zip(probes, expected):
        norm.closed(row, ["module", "layer", "route", "origin", "prefix_position", "prefix_fields", "final_position", "upstream_input_vs_own_full", "observed_output_vs_own_full", "frozen_arithmetic"], "actual prefix projection")
        same(row["origin"], "actual_same_cell_prefix_continuation", "frozen prefix projection is synthetic")
        same(row["prefix_position"], prefix, "frozen prefix position differs")
        same(row["prefix_fields"], value["prefix_fields"], "frozen projection does not belong to same-cell cloned prefix")
        same(row["final_position"], length, "frozen continuation final position differs")
        norm.detailed(row["upstream_input_vs_own_full"], [batch, suffix, cfg["d_model"]])
        norm.detailed(row["observed_output_vs_own_full"], [batch, suffix, projection_width(cfg)])
        projection_probe(row["frozen_arithmetic"], cfg, batch, suffix, active_cell=cell_id, offset=prefix, schedule=schedules[route])
    complete = value["execution_status"] == "completed"
    base.require(value["execution_status"] in {"completed", "incomplete"}, "unknown prefix execution status")
    if complete:
        base.require(len(probes) == len(expected) and value["failed_projection"] is None and "reason" not in value, "complete prefix missing frozen sites")
    else:
        base.require(isinstance(value.get("reason"), str) and bool(value["reason"]) and len(probes) < len(expected), "incomplete prefix missing failure scope")
        route, name, _layer = expected[len(probes)]
        same(value["failed_projection"], {"route": route, "module": name}, "failed frozen projection identity differs")
    base.flag(value["measured_routes_passed"], all(leaf["passed"] for leaf in prefix_leaves(value, frozen=False)), "measured prefix route aggregate")
    base.flag(value["passed"], complete and value["measured_routes_passed"], "prefix completion aggregate")


def cell_leaves(value):
    return [*norm.cell_leaves(value), *value["loss_comparisons"].values(), *(leaf for row in value["actual_prefix"]["frozen_projections"] for leaf in [row["upstream_input_vs_own_full"], row["observed_output_vs_own_full"], *row["frozen_arithmetic"]["checks"].values()])]


def validate_cell(value, cell_id, cfg, identity):
    norm.closed(value, ["cell_id", "input_projection_fp64_matmul", "normalization_cell", "projection_sites", "execution_status", "whole_model_comparisons", "stage_comparisons", "retained_state", "stateful_full_retained_state", "actual_prefix", "final_positions", "logits", "losses", "loss_comparisons", "next_token_effects", "actual_paths", "backward_executed", "BF16_executed", "whole_model_passed", "all_recorded_comparisons_passed"], "projection cell")
    same(value["cell_id"], cell_id, "projection cell order differs")
    base.require(type(value["input_projection_fp64_matmul"]) is bool, "invalid projection factor")
    same(value["input_projection_fp64_matmul"], cell_id == "P1", "projection factor differs")
    same(value["normalization_cell"], "N1H0", "fixed normalization/head policy changed")
    same(value["projection_sites"], [name for name, _ in projection_inventory(cfg)], "input projection family differs")
    base.require(value["execution_status"] in {"completed", "incomplete"} and value["backward_executed"] is False and value["BF16_executed"] is False, "cell execution scope differs")
    batch, length, prefix = (identity[key] for key in ("batch_size", "length", "prefix_length"))
    shape = [batch, length, cfg["vocab_size"]]
    same(list(value["next_token_effects"]), ["full_vs_original", "tokenwise_vs_original", "tokenwise_vs_own_full"], "descriptive effect coverage differs")
    targets = value["next_token_effects"]["full_vs_original"]["per_token"]["target_ids"]
    base.require(isinstance(targets, list) and len(targets) == batch * length, "scored target coverage differs")
    for token in targets:
        base.integer(token, "scored target")
        base.require(token < cfg["vocab_size"], "scored target outside vocabulary")
    same(norm.target_hash(targets), identity["targets_sha256"], "immutable targets differ")
    same(len(identity["windows"]), batch, "batch window coverage differs")
    for row, window in enumerate(identity["windows"]):
        same(norm.target_hash(targets[row * length:(row + 1) * length]), window["targets_sha256"], "per-row immutable targets differ")
    same(list(value["whole_model_comparisons"]), ENDPOINTS, "fifteen score endpoints differ")
    for leaf in value["whole_model_comparisons"].values():
        norm.detailed(leaf, shape)
    same(list(value["loss_comparisons"]), LOSS_ENDPOINTS, "seven actual-route loss endpoints differ")
    for leaf in value["loss_comparisons"].values():
        norm.detailed(leaf, [])
    same(list(value["losses"]), ["full", "chunked128"], "actual loss identities differ")
    for row in value["losses"].values():
        isolation.identities(row, [])
    same(list(value["stage_comparisons"]), PAIRINGS, "seven stage pairings differ")
    for name, row in value["stage_comparisons"].items():
        norm.trace(row, cfg, batch, length, value["whole_model_comparisons"][name])
    for key, anchors in (("retained_state", ANCHORS), ("stateful_full_retained_state", ANCHORS[1:])):
        same(list(value[key]), anchors, "direct retained-memory coverage differs")
        for row in value[key].values():
            norm.state_bundle(row, cfg, batch, length)
    prefix_probe(value["actual_prefix"], cfg, batch, length, prefix, targets, cell_id)
    same(value["actual_prefix"]["execution_status"], value["execution_status"], "cell prefix completion differs")
    same(value["final_positions"], {"tokenwise": length, "stateful_full": length}, "cell final positions differ")
    same(list(value["logits"]), ["full", "tokenwise", "chunked128"], "cell logits identity coverage differs")
    for row in value["logits"].values():
        isolation.identities(row, shape)
    for row in value["next_token_effects"].values():
        norm.effect(row, batch, length, 0, targets)
    same(list(value["actual_paths"]), ["full", "tokenwise"], "actual path coverage differs")
    norm.route_paths(value["actual_paths"]["full"], cfg, batch, length)
    norm.route_paths(value["actual_paths"]["tokenwise"], cfg, batch, length, True)
    complete = value["execution_status"] == "completed"
    whole = all(row["passed"] for row in [*value["whole_model_comparisons"].values(), *value["loss_comparisons"].values(), *value["retained_state"].values(), *value["stateful_full_retained_state"].values()]) and value["actual_prefix"]["passed"]
    base.flag(value["whole_model_passed"], complete and whole, "cell whole-model aggregate")
    base.flag(value["all_recorded_comparisons_passed"], complete and all(row["passed"] for row in cell_leaves(value)), "cell recorded comparison aggregate")


def reproduction_expectations(current, previous, original, old_original):
    checks = {"score." + name: current["whole_model_comparisons"][name] == previous["whole_model_comparisons"][name] for name in norm.ENDPOINTS}
    checks.update({"stage." + name: current["stage_comparisons"][name] == previous["stage_comparisons"][name] for name in norm.PAIRINGS})
    for bank in ("retained_state", "stateful_full_retained_state"):
        checks[bank] = all(current[bank][name] == value for name, value in previous[bank].items())
    for key in ("logits", "next_token_effects", "final_positions"):
        checks[key] = current[key] == previous[key]
    for key in ("logits", "loss", "stateful_logits", "state_fields", "stateful_vs_stateless", "repeat", "actual_paths"):
        checks["original." + key] = original[key] == old_original[key]
    old, new = previous["actual_prefix"], current["actual_prefix"]
    for key in ("prefix_fields", "prefix_logits", "suffix_logits", "route_comparisons", "final_positions", "suffix_state_fields", "chunked_schedule"):
        checks["prefix." + key] = new[key] == old[key]
    checks["prefix.prefix_comparisons"] = all(new["prefix_comparisons"][name] == value for name, value in old["prefix_comparisons"].items())
    for bank in ("suffix_comparisons", "retained_state"):
        checks["prefix." + bank] = all(new[bank][route][name] == value for route, values in old[bank].items() for name, value in values.items())
    checks["prefix.next_token_effects_vs_original"] = new["next_token_effects_vs_original"] == old["next_token_effects_vs_original"]
    return checks


def prior_reproduction(value, current, previous, original, old_original):
    norm.closed(value, ["executed", "comparisons", "exact_equal", "scope"], "P0 prior reproduction")
    base.require(value["executed"] is True, "prior N1H0 reproduction unexecuted")
    expected = reproduction_expectations(current, previous, original, old_original)
    same(list(value["comparisons"]), list(expected), "prior P0 repeated field coverage differs")
    for key, passed in expected.items():
        base.flag(value["comparisons"][key], passed, "exact prior P0 " + key)
    base.flag(value["exact_equal"], all(expected.values()), "prior P0 control aggregate")
    same(value["scope"], "all prior N1H0 score/stage/state and genuine-prefix comparison fields repeated exactly; prior failures remain immutable", "prior control scope differs")


def operators(value, cfg, length, prefix, cells, failed_cell, has_original, has_decay):
    expected = {}
    layers = [layer for _name, layer in projection_inventory(cfg)]
    def add(stage, counts):
        for layer in layers:
            for span, calls in counts.items():
                expected[stage, layer, span] = calls
    for stage in ("original.stateful_full", "decay_baseline.stateful_full"):
        add(stage, norm.partition(length))
    stages = ["stateful_full", "full_chunked128", "prefix_prefill", "suffix_one_shot", "suffix_chunked128", "suffix_tokenwise"]
    for cell in CELLS:
        for stage in stages:
            counts = Counter({1: length - prefix}) if stage == "suffix_tokenwise" else norm.partition(prefix if stage == "prefix_prefill" else length - prefix if stage.startswith("suffix_") else length)
            add(cell + "." + stage, counts)
    allowed = set(CELLS[:max(len(cells), 0 if failed_cell is None else CELLS.index(failed_cell) + 1)])
    observed = {}
    for row in value:
        norm.closed(row, ["stage", "layer", "path", "length", "autocast", "input_dtypes", "calls"], "scan observation")
        key = row["stage"], row["layer"], row["length"]
        base.require(key not in observed and key in expected, "duplicate/unknown operator observation")
        base.require(row["stage"] in {"original.stateful_full", "decay_baseline.stateful_full"} or row["stage"].split(".")[0] in allowed, "future-cell operator observation")
        same(row["path"], "torch.chunked_ssd", "actual cached/chunked operator differs")
        base.require(row["autocast"] is False, "autocast observed in FP32 study")
        same(row["input_dtypes"], ["torch.float32"] * 7, "scan operand precision differs")
        base.integer(row["calls"], "operator calls", 1)
        base.require(row["calls"] <= expected[key], "extra operator calls")
        observed[key] = row["calls"]
    required = {key: calls for key, calls in expected.items() if key[0].split(".")[0] in {row["cell_id"] for row in cells}
                or has_original and key[0] == "original.stateful_full" or has_decay and key[0] == "decay_baseline.stateful_full"}
    base.require(all(observed.get(key) == calls for key, calls in required.items()), "executed model route lacks actual operator coverage")


def case_leaves(value):
    leaves = [row for cell in value["cells"] for row in cell_leaves(cell)]
    leaves += [leaf for probe in value["frozen_projections"] for leaf in probe["checks"].values()]
    if "original_anchor" in value:
        leaves += [value["original_anchor"]["stateful_vs_stateless"], value["original_anchor"]["repeat"]["comparison"]]
    return leaves


def rng_integrity(value):
    norm.closed(value, ["before", "after", "restoration_exact"], "RNG restoration")
    for key in ("before", "after"):
        norm.closed(value[key], ["python", "numpy", "torch_cpu", "torch_cuda"], "RNG identity")
        for name, digest in value[key].items():
            base.digest(digest, "RNG " + name)
    base.flag(value["restoration_exact"], value["before"] == value["after"], "RNG restoration")


def validate_case(value, identity, checkpoint, previous, declared_policy):
    required = {"ratio", *identity, "weights_sha256", "execution_status", "cells", "failed_cell", "frozen_projections", "backward_executed", "BF16_executed", "production_defaults_changed", "optimizer_executed", "runtime_policy", "rng_integrity", "tied_weight_integrity", "post_model_weights_unchanged", "operator_observations", "attribution_control_stable", "all_recorded_comparisons_passed", "processing_seconds", "cuda_peak_allocated_bytes"}
    optional = {"reason", "original_anchor", "decay_baseline_anchor", "projection_baseline_anchor", "prior_reproduction"}
    base.require(isinstance(value, dict) and required <= set(value) <= required | optional, "projection case schema differs")
    for key, expected in identity.items():
        same(value[key], expected, "shared immutable input differs: " + key)
    same(value["ratio"], checkpoint["ratio"], "case checkpoint ratio differs")
    same(value["weights_sha256"], previous["weights_sha256"], "loaded weight bytes differ from prior case")
    base.digest(value["weights_sha256"], "loaded model weights")
    cfg = checkpoint["model_config"]
    batch, length = identity["batch_size"], identity["length"]
    complete = value["execution_status"] == "completed"
    base.require(value["execution_status"] in {"completed", "incomplete"}, "unknown case execution status")
    for key in ("backward_executed", "BF16_executed", "production_defaults_changed", "optimizer_executed"):
        base.require(value[key] is False, "unapproved executed factor: " + key)
    norm.runtime_policy(value["runtime_policy"], declared_policy, nested=True)
    norm.tied_integrity(value["tied_weight_integrity"])
    rng_integrity(value["rng_integrity"])
    base.require(type(value["post_model_weights_unchanged"]) is bool, "invalid weight preservation flag")
    safeguards = all([value["post_model_weights_unchanged"], value["tied_weight_integrity"]["passed"], value["runtime_policy"]["restoration_exact"], value["rng_integrity"]["restoration_exact"]])
    if complete:
        base.require(value["failed_cell"] is None and "reason" not in value and safeguards, "completed case failed restoration")
    else:
        base.require(isinstance(value.get("reason"), str) and bool(value["reason"]), "incomplete case missing reason")
        base.require(value["failed_cell"] is None or value["failed_cell"] in CELLS, "unknown failed projection cell")
    base.number(value["processing_seconds"], "case processing time", 0)
    base.integer(value["cuda_peak_allocated_bytes"], "case CUDA allocation high-water mark", 1)
    cells = value["cells"]
    base.require(isinstance(cells, list) and len(cells) <= len(CELLS), "extra projection cell")
    same([row["cell_id"] for row in cells], CELLS if complete else CELLS[:len(cells)], "projection cell order/coverage differs")
    base.require(all(row["execution_status"] == "completed" for row in cells[:-1]), "continued after incomplete projection cell")
    base.require(not complete or all(row["execution_status"] == "completed" for row in cells), "completed case contains incomplete projection cell")
    if "original_anchor" in value:
        norm.original_anchor(value["original_anchor"], cfg, batch, length)
    base.require(not cells or {"original_anchor", "decay_baseline_anchor", "projection_baseline_anchor"} <= set(value), "executed cells missing direct anchors")
    for key in ("decay_baseline_anchor", "projection_baseline_anchor"):
        if key not in value:
            continue
        anchor = value[key]
        norm.closed(anchor, ["logits", "state_fields", *(["cell"] if key == "projection_baseline_anchor" else [])], "direct baseline anchor")
        isolation.identities(anchor["logits"], [batch, length, cfg["vocab_size"]])
        isolation.state_identities(anchor["state_fields"], cfg, batch, length)
        if key == "projection_baseline_anchor":
            same(anchor["cell"], "P0", "projection baseline belongs to wrong cell")
    for row, cell in zip(cells, CELLS):
        validate_cell(row, cell, cfg, identity)
    if cells:
        same(cells[0]["logits"]["full"], value["projection_baseline_anchor"]["logits"], "P0 baseline score identity differs")
        same(cells[0]["whole_model_comparisons"]["full_vs_projection_baseline"]["nonzero_error_count"], 0, "P0 self-score comparison differs")
        same(cells[0]["loss_comparisons"]["full_vs_projection_baseline"]["nonzero_error_count"], 0, "P0 self-loss comparison differs")
        base.require(cells[0]["stateful_full_retained_state"]["projection_baseline"]["passed"], "P0 self-memory differs")
    p0_complete = bool(cells) and cells[0]["execution_status"] == "completed"
    if p0_complete:
        base.require("prior_reproduction" in value, "completed P0 missing independent prior control")
        old = next(cell for cell in previous["cells"] if cell["cell_id"] == "N1H0")
        prior_reproduction(value["prior_reproduction"], cells[0], old, value["original_anchor"], previous["original_anchor"])
    else:
        base.require("prior_reproduction" not in value, "unexecuted prior control supplied flags")
    inventory = projection_inventory(cfg)
    probes = value["frozen_projections"]
    base.require(isinstance(probes, list) and len(probes) <= len(inventory), "extra frozen full projection")
    same([(row["module"], row["layer"]) for row in probes], inventory[:len(probes)], "frozen full projection inventory/order differs")
    base.require(not probes or p0_complete, "frozen full inputs lack completed P0")
    for row in probes:
        same(row["origin"], "identical_P0_stateless_full_operands", "frozen full origin differs")
        projection_probe({key: item for key, item in row.items() if key not in {"module", "layer", "origin"}}, cfg, batch, length, schedule=[length])
    base.require(not complete or len(probes) == len(inventory), "completed case missing frozen full projections")
    operators(value["operator_observations"], cfg, length, identity["prefix_length"], cells, value["failed_cell"], "original_anchor" in value, "decay_baseline_anchor" in value)
    stable = value.get("original_anchor", {}).get("repeat", {}).get("exact_equal", False) and value.get("prior_reproduction", {}).get("exact_equal", False)
    base.flag(value["attribution_control_stable"], stable, "exact prior attribution control")
    base.flag(value["all_recorded_comparisons_passed"], complete and stable and safeguards and all(row["passed"] for row in case_leaves(value)), "case explicit comparison aggregate")


REPORT_SCHEMA = {**norm.REPORT_SCHEMA, "frozen_norm_output_checks": [], "frozen_norm_stat_checks": [], "frozen_head_checks": [],
    "frozen_projection_checks": PROJECTION_CHECKS, "stage_pairings": PAIRINGS, "score_endpoints": ENDPOINTS, "anchors": ANCHORS, "loss_endpoints": LOSS_ENDPOINTS,
    "loss_policy": "actual model target cross-entropy on each GPU/CPU route; no cross-device recomputation",
    "frozen_full_sites": "all Mamba in_proj; identical P0 full inputs",
    "frozen_prefix_sites": "first and last Mamba in_proj; each cell's actual one_shot/chunked128/tokenwise suffix inputs",
    "prefix_layout_policy": "actual call layouts retained/reconstructed for observed replay; concatenated full/token contrasts include explicit shape/layout change",
    "prior_transport": "registered gzip or original JSON, exact original bytes verified before parsing"}
FIXED_DECLARATION = {"schema": 1, "date": "2026-10-06", "status_at_declaration": "planned_before_execution", "device": "cuda", "seed": 2027,
    "ratios": RATIOS, "cases": CASES, "cells": CELLS, "decay_treatment": "fp64_cumsum_decay_coefficients", "normalization_cell": "N1H0",
    "head_treatment": "historical FP32 lm_head", "projection_treatment": "fp64_mamba_input_projection_matmul",
    "projection_scope": "all Mamba in_proj instances only; FP64 F.linear inputs/weights/optional bias then FP32 output; original parameters unchanged",
    "projection_site_registry": {ratio: [f"blocks.{layer}.mixer.in_proj" for layer in range(16) if (layer + 1) % (16 if ratio == "1:15" else 4)] for ratio in RATIOS},
    "chunk_size": 128, "tolerances": base.TOLERANCES, "report_schema": REPORT_SCHEMA,
    "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}, "time_budget_seconds": 900, "minimum_large_case_free_bytes": 8 * 1024**3,
    "backward_executed": False, "BF16_executed": False, "production_defaults_changed": False, "optimizer_executed": False, "prior_control_cell": "N1H0"}


def validate_declaration(value, prior_plan, prior, source_root):
    norm.closed(value, {*FIXED_DECLARATION, "shared_inputs", "observed_inference_policy", "prior_declaration", "prior_summary", "prior_transport", "output", "source_sha256", "data", "checkpoints"}, "projection declaration")
    for key, expected in FIXED_DECLARATION.items():
        same(value[key], expected, "declared projection scope differs: " + key)
    for key in ("prior_declaration", "prior_summary"):
        norm.reference(value[key])
    norm.relative(value["output"])
    policy = value["observed_inference_policy"]
    norm.closed(policy, norm.POLICY_KEYS - {"cuda_matmul_allow_tf32", "cudnn_allow_tf32"}, "declared inference factors")
    norm.policy_flags({**value["precision_flags"], **policy})
    same(policy, prior_plan["observed_inference_policy"], "prior inference factors changed")
    for key in ("shared_inputs", "data", "checkpoints"):
        same(value[key], prior_plan[key], "immutable prior registry changed: " + key)
        same(value[key], prior["protocol"][key], "prior protocol registry changed: " + key)
    same([row["case_id"] for row in value["shared_inputs"]], CASE_IDS, "six-case shape registry differs")
    for identity, geometry in zip(value["shared_inputs"], CASES):
        for key, expected in geometry.items():
            same(identity[key], expected, "shared input geometry differs")
    same([row["ratio"] for row in value["checkpoints"]], RATIOS, "two checkpoint order differs")
    for row in value["checkpoints"]:
        same([name for name, _ in projection_inventory(row["model_config"])], value["projection_site_registry"][row["ratio"]], "declared projection registry differs from actual geometry")
    isolation.source_check(value["source_sha256"], source_root, SOURCES)


def validate_prior_chain(declaration, *, root=ROOT, source_root=ROOT):
    prior_plan = read_public_json(declaration["prior_declaration"], root=root)
    transport = declaration["prior_transport"]
    norm.closed(transport, ["publication", "selected_representation", "original"], "prior projection transport chain")
    norm.reference(transport["publication"])
    norm.closed(transport["original"], ["path", "bytes", "sha256"], "prior original evidence")
    original = {"path": transport["original"]["path"], "path_scope": "repository-relative", "sha256": transport["original"]["sha256"]}
    norm.reference(original)
    base.integer(transport["original"]["bytes"], "prior original byte count", 1)
    prior = read_lossless_prior(original, transport["publication"], root=root, maximum_bytes=160 * 1024**2)
    publication = read_public_json(transport["publication"], root=root)
    same(transport["original"]["bytes"], publication["uncompressed_bytes"], "prior original transport size differs")
    selected = transport["selected_representation"]
    norm.closed(selected, ["path", "path_scope", "bytes", "sha256", "compressed"], "selected prior representation")
    base.require(type(selected["compressed"]) is bool, "selected compression flag differs")
    norm.reference({key: selected[key] for key in ("path", "path_scope", "sha256")})
    same(selected["path"], publication["path"] if selected["compressed"] else publication["uncompressed_path"], "selected historical path differs")
    same(selected["bytes"], publication["bytes"] if selected["compressed"] else publication["uncompressed_bytes"], "selected historical size differs")
    same(selected["sha256"], publication["sha256"] if selected["compressed"] else publication["uncompressed_sha256"], "selected historical hash differs")
    selected_path = repository_path(root, selected["path"])
    same(selected_path.stat().st_size, selected["bytes"], "actual selected historical size differs")
    same(base.file_sha256(selected_path), selected["sha256"], "actual selected historical hash differs")
    same(prior.get("kind"), norm.KIND, "prior is not normalization/head evidence")
    base.require(prior.get("execution_status") == "completed" and prior.get("certified") is False, "prior study incomplete/promoted")
    old_plan, old_report = norm.validate_prior_chain(prior_plan, Path(root), Path(source_root))
    norm.validate_declaration(prior_plan, old_plan, old_report, source_root)
    norm.validate(prior, prior_plan, declaration["prior_declaration"]["sha256"], declaration["prior_declaration"]["path"], old_report, source_root)
    summary = read_public_json(declaration["prior_summary"], root=root)
    same(summary.get("kind"), norm.KIND + "_summary", "prior summary kind differs")
    base.require(summary.get("certified") is False and summary.get("final_audit_passed") is True, "prior summary lacks validated audit")
    same(summary["declaration"], {key: declaration["prior_declaration"][key] for key in ("path", "sha256")}, "prior summary declaration chain differs")
    same(summary["raw_report"], {key: original[key] for key in ("path", "sha256")}, "prior summary original bytes differ")
    for key in ("protocol", "protocol_sha256", "status", "execution_status", "runtime_policy", "execution_allowance", "post_execution_integrity", "post_execution_audit_files"):
        same(summary[key], prior[key], "prior summary scope differs: " + key)
    configs = {row["ratio"]: row["model_config"] for row in prior_plan["checkpoints"]}
    same(summary["cases"], [norm.compact_case(row, configs[row["ratio"]]) for row in prior["cases"]], "prior summary cases differ from validated original")
    return prior_plan, prior


def audit_registry(declaration, declaration_path, declaration_hash, prior):
    registry = norm.audit_registry({**declaration, "prior_report": declaration["prior_transport"]["selected_representation"]}, declaration_path, declaration_hash)
    transport = declaration["prior_transport"]
    # Producer order is declaration, prior declaration, selected raw, transport,
    # prior summary; decoded JSON is proven through the transport hash chain.
    registry["prior_and_declaration"] += [(transport["publication"]["path"], transport["publication"]["sha256"]), (declaration["prior_summary"]["path"], declaration["prior_summary"]["sha256"])]
    # JSON object key sorting in the new declaration does not change producer
    # file-audit execution order. The unchanged inherited file inventories were
    # independently validated from the immutable prior report above.
    old_files = prior["post_execution_audit_files"]
    for group in ("prepared_data_and_tokenizer", "sources"):
        inherited = [(row["path"], row["expected_sha256"]) for row in old_files if row["group"] == group]
        expected = inherited + ([("scripts/study_projection_precision.py", declaration["source_sha256"]["scripts/study_projection_precision.py"])] if group == "sources" else [])
        same(sorted(registry[group]), sorted(expected), "inherited actual audit inventory changed: " + group)
        registry[group] = expected
    return registry


def validate_coverage(value, expected):
    rows = value["cases"]
    base.require(isinstance(rows, list) and len(rows) <= 6, "extra recorded case")
    same([(row["ratio"], row["case_id"]) for row in rows], expected[:len(rows)], "six-case coverage/order differs")
    if value["execution_status"] == "completed":
        base.require(len(rows) == 6 and all(row["execution_status"] == "completed" for row in rows), "completed projection study lacks six completed cases")
        base.require(value["reason"] is None and value["failed_case"] is None, "complete projection workflow has failure scope")
        return
    base.require(isinstance(value["reason"], str) and bool(value["reason"]), "incomplete workflow missing reason")
    base.require(all(row["execution_status"] == "completed" for row in rows[:-1]), "continued after incomplete case")
    failed = value["failed_case"]
    norm.closed(failed, ["ratio", "case_id", "cell", "stage"], "workflow interruption")
    if len(rows) == 6 and all(row["execution_status"] == "completed" for row in rows):
        same(failed, {"ratio": None, "case_id": None, "cell": None, "stage": "final audit / allowance"}, "final audit failure scope differs")
    elif rows and rows[-1]["execution_status"] == "incomplete":
        case = rows[-1]
        same((failed["ratio"], failed["case_id"]), (case["ratio"], case["case_id"]), "failed recorded case differs")
        stage = failed["stage"]
        allowed = {"original anchors", "full/tokenwise/stateful/continuation", "frozen full input projections"}
        allowed |= {cell + ".actual_suffix_" + route for cell in CELLS for route in norm.SUFFIX_ROUTES}
        allowed |= {cell + ".frozen_suffix_" + route + "." + name for cell in CELLS for route in norm.SUFFIX_ROUTES for name in FIXED_DECLARATION["projection_site_registry"][case["ratio"]]}
        base.require(stage in allowed, "unknown recorded-case interruption stage")
        post_guard = len(case["cells"]) == 2 and case["failed_cell"] is None and all(cell["execution_status"] == "completed" for cell in case["cells"])
        if post_guard:
            base.require(not all([case["post_model_weights_unchanged"], case["tied_weight_integrity"]["passed"], case["runtime_policy"]["restoration_exact"], case["rng_integrity"]["restoration_exact"]]), "cleared failed cell without actual restoration failure")
            same(failed["cell"], "P1", "post-case restoration failure active cell differs")
        else:
            same(failed["cell"], case["failed_cell"], "failed projection cell differs")
    else:
        ratio, case_id = expected[len(rows)]
        if failed["stage"] == "load model":
            same(failed, {"ratio": ratio, "case_id": None, "cell": None, "stage": "load model"}, "unexecuted model-load scope differs")
            same(case_id, CASE_IDS[0], "model load failure is not first ratio case")
        else:
            same(failed, {"ratio": ratio, "case_id": case_id, "cell": None, "stage": "case preflight"}, "unexecuted case scope differs")


def validate(value, declaration, declaration_hash, declaration_path, prior, source_root):
    norm.closed(value, ["schema", "kind", "certified", "execution_status", "status", "reason", "failed_case", "timestamp_utc", "protocol", "protocol_sha256", "cases", "large_case_headroom", "post_execution_integrity", "post_execution_audit_files", "runtime_policy", "execution_allowance", "limits"], "audited projection report")
    same(value["schema"], 1, "projection raw schema differs")
    base.require(value["kind"] == KIND and value["certified"] is False and value["execution_status"] in {"completed", "incomplete"}, "unknown/promoted raw identity")
    protocol = value["protocol"]
    norm.closed(protocol, {*declaration, "declaration", "runtime", "runtime_sha256", "quality_equivalence_margin"}, "executed projection protocol")
    same(value["protocol_sha256"], base.canonical_sha256(protocol), "protocol canonical fingerprint differs")
    for key, expected in declaration.items():
        same(protocol[key], expected, "executed protocol differs: " + key)
    same(protocol["declaration"], {"path": declaration_path, "path_scope": "repository-relative", "sha256": declaration_hash}, "actual declaration bytes differ")
    base.require(protocol["quality_equivalence_margin"] is None, "invented quality equivalence margin")
    same(protocol["runtime"], prior["protocol"]["runtime"], "runtime differs from matched historical control")
    same(protocol["runtime_sha256"], base.canonical_sha256(protocol["runtime"]), "runtime canonical fingerprint differs")
    norm.runtime_policy(value["runtime_policy"], declaration["observed_inference_policy"])
    same(protocol["runtime"]["inference_policy"], value["runtime_policy"]["active"], "actual policy metadata differs")
    for key in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "deterministic_algorithms"):
        same(protocol["runtime"]["precision_flags"][key], value["runtime_policy"]["active"][key], "runtime precision flags differ")
    expected = [(ratio, identity["case_id"]) for ratio in RATIOS for identity in declaration["shared_inputs"]]
    validate_coverage(value, expected)
    identities = {row["case_id"]: row for row in declaration["shared_inputs"]}
    checkpoints = {row["ratio"]: row for row in declaration["checkpoints"]}
    old = {(row["ratio"], row["case_id"]): row for row in prior["cases"]}
    for row in value["cases"]:
        validate_case(row, identities[row["case_id"]], checkpoints[row["ratio"]], old[row["ratio"], row["case_id"]], declaration["observed_inference_policy"])
    complete = value["execution_status"] == "completed"
    same(value["status"], "incomplete" if not complete else "completed" if all(row["all_recorded_comparisons_passed"] for row in value["cases"]) else "completed_with_parity_failures", "workflow status hides negatives")
    audit_ok = norm.final_audit(value, audit_registry(declaration, declaration_path, declaration_hash, prior), complete)
    allowance = value["execution_allowance"]
    norm.closed(allowance, ["allowance_seconds", "elapsed_seconds", "overshoot_seconds", "scope"], "execution allowance")
    same(allowance["allowance_seconds"], 900, "cooperative allowance differs")
    same(allowance["scope"], norm.ALLOWANCE_SCOPE, "allowance measurement scope differs")
    base.number(allowance["elapsed_seconds"], "elapsed wall time", 0)
    base.close_double(allowance["overshoot_seconds"], max(0, allowance["elapsed_seconds"] - 900), "in-flight allowance overshoot")
    base.require(not complete or allowance["elapsed_seconds"] < 900 and value["runtime_policy"]["restoration_exact"], "complete workflow exceeded allowance/restoration")
    seen, required = [], [(row["ratio"], row["case_id"]) for row in value["cases"] if row["batch_size"] == 2]
    for row in value["large_case_headroom"]:
        norm.closed(row, ["ratio", "case_id", "free_bytes", "total_bytes", "minimum_free_bytes"], "large-case headroom")
        pair = row["ratio"], row["case_id"]
        base.require(pair not in seen and pair in [(ratio, CASE_IDS[-1]) for ratio in RATIOS], "unknown/duplicate memory guard")
        seen.append(pair)
        for key in ("free_bytes", "total_bytes", "minimum_free_bytes"):
            base.integer(row[key], "headroom " + key, 0 if key == "free_bytes" else 1)
        same(row["minimum_free_bytes"], 8 * 1024**3, "free-memory guard changed")
        base.require(row["free_bytes"] <= row["total_bytes"] and (pair not in required or row["free_bytes"] >= 8 * 1024**3), "executed large case lacked guard")
    failed = value["failed_case"]
    extra = [] if complete or failed is None or failed["case_id"] != CASE_IDS[-1] else [(failed["ratio"], CASE_IDS[-1])]
    base.require(seen == required or seen == required + [pair for pair in extra if pair not in required], "memory-guard coverage differs")
    base.require(isinstance(value["timestamp_utc"], str) and bool(value["timestamp_utc"]) and isinstance(value["limits"], list) and all(isinstance(item, str) for item in value["limits"]), "invalid public annotations")
    return audit_ok


# Stable field order preserves the first authoritative summary bytes. The
# borrowed historical helper iterates a set for these keys; never alter that
# measured source, and never let process hash randomization order this export.
FAILED_STAGE_SAMPLE_KEYS = ("violation_samples", "worst_tolerance_ratio", "first_nonzero", "first_violation", "worst_absolute_error")


def compact_stage(value):
    result = {key: deepcopy(value[key]) for key in ("layer", "stage", "passed", "finite", "shape", "actual_dtype", "reference_dtype", "comparison_dtype", "elements", "nonzero_error_count", "violation_count", "max_absolute_error", "max_tolerance_ratio")}
    if value["passed"]:
        result["first_nonzero_coordinate"] = deepcopy(value["first_nonzero_coordinate"])
    else:
        result.update({key: deepcopy(value[key]) for key in FAILED_STAGE_SAMPLE_KEYS})
    return result


def compact_cell(value, cfg, batch, length):
    result = norm.compact_cell(value, cfg, batch, length)
    result["stage_comparisons"] = {name: {"passed": row["passed"], "first_nonzero": deepcopy(row["first_nonzero"]), "first_violation": deepcopy(row["first_violation"]),
        "counts": isolation.count(row["stages"]), "stages": [compact_stage(stage) for stage in row["stages"]]} for name, row in value["stage_comparisons"].items()}
    result["frozen_prefix_comparison_counts"] = isolation.count(leaf for probe in value["actual_prefix"]["frozen_projections"] for leaf in probe["frozen_arithmetic"]["checks"].values())
    result["prefix_upstream_comparison_counts"] = isolation.count(leaf for probe in value["actual_prefix"]["frozen_projections"] for leaf in [probe["upstream_input_vs_own_full"], probe["observed_output_vs_own_full"]])
    return result


def compact_case(value, cfg):
    result = {key: deepcopy(item) for key, item in value.items() if key not in {"cells", "original_anchor", "processing_seconds", "cuda_peak_allocated_bytes"}}
    batch, length = value["batch_size"], value["length"]
    result["cells"] = [compact_cell(cell, cfg, batch, length) for cell in value["cells"]]
    if "original_anchor" in value:
        result["original_anchor"] = {key: deepcopy(item) for key, item in value["original_anchor"].items() if key != "actual_paths"}
        result["original_anchor"]["actual_paths"] = norm.route_paths(value["original_anchor"]["actual_paths"], cfg, batch, length)
    result["resources"] = {"processing_seconds": value["processing_seconds"], "cuda_peak_allocated_bytes": value["cuda_peak_allocated_bytes"],
        "scope": "case full/tokenwise/cached routes, stage comparisons and frozen CPU/GPU projection replays; allocation peak reset before each case; not throughput"}
    inventory = projection_inventory(cfg)
    selected_sites = 1 if len(inventory) == 1 else 2
    result["frozen_comparison_counts"] = {"full_projections": isolation.count(leaf for probe in value["frozen_projections"] for leaf in probe["checks"].values()),
        "prefix_projections": isolation.count(leaf for cell in value["cells"] for probe in cell["actual_prefix"]["frozen_projections"] for leaf in probe["frozen_arithmetic"]["checks"].values()),
        "prefix_upstream": isolation.count(leaf for cell in value["cells"] for probe in cell["actual_prefix"]["frozen_projections"] for leaf in [probe["upstream_input_vs_own_full"], probe["observed_output_vs_own_full"]])}
    result["frozen_coverage"] = {"expected_full_sites": len(inventory), "executed_full_sites": len(value["frozen_projections"]), "common_input_cell": "P0",
        "expected_prefix_sites_per_cell": 3 * selected_sites, "executed_prefix_sites": sum(len(cell["actual_prefix"]["frozen_projections"]) for cell in value["cells"]),
        "independent_full_repetitions_per_cell": False}
    result["missing_cells"] = [cell for cell in CELLS if not any(row["cell_id"] == cell and row["execution_status"] == "completed" for row in value["cells"])]
    return result


def derive_summary(report, declaration, declaration_path, declaration_hash, raw_hash, final_ok):
    audited = "protocol" in report
    configs = {row["ratio"]: row["model_config"] for row in declaration["checkpoints"]}
    cases = [compact_case(row, configs[row["ratio"]]) for row in report["cases"]]
    raw_cells = [cell for case in report["cases"] for cell in case["cells"]]
    complete_cells = [cell for cell in raw_cells if cell["execution_status"] == "completed"]
    expected = [(ratio, identity["case_id"]) for ratio in RATIOS for identity in declaration["shared_inputs"]]
    completed = {(case["ratio"], case["case_id"]) for case in cases if case["execution_status"] == "completed"}
    missing_cells = [{"ratio": ratio, "case_id": case_id, "cell_id": cell_id} for ratio, case_id in expected for cell_id in CELLS if not any(case["ratio"] == ratio and case["case_id"] == case_id and any(cell["cell_id"] == cell_id and cell["execution_status"] == "completed" for cell in case["cells"]) for case in cases)]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"], "execution_status": report["execution_status"], "certified": False,
        "reason": report["reason"], "failed_case": report.get("failed_case"), "execution_identities_available": audited, "final_audit_passed": final_ok,
        "declaration": {"path": declaration_path, "sha256": declaration_hash}, "raw_report": {"path": declaration["output"], "sha256": raw_hash},
        "protocol_sha256": report.get("protocol_sha256"), "protocol": report.get("protocol"),
        "coverage": {"expected_cases": 6, "recorded_cases": len(cases), "completed_cases": len(completed), "expected_cells": 12, "recorded_cells": len(raw_cells), "completed_cells": len(complete_cells),
            "missing_cases": [{"ratio": ratio, "case_id": case_id} for ratio, case_id in expected if (ratio, case_id) not in completed], "missing_cells": missing_cells,
            "execution_identities_available": audited, "backward_executed": False, "BF16_executed": False, "optimizer_executed": False, "production_defaults_changed": False},
        "gate_counts": {"scope": "explicit recorded comparison leaves; completed cell gates remain false for incomplete cells; no quality-equivalence or production qualification",
            "cells": {cell_id: {"whole_model_comparisons": {name: isolation.count(cell["whole_model_comparisons"][name] for cell in raw_cells if cell["cell_id"] == cell_id) for name in ENDPOINTS},
                "loss_comparisons": {name: isolation.count(cell["loss_comparisons"][name] for cell in raw_cells if cell["cell_id"] == cell_id) for name in LOSS_ENDPOINTS},
                "whole_model": {"passed": sum(cell["whole_model_passed"] for cell in raw_cells if cell["cell_id"] == cell_id), "total": sum(cell["cell_id"] == cell_id for cell in raw_cells)},
                "all_recorded": {"passed": sum(cell["all_recorded_comparisons_passed"] for cell in raw_cells if cell["cell_id"] == cell_id), "total": sum(cell["cell_id"] == cell_id for cell in raw_cells)}} for cell_id in CELLS},
            "original_repeat": {"exact_passed": sum(case["original_anchor"]["repeat"]["exact_equal"] for case in cases if "original_anchor" in case), "total": sum("original_anchor" in case for case in cases)},
            "prior_reproduction": {"exact_passed": sum(case["prior_reproduction"]["exact_equal"] for case in cases if "prior_reproduction" in case), "total": sum("prior_reproduction" in case for case in cases)},
            "frozen_full_projections": isolation.count(leaf for case in report["cases"] for probe in case["frozen_projections"] for leaf in probe["checks"].values()),
            "frozen_prefix_projections": isolation.count(leaf for case in report["cases"] for cell in case["cells"] for probe in cell["actual_prefix"]["frozen_projections"] for leaf in probe["frozen_arithmetic"]["checks"].values()),
            "prefix_upstream": isolation.count(leaf for case in report["cases"] for cell in case["cells"] for probe in cell["actual_prefix"]["frozen_projections"] for leaf in [probe["upstream_input_vs_own_full"], probe["observed_output_vs_own_full"]])},
        "cases": cases, "runtime_policy": report.get("runtime_policy"), "execution_allowance": report.get("execution_allowance"), "large_case_headroom": report.get("large_case_headroom", []),
        "post_execution_integrity": report.get("post_execution_integrity"), "post_execution_audit_files": report.get("post_execution_audit_files", []),
        "limits": [*report.get("limits", []), "Independent standard-library exporter validates public evidence and text source only; no checkpoint/corpus binary or producer module loaded.",
            "All aggregates use closed schemas and explicit executed leaves. Missing frozen sites or incomplete cells cannot qualify complete gates.",
            "All failed bounded coordinate samples and trace-first markers are retained; frozen local arithmetic passes remain separate from natural upstream differences and whole-model acceptance.",
            "Full frozen input probes reuse P0 operands once per case; genuine prefix probes use each cell's actual cloned state and captured call layouts.",
            "Recorded scalar loss comparisons are actual route cross-entropy; the prior report supplies no invented historical scalar loss repetition."]}


def build_summary(declaration_path, raw_path=None, *, root=ROOT, source_root=ROOT, raw_publication=None):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    base.require(declaration_path.is_relative_to(root), "declaration outside repository")
    declaration, declaration_hash = base.read_json(declaration_path)
    declared_raw = repository_path(root, declaration["output"])
    raw_path = declared_raw if raw_path is None else Path(raw_path).resolve()
    base.require(raw_path == declared_raw, "raw report path differs from declaration")
    report, raw_hash = read_raw_report(declaration["output"], root=root, publication_path=raw_publication)
    prior_plan, prior = validate_prior_chain(declaration, root=root, source_root=source_root)
    validate_declaration(declaration, prior_plan, prior, source_root)
    relative = declaration_path.relative_to(root).as_posix()
    if "protocol" in report:
        final_ok = validate(report, declaration, declaration_hash, relative, prior, source_root)
    else:
        norm.closed(report, ["schema", "kind", "certified", "status", "execution_status", "reason", "cases"], "pre-audit projection failure")
        same(report["schema"], 1, "pre-audit schema differs")
        base.require(report["kind"] == KIND and report["certified"] is False and report["status"] == report["execution_status"] == "incomplete" and report["cases"] == [] and isinstance(report["reason"], str) and bool(report["reason"]), "pre-audit report invents execution")
        final_ok = False
    return derive_summary(report, declaration, relative, declaration_hash, raw_hash, final_ok)


WEBSITE_PROJECTION = {**norm.WEBSITE_PROJECTION, "source_kind": KIND + "_summary",
    "frozen_input_scope": "all identical P0 full projection inputs once per case, plus first/last projections on each cell's genuine suffix routes; captured layouts/positions and every failed sample retained"}


def website_projection_probe(value):
    return {**{key: deepcopy(item) for key, item in value.items() if key != "checks"}, "checks": {name: norm.website_comparison(row) for name, row in value["checks"].items()}}


def project_website_view(validated_summary, source_summary):
    base.require(validated_summary.get("kind") == KIND + "_summary" and validated_summary.get("certified") is False, "unknown/promoted website source")
    norm.closed(source_summary, ["path", "sha256"], "source summary binding")
    norm.relative(source_summary["path"])
    base.digest(source_summary["sha256"], "full summary")
    result = {key: deepcopy(item) for key, item in validated_summary.items() if key != "cases"}
    result.update(kind=KIND + "_website_view", source_summary=deepcopy(source_summary), projection=deepcopy(WEBSITE_PROJECTION), cases=[])
    for case in validated_summary["cases"]:
        current = {key: deepcopy(item) for key, item in case.items() if key not in {"cells", "frozen_projections"}}
        current["frozen_projections"] = [website_projection_probe(probe) for probe in case["frozen_projections"]]
        current["cells"] = []
        for cell in case["cells"]:
            projected = {key: deepcopy(item) for key, item in cell.items() if key not in {"stage_comparisons", "retained_state", "stateful_full_retained_state", "actual_prefix"}}
            projected["stage_comparisons"] = {}
            for name, pair in cell["stage_comparisons"].items():
                markers = {(marker["layer"], marker["stage"]) for key in ("first_nonzero", "first_violation") if (marker := pair[key]) is not None}
                rows = [row for row in pair["stages"] if row["stage"] == "block_output" or row["layer"] is None and row["stage"] in {"final_norm", "lm_head.output"} or not row["passed"] or (row["layer"], row["stage"]) in markers]
                projected["stage_comparisons"][name] = {**{key: deepcopy(item) for key, item in pair.items() if key != "stages"}, "stages": deepcopy(rows),
                    "projection_coverage": {"full_stage_count": len(pair["stages"]), "retained_stage_count": len(rows), "omitted_passing_stages": len(pair["stages"]) - len(rows), "retained_all_failures": True}}
            for key in ("retained_state", "stateful_full_retained_state"):
                projected[key] = {name: norm.website_bundle(row) for name, row in cell[key].items()}
            prefix = cell["actual_prefix"]
            projected["actual_prefix"] = {key: deepcopy(item) for key, item in prefix.items() if key not in {"retained_state", "frozen_projections"}}
            projected["actual_prefix"]["retained_state"] = {route: {anchor: norm.website_bundle(row) for anchor, row in bundles.items()} for route, bundles in prefix["retained_state"].items()}
            projected["actual_prefix"]["frozen_projections"] = [{**{key: deepcopy(item) for key, item in probe.items() if key not in {"frozen_arithmetic", "upstream_input_vs_own_full", "observed_output_vs_own_full"}},
                "upstream_input_vs_own_full": norm.website_comparison(probe["upstream_input_vs_own_full"]), "observed_output_vs_own_full": norm.website_comparison(probe["observed_output_vs_own_full"]),
                "frozen_arithmetic": website_projection_probe(probe["frozen_arithmetic"])} for probe in prefix["frozen_projections"]]
            current["cells"].append(projected)
        result["cases"].append(current)
    return result


def build_website_view(summary_path, *, root=ROOT, source_root=ROOT, raw_publication=None):
    root = Path(root).resolve()
    path = Path(summary_path).resolve()
    base.require(path.is_relative_to(root) and path.stat().st_size <= MAX_PUBLIC_JSON_BYTES, "invalid full-summary path/byte bound")
    full, digest = base.read_json(path)
    base.require(full.get("kind") == KIND + "_summary", "website source is not a full summary")
    expected = build_summary(repository_path(root, full["declaration"]["path"]), root=root, source_root=source_root, raw_publication=raw_publication)
    same(full, expected, "full summary differs from independently validated evidence")
    return project_website_view(full, {"path": path.relative_to(root).as_posix(), "sha256": digest})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/projection-precision-protocol-2026-10-06.json")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--raw-publication", type=Path, help="Exact-byte gzip transport manifest; default auto-detects the dated publication file")
    parser.add_argument("--website-view-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.website_view_from is not None:
        base.require(args.raw is None, "website raw identity comes from the bound full summary")
        norm.write_website_view(args.output, build_website_view(args.website_view_from, raw_publication=args.raw_publication))
        print("Validated projection precision website view written.")
    else:
        base.write_summary(args.output, build_summary(args.declaration, args.raw, raw_publication=args.raw_publication))
        print("Validated projection precision summary written.")


if __name__ == "__main__":
    main()
