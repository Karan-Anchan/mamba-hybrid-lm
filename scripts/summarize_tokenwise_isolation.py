"""Independent standard-library validation/export of declared tokenwise isolation.

No producer module, PyTorch, CUDA, corpus binary or checkpoint binary is loaded.
Public hashes bind the producer's binary audit; every stored failed endpoint and
unexecuted route remains distinct from a measured pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import summarize_checkpoint_numerics as base  # noqa: E402
from scripts import summarize_decay_breadth as breadth  # noqa: E402

KIND = "trained_checkpoint_tokenwise_isolation"
RATIOS = ["1:15", "1:3"]
CASES = [row for row in breadth.CASES if row["length"] in (128, 129, 512)]
CASE_IDS = [breadth.identifier(row) for row in CASES]
ROUTES = ["original_full_reference", "candidate_full_reference", "candidate_tokenwise_reference"]
PAIRINGS = ["candidate_full_vs_original_stateless", "candidate_tokenwise_vs_candidate_full", "candidate_tokenwise_vs_original_stateless"]
ENDPOINTS = [*PAIRINGS, "original_stateful_full_vs_original_stateless", "candidate_stateful_full_vs_candidate_stateless"]
MEMORY = ["candidate_tokenwise_vs_candidate_stateful_full", "candidate_tokenwise_vs_original_stateful_full"]
PROBE_CHECKS = ["full_replay_vs_observed", "tokenwise_vs_full", "full_vs_fp64", "tokenwise_vs_fp64"]
SCAN_ROUTES = ["original_quadratic", "candidate_quadratic", "candidate_one_shot", "candidate_chunked128", "candidate_tokenwise"]
SOURCES = breadth.SOURCES | {"scripts/isolate_tokenwise_numerics.py"}
DETAIL_KEYS = {"finite", "passed", "shape", "actual_dtype", "reference_dtype", "comparison_dtype", "tolerance", "elements",
               "nonzero_error_count", "violation_count", "max_absolute_error", "max_tolerance_ratio", "first_nonzero",
               "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples"}


def same(actual, expected, label):
    # Canonical comparison distinguishes false/0 and integer/boolean contracts.
    breadth.same(actual, expected, label)


def source_check(registry, source_root, expected):
    same(sorted(registry), sorted(expected), "source coverage differs")
    for name, digest in registry.items():
        base.digest(digest, "source " + name)
        path = Path(source_root) / base.relative_path(name)
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        base.require(digest in {hashlib.sha256(normalized).hexdigest(), hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest()},
                     "stale measured source: " + name)


def detail(value, shape=None, *, oracle=False):
    """Validate finite bounded coordinate arithmetic without changing model tolerances."""
    passed = base.metric(value, "detailed comparison")
    base.require(set(value) - DETAIL_KEYS <= {"layer", "stage", "field"} and DETAIL_KEYS <= set(value), "detailed comparison schema differs")
    base.require(value["finite"] is True, "nonfinite detailed comparison")
    same(value["tolerance"], base.TOLERANCES["float32"], "changed FP32 tolerance")
    if shape is not None:
        same(value["shape"], shape, "comparison tensor shape differs")
    dimensions = value["shape"]
    base.require(isinstance(dimensions, list) and dimensions, "empty comparison shape")
    for size in dimensions:
        base.integer(size, "tensor dimension", 1)
    same(value["elements"], math.prod(dimensions), "comparison element coverage differs")
    same(value["actual_dtype"], "torch.float32", "actual precision differs")
    same(value["reference_dtype"], "torch.float64" if oracle else "torch.float32", "reference precision differs")
    same(value["comparison_dtype"], "torch.float64" if oracle else "torch.float32", "comparison arithmetic differs")
    for key in ("nonzero_error_count", "violation_count"):
        base.integer(value[key], key)
    base.require(0 <= value["violation_count"] <= value["nonzero_error_count"] <= value["elements"], "impossible error coverage")
    base.flag(value["passed"], value["violation_count"] == 0, "violation count")
    base.require((value["max_absolute_error"] == 0) == (value["nonzero_error_count"] == 0), "nonzero count differs")
    samples = value["violation_samples"]
    base.require(isinstance(samples, list) and len(samples) == min(8, value["violation_count"]), "bounded violation coverage differs")
    for key, count in (("first_nonzero", value["nonzero_error_count"]), ("first_violation", value["violation_count"])):
        base.require((value[key] is None) == (count == 0), "missing/extra first coordinate")

    def coordinate(sample):
        base.require(isinstance(sample, dict), "coordinate sample missing")
        same(sorted(sample), sorted(["coordinate", "actual", "reference", "absolute_error", "tolerance_ratio"]), "coordinate sample schema differs")
        indices = sample["coordinate"]
        base.require(isinstance(indices, list) and len(indices) == len(dimensions), "coordinate rank differs")
        flat = 0
        for index, size in zip(indices, dimensions):
            base.integer(index, "coordinate")
            base.require(index < size, "coordinate outside tensor")
            flat = flat * size + index
        for key in ("actual", "reference", "absolute_error", "tolerance_ratio"):
            base.number(sample[key], key, None if key in {"actual", "reference"} else 0)
        error = abs(sample["actual"] - sample["reference"])
        base.require(math.isclose(error, sample["absolute_error"], rel_tol=2e-6, abs_tol=1e-10), "coordinate absolute error differs")
        threshold = 3e-5 + 3e-4 * abs(sample["reference"])
        base.require(math.isclose(sample["absolute_error"] / threshold, sample["tolerance_ratio"], rel_tol=2e-6, abs_tol=1e-9), "coordinate tolerance ratio differs")
        base.require(sample["absolute_error"] <= value["max_absolute_error"] and sample["tolerance_ratio"] <= value["max_tolerance_ratio"], "coordinate exceeds recorded maximum")
        return flat

    indices = [coordinate(item) for item in samples]
    base.require(indices == sorted(set(indices)), "violation samples duplicate/out of order")
    base.require(all(item["tolerance_ratio"] > 1 for item in samples), "nonviolating violation sample")
    for key in ("first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio"):
        if value[key] is not None:
            coordinate(value[key])
    if value["first_nonzero"] is not None:
        base.require(value["first_nonzero"]["absolute_error"] > 0, "first drift is zero")
    if samples:
        same(value["first_violation"], samples[0], "first violation differs from bounded samples")
        base.require(coordinate(value["first_nonzero"]) <= indices[0], "first drift occurs after first violation")
    same(value["worst_absolute_error"]["absolute_error"], value["max_absolute_error"], "worst absolute coordinate differs")
    same(value["worst_tolerance_ratio"]["tolerance_ratio"], value["max_tolerance_ratio"], "worst ratio coordinate differs")
    return passed


def identities(value, shape, dtype="torch.float32"):
    same(sorted(value), ["dtype", "sha256", "shape"], "tensor identity schema differs")
    same(value["shape"], shape, "tensor identity shape differs")
    same(value["dtype"], dtype, "tensor identity dtype differs")
    base.digest(value["sha256"], "tensor identity")


def operand_layout(value, shape):
    same(sorted(value), sorted(["shape", "dtype", "stride", "storage_offset"]), "operand layout schema differs")
    same(value["shape"], shape, "operand layout shape differs")
    same(value["dtype"], "torch.float32", "operand layout precision differs")
    base.require(isinstance(value["stride"], list) and len(value["stride"]) == len(shape), "operand stride rank differs")
    for stride in value["stride"]:
        base.integer(stride, "operand stride")
    base.integer(value["storage_offset"], "operand storage offset")


def stage_inventory(cfg, batch, length):
    d, inner, state = cfg["d_model"], cfg["d_model"] * cfg["expand"], cfg["d_state"]
    heads, conv = inner // cfg["mamba_headdim"], inner + 2 * state
    hidden = ((int(cfg["mlp_ratio"] * d) + cfg["mlp_multiple_of"] - 1) // cfg["mlp_multiple_of"]) * cfg["mlp_multiple_of"]
    rows = [(None, "embedding", [batch, length, d])]
    for layer, kind in enumerate(cfg["layer_types"]):
        def add(name, width=d, shape=None):
            rows.append((layer, name, [batch, length, width] if shape is None else shape))
        add("block_input")
        add("norm1_output")
        if kind == "mamba":
            add("in_projection.input")
            add("in_projection.output", 2 * inner + 2 * state + heads)
            add("convolution_raw", conv)
            for name, shape in (("x", [batch, length, heads, cfg["mamba_headdim"]]), ("dt", [batch, length, heads]),
                                ("A", [heads]), ("B", [batch, length, state]), ("C", [batch, length, state]), ("D", [heads])):
                add("scan_input." + name, shape=shape)
            add("scan_output", shape=[batch, length, heads, cfg["mamba_headdim"]])
            for name in ("gated_norm_input", "gated_norm_output", "out_projection.input"):
                add(name, inner)
            add("out_projection.output")
        else:
            add("attention_qkv.input")
            add("attention_qkv.output", 3 * d)
            for name in ("query", "key", "value", "output"):
                add("attention_" + name, shape=[batch, length, d // cfg["head_dim"], cfg["head_dim"]])
            add("attention_projection.input")
            add("attention_projection.output")
        add("mixer_output")
        if cfg["mlp_on_every_layer"]:
            add("mixer_residual")
            add("norm2_output")
            for name, input_width, output_width in (("gate", d, hidden), ("up", d, hidden), ("down", hidden, d)):
                add("mlp_" + name + ".input", input_width)
                add("mlp_" + name + ".output", output_width)
            add("mlp_output")
        add("block_output")
    return [*rows, (None, "final_norm", [batch, length, d]), (None, "lm_head.input", [batch, length, d]),
            (None, "lm_head.output", [batch, length, cfg["vocab_size"]])]


def first_stage(stages, key):
    row = next((row for row in stages if row[key] is not None), None)
    return None if row is None else {"layer": row["layer"], "stage": row["stage"], "sample": row[key]}


def selected_mixers(stages, cfg):
    rows = [row for row in stages if row["layer"] is not None
            and not row["stage"].startswith(("mlp_", "norm2", "block_input", "block_output", "mixer_residual"))]
    selected = {}
    def add(row, reason):
        if row is not None:
            selected.setdefault(row["layer"], []).append(reason)
    add(next((row for row in rows if row["nonzero_error_count"]), None), "first_mixer_nonzero_drift")
    failure = next((row for row in rows if not row["passed"]), None)
    add(failure, "first_mixer_fixed_tolerance_violation")
    if failure is None:
        selected.setdefault(len(cfg["layer_types"]) - 1, []).append("last_mixer_no_intermediate_mixer_violation")
    return [{"layer": layer, "kind": cfg["layer_types"][layer], "reasons": reasons} for layer, reasons in selected.items()][:3]


def state_inventory(cfg, batch, length):
    return {name: ([batch, *shape[1:]], "torch.float32") for name, (shape, _dtype) in base.state_inventory(cfg, length).items()}


def state_fields(fields, cfg, batch, length):
    inventory = state_inventory(cfg, batch, length)
    same([row["field"] for row in fields], list(inventory), "state field coverage/order differs")
    for row in fields:
        detail(row, inventory[row["field"]][0])
    return all(row["passed"] for row in fields)


def state_identities(fields, cfg, batch, length):
    inventory = state_inventory(cfg, batch, length)
    same([row["field"] for row in fields], list(inventory), "state identity coverage/order differs")
    for row in fields:
        same(sorted(row), sorted(["field", "shape", "dtype", "sha256", "nonzero_elements"]), "state identity schema differs")
        shape, dtype = inventory[row["field"]]
        same(row["shape"], shape, "state identity shape differs")
        same(row["dtype"], dtype, "state identity precision differs")
        base.digest(row["sha256"], "state field")
        base.integer(row["nonzero_elements"], "state nonzero elements")
        base.require(row["nonzero_elements"] <= math.prod(shape), "impossible state nonzero count")
        if row["field"].endswith(".ssm"):
            base.require(row["nonzero_elements"] > 0, "real prefix/end memory is zero-only")
    return {row["field"]: row for row in fields}


def linear_probe(value, batch, length, input_width, output_width, bias):
    same(sorted(value), sorted(["input", "weight", "bias", "identical_operands", *PROBE_CHECKS]), "linear probe schema differs")
    identities(value["input"], [batch, length, input_width])
    identities(value["weight"], [output_width, input_width])
    base.require(value["identical_operands"] is True, "linear operands not frozen")
    if bias:
        identities(value["bias"], [output_width])
    else:
        base.require(value["bias"] is None, "unexpected linear bias")
    for name in PROBE_CHECKS:
        detail(value[name], [batch, length, output_width], oracle=name.endswith("vs_fp64"))


def scan_probe(value, cfg, batch, length, prefix_field=None):
    same(sorted(value), sorted(["operands", "initial", "initial_nonzero_elements", "initial_condition", "origin", "synthetic",
                               "initial_unchanged", "oracle", "routes", "exploratory"]), "frozen scan probe schema differs")
    inner, state = cfg["d_model"] * cfg["expand"], cfg["d_state"]
    heads, width = inner // cfg["mamba_headdim"], cfg["mamba_headdim"]
    shapes = {"x": [batch, length, heads, width], "dt": [batch, length, heads], "A": [heads],
              "B": [batch, length, state], "C": [batch, length, state], "D": [heads]}
    same(list(value["operands"]), list(shapes), "scan operand coverage differs")
    for name, shape in shapes.items():
        identities(value["operands"][name], shape)
    memory_shape = [batch, heads, width, state]
    identities(value["initial"], memory_shape)
    base.require(value["synthetic"] is False and value["initial_unchanged"] is True and value["exploratory"] is True,
                 "changed/promoted frozen scan scope")
    same(value["oracle"], "independent FP64 elementwise sequential recurrence", "scan oracle differs")
    base.integer(value["initial_nonzero_elements"], "initial nonzero count")
    if prefix_field is None:
        same(value["initial_nonzero_elements"], 0, "zero-state replay initial differs")
        same(value["initial"]["sha256"], hashlib.sha256(bytes(math.prod(memory_shape) * 4)).hexdigest(), "zero-state memory hash differs")
        same(value["initial_condition"], "zero state on actual stateless projected operands", "zero-state scope differs")
        same(value["origin"], "actual_model_stateless_operands_zero_state", "zero-state origin differs")
    else:
        same(value["initial"]["sha256"], prefix_field["sha256"], "real prefix scan memory differs")
        same(value["initial_nonzero_elements"], prefix_field["nonzero_elements"], "real prefix nonzero memory differs")
        same(value["initial_condition"], "actual model prefix state", "prefix state scope differs")
        same(value["origin"], "actual_model_prefix_continuation", "real-prefix origin differs")
    same(list(value["routes"]), SCAN_ROUTES, "frozen scan route coverage differs")
    for name, row in value["routes"].items():
        quadratic = name.endswith("quadratic")
        executed = prefix_field is None or not quadratic
        base.flag(row["executed"], executed, "frozen route execution")
        if not executed:
            same(sorted(row), ["executed", "reason"], "unexecuted route contains measurements")
            same(row["reason"], "stateless quadratic scan cannot represent real nonzero prefix state", "unexecuted reason differs")
            continue
        observed = name == ("candidate_quadratic" if prefix_field is None else "candidate_chunked128")
        expected_keys = ["executed", "output_vs_fp64", "retained_state" if quadratic else "retained_state_vs_fp64"]
        if observed:
            expected_keys.append("replay_vs_observed")
        same(sorted(row), sorted(expected_keys), "executed frozen scan schema differs")
        detail(row["output_vs_fp64"], shapes["x"], oracle=True)
        if quadratic:
            base.require(row["retained_state"] is None, "quadratic scan falsely provides memory")
        else:
            detail(row["retained_state_vs_fp64"], memory_shape, oracle=True)
        base.require(("replay_vs_observed" in row) == observed, "observed replay coverage differs")
        if observed:
            detail(row["replay_vs_observed"], shapes["x"])


def component_probes(case, cfg):
    batch, length = case["batch_size"], case["length"]
    d, inner = cfg["d_model"], cfg["d_model"] * cfg["expand"]
    hidden = ((int(cfg["mlp_ratio"] * d) + cfg["mlp_multiple_of"] - 1) // cfg["mlp_multiple_of"]) * cfg["mlp_multiple_of"]
    selections = selected_mixers(case["stage_comparisons"][PAIRINGS[1]]["stages"], cfg)
    probes = case["selected_frozen_mixers"]
    base.require(1 <= len(probes) <= 3, "frozen mixer bound differs")
    same([{key: row[key] for key in ("layer", "kind", "reasons")} for row in probes], selections, "selected frozen mixers differ")
    for row in probes:
        attention = row["kind"] == "attention"
        same(sorted(row), sorted(["layer", "kind", "reasons", "linear", *( ["attention"] if attention else ["convolution", "scan"])]),
             "frozen mixer probe schema differs")
        linears = {"attention_qkv": (d, 3 * d, cfg["attn_bias"]), "attention_projection": (d, d, cfg["attn_bias"])} if attention else {
            "in_projection": (d, 2 * inner + 2 * cfg["d_state"] + inner // cfg["mamba_headdim"], False), "out_projection": (inner, d, False)}
        if cfg["mlp_on_every_layer"]:
            linears.update({"mlp_gate": (d, hidden, cfg["mlp_bias"]), "mlp_up": (d, hidden, cfg["mlp_bias"]), "mlp_down": (hidden, d, cfg["mlp_bias"])})
        same(list(row["linear"]), list(linears), "frozen linear coverage differs")
        for name, dimensions in linears.items():
            linear_probe(row["linear"][name], batch, length, *dimensions)
        if attention:
            base.require("scan" not in row and "convolution" not in row, "attention falsely includes Mamba replay")
            probe = row["attention"]
            same(sorted(probe), sorted(["operands", "original_layout", "replay_layout", "identical_operands", "mask", *PROBE_CHECKS]), "attention probe schema differs")
            base.require(probe["identical_operands"] is True, "attention operands not frozen")
            same(probe["mask"], "query i sees key positions 0..i, token queries use the same sliced prefix", "attention causal mask differs")
            shape = [batch, d // cfg["head_dim"], length, cfg["head_dim"]]
            same(list(probe["operands"]), ["post_rope_q", "post_rope_k", "v"], "attention operand coverage differs")
            for value in probe["operands"].values():
                identities(value, shape)
            original_layout = next(event["operand_layout"] for event in case["actual_paths"]["candidate_full_reference"]["attention"] if event["layer"] == row["layer"])
            same(probe["original_layout"], original_layout, "attention replay original layout differs from observed model")
            same(probe["replay_layout"], original_layout, "frozen attention changed original strides/offsets")
            for layout in original_layout.values():
                operand_layout(layout, shape)
            for name in PROBE_CHECKS:
                detail(probe[name], [batch, length, d // cfg["head_dim"], cfg["head_dim"]] if name == "full_replay_vs_observed" else shape,
                       oracle=name.endswith("vs_fp64"))
        else:
            base.require("attention" not in row, "Mamba falsely includes attention replay")
            probe = row["convolution"]
            same(sorted(probe), sorted(["input", "weight", "identical_operands", "initial_condition", *PROBE_CHECKS]), "convolution probe schema differs")
            channels = inner + 2 * cfg["d_state"]
            identities(probe["input"], [batch, length, channels])
            identities(probe["weight"], [channels, 1, cfg["d_conv"]])
            base.require(probe["identical_operands"] is True, "convolution operands not frozen")
            same(probe["initial_condition"], "zero causal convolution tail, same frozen full-route projected input", "convolution initial scope differs")
            for name in PROBE_CHECKS:
                detail(probe[name], [batch, length, channels], oracle=name.endswith("vs_fp64"))
            scan_probe(row["scan"], cfg, batch, length)
    linear_probe(case["frozen_lm_head"], batch, length, d, cfg["vocab_size"], False)


def actual_paths(value, cfg, length, batch=1):
    same(list(value), ROUTES, "observed model route coverage differs")
    mamba_layers = [i for i, kind in enumerate(cfg["layer_types"]) if kind == "mamba"]
    attention_layers = [i for i, kind in enumerate(cfg["layer_types"]) if kind == "attention"]
    compact = {}
    for route, observations in value.items():
        tokenwise = route == ROUTES[-1]
        positions = range(length) if tokenwise else (0,)
        scan = [{"layer": layer, "position": position, "length": 1 if tokenwise else length,
                 "stateful": tokenwise, "path": "torch.chunked_ssd" if tokenwise else "torch.quadratic_ssd",
                 "backend": "reference", "chunk_size": 128} for position in positions for layer in mamba_layers]
        attention = [{"layer": layer, "position": position, "query_length": 1 if tokenwise else length,
                      "key_length": position + 1 if tokenwise else length, "is_causal": position == 0,
                      "mask_present": False, "mask_shape": None} for position in positions for layer in attention_layers]
        same(list(observations), ["scan", "attention"], "observed route schema differs")
        same(observations["scan"], scan, "actual scan positional coverage differs")
        same([{key: item for key, item in event.items() if key != "operand_layout"} for event in observations["attention"]], attention,
             "actual attention positional coverage differs")
        for event in observations["attention"]:
            same(list(event["operand_layout"]), ["q", "k", "v"], "attention layout operand coverage differs")
            for name, layout in event["operand_layout"].items():
                operand_layout(layout, [batch, cfg["d_model"] // cfg["head_dim"],
                                       event["query_length"] if name == "q" else event["key_length"], cfg["head_dim"]])
        compact[route] = {"backend": "reference", "scan_paths": sorted({row["path"] for row in scan}),
                          "scan_calls": len(scan), "attention_calls": len(attention),
                          "query_policy": "one token with all causal prefix keys" if tokenwise else "full causal sequence"}
    return compact


def validate_case(case, identity, checkpoint, prior_case):
    expected_keys = {"ratio", *identity, "weights_sha256", "whole_model_comparisons", "stage_comparisons", "retained_state",
                     "original_repeat", "selected_frozen_mixers", "frozen_lm_head", "actual_prefix", "actual_paths", "final_positions",
                     "backward_executed", "BF16_executed", "attribution_control_stable", "whole_model_passed",
                     "exploratory_components_are_not_whole_model_qualification", "all_recorded_comparisons_passed",
                     "processing_seconds", "cuda_peak_allocated_bytes"}
    same(sorted(case), sorted(expected_keys), "case schema differs")
    for key, expected in identity.items():
        same(case[key], expected, "case input identity differs: " + key)
    cfg, batch, length = checkpoint["model_config"], case["batch_size"], case["length"]
    same(case["weights_sha256"], prior_case["weights_sha256"], "checkpoint weight identity differs")
    base.require(case["backward_executed"] is False and case["BF16_executed"] is False
                 and case["exploratory_components_are_not_whole_model_qualification"] is True, "unexpected execution/promotion")
    same(list(case["whole_model_comparisons"]), ENDPOINTS, "whole-model endpoint coverage differs")
    for value in case["whole_model_comparisons"].values():
        detail(value, [batch, length, cfg["vocab_size"]])
    same(list(case["stage_comparisons"]), PAIRINGS, "aligned stage pairing coverage differs")
    inventory = stage_inventory(cfg, batch, length)
    for name, pairing in case["stage_comparisons"].items():
        same(sorted(pairing), sorted(["stages", "first_nonzero", "first_violation", "passed"]), "stage pairing schema differs")
        stages = pairing["stages"]
        same([(row["layer"], row["stage"]) for row in stages], [(layer, stage) for layer, stage, _shape in inventory], "stage names/order differ")
        for row, (_layer, _stage, shape) in zip(stages, inventory):
            detail(row, shape)
        same(pairing["first_nonzero"], first_stage(stages, "first_nonzero"), "first nonzero stage differs")
        same(pairing["first_violation"], first_stage(stages, "first_violation"), "first threshold violation stage differs")
        base.flag(pairing["passed"], all(row["passed"] for row in stages), "aligned trace aggregate")
        same(case["whole_model_comparisons"][name], {key: value for key, value in stages[-1].items() if key not in {"layer", "stage"}}, "whole endpoint differs from final stage")
    same(list(case["retained_state"]), MEMORY, "memory anchor coverage differs")
    for row in case["retained_state"].values():
        same(sorted(row), sorted(["tolerance", "fields", "passed"]), "state comparison bundle schema differs")
        same(row["tolerance"], base.TOLERANCES["float32"], "changed memory tolerance")
        base.flag(row["passed"], state_fields(row["fields"], cfg, batch, length), "memory aggregate")
    repeat = case["original_repeat"]
    same(sorted(repeat), sorted(["exact_equal", "comparison"]), "original repeat schema differs")
    detail(repeat["comparison"], [batch, length, cfg["vocab_size"]])
    base.flag(repeat["exact_equal"], repeat["comparison"]["nonzero_error_count"] == 0, "original exact repeat")
    base.flag(case["attribution_control_stable"], repeat["exact_equal"], "attribution control")
    base.flag(case["whole_model_passed"], all(row["passed"] for row in case["whole_model_comparisons"].values())
              and all(row["passed"] for row in case["retained_state"].values()), "whole-model aggregate")
    same(case["final_positions"], {"candidate_tokenwise": length, "candidate_stateful_full": length}, "whole route final position differs")
    prefix = case["actual_prefix"]
    same(sorted(prefix), sorted(["synthetic", "prefix_position", "final_position", "suffix_length", "prefix_unchanged",
                                "prefix_fields", "scan_probes", "suffix_logits", "retained_state"]), "real prefix schema differs")
    base.require(prefix["synthetic"] is False and prefix["prefix_unchanged"] is True, "prefix is synthetic or mutated")
    same((prefix["prefix_position"], prefix["final_position"], prefix["suffix_length"]),
         (case["prefix_length"], length, length - case["prefix_length"]), "real continuation positions differ")
    fields = state_identities(prefix["prefix_fields"], cfg, batch, case["prefix_length"])
    state_identities(prefix["retained_state"], cfg, batch, length)
    identities(prefix["suffix_logits"], [batch, length - case["prefix_length"], cfg["vocab_size"]])
    component_probes(case, cfg)
    expected_layers = [row["layer"] for row in case["selected_frozen_mixers"] if row["kind"] == "mamba"]
    same([row["layer"] for row in prefix["scan_probes"]], expected_layers, "real-prefix replay layer coverage differs")
    for row in prefix["scan_probes"]:
        same(sorted(row), ["layer", "scan"], "prefix scan schema differs")
        scan_probe(row["scan"], cfg, batch, prefix["suffix_length"], fields[f"layer_{row['layer']}.ssm"])
    actual_paths(case["actual_paths"], cfg, length, batch)
    def all_flags(value):
        if isinstance(value, dict):
            return ([value["passed"]] if "passed" in value else []) + [flag for item in value.values() for flag in all_flags(item)]
        if isinstance(value, list):
            return [flag for item in value for flag in all_flags(item)]
        return []
    base.flag(case["all_recorded_comparisons_passed"], all(all_flags(case)), "all recorded comparison aggregate")
    base.number(case["processing_seconds"], "diagnostic processing time", 0)
    base.integer(case["cuda_peak_allocated_bytes"], "diagnostic CUDA peak", 1)


def validate_declaration(declaration, prior_declaration, prior, source_root):
    expected = {"schema": 1, "status_at_declaration": "planned_before_execution", "device": "cuda", "seed": 2027,
                "ratios": RATIOS, "cases": CASES, "routes": ROUTES, "chunk_size": 128, "treatment_id": breadth.decay.TREATMENT,
                "tolerances": base.TOLERANCES, "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
                "time_budget_seconds": 900, "minimum_large_case_free_bytes": 8 * 1024**3,
                "coordinate_limit": 8, "maximum_frozen_mixers": 3, "backward_executed": False, "BF16_executed": False}
    for key, value in expected.items():
        same(declaration[key], value, "declaration differs: " + key)
    shared = [row for row in prior["protocol"]["shared_inputs"] if row["case_id"] in CASE_IDS]
    same(declaration["shared_inputs"], shared, "shared breadth inputs differ")
    same(prior["protocol"]["shared_inputs"], prior_declaration["shared_inputs"], "prior declaration inputs differ")
    same([row["case_id"] for row in shared], CASE_IDS, "declared shape coverage differs")
    source_check(declaration["source_sha256"], source_root, SOURCES)


def validate(report, declaration, declaration_hash, declaration_relative, prior, source_root):
    base.require(type(report["schema"]) is int and report["schema"] == 1 and report["kind"] == KIND
                 and report["certified"] is False and report["execution_status"] in {"completed", "incomplete"}, "unknown report identity")
    protocol = report["protocol"]
    same(report["protocol_sha256"], base.canonical_sha256(protocol), "protocol canonical fingerprint differs")
    for key, expected in declaration.items():
        same(protocol[key], expected, "protocol declaration differs: " + key)
    same(protocol["declaration"], {"path": declaration_relative, "path_scope": "repository-relative", "sha256": declaration_hash}, "declaration identity differs")
    base.require(protocol["production_defaults_changed"] is False and protocol["optimizer_executed"] is False
                 and protocol["quality_equivalence_margin"] is None, "unapproved policy/training/quality margin")
    same(protocol["data"], prior["protocol"]["data"], "prior data/tokenizer identity differs")
    checkpoints = {row["ratio"]: row for row in prior["protocol"]["checkpoints"]}
    same(protocol["checkpoints"], [checkpoints[ratio] for ratio in RATIOS], "checkpoint registry differs")
    same(protocol["runtime"], prior["protocol"]["runtime"], "actual runtime differs from immutable breadth")
    same(protocol["runtime_sha256"], base.canonical_sha256(protocol["runtime"]), "runtime canonical fingerprint differs")
    for name in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
        base.require(protocol["runtime"]["precision_flags"][name] is False, "actual TF32 flags differ")
    source_check(protocol["source_sha256"], source_root, SOURCES)
    expected = [(ratio, identity["case_id"]) for ratio in RATIOS for identity in declaration["shared_inputs"]]
    completed = [(row["ratio"], row["case_id"]) for row in report["cases"]]
    complete = report["execution_status"] == "completed"
    same(completed, expected if complete else expected[:len(completed)], "six-case coverage/order differs")
    base.require(len(completed) <= len(expected), "extra completed cases")
    if complete:
        base.require(report["reason"] is None and report["failed_case"] is None, "complete report has failure metadata")
    else:
        base.require(isinstance(report["reason"], str) and report["reason"], "incomplete reason missing")
        failed = report["failed_case"]
        if len(completed) < len(expected):
            base.require(isinstance(failed, dict), "failed case missing")
            ratio, case_id = expected[len(completed)]
            base.require(failed["ratio"] == ratio and failed["case_id"] in (None, case_id), "failure is not next incomplete case")
            base.require(failed["stage"] in {"load model", "trace and frozen components"}, "unknown incomplete stage")
            if failed["stage"] == "load model":
                base.require(failed["case_id"] is None and case_id == CASE_IDS[0], "invalid model-load failure")
        else:
            if failed is None:
                base.require(not all(report["post_execution_integrity"].values()), "completed cases have no final integrity failure")
            else:
                base.require(isinstance(failed, dict) and failed["stage"] in {"trace and frozen components", "post-execution integrity / allowance"}, "unknown final failure")
    identities_by_case = {row["case_id"]: row for row in declaration["shared_inputs"]}
    for case in report["cases"]:
        prior_case = next(row for row in prior["cases"] if (row["ratio"], row["case_id"]) == (case["ratio"], case["case_id"]))
        validate_case(case, identities_by_case[case["case_id"]], checkpoints[case["ratio"]], prior_case)
    same(report["status"], "incomplete" if not complete else "completed" if all(row["all_recorded_comparisons_passed"] and row["attribution_control_stable"] for row in report["cases"]) else "completed_with_parity_failures", "status hides failures")
    integrity = report["post_execution_integrity"]
    same(sorted(integrity), sorted(["checkpoints", "prepared_data_and_tokenizer", "declarations_and_prior_evidence", "sources"]), "post-execution integrity coverage differs")
    base.require(all(type(value) is bool for value in integrity.values()) and (not complete or all(integrity.values())), "unstable completed evidence")
    allowance = report["execution_allowance"]
    same(allowance["allowance_seconds"], 900, "cooperative allowance differs")
    base.number(allowance["elapsed_seconds"], "elapsed time", 0)
    base.close_double(allowance["overshoot_seconds"], max(0, allowance["elapsed_seconds"] - 900), "wall overshoot")
    same(allowance["scope"], "from run_study entry, including evidence/data preflight and final integrity; excludes immutable report publication", "resource scope differs")
    base.require(not complete or allowance["elapsed_seconds"] < 900, "completed study exceeded declared allowance")
    seen = []
    expected_headroom = [(ratio, CASE_IDS[-1]) for ratio, case_id in completed if case_id == CASE_IDS[-1]]
    for row in report["large_case_headroom"]:
        pair = row["ratio"], row["case_id"]
        base.require(pair not in seen and pair in [(ratio, CASE_IDS[-1]) for ratio in RATIOS], "duplicate/unknown headroom case")
        seen.append(pair)
        for key in ("free_bytes", "total_bytes", "minimum_free_bytes"):
            base.integer(row[key], "headroom " + key, 0 if key == "free_bytes" else 1)
        same(row["minimum_free_bytes"], 8 * 1024**3, "large-case headroom guard changed")
        base.require(row["free_bytes"] <= row["total_bytes"] and (pair not in completed or row["free_bytes"] >= row["minimum_free_bytes"]), "executed case lacked declared headroom")
    failed = report["failed_case"]
    extra = [] if complete or failed is None or failed["case_id"] != CASE_IDS[-1] else [(failed["ratio"], CASE_IDS[-1])]
    base.require(seen == expected_headroom or seen == expected_headroom + extra, "headroom case coverage differs")


def compact_detail(value):
    return {key: deepcopy(value[key]) for key in ("passed", "finite", "shape", "actual_dtype", "reference_dtype", "comparison_dtype",
             "elements", "nonzero_error_count", "violation_count", "max_absolute_error", "max_tolerance_ratio", "first_nonzero",
             "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples")}


def compact_stage(row):
    """Retain all stage ratios/counts and failed samples; passing coordinates live in immutable raw JSON."""
    value = {key: deepcopy(row[key]) for key in ("layer", "stage", "passed", "shape", "max_absolute_error", "max_tolerance_ratio",
                                                "nonzero_error_count", "violation_count")}
    if not row["passed"]:
        for key in ("first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples"):
            value[key] = deepcopy(row[key])
    return value


def compact_probe(value):
    result = {key: deepcopy(item) for key, item in value.items() if key not in PROBE_CHECKS}
    result.update({key: compact_detail(value[key]) for key in PROBE_CHECKS})
    return result


def compact_scan(value):
    result = {key: deepcopy(item) for key, item in value.items() if key != "routes"}
    result["routes"] = {}
    for name, row in value["routes"].items():
        result["routes"][name] = {key: compact_detail(item) if isinstance(item, dict) and "max_tolerance_ratio" in item else deepcopy(item)
                                   for key, item in row.items()}
    executed = [row for row in value["routes"].values() if row["executed"]]
    outputs = [row["output_vs_fp64"] for row in executed]
    states = [row["retained_state_vs_fp64"] for row in executed if "retained_state_vs_fp64" in row]
    result["counts"] = {"output_vs_fp64": count(outputs), "retained_state_vs_fp64": count(states),
                        "unexecuted_routes": [name for name, row in value["routes"].items() if not row["executed"]]}
    return result


def count(values):
    values = list(values)
    return {"passed": sum(row["passed"] for row in values), "total": len(values),
            "worst_tolerance_ratio": max((row["max_tolerance_ratio"] for row in values), default=None)}


def frozen_leaves(case):
    """Count explicitly validated, actually executed routes; never infer a new comparison from extra metadata."""
    leaves = [case["frozen_lm_head"][key] for key in PROBE_CHECKS]
    def scan_rows(scan):
        for row in scan["routes"].values():
            if row["executed"]:
                leaves.append(row["output_vs_fp64"])
                if "retained_state_vs_fp64" in row:
                    leaves.append(row["retained_state_vs_fp64"])
                if "replay_vs_observed" in row:
                    leaves.append(row["replay_vs_observed"])
    for mixer in case["selected_frozen_mixers"]:
        leaves.extend(probe[key] for probe in mixer["linear"].values() for key in PROBE_CHECKS)
        for kind in ("attention", "convolution"):
            if kind in mixer:
                leaves.extend(mixer[kind][key] for key in PROBE_CHECKS)
        if "scan" in mixer:
            scan_rows(mixer["scan"])
    for probe in case["actual_prefix"]["scan_probes"]:
        scan_rows(probe["scan"])
    return leaves


def compact_case(case, cfg):
    result = {key: deepcopy(case[key]) for key in ("ratio", "case_id", "length", "batch_size", "prefix_length", "gradient_check",
              "weights_sha256", "backward_executed", "BF16_executed", "whole_model_passed", "all_recorded_comparisons_passed", "attribution_control_stable", "final_positions")}
    result["whole_model_comparisons"] = {name: compact_detail(row) for name, row in case["whole_model_comparisons"].items()}
    result["retained_state"] = {name: {"passed": row["passed"], "fields": [{"field": field["field"], **compact_detail(field)} for field in row["fields"]],
                                              "counts": count(row["fields"])} for name, row in case["retained_state"].items()}
    result["stage_comparisons"] = {name: {"passed": pairing["passed"], "first_nonzero": first_stage(pairing["stages"], "first_nonzero"),
                                         "first_violation": first_stage(pairing["stages"], "first_violation"), "counts": count(pairing["stages"]),
                                         "stages": [compact_stage(row) for row in pairing["stages"]]}
                                   for name, pairing in case["stage_comparisons"].items()}
    result["original_repeat"] = {"exact_equal": case["original_repeat"]["exact_equal"], "comparison": compact_detail(case["original_repeat"]["comparison"])}
    probes = []
    for row in case["selected_frozen_mixers"]:
        compact = {key: deepcopy(row[key]) for key in ("layer", "kind", "reasons")}
        compact["linear"] = {name: compact_probe(probe) for name, probe in row["linear"].items()}
        for kind in ("attention", "convolution"):
            if kind in row:
                compact[kind] = compact_probe(row[kind])
        if "scan" in row:
            compact["scan"] = compact_scan(row["scan"])
        probes.append(compact)
    result["selected_frozen_mixers"] = probes
    result["frozen_lm_head"] = compact_probe(case["frozen_lm_head"])
    result["actual_prefix"] = {key: deepcopy(item) for key, item in case["actual_prefix"].items() if key != "scan_probes"}
    result["actual_prefix"]["scan_probes"] = [{"layer": row["layer"], "scan": compact_scan(row["scan"])} for row in case["actual_prefix"]["scan_probes"]]
    result["actual_paths"] = actual_paths(case["actual_paths"], cfg, case["length"], case["batch_size"])
    result["resources"] = {"processing_seconds": case["processing_seconds"], "cuda_peak_allocated_bytes": case["cuda_peak_allocated_bytes"],
                            "scope": "combined model traces and frozen diagnostic/oracle replays, not training/inference throughput"}
    result["frozen_comparison_counts"] = count(frozen_leaves(case))
    result["stage_comparison_counts"] = count(row for pairing in case["stage_comparisons"].values() for row in pairing["stages"])
    return result


def build_summary(declaration_path, raw_path=None, *, root=ROOT, source_root=ROOT):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    declaration_path = Path(declaration_path).resolve()
    declaration, declaration_hash = base.read_json(declaration_path)
    declared_raw = root / base.relative_path(declaration["output"])
    raw_path = declared_raw if raw_path is None else Path(raw_path).resolve()
    same(str(raw_path.resolve()), str(declared_raw.resolve()), "raw path differs from declaration")
    report, raw_hash = base.read_json(raw_path)
    priors = {}
    for name in ("breadth_declaration", "breadth_report"):
        record = declaration[name]
        prior, digest = base.read_json(root / base.relative_path(record["path"]))
        same(digest, record["sha256"], "prior immutable bytes differ: " + name)
        priors[name] = prior
    prior = priors["breadth_report"]
    base.require(prior["kind"] == "trained_checkpoint_decay_breadth" and prior["certified"] is False
                 and prior["execution_status"] == "completed", "prior breadth is incomplete/certified/unknown")
    same(prior["protocol_sha256"], base.canonical_sha256(prior["protocol"]), "prior canonical protocol differs")
    same(prior["protocol"]["declaration"], declaration["breadth_declaration"], "prior declaration chain differs")
    source_check(prior["protocol"]["source_sha256"], source_root, breadth.SOURCES)
    expected_prior = [(ratio, breadth.identifier(case)) for ratio in base.RATIOS for case in breadth.CASES]
    same([(row["ratio"], row["case_id"]) for row in prior["cases"]], expected_prior, "prior complete coverage differs")
    validate_declaration(declaration, priors["breadth_declaration"], prior, source_root)
    relative = declaration_path.relative_to(root).as_posix()
    expected = [{"ratio": ratio, "case_id": case_id} for ratio in RATIOS for case_id in CASE_IDS]
    audited = "protocol" in report
    if audited:
        validate(report, declaration, declaration_hash, relative, prior, source_root)
    else:
        same(sorted(report), sorted(["schema", "kind", "certified", "status", "execution_status", "reason", "cases"]), "pre-audit report schema differs")
        base.require(type(report["schema"]) is int and report["schema"] == 1 and report["kind"] == KIND and report["certified"] is False
                     and report["status"] == report["execution_status"] == "incomplete" and report["cases"] == []
                     and isinstance(report["reason"], str) and report["reason"], "invalid pre-audit incomplete evidence")
    cps = {} if not audited else {row["ratio"]: row["model_config"] for row in report["protocol"]["checkpoints"]}
    cases = [compact_case(case, cps[case["ratio"]]) for case in report["cases"]]
    return {"schema": 1, "kind": KIND + "_summary", "date": declaration["date"], "status": report["status"],
            "execution_status": report["execution_status"], "certified": False, "reason": report["reason"], "failed_case": report.get("failed_case"),
            "declaration": {"path": relative, "sha256": declaration_hash}, "raw_report": {"path": declaration["output"], "sha256": raw_hash},
            "protocol_sha256": report.get("protocol_sha256"), "protocol": report.get("protocol"),
            "coverage": {"expected_cases": 6, "completed_cases": len(cases), "missing_cases": expected[len(cases):],
                         "execution_identities_available": audited, "backward_executed": False, "BF16_executed": False},
            "gate_counts": {"whole_model_comparisons": {name: count(case["whole_model_comparisons"][name] for case in cases) for name in ENDPOINTS},
                            "retained_state": {name: {"passed": sum(case["retained_state"][name]["passed"] for case in cases), "total": len(cases)} for name in MEMORY},
                            "original_repeat": {"exact_passed": sum(case["original_repeat"]["exact_equal"] for case in cases), "total": len(cases)},
                            "stage_comparisons": {name: count(row for case in cases for row in case["stage_comparisons"][name]["stages"]) for name in PAIRINGS},
                            "frozen_comparisons": count(row for case in report["cases"] for row in frozen_leaves(case))},
            "cases": cases, "execution_allowance": report.get("execution_allowance"), "large_case_headroom": report.get("large_case_headroom", []),
            "post_execution_integrity": report.get("post_execution_integrity"),
            "limits": [*report.get("limits", declaration.get("limits", [])),
                       "Public JSON fingerprints bind the producer binary audit; exporter reads no checkpoint/corpus binaries or activation arrays.",
                       "All finite coordinate reporting arithmetic is checked; first stage markers are recomputed from validated bounded samples.",
                       "All stage ratios/counts remain; individual coordinate samples are omitted only for passing stages. Failed-stage and pair-first-marker samples retain their numeric values; complete details remain in immutable raw JSON.",
                       "Exact original repeats, own full scores, original stateless scores and original stateful memory remain separate controls.",
                       "Frozen components and FP64 anchors omit upstream propagation and cannot promote a whole-model policy.",
                       "Missing/incomplete cases and explicitly unexecuted nonzero-state quadratic routes provide no passes."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=ROOT / "docs/research/tokenwise-isolation-protocol-2026-10-04.json")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    base.write_summary(args.output, build_summary(args.declaration, args.raw))
    print("Validated tokenwise isolation summary written.")


if __name__ == "__main__":
    main()
