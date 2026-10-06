"""CPU standard-library public-evidence checks; no private binaries or Torch."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import summarize_projection_precision as summary

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def historical_transport(tmp_path):
    original_bytes = b'{"status":"completed_with_parity_failures","passed":false}\r\n'
    original_path = "docs/research/checks/prior/raw.json"
    archive = tmp_path / (original_path + ".gz")
    archive.parent.mkdir(parents=True)
    archive.write_bytes(gzip.compress(original_bytes, mtime=0))
    record = {"path": original_path, "path_scope": "repository-relative", "sha256": digest(original_bytes)}
    manifest = {"schema": 1, "kind": "lossless_normalization_head_raw_publication", "compression": "gzip", "path": original_path + ".gz",
                "bytes": archive.stat().st_size, "sha256": digest(archive.read_bytes()), "uncompressed_path": original_path,
                "uncompressed_bytes": len(original_bytes), "uncompressed_sha256": record["sha256"], "round_trip_exact": True,
                "scope": "Lossless transport of the original immutable report; no JSON reserialization or evidence removal."}
    path = tmp_path / "docs/research/transport.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    transport = {"path": "docs/research/transport.json", "path_scope": "repository-relative", "sha256": digest(path.read_bytes())}
    return tmp_path, record, transport, manifest, original_bytes


def rewrite_manifest(fixture):
    root, _record, transport, manifest, _bytes = fixture
    path = root / transport["path"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    transport["sha256"] = digest(path.read_bytes())


def test_exact_gzip_only_public_clone_preserves_original_crlf_bytes(historical_transport):
    root, record, transport, _manifest, _bytes = historical_transport
    before = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
    result = summary.read_lossless_prior(record, transport, root=root)
    assert result == {"status": "completed_with_parity_failures", "passed": False}
    assert not (root / record["path"]).exists()
    assert sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file()) == before


def test_existing_original_must_match_exact_bytes_without_evidence_normalization(historical_transport):
    root, record, transport, _manifest, raw = historical_transport
    path = root / record["path"]
    path.write_bytes(raw)
    assert summary.read_lossless_prior(record, transport, root=root)["passed"] is False
    path.write_bytes(raw.replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="local original"):
        summary.read_lossless_prior(record, transport, root=root)


@pytest.mark.parametrize("change", [
    lambda manifest: manifest.__setitem__("round_trip_exact", False),
    lambda manifest: manifest.__setitem__("uncompressed_sha256", "f" * 64),
    lambda manifest: manifest.__setitem__("uncompressed_path", "other.json"),
    lambda manifest: manifest.__setitem__("path", "../../outside.gz"),
    lambda manifest: manifest.__setitem__("bytes", True),
    lambda manifest: manifest.__setitem__("compression", "plain"),
    lambda manifest: manifest.__setitem__("synthetic_leaf", {"passed": True}),
])
def test_bound_transport_schema_paths_hashes_flags_and_sizes_are_closed(historical_transport, change):
    change(historical_transport[3])
    rewrite_manifest(historical_transport)
    with pytest.raises(ValueError):
        summary.read_lossless_prior(historical_transport[1], historical_transport[2], root=historical_transport[0])


@pytest.mark.parametrize("offset", [-1, 1])
def test_decoded_size_checked_independently_of_resigned_transport(historical_transport, offset):
    historical_transport[3]["uncompressed_bytes"] += offset
    rewrite_manifest(historical_transport)
    with pytest.raises(ValueError, match="decoded bytes|decompressed byte"):
        summary.read_lossless_prior(historical_transport[1], historical_transport[2], root=historical_transport[0])


def test_bounded_decompression_rejects_oversized_declaration_before_read(historical_transport):
    root, record, transport, manifest, _raw = historical_transport
    with pytest.raises(ValueError, match="decompression exceeds"):
        summary.read_lossless_prior(record, transport, root=root, maximum_bytes=manifest["uncompressed_bytes"] - 1)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'[]'])
def test_public_json_rejects_duplicate_nonfinite_or_nonobject_payload(raw):
    with pytest.raises(ValueError):
        summary.parse_public_json(raw)


def test_gzip_payload_corruption_rejected_even_when_archive_hash_is_resigned(historical_transport):
    root, record, transport, manifest, _raw = historical_transport
    archive = root / manifest["path"]
    archive.write_bytes(archive.read_bytes()[:-6])
    manifest.update(bytes=archive.stat().st_size, sha256=digest(archive.read_bytes()))
    rewrite_manifest(historical_transport)
    with pytest.raises(ValueError, match="gzip bytes"):
        summary.read_lossless_prior(record, transport, root=root)


def test_invalid_deflate_block_rejected_as_public_evidence_error(historical_transport):
    root, record, transport, manifest, _raw = historical_transport
    archive = root / manifest["path"]
    raw = bytearray(archive.read_bytes())
    raw[10] = (raw[10] & ~7) | 7  # Reserved deflate block type.
    archive.write_bytes(raw)
    manifest["sha256"] = digest(raw)
    rewrite_manifest(historical_transport)
    with pytest.raises(ValueError, match="gzip bytes"):
        summary.read_lossless_prior(record, transport, root=root)


def test_read_only_module_import_does_not_load_cuda_numerical_or_producer_modules():
    code = "import sys; from scripts import summarize_projection_precision; assert not any(k in sys.modules for k in ['torch','numpy','scripts.study_projection_precision','scripts.study_normalization_head_precision'])"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


# These old, standard-library-only fixture builders allocate metadata, never
# tensors. Import the test helper, not the measured producer or its runtime.
_spec = importlib.util.spec_from_file_location("norm_summary_fixtures", ROOT / "tests/test_normalization_head_precision_summary.py")
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
identity, layout, comparison = _fixtures.identity, _fixtures.layout, _fixtures.comparison


def frozen_projection(cfg, batch, length, *, cell="P0", offset=0, schedule=None):
    shape, out = [batch, length, cfg["d_model"]], [batch, length, summary.projection_width(cfg)]
    calls, position = [], offset
    for size in schedule or [length]:
        calls.append({"position": position, "length": size, "input_layout": layout([batch, size, shape[-1]]), "output_layout": layout([batch, size, out[-1]])})
        position += size
    return {"active_cell": cell, "input": identity(shape), "observed_output": identity(out), "weight": identity([out[-1], shape[-1]]), "bias": None,
        "actual_calls": calls, "position_offset": offset, "replay_layout": layout(shape), "layout_contrast": "captured single-call layout restored" if len(calls) == 1 else "combined contiguous full replay versus actual captured call layouts",
        "tokenwise_shape": [batch, 1, shape[-1]], "identical_operands": True, "input_unchanged": True,
        "oracle": "independent detached CPU NumPy FP64 matrix multiplication and optional bias addition", "oracle_output": identity(out, "torch.float64"),
        "checks": {name: comparison(out, reference="torch.float64" if name.endswith("vs_fp64") else "torch.float32") for name in summary.PROJECTION_CHECKS},
        "outputs": {name: identity(out) for name in ("p0_full", "p0_tokenwise", "p1_full", "p1_tokenwise")}, "passed": True, "exploratory": True}


def projection_cell(cell_id, old, cfg, shared):
    cell = deepcopy(old)
    for key in ("normalization_fp64_coefficients", "head_fp64_matmul"):
        del cell[key]
    cell.update(cell_id=cell_id, input_projection_fp64_matmul=cell_id == "P1", normalization_cell="N1H0", projection_sites=[name for name, _ in summary.projection_inventory(cfg)])
    batch, length, prefix = (shared[key] for key in ("batch_size", "length", "prefix_length"))
    shape = [batch, length, cfg["vocab_size"]]
    for name in summary.ENDPOINTS[len(summary.norm.ENDPOINTS):]:
        cell["whole_model_comparisons"][name] = comparison(shape)
    for name in summary.PAIRINGS[len(summary.norm.PAIRINGS):]:
        cell["stage_comparisons"][name] = _fixtures.stages(cfg, batch, length)
    cell["losses"] = {name: identity([]) for name in ("full", "chunked128")}
    cell["loss_comparisons"] = {name: comparison([]) for name in summary.LOSS_ENDPOINTS}
    for bank in ("retained_state", "stateful_full_retained_state"):
        cell[bank]["projection_baseline"] = _fixtures.state_bundle(cfg, batch, length)
    actual = cell["actual_prefix"]
    actual["prefix_comparisons"]["projection_baseline"] = comparison([batch, 1, cfg["vocab_size"]])
    for route in summary.norm.SUFFIX_ROUTES:
        actual["suffix_comparisons"][route]["projection_baseline"] = comparison([batch, length - prefix, cfg["vocab_size"]])
        actual["retained_state"][route]["projection_baseline"] = _fixtures.state_bundle(cfg, batch, length)
    sites = summary.projection_inventory(cfg)
    selected = sites[:1] if len(sites) == 1 else [sites[0], sites[-1]]
    actual.update(projection_sites=[name for name, _ in selected], frozen_projections=[], execution_status="completed", failed_projection=None, measured_routes_passed=True)
    suffix = length - prefix
    schedules = {"one_shot": [suffix], "chunked128": [min(128, suffix - start) for start in range(0, suffix, 128)], "tokenwise": [1] * suffix}
    for route in summary.norm.SUFFIX_ROUTES:
        for name, layer in selected:
            actual["frozen_projections"].append({"module": name, "layer": layer, "route": route, "origin": "actual_same_cell_prefix_continuation",
                "prefix_position": prefix, "prefix_fields": deepcopy(actual["prefix_fields"]), "final_position": length,
                "upstream_input_vs_own_full": comparison([batch, suffix, cfg["d_model"]]), "observed_output_vs_own_full": comparison([batch, suffix, summary.projection_width(cfg)]),
                "frozen_arithmetic": frozen_projection(cfg, batch, suffix, cell=cell_id, offset=prefix, schedule=schedules[route])})
    return cell


def control(current, previous, original, old_original):
    checks = summary.reproduction_expectations(current, previous, original, old_original)
    return {"executed": True, "comparisons": checks, "exact_equal": all(checks.values()), "scope": "all prior N1H0 score/stage/state and genuine-prefix comparison fields repeated exactly; prior failures remain immutable"}


@pytest.fixture
def toy():
    case, shared, checkpoint, _previous, policy = _fixtures.toy.__wrapped__()
    previous = deepcopy(case)
    cfg = checkpoint["model_config"]
    old = deepcopy(case["cells"][1])
    case["cells"] = [projection_cell(name, old, cfg, shared) for name in summary.CELLS]
    case.pop("frozen_norms")
    case.pop("frozen_head")
    case["frozen_projections"] = [{"module": name, "layer": layer, "origin": "identical_P0_stateless_full_operands", **frozen_projection(cfg, shared["batch_size"], shared["length"])} for name, layer in summary.projection_inventory(cfg)]
    case["decay_baseline_anchor"].pop("cell")
    case["projection_baseline_anchor"] = {"logits": identity([shared["batch_size"], shared["length"], cfg["vocab_size"]]), "state_fields": _fixtures.state_identities(cfg, shared["batch_size"], shared["length"]), "cell": "P0"}
    case["prior_reproduction"] = control(case["cells"][0], old, case["original_anchor"], previous["original_anchor"])
    case["rng_integrity"] = {"before": {name: "a" * 64 for name in ("python", "numpy", "torch_cpu", "torch_cuda")}, "after": {name: "a" * 64 for name in ("python", "numpy", "torch_cpu", "torch_cuda")}, "restoration_exact": True}
    observed = [deepcopy(case["operator_observations"][0])]
    observed.append({**deepcopy(observed[0]), "stage": "decay_baseline.stateful_full"})
    observed += [{**deepcopy(row), "stage": ("P0" if row["stage"].startswith("N0H0") else "P1") + "." + row["stage"].split(".", 1)[1]} for row in case["operator_observations"] if row["stage"].startswith(("N0H0", "N1H0"))]
    case["operator_observations"] = observed
    return case, shared, checkpoint, previous, policy


def validate_toy(toy):
    summary.validate_case(*toy)


def refresh_recorded(case):
    for cell in case["cells"]:
        complete = cell["execution_status"] == "completed"
        cell["actual_prefix"]["measured_routes_passed"] = all(row["passed"] for row in summary.prefix_leaves(cell["actual_prefix"], frozen=False))
        cell["actual_prefix"]["passed"] = complete and cell["actual_prefix"]["measured_routes_passed"]
        cell["whole_model_passed"] = complete and all(row["passed"] for row in [*cell["whole_model_comparisons"].values(), *cell["loss_comparisons"].values(), *cell["retained_state"].values(), *cell["stateful_full_retained_state"].values()]) and cell["actual_prefix"]["passed"]
        cell["all_recorded_comparisons_passed"] = complete and all(row["passed"] for row in summary.cell_leaves(cell))
    case["all_recorded_comparisons_passed"] = case["execution_status"] == "completed" and case["attribution_control_stable"] and all(row["passed"] for row in summary.case_leaves(case))


def test_complete_projection_matrix_and_scalar_losses_validate(toy):
    validate_toy(toy)
    case, shared, checkpoint, _previous, _policy = toy
    compact = summary.compact_case(case, checkpoint["model_config"])
    assert len(compact["cells"]) == 2 and compact["missing_cells"] == []
    assert len(compact["cells"][0]["stage_comparisons"]) == 7
    assert len(compact["cells"][0]["whole_model_comparisons"]) == 15
    assert len(compact["cells"][0]["loss_comparisons"]) == 7
    assert compact["frozen_comparison_counts"]["prefix_projections"] == {"passed": 48, "total": 48, "worst_tolerance_ratio": 0.}
    assert compact["cells"][0]["actual_paths"]["tokenwise"]["scan_calls"] == shared["length"]


@pytest.mark.parametrize("change", [
    lambda c: c["cells"].pop(),
    lambda c: c["cells"].__setitem__(1, deepcopy(c["cells"][0])),
    lambda c: c["cells"][0]["whole_model_comparisons"].pop("full_vs_projection_baseline"),
    lambda c: c["cells"][0]["loss_comparisons"].pop("chunked_vs_projection_baseline"),
    lambda c: c["cells"][0]["stage_comparisons"].pop("tokenwise_vs_projection_baseline"),
    lambda c: c["cells"][0]["retained_state"].pop("projection_baseline"),
    lambda c: c["cells"][0]["stateful_full_retained_state"].pop("original"),
    lambda c: c["cells"][0]["actual_prefix"]["suffix_comparisons"]["chunked128"].pop("projection_baseline"),
    lambda c: c["cells"][0]["actual_prefix"]["frozen_projections"].pop(),
    lambda c: c["frozen_projections"].pop(),
    lambda c: c["frozen_projections"][0]["checks"].pop(summary.PROJECTION_CHECKS[-1]),
    lambda c: c["frozen_projections"][0].__setitem__("oracle", "shared torch.double linear"),
    lambda c: c["frozen_projections"][0]["oracle_output"].__setitem__("dtype", "torch.float32"),
    lambda c: c["frozen_projections"][0]["checks"][summary.PROJECTION_CHECKS[0]]["tolerance"].__setitem__("atol", .1),
    lambda c: c["cells"][0]["loss_comparisons"][summary.LOSS_ENDPOINTS[0]].__setitem__("shape", [1]),
    lambda c: c["cells"][0]["actual_prefix"]["frozen_projections"][0].__setitem__("origin", "synthetic_nonzero_prefix"),
    lambda c: c["cells"][0]["actual_prefix"]["frozen_projections"][0]["prefix_fields"][0].__setitem__("sha256", "b" * 64),
    lambda c: c["cells"][0]["actual_prefix"]["frozen_projections"][0]["frozen_arithmetic"]["actual_calls"][0].__setitem__("position", 0),
    lambda c: c["cells"][0]["actual_prefix"]["frozen_projections"][-1]["frozen_arithmetic"]["actual_calls"].pop(),
    lambda c: c["cells"][0]["actual_prefix"].__setitem__("synthetic", True),
    lambda c: c["cells"][0]["actual_prefix"]["final_positions"].__setitem__("one_shot", 3),
    lambda c: c["cells"][0]["actual_paths"]["tokenwise"]["scan"].pop(),
    lambda c: c["operator_observations"].pop(),
    lambda c: c["runtime_policy"]["active"].__setitem__("cuda_matmul_allow_tf32", True),
    lambda c: c["tied_weight_integrity"].__setitem__("parameter_identity_unchanged", False),
    lambda c: c["rng_integrity"]["after"].__setitem__("torch_cpu", "b" * 64),
    lambda c: c.__setitem__("post_model_weights_unchanged", False),
    lambda c: c["prior_reproduction"]["comparisons"].pop("original.logits"),
])
def test_missing_or_tampered_routes_controls_or_arithmetic_rejected(toy, change):
    change(toy[0])
    with pytest.raises(ValueError):
        validate_toy(toy)


@pytest.mark.parametrize("where", ["full", "prefix", "loss"])
def test_extra_pass_leaves_cannot_inflate_explicit_counts(toy, where):
    case = toy[0]
    before = len(summary.case_leaves(case))
    if where == "full":
        target = case["frozen_projections"][0]
    elif where == "prefix":
        target = case["cells"][0]["actual_prefix"]["frozen_projections"][0]["frozen_arithmetic"]
    else:
        target = case["cells"][0]["loss_comparisons"][summary.LOSS_ENDPOINTS[0]]
    target["synthetic_leaf"] = {"passed": True, "finite": True, "max_tolerance_ratio": 0.}
    assert len(summary.case_leaves(case)) == before
    with pytest.raises(ValueError, match="schema"):
        validate_toy(toy)


def test_prior_negative_control_can_fail_without_erasing_passing_candidate(toy):
    case, _shared, _cp, previous, _policy = toy
    old = next(cell for cell in previous["cells"] if cell["cell_id"] == "N1H0")
    old["logits"]["full"]["sha256"] = "b" * 64
    case["prior_reproduction"] = control(case["cells"][0], old, case["original_anchor"], previous["original_anchor"])
    case["attribution_control_stable"] = case["all_recorded_comparisons_passed"] = False
    validate_toy(toy)
    assert case["cells"][1]["whole_model_passed"] is True
    assert case["prior_reproduction"]["comparisons"]["logits"] is False
    case["prior_reproduction"]["comparisons"]["logits"] = True
    with pytest.raises(ValueError, match="exact prior"):
        validate_toy(toy)


def test_actual_loss_failure_is_separate_and_cannot_be_promoted(toy):
    case = toy[0]
    case["cells"][1]["loss_comparisons"]["chunked_vs_projection_baseline"] = comparison([], error=6e-5)
    refresh_recorded(case)
    validate_toy(toy)
    assert not case["cells"][1]["whole_model_passed"]
    assert all(leaf["passed"] for leaf in case["cells"][1]["whole_model_comparisons"].values())


def test_hidden_intermediate_failure_retained_when_final_scores_pass(toy):
    case = toy[0]
    trace = case["cells"][1]["stage_comparisons"]["tokenwise_vs_original"]
    index = next(i for i, row in enumerate(trace["stages"]) if row["stage"] == "lm_head.input")
    old = trace["stages"][index]
    row = {"layer": old["layer"], "stage": old["stage"], **comparison(old["shape"], error=6e-5), "sample_policy": "full_failed"}
    trace["stages"][index] = row
    trace["first_nonzero"] = {"layer": row["layer"], "stage": row["stage"], "sample": deepcopy(row["first_nonzero"])}
    trace["first_violation"] = {"layer": row["layer"], "stage": row["stage"], "sample": deepcopy(row["first_violation"])}
    trace["passed"] = False
    refresh_recorded(case)
    validate_toy(toy)
    assert case["cells"][1]["whole_model_passed"]
    view = view_toy(toy)
    pair = view["cases"][0]["cells"][1]["stage_comparisons"]["tokenwise_vs_original"]
    retained = next(stage for stage in pair["stages"] if stage["stage"] == "lm_head.input")
    assert retained["violation_samples"] == row["violation_samples"]
    assert pair["first_violation"] == trace["first_violation"]
    assert pair["counts"]["total"] == len(trace["stages"])


def view_toy(toy):
    case, _identity, cp, _old, _policy = toy
    full = {"kind": summary.KIND + "_summary", "certified": False, "gate_counts": {"opaque_validated": "unchanged"}, "coverage": {"expected_cells": 12}, "cases": [summary.compact_case(case, cp["model_config"])]}
    return summary.project_website_view(full, {"path": "docs/research/full.json", "sha256": "a" * 64})


def test_website_projection_keeps_gates_counts_and_binding_and_omits_only_passing_samples(toy):
    validate_toy(toy)
    view = view_toy(toy)
    assert view["kind"] == summary.KIND + "_website_view"
    assert view["source_summary"] == {"path": "docs/research/full.json", "sha256": "a" * 64}
    assert view["gate_counts"] == {"opaque_validated": "unchanged"}
    assert view["coverage"] == {"expected_cells": 12}
    assert "worst_tolerance_ratio" not in view["cases"][0]["frozen_projections"][0]["checks"][summary.PROJECTION_CHECKS[0]]
    field = view["cases"][0]["cells"][0]["retained_state"]["original"]["fields"][0]
    assert field["tolerance"] == summary.base.TOLERANCES["float32"]
    assert "worst_absolute_error" not in field
    for pair in view["cases"][0]["cells"][0]["stage_comparisons"].values():
        assert sum(row["stage"] == "block_output" for row in pair["stages"]) == 2
        assert pair["projection_coverage"]["full_stage_count"] == pair["counts"]["total"]


def test_incomplete_frozen_prefix_retains_measured_endpoints_without_complete_gate(toy):
    case = toy[0]
    cell, prefix = case["cells"][1], case["cells"][1]["actual_prefix"]
    failed = prefix["frozen_projections"].pop()
    prefix.update(execution_status="incomplete", failed_projection={"route": failed["route"], "module": failed["module"]}, reason="cooperative allowance exhausted")
    cell["execution_status"] = case["execution_status"] = "incomplete"
    case.update(failed_cell="P1", reason="cooperative allowance exhausted")
    refresh_recorded(case)
    validate_toy(toy)
    assert prefix["measured_routes_passed"] and not prefix["passed"]
    assert len(cell["whole_model_comparisons"]) == 15 and not cell["whole_model_passed"]
    compact = summary.compact_case(case, toy[2]["model_config"])
    assert compact["missing_cells"] == ["P1"]
    cell["whole_model_passed"] = True
    with pytest.raises(ValueError, match="aggregate"):
        validate_toy(toy)


def test_incomplete_P0_prefix_has_no_invented_prior_control_or_full_replays(toy):
    case = toy[0]
    case["cells"] = case["cells"][:1]
    cell, prefix = case["cells"][0], case["cells"][0]["actual_prefix"]
    failed = prefix["frozen_projections"].pop(0)
    prefix["frozen_projections"] = []
    prefix.update(execution_status="incomplete", failed_projection={"route": failed["route"], "module": failed["module"]}, reason="frozen memory limit")
    case["frozen_projections"] = []
    case.pop("prior_reproduction")
    case["operator_observations"] = [row for row in case["operator_observations"] if not row["stage"].startswith("P1.")]
    cell["execution_status"] = case["execution_status"] = "incomplete"
    case.update(failed_cell="P0", reason="frozen memory limit", attribution_control_stable=False)
    refresh_recorded(case)
    validate_toy(toy)
    assert not case["attribution_control_stable"]
    assert len(summary.case_leaves(case)) > 0


def test_post_case_rng_restoration_failure_is_legitimate_incomplete_evidence(toy):
    case = toy[0]
    case["rng_integrity"]["after"]["torch_cpu"] = "b" * 64
    case["rng_integrity"]["restoration_exact"] = False
    case.update(execution_status="incomplete", failed_cell=None, reason="RNG restoration failed", all_recorded_comparisons_passed=False)
    validate_toy(toy)
    # This coverage test uses public registered case names while the toy case
    # keeps metadata-only tensor geometry.
    case["case_id"] = summary.CASE_IDS[0]
    value = {"cases": [case], "execution_status": "incomplete", "reason": "RNG restoration failed", "failed_case": {"ratio": case["ratio"], "case_id": case["case_id"], "cell": "P1", "stage": "P1.frozen_suffix_tokenwise.blocks.0.mixer.in_proj"}}
    expected = [(case["ratio"], case["case_id"])] + [(ratio, identity) for ratio in summary.RATIOS for identity in summary.CASE_IDS if (ratio, identity) != (case["ratio"], case["case_id"])]
    summary.validate_coverage(value, expected)
    case["rng_integrity"]["restoration_exact"] = True
    with pytest.raises(ValueError, match="actual restoration failure"):
        summary.validate_coverage(value, expected)


def test_inherited_file_audit_order_ignores_json_object_key_sorting(toy):
    case, _identity, cp, _previous, _policy = toy
    paths = ["src/model/z.py", "src/model/a.py", "scripts/study_projection_precision.py"]
    artifacts = {"z": {"path": "data/z.bin", "sha256": "a" * 64}, "a": {"path": "data/a.bin", "sha256": "a" * 64}}
    declaration = {"data": {"manifest": {"path": "data/manifest.json", "sha256": "a" * 64}, "tokenizer": {"path": "data/tokenizer.json", "sha256": "a" * 64}, "artifacts": dict(sorted(artifacts.items()))},
        "checkpoints": [{"path": "cp/best.pt", "sha256": "a" * 64, "training_manifest": {"path": "cp/manifest.json", "sha256": "a" * 64}}],
        "prior_declaration": {"path": "old/decl.json", "sha256": "a" * 64}, "prior_summary": {"path": "old/summary.json", "sha256": "a" * 64},
        "prior_transport": {"selected_representation": {"path": "old/raw.json.gz", "sha256": "a" * 64}, "publication": {"path": "old/transport.json", "sha256": "a" * 64}},
        "source_sha256": {path: "a" * 64 for path in sorted(paths)}}
    prior = {"post_execution_audit_files": [{"group": group, "path": path, "expected_sha256": "a" * 64} for group, path in
        [("prepared_data_and_tokenizer", "data/manifest.json"), ("prepared_data_and_tokenizer", "data/tokenizer.json"), ("prepared_data_and_tokenizer", "data/z.bin"), ("prepared_data_and_tokenizer", "data/a.bin"), ("sources", "src/model/z.py"), ("sources", "src/model/a.py")]]}
    registry = summary.audit_registry(declaration, "new/decl.json", "a" * 64, prior)
    assert [row[0] for row in registry["sources"]] == paths
    assert [row[0] for row in registry["prepared_data_and_tokenizer"]] == ["data/manifest.json", "data/tokenizer.json", "data/z.bin", "data/a.bin"]
    declaration["data"]["artifacts"].pop("z")
    with pytest.raises(ValueError, match="inventory changed"):
        summary.audit_registry(declaration, "new/decl.json", "a" * 64, prior)


def test_bound_actual_declaration_keeps_frozen_protocol_and_text_source_contract():
    path = ROOT / "docs/research/projection-precision-protocol-2026-10-06.json"
    if not path.exists():
        pytest.skip("optional locally bound public declaration absent")
    declaration, _hash = summary.base.read_json(path)
    prior_plan = summary.read_public_json(declaration["prior_declaration"])
    # This unit checks declaration/source closure, not prior measurements;
    # the full chain is validated in build_summary, never assumed here.
    summary.validate_declaration(declaration, prior_plan, {"protocol": prior_plan}, ROOT)
    for key, value in (("tolerances", {"float32": {"atol": .1, "rtol": .1}}), ("cells", ["P0"]), ("optimizer_executed", True), ("seed", True)):
        changed = deepcopy(declaration)
        changed[key] = value
        with pytest.raises(ValueError):
            summary.validate_declaration(changed, prior_plan, {"protocol": prior_plan}, ROOT)


def test_source_line_ending_equivalence_is_only_for_text_source(tmp_path):
    relative = "scripts/source.py"
    path = tmp_path / relative
    path.parent.mkdir()
    raw = b"value = 1\n# preserved source\n"
    path.write_bytes(raw.replace(b"\n", b"\r\n"))
    summary.isolation.source_check({relative: digest(raw)}, tmp_path, {relative})
    path.write_bytes(b"value = 2\n# altered source\n")
    with pytest.raises(ValueError, match="stale measured source"):
        summary.isolation.source_check({relative: digest(raw)}, tmp_path, {relative})


def test_complete_six_case_coverage_rejects_duplicates_omissions_and_incomplete_rows():
    expected = [(ratio, case) for ratio in summary.RATIOS for case in summary.CASE_IDS]
    value = {"cases": [{"ratio": ratio, "case_id": case, "execution_status": "completed"} for ratio, case in expected], "execution_status": "completed", "reason": None, "failed_case": None}
    summary.validate_coverage(value, expected)
    for altered in [value["cases"][:-1], [*value["cases"][:-1], deepcopy(value["cases"][0])]]:
        with pytest.raises(ValueError):
            summary.validate_coverage({**value, "cases": altered}, expected)
    value["cases"][-1]["execution_status"] = "incomplete"
    with pytest.raises(ValueError):
        summary.validate_coverage(value, expected)


def test_actual_negative_suffix_window_207_is_not_cropped_or_repositioned_in_view():
    # Synthetic bounded comparison at the same recorded negative position;
    # preserve the local coordinate and the genuine prefix separately.
    leaf = comparison([2, 384, 16], error=6e-5)
    for key in summary.norm.SAMPLES - {"violation_samples"}:
        if leaf[key] is not None:
            leaf[key]["coordinate"] = [0, 79, 0]
    leaf["violation_samples"][0]["coordinate"] = [0, 79, 0]
    summary.norm.detailed(leaf, [2, 384, 16])
    prefix = {"prefix_position": 128, "suffix_comparisons": {"chunked128": {"original": leaf}}}
    assert prefix["prefix_position"] + prefix["suffix_comparisons"]["chunked128"]["original"]["first_violation"]["coordinate"][1] == 207
    assert summary.norm.website_comparison(leaf) == leaf


def test_failed_memory_and_local_arithmetic_samples_survive_website_projection(toy):
    case, _shared, cp, _previous, _policy = toy
    cell = case["cells"][1]
    field = cell["retained_state"]["original"]["fields"][0]
    changed = {"field": field["field"], **comparison(field["shape"], error=6e-5)}
    cell["retained_state"]["original"]["fields"][0] = changed
    cell["retained_state"]["original"]["passed"] = False
    frozen = cell["actual_prefix"]["frozen_projections"][0]["frozen_arithmetic"]
    name = summary.PROJECTION_CHECKS[-1]
    frozen["checks"][name] = comparison(frozen["observed_output"]["shape"], reference="torch.float64", error=6e-5)
    frozen["passed"] = False
    refresh_recorded(case)
    validate_toy(toy)
    view = view_toy(toy)
    rendered = view["cases"][0]["cells"][1]
    assert rendered["retained_state"]["original"]["fields"][0] == changed
    assert rendered["actual_prefix"]["frozen_projections"][0]["frozen_arithmetic"]["checks"][name] == frozen["checks"][name]
    assert rendered["whole_model_passed"] is False
    assert rendered["frozen_prefix_comparison_counts"]["passed"] == 23


def test_preaudit_failure_keeps_all_twelve_cells_missing_and_path_identity(tmp_path, monkeypatch, toy):
    _case, shared, cp, _previous, _policy = toy
    declaration = {"output": "docs/research/raw.json", "date": "2026-10-06", "checkpoints": [cp], "shared_inputs": [{**shared, "case_id": case} for case in summary.CASE_IDS]}
    declaration_path = tmp_path / "docs/research/decl.json"
    raw_path = tmp_path / declaration["output"]
    declaration_path.parent.mkdir(parents=True)
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    failure = {"schema": 1, "kind": summary.KIND, "certified": False, "status": "incomplete", "execution_status": "incomplete", "reason": "public preflight failed", "cases": []}
    raw_path.write_text(json.dumps(failure), encoding="utf-8")
    monkeypatch.setattr(summary, "validate_prior_chain", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(summary, "validate_declaration", lambda *_a, **_k: None)
    exported = summary.build_summary(declaration_path, root=tmp_path, source_root=tmp_path)
    assert not exported["execution_identities_available"] and not exported["final_audit_passed"]
    assert exported["coverage"]["completed_cells"] == 0
    assert len(exported["coverage"]["missing_cells"]) == 12
    assert all(gates["whole_model"]["total"] == 0 for gates in exported["gate_counts"]["cells"].values())
    with pytest.raises(ValueError, match="raw report path differs"):
        summary.build_summary(declaration_path, tmp_path / "different.json", root=tmp_path, source_root=tmp_path)


def test_website_view_atomic_publication_does_not_overwrite_prior_bytes(tmp_path, toy):
    path = tmp_path / "view.json"
    view = view_toy(toy)
    summary.norm.write_website_view(path, view)
    original = path.read_bytes()
    assert original.endswith(b"\n") and b"\r\n" not in original
    with pytest.raises(ValueError, match="already exists"):
        summary.norm.write_website_view(path, {"different": True})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def projection_transport(fixture, *, auto=True):
    root, record, transport, manifest, raw = fixture
    manifest.update(kind="lossless_projection_precision_raw_publication", scope=summary.RAW_TRANSPORT_SCOPE)
    path = root / (summary.RAW_PUBLICATION if auto else transport["path"])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_new_raw_gzip_only_fallback_and_existing_original_have_identical_identity(historical_transport):
    root, record, _transport, _manifest, raw = historical_transport
    projection_transport(historical_transport)
    decoded, sha = summary.read_raw_report(record["path"], root=root)
    assert decoded == {"status": "completed_with_parity_failures", "passed": False}
    assert sha == digest(raw)
    assert not (root / record["path"]).exists()
    (root / record["path"]).write_bytes(raw)
    assert summary.read_raw_report(record["path"], root=root) == (decoded, sha)
    (root / record["path"]).write_bytes(raw.replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="local original"):
        summary.read_raw_report(record["path"], root=root)


@pytest.mark.parametrize("change", [
    lambda m: m.__setitem__("scope", "partial cropped trace"),
    lambda m: m.__setitem__("uncompressed_path", "changed/original.json"),
    lambda m: m.__setitem__("kind", "lossless_normalization_head_raw_publication"),
    lambda m: m.__setitem__("uncompressed_sha256", "b" * 64),
    lambda m: m.__setitem__("uncompressed_bytes", 2**40),
    lambda m: m.__setitem__("bytes", True),
    lambda m: m.__setitem__("extra_public_pass", True),
])
def test_raw_fallback_cannot_change_path_hash_scope_or_size(historical_transport, change):
    root, record, _transport, manifest, _raw = historical_transport
    path = projection_transport(historical_transport)
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        summary.read_raw_report(record["path"], root=root)


def test_explicit_raw_transport_is_required_and_must_stay_inside_repository(historical_transport):
    root, record, _transport, _manifest, _raw = historical_transport
    path = projection_transport(historical_transport, auto=False)
    assert summary.read_raw_report(record["path"], root=root, publication_path=path)[1] == record["sha256"]
    with pytest.raises(ValueError, match="manifest missing"):
        summary.read_raw_report(record["path"], root=root, publication_path=root / "absent.json")
    with pytest.raises(ValueError, match="outside repository"):
        summary.read_raw_report(record["path"], root=root, publication_path=root.parent / "outside.json")


def test_gzip_only_public_clone_rebuilds_the_exact_full_summary_and_view_bytes(tmp_path, monkeypatch, toy):
    _case, shared, cp, _previous, _policy = toy
    declaration = {"output": "docs/research/raw.json", "date": "2026-10-06", "checkpoints": [cp], "shared_inputs": [{**shared, "case_id": case} for case in summary.CASE_IDS]}
    declaration_path = tmp_path / "docs/research/decl.json"
    raw_path = tmp_path / declaration["output"]
    declaration_path.parent.mkdir(parents=True)
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    failure = {"schema": 1, "kind": summary.KIND, "certified": False, "status": "incomplete", "execution_status": "incomplete", "reason": "public preflight failed", "cases": []}
    raw_bytes = (json.dumps(failure) + "\r\n").encode()
    raw_path.write_bytes(raw_bytes)
    # Isolate raw transport from the separately tested immutable historical
    # validators. No measurement/certification is simulated by this fixture.
    monkeypatch.setattr(summary, "validate_prior_chain", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(summary, "validate_declaration", lambda *_a, **_k: None)
    full = summary.build_summary(declaration_path, root=tmp_path, source_root=tmp_path)
    full_path = tmp_path / "docs/research/summary.json"
    summary.base.write_summary(full_path, full)
    view = summary.build_website_view(full_path, root=tmp_path, source_root=tmp_path)
    view_path = tmp_path / "docs/research/view.json"
    summary.norm.write_website_view(view_path, view)
    full_bytes, view_bytes = full_path.read_bytes(), view_path.read_bytes()
    archive = raw_path.with_suffix(".json.gz")
    archive.write_bytes(gzip.compress(raw_bytes, mtime=0))
    manifest = {"schema": 1, "kind": "lossless_projection_precision_raw_publication", "compression": "gzip", "path": declaration["output"] + ".gz", "bytes": archive.stat().st_size,
        "sha256": digest(archive.read_bytes()), "uncompressed_path": declaration["output"], "uncompressed_bytes": len(raw_bytes), "uncompressed_sha256": digest(raw_bytes), "round_trip_exact": True,
        "scope": summary.RAW_TRANSPORT_SCOPE}
    (tmp_path / summary.RAW_PUBLICATION).write_text(json.dumps(manifest), encoding="utf-8")
    raw_path.unlink()
    regenerated = summary.build_summary(declaration_path, root=tmp_path, source_root=tmp_path)
    assert regenerated == full
    assert (json.dumps(regenerated, indent=2, allow_nan=False) + "\n").encode() == full_bytes
    regenerated_view = summary.build_website_view(full_path, root=tmp_path, source_root=tmp_path)
    assert (json.dumps(regenerated_view, separators=(",", ":"), allow_nan=False) + "\n").encode() == view_bytes
    assert regenerated_view["raw_report"]["sha256"] == digest(raw_bytes)
    assert not raw_path.exists()
    assert full_path.read_bytes() == full_bytes and view_path.read_bytes() == view_bytes
    changed = deepcopy(full)
    changed["gate_counts"]["cells"]["P1"]["whole_model"]["passed"] = 1
    full_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from independently validated"):
        summary.build_website_view(full_path, root=tmp_path, source_root=tmp_path)


def test_failed_stage_serialization_is_identical_across_python_hash_seeds():
    code = """
import importlib.util, json
from pathlib import Path
from scripts import summarize_projection_precision as export
spec = importlib.util.spec_from_file_location('projection_test_metadata', Path('tests/test_projection_precision_summary.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
row = {'layer': None, 'stage': 'lm_head.input', **fixture.comparison([2, 4, 8], error=6e-5), 'sample_policy': 'full_failed'}
print(json.dumps(export.compact_stage(row), separators=(',', ':')))
"""
    outputs = [subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": seed}, check=True, capture_output=True).stdout for seed in ("1", "873")]
    assert outputs[0] == outputs[1]
    row = json.loads(outputs[0])
    assert tuple(key for key in row if key in summary.norm.SAMPLES) == summary.FAILED_STAGE_SAMPLE_KEYS
    assert row["violation_samples"] and row["first_violation"]["tolerance_ratio"] > 1
