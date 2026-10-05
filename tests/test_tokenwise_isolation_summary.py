"""Standard-library public-evidence validation; no model, CUDA or private binary fixture."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import summarize_tokenwise_isolation as summary

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/research/tokenwise-isolation-protocol-2026-10-04.json"


@pytest.fixture(scope="module")
def public_evidence():
    declaration, declaration_hash = summary.base.read_json(DECLARATION)
    report, raw_hash = summary.base.read_json(ROOT / declaration["output"])
    prior, _prior_hash = summary.base.read_json(ROOT / declaration["breadth_report"]["path"])
    paths = [DECLARATION.relative_to(ROOT).as_posix(), declaration["output"],
             declaration["breadth_declaration"]["path"], declaration["breadth_report"]["path"]]
    return {"declaration": declaration, "declaration_hash": declaration_hash, "report": report, "prior": prior,
            "raw_hash": raw_hash, "bytes": {path: (ROOT / path).read_bytes() for path in paths}}


@pytest.fixture
def report(public_evidence):
    return deepcopy(public_evidence["report"])


def validate(report, evidence):
    summary.validate(report, evidence["declaration"], evidence["declaration_hash"], DECLARATION.relative_to(ROOT).as_posix(),
                     evidence["prior"], ROOT)


def write_evidence(tmp_path, evidence, report=None):
    for name, content in evidence["bytes"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if report is not None:
        path = tmp_path / evidence["declaration"]["output"]
        path.write_text(json.dumps(report), encoding="utf-8")
    return tmp_path / DECLARATION.relative_to(ROOT)


def test_actual_export_keeps_six_cases_endpoints_controls_and_hidden_stage_failures():
    result = summary.build_summary(DECLARATION)
    assert result["status"] == "completed_with_parity_failures" and not result["certified"]
    assert result["coverage"]["completed_cases"] == result["coverage"]["expected_cases"] == 6
    assert result["coverage"]["missing_cases"] == []
    assert not result["coverage"]["backward_executed"] and not result["coverage"]["BF16_executed"]
    gates = result["gate_counts"]
    assert gates["whole_model_comparisons"][summary.PAIRINGS[0]]["passed"] == 6
    assert gates["whole_model_comparisons"][summary.PAIRINGS[1]]["passed"] == 4
    assert gates["whole_model_comparisons"][summary.PAIRINGS[2]]["passed"] == 3
    assert gates["original_repeat"] == {"exact_passed": 6, "total": 6}
    assert all(row == {"passed": 6, "total": 6} for row in gates["retained_state"].values())
    case = next(row for row in result["cases"] if row["ratio"] == "1:3" and row["batch_size"] == 2)
    assert case["whole_model_passed"] and not case["all_recorded_comparisons_passed"]
    assert case["attribution_control_stable"] and not case["backward_executed"] and case["gradient_check"]
    failed_names = {row["stage"] for pairing in case["stage_comparisons"].values() for row in pairing["stages"] if not row["passed"]}
    assert {"final_norm", "lm_head.input"} <= failed_names
    assert case["frozen_comparison_counts"]["passed"] == case["frozen_comparison_counts"]["total"]
    assert case["stage_comparison_counts"]["passed"] < case["stage_comparison_counts"]["total"]
    for row in result["cases"]:
        assert len(row["selected_frozen_mixers"]) <= 3
        assert all(not probe["scan"]["routes"][name]["executed"] for probe in row["actual_prefix"]["scan_probes"]
                   for name in ("original_quadratic", "candidate_quadratic"))
        assert all("passed" not in probe["scan"]["routes"][name] for probe in row["actual_prefix"]["scan_probes"]
                   for name in ("original_quadratic", "candidate_quadratic"))


def test_stage_compaction_retains_negative_samples_and_positive_shapes_ratios_counts(public_evidence):
    case = public_evidence["report"]["cases"][1]
    stages = case["stage_comparisons"][summary.PAIRINGS[1]]["stages"]
    failed = next(row for row in stages if not row["passed"])
    positive = next(row for row in stages if row["passed"] and row["first_nonzero"] is not None)
    compact = summary.compact_stage(failed)
    for key in ("first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples"):
        assert compact[key] == failed[key]
        assert summary.compact_detail(failed)[key] == failed[key]
    passing = summary.compact_stage(positive)
    assert all(key not in passing for key in ("first_nonzero", "first_violation", "worst_absolute_error", "worst_tolerance_ratio", "violation_samples"))
    for key in ("layer", "stage", "passed", "shape", "max_absolute_error", "max_tolerance_ratio", "nonzero_error_count", "violation_count"):
        assert passing[key] == positive[key]


@pytest.mark.parametrize("change", [
    lambda raw: raw["cases"].pop(),
    lambda raw: raw["cases"].__setitem__(1, deepcopy(raw["cases"][0])),
    lambda raw: raw["cases"][0]["stage_comparisons"].pop(summary.PAIRINGS[0]),
    lambda raw: raw["cases"][0]["stage_comparisons"][summary.PAIRINGS[1]]["stages"].pop(),
    lambda raw: raw["cases"][0]["retained_state"].pop(summary.MEMORY[0]),
    lambda raw: raw["cases"][0]["whole_model_comparisons"].pop(summary.ENDPOINTS[-1]),
    lambda raw: raw["cases"][0]["actual_paths"][summary.ROUTES[-1]]["scan"].pop(),
    lambda raw: raw["cases"][0]["actual_paths"][summary.ROUTES[-1]]["attention"][1].__setitem__("key_length", 1),
])
def test_missing_duplicate_stage_endpoint_memory_or_real_route_coverage_rejected(report, public_evidence, change):
    change(report)
    with pytest.raises(ValueError):
        validate(report, public_evidence)


@pytest.mark.parametrize("change", [
    lambda case: case["stage_comparisons"][summary.PAIRINGS[1]].__setitem__("first_nonzero", None),
    lambda case: case["stage_comparisons"][summary.PAIRINGS[1]].__setitem__("first_violation", None),
    lambda case: case.__setitem__("whole_model_passed", True),
    lambda case: case.__setitem__("all_recorded_comparisons_passed", True),
    lambda case: case["original_repeat"].__setitem__("exact_equal", False),
    lambda case: case.__setitem__("attribution_control_stable", False),
    lambda case: case["whole_model_comparisons"][summary.PAIRINGS[1]]["tolerance"].__setitem__("atol", 1e-3),
])
def test_first_markers_boolean_aggregates_original_control_and_tolerance_not_relabelled(report, public_evidence, change):
    case = report["cases"][1]  # Actual 1:15 L129 threshold failure.
    change(case)
    with pytest.raises(ValueError):
        validate(report, public_evidence)


@pytest.mark.parametrize("change", [
    lambda prefix: prefix.__setitem__("prefix_unchanged", False),
    lambda prefix: prefix.__setitem__("synthetic", True),
    lambda prefix: prefix.__setitem__("prefix_position", 0),
    lambda prefix: prefix["scan_probes"][0]["scan"]["initial"].__setitem__("sha256", "0" * 64),
    lambda prefix: prefix["scan_probes"][0]["scan"].__setitem__("origin", "exploratory_supplied_operands"),
    lambda prefix: prefix["scan_probes"][0]["scan"]["routes"]["candidate_quadratic"].__setitem__("executed", True),
])
def test_real_prefix_clone_position_hash_origin_and_unexecuted_quadratic_contracts(report, public_evidence, change):
    change(report["cases"][0]["actual_prefix"])
    with pytest.raises(ValueError):
        validate(report, public_evidence)


def test_attention_same_layout_replay_and_batch_shape_cannot_change(report, public_evidence):
    case = next(case for case in report["cases"] if any("attention" in row for row in case["selected_frozen_mixers"]))
    replay = next(row["attention"] for row in case["selected_frozen_mixers"] if "attention" in row)
    replay["replay_layout"]["v"]["storage_offset"] += 1
    with pytest.raises(ValueError, match="strides/offsets"):
        validate(report, public_evidence)


def test_unvalidated_extra_frozen_leaf_cannot_inflate_success_counts(report, public_evidence):
    case = report["cases"][0]
    before = summary.count(summary.frozen_leaves(case))
    case["frozen_lm_head"]["synthetic_leaf"] = {"passed": True, "finite": True, "max_tolerance_ratio": 0.0}
    assert summary.count(summary.frozen_leaves(case)) == before
    with pytest.raises(ValueError, match="linear probe schema"):
        validate(report, public_evidence)


@pytest.mark.parametrize("change", [
    lambda value: value.__setitem__("comparison_dtype", "torch.float32"),
    lambda value: value.__setitem__("reference_dtype", "torch.float32"),
    lambda value: value.__setitem__("elements", True),
    lambda value: value["worst_tolerance_ratio"]["coordinate"].__setitem__(0, 99),
    lambda value: value["worst_tolerance_ratio"].__setitem__("absolute_error", 0.5),
])
def test_finite_fp64_comparison_coordinate_arithmetic_and_counts_are_checked(report, public_evidence, change):
    value = report["cases"][0]["frozen_lm_head"]["full_vs_fp64"]
    change(value)
    with pytest.raises(ValueError):
        validate(report, public_evidence)


def test_incomplete_prefix_coverage_keeps_missing_cases_and_completed_failures(tmp_path, public_evidence, report):
    report["cases"] = report["cases"][:2]
    report.update({"execution_status": "incomplete", "status": "incomplete", "reason": "RuntimeError: injected allowance stop",
                   "failed_case": {"ratio": "1:15", "case_id": summary.CASE_IDS[-1], "stage": "trace and frozen components"},
                   "large_case_headroom": []})
    declaration_path = write_evidence(tmp_path, public_evidence, report)
    result = summary.build_summary(declaration_path, root=tmp_path, source_root=ROOT)
    assert result["status"] == "incomplete" and result["coverage"]["completed_cases"] == 2
    assert len(result["coverage"]["missing_cases"]) == 4
    assert result["gate_counts"]["whole_model_comparisons"][summary.PAIRINGS[1]]["total"] == 2
    assert not result["cases"][1]["whole_model_comparisons"][summary.PAIRINGS[1]]["passed"]


def test_final_integrity_failure_with_all_cases_and_no_active_case_is_retained(report, public_evidence):
    report.update({"execution_status": "incomplete", "status": "incomplete", "reason": "post-execution identity check failed", "failed_case": None})
    report["post_execution_integrity"]["sources"] = False
    validate(report, public_evidence)


def test_pre_audit_unexecuted_report_has_no_runtime_or_green_rows(tmp_path, public_evidence):
    report = {"schema": 1, "kind": summary.KIND, "certified": False, "execution_status": "incomplete",
              "status": "incomplete", "reason": "CUDA unavailable; no fallback", "cases": []}
    declaration_path = write_evidence(tmp_path, public_evidence, report)
    result = summary.build_summary(declaration_path, root=tmp_path, source_root=ROOT)
    assert result["protocol"] is None and result["cases"] == []
    assert not result["coverage"]["execution_identities_available"] and len(result["coverage"]["missing_cases"]) == 6
    assert result["gate_counts"]["frozen_comparisons"]["total"] == 0


def test_prior_bytes_and_canonical_fingerprint_cannot_be_replaced(tmp_path, public_evidence):
    declaration_path = write_evidence(tmp_path, public_evidence)
    prior_path = tmp_path / public_evidence["declaration"]["breadth_report"]["path"]
    prior_path.write_bytes(prior_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="prior immutable bytes"):
        summary.build_summary(declaration_path, root=tmp_path, source_root=ROOT)


def test_lf_crlf_only_source_equivalence_accepts_line_endings_but_rejects_code_change(tmp_path):
    source = tmp_path / "example.py"
    lf = b"answer = 42\nprint(answer)\n"
    source.write_bytes(lf.replace(b"\n", b"\r\n"))
    summary.source_check({"example.py": hashlib.sha256(lf).hexdigest()}, tmp_path, {"example.py"})
    source.write_bytes(b"answer = 43\nprint(answer)\n")
    with pytest.raises(ValueError, match="stale measured source"):
        summary.source_check({"example.py": hashlib.sha256(lf).hexdigest()}, tmp_path, {"example.py"})


def test_exporter_import_is_standard_library_only():
    command = [sys.executable, "-c", "import sys; import scripts.summarize_tokenwise_isolation; print('torch' in sys.modules); print('scripts.isolate_tokenwise_numerics' in sys.modules)"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert completed.stdout.splitlines() == ["False", "False"]
