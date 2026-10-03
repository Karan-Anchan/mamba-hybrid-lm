"""Exported precision results reject incomplete, altered or misleading evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.summarize_precision_study as summary

RESEARCH = Path(__file__).resolve().parents[1] / "docs/research"


@pytest.fixture
def evidence(tmp_path):
    declaration = tmp_path / "docs/protocol.json"
    declaration.parent.mkdir()
    declaration.write_bytes((RESEARCH / "precision-protocol-2026-10-04.json").read_bytes())
    paths = []
    for original in sorted((RESEARCH / "checks/precision-2026-10-04").glob("*.json")):
        path = declaration.parent / original.name
        path.write_bytes(original.read_bytes())
        paths.append(path)
    return declaration, paths, tmp_path


def resign_protocol(report):
    report["protocol_sha256"] = hashlib.sha256(json.dumps(report["protocol"], sort_keys=True).encode()).hexdigest()


def alter(path, change, resign=False):
    report = json.loads(path.read_text(encoding="utf-8"))
    change(report)
    if resign:
        resign_protocol(report)
    path.write_text(json.dumps(report), encoding="utf-8")


def build(evidence):
    declaration, paths, root = evidence
    return summary.build_summary(declaration, paths, root)


def test_complete_matrix_preserves_primary_anchor_and_failed_probe_counts(evidence):
    result = build(evidence)
    assert result["schema"] == 1 and result["date"] == "2026-10-04"
    assert result["certified"] is False and result["status"] == "completed_with_parity_failures"
    assert len(result["raw_reports"]) == 6 and len(result["rows"]) == 10
    assert all(row["cases"] == row["internal"]["total"] == 15 for row in result["rows"])
    assert all(row["fp32_anchor"] is None for row in result["rows"] if row["treatment"] == "fp32_reference")
    original = next(row for row in result["rows"] if row["ratio"] == "0:1" and row["treatment"] == "bf16_original")
    assert original["internal"]["passed"] < 15 and original["fp32_anchor"]["passed"] < 15
    assert original["failures"]
    scan = next(row for row in result["exploratory_probes"] if row["ratio"] == "0:1" and row["autocast"])
    assert scan["scan"] == {**scan["scan"], "passed": 4, "total": 15}
    assert scan["direct_gradients"]["passed"] == scan["mixer_projection"]["passed"] == scan["lm_projection"]["passed"] == 15
    assert scan["future_gradient_causality"] == {"passed": 12, "total": 12, "not_applicable": 3}
    assert scan["cb_contraction_output_dtypes"] == ["torch.bfloat16"]
    assert any("attention/MLP" in limit for limit in result["limits"])
    assert any("final token" in limit for limit in result["limits"])
    assert all(not Path(raw["path"]).is_absolute() and raw["sha256"] and raw["canonical_sha256"] for raw in result["raw_reports"])


def test_missing_and_duplicate_matrix_arms_reject(evidence):
    declaration, paths, root = evidence
    with pytest.raises(ValueError, match="missing/extra"):
        summary.build_summary(declaration, paths[:-1], root)
    with pytest.raises(ValueError, match="duplicate raw"):
        summary.build_summary(declaration, [*paths[:-1], paths[0]], root)


def test_protocol_hash_detects_altered_identity(evidence):
    alter(evidence[1][0], lambda report: report["protocol"].update({"seed": 999}))
    with pytest.raises(ValueError, match="protocol hash"):
        build(evidence)


def test_tolerance_widening_rejected_even_with_updated_protocol_hash(evidence):
    alter(evidence[1][0], lambda report: report["protocol"]["tolerances"]["bfloat16"].update({"atol": 0.2}), resign=True)
    with pytest.raises(ValueError, match="tolerances changed"):
        build(evidence)


@pytest.mark.parametrize("field", ["weights_sha256", "tokens_sha256", "targets_sha256"])
def test_changed_case_weights_or_inputs_are_rejected(evidence, field):
    alter(evidence[1][0], lambda report: report["cases"][7].update({field: "0" * 64}))
    with pytest.raises(ValueError, match="changed within a contrast"):
        build(evidence)


def test_missing_case_cannot_shrink_the_denominator(evidence):
    alter(evidence[1][0], lambda report: report["cases"].pop())
    with pytest.raises(ValueError, match="report cases"):
        build(evidence)


def test_inconsistent_metric_flag_is_rejected(evidence):
    alter(evidence[1][0], lambda report: report["cases"][0]["comparisons"]["internal_consistency"]["token_decode_vs_full"].update({"passed": False}))
    with pytest.raises(ValueError, match="metric passed flag"):
        build(evidence)


def test_inconsistent_internal_flag_is_rejected(evidence):
    alter(evidence[1][0], lambda report: report["cases"][0]["comparisons"].update({"internal_passed": False}))
    with pytest.raises(ValueError, match="internal passed flag"):
        build(evidence)


def test_false_backend_path_claim_rejected(evidence):
    alter(evidence[1][0], lambda report: report["cases"][0]["observed_paths"].update({"full": ["fused_cuda"]}))
    with pytest.raises(ValueError, match="observed paths differ"):
        build(evidence)


def test_source_versions_must_agree_across_the_matrix(evidence):
    alter(evidence[1][0], lambda report: report["protocol"]["source_sha256"].update({"src/model/mamba2.py": "0" * 64}), resign=True)
    with pytest.raises(ValueError, match="inconsistent source"):
        build(evidence)


def test_failed_probe_cannot_be_relabeled_as_success(evidence):
    def hide_failed_probe(report):
        for probe in report["exploratory_probes"]:
            for case in probe["scan"]["cases"]:
                if not case["passed"]:
                    case["passed"] = True
                    return
        pytest.fail("real matrix must contain an informative failed probe")
    alter(evidence[1][0], hide_failed_probe)
    with pytest.raises(ValueError, match="probe passed flag"):
        build(evidence)


def test_diagnostic_cannot_claim_certification(evidence):
    alter(evidence[1][0], lambda report: report.update({"certified": True}))
    with pytest.raises(ValueError, match="incorrectly certified"):
        build(evidence)


def test_summary_order_and_canonical_hash_are_stable_when_json_line_endings_change(evidence):
    declaration, paths, root = evidence
    before = build(evidence)
    path = paths[0]
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    after = summary.build_summary(declaration, list(reversed(paths)), root)
    assert after["rows"] == before["rows"]
    before_raw = {item["path"]: item for item in before["raw_reports"]}
    assert all(item["canonical_sha256"] == before_raw[item["path"]]["canonical_sha256"] for item in after["raw_reports"])


def test_output_cannot_replace_raw_reports(evidence, monkeypatch):
    declaration, paths, _ = evidence
    monkeypatch.setattr(summary, "build_summary", lambda *_: pytest.fail("must reject overwriting input before aggregation"))
    with pytest.raises(ValueError, match="must not overwrite"):
        summary.main(["--protocol", str(declaration), "--reports", *map(str, paths), "--output", str(paths[0])])
