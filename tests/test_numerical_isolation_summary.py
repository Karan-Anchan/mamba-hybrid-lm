"""Public-report tamper tests; no private checkpoint, corpus or CUDA is needed."""
import copy
from pathlib import Path

import pytest
from scripts import summarize_numerical_isolation as export

DECLARATION = export.ROOT / "docs/research/numerical-isolation-protocol-2026-10-04.json"


@pytest.fixture
def evidence():
    declaration, digest = export.read_json(DECLARATION)
    report, _ = export.read_json(export.ROOT / declaration["output"])
    baseline, _ = export.read_json(export.ROOT / declaration["baseline_report"])
    return report, declaration, digest, baseline


def test_valid_export_preserves_one_failed_score_and_passing_control(evidence):
    report, declaration, digest, baseline = evidence
    export.validate(report, declaration, digest, baseline)
    summary = export.build_summary(DECLARATION)
    assert summary["certified"] is False
    assert [(row["ratio"], row["logits"]["violation_count"]) for row in summary["rows"]] == [("1:15", 1), ("1:3", 0)]
    assert summary["rows"][0]["first_tolerance_violation"]["stage"] == "logits"


@pytest.mark.parametrize("mutation", [
    lambda r: r["cases"].pop(),
    lambda r: r["cases"][0]["trace"].pop(5),
    lambda r: r["cases"][0].update(start_token=0),
    lambda r: r["cases"][0].update(first_tolerance_violation=None),
    lambda r: r["cases"][0]["frozen_replays"][0].update(operand_pair_checks={}),
    lambda r: r["cases"][0]["frozen_replays"][0]["full_zero_state"]["treatments"]["tokenwise"].update(state_vs_fp64=None),
    lambda r: r["cases"][0]["frozen_replays"][0]["real_prefix_continuation"]["initial_state"].update(sha256="0" * 64),
    lambda r: r["cases"][0]["logits"]["worst_tolerance_ratio"].update(tolerance_ratio=0.5),
    lambda r: r["cases"][0]["candidate_experiments"][0].update(approved_for_production=True),
    lambda r: r["cases"][0].update(all_frozen_checks_passed=False),
    lambda r: r.update(status="completed"),
    lambda r: r.update(certified=True),
])
def test_rejects_missing_checks_changed_inputs_and_false_approval(evidence, mutation):
    report, declaration, digest, baseline = evidence
    changed = copy.deepcopy(report)
    mutation(changed)
    with pytest.raises((ValueError, KeyError)):
        export.validate(changed, declaration, digest, baseline)


def test_rejects_stale_source_even_with_recomputed_protocol_hash(evidence):
    report, declaration, digest, baseline = evidence
    report["protocol"]["source_sha256"]["src/model/mamba2.py"] = "0" * 64
    report["protocol_sha256"] = export.canonical_sha256(report["protocol"])
    with pytest.raises(ValueError, match="stale measured source"):
        export.validate(report, declaration, digest, baseline)


def test_refuses_to_overwrite_existing_summary(tmp_path):
    path = tmp_path / "historical.json"
    path.write_text("retained", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        export.write_summary(path, {})
    assert path.read_text(encoding="utf-8") == "retained"
