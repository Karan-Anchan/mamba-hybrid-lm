"""Public breadth reports retain failures, batch spans and incomplete coverage."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import summarize_decay_breadth as summary

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/research/decay-breadth-protocol-2026-10-04.json"


@pytest.fixture(scope="module")
def public_bytes():
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    names = [DECLARATION.relative_to(ROOT).as_posix(), declaration["output"],
             declaration["baseline_report"], declaration["decay_report"], declaration["baseline_declaration"]]
    return {name: (ROOT / name).read_bytes() for name in names}


@pytest.fixture
def evidence(tmp_path, public_bytes):
    # Only public JSON is copied. source_root is read-only tracked source; no
    # private checkpoint binary, validation corpus, PyTorch or CUDA is used.
    for name, content in public_bytes.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    declaration_path = tmp_path / DECLARATION.relative_to(ROOT)
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    return declaration_path, tmp_path / declaration["output"], tmp_path


def build(evidence):
    declaration, raw, root = evidence
    return summary.build_summary(declaration, raw, root=root, source_root=ROOT)


def alter(evidence, change, resign=False):
    raw = json.loads(evidence[1].read_text(encoding="utf-8"))
    change(raw)
    if resign:
        raw["protocol_sha256"] = summary.base.canonical_sha256(raw["protocol"])
    evidence[1].write_text(json.dumps(raw), encoding="utf-8")


def arm(raw, case=0, policy=1):
    return raw["cases"][case]["treatments"][policy]


def test_complete_export_separates_all_gates_and_keeps_candidate_failures(evidence):
    result = build(evidence)
    assert result["status"] == "completed_with_parity_failures" and result["certified"] is False
    assert result["coverage"]["completed_cases"] == result["coverage"]["expected_cases"] == 15
    assert result["coverage"]["completed_policy_rows"] == 30 and result["coverage"]["missing_cases"] == []
    assert result["treatment_counts"] == {"original": {"passed": 11, "completed": 15, "expected": 15},
        summary.decay.TREATMENT: {"passed": 12, "completed": 15, "expected": 15}}
    original, candidate = (result["gate_counts"][name] for name in summary.decay.TREATMENTS)
    assert original["full_backend"] == {"passed": 14, "total": 15}
    assert candidate["full_backend"] == {"passed": 15, "total": 15}
    assert original["cached"] == {"passed": 11, "total": 15}
    assert candidate["cached"] == {"passed": 12, "total": 15}
    assert candidate["cached_internal_logits"] == {"passed": 133, "total": 135}
    assert candidate["cached_original_stateless_logits"] == {"passed": 102, "total": 105}
    assert original["original_stateful_anchor_bundles"] is None
    assert candidate["original_stateful_anchor_bundles"] == {"passed": 90, "total": 90}
    for policy in (original, candidate):
        assert policy["retained_state_bundles"] == {"passed": 75, "total": 75}
        assert policy["internal_unique_parameter_gradients"] == {"passed": 588, "total": 588}
    failed = [row for row in result["rows"] if not row["passed"] and row["treatment"] == summary.decay.TREATMENT]
    assert [(row["ratio"], row["case_id"]) for row in failed] == [
        ("1:15", "b1-l128-p64"), ("1:15", "b1-l129-p64"), ("1:15", "b2-l512-p128")]
    assert failed[0]["internal"]["cached_logits"]["failed_checks"] == []
    assert failed[0]["original_stateless_anchor"]["cached_logits"]["failed_checks"] == ["full_tokenwise"]
    assert failed[0]["original_stateless_anchor"]["cached_logits"]["worst_tolerance_ratio"] == pytest.approx(1.0044056177139282)
    assert [row["internal"]["cached_logits"]["worst_tolerance_ratio"] for row in failed[1:]] == pytest.approx([1.117571234703064, 1.0986982583999634])
    assert len(result["original_controls"]) == 3
    assert all(row["narrow_full_fields_and_anchor_hashes_equal"] for row in result["original_controls"])
    assert len(result["diagnostic_resources"]) == 15
    assert all(row["peak_allocated_mib"] > 0 and "not throughput" in row["scope"] for row in result["diagnostic_resources"])
    assert str(evidence[2]) not in json.dumps(result)


def test_batch_two_effects_keep_positions_per_row_and_gradients_only_where_executed(evidence):
    result = build(evidence)
    for row in result["rows"]:
        full = row["internal"]["full"]
        if row["batch_size"] == 1:
            assert full["gradients"] is None and "all parameter gradients" in row["unexecuted"]
            continue
        assert row["length"] == 512 and row["gradient_check"] is True
        assert full["gradients"]["passed"] == full["gradients"]["total"]
        for route, comparison in row["original_stateless_anchor"]["full"].items():
            if row["treatment"] == "original" and route == "reference":
                assert comparison is None
            else:
                assert comparison["gradients"]["passed"] == comparison["gradients"]["total"]
        effects = row["next_token_effects"]
        assert effects["internal_full"]["scored_tokens"] == 1024
        assert effects["internal_full"]["position_stop_exclusive"] == 512
        suffix = effects["cached_original_anchor"]["suffix_tokenwise"]
        assert suffix["scored_tokens"] == 768 and suffix["position_start"] == 128 and suffix["position_stop_exclusive"] == 512
        prefix = effects["cached_original_anchor"]["prefix"]
        assert prefix["scored_tokens"] == 2 and prefix["position_start"] == 127 and prefix["position_stop_exclusive"] == 128
        for effect in [effects["internal_full"], *effects["cached_original_anchor"].values()]:
            assert effect["batch_size"] == 2 and "each row" in effect["position_interpretation"]
            assert "per_token" not in effect and "passed" not in effect


@pytest.mark.parametrize("change", [
    lambda raw: raw["cases"].pop(),
    lambda raw: raw["cases"].__setitem__(1, raw["cases"][0]),
    lambda raw: arm(raw)["fp32_cached"]["internal_logits"].pop("full_tokenwise"),
    lambda raw: arm(raw)["fp32_cached"]["original_fp32_anchor_logits"].pop("full_tokenwise"),
    lambda raw: arm(raw)["fp32_cached"]["retained_state"].pop("prefill_full"),
    lambda raw: arm(raw)["fp32_cached"]["original_reference_stateful_anchor"].pop("stateful_full"),
    lambda raw: arm(raw)["operator_observations"].pop(),
    lambda raw: arm(raw)["operator_observations"].append(arm(raw)["operator_observations"][0]),
    lambda raw: arm(raw)["fp32_cached"]["real_prefix"]["fields"].pop(),
    lambda raw: raw["cases"][0]["treatments"].__setitem__(1, raw["cases"][0]["treatments"][0]),
])
def test_missing_duplicate_cases_routes_memory_and_operators_rejected(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda p: p.__setitem__("runtime_sha256", "0" * 64),
    lambda p: p["runtime"]["precision_flags"].__setitem__("cuda_matmul_allow_tf32", True),
    lambda p: p["source_sha256"].__setitem__("scripts/check_decay_breadth.py", "0" * 64),
    lambda p: p["checkpoints"][0].__setitem__("sha256", "0" * 64),
    lambda p: p["data"].__setitem__("signature", "0" * 64),
    lambda p: p["declaration"].__setitem__("sha256", "0" * 64),
    lambda p: p["baseline_report"].__setitem__("sha256", "0" * 64),
    lambda p: p["decay_report"].__setitem__("sha256", "0" * 64),
    lambda p: p.__setitem__("BF16_executed", True),
    lambda p: p.__setitem__("optimizer_executed", True),
    lambda p: p["shared_inputs"][0].__setitem__("tokens_sha256", "0" * 64),
])
def test_identity_tampering_rejected_after_protocol_resigning(evidence, change):
    alter(evidence, lambda raw: change(raw["protocol"]), resign=True)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda raw: arm(raw)["fp32_backend"].__setitem__("gradients", {}),
    lambda raw: arm(raw, 4)["fp32_backend"]["gradients"]["checks"].pop(),
    lambda raw: arm(raw, 4)["common_original_fp32_anchor"]["reference"]["gradients"]["checks"].pop(),
    lambda raw: arm(raw, 4)["fp32_backend"]["gradients"]["checks"][0].__setitem__("cosine_similarity", 2),
    lambda raw: arm(raw)["fp32_cached"]["real_prefix"].__setitem__("synthetic", True),
    lambda raw: arm(raw)["fp32_cached"]["retained_state"]["prefill_full"]["fields"][0].__setitem__("shape", [1]),
    lambda raw: arm(raw)["fp32_cached"]["retained_state"]["prefill_full"]["tolerance"].__setitem__("atol", .1),
    lambda raw: arm(raw)["fp32_cached"]["internal_logits"]["full_tokenwise"].__setitem__("passed", False),
    lambda raw: arm(raw)["scan_function_bindings"].__setitem__("stateful_scan", "src.model.mamba2.ssd_stateful"),
    lambda raw: arm(raw)["fp32_cached"]["next_token_effects"]["suffix_tokenwise"].__setitem__("mean_nll_delta", 1),
    lambda raw: arm(raw)["fp32_backend"]["next_token_effects"].__setitem__("passed", True),
])
def test_gradient_state_arithmetic_and_descriptive_statistics_rejected(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda value: value.__setitem__("position_stop_exclusive", 896),
    lambda value: value.__setitem__("scored_tokens", 384),
    lambda value: value["per_token"]["target_ids"].reverse(),
])
def test_batch_two_flattening_cannot_change_row_spans_or_target_order(evidence, change):
    alter(evidence, lambda raw: change(arm(raw, 4)["fp32_cached"]["next_token_effects"]["suffix_tokenwise"]))
    with pytest.raises(ValueError):
        build(evidence)


def test_narrow257_control_cannot_change_even_with_consistent_pass_flags(evidence):
    alter(evidence, lambda raw: raw["cases"][3]["original_anchor"].__setitem__("logits_sha256", "0" * 64))
    with pytest.raises(ValueError, match="does not reproduce narrow decay"):
        build(evidence)


def test_original_self_anchors_remain_null(evidence):
    alter(evidence, lambda raw: arm(raw, policy=0)["common_original_fp32_anchor"].__setitem__(
        "reference", arm(raw, policy=0)["common_original_fp32_anchor"]["torch_chunked"]))
    with pytest.raises(ValueError, match="self-anchor must be null"):
        build(evidence)


def make_partial(raw, completed=4):
    raw["cases"] = raw["cases"][:completed]
    raw["status"] = raw["execution_status"] = "incomplete"
    raw["reason"] = "RuntimeError: diagnostic time allowance exhausted"
    case_ids = [summary.identifier(case) for case in summary.CASES]
    raw["failed_case"] = {"ratio": summary.base.RATIOS[completed // 5], "case_id": case_ids[completed % 5],
                          "stage": "full/cached model diagnostic"}
    raw["large_case_headroom"] = [entry for entry in raw["large_case_headroom"] if (entry["ratio"], entry["case_id"])
                                 in {(case["ratio"], case["case_id"]) for case in raw["cases"]}]


def test_partial_run_retains_completed_results_and_never_counts_missing_passes(evidence):
    alter(evidence, make_partial)
    result = build(evidence)
    assert result["status"] == "incomplete" and result["certified"] is False
    assert result["coverage"]["completed_cases"] == 4 and result["coverage"]["expected_cases"] == 15
    assert len(result["coverage"]["missing_cases"]) == 11 and len(result["rows"]) == 8
    assert all(item["completed"] == item["passed"] == 4 and item["expected"] == 15 for item in result["treatment_counts"].values())
    assert result["gate_counts"][summary.decay.TREATMENT]["internal_unique_parameter_gradients"] == {"passed": 0, "total": 0}


@pytest.mark.parametrize("bad", ["order", "reason", "stage", "claimed_complete", "claimed_passed"])
def test_partial_metadata_cannot_hide_missing_work(evidence, bad):
    def change(raw):
        make_partial(raw)
        if bad == "order":
            raw["cases"].reverse()
        elif bad == "reason":
            raw["reason"] = None
        elif bad == "stage":
            raw["failed_case"]["case_id"] = "b1-l127-p63"
        elif bad == "claimed_complete":
            raw["execution_status"] = "completed"
        else:
            raw["status"] = "completed"
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


def test_pre_audit_failure_exports_no_protocol_or_measured_passes(evidence):
    raw = {"schema": 1, "kind": summary.KIND, "certified": False, "status": "incomplete",
           "execution_status": "incomplete", "reason": "RuntimeError: CUDA unavailable", "cases": []}
    evidence[1].write_text(json.dumps(raw), encoding="utf-8")
    result = build(evidence)
    assert result["status"] == "incomplete" and result["protocol"] is result["protocol_sha256"] is None
    assert result["rows"] == result["diagnostic_resources"] == []
    assert result["coverage"]["execution_identities_available"] is False
    assert len(result["coverage"]["missing_cases"]) == 15
    assert all(item["completed"] == item["passed"] == 0 for item in result["treatment_counts"].values())


@pytest.mark.parametrize("field,value", [("reason", None), ("certified", True), ("status", "completed")])
def test_invalid_pre_audit_failure_rejected(evidence, field, value):
    raw = {"schema": 1, "kind": summary.KIND, "certified": False, "status": "incomplete",
           "execution_status": "incomplete", "reason": "CUDA unavailable", "cases": []}
    raw[field] = value
    evidence[1].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda raw: raw.__setitem__("status", "completed"),
    lambda raw: raw.__setitem__("certified", True),
    lambda raw: raw.__setitem__("schema", True),
    lambda raw: raw["post_execution_integrity"].__setitem__("checkpoints", False),
    lambda raw: raw["large_case_headroom"].pop(),
    lambda raw: raw["large_case_headroom"][0].__setitem__("free_bytes", 1),
    lambda raw: raw["execution_allowance"].__setitem__("overshoot_seconds", 2),
])
def test_false_completion_integrity_resource_and_time_claims_rejected(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


def test_cli_is_atomic_and_refuses_overwrite(evidence, tmp_path, capsys):
    output = tmp_path / "summary.json"
    argv = ["--declaration", str(evidence[0]), "--raw", str(evidence[1]), "--root", str(evidence[2]),
            "--source-root", str(ROOT), "--output", str(output)]
    assert summary.main(argv) == 0
    original = output.read_bytes()
    assert json.loads(original)["certified"] is False
    with pytest.raises(SystemExit):
        summary.main(argv)
    assert output.read_bytes() == original
    alter(evidence, lambda raw: raw.__setitem__("certified", True))
    invalid = tmp_path / "invalid.json"
    with pytest.raises(SystemExit):
        summary.main(argv[:-1] + [str(invalid)])
    assert not invalid.exists() and not list(tmp_path.glob("*.tmp"))
    capsys.readouterr()


@pytest.mark.parametrize("name", ["baseline_report", "decay_report"])
def test_immutable_prior_bytes_cannot_be_replaced(evidence, name):
    declaration = json.loads(evidence[0].read_text(encoding="utf-8"))
    path = evidence[2] / declaration[name]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="immutable"):
        build(evidence)


def test_duplicate_keys_and_nonfinite_json_are_rejected(evidence):
    original = evidence[1].read_text(encoding="utf-8")
    evidence[1].write_text(original.replace('"schema": 1', '"schema": 1, "schema": 1', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        build(evidence)
    evidence[1].write_text(original.replace('"schema": 1', '"schema": NaN', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        build(evidence)


def test_import_does_not_load_torch_or_numpy():
    result = subprocess.run([sys.executable, "-c", "import sys; import scripts.summarize_decay_breadth; "
        "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
