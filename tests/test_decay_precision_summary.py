"""Public decay evidence is validated without checkpoint binaries or CUDA."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.summarize_decay_precision as summary
from test_checkpoint_numerics_summary import evidence as numerics_evidence  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/research/decay-precision-protocol-2026-10-04.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(numerics_evidence):
    _baseline_declaration, baseline_path, root = numerics_evidence
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_hash = summary.base.file_sha256(baseline_path)
    # Copy and rebind the public isolation evidence to the same small synthetic
    # data registry. Actual tokens and saved numerical controls stay unchanged.
    iso_declaration_path = root / summary.ISOLATION_DECLARATION
    iso_declaration = json.loads((ROOT / summary.ISOLATION_DECLARATION).read_text(encoding="utf-8"))
    iso_declaration["baseline_sha256"] = baseline_hash
    save(iso_declaration_path, iso_declaration)
    iso_path = root / iso_declaration["output"]
    iso = json.loads((ROOT / iso_declaration["output"]).read_text(encoding="utf-8"))
    iso["protocol"]["data"] = baseline["protocol"]["data"]
    checkpoints = {item["ratio"]: item for item in baseline["protocol"]["checkpoints"]}
    iso["protocol"]["checkpoints"] = [checkpoints[ratio] for ratio in iso["protocol"]["ratios"]]
    iso["protocol"]["baseline_report"].update(sha256=baseline_hash, protocol_sha256=baseline["protocol_sha256"])
    iso["protocol"]["isolation_declaration"]["sha256"] = summary.base.file_sha256(iso_declaration_path)
    iso["protocol_sha256"] = summary.base.canonical_sha256(iso["protocol"])
    save(iso_path, iso)
    for name in ("scripts/isolate_checkpoint_numerics.py", "scripts/study_decay_precision.py"):
        (root / name).write_bytes((ROOT / name).read_bytes())
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    declaration["baseline_sha256"] = baseline_hash
    declaration["isolation_sha256"] = summary.base.file_sha256(iso_path)
    declaration_path = root / DECLARATION.relative_to(ROOT)
    save(declaration_path, declaration)
    raw_path = root / declaration["output"]
    raw = json.loads((ROOT / declaration["output"]).read_text(encoding="utf-8"))
    for name in ("data", "checkpoints"):
        raw["protocol"][name] = baseline["protocol"][name]
    raw["protocol"]["declaration"]["sha256"] = summary.base.file_sha256(declaration_path)
    raw["protocol"]["baseline_report"].update(sha256=baseline_hash, protocol_sha256=baseline["protocol_sha256"])
    raw["protocol"]["isolation_report"].update(sha256=summary.base.file_sha256(iso_path), protocol_sha256=iso["protocol_sha256"])
    raw["protocol_sha256"] = summary.base.canonical_sha256(raw["protocol"])
    save(raw_path, raw)
    return declaration_path, raw_path, root


def build(evidence):
    declaration, raw, root = evidence
    return summary.build_summary(declaration, raw, root=root, source_root=root)


def alter(evidence, change, resign=False):
    raw = json.loads(evidence[1].read_text(encoding="utf-8"))
    change(raw)
    if resign:
        raw["protocol_sha256"] = summary.base.canonical_sha256(raw["protocol"])
    save(evidence[1], raw)


def test_valid_export_retains_original_failure_and_unpromoted_treatment_passes(evidence):
    result = build(evidence)
    assert result["schema"] == 1 and result["kind"] == "trained_checkpoint_decay_precision_summary"
    assert result["status"] == "completed_with_parity_failures" and result["certified"] is False
    assert result["treatment_counts"] == {"original": {"passed": 2, "total": 3},
                                          summary.TREATMENT: {"passed": 3, "total": 3}}
    assert [(row["ratio"], row["treatment"]) for row in result["rows"]] == [
        (ratio, treatment) for ratio in summary.base.RATIOS for treatment in summary.TREATMENTS]
    assert result["raw_report"]["sha256"] == hashlib.sha256(evidence[1].read_bytes()).hexdigest()
    for row in result["rows"]:
        expected = {"1:3": 186, "1:7": 198, "1:15": 204}[row["ratio"]]
        assert row["internal"]["gradients"] == {"passed": expected, "total": expected, "failed_parameters": []}
        assert row["internal"]["loss"]["passed"] is True
        assert row["internal"]["logits"]["passed"] is not (row["ratio"] == "1:15" and row["treatment"] == "original")
        for route, anchor in row["original_fp32_anchor"].items():
            if row["treatment"] == "original" and route == "reference":
                assert anchor is None and row["next_token_effects"]["original_anchor_reference"] is None
            else:
                assert anchor["gradients"]["passed"] == anchor["gradients"]["total"] == expected
                assert anchor["loss"]["passed"] is True
        effect = row["next_token_effects"]["internal"]
        assert effect["scored_tokens"] == 257 and "per_token" not in effect and "passed" not in effect
    assert all(row["exact_fp32_fields_equal"] for row in result["original_controls"])
    assert result["coverage"]["bf16_executed"] is False
    assert result["coverage"]["retained_state_comparisons_executed"] is False
    assert result["coverage"]["cached_continuation_executed"] is False
    assert result["optimizer_executed"] is False and result["production_defaults_changed"] is False
    assert result["quality_equivalence_margin"] is None
    assert all(item["peak_allocated_mib"] > 0 and "combined" in item["scope"] for item in result["diagnostic_resources"])
    assert str(evidence[2]) not in json.dumps(result)


@pytest.mark.parametrize("change", [
    lambda raw: raw["cases"].pop(),
    lambda raw: raw["cases"].__setitem__(1, raw["cases"][0]),
    lambda raw: raw["cases"][0]["treatments"].pop(),
    lambda raw: raw["cases"][0]["treatments"].__setitem__(1, raw["cases"][0]["treatments"][0]),
    lambda raw: raw["cases"][0].__setitem__("tokens_sha256", "0" * 64),
    lambda raw: raw["cases"][0].__setitem__("weights_sha256", "0" * 64),
])
def test_missing_duplicate_models_treatments_and_input_weights_rejected(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda p: p.__setitem__("runtime_sha256", "0" * 64),
    lambda p: p["runtime"]["precision_flags"].__setitem__("cuda_matmul_allow_tf32", True),
    lambda p: p["source_sha256"].__setitem__("scripts/study_decay_precision.py", "0" * 64),
    lambda p: p["source_sha256"].pop("src/model/attention.py"),
    lambda p: p["checkpoints"][0].__setitem__("sha256", "0" * 64),
    lambda p: p["data"].__setitem__("signature", "0" * 64),
    lambda p: p["baseline_report"].__setitem__("protocol_sha256", "0" * 64),
    lambda p: p["isolation_report"].__setitem__("sha256", "0" * 64),
    lambda p: p["declaration"].__setitem__("sha256", "0" * 64),
    lambda p: p["treatment_policy"].__setitem__("dt_times_A", "FP64 product"),
    lambda p: p.__setitem__("include_bf16", True),
    lambda p: p.__setitem__("optimizer_executed", True),
    lambda p: p.__setitem__("quality_equivalence_margin", 0.01),
])
def test_identity_arithmetic_and_declaration_tamper_rejected_even_after_resigning(evidence, change):
    alter(evidence, lambda raw: change(raw["protocol"]), resign=True)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda arm: arm["fp32_backend"]["gradients"]["checks"].pop(),
    lambda arm: arm["fp32_backend"]["gradients"]["checks"].__setitem__(1, arm["fp32_backend"]["gradients"]["checks"][0]),
    lambda arm: arm["fp32_backend"]["gradients"]["checks"][0].__setitem__("shape", [1]),
    lambda arm: arm["fp32_backend"]["gradients"]["checks"][0].__setitem__("cosine_similarity", 2),
    lambda arm: arm["fp32_backend"]["gradients"].__setitem__("failed_parameters", ["embedding.weight"]),
    lambda arm: arm["fp32_backend"]["logits"].__setitem__("passed", False),
    lambda arm: arm["fp32_backend"]["tolerance"].__setitem__("atol", 0.1),
    lambda arm: arm["common_original_fp32_anchor"].pop("reference"),
    lambda arm: arm["common_original_fp32_anchor"].__setitem__("reference", None),
    lambda arm: arm["common_original_fp32_anchor"]["reference"]["gradients"]["checks"].pop(),
    lambda arm: arm["common_original_fp32_anchor"]["torch_chunked"]["loss"].__setitem__("passed", False),
    lambda arm: arm["common_original_fp32_anchor"]["reference"]["next_token_effects"]["per_token"]["target_ids"].__setitem__(0, 0),
    lambda arm: arm["fp32_backend"]["next_token_effects"].__setitem__("mean_nll_delta", 1),
    lambda arm: arm["fp32_backend"]["next_token_effects"].__setitem__("passed", True),
    lambda arm: arm["operator_observations"].pop(),
    lambda arm: arm["operator_observations"][0].__setitem__("autocast", True),
    lambda arm: arm["scan_function_bindings"].__setitem__("stateful_scan", "src.model.mamba2.ssd_stateful"),
    lambda arm: arm.__setitem__("bf16_cached", {}),
    lambda arm: arm["unexecuted_stages"].pop(),
    lambda arm: arm.__setitem__("passed", False),
])
def test_candidate_metrics_gradients_anchors_tokens_and_execution_coverage_rejected(evidence, change):
    alter(evidence, lambda raw: change(raw["cases"][0]["treatments"][1]))
    with pytest.raises(ValueError):
        build(evidence)


def test_original_control_cannot_change_even_when_its_metric_flag_stays_consistent(evidence):
    def tamper(raw):
        check = raw["cases"][2]["treatments"][0]["fp32_backend"]["logits"]
        check["max_absolute_error"] *= 2
        check["max_tolerance_ratio"] *= 2
    alter(evidence, tamper)
    with pytest.raises(ValueError, match="original full FP32 numerical fields"):
        build(evidence)


def test_original_reference_self_anchor_cannot_be_counted_as_a_pass(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["treatments"][0]["common_original_fp32_anchor"].__setitem__(
        "reference", raw["cases"][0]["treatments"][0]["common_original_fp32_anchor"]["torch_chunked"]))
    with pytest.raises(ValueError, match="self-anchor must remain null"):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda raw: raw.__setitem__("status", "completed"),
    lambda raw: raw.__setitem__("certified", True),
    lambda raw: raw.__setitem__("schema", True),
    lambda raw: raw["cases"][2].__setitem__("passed", True),
    lambda raw: raw["cases"][0].__setitem__("baseline_pass_flag_reproduced", False),
    lambda raw: raw["cases"][0]["common_original_fp32_anchor"]["gradients"].pop(),
])
def test_aggregate_flags_and_anchor_inventory_cannot_hide_failures(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("name", ["baseline_report", "isolation_report"])
def test_immutable_prior_reports_cannot_be_replaced(evidence, name):
    declaration = json.loads(evidence[0].read_text(encoding="utf-8"))
    path = evidence[2] / declaration[name]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="immutable"):
        build(evidence)


def test_strict_reader_rejects_duplicate_keys_and_nonfinite_json(evidence):
    original = evidence[1].read_text(encoding="utf-8")
    save_text = lambda value: evidence[1].write_text(value, encoding="utf-8")
    save_text(original.replace('"schema": 1', '"schema": 1, "schema": 1', 1))
    with pytest.raises(ValueError, match="duplicate"):
        build(evidence)
    save_text(original.replace('"schema": 1', '"schema": Infinity', 1))
    with pytest.raises(ValueError, match="nonfinite"):
        build(evidence)


def test_cli_atomic_write_refuses_overwrite_and_invalid_evidence(evidence, tmp_path, capsys):
    declaration, raw, root = evidence
    output = tmp_path / "summary.json"
    argv = ["--declaration", str(declaration), "--raw", str(raw), "--root", str(root),
            "--source-root", str(root), "--output", str(output)]
    assert summary.main(argv) == 0
    original = output.read_bytes()
    assert json.loads(original)["certified"] is False and b"\r\n" not in original
    with pytest.raises(SystemExit):
        summary.main(argv)
    assert output.read_bytes() == original
    alter(evidence, lambda raw: raw.__setitem__("certified", True))
    failed = tmp_path / "invalid.json"
    with pytest.raises(SystemExit):
        summary.main(argv[:-1] + [str(failed)])
    assert not failed.exists() and not list(tmp_path.glob("*.tmp"))
    capsys.readouterr()


def test_exporter_imports_without_pytorch_or_numpy():
    result = subprocess.run([sys.executable, "-c", "import sys; import scripts.summarize_decay_precision; "
        "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
