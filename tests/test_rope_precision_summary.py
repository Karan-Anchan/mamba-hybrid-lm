"""Compact RoPE exports must bind controls and retain failures without PyTorch."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.summarize_rope_precision as summary
from test_checkpoint_numerics_summary import evidence as numerics_evidence  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/research/rope-precision-protocol-2026-10-04.json"


@pytest.fixture
def evidence(numerics_evidence):
    _base_declaration, baseline_path, root = numerics_evidence
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    declaration["baseline_sha256"] = summary.base.file_sha256(baseline_path)
    declaration_path = root / DECLARATION.relative_to(ROOT)
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    raw_path = root / declaration["output"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads((ROOT / declaration["output"]).read_text(encoding="utf-8"))
    for name in ("data", "checkpoints", "shared_inputs"):
        raw["protocol"][name] = baseline["protocol"][name]
    raw["protocol"]["declaration"]["sha256"] = summary.base.file_sha256(declaration_path)
    raw["protocol"]["baseline_report"]["sha256"] = summary.base.file_sha256(baseline_path)
    raw["protocol_sha256"] = summary.base.canonical_sha256(raw["protocol"])
    producer = root / "scripts/study_rope_precision.py"
    producer.write_bytes((ROOT / "scripts/study_rope_precision.py").read_bytes())
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    return declaration_path, raw_path, root


def build(evidence):
    declaration, raw, root = evidence
    return summary.build_summary(declaration, raw, root=root, source_root=root)


def alter(evidence, change, resign=False):
    path = evidence[1]
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    if resign:
        raw["protocol_sha256"] = summary.base.canonical_sha256(raw["protocol"])
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_valid_export_retains_full_model_failures_and_frozen_exact_agreements(evidence):
    result = build(evidence)
    assert result["kind"] == "rope_precision_isolation_summary" and result["schema"] == 1
    assert result["certified"] is False and result["status"] == "completed_with_parity_failures"
    assert len(result["rows"]) == 9
    assert [(row["ratio"], row["policy"]) for row in result["rows"]] == [
        (ratio, policy) for ratio in summary.base.RATIOS for policy in summary.POLICIES]
    assert result["raw_report"]["sha256"] == hashlib.sha256(evidence[1].read_bytes()).hexdigest()
    assert all(control["exact_fields_equal"] for control in result["original_controls"])
    for row in result["rows"]:
        assert row["internal"]["passed"] == 0 and row["internal"]["total"] == 8
        assert row["retained_state"]["passed"] == 1 and row["retained_state"]["total"] == 4
        assert row["fp32_anchor"]["passed"] is False and row["passed"] is False
        layers = {"1:3": 4, "1:7": 2, "1:15": 1}[row["ratio"]]
        assert row["frozen"]["attention_layers"] == layers
        for phase in ("post_rotation", "post_cast"):
            assert row["frozen"][phase]["total"] == layers * 2
            assert row["frozen"][phase]["total_layers"] == layers
            if row["policy"] != "original":
                assert row["frozen"][phase]["passed"] == layers * 2
                assert row["frozen"][phase]["exact_layers"] == layers
        effect = row["next_token_effects"]["full_vs_fp32_anchor"]
        assert effect["scored_tokens"] == 257 and "per_token" not in effect and "passed" not in effect
        assert row["real_prefix"]["synthetic"] is False and row["segmented_schedule"] == [128, 1]
    assert result["quality_equivalence_margin"] is None
    assert str(evidence[2]) not in json.dumps(result)


@pytest.mark.parametrize("change", [
    lambda raw: raw["cases"].pop(),
    lambda raw: raw["cases"].__setitem__(1, raw["cases"][0]),
    lambda raw: raw["cases"][0]["policies"].pop(),
    lambda raw: raw["cases"][0]["policies"].__setitem__(1, raw["cases"][0]["policies"][0]),
    lambda raw: raw["cases"][0]["frozen_probes"].pop(),
    lambda raw: raw["cases"][0]["frozen_probes"].__setitem__(1, raw["cases"][0]["frozen_probes"][0]),
])
def test_missing_duplicate_ratio_policy_and_frozen_layer_coverage_rejected(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda raw: raw["protocol"].__setitem__("runtime_sha256", "0" * 64),
    lambda raw: raw["protocol"]["runtime"]["precision_flags"].__setitem__("cuda_matmul_allow_tf32", True),
    lambda raw: raw["protocol"]["source_sha256"].__setitem__("scripts/study_rope_precision.py", "0" * 64),
    lambda raw: raw["protocol"]["source_sha256"].pop("src/model/attention.py"),
    lambda raw: raw["protocol"]["checkpoints"][0].__setitem__("sha256", "0" * 64),
    lambda raw: raw["protocol"]["shared_inputs"][0].__setitem__("targets_sha256", "0" * 64),
    lambda raw: raw["protocol"]["baseline_report"].__setitem__("sha256", "0" * 64),
    lambda raw: raw["protocol"]["declaration"].__setitem__("sha256", "0" * 64),
    lambda raw: raw["protocol"].__setitem__("quality_equivalence_margin", .001),
    lambda raw: raw["protocol"]["tolerances"]["bfloat16"].__setitem__("rtol", .03),
])
def test_bound_source_runtime_checkpoint_input_and_policy_tamper_rejected(evidence, change):
    alter(evidence, change, resign=True)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda raw: raw["cases"][0]["frozen_probes"][0]["operands"]["q"].__setitem__("dtype", "torch.float32"),
    lambda raw: raw["cases"][0]["frozen_probes"][0]["operands"]["cos"].__setitem__("sha256", "not-a-hash"),
    lambda raw: raw["cases"][0]["frozen_probes"][0]["policies"][0]["post_rotation"]["full"]["q"].__setitem__("dtype", "torch.bfloat16"),
    lambda raw: raw["cases"][0]["frozen_probes"][0]["policies"][1]["post_rotation"]["cached"]["q"].__setitem__("sha256", "0" * 64),
    lambda raw: raw["cases"][0]["frozen_probes"][0]["policies"][1]["post_cast"].__setitem__("bitwise_equal", False),
    lambda raw: raw["cases"][0]["frozen_probes"][0]["policies"][0]["post_cast"]["comparisons"]["q"].__setitem__("passed", True),
])
def test_frozen_dtypes_hash_relationships_and_metric_flags_rejected(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda check: check["internal_logits"].pop("full_tokenwise"),
    lambda check: check["retained_state"].pop("suffix_one_shot"),
    lambda check: check["next_token_effects"]["full_vs_fp32_anchor"]["per_token"]["target_ids"].__setitem__(0, 0),
    lambda check: check["next_token_effects"]["full_vs_fp32_anchor"].__setitem__("mean_nll_delta", 1),
    lambda check: check["next_token_effects"]["full_vs_fp32_anchor"].__setitem__("passed", True),
    lambda check: check["real_prefix"].__setitem__("synthetic", True),
    lambda check: check["fp32_anchor"].__setitem__("passed", True),
])
def test_full_model_endpoint_target_statistic_and_prefix_tamper_rejected(evidence, change):
    alter(evidence, lambda raw: change(raw["cases"][0]["policies"][1]["bf16_cached"]))
    with pytest.raises(ValueError):
        build(evidence)


def test_original_control_cannot_be_changed_even_with_consistent_metric_flag(evidence):
    def tamper(raw):
        check = raw["cases"][0]["policies"][0]["bf16_cached"]["internal_logits"]["stateful_full"]
        check["max_absolute_error"] *= 2
        check["max_tolerance_ratio"] *= 2
    alter(evidence, tamper)
    with pytest.raises(ValueError, match="original full-model numerical fields"):
        build(evidence)


def test_baseline_cannot_be_replaced_or_control_hash_forged(evidence):
    declaration = json.loads(evidence[0].read_text(encoding="utf-8"))
    baseline = evidence[2] / declaration["baseline_report"]
    baseline.write_bytes(baseline.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="immutable baseline bytes"):
        build(evidence)


@pytest.mark.parametrize("change", [
    lambda raw: raw.__setitem__("status", "completed"),
    lambda raw: raw.__setitem__("certified", True),
    lambda raw: raw.__setitem__("schema", True),
    lambda raw: raw["cases"][0].__setitem__("full_model_passed", True),
    lambda raw: raw["cases"][0]["baseline_control_comparison"].__setitem__("exact_fields_equal", False),
])
def test_overall_flags_cannot_hide_failed_endpoints(evidence, change):
    alter(evidence, change)
    with pytest.raises(ValueError):
        build(evidence)


def test_nonfinite_and_duplicate_json_keys_rejected(evidence):
    original = evidence[1].read_text(encoding="utf-8")
    evidence[1].write_text(original.replace('"schema": 1', '"schema": 1, "schema": 1', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        build(evidence)
    evidence[1].write_text(original.replace('"schema": 1', '"schema": NaN', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        build(evidence)


def test_cli_publishes_atomically_refuses_overwrite_and_invalid_evidence(evidence, tmp_path, capsys):
    declaration, raw, root = evidence
    output = tmp_path / "summary.json"
    argv = ["--protocol", str(declaration), "--report", str(raw), "--root", str(root), "--source-root", str(root), "--output", str(output)]
    assert summary.main(argv) == 0
    original = output.read_bytes()
    assert json.loads(original)["certified"] is False and b"\r\n" not in original
    with pytest.raises(SystemExit):
        summary.main(argv)
    assert output.read_bytes() == original
    alter(evidence, lambda value: value.__setitem__("certified", True))
    failed = tmp_path / "invalid-summary.json"
    with pytest.raises(SystemExit):
        summary.main(argv[:-1] + [str(failed)])
    assert not failed.exists() and not list(tmp_path.glob("*.tmp"))
    capsys.readouterr()


def test_exporter_import_uses_only_standard_library():
    script = "import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); import scripts.summarize_rope_precision; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
