"""Compact trained results must reject missing, altered and misleading evidence."""
import hashlib
import json
from pathlib import Path
import struct

import pytest

import scripts.summarize_checkpoint_numerics as summary

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/research/checkpoint-numerics-protocol-2026-10-04.json"


@pytest.fixture
def evidence(tmp_path):
    declaration = tmp_path / DECLARATION.relative_to(ROOT)
    declaration.parent.mkdir(parents=True)
    declaration.write_bytes(DECLARATION.read_bytes())
    metadata = json.loads(declaration.read_text(encoding="utf-8"))
    raw = tmp_path / metadata["output"]
    raw.parent.mkdir(parents=True)
    raw.write_bytes((ROOT / metadata["output"]).read_bytes())
    # All fixture material comes from public reports and tracked source. No
    # private corpus/checkpoints/manifests, GPU, or 10GB sparse file is needed.
    report = json.loads(raw.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    for relative in protocol["source_sha256"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    targets = report["cases"][0]["fp32_backend"]["next_token_effects"]["per_token"]["target_ids"]
    suffix = struct.pack("<" + "q" * (len(targets) - 1), *targets[:-1])
    token_hash = protocol["shared_inputs"][0]["tokens_sha256"]
    first = next(token for token in range(protocol["data"]["tokenizer"]["vocab_size"])
                 if hashlib.sha256(struct.pack("<q", token) + suffix).hexdigest() == token_hash)
    data = protocol["data"]
    val_path = tmp_path / data["artifacts"]["val"]["path"]
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with val_path.open("wb") as handle:
        handle.seek(protocol["shared_inputs"][0]["start_token"] * 2)
        handle.write(struct.pack("<" + "H" * (len(targets) + 1), first, *targets))
    for label in ("train", "meta"):
        path = tmp_path / data["artifacts"][label]["path"]
        path.write_bytes(b"small synthetic fixture" if label == "train" else b"{}")
    for item in data["artifacts"].values():
        path = tmp_path / item["path"]
        item.update({"sha256": summary.file_sha256(path), "bytes": path.stat().st_size})
    tokenizer = tmp_path / data["tokenizer"]["path"]
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_bytes(b"small synthetic tokenizer identity fixture")
    data["tokenizer"]["sha256"] = summary.file_sha256(tokenizer)
    manifest = {"build_signature": data["build_signature"], "signature": data["signature"],
                "outputs": {label: {"file": Path(item["path"]).name, "sha256": item["sha256"], "bytes": item["bytes"]}
                            for label, item in data["artifacts"].items()}, "tokenizer": data["tokenizer"]}
    data_manifest = tmp_path / data["manifest"]["path"]
    data_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    data["manifest"]["sha256"] = summary.file_sha256(data_manifest)
    for checkpoint in protocol["checkpoints"]:
        manifest = {"status": "completed", "ratio": checkpoint["ratio"], "signature": checkpoint["training_signature"],
                    "artifact_sha256": {"best": checkpoint["sha256"]}, "model_config": checkpoint["model_config"],
                    "data": {"files": {Path(item["path"]).name: {"sha256": item["sha256"], "bytes": item["bytes"]}
                                       for item in data["artifacts"].values()}}}
        path = tmp_path / checkpoint["training_manifest"]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        checkpoint["training_manifest"]["sha256"] = summary.file_sha256(path)
    report["protocol_sha256"] = summary.canonical_sha256(protocol)
    raw.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return declaration, raw, tmp_path


def build(evidence):
    declaration, raw, root = evidence
    return summary.build_summary(declaration, raw, root=root, source_root=root)


def alter(evidence, change, resign=False):
    path = evidence[1]
    report = json.loads(path.read_text(encoding="utf-8"))
    change(report)
    if resign:
        report["protocol_sha256"] = summary.canonical_sha256(report["protocol"])
    path.write_text(json.dumps(report), encoding="utf-8")


def test_complete_export_preserves_failures_gradients_and_descriptive_effects(evidence):
    result = build(evidence)
    assert result["schema"] == 1 and result["date"] == "2026-10-04"
    assert result["certified"] is False and result["status"] == "completed_with_parity_failures"
    assert result["quality_equivalence_margin"] is None
    assert [row["ratio"] for row in result["rows"]] == ["1:3", "1:7", "1:15"]
    assert result["raw_report"]["cases"] == 3
    assert result["raw_report"]["sha256"] == hashlib.sha256(evidence[1].read_bytes()).hexdigest()
    assert not Path(result["raw_report"]["path"]).is_absolute()
    for row, total in zip(result["rows"], (186, 198, 204)):
        assert row["fp32"]["gradients"]["passed"] == row["fp32"]["gradients"]["total"] == total
        assert row["fp32"]["gradients"]["failed_parameters"] == []
        assert row["bf16"]["internal"]["passed"] == 0 and row["bf16"]["internal"]["total"] == 8
        assert row["bf16"]["retained_state"]["passed"] == 1 and row["bf16"]["retained_state"]["total"] == 4
        assert row["bf16"]["retained_state"]["failed_fields"]
        assert row["bf16"]["fp32_anchor"]["passed"] is False
        assert row["bf16"]["suffix_length"] == 129 and row["bf16"]["segmented_schedule"] == [128, 1]
        assert row["bf16"]["real_prefix"]["synthetic"] is False
        effect = row["bf16"]["next_token_effects"]["full_vs_fp32_anchor"]
        assert effect["scored_tokens"] == 257 and "passed" not in effect and "per_token" not in effect
    assert result["rows"][2]["fp32"]["logits"]["passed"] is False
    assert result["rows"][2]["fp32"]["passed"] is False
    assert str(ROOT) not in json.dumps(result)


def test_missing_duplicate_matrix_rows_rejected(evidence):
    original = json.loads(evidence[1].read_text(encoding="utf-8"))
    alter(evidence, lambda raw: raw["cases"].pop())
    with pytest.raises(ValueError, match="missing/duplicate"):
        build(evidence)
    original["cases"][1] = original["cases"][0]
    evidence[1].write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="missing/duplicate"):
        build(evidence)


def test_protocol_hash_rejects_unsigned_changes(evidence):
    alter(evidence, lambda raw: raw["protocol"].update({"seed": 42}))
    with pytest.raises(ValueError, match="protocol hash"):
        build(evidence)


def test_resigned_tolerance_widening_rejects(evidence):
    alter(evidence, lambda raw: raw["protocol"]["tolerances"]["float32"].update({"atol": 0.1}), resign=True)
    with pytest.raises(ValueError, match="tolerances changed"):
        build(evidence)


def test_resigned_source_and_runtime_hash_tampering_rejected(evidence):
    alter(evidence, lambda raw: raw["protocol"]["source_sha256"].update({"src/model/mamba2.py": "0" * 64}), resign=True)
    with pytest.raises(ValueError, match="source hash"):
        build(evidence)


def test_runtime_contents_must_match_runtime_hash(evidence):
    alter(evidence, lambda raw: raw["protocol"]["runtime"]["packages"].update({"torch": "changed"}), resign=True)
    with pytest.raises(ValueError, match="runtime hash"):
        build(evidence)


def test_both_resigned_tf32_flags_cannot_relax_precision(evidence):
    def change(raw):
        runtime = raw["protocol"]["runtime"]
        runtime["precision_flags"]["cuda_matmul_allow_tf32"] = True
        raw["protocol"]["runtime_sha256"] = summary.canonical_sha256(runtime)
    alter(evidence, change, resign=True)
    with pytest.raises(ValueError, match="TF32 flags"):
        build(evidence)


@pytest.mark.parametrize("field", ["tokens_sha256", "targets_sha256"])
def test_shared_input_hashes_cannot_change_between_checkpoints(evidence, field):
    alter(evidence, lambda raw: raw["cases"][1].update({field: "0" * 64}))
    with pytest.raises(ValueError, match="shared input identity"):
        build(evidence)


def test_resigned_shared_hash_must_match_actual_validation_bytes(evidence):
    def change(raw):
        raw["protocol"]["shared_inputs"][0]["tokens_sha256"] = "0" * 64
        for case in raw["cases"]:
            case["tokens_sha256"] = "0" * 64
    alter(evidence, change, resign=True)
    with pytest.raises(ValueError, match="window byte/hash"):
        build(evidence)


def test_checkpoint_must_match_pre_run_declaration(evidence):
    alter(evidence, lambda raw: raw["protocol"]["checkpoints"][0].update({"sha256": "0" * 64}), resign=True)
    with pytest.raises(ValueError, match="pre-run declaration"):
        build(evidence)


@pytest.mark.parametrize("missing", [True, False])
def test_all_unique_parameter_gradients_required(evidence, missing):
    def change(raw):
        gradients = raw["cases"][0]["fp32_backend"]["gradients"]
        if missing:
            gradients["checks"].pop()
            gradients["parameter_tensors"] -= 1
        else:
            gradients["checks"][1] = gradients["checks"][0]
    alter(evidence, change)
    with pytest.raises(ValueError, match="parameter gradient checks"):
        build(evidence)


def test_wrong_parameter_shape_rejected(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["fp32_backend"]["gradients"]["checks"][0].update({"shape": [1]}))
    with pytest.raises(ValueError, match="gradient shape"):
        build(evidence)


def test_tensor_flag_cannot_hide_known_forward_failure(evidence):
    alter(evidence, lambda raw: raw["cases"][2]["fp32_backend"]["logits"].update({"passed": True}))
    with pytest.raises(ValueError, match="passed flag"):
        build(evidence)


def test_aggregate_status_cannot_hide_failed_checks(evidence):
    alter(evidence, lambda raw: raw.update({"status": "completed"}))
    with pytest.raises(ValueError, match="status hides"):
        build(evidence)


def test_failed_gradient_names_must_match_actual_checks(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["fp32_backend"]["gradients"].update({"failed_parameters": ["embed.weight"]}))
    with pytest.raises(ValueError, match="failed gradient names"):
        build(evidence)


def test_state_bundle_cannot_hide_field_failures(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["bf16_cached"]["retained_state"]["full_tokenwise"].update({"passed": True}))
    with pytest.raises(ValueError, match="passed flag"):
        build(evidence)


def test_clone_and_state_shape_facts_are_required(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["bf16_cached"]["real_prefix"].update({"unchanged_after_cloned_continuations": False}))
    with pytest.raises(ValueError, match="prefix clone"):
        build(evidence)


def test_observed_operator_counts_cannot_be_invented(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["operator_observations"][0].update({"calls": 999}))
    with pytest.raises(ValueError, match="operator calls"):
        build(evidence)


def test_probability_statistics_and_shifted_targets_are_checked(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["bf16_cached"]["next_token_effects"]["suffix_tokenwise"].update({"mean_nll_delta": 1.0}))
    with pytest.raises(ValueError, match="descriptive mean"):
        build(evidence)


def test_probability_effect_cannot_claim_a_quality_margin(evidence):
    alter(evidence, lambda raw: raw["cases"][0]["bf16_cached"]["next_token_effects"]["suffix_tokenwise"].update({"passed": True}))
    with pytest.raises(ValueError, match="cannot claim equivalence"):
        build(evidence)


def test_nonfinite_json_and_duplicate_keys_are_rejected(evidence):
    text = evidence[1].read_text(encoding="utf-8")
    evidence[1].write_text(text.replace('"schema": 1', '"schema": 1, "schema": 1', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        build(evidence)
    evidence[1].write_text(text.replace('"schema": 1', '"schema": 1, "invalid": 1e999', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON"):
        build(evidence)


def test_raw_lf_crlf_byte_hash_is_preserved_separately_from_semantic_hash(evidence):
    first = build(evidence)
    evidence[1].write_bytes(evidence[1].read_bytes().replace(b"\n", b"\r\n"))
    second = build(evidence)
    assert first["raw_report"]["sha256"] != second["raw_report"]["sha256"]
    assert first["raw_report"]["canonical_sha256"] == second["raw_report"]["canonical_sha256"]


def test_cli_and_publisher_never_overwrite_existing_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    output.write_text('{"historical":true}', encoding="utf-8")
    monkeypatch.setattr(summary, "build_summary", lambda *_: pytest.fail("cannot build with existing output"))
    with pytest.raises(SystemExit) as stopped:
        summary.main(["--output", str(output)])
    assert stopped.value.code == 2
    with pytest.raises(ValueError, match="already exists"):
        summary.write_summary(output, {"changed": True})
    assert output.read_text(encoding="utf-8") == '{"historical":true}'
    fresh = tmp_path / "fresh.json"
    summary.write_summary(fresh, {"complete": True})
    assert b"\r\n" not in fresh.read_bytes()
