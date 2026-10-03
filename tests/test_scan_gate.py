import json

import pytest
import torch

from scripts import check_scan_backend as gate


def test_nonfinite_comparison_cannot_pass_or_break_json_serialization():
    result = gate.compare(torch.tensor([float("nan"), float("inf")]), torch.zeros(2),
                          gate.TOLERANCES["float32"])
    assert result["passed"] is False and result["finite"] is False
    assert result["max_absolute_error"] is None
    json.dumps(result, allow_nan=False)


def test_comparison_rejects_broadcastable_wrong_shape():
    with pytest.raises(ValueError, match="identical shapes"):
        gate.compare(torch.zeros(2, 1), torch.zeros(2), gate.TOLERANCES["float32"])


def test_cpu_gate_checks_gradients_cache_and_source_provenance():
    result = gate.run_gate("torch_chunked", "cpu", "float32", 8, [1, 9], 123)
    assert result["status"] == "passed"
    assert result["backend"]["fused"] is False
    assert result["model_config"]["layer_types"] == ["mamba", "mamba", "mamba", "attention"]
    assert "scripts/check_scan_backend.py" in result["source_sha256"]
    assert all(case["all_gradients_passed"] and case["gradient_tensors"] > 20 for case in result["cases"])
    assert result["cases"][1]["decode_logits"]["passed"]
    assert result["cases"][1]["final_position"] == 9


@pytest.mark.parametrize("failure,status", [(gate.BackendUnavailableError("no kernel"), "unavailable"),
                                           (RuntimeError("CUDA failure"), "failed")])
def test_failure_report_is_saved_and_exits_nonzero(monkeypatch, tmp_path, capsys, failure, status):
    def fail(*args, **kwargs):
        raise failure
    output = tmp_path / "gate.json"
    monkeypatch.setattr(gate, "run_gate", fail)
    monkeypatch.setattr(gate.sys, "argv", ["check_scan_backend", "--output", str(output)])
    assert gate.main() == 2
    result = json.loads(output.read_text())
    assert result["status"] == status and result["certified"] is False
    assert result["observed_paths"] is None
    assert json.loads(capsys.readouterr().out) == result


def test_attention_only_control_cannot_certify_an_unused_scan():
    with pytest.raises(ValueError, match="at least one Mamba"):
        gate.run_gate("torch_chunked", "cpu", "float32", 8, [9], 123, "1:0")
