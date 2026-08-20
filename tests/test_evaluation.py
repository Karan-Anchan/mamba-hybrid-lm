"""Week 4 artifact, prompt, and report contracts."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from src.eval.report import write_artifacts
from src.eval.suite import (
    Week4Protocol,
    build_needle_prompt,
    discover_checkpoints,
    file_sha256,
)
from src.model.config import ModelConfig
from src.model.lm import HybridLM


def test_needle_prompt_has_exact_length_and_stable_depth():
    tokenizer = Tokenizer.from_file("data/tokenizer/openwebtext.json")
    prompt, target, actual_depth = build_needle_prompt(tokenizer, 512, 0.5, "orchid")

    assert len(prompt) == 512
    assert target
    assert 0.45 <= actual_depth <= 0.55
    assert "orchid" in tokenizer.decode(prompt)


def test_protocol_rejects_invalid_or_overlong_geometry():
    with pytest.raises(ValueError, match="unique and increasing"):
        Week4Protocol(context_lengths=(512, 512)).validate()
    with pytest.raises(ValueError, match="RoPE"):
        Week4Protocol(context_lengths=(16384,)).validate()
    with pytest.raises(ValueError, match="depth"):
        Week4Protocol(needle_depths=(1.1,)).validate()


def test_checkpoint_discovery_uses_manifest_hashes(tmp_path: Path):
    config = ModelConfig(
        name="tiny", ratio="1:3", d_model=64, n_layers=4, vocab_size=128,
        head_dim=32, mamba_headdim=32, d_state=16,
    )
    model = HybridLM(config)
    run_dir = tmp_path / "checkpoints" / "training" / "tiny"
    run_dir.mkdir(parents=True)
    best_path = run_dir / "best.pt"
    torch.save({
        "schema": 1,
        "signature": "signed",
        "model": model.state_dict(),
        "model_config": asdict(config),
        "step": 12,
        "val_loss": 3.5,
    }, best_path)
    (run_dir / "manifest.json").write_text(json.dumps({
        "status": "completed",
        "ratio": "1:3",
        "signature": "signed",
        "artifacts": {"best": "best.pt"},
        "artifact_sha256": {"best": file_sha256(best_path)},
    }), encoding="utf-8")

    discovered = discover_checkpoints(tmp_path / "checkpoints", "training", ("1:3",))
    assert discovered["1:3"].step == 12

    best_path.write_bytes(best_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        discover_checkpoints(tmp_path / "checkpoints", "training", ("1:3",))


def test_report_derives_markdown_svg_and_manifest_from_one_result(tmp_path: Path):
    protocol = asdict(Week4Protocol.smoke())
    result = {
        "run_id": "test",
        "training_run_id": "training",
        "created_at": "2026-08-20T00:00:00+00:00",
        "completed_at": "2026-08-20T00:01:00+00:00",
        "protocol": protocol,
        "git": {"commit": "abc", "branch": "main", "dirty": False},
        "runtime": {"device": "cpu"},
        "source_sha256": {"src/eval/suite.py": "123"},
        "variants": [{
            "ratio": "1:3",
            "validation": {"perplexity": 26.3},
            "checkpoint": {"ratio": "1:3", "sha256": "456"},
            "inference": [{
                "context_length": context,
                "prefill_tokens_per_second": 1000.0,
                "decode_tokens_per_second": 50.0,
                "logical_state_bytes": 2**20,
                "attention_kv_bytes": 2**19,
                "mamba_conv_bytes": 1024,
                "mamba_ssm_bytes": 2**18,
            } for context in protocol["context_lengths"]],
            "needle_retrieval": {
                "matches": 1,
                "trials": 2,
                "accuracy": 0.5,
                "by_context": [{"context_length": context, "accuracy": 0.5}
                               for context in protocol["context_lengths"]],
            },
        }],
    }

    manifest = write_artifacts(result, tmp_path)
    assert (tmp_path / "evaluation_results.json").is_file()
    assert "26.300" in (tmp_path / "evaluation_table.md").read_text(encoding="utf-8")
    assert "<svg" in (tmp_path / "plots/decode_throughput.svg").read_text(encoding="utf-8")
    assert "plots/logical_state_memory.svg" in manifest["artifact_sha256"]
