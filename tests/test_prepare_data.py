"""Local-stream checks for immutable, token-targeted dataset preparation."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import src.data.prepare_data as prepare_module
import scripts.run_sweep as sweep_module
from src.data.prepare_data import PrepareConfig, prepare_dataset, sha256_file, validate_prepared_dataset
from src.data.train_tokenizer import EOT


def _fixture(tmp_path: Path, *, train_tokens=7, val_tokens=5, run_id="build-1"):
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab = {f"v{i}": i for i in range(98)} | {"[UNK]": 98, EOT: 99}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_path))
    documents = [f"v{i} v{i + 20}" for i in range(20)]
    cfg = PrepareConfig(
        dataset="openwebtext",
        tokenizer=str(tokenizer_path),
        out_dir=str(tmp_path / "openwebtext-5b"),
        run_id=run_id,
        source="local/openwebtext",
        revision="0123456789abcdef0123456789abcdef01234567",
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        progress_docs=1,
    )
    return cfg, documents


def _write_manifest(path: Path, manifest: dict) -> None:
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def test_sweep_requires_explicit_data_directory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_sweep.py"])
    with pytest.raises(SystemExit) as exc_info:
        sweep_module.main()
    assert exc_info.value.code == 2


def _resign_manifest(manifest: dict) -> None:
    identity = {key: manifest[key] for key in prepare_module._BUILD_IDENTITY_KEYS}
    manifest["signature"] = prepare_module.canonical_hash(prepare_module._manifest_identity(
        identity, manifest["build_signature"], manifest["selection"], manifest["outputs"],
    ))


def test_token_targets_are_met_with_non_overlapping_deterministic_ranges(tmp_path):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))

    manifest = validate_prepared_dataset(output)
    assert manifest["selection"]["val"] == {
        "docs": 2, "overshoot_tokens": 1, "source_doc_end": 1, "source_doc_start": 0,
        "split": "train", "target": 5, "target_unit": "tokens", "tokens": 6,
    }
    assert manifest["selection"]["train"] == {
        "docs": 3, "overshoot_tokens": 2, "source_doc_end": 4, "source_doc_start": 2,
        "split": "train", "target": 7, "target_unit": "tokens", "tokens": 9,
    }
    assert manifest["selection"]["shared_source_stream"] is True
    assert np.fromfile(output / "val.bin", dtype="<u2").tolist() == [0, 20, 99, 1, 21, 99]
    assert np.fromfile(output / "train.bin", dtype="<u2").tolist() == [2, 22, 99, 3, 23, 99, 4, 24, 99]
    assert manifest["tokenizer"]["sha256"] == sha256_file(Path(cfg.tokenizer))
    assert manifest["config"]["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert manifest["tool"]["sha256"] and manifest["runtime"]["packages"]["tokenizers"]
    assert not (tmp_path / ".openwebtext-5b.staging-build-1").exists()


def test_legacy_document_targets_remain_exact(tmp_path):
    cfg, documents = _fixture(tmp_path)
    cfg = replace(cfg, train_tokens=None, val_tokens=None, train_docs=3, val_docs=2)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    manifest = validate_prepared_dataset(output)
    assert manifest["selection"]["val"]["docs"] == 2
    assert manifest["selection"]["train"]["docs"] == 3
    assert manifest["outputs"]["val"]["tokens"] == 6
    assert manifest["outputs"]["train"]["tokens"] == 9


def test_interruption_preserves_stage_and_same_run_refuses_unsafe_resume(tmp_path):
    cfg, documents = _fixture(tmp_path, train_tokens=9, val_tokens=9)

    def interrupted(_split):
        yield documents[0]
        yield documents[1]
        raise ConnectionError("stream stopped")

    with pytest.raises(ConnectionError, match="stream stopped"):
        prepare_dataset(cfg, iterator_factory=interrupted)

    stage = tmp_path / ".openwebtext-5b.staging-build-1"
    assert stage.is_dir() and not Path(cfg.out_dir).exists()
    state = json.loads((stage / "stage_manifest.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["progress"] == {"train": {"docs": 0, "tokens": 0}, "val": {"docs": 2, "tokens": 6}}
    partial_hash = sha256_file(stage / "val.bin")

    with pytest.raises(RuntimeError, match="Streaming cursor resume is not supported"):
        prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    assert sha256_file(stage / "val.bin") == partial_hash


def test_verified_output_is_never_overwritten_and_incompatible_reuse_rejects(tmp_path):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    original_hash = sha256_file(output / "train.bin")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    assert sha256_file(output / "train.bin") == original_hash

    reused = prepare_dataset(
        replace(cfg, reuse_existing=True),
        iterator_factory=lambda _split: pytest.fail("compatible reuse must not consume the stream"),
    )
    assert reused == output and sha256_file(output / "train.bin") == original_hash

    with pytest.raises(RuntimeError, match="incompatible"):
        prepare_dataset(
            replace(cfg, train_tokens=10, reuse_existing=True),
            iterator_factory=lambda _split: iter(documents),
        )

    with (output / "train.bin").open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(RuntimeError, match="integrity"):
        prepare_dataset(
            replace(cfg, reuse_existing=True),
            iterator_factory=lambda _split: iter(documents),
        )


def test_failed_pre_promotion_verification_leaves_output_absent(tmp_path, monkeypatch):
    cfg, documents = _fixture(tmp_path)
    real_validate = prepare_module.validate_prepared_dataset

    def reject_stage(path, expected_build_identity=None, expected_signature=None):
        if ".staging-" in path.name:
            raise RuntimeError("simulated integrity failure")
        return real_validate(path, expected_build_identity, expected_signature)

    monkeypatch.setattr(prepare_module, "validate_prepared_dataset", reject_stage)
    with pytest.raises(RuntimeError, match="simulated integrity failure"):
        prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    assert not Path(cfg.out_dir).exists()
    stage = tmp_path / ".openwebtext-5b.staging-build-1"
    assert stage.is_dir()
    assert json.loads((stage / "stage_manifest.json").read_text(encoding="utf-8"))["status"] == "interrupted"


@pytest.mark.parametrize("bad_name", ["../outside.bin", "C:/outside.bin"])
def test_manifest_artifact_names_reject_parent_and_absolute_paths(tmp_path, bad_name):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["outputs"]["train"]["file"] = bad_name
    _resign_manifest(manifest)
    _write_manifest(output, manifest)

    with pytest.raises(RuntimeError, match="exact filename"):
        validate_prepared_dataset(output)


def test_manifest_rejects_symlink_or_reparse_artifact(tmp_path):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    train_path = output / "train.bin"
    outside = tmp_path / "outside-train.bin"
    train_path.replace(outside)
    try:
        os.symlink(outside, train_path)
    except OSError as exc:
        outside.replace(train_path)
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(RuntimeError, match="regular file|reparse point"):
        validate_prepared_dataset(output)


def test_manifest_rejects_reparse_attribute_without_following_it(tmp_path, monkeypatch):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    train_path = output / "train.bin"
    real_lstat = prepare_module.os.lstat
    reparse_flag = getattr(prepare_module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    monkeypatch.setattr(
        prepare_module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", reparse_flag, raising=False,
    )

    def marked_lstat(path):
        result = real_lstat(path)
        if Path(path) == train_path:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=reparse_flag)
        return result

    monkeypatch.setattr(prepare_module.os, "lstat", marked_lstat)
    with pytest.raises(RuntimeError, match="reparse point"):
        validate_prepared_dataset(output)


def test_config_and_signatures_are_recomputed_from_canonical_contents(tmp_path):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    original = (output / "manifest.json").read_bytes()

    manifest = json.loads(original)
    manifest["config"]["train_tokens"] = 8
    _write_manifest(output, manifest)
    with pytest.raises(RuntimeError, match="build signature"):
        validate_prepared_dataset(output)

    (output / "manifest.json").write_bytes(original)
    manifest = json.loads(original)
    manifest["signature"] = "0" * 64
    _write_manifest(output, manifest)
    with pytest.raises(RuntimeError, match="canonical contents"):
        validate_prepared_dataset(output)


def test_reuse_binds_current_identity_even_if_stored_manifest_is_self_consistent(tmp_path):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["train_tokens"] = 8
    manifest["selection"]["train"]["target"] = 8
    manifest["selection"]["train"]["overshoot_tokens"] = 1
    identity = {key: manifest[key] for key in prepare_module._BUILD_IDENTITY_KEYS}
    manifest["build_signature"] = prepare_module.canonical_hash(identity)

    meta_path = output / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["build_signature"] = manifest["build_signature"]
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["outputs"]["meta"].update({
        "bytes": meta_path.stat().st_size,
        "sha256": sha256_file(meta_path),
    })
    _resign_manifest(manifest)
    _write_manifest(output, manifest)
    validate_prepared_dataset(output)

    with pytest.raises(RuntimeError, match="incompatible"):
        prepare_dataset(
            replace(cfg, reuse_existing=True),
            iterator_factory=lambda _split: pytest.fail("incompatible reuse must not stream"),
        )


def test_checksum_and_output_record_swaps_fail_closed(tmp_path):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    original = (output / "manifest.json").read_bytes()

    manifest = json.loads(original)
    manifest["outputs"]["train"]["sha256"], manifest["outputs"]["val"]["sha256"] = (
        manifest["outputs"]["val"]["sha256"], manifest["outputs"]["train"]["sha256"],
    )
    _resign_manifest(manifest)
    _write_manifest(output, manifest)
    with pytest.raises(RuntimeError, match="integrity"):
        validate_prepared_dataset(output)

    (output / "manifest.json").write_bytes(original)
    manifest = json.loads(original)
    manifest["outputs"]["train"], manifest["outputs"]["val"] = (
        manifest["outputs"]["val"], manifest["outputs"]["train"],
    )
    _resign_manifest(manifest)
    _write_manifest(output, manifest)
    with pytest.raises(RuntimeError, match="exact filename"):
        validate_prepared_dataset(output)


def test_openwebtext_requires_full_commit_revision(tmp_path):
    cfg, documents = _fixture(tmp_path)
    with pytest.raises(ValueError, match="40-character"):
        prepare_dataset(
            replace(cfg, revision="main"),
            iterator_factory=lambda _split: iter(documents),
        )


def test_disk_preflight_rejects_before_stage_or_stream(tmp_path, monkeypatch):
    cfg, _documents = _fixture(tmp_path)
    monkeypatch.setattr(prepare_module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))

    with pytest.raises(OSError, match="insufficient free space"):
        prepare_dataset(
            cfg,
            iterator_factory=lambda _split: pytest.fail("disk rejection must happen before streaming"),
        )
    assert not Path(cfg.out_dir).exists()
    assert not (tmp_path / ".openwebtext-5b.staging-build-1").exists()


def test_reuse_attests_the_exact_tokenizer_bytes_used_for_encoding(tmp_path):
    cfg, documents = _fixture(tmp_path)
    prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))
    tokenizer_path = Path(cfg.tokenizer)
    tokenizer_path.write_text(tokenizer_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incompatible"):
        prepare_dataset(
            replace(cfg, reuse_existing=True),
            iterator_factory=lambda _split: pytest.fail("changed-tokenizer reuse must not stream"),
        )


def test_hf_stream_does_not_enable_remote_repository_code(tmp_path, monkeypatch):
    cfg, _documents = _fixture(tmp_path)
    captured = {}

    def fake_load_dataset(source, **kwargs):
        captured.update({"source": source, **kwargs})
        return [{"text": "v0 v20"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    assert list(prepare_module.hf_iterator_factory(cfg)("train")) == ["v0 v20"]
    assert captured["revision"] == cfg.revision
    assert "trust_remote_code" not in captured


def test_validation_cli_and_sweep_reject_corruption_before_run_state(tmp_path, monkeypatch, capsys):
    cfg, documents = _fixture(tmp_path)
    output = prepare_dataset(cfg, iterator_factory=lambda _split: iter(documents))

    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--validate-only", str(output)])
    prepare_module.main()
    assert "verified prepared dataset" in capsys.readouterr().out

    with (output / "val.bin").open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))

    from scripts import run_sweep

    result_root = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "run_sweep.py", "--data-dir", str(output), "--out", str(result_root), "--run-id", "reject-data",
    ])
    with pytest.raises(RuntimeError, match="integrity"):
        run_sweep.main()
    assert not result_root.exists()
