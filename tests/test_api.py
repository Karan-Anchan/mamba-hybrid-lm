"""FastAPI response, streaming, and availability contracts."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.generation import GenerationMetrics, GenerationResult, SamplingSettings, TokenEvent
from src.serve.app import AppSettings, create_app
from src.serve.registry import RuntimeSettings, parameter_count
from src.model.config import ModelConfig
from src.model.lm import HybridLM


class FakeRuntime:
    def health(self) -> dict:
        return {"status": "ready", "mode": "cpu", "loaded_ratio": "1:3"}

    def models(self) -> list[dict]:
        return [{"ratio": "1:3", "parameters": 52_530_000, "loaded": True}]

    def generate(self, prompt, ratio, sampling, on_token=None) -> GenerationResult:
        if ratio != "1:3":
            raise ValueError("ratio is not available")
        if on_token is not None:
            on_token(TokenEvent(0, 4, " works", " works", 0.01))
        metrics = GenerationMetrics(
            prompt_tokens=2, generated_tokens=1, prefill_seconds=0.01,
            decode_seconds=0.01, total_seconds=0.02, tokens_per_second=50.0,
            decode_tokens_per_second=100.0, time_to_first_token_seconds=0.01,
            logical_state_mib=5.0, peak_vram_mib=None, current_vram_mib=None,
            device="cpu", stop_reason="length",
        )
        return GenerationResult(
            prompt=prompt, completion=" works", text=prompt + " works", token_ids=(4,),
            ratio=ratio, checkpoint_sha256="abc", settings=sampling, metrics=metrics,
        )


def test_registry_parameter_count_matches_the_model():
    config = ModelConfig(
        name="tiny", ratio="1:3", vocab_size=32, d_model=32, n_layers=4,
        head_dim=16, mamba_headdim=16, d_state=8, mlp_multiple_of=16,
    )
    assert parameter_count(config.__dict__) == HybridLM(config).num_params()


def test_container_manifest_matches_the_certified_checkpoint():
    deployment = json.loads(Path("demo/checkpoint_manifest.json").read_text(encoding="utf-8"))
    certified = json.loads(Path(
        "checkpoints/week3-700m-v1/hybrid-1_3-76f9c3a9/manifest.json"
    ).read_text(encoding="utf-8"))
    assert deployment["signature"] == certified["signature"]
    assert deployment["artifact_sha256"]["best"] == certified["artifact_sha256"]["best"]


def test_health_models_and_json_generation(tmp_path):
    settings = AppSettings(
        runtime=RuntimeSettings(tmp_path, tmp_path / "unused", eager_load=False),
        cors_origins=("http://localhost:5173",),
    )
    with TestClient(create_app(FakeRuntime(), settings)) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/v1/models").json()["models"][0]["ratio"] == "1:3"
        response = client.post("/v1/generate", json={"prompt": "small models", "max_new_tokens": 1})

    assert response.status_code == 200
    assert response.json()["completion"] == " works"
    assert response.json()["metrics"]["tokens_per_second"] == 50.0


def test_stream_emits_token_then_complete(tmp_path):
    settings = AppSettings(
        runtime=RuntimeSettings(tmp_path, tmp_path / "unused", eager_load=False),
        cors_origins=(),
    )
    with TestClient(create_app(FakeRuntime(), settings)) as client:
        response = client.post(
            "/v1/generate/stream", json={"prompt": "small models", "max_new_tokens": 1}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert "event: complete" in response.text
    assert response.text.index("event: token") < response.text.index("event: complete")


def test_invalid_request_and_runtime_error_are_friendly(tmp_path):
    settings = AppSettings(
        runtime=RuntimeSettings(tmp_path, tmp_path / "unused", eager_load=False),
        cors_origins=(),
    )
    with TestClient(create_app(FakeRuntime(), settings)) as client:
        invalid = client.post("/v1/generate", json={"prompt": "x", "temperature": 3.0})
        unavailable = client.post("/v1/generate", json={"prompt": "x", "ratio": "1:7"})

    assert invalid.status_code == 422
    assert unavailable.status_code == 422
    assert unavailable.json()["detail"] == "ratio is not available"
