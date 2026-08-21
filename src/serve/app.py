"""Single-worker FastAPI application for checkpoint-backed generation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import AsyncIterator, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.generation import GenerationResult, SamplingSettings, TokenEvent
from src.serve.registry import DemoRuntime, RATIOS, RuntimeSettings


ROOT = Path(__file__).resolve().parents[2]


class RuntimeProtocol(Protocol):
    def health(self) -> dict: ...
    def models(self) -> list[dict]: ...
    def generate(self, prompt: str, ratio: str, sampling: SamplingSettings, on_token=None) -> GenerationResult: ...


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    ratio: str = "1:3"
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=40, ge=1, le=100)
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)

    def sampling(self) -> SamplingSettings:
        return SamplingSettings(
            temperature=self.temperature,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )


@dataclass(frozen=True)
class AppSettings:
    runtime: RuntimeSettings
    cors_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "AppSettings":
        allowed = tuple(
            item.strip() for item in os.getenv("MAMBA_ALLOWED_RATIOS", ",".join(RATIOS)).split(",")
            if item.strip()
        )
        runtime = RuntimeSettings(
            checkpoint_root=Path(os.getenv("MAMBA_CHECKPOINT_ROOT", ROOT / "checkpoints")),
            tokenizer_path=Path(
                os.getenv("MAMBA_TOKENIZER_PATH", ROOT / "data/tokenizer/openwebtext.json")
            ),
            training_run_id=os.getenv("MAMBA_TRAINING_RUN_ID", "week3-700m-v1"),
            allowed_ratios=allowed,
            default_ratio=os.getenv("MAMBA_DEFAULT_RATIO", "1:3"),
            device=os.getenv("MAMBA_DEVICE", "auto"),
            eager_load=os.getenv("MAMBA_EAGER_LOAD", "true").lower() == "true",
        )
        origins = tuple(
            item.strip() for item in os.getenv(
                "MAMBA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",") if item.strip()
        )
        return cls(runtime=runtime, cors_origins=origins)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, allow_nan=False)}\n\n"


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=422, detail=str(error))
    if "out of memory" in str(error).lower():
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=500, detail="Generation failed. Check the server log for details.")


def create_app(
    runtime: RuntimeProtocol | None = None,
    settings: AppSettings | None = None,
) -> FastAPI:
    configured = settings or AppSettings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = runtime or await asyncio.to_thread(
            DemoRuntime, configured.runtime
        )
        application.state.generation_lock = asyncio.Lock()
        yield

    application = FastAPI(
        title="Mamba Hybrid LM API",
        version="0.5.0",
        description="Serialized recurrent generation from the certified Week 3 checkpoints.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.get("/health")
    async def health(request: Request) -> dict:
        return request.app.state.runtime.health()

    @application.get("/v1/models")
    async def models(request: Request) -> dict:
        return {"models": await asyncio.to_thread(request.app.state.runtime.models)}

    @application.post("/v1/generate")
    async def generate(payload: GenerationRequest, request: Request) -> dict:
        try:
            async with request.app.state.generation_lock:
                result = await asyncio.to_thread(
                    request.app.state.runtime.generate,
                    payload.prompt,
                    payload.ratio,
                    payload.sampling(),
                )
            return result.to_dict()
        except Exception as error:
            raise _http_error(error) from error

    @application.post("/v1/generate/stream")
    async def generate_stream(payload: GenerationRequest, request: Request) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_token(event: TokenEvent) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, ("token", asdict(event)))

            def run_generation() -> None:
                try:
                    result = request.app.state.runtime.generate(
                        payload.prompt, payload.ratio, payload.sampling(), on_token
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, ("complete", result.to_dict()))
                except Exception as error:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", error))

            async with request.app.state.generation_lock:
                worker = asyncio.create_task(asyncio.to_thread(run_generation))
                while True:
                    kind, value = await queue.get()
                    if kind == "token":
                        yield _sse("token", value)
                    elif kind == "complete":
                        yield _sse("complete", value)
                        break
                    else:
                        error = value if isinstance(value, Exception) else RuntimeError(str(value))
                        mapped = _http_error(error)
                        yield _sse("error", {"status": mapped.status_code, "detail": mapped.detail})
                        break
                await worker

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return application


app = create_app()
