# Generation API

The Week 5 backend exposes the certified checkpoints through one serialized FastAPI process. It uses the same
recurrent prefill/decode path as the Week 4 evaluator and loads only one ratio at a time.

## Local CUDA server

```powershell
$env:MAMBA_DEVICE = "cuda"
$env:MAMBA_CORS_ORIGINS = "http://localhost:5173"
.venv\Scripts\python.exe -m uvicorn src.serve.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Use `MAMBA_DEVICE=cpu` for a CPU process. Keep `--workers 1`: every worker owns a separate checkpoint, and
multiple GPU copies would waste VRAM. Requests are also serialized behind one process-wide lock.

## Endpoints

- `GET /health` reports the execution mode, device, loaded ratio, and available ratios.
- `GET /v1/models` reports checkpoint identity and model geometry.
- `POST /v1/generate` returns one complete JSON result.
- `POST /v1/generate/stream` emits `token`, `complete`, or `error` Server-Sent Events.

Example payload:

```json
{
  "prompt": "A practical reason to compare attention with state-space layers is",
  "ratio": "1:3",
  "temperature": 0.8,
  "top_k": 40,
  "max_new_tokens": 80,
  "seed": 1337
}
```

Prompts are limited to 1,024 tokenizer tokens and completions to 512 tokens. CUDA responses include current and
peak allocated VRAM. CPU responses return `null` for both fields instead of inventing a VRAM measurement.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `MAMBA_CHECKPOINT_ROOT` | `checkpoints` | Root containing the certified training run |
| `MAMBA_TOKENIZER_PATH` | `data/tokenizer/openwebtext.json` | 16K BPE tokenizer |
| `MAMBA_TRAINING_RUN_ID` | `week3-700m-v1` | Checkpoint namespace |
| `MAMBA_ALLOWED_RATIOS` | `1:3,1:7,1:15` | Ratios visible to the service |
| `MAMBA_DEFAULT_RATIO` | `1:3` | Checkpoint loaded at startup |
| `MAMBA_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `MAMBA_CORS_ORIGINS` | localhost Vite origins | Comma-separated frontend origins |

The 200+ MB checkpoint files are intentionally excluded from Git. A deployment must mount or download verified
checkpoint files and their manifests before application startup. The registry checks every advertised
`best.pt` SHA-256 before accepting requests.

## Container deployment

`demo/Dockerfile` builds a CPU image around the same API. At startup, `demo/start.py` downloads the public 1:3
release checkpoint to temporary model storage, verifies its fixed SHA-256, and only then starts Uvicorn. The
container advertises one ratio to keep startup download and RAM use predictable.

```powershell
docker build -f demo/Dockerfile -t mamba-hybrid-api .
docker run --rm -p 7860:7860 -e MAMBA_CORS_ORIGINS=https://your-site.example mamba-hybrid-api
```

The default checkpoint URL targets the `v0.5` GitHub release. `MAMBA_CHECKPOINT_URL` can override it, but the
download must still match the certified hash. This image is ready for a Docker-capable CPU host; publication to
a third-party compute account is intentionally separate from the public static showcase.
