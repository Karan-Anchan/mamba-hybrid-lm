"""Prepare the verified public checkpoint, then start one API worker."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.getenv("MAMBA_CHECKPOINT_ROOT", "/models"))
RUN_DIR = MODEL_ROOT / "week3-700m-v1" / "hybrid-1_3"
MODEL_PATH = RUN_DIR / "best.pt"
EXPECTED_SHA256 = "0995d848d8538a0169151e94388660554dcf90ce93c9a5bb2fa18ae1aec0504c"
DEFAULT_URL = (
    "https://github.com/Karan-Anchan/mamba-hybrid-lm/releases/download/"
    "v0.5/hybrid-1_3-best.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_checkpoint() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = RUN_DIR / "manifest.json"
    if not manifest.is_file():
        shutil.copyfile(ROOT / "demo/checkpoint_manifest.json", manifest)
    if MODEL_PATH.is_file() and sha256(MODEL_PATH) == EXPECTED_SHA256:
        return

    url = os.getenv("MAMBA_CHECKPOINT_URL", DEFAULT_URL)
    temporary = MODEL_PATH.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    print("Downloading the verified 1:3 checkpoint...", flush=True)
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = sha256(temporary)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"checkpoint checksum mismatch: expected {EXPECTED_SHA256}, received {actual}"
            )
        os.replace(temporary, MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    prepare_checkpoint()
    environment = os.environ.copy()
    environment.setdefault("MAMBA_CHECKPOINT_ROOT", str(MODEL_ROOT))
    environment.setdefault("MAMBA_TOKENIZER_PATH", str(ROOT / "data/tokenizer/openwebtext.json"))
    environment.setdefault("MAMBA_ALLOWED_RATIOS", "1:3")
    environment.setdefault("MAMBA_DEFAULT_RATIO", "1:3")
    environment.setdefault("MAMBA_DEVICE", "cpu")
    port = environment.get("PORT", "7860")
    command = [
        sys.executable, "-m", "uvicorn", "src.serve.app:app", "--host", "0.0.0.0",
        "--port", port, "--workers", "1",
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
