"""Read the token .bin files back and hand out training batches.

Nothing fancy here: I memmap the uint16 stream so the OS pages it in lazily (the file can be way
bigger than RAM), then grab random windows for each batch. This is the nanoGPT get_batch pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_split(data_dir: str | Path, split: str) -> np.ndarray:
    # read-only memmap; 'r' mode means I never accidentally scribble over the data
    path = Path(data_dir) / f"{split}.bin"
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text())


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str = "cuda",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One (x, y) batch. y is x shifted by one — standard next-token setup.

    I pick batch_size random start points, cut block_size+1 tokens from each, and split into
    input/target. Casting to int64 because that's what nn.Embedding / cross_entropy expect.
    """
    # high is exclusive and I need room for the +1 target, so stop at len - block_size - 1
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):
        # pin + non_blocking lets the copy overlap with compute; cheap win
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
