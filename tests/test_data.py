"""Batch-sampler checks on a tiny fake .bin, so I don't need the real dataset to run these."""

import numpy as np
import torch

from src.data.dataset import get_batch


def _fake_data(n=1000):
    # 0..n-1 as uint16; the ramp makes the "y is x shifted by 1" check trivial to verify
    return np.arange(n, dtype=np.uint16)


def test_batch_shapes_and_dtype():
    x, y = get_batch(_fake_data(), block_size=64, batch_size=8, device="cpu")
    assert x.shape == (8, 64) and y.shape == (8, 64)
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one():
    # on the 0..n-1 ramp, every target should be its input + 1
    x, y = get_batch(_fake_data(), block_size=32, batch_size=4, device="cpu")
    assert torch.equal(y, x + 1)


def test_reproducible_with_generator():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    x1, _ = get_batch(_fake_data(), 16, 4, device="cpu", generator=g1)
    x2, _ = get_batch(_fake_data(), 16, 4, device="cpu", generator=g2)
    assert torch.equal(x1, x2)
