"""Model checks: shapes, causality, and (the important one) that the real param count matches the
analytical counter I used to pick the config back in D-ARCH-01. If those two ever disagree, one of
them is wrong — and I made design decisions off the analytical one.
"""

import torch

from scripts.count_params import count
from src.model.config import ModelConfig
from src.model.lm import HybridLM


def _tiny(ratio="1:3"):
    # small enough to build fast on CPU, still exercises both mixer types
    return ModelConfig(ratio=ratio, d_model=128, n_layers=4, vocab_size=256,
                       head_dim=32, mamba_headdim=32, d_state=32)


def test_forward_shapes_and_loss():
    cfg = _tiny()
    model = HybridLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_is_causal():
    # changing the last input token must not move any earlier position's logits,
    # for both mixers (conv is left-padded, attention is masked, ssd is lower-triangular)
    torch.manual_seed(0)
    cfg = _tiny("1:3")
    model = HybridLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        a, _ = model(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size
        b, _ = model(idx2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_param_count_matches_analytical():
    # this is the validation of scripts/count_params.py against a real nn.Module
    for ratio in ("1:3", "1:7", "1:15"):
        cfg = ModelConfig(ratio=ratio, d_model=448, n_layers=16, vocab_size=16000)
        model = HybridLM(cfg)
        analytical = count(cfg)
        assert model.num_params() == analytical["total"], ratio
        assert model.num_params(non_embedding=True) == analytical["non_embed"], ratio
