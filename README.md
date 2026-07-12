<div align="center">

# 🐍⚡ Mamba-Transformer Hybrid LM

**A small (~50M) hybrid language model that interleaves Mamba-2 selective-SSM blocks with causal attention — and a study of how the attention:SSM ratio trades off quality, speed, and memory.**

[![Live Demo](https://img.shields.io/badge/🔴_Live_Demo-online-e11d48?style=for-the-badge)](#-live-demo)
[![Model](https://img.shields.io/badge/params-~50M-2563eb?style=for-the-badge)]()
[![Precision](https://img.shields.io/badge/precision-bf16-7c3aed?style=for-the-badge)]()
[![GPU](https://img.shields.io/badge/trained_on-RTX_5070_12GB-16a34a?style=for-the-badge)]()

</div>

---

## Research question

> **At a fixed compute budget, how does the attention:SSM layer ratio affect perplexity, generation speed, and KV-cache memory for a small (~50M) hybrid LM?**

Mamba-2 state-space layers are **O(L)** in sequence length with a **fixed-size** recurrent state,
while attention is **O(L²)** with a **KV-cache that grows** with context. Interleaving them (the
[Jamba](https://arxiv.org/abs/2403.19887) pattern) should keep most of attention's quality while
paying far less at long context. This project trains three ratios — **1:3, 1:7, 1:15** (attention:SSM)
— at **matched tokens-seen** and measures the trade-off.

---

## Headline result

<!-- TODO(Week 4): replace with the real plot -->
> 📊 _Perplexity vs. throughput vs. KV-cache memory across the three ratios — plot lands in Week 4._

| Variant | Val PPL | Train tok/s | Infer tok/s | KV-cache @ 8K |
|:--:|:--:|:--:|:--:|:--:|
| 1:3  | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 1:7  | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 1:15 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

---

## 🔴 Live Demo

<!-- TODO(Week 5): embed the recorded GIF + hosted link -->
**▶ Try it live:** _link lands in Week 5_ · **📹 30-second demo:** _GIF lands in Week 5_

The demo is a **Next.js frontend** (hosted on Vercel, always on) talking to a **FastAPI backend**.
You type a prompt and watch tokens stream in with **live tokens/sec** and, when run locally on a GPU,
**live VRAM usage**. You can switch between the trained ratio variants.

```
Recruiter → GitHub → Vercel frontend ──POST /generate (SSE)──▶ FastAPI backend ──▶ ~50M model
                       (always on)                              (free CPU host)      streams tokens
```

Because the model is only ~50M parameters, it runs at usable speed **on a free CPU host** — so the
live link stays up without a paid GPU. Full-fidelity GPU speed + the VRAM meter are available via
**run-locally** (see `demo/README.md`).

---

## Architecture

```
tokens ─▶ Embedding (tied) ─▶ N × HybridBlock ─▶ RMSNorm ─▶ LM head

HybridBlock:
  x = x + Mixer(RMSNorm(x))        # Mixer ∈ { Mamba-2 , CausalAttention }, set per-layer by the ratio
  x = x + SwiGLU_MLP(RMSNorm(x))
```

- **Mamba-2 mixer** — selective SSM, SSD chunked scan · d_state 128 · expand 2 · headdim 64 · fixed recurrent state at inference.
- **Attention mixer** — causal multi-head (head_dim 64) · RoPE · Flash-Attn-2 if available, else PyTorch SDPA · KV-cache at inference.
- **d_model 768** · bf16 · gradient checkpointing · trained on **OpenWebText**.

The attention:SSM ratio is a single config knob — a per-layer type list like
`['mamba','mamba','mamba','attention']` (1:3) — so the sweep is a config change, not a code change.

---

## Quickstart

```bash
# 1. environment (Python 3.11 recommended for the SSM CUDA kernels)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. install PyTorch for your CUDA first (Blackwell / RTX 5070 → cu128), then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 3. train the 16k tokenizer, then tokenize TinyStories into uint16 memmap shards
python -m src.data.train_tokenizer --dataset tinystories --docs 100000 --vocab 16000
python -m src.data.prepare_data   --dataset tinystories --train-docs 50000 --val-docs 22000

# 4. sanity-check the parameter budget for a config
python scripts/count_params.py --config configs/ratio_1_3.yaml

# 5. train  (coming soon — Week 2/3)
python -m src.train.train --config configs/ratio_1_3.yaml
```

> ⚠️ On RTX 5070 (Blackwell, sm_120) the `mamba-ssm` CUDA kernels may need to be built from source.
> If the build fails, the model falls back to a pure-PyTorch Mamba path (slower, identical math).
> This is also what lets the demo run on CPU-only hosts.

---

## Repository layout

```
src/
  model/   HybridBlock, Mamba-2 mixer, attention mixer, the LM
  data/    tokenization, sharding, dataloader
  train/   training loop, optimizer, scheduler, checkpointing
  eval/    perplexity, throughput, KV-cache, needle-in-haystack
configs/   one config per ratio variant
scripts/   count_params.py, prepare_data.py, plotting, sweep runner
tests/     unit + smoke tests
demo/      Next.js frontend + FastAPI backend
```

---

## Roadmap

- [x] Week 1 — Foundation: environment, data pipeline, param budget, baseline
- [x] Week 2 — Hybrid block + ratio-configurable stack + training loop (converges on TinyStories, val ppl 11.4)
- [ ] Week 3 — Train the three ratios at matched compute
- [ ] Week 4 — Evaluation: PPL, KV-cache vs context, needle-in-haystack, plots
- [ ] Week 5 — Live demo (Next.js + FastAPI), hosted + local
- [ ] Week 6 — Write-up + workshop-style paper

---

## References

Attention Is All You Need (Vaswani et al., 2017) · Mamba (Gu & Dao, 2023) ·
Transformers are SSMs / Mamba-2 (Dao & Gu, 2024) · Jamba (Lieber et al., 2024) ·
Hymba (NVIDIA, 2024).

## License

MIT © Karan Anchan
