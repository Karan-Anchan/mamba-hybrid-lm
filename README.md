<div align="center">

# Mamba-Transformer Hybrid LM

**A small (~50M) language model that interleaves Mamba-2 selective-SSM blocks with causal attention. The experiments compare quality, speed, and memory at different attention:SSM ratios.**

[![Live Demo](https://img.shields.io/badge/live_demo-online-e11d48?style=for-the-badge)](#live-demo)
[![Model](https://img.shields.io/badge/params-~50M-2563eb?style=for-the-badge)]()
[![Precision](https://img.shields.io/badge/precision-bf16-7c3aed?style=for-the-badge)]()
[![GPU](https://img.shields.io/badge/trained_on-RTX_5070_12GB-16a34a?style=for-the-badge)]()

</div>

---

## Research question

> **At a fixed compute budget, how does the attention:SSM layer ratio affect perplexity, generation speed, and KV-cache memory for a small (~50M) hybrid LM?**

Mamba-2 state-space layers are **O(L)** in sequence length and use a **fixed-size** recurrent state.
Attention is **O(L²)** and its **KV-cache grows** with context. The project uses the interleaving
pattern from [Jamba](https://arxiv.org/abs/2403.19887) and trains three attention:SSM ratios:
**1:3, 1:7, and 1:15**. Each variant sees the same number of tokens.

---

## Headline result

<!-- TODO(Week 4): full-scale runs + KV-cache/infer columns + the plot -->
> _The table below is a reduced-scale preview: 16.4M OpenWebText tokens per variant at matched
> compute. Full-scale runs, inference measurements, and KV-cache measurements are still pending._

| Variant | Attn layers | Val PPL | Train tok/s | Peak VRAM | Infer tok/s | KV-cache @ 8K |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1:3  | 4 | 105.4 | 25,285 | 6.7 GB | _tbd_ | _tbd_ |
| 1:7  | 2 | **102.4** | 22,747 | 7.3 GB | _tbd_ | _tbd_ |
| 1:15 | 1 | 106.9 | 21,755 | 7.5 GB | _tbd_ | _tbd_ |

> Note: training throughput and VRAM rise with more Mamba layers because the SSD scan currently uses
> the O(L²) dual form. The expected SSM memory advantage is an **inference KV-cache** property and
> has not been measured yet.

---

## Live demo

<!-- TODO(Week 5): embed the recorded GIF + hosted link -->
_The hosted link and recorded demo are still pending._

The demo is a Next.js frontend connected to a FastAPI backend. Enter a prompt to stream tokens
and see live throughput. A local GPU run also reports VRAM use, and the interface can switch
between trained ratio variants.

```
Recruiter → GitHub → Vercel frontend ──POST /generate (SSE)──▶ FastAPI backend ──▶ ~50M model
                       (always on)                              (free CPU host)      streams tokens
```

At roughly 50M parameters, the model can run on a free CPU host. Local execution is needed for
GPU throughput and the VRAM meter; see `demo/README.md`.

---

## Architecture

```
tokens ─▶ Embedding (tied) ─▶ N × HybridBlock ─▶ RMSNorm ─▶ LM head

HybridBlock:
  x = x + Mixer(RMSNorm(x))        # Mixer ∈ { Mamba-2 , CausalAttention }, set per-layer by the ratio
  x = x + SwiGLU_MLP(RMSNorm(x))
```

| | |
| :--- | :--- |
| Mamba-2 mixer | pure-PyTorch SSD dual form (masked-attention formulation, O(L²)) · d_state 128 · expand 2 · headdim 64 |
| Attention mixer | causal multi-head (head_dim 64) · RoPE · PyTorch scaled-dot-product attention (SDPA) |
| Shared | d_model 448 · 16 layers · bf16 autocast · trained on OpenWebText |

The attention:SSM ratio is one configuration value. A 1:3 variant, for example, uses the
per-layer list `['mamba','mamba','mamba','attention']`, so the sweep does not require separate
model implementations.

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

# 5. train one variant; reuse the run ID to resume or skip a completed run
python -m src.train.train --model-config configs/ratio_1_3.yaml --run-id debug-1
```

The sweep runner keeps summaries in `results/<run-id>/` and each variant's manifest, metrics,
resumable `last.pt`, and evaluation `best.pt` in `checkpoints/<run-id>/<variant>/`. The default
8,000-step recipe is a **reduced 131.072M-token run per variant**, not the full Week-3 target:

```bash
# first build the immutable authoritative sampling pool (not run as part of repository verification)
python -m src.data.prepare_data --dataset openwebtext \
  --tokenizer data/tokenizer/openwebtext.json --out-dir data/openwebtext-5b \
  --run-id owt-5b-20260809 --revision 79d93d786212f7344586290adb811d4ae6a1762c \
  --train-tokens 5000000000 --val-tokens 10000000

# standalone integrity check; the sweep performs this same check before creating run artifacts
python -m src.data.prepare_data --validate-only data/openwebtext-5b

# reduced readiness run: 8,000 × (8 batch × 4 accumulation × 512 tokens) = 131,072,000 tokens/variant
python scripts/run_sweep.py --data-dir data/openwebtext-5b --run-id week3-131m-v1 --max-steps 8000

# authoritative ~700M-token run: 42,725 steps = 700,006,400 tokens/variant
# 855 warmup steps = ceil(2% × 42,725), as specified in the training plan
python scripts/run_sweep.py --data-dir data/openwebtext-5b \
  --run-id week3-700m-v1 --max-steps 42725 --warmup-steps 855
```

Do not start the authoritative command from the current 72.7M-token preview memmap. Week 3 still
requires the planned approximately 5B-token OpenWebText sampling pool to be prepared and verified first.
The two uint16 token files require at least **10,020,000,000 bytes (9.33 GiB)** before whole-document
overshoot; staging is renamed into place rather than copied. Reserve at least 12 GiB for the prepared
artifacts, plus separate space for the Hugging Face cache. If streaming stops, the run-specific stage is
normally preserved and the final directory remains absent; the interrupted-manifest update is best effort
if the filesystem itself is failing. Choose a new run ID or inspect/remove the stage manually. The builder
preflights current free space against the exact token-file lower bound, but whole-document overshoot, the
download cache, and space already consumed by stale stages still require the documented reserve.
An existing final directory is never overwritten. `--reuse-existing` only accepts the exact same verified
source revision, targets, tokenizer, tool/runtime provenance, counts, and checksums. Authoritative
OpenWebText builds require a full 40-hex commit revision and do not execute dataset repository code.
Artifact names are fixed basenames, links/reparse points are rejected, and all reads use little-endian
uint16 explicitly. Every sweep revalidates the manifest identity and artifact hashes before training starts.

If training stops, run the same command with the same run ID. Resume is accepted only when the
training configuration, SHA-256 dataset identity, runtime-code fingerprint, and git provenance still
match. A completed variant is skipped only after its manifest, exact step/token counts, finite result
fields, artifact checksums, full metrics trajectory, and both `best.pt` and `last.pt` validate. Checkpoint
loading uses PyTorch's restricted `weights_only=True` path, and a per-variant lock rejects concurrent
writers. Earlier runs are never overwritten.

> The current implementation deliberately uses the pure-PyTorch Mamba-2 SSD dual form; it does not
> dynamically load or fall back from `mamba-ssm` CUDA kernels. This is portable to CPU, but its O(L²)
> training behavior does not provide the speed or memory profile of a fused linear-scan kernel.

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
- [x] Week 3 — Matched-compute sweep infra + OWT pipeline + reduced-scale preview (full-scale runs pending)
- [ ] Week 4 — Evaluation: PPL, KV-cache vs context, needle-in-haystack, plots
- [ ] Week 5 — Live demo (Next.js + FastAPI), hosted + local
- [ ] Week 6 — Write-up + workshop-style paper

---

## References

Attention Is All You Need (Vaswani et al., 2017) · Mamba (Gu & Dao, 2023) ·
Transformers are SSMs / Mamba-2 (Dao & Gu, 2024) · Jamba (Lieber et al., 2024) ·
Hymba (NVIDIA, 2024).

## License

Released under the MIT License. © Karan Anchan
