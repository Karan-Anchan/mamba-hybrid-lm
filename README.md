<div align="center">

# Mamba-Transformer Hybrid LM

A 52-54M parameter language-model study that interleaves Mamba-2 selective state-space blocks with causal attention.

[![Week 3](https://img.shields.io/badge/milestone-Week_3_complete-55a36f?style=for-the-badge)](#week-3-result)
[![Tests](https://img.shields.io/badge/tests-33_passed-3f7f5f?style=for-the-badge)](#verification)
[![Precision](https://img.shields.io/badge/precision-bf16-596c74?style=for-the-badge)](#training-protocol)
[![GPU](https://img.shields.io/badge/GPU-RTX_5070_12GB-66756c?style=for-the-badge)](#training-protocol)

</div>

## Research question

> At a fixed token budget and near-matched parameter scale, how does the attention:SSM layer ratio affect validation perplexity and observed training behavior in a small hybrid language model?

I trained three 16-layer variants with attention:SSM ratios of **1:3**, **1:7**, and **1:15**. Every run used the same prepared OpenWebText pool, optimizer schedule, batch geometry, and **700,006,400 sampled token positions**.

The current Mamba-2 path uses the portable PyTorch SSD dual form. It is mathematically useful for the controlled architecture comparison, but it materializes an `L x L` matrix and does not reproduce the speed or memory profile of a fused linear scan.

## Week 3 result

The authoritative matched-token sweep completed on 2026-08-10. Lower perplexity is better.

| Ratio | Attention / Mamba layers | Parameters | Best val PPL | Train tok/s | Peak allocated VRAM |
|:--:|:--:|--:|--:|--:|--:|
| **1:3** | 4 / 12 | 52.53M | **26.30** | **26,808** | **6,684 MB** |
| 1:7 | 2 / 14 | 53.58M | 26.47 | 24,162 | 7,251 MB |
| 1:15 | 1 / 15 | 54.11M | 26.51 | 21,187 | 7,538 MB |

The quality gap is small and favors 1:3 in this single-seed sweep. The 1:15 run was paused at a verified checkpoint when another GPU workload reduced headroom, then resumed with model, optimizer, counters, metrics, and RNG state restored. Its throughput therefore includes external host contention.

The training-efficiency ordering is specific to this implementation. More Mamba layers are slower here because each Mamba block builds a quadratic matrix across fourteen heads. These results do not support a general claim that Mamba is slower or uses more memory than attention.

Authoritative artifacts:

- [`results/week3-700m-v1/sweep_results.json`](results/week3-700m-v1/sweep_results.json) - machine-readable headline metrics and variant signatures
- [`results/week3-700m-v1/sweep_table.md`](results/week3-700m-v1/sweep_table.md) - compact comparison table
- [`results/week3-700m-v1/sweep_manifest.json`](results/week3-700m-v1/sweep_manifest.json) - run settings, data signature, completion state, and exact token exposure
- [`results/week3-700m-v1/README.md`](results/week3-700m-v1/README.md) - interpretation and evidence boundary

The earlier 16.384M-token preview is preserved in [`results/sweep_results.json`](results/sweep_results.json) for the development record, but it is not the headline result.

## What Week 3 establishes

- All three ratio variants train to the same token exposure under one schedule.
- The 1:3 run achieved the lowest validation perplexity in this sweep.
- The best-perplexity spread is only 0.21, so one seed cannot establish statistical separation.
- Run recovery is durable and auditable across checkpoints, append-only metrics, data identity, code identity, and RNG state.
- Training throughput and peak allocated memory describe this PyTorch SSD implementation, not a fused linear-scan Mamba kernel.

It does **not** yet establish inference throughput, recurrent-state memory, KV-cache scaling, 8K behavior, needle retrieval, or generation quality. Those are Week 4 measurements.

## Architecture

```text
token IDs
   |
tied embedding
   |
16 x HybridBlock
   |-- RMSNorm -> Mamba-2 SSD or causal attention -> residual
   `-- RMSNorm -> SwiGLU MLP -> residual
   |
RMSNorm -> tied LM head
```

| Component | Locked setting |
|---|---|
| Model width | 448 |
| Layers | 16 |
| Vocabulary | 16,000 byte-level BPE tokens |
| Attention | 7 heads x 64, RoPE, PyTorch SDPA |
| Mamba-2 | inner width 896, 14 heads x 64, state size 128, convolution width 4 |
| MLP | SwiGLU, hidden width 1,216 |
| Context used for training | 512 tokens |
| RoPE table limit | 8,192 positions, not yet evaluated end to end |

The ratio is configuration, not a separate implementation. For example, 1:3 repeats:

```python
["mamba", "mamba", "mamba", "attention"]
```

## Data pipeline

The authoritative dataset is a locally prepared, read-only OpenWebText sampling pool:

| Split | Documents | Tokens |
|---|---:|---:|
| Train | 4,113,788 | 5,000,001,094 |
| Validation | 8,353 | 10,001,312 |

Preparation pins the dataset revision, trains the 16k tokenizer separately, writes little-endian `uint16` memmaps, records hashes and source ranges, verifies split continuity, and only then promotes the staging directory. Training revalidates the prepared-data identity before opening a sweep namespace.

The 700M figure is processed token positions sampled with replacement, not 700M unique corpus tokens.

## Training protocol

| Setting | Value |
|---|---:|
| Optimizer | fused AdamW on CUDA |
| Precision | fp32 master weights, bf16 autocast, TF32 matmul |
| Micro-batch | 8 sequences |
| Gradient accumulation | 4 |
| Sequence length | 512 |
| Effective tokens / step | 16,384 |
| Optimizer steps | 42,725 |
| Tokens / variant | 700,006,400 |
| Peak / minimum LR | 1e-3 / 1e-4 |
| Warmup | 855 steps, then cosine decay |
| Gradient clipping | 1.0 |
| Gradient checkpointing | disabled for the sweep |

Validation at each gate averages 50 deterministic batches per split. It samples 204,800 token positions, so it is repeatable but not a complete pass over the validation memmap.

## Quickstart

Python 3.11 is recommended. Install the CUDA-enabled PyTorch build before the rest of the dependencies.

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Prepare the small TinyStories debug corpus:

```powershell
.\.venv\Scripts\python.exe -m src.data.train_tokenizer --dataset tinystories --docs 100000 --vocab 16000
.\.venv\Scripts\python.exe -m src.data.prepare_data --dataset tinystories --train-docs 50000 --val-docs 22000
```

Inspect one ratio and train a debug run:

```powershell
.\.venv\Scripts\python.exe scripts\count_params.py --config configs\ratio_1_3.yaml
.\.venv\Scripts\python.exe -m src.train.train --model-config configs\ratio_1_3.yaml --run-id debug-1
```

## Reproducing the authoritative sweep

The full prepared corpus requires at least 9.33 GiB for the two token files before whole-document overshoot, plus space for staging and the Hugging Face cache.

```powershell
# Build the immutable 5B-token sampling pool.
.\.venv\Scripts\python.exe -m src.data.prepare_data `
  --dataset openwebtext `
  --tokenizer data/tokenizer/openwebtext.json `
  --out-dir data/openwebtext-5b `
  --run-id owt-5b-20260809 `
  --revision 79d93d786212f7344586290adb811d4ae6a1762c `
  --train-tokens 5000000000 `
  --val-tokens 10000000

# Verify the exact prepared identity without downloading or tokenizing again.
.\.venv\Scripts\python.exe -m src.data.prepare_data --validate-only data/openwebtext-5b

# Run or resume the three variants.
.\.venv\Scripts\python.exe scripts\run_sweep.py `
  --data-dir data/openwebtext-5b `
  --run-id week3-700m-v1 `
  --max-steps 42725 `
  --warmup-steps 855
```

Reusing a run ID is accepted only when the numerical configuration, data identity, runtime-code fingerprint, and provenance agree. Completed variants are skipped only after their manifests, counters, metrics trajectory, result record, and `best.pt` / `last.pt` checks pass.

## Repository layout

```text
configs/   ratio-specific model configurations
data/      committed tokenizers; prepared corpora stay local
results/   committed aggregate experiment results
scripts/   parameter analysis and matched-token sweep runner
src/
  data/    tokenizer training, immutable preparation, memmap batching
  model/   attention, Mamba-2 SSD, hybrid blocks, language model
  train/   optimization, checkpointing, recovery, certification
  eval/    Week 4 evaluation modules
tests/     model, data, preparation, and training-reliability tests
demo/      Week 5 generation application
```

## Verification

Current repository test result:

```text
33 passed, 1 skipped
```

The skipped test requires CUDA in the test process. The authoritative runs themselves were executed on an RTX 5070 12 GB with CUDA available.

## Roadmap

- [x] Week 1 - environment, tokenizer, data pipeline, parameter budget
- [x] Week 2 - hybrid model, configurable ratios, training loop, TinyStories convergence
- [x] Week 3 - immutable 5B-token pool, recovery hardening, three-model 700M-token sweep
- [ ] Week 4 - inference throughput, state / KV memory, 8K evaluation, needle retrieval, plots
- [ ] Week 5 - Next.js and FastAPI generation demo, local GPU metrics, hosted CPU path
- [ ] Week 6 - final documentation and workshop-style report

## References

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Gu and Dao (2023), [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- Dao and Gu (2024), [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- Lieber et al. (2024), [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)
- Dong et al. (2024), [Hymba: A Hybrid-head Architecture for Small Language Models](https://arxiv.org/abs/2411.13676)

## License

Released under the MIT License. Copyright 2026 Karan Anchan.
