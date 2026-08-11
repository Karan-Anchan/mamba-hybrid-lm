# Week 3 authoritative sweep

Run ID: `week3-700m-v1`

Each ratio variant completed 42,725 optimizer steps at 16,384 tokens per step, for exactly 700,006,400 sampled token positions per model.

## Result

| Ratio | Parameters | Attention layers | Best validation perplexity | Average train tok/s | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| **1:3** | 52.53M | 4 | **26.30** | **26,808** | **6,684 MB** |
| 1:7 | 53.58M | 2 | 26.47 | 24,162 | 7,251 MB |
| 1:15 | 54.11M | 1 | 26.51 | 21,187 | 7,538 MB |

The 1:3 model achieved the lowest best validation loss in this single-seed sweep. The difference from 1:7 is 0.17 perplexity and the full spread is 0.21, so this is a descriptive ordering rather than evidence of statistical separation.

## Controlled variables

- Same prepared OpenWebText pool and tokenizer
- Same 512-token sequence length
- Same micro-batch 8 and gradient accumulation 4
- Same 42,725-step schedule
- Same optimizer, warmup, cosine decay, precision, and clipping settings
- Same fixed-window validation protocol

Parameter scale is near-matched rather than exact: 52.53M to 54.11M, a 3.02% spread. Token exposure is matched; FLOPs are not.

## Efficiency boundary

The current pure-PyTorch Mamba-2 SSD implementation materializes an `L x L` matrix across fourteen Mamba heads. The observed training throughput and memory ordering describes that implementation only. It must not be presented as evidence about a fused linear-scan Mamba kernel.

The 1:15 run was paused after a verified step-12,500 checkpoint because another host workload reduced GPU headroom. Resume restored the model, optimizer, counters, timing, metrics boundary, and RNG streams. Its final throughput includes that external contention.

## Files

- `sweep_results.json`: headline result rows and per-variant signatures
- `sweep_table.md`: compact Markdown rendering of the same rows
- `sweep_manifest.json`: run arguments, timestamps, data signature, sweep signature, and completion state

Large model checkpoints and the 5B-token prepared corpus remain local and are excluded from Git.

## Claims deferred to Week 4

- Inference tokens per second
- Attention KV-cache and recurrent Mamba state memory
- Context-length scaling through 8K
- Needle-in-haystack retrieval
- Generation quality
