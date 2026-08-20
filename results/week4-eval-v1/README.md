# Week 4 inference and long-context evaluation

This directory contains the certified `week4-eval-v1` evaluation of the three Week 3 checkpoints.
The run rechecks fixed-window validation quality, separates prefill from token-by-token decode speed, measures
logical recurrent/KV state alongside CUDA allocation deltas, and runs controlled exact-match retrieval.

## Protocol

- Checkpoints: best-validation artifacts from `week3-700m-v1` after manifest SHA-256 validation
- Context lengths: 512, 1,024, 2,048, 4,096, 8,192 tokens
- Decode: 32 timed greedy steps after each prefill
- Throughput repeats: 3 after warmup; medians are reported
- Validation: 50 fixed batches × 8 × 512 tokens
- Retrieval depths: 10%, 50%, 90%
- Precision: bf16 CUDA compute with fp32 recurrent SSM state

## Files

- `evaluation_results.json`: complete machine-readable measurements and trial records
- `evaluation_table.md`: compact 8K comparison derived from the JSON
- `manifest.json`: protocol, source/checkpoint provenance, and generated-artifact hashes
- `plots/validation_perplexity.svg`: quality comparison
- `plots/decode_throughput.svg`: decode speed across prompt lengths
- `plots/logical_state_memory.svg`: recurrent plus KV state scaling
- `plots/needle_retrieval.svg`: exact-match retrieval by context

## Interpretation boundary

These measurements describe this repository's plain-PyTorch recurrent/chunked implementation on the recorded
RTX 5070 runtime. Logical state bytes are exact tensor sizes. Allocated and peak deltas include framework and
kernel workspaces and therefore answer a different question. Retrieval uses one pre-registered code at each
context/depth pair; it is a controlled diagnostic, not a broad language-understanding benchmark.

See `evaluation_table.md` for the headline comparison and `evaluation_results.json` for every raw timing and
generated answer.
