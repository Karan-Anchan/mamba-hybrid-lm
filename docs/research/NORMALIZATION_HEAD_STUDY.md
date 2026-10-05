# Normalization and final-score arithmetic

**Status: completed with parity failures; no accepted precision change.** The CUDA diagnostic executed all six declared checkpoint/window cases and all four cells. The pre-run declaration is `normalization-head-protocol-2026-10-05.json`; `normalization-head-summary-2026-10-05.json` independently validates the complete raw report. Production defaults and trained weights remain unchanged. The initial implementation outline is retained in the parent PR2 workspace at `planning/NORMALIZATION_NEXT_PROTOCOL_OUTLINE_2026-10-05.md`.

## What the completed comparison shows

More precise normalization coefficients improve agreement with the original model's scores. They do not repair agreement between each changed model's full-reading and one-token routes. Every cell passes its own full-versus-one-token score check in only four of six cases. No cell passes every case's complete gate.

| Across six checkpoint/window combinations | N0H0: neither | N1H0: normalization | N0H1: head | N1H1: both |
|---|---:|---:|---:|---:|
| One-token scores versus original full scores | 3/6 | 6/6 | 4/6 | 6/6 |
| One-token scores versus own full scores | 4/6 | 4/6 | 4/6 | 4/6 |
| Complete whole-route gate, including continuation and memory | 3/6 | 4/6 | 4/6 | 4/6 |
| Whole route plus all recorded intermediate comparisons | 2/6 | 3/6 | 3/6 | 3/6 |

These counts are numerical cases, not independent training seeds or language-quality results. “Original” means the unchanged historical FP32 full-reading calculation. “Own” means the full-reading calculation of the same cell. A cell can move closer to the original while its own two reading routes still disagree.

The normalization treatment removes full-shape versus one-token drift on identical frozen inputs at all 279 recorded normalization sites. The first observed own-route difference then moves from `norm1_output` to the first Mamba `in_projection.output` in all six cases. This narrows the next investigation to subsequent arithmetic; it does not prove one projection caused every later failure.

The local normalization output's maximum error against the independent FP64 reference improves at 251 sites, ties at 22 and worsens at 6. All those comparisons pass the fixed allowance. Its mean-square and coefficient maximum errors improve at all 279 sites. More precise coefficients therefore do not make every final normalization output more accurate. The FP64 head's frozen full-shaped and token-shaped outputs are exactly equal in all six cases, and its maximum error against the independent head reference improves in all six. Propagated whole-model failures remain.

All 4,791 frozen-local comparisons pass, as do all 336 retained-memory bundles covering 10,752 fields. A memory pass does not erase a failed score or intermediate check. All six exact original repeats and earlier N0H0 control comparisons remain stable; the final identity audit passes.

## Failures that the improved original agreement can hide

The 1:15 length-129 case still fails its own route comparison. N1 increases the violating score count and maximum error/allowance ratio:

| Cell | Violating scores | Maximum error / allowance |
|---|---:|---:|
| N0H0 | 7 | 1.117571 |
| N1H0 | 52 | 1.489642 |
| N0H1 | 7 | 1.104264 |
| N1H1 | 53 | 1.484899 |

For 1:15, N1 passes original-score agreement on all three windows, but its own full-versus-one-token agreement passes only the length-128 window. Batch two at length 512 remains a failed own-route case. Head-only passes the 1:15 length-128 original-score comparison; it leaves the length-129 and batch-two own-route failures unresolved.

N1 also introduces real-prefix continuation failures in the 1:15 batch-two, length-512 case. After its genuine 128-token prefix, the chunk-128 suffix route fails these score references:

| Cell | Suffix reference | Violating scores | Maximum error / allowance |
|---|---|---:|---:|
| N1H0 | Original full scores | 1 | 1.059812 |
| N1H0 | Decay-only full scores | 2 | 1.169722 |
| N1H1 | Decay-only full scores | 1 | 1.098821 |

The first violating suffix position is 79, which is global text position **207** after adding the 128-token prefix. Positions are zero-based and apply to the recorded batch row. The website shows each route/reference failure and coordinate beside all four cells. Genuine prefix and final retained memory checks remain separate from these score failures.

All 1:3 final-score and retained-memory checks pass in all four cells. Its batch-two, length-512 case still fails intermediate comparisons against the original route at final normalization and the head input. This negative remains visible even when final scores pass. The normalization/head factorial has no accepted winner.

## Why test these two operations?

The preceding one-token isolation study first observes a numerical difference at layer one's RMS normalization in all six recorded cases. RMS normalization rescales the features describing one token. Finding the first difference tells us where to look; it does not prove that operation caused the later prediction failure.

The preceding study retains two 1:15 own-route final-score failures, a third case that passes its own comparison but fails the original reference, and the 1:3 batch-two intermediate failure. These earlier findings are preserved in `tokenwise-isolation-summary-2026-10-04.json` and motivate this factorial.

The final score projection, named `lm_head`, converts each token's final features into scores for possible next tokens. It can amplify or partly cancel differences produced earlier. This study tests normalization arithmetic and final-score arithmetic separately and together while keeping the decay-coefficient treatment fixed.

## Four cells, one controlled question

A factorial comparison tests two changes in four combinations. All four cells use the same trained weights, text and fixed decay-coefficient treatment, `fp64_cumsum_decay_coefficients`.

| Cell | RMS normalization | Final score projection |
|---|---|---|
| N0H0: neither | Original FP32 formula | Original FP32 multiplication |
| N1H0: normalization only | FP64 reduction coefficient | Original FP32 multiplication |
| N0H1: head only | Original FP32 formula | Temporary FP64 multiplication |
| N1H1: both | FP64 reduction coefficient | Temporary FP64 multiplication |

N1 preserves the original conversion to FP32 and **FP32 squaring**. It converts those squared values to FP64 for the feature average, epsilon addition and reciprocal square root. It casts the resulting coefficient back to FP32 before the original input and weight multiplications, and returns the original input format. It does not calculate every normalization step in FP64.

H1 temporarily converts the same final features and projection weights to FP64 for the final matrix multiplication, then casts its scores to FP32. The input embedding and output projection share a learned parameter. The diagnostic must preserve that parameter's identity and stored values; it must not convert the saved model or every linear layer. Temporary overrides must restore the original methods after success or an exception.

The independent normalization reference is a separate calculation. It squares FP64 inputs, averages over features, adds epsilon and rescales with FP64 arithmetic. Comparing N1 with that reference answers a different question from checking N1's full-reading and one-token agreement.

## Fixed checkpoints and real text

The initial factorial covers the trained 1:15 and 1:3 checkpoints, each on three cases: six checkpoint/window combinations and four cells per combination. Seed 2027 and chunk size 128 remain fixed.

| Case | Batch × text length | Genuine prefix / remaining text | Validation-window starts |
|---|---:|---:|---|
| `b1-l128-p64` | 1 × 128 | 64 / 64 | 2,297,865 |
| `b1-l129-p64` | 1 × 129 | 64 / 65 | 2,298,209 |
| `b2-l512-p128` | 2 × 512 | 128 / 384 | 2,429,961; 3,825,427 |

Use the same saved token and shifted-target fingerprints as the preceding declaration. Bind checkpoint, training-manifest, tokenizer and prepared-data fingerprints before model allocation. These windows come from the validation pool used to select the checkpoint. They support a numerical diagnostic; they are not an independent language-quality test.

## Freeze one operation before changing the whole route

A frozen replay captures an operation's actual inputs, then feeds those identical numbers to several calculations. Earlier layers can no longer introduce different inputs into that local comparison.

Capture every normalization input from the decay-only candidate's full-reading reference route. Keep the learned weights, epsilon, number format, shape, stride and storage offset. The capture includes every block's `norm1` and `norm2`, every Mamba mixer's gated normalization input, and final `norm_f`: 48 sites for 1:15 and 45 for 1:3. The gated input is recorded at the module boundary; recalculating the gate would introduce another rounding change.

Replay each captured input both as the full batch of tokens and as one token at a time. Reduce only the final feature dimension; different batch rows and text positions stay separate. Compare the original and coefficient formulas with their observed outputs and an independently written FP64 reference. Record input, statistic and coefficient hashes plus bounded coordinates and error samples. Do not serialize the full activation arrays.

Freeze final-head inputs separately for the same reason. Compare the H0 and H1 full-shaped and token-shaped calculations on identical hidden features with an independent CPU NumPy FP64 matrix multiplication and optional bias addition. The reference keeps FP64 scores for comparison. A head replay on identical inputs isolates the multiplication's arithmetic. A changed whole-model head input includes effects propagated from normalization and earlier layers. Those observations must remain separate.

## Keep the references and gates separate

The original stateless full FP32 scores remain the historical prediction reference. N0H0 full reading is a second, common reference: the decay-only factorial baseline. Each cell also has its own full-reading reference for checking one-token consistency. A treatment can agree with itself and still disagree with either common reference.

Retained memory uses stateful references because a stateless prediction run does not produce inference memory. Compare each cell's tokenwise end memory with its own stateful full run and with the original stateful full run. Check every convolution, state-scan and attention-cache field, including shape, finiteness and final text position.

For continuation, each cell reads its own real prefix and clones that memory. It then reads the suffix in one shot, chunks of 128 and one token at a time. Never reuse another cell's memory. Verify the original prefix memory remains byte-identical after every continuation probe.

Keep the acceptance rule fixed: FP32 absolute tolerance 0.00003 plus relative tolerance 0.0003 times the absolute reference value. A difference divided by its allowance above 1 fails. Record the first nonzero difference separately from the first tolerance violation. Exact repeated controls use exact equality, not this tolerance.

The viewer retains final scores, cached-route checks, memory checks, intermediate comparisons and frozen-local checks separately. A passing final score cannot erase a failed intermediate. A local replay pass cannot certify the whole route. Duplicate views of the same final-normalization/head-input values do not count as independent observations.

## Execution limits and honest partial results

This first factorial is forward-only. It does not execute gradients, an optimizer, BF16, new training or a fused CUDA Mamba kernel. A single CUDA process receives a cooperative 900-second allowance starting at runner entry, including preflight and final integrity checks. Checks between routes, tokens and frozen modules request stopping when the allowance is exhausted; an operation already running may overshoot.

Require at least 8 GiB free after model loading before the batch-two, length-512 case. This is a headroom guard, not a measured peak-memory promise. Record runtime, peak allocation and their scopes. A timeout, out-of-memory error, nonfinite value or headroom failure preserves completed cells and the identified incomplete cell. A missing or unexecuted cell supplies no passes. Do not silently substitute smaller inputs under the same declaration.

Both TF32 settings remain disabled and are restored afterward. Keep the actual deterministic, cuDNN and cuBLAS settings of the preceding isolation; changing them would add another experimental factor. Bind source and evidence bytes before execution and recheck historical evidence afterward. Save a new report atomically without replacing existing evidence.

## What would still be needed afterward?

The result narrows the next experiment without approving a production precision policy or showing that attention improves language quality. N1 changes a family of normalization sites, so an improvement cannot identify which single site caused it. Agreement with the original FP32 calculation is also different from closeness to an independent FP64 reference: the original reference rounds too.

Before expanding the case grid or running a training pilot, isolate projection arithmetic on identical actual operands and investigate the newly failed real-prefix chunk routes. Keep original, own and decay-only references separate and retain the exact negative windows. A successful local replay still needs whole-route checks.

Any candidate must next cover all three ratios, lengths 127, 128, 129 and 257, and batch two at length 512. It needs full-reference and chunk-128 scores and loss, tokenwise and genuine-prefix routes, retained memory, complete parameter gradients and future-token isolation. Start larger gradient probes with the exact 257-token negative window and batch-two, length-512 case; declare additional coverage separately.

The earlier 1:3 batch-two intermediate failure remains visible even if final scores pass. Numerical acceptance and deterministic checkpoint recovery require separate declared policies before an operational training pilot. None of these diagnostics supplies a multi-seed architecture ranking or an independent quality conclusion.

## Recorded resources and evidence transport

The shared workflow took **633.453 seconds**, including preflight, model routes, traces, frozen replay, independent CPU references and final integrity audits. Its cooperative allowance was 900 seconds, with zero recorded overshoot. These diagnostic timings and allocations are not throughput benchmarks. TF32 was disabled in both CUDA arithmetic settings and restored afterward. The recorded inference policy retained nondeterministic algorithms, cuDNN determinism disabled, cuDNN benchmarking disabled and no cuBLAS workspace environment value. It does not apply the separate strict recovery study's policy retrospectively.

The exact raw JSON is 152,444,374 bytes, larger than [GitHub's documented 100 MiB file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Its original local bytes and declaration hash remain unchanged. The published `checks/normalization-head-2026-10-05/natural-matched-seed-2027.json.gz` is a deterministic, lossless gzip transport of that same report: 5,975,145 bytes. `normalization-head-raw-publication-2026-10-05.json` records its compressed and decompressed identities and verifies an exact round trip.

Raw SHA-256: `1cd4a0cbbf414f65a3712d2aaec07f6a35098a318bae5699f5e60d3302c4a0ee`.

Gzip SHA-256: `36d7efa1fad3497f1d7ba3b0b9fb61853408df4d6781e2d9c5046e9f9aa741a3`.

The complete validated summary remains a separate detailed download. A smaller website view retains all score and memory gates, exact controls, original full trace counts, first drift/violation markers, layer outputs, every failed stage and failed coordinate sample. It removes redundant passing coordinate samples and records its source-summary fingerprint. The website does not load the full 81 MB summary; its reduced transport does not turn omitted or unexecuted checks into passes.

To rebuild the summary after cloning, unpack the archive to the declared JSON path first. This PowerShell example keeps an existing local report and writes the rechecked summary to a new output:

```powershell
if (-not (Test-Path 'docs/research/checks/normalization-head-2026-10-05/natural-matched-seed-2027.json')) {
  .\.venv\Scripts\python.exe -m gzip -d docs/research/checks/normalization-head-2026-10-05/natural-matched-seed-2027.json.gz
}
.\.venv\Scripts\python.exe scripts/summarize_normalization_head_precision.py --output outputs/normalization-head-summary-recheck.json
```

The exporter checks the uncompressed report's declared identities and comparisons before publishing. It uses public records and text source; it does not load the private corpus or checkpoint tensors. Use another unused output path for a repeat. The existing declaration, raw report and validated summary remain the evidence source.
