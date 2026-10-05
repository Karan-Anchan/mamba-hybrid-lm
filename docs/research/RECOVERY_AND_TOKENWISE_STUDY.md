# Repeated training controls and one-token arithmetic

This study follows two unresolved checks recorded on 4 October 2026. It keeps the old evidence and original acceptance rules. The longer training pilot is still prepared.

## Why repeat an uninterrupted run?

The first CUDA recovery smoke saved a complete second update, stopped, and resumed to its fourth update. Its random state and progress counters matched an uninterrupted run, but some learned arrays, optimizer memory and scores differed exactly. That observation cannot tell us whether the restart caused the difference: two uninterrupted GPU runs might also differ.

The next comparison therefore has three trajectories per policy. Control A and control B each start from the same declared seeds and run four updates. A third starts from those seeds, stops after its complete second saved update, and resumes to four. Compare A with B, then A with the resumed trajectory. Keep those questions separate.

One policy retains the historical nondeterministic algorithm settings. Another requests strict deterministic algorithms and deterministic convolution. Both receive the same declared cuBLAS workspace setting in fresh processes, before PyTorch loads. This is a controlled new environment. It does not reproduce every ambient setting of the historical smoke.

**Deterministic** means repeated execution should produce the same numbers under the same inputs, state and settings. **cuBLAS** is NVIDIA's GPU matrix-multiplication library. Its workspace is temporary calculation memory; the environment setting controls an execution choice used by the deterministic study. Strict mode may reject a calculation with no supported deterministic implementation. Such a rejection is an incomplete result, not permission to downgrade the policy.

Exact comparison checks learned arrays, optimizer state, random state, update counts, training positions, best validation loss, the next-batch random generator and the score ledger. Timing fields are excluded because separate invocations take different amounts of time. Small numerical differences are still failures under an exact rule.

## Why trace one-token calculations?

The experimental FP64 cumulative-decay treatment passed all 15 full-reading comparisons in the breadth study. Three 1:15 one-token reading cases still failed a prediction check. Full reading calculates many positions together. Tokenwise reading calculates one position, updates real memory, and repeats. They use the same learned model and text, but some calculations have different shapes and addition orders.

**FP32** stores numbers with less detail than **FP64**. Floating-point addition rounds intermediate results, so changing the order of additions can change the final number slightly. Disabling TF32 does not require differently shaped FP32 calculations to be bit-for-bit identical.

Keep three references separate:

- The original stateless full-reading scores ask whether the treatment agrees with the original calculation.
- The treatment's own full-reading scores ask whether reading one token at a time agrees with that treatment.
- The original stateful end memory asks whether the complete saved recurrent state agrees. Passing memory cannot erase a failed score comparison.

Temporary hooks record stages inside the model. The trace finds the first nonzero difference and the first difference outside the fixed allowance. Those are different questions: a small early difference may remain acceptable before later calculations amplify it.

At selected mixers, frozen replays feed identical inputs to projections, causal convolution, attention and the state scan. A projection combines features through learned matrix weights. Convolution reads a short recent token window. Attention compares a query with earlier keys and values. The state scan updates compact numerical memory. An independent FP64 calculation helps diagnose rounding in a frozen operation; it does not approve a whole-model replacement.

## What the evidence can establish

These are numerical and operational checks on one device and runtime. They do not rank language quality, measure an official fused Mamba kernel, or replace the planned independent-seed research campaign. Each workflow has its own shared 900-second allowance, an 8 GiB free-memory guard and fresh output paths. Preserve incomplete cases, the historical checkpoints and every negative result.

Runtime background: [PyTorch 2.11 reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html) explains the distinction between seeded randomness and deterministic algorithms; [the deterministic-algorithm API](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html) documents strict errors and CUDA workspace requirements.

## Actual CUDA outcomes

Both declarations were recorded before execution on 4 October. Publication and explanation continued on 5 October. Historical checkpoint bytes, input identities and producer sources passed the final integrity checks.

The recovery workflow completed in **472.031 seconds**, with no overshoot of its shared 900-second allowance.

| Algorithm policy | Uninterrupted A versus B | Control A versus stop/resume |
|---|---:|---:|
| Legacy, shared declared workspace | 8/8 exact categories | 6/8 exact categories |
| Strict deterministic, same workspace | 8/8 exact categories | 8/8 exact categories |

Legacy restart differs in 130 of 205 model arrays and 408 of 612 optimizer arrays. The largest absolute differences are approximately 0.0000002384 and 0.0000000005821 respectively. Random state, progress, the batch generator, best loss and semantic scores agree in this new controlled arm. Keep the earlier smoke's different score outcome as its own historical record.

Strict mode provides a reproducible short CUDA recovery control on this device and runtime. The two policies differ in a group of deterministic settings; this does not identify a single responsible operator. No production setting or default was promoted. Future training must sign any accepted execution policy into its configuration rather than relying on an ambient flag.

The tokenwise workflow completed all six cases in **111.156 seconds**, with no overshoot. Every original full-reading repeat is exactly equal. All twelve complete retained-memory comparisons pass. The frozen projection, convolution, scan, attention and final projection probes pass **408/408** recorded comparisons. These local passes do not certify every stage of the whole model.

| Model / shape | Tokenwise versus own full scores | Tokenwise versus original full scores |
|---|---:|---:|
| 1:15, batch 1 × 128 | Pass; ratio 0.633806 | Fail; one score, ratio 1.004406 |
| 1:15, batch 1 × 129 | Fail; seven scores, ratio 1.117571 | Fail; seven scores, ratio 1.117571 |
| 1:15, batch 2 × 512 | Fail; one score, ratio 1.098698 | Fail; one score, ratio 1.098698 |
| 1:3, batch 1 × 128 | Pass | Pass |
| 1:3, batch 1 × 129 | Pass | Pass |
| 1:3, batch 2 × 512 | Pass | Pass |

A tolerance ratio divides the observed difference by its fixed allowance. A value above one fails. The limits were not changed after observing these outcomes.

All six tokenwise-versus-own-full traces first differ at **layer one's RMS normalization**, before the first Mamba projection. RMS normalization rescales features using the square root of their average squared value. The operation contains a reduction: adding many values into one result. Frozen normalization was not tested here, so this is a focused next question rather than a certified replacement.

The internally failing 1:15 cases cross the threshold only at the final score projection. Their violating scores occur at the first text position. Accumulated long-token memory alone cannot explain a difference before that history exists. Passing frozen local operations also cannot erase a propagated whole-model difference.

An additional negative control matters: **1:3, batch two × 512**, passes final scores and memory but fails two coordinates at both final normalization and the final projection's input against the original route. The maximum ratios are 1.697286 for full treatment versus original and 1.653612 for tokenwise versus original. The overall recorded-comparison gate therefore remains negative even for this passing prediction control. Do not reduce the study to the final-score table.

## Follow the code

| Function | What it does in plain language |
|---|---|
| `run_declared` | Check the declared recipe, launch a fresh process for each policy, then verify the final evidence fingerprints. |
| `deterministic_policy` | Save existing runtime flags, apply the declared settings, record what is active, and restore the originals afterward. |
| `execute_policy` | Run two uninterrupted controls, compare them, stop a third after a complete second save, resume and compare again. |
| `compare_trajectories` | Compare all eight saved-state categories exactly; keep timing outside that rule. |
| `descriptive_magnitudes` | Explain the size of differences in bounded chunks without turning an exact failure into a pass. |
| `validate_worker_result` | Reject a worker whose identity, flags, exit code or conclusion does not match its actual recorded checks. |
| `TokenTrace.capture` | Save detached stage observations and token positions; transfer and validate aligned values once per stage. |
| `stage_comparisons` | Align full and tokenwise stages, count differences and violations, and locate the first of each. |
| `frozen_mixer_probes` | Replay selected operations on the exact same saved inputs, avoiding differences inherited from earlier stages. |
| `actual_prefix_probes` | Read a real prefix, clone its memory, read the suffix and verify the original prefix is unchanged. |
| `attention_oracle_fp64` | Calculate causal attention independently from identical rotated queries, keys and values. |

## Next dependency

Declare a focused frozen RMS-normalization replay and a separate full-model normalization/head precision factorial. Keep the three failing cases, the 1:3 intermediate negative control, original stateless scores and complete stateful memory anchors. Hold tolerances fixed. Broaden any successful treatment to all three models, boundary shapes and gradients before selecting a signed production policy.

The pilot remains behind that numerical gate and a signed recovery policy. Official fused-backend compatibility, independent training seeds, untouched quality evaluation, retrieval and similarity campaigns remain later research work.

Reproduction records: `cuda-recovery-determinism-protocol-2026-10-04.json`, `checks/cuda-recovery-determinism-2026-10-04/report.json`, `tokenwise-isolation-protocol-2026-10-04.json`, and `checks/tokenwise-isolation-2026-10-04/natural-remaining-seed-2027.json`. Independent exporters validate the compact public summaries without importing model code.
