# Trained-checkpoint precision study — 4 October 2026

The precision changes that helped the earlier tiny-model grid did **not** make the historical trained models agree across reading routes. Every BF16 treatment passed **0 of 4 internal cases and 0 of 4 FP32-anchor cases for each of the three hybrids**. Full FP32 passed 4/4 cases for 1:3 and 1:7, but passed only 3/4 for 1:15: token-by-token prediction at length 127 exceeded the unchanged tolerance.

K1 is a completed, read-only numerical diagnostic with retained failures. It does not certify or promote a backend, approve a new production precision policy, or rank language quality. Historical checkpoints and BF16 defaults remain intact. A follow-up natural-text/full-model-FP32-gradient report and validated compact summary are recorded below. The bounded training pilot is prepared and unexecuted while the numerical gate remains unresolved.

The evidence consists of the [protocol declared before execution](checkpoint-precision-protocol-2026-10-04.json), the [validated summary](checkpoint-precision-summary-2026-10-04.json), and three raw reports: [1:3](checks/trained-precision-2026-10-04/hybrid-1-3-seed-1337.json), [1:7](checks/trained-precision-2026-10-04/hybrid-1-7-seed-1337.json) and [1:15](checks/trained-precision-2026-10-04/hybrid-1-15-seed-1337.json). The [earlier tiny-model study](PRECISION_STUDY.md) remains a separate experiment.

## Why repeat the check with trained models?

An untrained model starts with randomly chosen numbers. Training adjusts those numbers using examples. Learned weights can produce different intermediate values and memory, so a useful finding on a small random model might not transfer to a trained model.

The earlier grid had four layers, width 64 and a vocabulary of 128. K1 loads the historical `week3-700m-v1` best checkpoints at their actual geometry: 16 layers, width 448, vocabulary 16,000, recurrent state size 128 and Mamba head dimension 64. A width is the number of features in a hidden vector; a vocabulary is the set of possible token IDs. The run ID is a historical name and is not a claim that the model has 700 million parameters.

Each model has a different attention:Mamba ratio. Across 16 layers, 1:3 has four attention layers, 1:7 has two, and 1:15 has one. K1 changes precision within each existing model; it does not train a new architecture or isolate the causal effect of adding attention.

| Term | Beginner explanation |
|---|---|
| Checkpoint / weights | A saved model and its learned numbers. K1 reads the weights without changing them. |
| Token / logits | A token is a numbered piece of text. Logits are the model's next-token scores before conversion into probabilities. |
| FP32 / BF16 | Numerical formats with different precision. FP32 retains more detail; BF16 uses fewer bits. |
| Autocast | PyTorch chooses lower-precision arithmetic for supported operations in a selected region. FP32 storage alone does not force FP32 arithmetic inside that region. |
| Scan / projection | A scan carries Mamba's numerical memory through tokens. A projection uses learned numbers to convert one vector into another. |
| Chunk / boundary | A chunk is a group of positions processed together. A boundary is where one such group ends and the next starts. |
| Prefix / cached state | A prefix is the already-read part of a prompt. Cached state stores the information needed to continue from it. |
| Prefill / decode | Prefill reads a prompt into memory. Decode processes a new token using that retained memory. |
| Internal agreement / anchor | Internal agreement compares different reading routes under one precision policy. The FP32 anchor compares each route with its corresponding FP32 calculation. |
| Tolerance / parity | The allowed numerical difference and agreement within that limit. A parity failure can contain entirely finite numbers. |
| Gradient | The training signal describing how a loss changes when a learned number changes. |
| Frozen operands / synthetic state | Fixed numerical inputs isolate one operation. A synthetic state is a made-up initial memory, rather than memory produced by a real prefix. |
| Hash / provenance | A hash fingerprints exact contents. Provenance records the files, settings and environment behind a measurement. |
| TF32 | An NVIDIA arithmetic mode that can round FP32 inputs more coarsely during some GPU operations. K1 explicitly disables it. |

## The declared K1 grid

The grid contains three checkpoints × four lengths × five treatments = **60 full-model cases**, or 20 per raw report. It uses CUDA, the portable `reference` backend, batch size 1 and diagnostic seed 1337. These are three historical trained models with one diagnostic seed, rather than three independently trained replicas of each architecture.

The scan chunk size is 128. Length 127 stops just before the first boundary; 128 stops on it; 129 crosses it; 257 crosses two boundaries. This bounded grid checks grouping and leftover tokens. It cannot guarantee agreement for every length, batch or memory state.

The five treatments are:

1. Full FP32 reference.
2. Original BF16 arithmetic.
3. BF16 with autocast disabled only inside the scan.
4. BF16 with autocast disabled for every Mamba output projection and the vocabulary projection.
5. BF16 with both scan and output-projection autocast disabled. This changes two factors and is exploratory.

The output-projection treatment leaves input projections, convolution, attention and feed-forward projections outside the intervention. Each treatment uses a copy of the same loaded weights and exactly the same tokens for that checkpoint. Weight fingerprints are checked after execution. Random targets are drawn to preserve the diagnostic input-draw order; K1 does not use them to calculate language-model loss.

The recorded token and target hashes also match across all three checkpoint reports. That gives identical random inputs in this saved grid, but still does not isolate attention: the architectures and their learned weights differ. A shared seed by itself would not prove identical inputs; the recorded hashes establish that fact here.

The declared precision controls are FP32 matmul precision `highest`, CUDA matrix TF32 disabled and cuDNN TF32 disabled. These controls belong to this new diagnostic. The earlier tiny-grid reports keep their originally recorded settings. No optimizer runs, no trained checkpoint is overwritten, and no official fused Mamba kernel executes.

The comparison rule remains:

```text
absolute difference <= atol + rtol × absolute reference value
FP32: atol = 0.00003, rtol = 0.0003
BF16: atol = 0.002,   rtol = 0.02
```

The absolute part allows a small fixed difference. The relative part scales with the reference value. The thresholds were not widened after observing failures.

## What the reading routes compare

Full forward reads the complete sequence together. Bounded prefill reads the prompt through the cached implementation. A prefix/suffix route first reads all but the last three tokens, then decodes those remaining tokens using the memory that the prefix actually produced. A whole-prompt tokenwise route starts from empty memory and reads every token individually.

The cached model routes retain real prefix memory: Mamba convolution and recurrent state, or attention keys and values. Their internal endpoint compares logits, retained memory and the limited causality check. Causality means that changing a future token must not change earlier predictions; this diagnostic changes the final token only.

The anchor endpoint compares each BF16 route's logits and memory with the matching FP32 route. Internal agreement and anchor agreement are separate requirements. Two BF16 routes can agree with each other while both differ from FP32.

## Recorded results

Each table entry counts the four tested lengths. “0/4” means that every case fails at least one required comparison, rather than every individual array comparison failing. The summary also preserves the stricter whole-case endpoint, layer-stage diagnostics and failed lengths.

| Treatment | 1:3 internal | 1:3 FP32 anchor | 1:7 internal | 1:7 FP32 anchor | 1:15 internal | 1:15 FP32 anchor |
|---|---:|---:|---:|---:|---:|---:|
| Full FP32 reference | 4/4 | Reference | 4/4 | Reference | 3/4 | Reference |
| Original BF16 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| BF16, scan autocast disabled | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| BF16, output-projection autocast disabled | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| BF16, scan and output-projection autocast disabled | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |

All BF16 rows retain failures at 127, 128, 129 and 257. Full FP32's only failed case is 1:15 at length 127. All three raw reports and their summary state `completed_with_parity_failures` and `certified: false`. Completion means that measurements finished; it is not numerical approval.

The earlier tiny grid showed an internal-agreement improvement when scan autocast was disabled. K1 shows that this repair was insufficient for these trained shapes and weights. Because model size, learned weights, chunk size and some runtime controls differ between the two studies, their contrast cannot identify one universal cause. The useful conclusion is that the proposed repair did not pass the required trained-checkpoint gate.

## Keep the FP32 failure visible

In the 1:15 FP32 case at length 127, `token_decode_vs_full` fails with maximum absolute logit difference **0.00008535385131835938** (`8.535385131835938e-5`) and maximum tolerance ratio **1.6785069704055786**. The tolerance ratio divides an element's difference by its allowed difference; a value above 1 exceeds the limit. The largest absolute difference and largest ratio need not belong to the same element.

The compared values are finite. Bounded prefill, the prefix/suffix decode route, retained memory and the final-token causality check pass in that case. Its recorded Mamba layer-stage comparisons also pass, and `first_failed_layer_stage` is null. These observations do not localize the final logit failure to a recorded stage. A smaller discrepancy is still a failed preregistered endpoint; it is not silently rounded into a pass.

This result also limits the phrase “FP32 reference.” It is the common calculation used for comparison, not an exact real-number oracle or proof that every FP32 execution route agrees.

## What the frozen probes establish

The exploratory scan probes freeze the first Mamba layer's projected inputs from the original BF16 full forward. They compare several scan routes with a separately written FP32 sequential recurrence. That recurrence calculates the same memory update one token at a time. It offers a different computational route, while still having finite rounding error.

| Frozen probe endpoint | 1:3 | 1:7 | 1:15 |
|---|---:|---:|---:|
| Complete scan probe, autocast active | 0/4 | 0/4 | 0/4 |
| Complete scan probe, autocast disabled | 4/4 | 4/4 | 4/4 |
| Direct scan gradients, autocast active | 4/4 | 4/4 | 4/4 |
| Direct scan gradients, autocast disabled | 4/4 | 4/4 | 4/4 |
| First Mamba output projection, either setting | 4/4 | 4/4 | 4/4 |
| Vocabulary projection, either setting | 4/4 | 4/4 | 4/4 |

With autocast active, the failed scan checks concern synthetic nonzero-memory tokenwise outputs and outputs compared with the sequential recurrence. The recorded zero-memory and retained-state checks pass. Disabling scan autocast passes the complete frozen scan probe in all four lengths for each model, yet the full-model scan-only treatment still fails. Fixing one isolated operation is insufficient to approve the complete model.

Synthetic nonzero memory is deliberately constructed. It is separate from memory produced by an actual model prefix. Frozen values also cannot recover detail already lost in earlier calculations. The direct gradient probe measures derivatives of scan inputs and initial memory, including an objective involving final memory; it does not check every learned parameter through the complete network.

The projection probes reuse identical hidden vectors and learned projection weights. Their passes show no observed same-input projection-shape failure on this grid. They do not exclude a projection amplifying differences that arrive from earlier layers. Likewise, a recorded `gate_norm` first-failure stage is where a threshold is first crossed among the observed stages, not proof that this stage caused the original discrepancy.

## Execution resources are not a benchmark

The reports record Python 3.11.9, PyTorch 2.11.0+cu128, NumPy 2.4.6, CUDA runtime 12.8 and an NVIDIA GeForce RTX 5070. `mamba-ssm`, Triton and `causal-conv1d` are absent from the recorded environment. CUDA execution here uses portable PyTorch reference code.

| Complete diagnostic report | Recorded wall time | Peak PyTorch allocated memory |
|---|---:|---:|
| 1:3 | 78.78 seconds | 957.82 MiB |
| 1:7 | 83.68 seconds | 976.56 MiB |
| 1:15 | 85.34 seconds | 990.07 MiB |

These measurements cover each report's complete diagnostic workload, including multiple treatments, copying models, comparisons and probes. They are not synchronized repeated throughput benchmarks and cannot rank training or generation speed. PyTorch allocated memory is different from total device occupancy, which also includes runtime and other allocations. One MiB is 1,048,576 bytes.

## Code walkthrough and evidence validation

[`scripts/study_scan_precision.py`](../../scripts/study_scan_precision.py) performs the diagnostic. `_load_model` reads the checkpoint with its stored model configuration. `run_study` builds the fixed input grid, copies learned weights into treatments, records hashes and checks that weights remain unchanged. `experimental_precision` temporarily changes selected calculation regions and restores wrappers and hooks afterward. `_evaluate` runs the reading routes, and `_comparisons` keeps internal and anchor outcomes separate. `scan_isolation`, `gradient_isolation` and `projection_isolation` investigate frozen numerical inputs without replacing the full-model results.

[`scripts/summarize_checkpoint_precision.py`](../../scripts/summarize_checkpoint_precision.py) checks the declaration, all three expected checkpoint identities, source and input fingerprints, runtime controls, model geometry, treatment/length coverage and result flags before counting outcomes. The compact table therefore points back to complete raw evidence rather than discarding failed cases.

To revalidate the saved reports without launching a model, choose a new output path:

```text
python scripts/summarize_checkpoint_precision.py --protocol docs/research/checkpoint-precision-protocol-2026-10-04.json --reports docs/research/checks/trained-precision-2026-10-04/hybrid-1-3-seed-1337.json docs/research/checks/trained-precision-2026-10-04/hybrid-1-7-seed-1337.json docs/research/checks/trained-precision-2026-10-04/hybrid-1-15-seed-1337.json --output docs/research/checkpoint-summary-recheck.json
```

The summarizer rejects an existing output path. Re-executing the CUDA diagnostic is a separate operation: match the declared TF32 and precision flags as well as the checkpoint, seed, batch size, lengths and chunk size. Source-file hashes identify measured code contents; a later Git publication commit identifies where the evidence was stored. Those identities should not be substituted for one another.

## K2/K3: one natural window and full-model FP32 gradients

The [natural-text protocol](checkpoint-numerics-protocol-2026-10-04.json) and [completed raw report](checks/trained-numerics-2026-10-04/natural-257-seed-2027.json) cover one shared validation window of 257 tokens, with seed 2027, batch size 1 and chunk size 128. All three checkpoints receive identical tokens and shifted next-token targets. The window begins at validation token offset 2,342,241. Its real prefix has 128 tokens and its continuation has 129, so the continuation crosses a chunk boundary rather than reducing to a one-token call. The raw report and [validated compact summary](checkpoint-numerics-summary-2026-10-04.json) retain `certified: false` and `completed_with_parity_failures`.

Teacher forcing supplies the true preceding tokens and asks the model to score the true next token. Next-token negative log-likelihood, or NLL, measures the probability assigned to that target; lower loss means higher target probability on average. This keeps the treatments on identical text instead of allowing an earlier generated mistake to change later inputs.

K3 compares full stateless forward and backward under FP32 using `reference` versus `torch_chunked`, the two portable scan paths. It computes next-token loss through the complete model, then compares every unique learned parameter-gradient tensor. The reference gradient arrays are copied to CPU and its graph is freed before the candidate graph is built. There is no optimizer, clipping or weight update. This is broader than a frozen first-scan derivative probe, but it covers FP32 backends on this one recorded window. **BF16 full-model gradient parity remains unexecuted.**

| Model | FP32 logit maximum tolerance ratio | FP32 logits | FP32 loss | FP32 gradient tensors passing |
|---|---:|---|---|---:|
| 1:3 | 0.529364 | Pass | Pass | 186/186 |
| 1:7 | 0.541036 | Pass | Pass | 198/198 |
| 1:15 | 1.159336 | Fail | Pass | 204/204 |

The gradient counts refer to parameter **arrays**, rather than numbers of scalar parameters or independent experiments. Tied weights are counted once. Every recorded FP32 parameter-gradient comparison and every FP32 loss comparison passes. The 1:15 forward-logit comparison still fails: its maximum absolute difference is `5.8650970458984375e-5`. Finite gradients and an agreeing aggregate loss therefore do not approve the stricter forward endpoint. This is a separate comparison from K1's length-127 tokenwise failure.

The original BF16 inference probe reads the full window, loads its real 128-token prefix and clones that naturally produced prefix memory for each continuation schedule. It compares a single 129-token continuation, a segmented 128+1 continuation and tokenwise continuation. The original prefix remains unchanged after the clones are used. Whole-window tokenwise reading from empty memory is also recorded.

| Model | BF16 internal logit comparisons passing | BF16 retained-state bundles passing | BF16 full logits versus FP32 anchor |
|---|---:|---:|---|
| 1:3 | 0/8 | 1/4 | Fail |
| 1:7 | 0/8 | 1/4 | Fail |
| 1:15 | 0/8 | 1/4 | Fail |

The one passing retained-state bundle for each model is the single-call suffix continuation compared with stateful full reading. A bundle passes only when all its required memory fields pass. These eight logit comparisons and four memory bundles are correlated checks on one window, rather than statistical replication. They demonstrate that the unresolved numerical issue also appears with naturally produced prefix memory.

The raw report also describes the effect of full BF16 versus full FP32 on target probabilities:

| Model | Mean next-token NLL change, BF16 minus FP32 | Greedy next-token choices that differ |
|---|---:|---:|
| 1:3 | +0.00148950 | 2/257 |
| 1:7 | −0.00153022 | 2/257 |
| 1:15 | +0.00023785 | 5/257 |

A greedy choice is the token with the largest score. These counts ask whether that choice changes under the two numerical policies at fixed true-text positions; they do not measure correctness or the quality of a freely generated passage. A positive NLL change means BF16 assigned lower average target probability on this window; a negative change means higher. The study declares no quality-equivalence margin, so these small average changes cannot establish that the policies are equally useful or rank the architectures.

The historical best checkpoints were selected using the existing validation pool. Reusing that pool is a descriptive numerical check, not an independent quality test. One window provides limited numerical coverage, and its 257 positions do not substitute for independent data samples or training seeds. A generalization conclusion needs separately frozen evaluation data.

The recorded processing times are 10.53, 10.48 and 10.69 seconds for 1:3, 1:7 and 1:15 respectively; their peak PyTorch allocations are 2052.53, 2309.12 and 2434.69 MiB. These times include transfers, hashing and checks, and are not throughput benchmarks. [`scripts/study_checkpoint_numerics.py`](../../scripts/study_checkpoint_numerics.py) records the underlying forward, gradient, real-prefix and per-token comparisons; a compact exporter must retain the failed strict forward endpoint as well as the passing gradients.

## Prepared pilot and decisions

**K4: bounded operational training pilot — prepared, unexecuted.** A short run with one declared seed can test training stability, evaluation, saving, resume behavior and resource cost. The prospective FP32 policy must be explicitly recorded and scope its TF32 controls; BF16 remains the historical default. The pilot is held while the numerical gate remains unresolved, including the 1:15 strict FP32 forward failure. This pilot cannot establish an architecture-quality ranking, and pure-attention/pure-Mamba quality baselines remain unmeasured. No pilot outcome is reported here.

**R-019:** test actual trained shapes and memory from real prefixes before relying on the tiny-grid repair. **R-020:** give a new training precision policy its own recorded identity instead of silently changing historical runs. **R-021:** use natural next-token diagnostics to investigate numerical effects while retaining the validation-selection limitation. **R-022:** keep the first operational pilot bounded; require controlled training, repeated seeds and measured baselines before an architecture conclusion.

K1 and the natural-window report retain the original numerical thresholds and all failures. The next work should investigate the remaining whole-model discrepancies, including both FP32 forward failures, broaden natural-window and shape coverage, and keep untested BF16 full-model gradients explicit before making a production precision recommendation.

## Revalidate the natural-text summary

The natural-text exporter checks the expected checkpoints, source fingerprints, original thresholds, all unique gradient arrays, real-prefix memory and descriptive statistics against the retained per-token values. It keeps passing gradients separate from failed predictions. Source validation accepts LF/CRLF line-ending forms of the same text, while preserving the original measured byte hashes.

```text
python scripts/summarize_checkpoint_numerics.py --output docs/research/natural-summary-recheck.json
```

Choose a new output path: the command refuses to replace an existing artifact. This strict revalidation uses the original local prepared-data metadata, validation window, tokenizer and historical training manifests. Those large/local artifacts are not all in a fresh source checkout. The public raw report and compact summary remain available for inspection; the automated exporter tests build small fixtures from public evidence and do not require private checkpoints or the full corpus. Revalidation performs no model or optimizer calculation.

## Recovery audit before a wall-bounded pilot

The trainer saves individual files atomically: a reader sees the complete old or complete new version of one file. Several file replacements are not automatically one transaction. A process killed after `best.pt` is replaced but before `last.pt` advances can leave a best/latest pair that strict resume validation rejects. A kill during the initial evaluation can also occur before a resumable baseline is saved.

This is a source-audit finding, not an observed failure or corruption of the historical runs. Before adding a time-limit watchdog, test a coordinated stopping point after the model snapshots, counters and metrics agree. Retain a forced-stop arm as incomplete if recovery cannot be validated. No watchdog or pilot was executed in this increment.

The next numerical isolation will keep the same failing 1:15 checkpoint and natural text, record the violating score coordinates, and trace the difference through its layers. At the first divergent operation, freeze its inputs and compare with an independent FP64 sequential recurrence. FP64 retains more numerical detail than FP32 and serves as an exploratory diagnostic reference; it does not change the production policy or the original pass threshold. A passing 1:3 checkpoint provides a control.
