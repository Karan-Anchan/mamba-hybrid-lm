# Numerical isolation and recoverable stopping

Study record, 4 October 2026. This extends the earlier trained-checkpoint checks. It retains their weights, input window and numerical thresholds. Each declared experiment runs on CUDA with both TF32 controls disabled. These are numerical and software studies, not new language-quality rankings.

## What the layer trace found

The exact earlier natural-text window starts at validation token 2,342,241. It contains 257 input tokens and their true next-token targets. The 1:15 checkpoint is the failing contrast; 1:3 is the passing control. Neither is retrained.

| Observation | 1:15 | 1:3 |
|---|---|---|
| First nonzero difference | Layer 1 scan output | Layer 1 scan output |
| First threshold violation | Final next-token scores | None |
| Final score violations | 1 of 4,112,000 | 0 of 4,112,000 |
| Largest tolerance ratio | 1.159336 | 0.529364 |

The failing coordinate is `[batch 0, token 209, vocabulary 9]`, using zero-based indexing. Token 209 is the 210th input position. Its reference score is -0.01034456398 and its chunked score is -0.01038294192. Their difference is 0.00003837794. The largest absolute difference occurs at another vocabulary option; maximum absolute error and maximum tolerance ratio need not identify the same number.

A tolerance ratio divides the observed difference by the allowed difference. Values at most 1 pass. The original allowance is `0.00003 + 0.0003 × abs(reference)` for FP32. We keep the one failing value. Counting many passing values cannot erase it.

Every captured intermediate comparison passes on this window. Small differences start in the first scan and propagate through later calculations. This narrows the investigation, but does not prove that one operation is the only cause. The earlier full-model loss and all gradient-array checks remain separate passing observations.

## Freeze the operands to isolate one calculation

An operand is an input number or array used by a calculation. If earlier layers already disagree, later layers may receive different operands. To compare scan formulas themselves, every replay receives an identical set of actual reference-model operands.

We replay layer 1 in both models and layer 15 in 1:15. Those choices follow the predeclared first-drift/first-violation rule. Each selected layer has two separate starting conditions:

- Zero starting memory, followed by all 257 tokens.
- Actual memory produced by reading the 128-token prefix, followed by the remaining 129 tokens. Its convolution history is also captured. This is model-produced memory, not a random artificial state.

The independent FP64 oracle follows a separately written recurrence:

`new memory = decay × old memory + current token contribution`

`output = read new memory + shortcut contribution`

FP64 keeps more numerical detail than FP32. The oracle uses elementary operations and a token loop; it does not call either SSD implementation or reuse their cumulative sums. Simply converting the existing scan to double would not produce this oracle, because those implementations explicitly convert their operands back to FP32.

The zero-memory replay compares quadratic, one-call stateful, chunked-128 and tokenwise routes. The real-prefix replay compares the three stateful routes; the stateless quadratic route has no retained starting-state input. Every tested frozen output and retained-state comparison passes against the independent recurrence under the unchanged FP32 allowance. That result covers these frozen operands and states only.

## A separately named decay experiment

Decay decides how much earlier memory remains. The quadratic formula accumulates log-decay totals, then subtracts two totals to obtain a shorter interval. Rounded totals can make that subtraction less accurate.

The frozen experiment changes cumulative sums, subtraction and exponentiation to FP64, then casts their coefficients back to FP32 before the original contractions. A second contrast also changes the initial decay multiplication to FP64. Both reduce the maximum scan error on the selected zero-memory operands. This proposes a follow-up; it does not certify a full model.

The distinct `fp64_cumsum_decay_coefficients` declaration tests the accumulation-only candidate through full-model predictions, loss and every unique parameter-gradient array. Its reference and chunked routes also compare with the common original FP32 anchor. That declaration follows the frozen observation and precedes execution. All three trained models pass the narrow comparison, including all 588 unique parameter-gradient arrays. Both candidate routes also pass against the original reference. Original controls reproduce their earlier fields, including the failed 1:15 prediction check.

## Broaden the successful window before choosing a policy

The next declaration fixes five shapes before execution: batch one at lengths 127, 128, 129 and 257; batch two at length 512. Chunk size stays 128. Each case uses identical natural text and targets across models and treatments. Genuine prefix memory is cloned before one-call, chunked and tokenwise continuation. Only the batch-two/512 case adds full-model gradients; shorter cases explicitly omit them.

| Gate | Original | FP64 cumulative-decay treatment |
|---|---:|---:|
| Full reference/chunked comparison | 14/15 | 15/15 |
| Complete cached gate, including anchors | 11/15 | 12/15 |
| Internal cached score comparisons | 130/135 | 133/135 |
| Cached scores against original full-reading anchor | 100/105 | 102/105 |
| Internal complete memory bundles | 75/75 | 75/75 |
| Memory bundles against original stateful anchor | Self-comparison omitted | 90/90 |
| Unique gradient arrays across three batch-two/512 models | 588/588 | 588/588 |

All 15 cases completed within the declared cooperative diagnostic allowance. The positive narrow result transfers to full forward comparisons; complete cached approval still fails. The treatment's three remaining failures are all 1:15 whole-text tokenwise scores:

- Length 128: internal agreement passes, but scores against the original full-reading anchor fail at ratio **1.004406**.
- Length 129: internal and original score-anchor comparisons fail at ratio **1.117571**.
- Batch two, length 512: internal and original score-anchor comparisons fail at ratio **1.098698**.

The original 1:3 batch-two/512 tokenwise failure is removed by the treatment. Its full forward result also removes the original 1:15 length-257 failure. These improvements do not erase the three remaining treatment failures. The original stateless score anchor and original stateful memory anchor answer separate questions; a missing self-anchor is not a measured pass.

In the batch-two case, 512 positions are scored in each row: 1,024 positions total. Position bounds describe each row, and stored arrays flatten the first row followed by the second. The exporter validates this layout without changing scores or tolerances. The text comes from the checkpoint-selection validation pool. Neither these small numerical effects nor passing gradients establish independent language quality. BF16 and official fused kernels were unexecuted in the decay studies, and production defaults remain unchanged.

## Why attention position tables needed their own study

RoPE means rotary position embedding. It rotates attention query/key values according to token position. Full-prompt attention originally uses FP32 cosine/sine tables; cached attention rounds those tables to BF16. Even identical query/key operands can therefore rotate differently.

The declared counterfactual freezes actual pre-rotation queries, keys and tables. It checks the post-rotation tensors and a separately labeled post-cast replay. It also runs the complete model with three temporary policies:

| Policy | Change | Full-model BF16 predictions | Complete memory bundles | Common FP32 anchor |
|---|---|---|---|---|
| Original | Cached route alone rounds tables | 0/8 for every model | 1/4 for every model | Fails for every model |
| Shared FP32 | Both rotate in FP32 and explicitly cast back | 0/8 for every model | 1/4 for every model | Fails for every model |
| Shared BF16 | Both round tables to BF16 before rotation | 0/8 for every model | 1/4 for every model | Fails for every model |

The original controls exactly reproduce their saved baseline numerical fields. Matching the frozen rotation policy makes the frozen routes agree by construction. Complete-model agreement still fails. The isolated discrepancy exists, but changing it alone does not resolve the broader BF16 disagreement. It also cannot explain the separate FP32 scan contrast.

Shared FP32 includes an explicit final cast. It is a declared treatment, rather than every detail of the original full route. Shared BF16 changes descriptive next-token loss on this one window; those changes do not establish a quality improvement. This validation pool also helped select the checkpoints, so it is not an untouched quality test.

## Save a complete generation of training files

`best.pt` holds the lowest-loss saved model. `last.pt` holds the resumable model, optimizer, random state and progress. `metrics.jsonl` records the update and evaluation history. Previously, a process could stop after a new best file was published but before the matching latest file advanced. Atomic replacement of one file does not coordinate all three.

The new transaction stages a complete next generation and verified previous snapshots. A write-ahead journal records which generation is valid before public replacements begin. An interrupted pending publication restores the previous generation; a committed journal completes the next generation. Recovery validates the run signature, checksums and allowed paths before writing. A run without a journal retains strict legacy validation; ordinary corruption is not silently repaired.

Checkpoint backups use immutable hard links when available, with a copied and flushed fallback. Metrics receive a copy because they can be appended. Cleanup validates the private filenames and paths before removing inert staging files. Historical training checkpoints are not edited.

Fault tests interrupt baseline publication and later improved checkpoints at the staging, journal, file-replacement, commit and cleanup steps. They compare resumed learned numbers, optimizer state, random state, counters and semantic metrics with uninterrupted runs. One test calls `os._exit` after best advances while latest remains at step zero, then verifies resumed equality. Timing fields naturally differ between invocations.

This contract addresses process interruption. It does not promise power-loss durability across every filesystem operation.

## Cooperative stopping and the pilot boundary

A cooperative stop requests that training finish current work, publish a complete checkpoint generation, then return an explicitly incomplete status. The controller records elapsed time and overshoot. An in-flight GPU update, evaluation or save can exceed the wall allowance; the allowance is not an arbitrary forced-kill timer.

Sweeps share one invocation deadline across arms. If an arm stops, completed campaign comparison files are withheld. Resume starts a fresh invocation clock under the same signed policy. The optional wall setting does not change maximum updates or the learning-rate schedule. Missing/default settings retain their historical identity; a nondefault policy is recorded and signed.

The prepared five-model research pilot remains behind its numerical and recovery gates. The 64 CPU fault/stopping tests pass, including real abrupt process exit and exact resumed comparison. CUDA requires its own evidence.

## CUDA recovery: useful operation, failed exact endpoint

A separate predeclared operational smoke uses one fresh full-size 1:15 model, FP32, the portable reference backend, block 512, batch one and four optimizer updates. A control runs uninterrupted. The paired trajectory requests a stop only after complete step-two checkpoint publication, then resumes under the same learning-rate schedule to step four. Both trajectories see 2,048 training positions. This is an operational check, not the five-arm research pilot.

The complete workflow takes **157.156 seconds**, within its 300-second allowance, including input validation and evidence checks. It stops durably at step two and resumes successfully to step four. All three historical checkpoint fingerprints remain unchanged. Exact saved-state comparison nevertheless fails:

| Category | Exactly equal? | Differing entries/arrays |
|---|---|---:|
| Model | No | 130 of 205 saved tensors |
| Optimizer | No | 408 of 612 tensors |
| Complete random state | Yes | 0 |
| Training-batch generator | Yes | 0 |
| Completed updates and token counters | Yes | 0 |
| Best validation loss | No | 1 scalar |
| Semantic score records, excluding timing | No | 2 fields |

An array difference counts the whole tensor once, even if very few elements differ. Timing and resource fields are excluded from the equality comparison; learned values and score records are not. The runtime reports deterministic algorithms disabled. Nondeterministic CUDA arithmetic is a possible explanation, but this run cannot establish it or exclude a recovery issue. Repeated uninterrupted controls and a separately declared deterministic policy are needed to isolate the cause. Keep the exact failed endpoint and do not relabel it as approximate success.

The raw operational status is `completed_with_recovery_failures`, with `certified: false`, `research_pilot_executed: false` and `backend_parity_certified: false`. Snapshot publication checks and CPU equality evidence are valuable, but the CUDA recovery gate remains open. No quality ranking, fused speed gain or completed research pilot follows from this smoke.

## Follow the functions

| Function | Beginner walkthrough |
|---|---|
| `trace_execution` | Attach temporary stage recorders, capture detached outputs, then remove recorders and restore backends in `finally`. |
| `detailed_comparison` | Check shapes and finite values; count differences and violations; save a bounded sample of their exact locations. |
| `recurrence_fp64` | Update a private memory copy one token at a time with an independent formula. Return outputs and final memory. |
| `real_prefix_scans` | Read the actual prefix, clone its memory, capture suffix operands and verify the original prefix remained unchanged. |
| `frozen_rope_probe` | Rotate identical actual queries/keys under three policies; distinguish rotation precision from a comparison-only final cast. |
| `rope_policy` | Install a temporary per-instance attention treatment and restore original methods/hooks afterward. |
| `temporary_decay_treatment` | Replace both scan functions only inside the diagnostic scope, then restore them in `finally`. |
| `prepare_inputs` | Choose shared natural text once per declared shape and fingerprint the inputs and shifted targets. |
| `study_case` | Compare full routes, real cached continuations, complete memory and the gradients declared for that shape. |
| `_publish_checkpoint_transaction` | Stage and validate a complete generation, write the journal, publish files, mark committed and clean staging. |
| `_recover_checkpoint_transaction` | Verify identity and snapshots, then restore the journal’s valid generation without changing historical files. |
| `TrainingStopControl` | Record a request or deadline and describe actual elapsed time; training chooses a durable stopping boundary. |
| `training_stop_scope` | Share the same stop controller across sequential arms, then restore the caller’s previous scope. |

Evidence: the declared protocols and immutable reports are under `docs/research/`. The site exports compact data through independent validators. A publication commit locates the files; measured source hashes identify the exact code used during each execution.
