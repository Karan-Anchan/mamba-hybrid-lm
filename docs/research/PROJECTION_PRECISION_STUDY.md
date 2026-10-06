# Mamba input projection and genuine-prefix arithmetic

**Status: completed with parity failures; no accepted projection change.** The single declared run completed all six checkpoint/window cases and twelve cells. The independent exporter validates its immutable raw report, complete summary and reduced website view. P0 repeats decay + N1H0 exactly; P1 changes only Mamba input-projection arithmetic. Production defaults and trained weights remain unchanged, and the longer training programme remains gated.

## What the recorded comparison shows

P1 improves the frozen projection calculation but does not qualify the whole model. On identical captured P0 full inputs, its full-shaped and one-token-shaped outputs are bit-identical at all 81 projection sites. Maximum error against the independent FP64 reference improves at all 81 sites. The first observed own-route drift moves from `in_projection.output` to `scan_output` in all six cases, while the two 1:15 own-route score failures remain.

| Across six fixed numerical cases | P0: repeated control | P1: projection treatment |
|---|---:|---:|
| One-token scores / own full scores | 4/6 | 4/6 |
| Direct final-score endpoints | 85/90 | 82/90 |
| Complete whole-route scores, loss and memory | 4/6 | 3/6 |
| Every recorded comparison | 3/6 | 3/6 |
| Intermediate trace pairings | 35/42 | 34/42 |
| Genuine-prefix continuation bundles | 5/6 | 4/6 |
| Suffix score endpoints | 70/72 | 68/72 |

Each count has its own scope. These are checkpoint/window cases, not independent training seeds or a language-quality ranking. The original repeat, prior P0 reproduction, weight/tied-parameter checks, random-state restoration and final identity audit all pass.

All 84 target-loss checks and 228 retained-memory bundles pass, covering 7,296 memory fields. The 648 frozen-full arithmetic checks across 81 captures and 576 suffix arithmetic checks across 72 captures also pass; all 144 suffix upstream/input-output checks remain below the allowance. Passing those scopes does not erase failed whole-route scores or intermediate stages. The reduced view retains all 13 failed direct score endpoints, 19 failed stage rows and six failed suffix endpoints with their bounded coordinates unchanged.

## Failures that remain or appear under P1

The length-129 1:15 own-route failure decreases from 52 violating scores in P0 to 47 in P1, but remains above the fixed allowance. P1 also introduces failure against the unchanged original scores: 13 values violate that comparison. Its batch-two own-route comparison fails at three score values.

| 1:15 score comparison | Violating scores | Maximum error / allowance |
|---|---:|---:|
| P0, length 129, one-token / own full | 52 | 1.489642 |
| P1, length 129, one-token / own full | 47 | 1.438168 |
| P1, length 129, one-token / original full | 13 | 1.223741 |
| P1, batch two × 512, one-token / own full | 3 | 1.099822 |

P0 exactly reproduces the earlier 1:15 chunked suffix failures at global text position 207 against original and decay-only references. Those two failed leaves pass under P1. However, P1 introduces own-reference suffix failures in all three reading routes at suffix position 55, global position 183. It also adds a 1:3 batch-two original-reference chunk failure at suffix position 352, global position 480.

| P1 genuine-prefix failure | Reference | Violating scores | Maximum error / allowance | First coordinate; suffix → global |
|---|---|---:|---:|---|
| 1:15, one-shot suffix | Own full | 10 | 1.201106 | `[0, 55, 771]`; 55 → 183 |
| 1:15, chunk-128 suffix | Own full | 5 | 1.116414 | `[0, 55, 2402]`; 55 → 183 |
| 1:15, one-token suffix | Own full | 2 | 1.036419 | `[0, 55, 6177]`; 55 → 183 |
| 1:3, chunk-128 suffix | Original full | 1 | 1.020100 | `[0, 352, 4390]`; 352 → 480 |

The 1:3 batch-two stateful-full and full chunk-128 comparisons also newly fail original scores at `[0, 480, 4390]`, maximum ratio 1.000998. The earlier 1:3 intermediate final-normalization/head-input negative at `[0, 416, 22]` remains visible. A barely exceeded allowance is still a failed unchanged gate.

All 72 descriptive target-score effects record zero greedy disagreements, with maximum absolute mean NLL change about `5.31e-7`. Those small descriptive changes do not establish language-quality equivalence or override the numerical negatives. The next supported local investigation concerns actual scan-output arithmetic, with recurrence state, storage and addition-order references kept separate. It does not automatically authorize an expanded grid or training pilot.

## The question

A projection multiplies a token's feature vector by a table of learned weights, then adds a bias when that module has one. Processing many tokens together and processing one token at a time use the same mathematical rule, but can group floating-point operations differently.

The preceding normalization/head study removes local normalization shape drift with N1, yet whole-route failures remain. Its first observed own-route difference moves to the first Mamba input projection in all six cases. That locates the next arithmetic contrast; it does not prove that projection caused every later failure.

This experiment asks whether a more precise Mamba input-projection calculation improves agreement on the same recorded windows, including the real-prefix failures. It tests the input-projection family across the model. It does not attribute an effect to one layer, interpret hidden features as thoughts, rank language quality, or measure production speed.

## Two cells with the same references

| Choice | Mamba `in_proj` | Fixed calculations |
|---|---|---|
| P0: repeated control | Original FP32 linear calculation | Precise decay + N1 normalization coefficient + original FP32 final head |
| P1: projection treatment | Temporary FP64 inputs, weights and optional bias; FP64 multiplication/addition; FP32 output | Same decay, normalization and final head as P0 |

P1 changes every registered Mamba `in_proj` instance. It leaves attention projections, Mamba output projections, feed-forward projections and the vocabulary head at their existing arithmetic. Learned parameter objects, storage and saved values stay fixed. Temporary instance methods are restored after success or an exception.

Both cells compare directly with four separate score references:

1. **Own:** that cell's full-reading calculation, used to check its internal route agreement.
2. **Original:** unchanged historical stateless full FP32 scores.
3. **Decay-only:** precise decay with the original normalization and head arithmetic.
4. **Projection baseline:** P0 full reading, which already includes precise decay and N1H0.

A pass against one reference is not a pass against another. A treatment can agree with its own two routes while disagreeing with the historical model. Stateless score references also do not supply memory: retained memory is compared with actual stateful full-reading runs.

## Fixed trained windows

The declared grid uses the trained 1:15 and 1:3 checkpoints, each on the three earlier cases. Seed 2027 and chunk size 128 stay fixed. The same token and shifted-target fingerprints, validation-window starts, checkpoints, tokenizer and prepared-data records are bound before model allocation.

| Case | Batch × text length | Genuine prefix / suffix | Validation-window starts |
|---|---:|---:|---|
| `b1-l128-p64` | 1 × 128 | 64 / 64 | 2,297,865 |
| `b1-l129-p64` | 1 × 129 | 64 / 65 | 2,298,209 |
| `b2-l512-p128` | 2 × 512 | 128 / 384 | 2,429,961; 3,825,427 |

These windows come from the checkpoint-selection validation pool. They support a paired numerical diagnostic, not an independent language-quality test. Batch two contains two separate 512-token rows; it is not one 1,024-token sequence.

## Exact controls before attribution

The unchanged original full reading repeats exactly. P0 must reproduce the prior measured N1H0 score, stage, memory and genuine-prefix comparison records exactly. An ordinary tolerance pass cannot replace that attribution control.

The earlier negatives remain part of the evidence: 1:15 length 129 and batch-two own-route failures, the batch-two chunked suffix failure at suffix position 79/global text position 207, and the 1:3 batch-two intermediate failure despite passing final scores. These historical anchors come from `normalization-head-summary-2026-10-05.json`. The new P0 measurement reproduces them exactly; P1 is compared separately. A changed or incomplete P0 repeat would leave attribution unverified.

## What is recorded for each cell

Keep all 15 direct final-score endpoints: full, one-token, stateful-full and chunk-128 readings against their applicable own, original, decay-only and P0 references. Seven stage trace pairings compare full and one-token routes against the same direct references. Retain first nonzero drift separately from the first tolerance violation, complete full-stage counts, every failed stage and its bounded coordinate samples.

The seven target-loss endpoints use actual route-produced cross-entropy. The independent CPU local reference must not silently replace an actual CUDA/CPU model loss. Changes in true-target NLL and greedy choices remain descriptive numerical effects, with no declared quality-equivalence margin.

Compare complete convolution, state-scan and attention-cache memory fields against the four stateful anchors; stateful-full memory excludes its identical own anchor. Preserve every failed memory field. Final scores, intermediate stages, target loss and retained memory have separate gates. A local pass cannot erase any of those failures.

The fixed FP32 allowance remains:

`absolute error <= 0.00003 + 0.0003 × absolute reference value`

An error divided by its allowance above 1 fails. This run does not widen the limit in response to a negative.

## Actual operands before whole-model conclusions

A frozen operand is an input, weight or bias actually observed at an operation's boundary and then held fixed for local replay. Capture P0 full-reading inputs and outputs at every registered Mamba input projection: the planned inventories contain 15 sites for 1:15 and 12 for 1:3. Report actual completed capture counts separately; a missing capture supplies no pass.

Replay each observed call using its recorded shape, stride and storage offset. Compare P0 and P1 on the same numerical inputs at a full shape and one-token shapes. Check them against a separately written detached CPU NumPy FP64 matrix multiplication and optional bias addition. The independent reference retains FP64 output for comparison.

Each probe retains eight named checks: observed-layout replay against the observed output, P0 tokenwise against P0 full, P0 full/tokenwise against the FP64 reference, P1 full against P0 full, P1 tokenwise against P1 full, and P1 full/tokenwise against the FP64 reference. It also retains actual call boundaries, input/output/weight fingerprints, unchanged-input checks and bounded errors. All declared 81 full-reading captures and 72 suffix captures completed; planned inventory and actual completed counts remain separate in the viewer.

Numeric operands stay identical in that local replay. Shape and layout are deliberate contrasts. Combining multiple captured calls into one contiguous full replay introduces an explicit layout change; it must not be described as the original observed call. The viewer shows captured scalar counts, actual call counts and position spans rather than inventing activation histories. Full activation arrays are not published.

## Continue from each cell's real prefix

Each cell reads its own prefix and clones that exact memory. Its suffix runs together, in chunks of 128 and one token at a time. Record all three suffix score routes against all four references, route-to-route agreement, complete retained memory, final text positions and the byte-identical original prefix.

Capture the first and last Mamba input projections for each suffix route and cell. Compare their actual upstream inputs and observed projection outputs with the same cell's full-reading values. Then replay each captured suffix input locally under both arithmetic choices. These are two different questions: a route can supply different inputs before projection even when identical-input projection arithmetic agrees.

Suffix and global text positions are zero-based. Add the genuine prefix length to the suffix coordinate to obtain the global position. In the earlier batch-two case, suffix position 79 plus prefix length 128 equals global text position 207. Displayed layer labels are one-based. Keep the batch coordinate beside that position and preserve every failed route/reference; do not reuse another cell's prefix or substitute synthetic state.

## Functions and their limits

| Function | What it does | What it cannot establish |
|---|---|---|
| `projection_arithmetic` | P0/P1 linear arithmetic with original parameters and FP32 output | Whole-model agreement or quality |
| `temporary_projection` | Applies instance-local methods and restores them in `finally` | An accepted production setting |
| `ProjectionCapture` | Captures actual inputs/outputs and ordered call/layout metadata | Semantic meaning of a feature |
| `frozen_projection_probe` | Eight arithmetic checks on identical captured operands | Agreement after differing upstream model calculations |
| `continuation_probe` | Uses genuine prefix clones and records suffix routes, memory and local operands | A synthetic-state conclusion or a distance-only quality effect |
| `reproduction` | Repeats the earlier N1H0 negative evidence exactly | Attribution when the repeat is missing or changed |
| `study_case` | Separates anchors, scores, loss, stages, memory, local replay and integrity | Training qualification from forward-only evidence |

All functions are in `scripts/study_projection_precision.py`. The independent exporter is `scripts/summarize_projection_precision.py`; it validates public records and source bytes without importing the measured producer, Torch, CUDA or private checkpoint/corpus tensors.

## Execution and incomplete records

The single bounded run is forward-only. It executes no gradients, optimizer, BF16, new training or fused Mamba kernel. The cooperative allowance is 900 seconds from runner entry through preflight, model work and final audits. An operation already running can overshoot. Require at least 8 GiB free after model loading before the batch-two case; this is a headroom guard, not a measured peak-allocation promise.

TF32 remains disabled and is restored afterward. Preserve the preceding measured inference policy, random state, tied-parameter identity, source bytes, historical evidence and data/checkpoint identities. Publish a new immutable report. A timeout, nonfinite value, missing headroom or incomplete local probe preserves finished measurements and identifies the interrupted work. Missing or unverified cells supply no positive qualification.

The website lazily loads a reduced validated view. Full-stage counts still include omitted redundant passing stages; every failed stage and bounded error sample remains intact. The view is bound to the complete source-summary fingerprint. Code/evidence links use one actual published model revision; earlier pins remain unchanged.

Expanded gradients, future-token isolation, three-ratio/boundary qualification and a declared resource pilot remain later dependencies. This projection diagnostic alone cannot open the longer training cohorts.

## Recorded resources and immutable evidence

The recorded workflow took **408.64 seconds** under its 900-second cooperative allowance, with zero overshoot. That scope includes runner preflight, model routes, capture/replay and final audits; immutable publication is excluded. These timings are diagnostic resources, not generation throughput or a P0/P1 speed comparison.

| Evidence | Bytes | SHA-256 |
|---|---:|---|
| `checks/projection-precision-2026-10-06/natural-matched-seed-2027.json` | 94,974,553 | `b29e3f05da2ce4337e468d2fa0118f84316262cca66ae74df03b354f4b211635` |
| `projection-precision-summary-2026-10-06.json` | 57,236,325 | `b83243a847a4c418886782d18d3cf07aaa2bb8cfe1d54945a066bd60fdfb1b1b` |
| `projection-precision-view-2026-10-06.json` | 6,782,460 | `35d141304e354aaa4dc96c3ac6f8e4d6a59c15ecaaaead01aabd335ea37ba62e` |
| `checks/projection-precision-2026-10-06/natural-matched-seed-2027.json.gz` | 3,641,586 | `6ada18b7c64061d1f9d36008e60699d1731925dff72fcc23103b32dda81e1e0d` |

The view is copied byte-for-byte into the study site's lazy evidence file. Its reduction changes transport size, not scientific counts or failed samples. The independent exporter uses public records and source text; it does not load private checkpoint/corpus tensors or import the measured producer.

The original raw JSON stays unchanged locally. Its gzip companion is complete lossless transport, bound by `projection-precision-raw-publication-2026-10-06.json`. A public clone can rebuild both exports from that companion and the preceding public evidence without the private checkpoint or corpus files. Gzip-only reconstruction reproduced the existing summary and website view byte-for-byte; dictionary ordering is fixed in this new exporter so a different Python process hash seed cannot change their serialized bytes.

The complete software regression passes 1,222 tests with one existing Windows file-symlink skip in 243.76 seconds. The focused producer has 37 passing CPU checks; its independent exporter has 77. The [verification receipt](projection-precision-verification-2026-10-06.json) binds those checks to the source and evidence hashes. Passing software tests verifies the experiment machinery; the separately recorded numerical failures still prevent policy qualification.
