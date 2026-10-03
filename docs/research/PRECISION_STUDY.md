# Precision study — 4 October 2026

Several experimental precision settings improve full-sequence/cached agreement in this small CUDA study, but **every BF16 treatment still fails the complete comparison with its common FP32 control in all 15 tested cases per architecture**. These settings remain diagnostic experiments. The production defaults, historical checkpoints and original tolerances are unchanged.

This record explains the method, the results and the next checks needed before changing numerical execution. It adds no language-quality, speed or fused-kernel result. Read the [implementation record](IMPLEMENTATION.md) for the earlier backend and recovery work, inspect the [declared protocol](precision-protocol-2026-10-04.json), compare the [validated summary](precision-summary-2026-10-04.json), and open the [six raw CUDA reports](checks/precision-2026-10-04/) for every comparison.

## Why investigate precision?

The earlier pure-Mamba BF16 gate found differences between processing a sequence together and processing it through saved memory. Both routes implement the same mathematical update. Their floating-point operations can nevertheless round values differently. We need to locate those differences before using a new route for a research campaign.

The historical scan converts its inputs to FP32. That conversion does not force every later operation to use FP32 while PyTorch autocast is active. The new reports observe a matrix contraction with FP32 inputs producing BF16 output. The final scan output and stored recurrent state are FP32. Storage precision and operation precision answer different questions.

| Term | Beginner explanation |
|---|---|
| Token | One numbered piece of text. This diagnostic uses random token IDs, rather than meaningful prompts. |
| Logits | The model's scores for possible next tokens, before turning them into probabilities. |
| FP32 / BF16 | Two ways to store numbers. FP32 keeps more numerical detail; BF16 uses fewer bits. |
| Autocast | PyTorch's rule for choosing the precision of individual operations inside a region of code. |
| SSM / attention | A state-space model carries a compact memory forward. Attention reads stored representations of earlier tokens. |
| Scan / SSD | Code that applies Mamba's sequence update. Structured State Space Duality gives a parallel matrix form of that update. |
| State / cache | Saved information used when the next token arrives: Mamba convolution and recurrent memory, or attention keys and values. |
| Convolution | A calculation over a short neighboring window; its saved tail keeps recent inputs available. |
| Tensor | An array of numbers, with dimensions such as batch, sequence position and feature. |
| Prefill / decode | Load the prompt into saved memory, then process additional tokens using that memory. |
| Projection | A learned multiplication that converts one vector into another. The language-model head projects a hidden vector into token scores. |
| Gradient | How much a computed result changes when an input or learned value changes. Training uses these derivatives to adjust weights. |
| Optimizer / checkpoint | An optimizer changes learned weights using gradients. A checkpoint saves a recoverable model and training state. |
| NaN / infinity | Numerical values that cannot be used as valid finite measurements. NaN means “not a number.” |
| TF32 | A GPU format used by some FP32 matrix multiplications. The recorded CUDA matrix setting disables it here. |
| Seed | A number used to repeat a sequence of random draws. |
| Isolation | Hold relevant values fixed and change one part of the calculation to test a specific explanation. |
| Parity / tolerance | Agreement within an allowed numerical difference; it does not require bit-for-bit equality. |
| Anchor / oracle | An anchor is a shared comparison run. Here, the oracle is a separately written sequential FP32 update used to check scan calculations. It still has finite rounding error. |
| Hash / provenance | A hash identifies exact file or tensor contents. Provenance records which code, settings, hardware and inputs produced a result. |

## Fixed protocol

Six reports cover two architectures and three initialization seeds, 1337, 2027 and 3141. Pure Mamba has attention:SSM ratio 0:1, with four Mamba layers; the 1:3 hybrid has three Mamba layers and one attention layer. Each tiny model has width 64, vocabulary 128, recurrent state size 16 and batch size 2. Width counts features in a hidden vector; vocabulary counts possible token IDs; a batch groups two sequences for one execution. The lengths are 1, 15, 16, 17 and 33, with chunk size 16. Lengths around 16 test behavior immediately before, at and after a chunk boundary.

Each architecture/seed report runs five precision settings on copies of the **same weights and input tokens**. There are 25 model cases per report and 150 across the six reports. No optimizer runs. Weight hashes are checked after the experiments. Target-token draws are retained to reproduce the earlier gate's random-draw order; this study does not use them to measure training loss or language quality.

The backend is `reference`: full forward uses the historical quadratic PyTorch SSD; stateful prefill and decode use bounded PyTorch SSD. This matrix does not run `torch_chunked` as a separate full-model backend treatment. No official fused Mamba kernel executes.

The recorded runtime is Python 3.11.9, PyTorch 2.11.0+cu128, CUDA runtime 12.8 and an RTX 5070. CUDA matrix TF32 is disabled and float32 matmul precision is `highest`; the reports also retain cuDNN and determinism flags. These are diagnostic settings, so the runtime records matter when reproducing a comparison.

The thresholds remain those declared before the original checks: FP32 `atol=0.00003, rtol=0.0003`; BF16 `atol=0.002, rtol=0.02`. Each element must satisfy:

```text
absolute difference <= atol + rtol × absolute reference value
```

The reports retain failures without widening these thresholds. All six say `execution_status: completed`, `status: completed_with_parity_failures` and `certified: false`. A completed diagnostic means the measurements finished; it does not approve a backend or treatment.

## Two questions, ten result rows

**Internal agreement** asks whether a precision setting agrees with itself when execution changes: full forward, bounded prefill, prefix prefill followed by tokenwise suffix decode, and tokenwise execution from an empty state. It includes retained convolution, recurrent and attention memory plus the tested causality check.

**FP32-anchor agreement** asks whether the BF16 setting's logits and retained memory agree with the matching FP32 reference paths. This is a separate question. A treatment can make two BF16 routes agree while both remain different from FP32.

Each row below contains three seeds × five lengths = 15 cases. The projection treatment disables autocast for **every Mamba `out_proj` and the language-model head**, while leaving input projections, convolution, attention and MLP projections outside that intervention. The combined treatment changes both factors and is an exploratory interaction check.

| Architecture | Precision setting | Internal agreement | Complete FP32-anchor agreement |
|---|---|---:|---:|
| Pure Mamba 0:1 | FP32 reference | 15/15 | Reference |
| Pure Mamba 0:1 | Original BF16 | 4/15 | 0/15 |
| Pure Mamba 0:1 | BF16, scan autocast disabled | 15/15 | 0/15 |
| Pure Mamba 0:1 | BF16, output-projection autocast disabled | 15/15 | 0/15 |
| Pure Mamba 0:1 | BF16, scan and output-projection autocast disabled | 15/15 | 0/15 |
| Hybrid 1:3 | FP32 reference | 15/15 | Reference |
| Hybrid 1:3 | Original BF16 | 4/15 | 0/15 |
| Hybrid 1:3 | BF16, scan autocast disabled | 15/15 | 0/15 |
| Hybrid 1:3 | BF16, output-projection autocast disabled | 14/15 | 0/15 |
| Hybrid 1:3 | BF16, scan and output-projection autocast disabled | 15/15 | 0/15 |

“0/15” means every case fails at least one required anchor comparison. It does not mean every individual tensor comparison fails. The raw `case.passed` flag also requires layer-stage and hidden-input comparisons; all BF16 cases remain false because of the anchor requirement. These 15 cases are a bounded check across three random initializations and correlated sequence lengths, not independent trained replicas or a confidence interval.

The original BF16 baseline fails under this broader protocol in both architectures. That does not replace the earlier hybrid gate's pass: that gate tested prefix prefill plus the last three decoded tokens, backend outputs and parameter gradients. The new diagnostic adds tokenwise execution from empty memory across the whole prompt, full-prompt prefill, more seeds, layer-stage checks and a common FP32 anchor. In the new hybrid seed-1337 report, the prefix/suffix comparisons still pass; lengths 15/16/17/33 fail whole-prompt tokenwise logits and retained memory. A pass on a narrower check does not guarantee a pass on a broader one.

## Freeze the operands: what changes inside the scan?

The exploratory scan probe takes the first Mamba layer's projected arrays from the original BF16 full forward, freezes them, and reuses those exact values. It compares a one-shot scan, bounded chunks, tokenwise execution and the independent FP32 recurrence. It tries both empty memory and a reproducible synthetic nonzero memory. This removes changes caused by recomputing upstream projections; it cannot recover precision already lost in those frozen values.

| Frozen-scan checks | Pure Mamba | Hybrid 1:3 |
|---|---:|---:|
| All required checks with BF16 autocast active | 4/15 | 3/15 |
| Nonzero-memory tokenwise output versus one-shot output | 4/15 | 7/15 |
| Nonzero-memory bounded output versus FP32 recurrence | 7/15 | 8/15 |
| All required checks with scan autocast disabled | 15/15 | 15/15 |

All zero-memory output checks pass 15/15 in each architecture with autocast active. All tested retained-state comparisons also pass 15/15, including nonzero-memory continuation. The failures concern some outputs from nonzero-memory continuation. Correct-looking saved memory is therefore insufficient evidence that every emitted output agrees.

These fixed-input checks demonstrate precision-sensitive scan outputs on the tested operands. They do not identify a universal cause for full-model mismatches or show that synthetic memory represents every trained checkpoint's state.

The direct-gradient probe also freezes the operands and includes nonzero initial memory. Its objective uses both outputs and final memory, so a detached state would affect the derivative check. Derivatives with respect to `x`, `dt`, `A`, `B`, `C`, `D` and initial memory agree with the separate FP32 recurrence in **15/15 cases per architecture**, with autocast both active and disabled. This result covers the direct bounded scan, not full-model parameter gradients or training stability.

For lengths above one, the probes change only the final input token or scan input row and verify earlier outputs. They also require exactly zero gradients from earlier outputs to the final row of `x`, `dt`, `B` and `C`. These checks pass in all 12 eligible cases per architecture and precision. They test final-token causality only; they do not cover every cut position or every continuation path.

## Freeze the hidden vectors: what changes inside a projection?

The projection probe takes the same hidden vectors and weights, then applies the first Mamba output projection or language-model head to the whole sequence, chunks and individual tokens. It holds the upstream scan fixed. Both projections pass **15/15 cases per architecture**, with autocast active and disabled.

This provides no observed same-input projection-shape failure in this small matrix. It does not rule out a projection amplifying small differences that arrive from earlier operations. The full-model projection treatment improves internal agreement, but that improvement alone cannot identify the original cause. Its scope is two kinds of projection, and the combined treatment changes two factors at once.

## Walk through the code

The diagnostic lives in [`scripts/study_scan_precision.py`](../../scripts/study_scan_precision.py):

1. **`run_study`** validates the bounded protocol, protects the caller's random streams, builds or strictly loads a model, and hashes weights, source files and inputs. It draws each input once, copies the model for each treatment, records an FP32 anchor and assembles comparisons plus exploratory probes. It verifies that weights stayed unchanged. Optional checkpoint loading is read-only; the six reported runs use random models.
2. **`experimental_precision`** temporarily wraps the scan and selected projections on that model instance. It records operation dtypes, executed paths and intermediate values. Scan autocast can be disabled separately from Mamba output-projection/LM-head autocast. A `finally` block removes hooks and restores wrappers even when a diagnostic raises.
3. **`_evaluate`** runs full forward; prefill of the entire prompt, returning its final-token logits; prefix prefill plus up to three tokenwise suffix steps; and tokenwise execution of the complete sequence from empty memory. It snapshots retained memory, checks finite values and expected logit shapes, and perturbs the final token for the limited causality check. The comparison helper keeps internal agreement and FP32-anchor agreement separate.
4. **`recurrence_oracle`** uses FP32 elementwise products and sums rather than the SSD matrix contractions. At each token it decays old memory, adds the current input contribution, and reads the new memory into an output. It implements the same update through a different computational route; it is a numerical control rather than an exact real-number answer.
5. **`scan_isolation`, `gradient_isolation` and `projection_isolation`** reuse frozen values to investigate one part of execution. The direct-gradient objective includes final memory. Projection replay uses identical hidden vectors, so it does not silently substitute an upstream full-model contrast.

The scan's short variable names describe parts of this update:

| Name | Role in the recurrence |
|---|---|
| `x` | Current input features. |
| `dt` | The step size that scales the current input and memory decay. |
| `A` | The learned decay parameter controlling how old memory fades. |
| `B` | How the current input writes into recurrent memory. |
| `C` | How recurrent memory is read into the current output. |
| `D` | A direct contribution from the current input to the output. |

The independent recurrence performs `memory = decay × old_memory + current_input_contribution`, then reads that memory and adds the direct input contribution. The gradient probe checks both those output effects and the final saved memory.

[`scripts/summarize_precision_study.py`](../../scripts/summarize_precision_study.py) validates the declared matrix and raw report structure, settings, identities and outcomes before deriving compact tables. It performs no model execution. It keeps internal counts, anchor counts and exploratory probes separate, and records raw-file hashes so the exported data can be traced back to the evidence.

The training safeguards live in [`src/train/train.py`](../../src/train/train.py):

1. **`estimate_loss`** temporarily switches the model to evaluation mode and repeatedly samples the same evaluation windows without advancing the training sampler. It checks every batch loss and the aggregate for NaN or infinity, naming the split and batch on failure. The original finite-result arithmetic stays unchanged. A `finally` block restores the model's prior training/evaluation mode on success or error.
2. **The evaluation publication step** computes validation perplexity, `exp(validation loss)`, before appending an evaluation metric or replacing `best.pt`. An overflowing or nonfinite value raises `FloatingPointError` first. A very large finite loss can overflow this conversion, so checking loss alone is insufficient. Perplexity is a prediction-difficulty score; this safeguard creates no new quality measurement.
3. **Configuration checks** reject malformed integer budgets/intervals and nonnumeric, Boolean, nonfinite or out-of-range optimizer scalars before opening the run lock or creating training artifacts. Zero training steps and zero warmup remain valid. No optimizer grouping or numerical schedule change accompanies these checks.
4. **Recovery** retains the prior durable `last.pt` and `best.pt` after a rejected evaluation. An update completed before that evaluation may have changed the live model; recovery restarts from the last checkpoint, restoring weights, optimizer, counters and random streams, and reconciling later metrics. Resume requires matching numerical settings, data, runtime and source identity. Changed code needs a fresh run namespace; these safeguards do not authorize resuming historical runs under new code.

Failure-injection tests in [`tests/test_training_reliability.py`](../../tests/test_training_reliability.py) check mode restoration, invalid batch/aggregate values, perplexity overflow, unchanged durable artifacts and recovery. This record does not substitute a diagnostic pass count for the final repository test result.

## Decisions and their beginner reasons

| Decision | Reason |
|---|---|
| R-015 — Validate evaluation/settings before publishing, restore model mode, preserve durable recovery. | A broken measurement must not become a “best” result. Recover from a saved point whose progress and random state agree. |
| R-016 — Keep weights and tolerances fixed; isolate scan/projection arithmetic with frozen values and direct gradients. | Changing several hidden ingredients makes an explanation hard to test. Reusing exact values helps locate a difference. |
| R-017 — Compare all three historical showcase models on the same recorded prompt; fill attention markers and label unmeasured baselines. | Readers can see the architecture and responses together. A baseline configuration without a measured result must not appear in a quality ranking. This is a presentation decision, not a precision-study finding. |
| R-018 — Separate internal agreement from FP32-anchor agreement, retain anchor failures and do not promote defaults. | Two routes can agree with each other while both differ from the control. Approval needs more shapes, trained checkpoints and full-model checks. |

## Reproduce and inspect

Run from the model repository with the existing environment. Use a **new output path**; the diagnostic refuses to overwrite a report.

```powershell
.\.venv\Scripts\python.exe scripts/study_scan_precision.py `
  --device cuda --ratio 0:1 --seed 1337 --backend reference `
  --chunk-size 16 --batch-size 2 --lengths 1 15 16 17 33 `
  --output outputs/precision-recheck/pure-mamba-seed-1337.json
```

Repeat with ratios `0:1` and `1:3`, and seeds `1337`, `2027` and `3141`, using distinct output paths. CUDA must be available. A CPU run is a different runtime comparison. Optional `--checkpoint <path>` strictly loads an existing checkpoint without changing it; it is a new study protocol, not a replacement for these random-model records.

The CLI returns zero when measurements complete, including measured parity failures. Check `status`, the comparison fields and `certified`; exit status alone does not mean numerical approval.

Build a fresh summary from the six committed raw reports without rerunning a model:

```powershell
$precisionReports = Get-ChildItem docs/research/checks/precision-2026-10-04/*.json |
  ForEach-Object { $_.FullName }
.\.venv\Scripts\python.exe scripts/summarize_precision_study.py `
  --protocol docs/research/precision-protocol-2026-10-04.json `
  --reports $precisionReports `
  --output outputs/precision-recheck/summary.json
```

The protocol requires exactly the declared six architecture/seed reports. Additional or missing files cause validation to fail. The summary reproduces measured failures; it does not certify them.

## What would support a numerical change?

Next checks should retain the thresholds and compare new interventions on held-out trained checkpoints, production dimensions and chunk size 128, including lengths 127/128/129 and longer uneven chunks. Use genuine nonzero continuation states as well as synthetic states; perturb multiple future suffixes at cuts across chunk boundaries; compare retained state and full-model parameter gradients. Report each intervention's runtime and memory costs under a fixed protocol.

The current evidence supports investigating scan precision and upstream difference propagation. It does not select a universal cause, approve a BF16 replacement, establish attention as a stabilizing mechanism, or rank architecture quality. Pure and hybrid initialization consume random draws differently, so sharing a seed does not isolate the effect of adding attention. Training campaigns and actual fused-kernel validation remain separate work.
