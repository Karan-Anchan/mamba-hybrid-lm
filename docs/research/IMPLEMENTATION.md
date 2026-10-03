# Research implementation record — 3 October 2026

The first implementation increment makes failures easier to detect and controlled experiments easier to prepare. It adds no new language-quality result. The completed 700M-token study and its checkpoints remain the historical evidence.

## What changed, and why

| Change | Beginner explanation | Check |
|---|---|---|
| Explicit configuration checks | A model is built from arrays of a particular size. Invalid sizes or layer names now stop with a useful message, even when Python runs with optimization enabled. | Dimensions, ratios, unsupported groups, layer patterns and Python `-O` tests. |
| Safe mixer memory updates | A mixer checks the input and saved memory before using them. It writes new memory only after its output succeeds. | Invalid-state and injected output-failure tests; convolution width 1; uneven chunks. |
| Finite training checks | NaN means “not a number”; infinity is also unusable here. A bad loss or gradient now stops before the optimizer changes weights or progress counters. | Failure injection, accumulated-loss overflow, optimizer/LR preservation and recovery tests. |
| Independent random streams | Change how the model starts while keeping its training text and development examples the same. | Initialization, training-window and evaluation-window separation tests. |
| Explicit scan engines | An engine is the code that performs the sequence update. Its name and intended/observed paths are recorded separately. | Same-weight output, gradient, tokenwise decode and retained-state checks. |
| Configurable campaign preparation | Verify every experiment arm and its budget before starting any of them. A dry-run prints a plan without creating training artifacts. | Duplicates, malformed settings, namespace changes, config hashes and dry-run tests. |
| Cancellation-safe model access | Closing a browser request does not immediately stop its background thread. That worker keeps the model lock until it actually finishes. | Real cancelled JSON/SSE requests followed by a competing ratio switch. |

Mixer memory protection applies to one mixer. If a later layer fails, earlier layers may already have updated their caches. Discard that whole inference state after a model-level failure. Shutdown waits for active generation workers; a hung Python thread cannot be safely killed by this mechanism.

## Scan choices

| Choice | Training | Prefill/decode | Current status |
|---|---|---|---|
| `reference` | Historical full-sequence quadratic PyTorch SSD | Bounded PyTorch SSD | Default. Existing numerical execution is retained. |
| `torch_chunked` | Differentiable bounded PyTorch SSD | Bounded PyTorch SSD | Available on CPU/CUDA. Portable, not a fused Mamba kernel. |
| `fused_mamba` | Official `mamba_chunk_scan_combined` adapter | The same official scan with recurrent memory | Adapter implemented; actual execution unavailable on this Windows environment. |

Chunking breaks the sequence into smaller pieces and carries memory between them. The largest local decay matrix is bounded by chunk size. Projection tensors and the training graph still grow with sequence length; this is not a constant-memory claim or a measured speedup. Training memory remains attached to autograd, the system that calculates gradients. Detaching it would silently change learning.

Backend selection adds no fields to `ModelConfig` and no new parameter keys to `state_dict`. A requested fused engine never silently falls back. The adapter keeps the local projection, convolution, gate and RMS normalization. It adds the one-group dimension to B/C and asks for FP32 recurrent state. `dt` already includes its bias and softplus transformation; applying them again would change the model. API, device, output shape and state checks fail explicitly. The adapter still needs real kernel execution, gradient, checkpoint, quality and performance validation in a supported environment.

`configure_scan_backend` chooses the default chunk size. Portable mixer calls can explicitly override it for stateful inference; fused per-call overrides are rejected. Metadata describes configured paths. The trainer reports observed training only after successful updates through Mamba layers, and leaves unexecuted prefill/decode fields null. Pure-attention arms report no executed Mamba scan.

## What the checks actually found

These are software checks on small random models, not new research conclusions. The tiny CUDA gate uses a four-layer, width-64 model, seed 1337, chunk size 16 and lengths 1/15/16/17/33. It compares logits, loss, every parameter gradient, prompt loading, tokenwise decode and retained convolution/SSM/KV tensors.

| Recorded check | Outcome | Meaning |
|---|---|---|
| Hybrid 1:3, FP32, chunked versus reference | Passed | Agreement on the tested shapes under absolute tolerance 0.00003 and relative tolerance 0.0003. |
| Hybrid 1:3, BF16, chunked versus reference | Passed | Agreement on the tested shapes under absolute tolerance 0.002 and relative tolerance 0.02. |
| Pure Mamba, FP32, chunked versus reference | Passed | The tested endpoint agrees in FP32. |
| Pure Mamba, BF16, reference cached versus full-sequence | Failed | Cached/parallel logits exceed the declared tolerance at some tested positions. |
| Pure Mamba, BF16, chunked versus reference | Failed | Some output comparisons also exceed tolerance; this endpoint is not approved for a BF16 campaign by these checks. |
| Fused Mamba on this machine | Unavailable | Supported Linux environment is absent; no fused numerical or speed certification. |
| Three registered historical checkpoints, CUDA FP32, 17 tokens | Passed | Registered hashes, strict loading, parameter keys, short prefill/decode and bounded-forward compatibility hold. |
| Nine-arm initialization-seed pilot dry-run | Prepared, not executed | Three ratios × three seeds, 32,768,000 token positions per arm; 294,912,000 total. |

The failures remain in [the JSON check records](checks/). Tolerances were declared before execution and were not widened after failure. A tolerance passes when every absolute difference is at most `atol + rtol × abs(reference)`. Small scores therefore rely mostly on the absolute tolerance.

BF16 rounding is a follow-up hypothesis, not a proven cause. The historical SSD converts inputs to float32, but active PyTorch autocast may execute matrix contractions at lower precision. FP32 recurrent storage alone does not make every operation FP32. The next precision study should compare autocast settings, accumulation paths, several seeds and larger shapes, then verify actual held-out checkpoints. Any numerical fix gets a new protocol; historical results are not relabeled.

## Follow the new functions

1. **`ModelConfig.__post_init__` / `build_layer_pattern`** receive dimensions and an attention:SSM ratio. They reject malformed input and build an ordered list of layers. `realized_ratio` reports actual counts; a partial repeat or an explicit placement can differ from the requested ratio.
2. **`Mamba2Mixer.init_state` / `_validate_forward`** create and check convolution memory plus FP32 recurrent memory. For a batch of text, array dimensions, device and projected dtype must agree. A mismatch raises before memory changes.
3. **`resolve_scan_backend`** receives an engine name and chunk size. Portable engines resolve directly. Fused selection checks the environment and required official API fields. Success means the operator was located, not that its numerical gate passed.
4. **`HybridLM.configure_scan_backend`** resolves the engine before touching any layer. It assigns the resolved engine to each Mamba mixer and returns descriptive metadata. A failed resolution leaves the previous selections intact.
5. **`ScanBackend.scan`** receives the projected arrays and optional initial memory. The reference path keeps its historical full-sequence training rule. The bounded path loops over chunks and returns the final differentiable memory. The fused path checks its returned arrays before they can enter inference state.
6. **`_run_locked`** computes each microbatch loss, checks it, runs backward, checks the accumulated loss and gradient norm, then clips and updates. Learning-rate and progress changes follow successful checks. Invalid work leaves the last durable checkpoint available for recovery.
7. **`make_batch_generator` / seed fields** separate model initialization, training-window sampling and repeated development evaluation. Leaving new seed fields unspecified preserves the original single-seed policy. Explicit initialization seeds hold shared data/evaluation streams fixed for paired contrasts.
8. **`prepare_matrix` / sweep `main`** verify all configurations, vocabulary/window fit, unique model names and seeds. The matrix contains content hashes, realized counts, namespaces and token budgets. Changing configuration, backend or chunk size cannot reuse the existing signed campaign identity.
9. **`check_scan_backend.run_gate` / `compare`** construct identical small models, run both engines on identical inputs, and compare finite arrays with identical shapes. JSON records contain model/script hashes, package versions, precision flags, actual paths and outcomes. An unavailable engine or runtime failure exits nonzero.
10. **`check_checkpoint_compatibility.main`** verifies registered hashes, loads each old model strictly, and runs a short read-only FP32 check. It neither updates weights nor replaces long-context/quality evaluation.
11. **`_start_generation_worker`** creates a task whose background work owns the model lock. Request handlers shield that task from client cancellation. A completion callback releases tracking and retrieves unobserved worker errors; shutdown drains remaining workers.

## Reproduce checks and inspect a campaign

Run from the model repository using its existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_scan_backend.py --backend torch_chunked --device cuda --dtype float32
.\.venv\Scripts\python.exe scripts/check_scan_backend.py --backend torch_chunked --device cuda --dtype bfloat16 --ratio 0:1
.\.venv\Scripts\python.exe scripts/check_checkpoint_compatibility.py
```

The pure-Mamba BF16 command is expected to fail this recorded gate. This negative result is part of the evidence.

```powershell
.\.venv\Scripts\python.exe scripts/run_sweep.py `
  --data-dir data/openwebtext-5b --run-id research-pilot-plan-v1 `
  --max-steps 2000 --warmup-steps 40 `
  --model-seeds 1337 2027 3141 --data-seed 1337 --eval-seed 1337 `
  --scan-backend reference --dry-run
```

The corpus is verified even in dry-run. The command records equal tokens, not equal GPU time or FLOPs. Seeds receive distinct checkpoint namespaces. Add `--configs configs/attention_only.yaml configs/mamba_only.yaml` to inspect the pure endpoints. They share depth/width/tokenizer geometry, but their parameter counts differ; their later comparison must report that confound. The pilot budget is a preparation example, not a completed or statistically powered campaign.

## Decisions for this increment

- **R-009 — Keep the default and checkpoint schema.** Existing weights are useful anchors. Runtime engine selection makes implementation contrasts possible without rewriting them.
- **R-010 — Keep failed numerical gates.** Failure is information about the tested protocol. Widening the threshold to obtain a green label would hide that information.
- **R-011 — Pair data while changing initialization.** It makes the first seed study easier to interpret. A separate later study can vary sampled training text.
- **R-012 — Let workers own model access.** A disconnected client and a stopped computation are different events. Switching ratios early could unload a model still in use.
- **R-013 — Use minimal research presentation on both sites.** Remove decoration that competes with the measurements. Preserve semantic data colors, controls, charts, recorded/live labels and readable explanations.
- **R-014 — Prepare, then pilot.** This increment adds tools and check records. Kernel certification, synchronized timing, retrieval expansion, similarity analysis and replicated quality conclusions remain later work.

The study website explains these choices and functions for a beginner. The separate showcase retains result-focused diagrams, interactive ratio controls and recorded/live generation behavior. Historical function links remain pinned to the version that produced those results; new implementation links identify this increment separately.
