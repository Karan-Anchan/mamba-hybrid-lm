# week5-generation-cuda-v1

Measured sampled generation from the certified Week 3 checkpoints. Speed includes prompt prefill,
sampling, and recurrent decode; decode speed is reported separately. Each completion is retained in JSON.

| Ratio | Samples | End-to-end tok/s | Decode tok/s | TTFT (s) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| 1:3 | 3 | 52.32 | 52.36 | 0.020 | 238.5 |
| 1:7 | 3 | 48.26 | 48.27 | 0.021 | 244.6 |
| 1:15 | 3 | 48.28 | 48.35 | 0.023 | 246.8 |

Device: `NVIDIA GeForce RTX 5070`. Git commit: `d6a4613653b4b5259342c3340e28a1be7231c632`.
