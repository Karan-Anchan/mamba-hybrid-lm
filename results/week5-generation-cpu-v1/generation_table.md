# week5-generation-cpu-v1

Measured sampled generation from the certified Week 3 checkpoints. Speed includes prompt prefill,
sampling, and recurrent decode; decode speed is reported separately. Each completion is retained in JSON.

| Ratio | Samples | End-to-end tok/s | Decode tok/s | TTFT (s) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| 1:3 | 3 | 41.56 | 42.37 | 0.032 | n/a |

Device: `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`. Git commit: `61d14375bd1169aa0bb68a7954d352ad81696aee`.
