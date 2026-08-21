# week5-generation-cpu-matched-v1

Measured sampled generation from the certified Week 3 checkpoints. Speed includes prompt prefill,
sampling, and recurrent decode; decode speed is reported separately. Each completion is retained in JSON.

| Ratio | Samples | End-to-end tok/s | Decode tok/s | TTFT (s) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| 1:3 | 3 | 40.44 | 40.52 | 0.028 | n/a |

Device: `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`. Git commit: `4d56338d87dbd0dad59a5f65893a7be70966c48b`.
