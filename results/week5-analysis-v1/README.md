# Week 5 analysis

This report joins certified Week 4 evaluation with protocol-matched Week 5 sampled generation.

| Ratio | PPL | GPU generation tok/s | TTFT (ms) | Peak VRAM (MiB) | State near 57 tokens (MiB) | State at 8K (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 1:3 | 26.301 | 52.32 | 20.3 | 238.5 | 5.72 | 61.33 |
| 1:7 | 26.466 | 48.26 | 21.3 | 244.6 | 6.41 | 34.22 |
| 1:15 | 26.513 | 48.28 | 23.1 | 246.8 | 6.76 | 20.66 |

## Findings

1. **Quality and long-context memory form the clearest trade-off.** 1:15 saves 66.3% logical state at 8K while adding 0.212 perplexity versus 1:3.
2. **Short sampled generation favors 1:3 on this implementation.** Its median end-to-end rate is 8.4% above 1:7. The 1:7 and 1:15 rates differ by only 0.04% in this nine-sample protocol.
3. **State ordering reverses with context.** Near 57 cached tokens, 1:3 uses 15.4% less logical state than 1:15 because each Mamba layer owns a fixed recurrent state. Their calculated curves cross near 260 tokens; beyond that, attention KV growth dominates.
4. **Local CPU serving is viable for this narrow demo.** The Ryzen 7700 reaches 77.3% of the RTX 5070's 1:3 end-to-end rate under the matched 48-token protocol. This is a local result, not a claim about a two-core cloud host.
5. **Generation quality remains the limiting factor.** The samples are locally coherent but often drift, repeat phrases, or make unsupported factual statements. The demo therefore presents the model as a controlled systems experiment, not a production assistant.

All findings are descriptive: one training seed and three generation prompts are not statistical evidence of a universal ratio ordering.
