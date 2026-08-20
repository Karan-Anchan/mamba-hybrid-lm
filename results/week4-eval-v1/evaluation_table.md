# Week 4 comparison

All speed and state-memory columns below use the 8,192-token context. 
Logical tensor memory is kept separate from CUDA allocator deltas.

| Ratio | Validation PPL | Prefill tok/s | Decode tok/s | Attention KV MiB | Mamba state MiB | Total state MiB | Needle exact match |
|:--:|--:|--:|--:|--:|--:|--:|--:|
| 1:3 | 26.301 | 14,143.1 | 46.8 | 56.00 | 5.33 | 61.33 | 3/15 (20.0%) |
| 1:7 | 26.466 | 12,126.7 | 49.0 | 28.00 | 6.22 | 34.22 | 3/15 (20.0%) |
| 1:15 | 26.513 | 11,876.2 | 44.6 | 14.00 | 6.66 | 20.66 | 3/15 (20.0%) |

The Mamba state column is recurrent convolution plus SSM state and is constant with context length. Attention KV grows linearly with the number of cached tokens and attention layers.
