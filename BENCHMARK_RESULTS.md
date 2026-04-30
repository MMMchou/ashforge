# Ashforge Performance Report

## Methodology

Ashforge uses a multi-phase warmup benchmark to find optimal parameters:

1. **Coarse Search** — Power-of-2 context sizes, measures tok/s at each level
2. **Fine Search** — Binary search between best coarse points
3. **Ubatch Tuning** — Tests multiple ubatch sizes at the winning context
4. **Profile Caching** — Results saved to `~/.ashforge/profiles/` for instant startup

## Key Optimizations

| Technique | Impact |
|-----------|--------|
| MoE Expert Offload | Attention on GPU, experts on CPU — 2-3x speedup for MoE models |
| iso3 Quantization | TurboQuant KV cache — 20-40% VRAM savings on Ampere+ |
| Adaptive Context | Auto-sizes context to maximize speed above useful threshold |
| Smart Thread Count | Measured, not guessed — prevents CPU over-subscription |

## Run Your Own Benchmarks

```bash
ashforge bench          # Quick benchmark of current model
ashforge run <model>    # Includes warmup benchmark on first run
```

Results are hardware-specific and cached per GPU fingerprint.
