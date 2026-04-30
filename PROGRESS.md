# Ashforge Roadmap

> Auto-tuned local LLM serving — forged in embers, tuned to perfection.

## Architecture

```
ashforge run <model>
    ├── Hardware Probe (GPU/CPU/RAM detection)
    ├── Model Matcher (GGUF metadata + knowledge base)
    ├── KV Cache Optimizer (f16 → q8_0 → iso3 → q4_0)
    ├── Warmup Benchmark (binary search for optimal context)
    ├── Parameter Tuning (ubatch, threads, mlock)
    └── API Proxy (OpenAI-compatible, streaming, repetition detection)
```

## Current Status (v0.1.0)

### Core Features
- Hardware auto-detection (NVIDIA CUDA, AMD ROCm)
- Model-aware GGUF metadata parsing
- Automatic KV cache type selection
- Warmup-based parameter optimization
- OpenAI-compatible API proxy with streaming
- IDE integration (Cursor, Codex CLI, Claude Code)
- Real-time terminal monitoring dashboard

### Platform Support
| Platform | Status |
|----------|--------|
| Linux x86_64 (CUDA) | Stable |
| Windows x86_64 (CUDA) | Stable |
| macOS Apple Silicon (Metal) | In Development |
| Linux (ROCm/Vulkan) | Experimental |

## Roadmap

### v0.2.0 — Apple Silicon
- [ ] macOS hardware probe (M-series chip detection)
- [ ] Metal GPU backend support
- [ ] Unified Memory optimization
- [ ] macOS binary distribution

### v0.3.0 — Intelligence
- [ ] Auto port discovery (find free ports automatically)
- [ ] Web-based monitoring dashboard
- [ ] Multi-language UI support (EN/CN)
- [ ] Expanded model knowledge base

### v0.4.0 — Community
- [ ] Plugin system for custom optimizers
- [ ] Model recommendation engine
- [ ] Performance leaderboard

## Tech Debt
- Warmup profile needs version stamping for cache invalidation
- Local model scan should be cached, not rescanned every invocation
- Warmup speed can diverge from production speed at very large contexts
