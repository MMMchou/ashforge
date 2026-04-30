# Ashforge Benchmark Suite

## Overview

The benchmark suite (`benchmark/bench.py`) provides reproducible performance comparisons between Ashforge and other local LLM serving tools.

## Usage

```bash
# Install dependencies
pip install requests psutil

# Run benchmarks
python benchmark/bench.py 8b-ashforge    # Test 8B model on Ashforge
python benchmark/bench.py 30b-ashforge   # Test 30B model on Ashforge
python benchmark/bench.py all            # Run all tests
```

## Metrics Collected

- **tok/s** — Generation throughput (via `usage.completion_tokens / elapsed`)
- **VRAM Peak** — Maximum GPU memory during inference
- **CPU %** — CPU utilization during generation
- **RAM** — System memory usage

## Configuration

Edit the constants at the top of `bench.py`:
- `PROMPT` — Test prompt
- `MAX_TOKENS` — Maximum tokens to generate
- `RUNS` — Number of runs per test (median is reported)
