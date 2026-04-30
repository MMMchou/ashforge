<div align="center">

# ASHFORGE

**Auto-tuned local LLM serving — forged in embers, tuned to perfection.**

Same model. Faster speed. Zero manual tuning.

[Features](#features) · [Install](#installation) · [Quick Start](#quick-start) · [How It Works](#how-it-works) · [中文](#中文)

</div>

---

## Why Ashforge?

LM Studio and Ollama make models run. Ashforge makes them run **well** — by measuring, not guessing.

```bash
ashforge run Qwen3-30B-A3B
```

That's it. Ashforge probes your hardware, reads the model architecture, benchmarks KV cache options, and finds the optimal configuration automatically. The result is cached — second launch takes 2 seconds.

## Features

- **Hardware Auto-Detection** — GPU (NVIDIA CUDA, Apple Silicon Metal), CPU, RAM
- **Model-Aware Optimization** — reads GGUF metadata, detects MoE/hybrid architectures
- **KV Cache Selection** — f16 → q8_0 → iso3 → q4_0 based on available VRAM
- **Warmup Benchmark** — binary search for optimal context length at target speed
- **MoE Expert Offload** — keeps attention on GPU, routes experts to CPU
- **OpenAI-Compatible API** — `http://localhost:21435/v1` with streaming
- **Repetition Detection** — auto-stops loops via n-gram + pattern analysis
- **Context Compression** — extractive summary when context fills up
- **IDE Integration** — one command setup for Cursor, Codex CLI, Claude Code
- **TUI Monitor** — real-time terminal dashboard (VRAM, speed, context usage)
- **Multi-Language** — auto-detects system language (EN/中文)
- **Smart Port Discovery** — auto-finds available port if default is taken

### Platform Support

| Platform | GPU | Status |
|----------|-----|--------|
| Linux x86_64 | NVIDIA CUDA | Stable |
| Windows x86_64 | NVIDIA CUDA | Stable |
| macOS arm64 | Apple Silicon Metal | Beta |
| macOS x86_64 | Intel (CPU) | Supported |
| Linux x86_64 | AMD ROCm / Vulkan | Experimental |

## Installation

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/MMMchou/ashforge/main/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/MMMchou/ashforge/main/install.ps1 | iex
```

Or download from [Releases](https://github.com/MMMchou/ashforge/releases).

## Quick Start

```bash
# Run a model (auto-downloads if needed)
ashforge run Qwen3-30B-A3B

# Run a local GGUF file
ashforge run /path/to/model.gguf

# List available models
ashforge list

# Check hardware
ashforge probe

# Stop the server
ashforge stop
```

Connect any OpenAI-compatible tool to `http://localhost:21435/v1`.

## How It Works

```
ashforge run <model>
    │
    ├── 1. Probe Hardware
    │       GPU model, VRAM, bandwidth, SM version, CPU cores, RAM
    │
    ├── 2. Read Model
    │       Architecture, layers, KV heads, context limit, MoE structure
    │
    ├── 3. Select KV Cache
    │       Calculate f16 footprint → fit f16 / q8_0 / iso3 by VRAM
    │
    ├── 4. Warmup Benchmark
    │       Walk context from max downward, find sweet spot
    │
    ├── 5. Tune Parameters
    │       ubatch, threads, mlock — all measured, not guessed
    │
    └── 6. Cache & Serve
            OpenAI API at localhost:21435/v1
```

On subsequent runs:
```
✓ Using cached config  (64K ctx · 26.2 tok/s · 3 days ago)
```

## Auto-Tuned Parameters

| Parameter | How Ashforge decides |
|-----------|---------------------|
| Context length | Walks from model max downward; stops where speed ≥ threshold |
| KV cache type | Calculates f16 footprint; uses f16 → q8_0+q4_0 → iso3 by VRAM fit |
| MoE placement | Detects expert tensors; routes to CPU automatically |
| Ubatch size | Benchmarks 128 vs 512; picks the faster one |
| Thread count | 2 for full-GPU, physical_cores/2 for MoE offload |
| GPU tensor split | Weighted by VRAM × bandwidth for multi-GPU setups |

## Commands

| Command | Description |
|---------|-------------|
| `ashforge run <model>` | Start serving a model with auto-tuned parameters |
| `ashforge stop` | Stop the running model |
| `ashforge status` | Show model status, speed, VRAM usage |
| `ashforge list` | List available and downloaded models |
| `ashforge probe` | Display detected hardware info |
| `ashforge inject` | Auto-configure IDE (Cursor, Codex, Claude Code) |
| `ashforge config show` | View current configuration |
| `ashforge config set <k=v>` | Modify configuration |
| `ashforge cache clear` | Clear warmup cache |
| `ashforge version` | Show version |

## Advanced Usage

```bash
# Override context size
ashforge run Qwen3-8B --ctx-size 12000

# Force re-tune after hardware change
ashforge run Qwen3-8B --reset

# Fast start — skip warmup, use cached config
ashforge run Qwen3-8B --fast

# Choose optimization mode
ashforge run Qwen3-8B --mode speed      # fastest tok/s
ashforge run Qwen3-8B --mode balanced   # balanced (default)
ashforge run Qwen3-8B --mode context    # maximum context

# Listen on all interfaces (LAN access)
ashforge run Qwen3-8B --host 0.0.0.0

# Use custom llama-server binary
ashforge run Qwen3-8B --llama-server /path/to/llama-server

# Set custom model directory
ashforge config set model_dir=/data/models

# Use HuggingFace mirror
ashforge config set hf_mirror=https://hf-mirror.com
```

## Configuration

Ashforge supports both config file (`~/.ashforge/config.yaml`) and environment variable overrides:

| Env Variable | Config Key | Default |
|-------------|------------|---------|
| `ASHFORGE_HF_MIRROR` | `hf_mirror` | `https://hf-mirror.com` |
| `ASHFORGE_LLAMA_PORT` | `llama_port` | `21434` |
| `ASHFORGE_PROXY_PORT` | `proxy_port` | `21435` |
| `ASHFORGE_MODEL_DIR` | `model_dir` | `~/.ashforge/models` |
| `ASHFORGE_LOG_LEVEL` | `log_level` | `info` |
| `ASHFORGE_LLAMA_TAG` | — | Built-in release tag |

## Supported Models

Ashforge ships with a knowledge base of popular models and also auto-detects any GGUF file:

| Family | Models |
|--------|--------|
| Qwen | Qwen3-235B, Qwen3.6-35B, Qwen3-32B, Qwen3-30B, Qwen3-14B, Qwen3-8B, Qwen3-4B, Qwen3-1.7B, Qwen3-0.6B |
| Llama | Llama 4 Scout, Llama 4 Maverick |
| Gemma | Gemma 3 27B, 12B, 4B |
| DeepSeek | DeepSeek R1, DeepSeek R1 0528 |
| Phi | Phi-4 14B, Phi-4 Mini |
| Mistral | Mistral Small 24B, Codestral 25.01 |
| Mixtral | Mixtral 8x7B |
| Yi | Yi-1.5 34B, 9B |
| InternLM | InternLM3 8B |
| GLM | GLM-4 9B |
| Command R | Command R 7B |

## Requirements

- **GPU**: NVIDIA (CUDA 12.4+) or Apple Silicon (Metal)
- **OS**: Linux, Windows 10/11, macOS 12+
- **RAM**: 8GB+ (16GB+ for 30B+ models)
- **Model format**: GGUF

CPU-only inference is supported but not the primary focus.

## Build from Source

```bash
git clone https://github.com/MMMchou/ashforge.git
cd ashforge
make build-linux    # or build-windows
```

Requires Go 1.22+.

---

<a name="中文"></a>

## 中文

Ashforge 是一个本地大模型自动调优部署工具。输入一条命令，自动完成硬件探测、模型分析、参数优化、API 启动。

```bash
ashforge run Qwen3-30B-A3B
```

### 核心特性

- 硬件自动探测（NVIDIA CUDA / Apple Silicon Metal）
- 模型感知优化（自动读取 GGUF 元数据）
- KV Cache 智能选择（f16 → q8_0 → iso3）
- Warmup 自动调参（二分搜索最优上下文）
- MoE 专家层卸载（attention 留 GPU，experts 走 CPU）
- OpenAI 兼容 API（`http://localhost:21435/v1`）
- 重复检测 & 上下文压缩
- IDE 一键配置（Cursor / Codex / Claude Code）
- TUI 实时监控面板
- 自动端口发现
- 环境变量覆盖配置

### 安装

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/MMMchou/ashforge/main/install.sh | sh

# Windows
irm https://raw.githubusercontent.com/MMMchou/ashforge/main/install.ps1 | iex
```

### 快速开始

```bash
ashforge run Qwen3-30B-A3B    # 运行模型
ashforge list                  # 查看可用模型
ashforge status                # 查看运行状态
ashforge stop                  # 停止服务
ashforge inject                # 配置 IDE
ashforge cache clear           # 清除调优缓存
ashforge config show           # 查看配置
```

---

<div align="center">

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) · by [Ashan](https://github.com/MMMchou)

</div>
