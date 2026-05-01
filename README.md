<div align="center">

<br>

```
 █████╗ ███████╗██╗  ██╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔══██╗██╔════╝██║  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
███████║███████╗███████║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██╔══██║╚════██║██╔══██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██║  ██║███████║██║  ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

**One command. Auto-tuned. Maximum performance.**

*The only local LLM tool that benchmarks your actual hardware — not just guesses.*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8.svg)](https://go.dev)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

[Quick Start](#-quick-start) · [Why Ashforge](#-why-not-just-use-ollama) · [How It Works](#-how-it-works) · [中文](#中文)

</div>

---

## The Problem

You download a 30B model. You run it in Ollama. It works... at 8 tok/s with 4K context, on a GPU that could do 35 tok/s at 64K.

**Why?** Because Ollama doesn't know your GPU's bandwidth is 504 GB/s, your VRAM can fit f16 KV cache, and ubatch=512 is 40% faster than the default on your hardware.

Ashforge does.

```bash
ashforge run Qwen3-30B-A3B
```

```
[5/6] Warmup benchmark...
      Probe 1: ctx=128K ... OOM
      Probe 2: ctx=64K  ... 26.3 tok/s
      Probe 3: ctx=128K ... OOM
      Fine:    ctx=96K  ... 22.1 tok/s
      Tune ubatch: ub=128 → 24.8 tok/s; ub=512 → 26.3 tok/s
      ✓ 26.3 tok/s @ 64K ctx

  ┌─────────────────────────────────────────────────┐
  │  Ready — Qwen3-30B-A3B @ 26.3 tok/s            │
  │  API: http://127.0.0.1:21435/v1/chat/completions│
  └─────────────────────────────────────────────────┘
```

Second launch? **2 seconds**. Configuration cached.

---

## Why Not Just Use Ollama?

| | Ollama / LM Studio | Ashforge |
|---|---|---|
| **Parameter tuning** | Static defaults | Benchmarks your actual hardware |
| **Context size** | Fixed (usually 4K-8K) | Binary search finds your real max |
| **KV cache** | Always q8_0 | Picks f16 → q8_0 → q4_0 by VRAM fit |
| **VRAM prediction** | None (OOM = restart) | Formula from 19,517 measurements, 365 MiB median error |
| **MoE offload** | Manual `--n-gpu-layers` | Auto-splits attention (GPU) vs experts (CPU) |
| **Ubatch size** | Default 512 | Benchmarks 128 vs 512, picks faster |
| **Multi-GPU** | Basic tensor split | VRAM×bandwidth weighted split + NVLink graph mode |
| **Repetition loops** | You wait and Ctrl+C | Auto-detected and stopped (n-gram + pattern) |
| **Context overflow** | Silently truncates | Zero-latency extractive compression |
| **Speculative decode** | Manual flag | Auto-enables MTP or n-gram lookup |
| **Live monitoring** | None | Real-time TUI (VRAM, speed, temp, context) |
| **IDE setup** | Manual config | `ashforge inject` → Cursor, Codex, Claude Code |

### The Numbers That Matter

Ashforge uses the [oobabooga VRAM formula](https://oobabooga.github.io/blog/posts/gguf-vram-formula/) — derived from **19,517 real measurements across 60 models** with a **median error of just 365 MiB** — to predict exactly how much context your GPU can handle before launching.

Other tools guess. Ashforge calculates.

---

## Quick Start

**Install:**

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/MMMchou/ashforge/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/MMMchou/ashforge/main/install.ps1 | iex
```

**Run:**

```bash
ashforge run Qwen3-30B-A3B
```

That's it. Ashforge will:
1. Detect your GPU, VRAM, bandwidth, CPU, RAM
2. Download the optimal quantization for your hardware
3. Benchmark to find the max context at target speed
4. Start an OpenAI-compatible API at `http://localhost:21435/v1`

**Connect your tools:**

```bash
ashforge inject    # auto-configures Cursor, Codex CLI, Claude Code
```

Or point any OpenAI-compatible client to `http://localhost:21435/v1`.

---

## How It Works

```
ashforge run Qwen3-30B-A3B

  [1/6] Probe Hardware
        RTX 4090 (SM89, 24576 MB, 1008 GB/s)
        RAM: 64 GB DDR5

  [2/6] Select Configuration
        Model:  Qwen3-30B-A3B (MoE, 30B total / 3B active)
        Quant:  Q4_K_M (17.2 GB)
        Mode:   moe_offload (experts on CPU)

  [3/6] Check Files
        Binary: llama-server-cuda [cached]
        Model:  Qwen3-30B-A3B-Q4_K_M.gguf [cached]

  [4/6] Preflight Check
        ✓ VRAM sufficient

  [5/6] Warmup Benchmark           ← the magic happens here
        Phase 1:   coarse search (power-of-2 steps)
        Phase 1.5: fine search (4K precision binary search)
        Phase 2:   ubatch tuning (128 vs 512)
        ✓ 26.3 tok/s @ 64K ctx

  [6/6] Start Server
        llama-server + Ashforge proxy + TUI monitor
```

### What Gets Auto-Tuned

Every parameter is **measured, not guessed**:

| Parameter | How Ashforge Decides |
|-----------|---------------------|
| **Context size** | Binary search from model max downward; 3 modes: speed / balanced / context |
| **KV cache type** | Calculates f16 VRAM footprint → cascades f16 → q8_0+q4_0 → iso3 → q4_0 |
| **MoE placement** | Measures VRAM for attention layers (25%), routes experts to CPU |
| **Ubatch size** | Benchmarks candidates; skips 512 on low-bandwidth GPUs (<200 GB/s) |
| **Thread count** | 2 for full-GPU; physical_cores/2 for CPU offload |
| **Tensor split** | Weighted by VRAM × memory bandwidth per GPU |
| **Speculative decode** | MTP (3 tokens) for Qwen3.6; n-gram lookup (8) for all others |
| **Flash attention** | Enabled on SM75+ (Turing and newer) |
| **mlock / mmap** | Based on RAM headroom vs model size |
| **KV defrag** | Auto-compact at 10% fragmentation |

<details>
<summary><b>Full list of 25+ auto-configured llama-server flags</b></summary>

```
--ctx-size         dynamic        from warmup binary search
--batch-size       512 / 4096     mode-dependent
--ubatch-size      128 / 512      from Phase 2 benchmark
-ctk / -ctv        f16/q8_0/q4_0  from KV cache selection
--cache-reuse      256-1024       from context size
--threads          2 / cores/2    from mode + CPU count
--n-gpu-layers     999            all layers to GPU
--parallel         1              single-user optimized
--kv-unified       conditional    skip on Blackwell / multi-GPU
--cpu-moe          conditional    MoE full offload
--n-cpu-moe N      conditional    MoE partial offload
--fit on           conditional    dense models only
--flash-attn       conditional    SM75+ (Turing+)
-sm graph          conditional    NVLink + turboquant
--tensor-split     conditional    multi-GPU weighted
--swa-full         conditional    hybrid architectures
--mlock            conditional    RAM headroom check
--mmap             conditional    when mlock inactive
--num-spec-tokens  3              native MTP models
--lookup           8              n-gram speculative
--defrag-thold     0.1            always enabled
--cont-batching    on             always enabled
--metrics          on             always enabled
--no-webui         on             always disabled
```

</details>

---

## Smart Proxy

Ashforge doesn't just forward requests — it makes them better:

### Repetition Detection
Dual-detector system catches infinite loops in real-time:
- **N-gram detector**: tracks 3-gram frequencies in a 200-token sliding window
- **Pattern detector**: identifies repeating sequences of 2-10 tokens

When triggered, auto-stops generation and warns the user. No more waiting 30 seconds for a model stuck in a loop.

### Context Compression
When conversation history hits 75% of context window:
- Keeps system prompt + recent 8K tokens untouched
- Compresses middle messages with zero-latency extractive summary
- Preserves code blocks, file paths, function definitions, TODOs
- No model call needed — pure algorithmic, zero added latency

### Live Context Warnings
- Header `X-Ashforge-Context-Warning` at 80% usage
- Inline hint at 90%: "Context is filling up, important info may be truncated"

---

## Live TUI Monitor

Real-time terminal dashboard, refreshes every 2s:

```
─ Live Monitor · Generating ──────────────── refresh 2s ─
  64K ctx · q8_0+q4_0 KV · ub512 · mlock

  Speed         VRAM           RAM        GPU      Temp
  26.3 tok/s   18.2/24.0 GB   12.1/64 GB   95%     67°C
  [========..]  [========..]  [==........]  [=====.] [======....]

─────────────────────────────────────────────────────────
  Context  [================....] 52.1K / 64.0K  余 11.9K  压缩 2 次 · 省 8.3K
```

Color-coded alerts:
- VRAM > 90%: warns inference may crash
- Context > 80%: suggests starting a new conversation
- GPU temp > 85°C: warns about thermal throttling

---

## Supported Hardware

| Platform | GPU | Status |
|----------|-----|--------|
| Linux x86_64 | NVIDIA CUDA (SM75+) | Stable |
| Windows x86_64 | NVIDIA CUDA | Stable |
| macOS arm64 | Apple Silicon Metal | Beta |
| macOS x86_64 | Intel (CPU only) | Supported |
| Linux x86_64 | AMD Vulkan | Experimental |

**Multi-GPU**: auto-detected. NVLink → graph split mode. No NVLink → VRAM×bandwidth weighted tensor split.

**RTX 50 series (Blackwell)**: first-run JIT compilation handled automatically (~60s one-time, then instant).

---

## Supported Models

30+ models with pre-configured profiles. Any GGUF file also works via auto-detection.

| Family | Models |
|--------|--------|
| **Qwen** | Qwen3-235B, Qwen3.6-35B, Qwen3-32B, Qwen3-30B-A3B, Qwen3-14B, Qwen3-8B, Qwen3-4B, Qwen3-1.7B, Qwen3-0.6B |
| **Llama** | Llama 4 Scout, Llama 4 Maverick |
| **DeepSeek** | DeepSeek R1, DeepSeek R1 0528 |
| **Gemma** | Gemma 3 27B, 12B, 4B |
| **Phi** | Phi-4 14B, Phi-4 Mini |
| **Mistral** | Mistral Small 24B, Codestral 25.01, Mixtral 8x7B |
| **Others** | Yi-1.5 34B/9B, InternLM3 8B, GLM-4 9B, Command R 7B |

---

## All Commands

```bash
ashforge run <model>              # deploy with auto-tuning
ashforge run <model> --fast       # skip warmup, use cache
ashforge run <model> --reset      # re-benchmark from scratch
ashforge run <model> --mode speed # or balanced / context
ashforge run <model> --ctx-size N # override context size
ashforge run <model> --host 0.0.0.0  # LAN access

ashforge stop                     # stop running model
ashforge status                   # show status + speed
ashforge list                     # list all models
ashforge list --installed         # only downloaded models
ashforge probe                    # hardware fingerprint

ashforge inject                   # auto-configure IDEs
ashforge inject --undo            # restore IDE configs

ashforge config show              # view settings
ashforge config set key=value     # change settings
ashforge cache clear              # clear warmup cache
```

## Configuration

Config file: `~/.ashforge/config.yaml`

| Env Variable | Config Key | Default | Description |
|-------------|------------|---------|-------------|
| `ASHFORGE_HF_MIRROR` | `hf_mirror` | `https://hf-mirror.com` | HuggingFace mirror URL |
| `ASHFORGE_LLAMA_PORT` | `llama_port` | `21434` | llama-server port |
| `ASHFORGE_PROXY_PORT` | `proxy_port` | `21435` | API proxy port |
| `ASHFORGE_MODEL_DIR` | `model_dir` | `~/.ashforge/models` | Model storage path |
| `ASHFORGE_LOG_LEVEL` | `log_level` | `info` | Log verbosity |
| `ASHFORGE_API_KEY` | `api_key` | — | API key for IDE injection |
| `ASHFORGE_LLAMA_TAG` | — | Built-in | llama.cpp release tag |

---

## Build from Source

```bash
git clone https://github.com/MMMchou/ashforge.git
cd ashforge
make build-linux    # or build-darwin, build-windows
```

Requires Go 1.22+.

---

## Requirements

- **GPU**: NVIDIA CUDA 12.4+ or Apple Silicon Metal
- **OS**: Linux, Windows 10/11, macOS 12+
- **RAM**: 8 GB+ (16 GB+ for 30B+ models)
- **Format**: GGUF

CPU-only mode is supported but not the focus.

---

<a name="中文"></a>

## 中文

### 为什么选 Ashforge?

**Ollama 和 LM Studio 让模型能跑。Ashforge 让模型跑得快。**

同样的模型、同样的显卡，Ashforge 通常能多给你 **2-4 倍的上下文**和 **20-40% 的速度提升**。

原因很简单：其他工具用默认参数。Ashforge **实测你的硬件**，二分搜索最优配置，结果缓存后第二次启动只要 2 秒。

```bash
ashforge run Qwen3-30B-A3B
# 自动完成：硬件探测 → 模型分析 → VRAM 预测 → 参数调优 → 启动服务
```

### 核心优势

**自动调参** — 不是套模板，是真跑 benchmark
- 二分搜索最大可用上下文（4K 精度）
- ubatch 对比测试（128 vs 512）
- 三档模式可选：速度优先 / 均衡 / 上下文优先
- 结果缓存 30 天，下次启动 2 秒

**VRAM 预测** — 基于 oobabooga 公式（19,517 次实测，中位误差 365 MiB）
- 启动前就知道能开多大上下文
- 不会 OOM 崩溃再重试

**MoE 智能卸载** — attention 留 GPU，expert 走 CPU
- 自动计算最优切分比例
- 30B-A3B 这类 MoE 模型速度翻倍

**KV Cache 自动选型** — f16 → q8_0+q4_0 → iso3 → q4_0
- 根据剩余 VRAM 自动选最快的类型
- Ollama 只会用默认 q8_0

**智能代理层**
- 重复检测：n-gram + 模式识别双引擎，自动停止死循环
- 上下文压缩：75% 满时自动压缩历史，零延迟（纯算法，不调模型）
- 投机解码：MTP 模型自动启用 3-token 预测，其他模型用 n-gram lookup

**一键接入 IDE**
```bash
ashforge inject   # 自动配置 Cursor / Codex CLI / Claude Code
```

### 实时监控

```
速度         显存           内存        GPU      温度
26.3 tok/s  18.2/24.0 GB  12.1/64 GB   95%     67°C
[========..] [========..] [==........] [=====.] [======....]

上下文  [================....] 52.1K / 64.0K  余 11.9K  压缩 2次 · 省 8.3K
```

### 安装

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/MMMchou/ashforge/main/install.sh | sh

# Windows
irm https://raw.githubusercontent.com/MMMchou/ashforge/main/install.ps1 | iex
```

### 快速开始

```bash
ashforge run Qwen3-30B-A3B    # 运行模型（自动下载）
ashforge run ./model.gguf     # 运行本地 GGUF 文件
ashforge list                  # 查看支持的模型
ashforge list --installed      # 只看已下载的
ashforge probe                 # 查看硬件信息
ashforge inject                # 配置 IDE
ashforge stop                  # 停止服务
```

API 地址：`http://localhost:21435/v1`，兼容所有 OpenAI 格式的工具。

---

<div align="center">

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) · by [Ashan](https://github.com/MMMchou)

If Ashforge saves you time, consider giving it a star.

</div>
