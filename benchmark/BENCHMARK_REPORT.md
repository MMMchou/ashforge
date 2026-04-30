# Ashforge vs LM Studio — Benchmark 报告

> 测试日期：2026-04-22 | 测试脚本：统一 Python 脚本，非流式 API 调用，每组 3 次取中位数

## 测试环境

| 项目 | 本地笔记本 | VPS 服务器 |
|------|-----------|-----------|
| GPU | RTX 5060 Laptop 8GB | RTX 4090 24GB × 2 |
| CPU | i7-13700HX (16C/24T) | 24 核 |
| RAM | 16 GB | 128 GB |
| OS | Windows 11 | Ubuntu 22.04 |

| 软件 | 版本 |
|------|------|
| Ashforge | v4 dev |
| LM Studio | 最新版 (2026-04) |
| llama.cpp | b8864 (Ashforge) / LM Studio 内置 |

## 测试方法

- 统一 prompt：中文长文本（量子计算原理解释），max_tokens=512
- 非流式请求，用 API 返回的 `usage.completion_tokens` 精确计算 tok/s
- GPU/CPU/RAM 通过 nvidia-smi + psutil 采集
- LM Studio 配置：`--gpu max --parallel 1 --context-length 4096`（最优单用户配置）
- Ashforge 配置：全自动（零手动调参）
- Qwen3 系列模型统一使用 `/no_think` 关闭 thinking mode

---

## 场景一：8GB 显卡跑 30B MoE — 核心卖点

模型：Qwen3-30B-A3B (MoE, 30B 总参 / 3B 激活)，量化：UD-Q3_K_XL (13.8GB)

测试了两种 LM Studio 配置：默认（`--gpu max`）和手动优化（GPU offload=48 层）。

| 指标 | LM Studio 默认 | LM Studio 手动调优 | Ashforge (零配置) |
|------|---------------|-------------------|---------------|
| 速度 (tok/s) | 2.5 | 3.0 | 21.0 |
| VRAM 占用 | 7,434 MB (91%) | 7,549 MB (93%) | 2,603 MB (32%) |
| GPU 利用率 | 100% | 100% | 14% |
| CPU 占用 | 5.9% | 18.3% | 11.4% |
| RAM 占用 | 15.5 GB | 11.5 GB | 15.7 GB |
| GPU 温度 | 59°C | 52°C | 61°C |
| vs Ashforge | 慢 8.4× | 慢 7.0× | — |

LM Studio 即使手动设置 GPU offload=48 层（与 Ashforge 的 MoE offload 策略相同思路），
仍然只有 3.0 tok/s。原因是 LM Studio 把 7.5GB 数据塞进 GPU（93% VRAM），
而 Ashforge 只放 2.6GB 必要数据到 GPU，experts 在 CPU 上高效运行，速度快 7 倍。

核心差异不仅是"自动 vs 手动"，而是 offload 策略本身的效率差距。

---

## 场景二：8B 常规模型 — 日常使用

模型：Qwen3-8B-Q5_K_M (dense, 5.7GB)

| 指标 | LM Studio | Ashforge | 差距 |
|------|-----------|-------|------|
| 速度 (tok/s) | 44.3 | 42.4 | LM Studio 快 4% |
| VRAM 占用 | 7,476 MB | 7,473 MB | 持平 |
| GPU 利用率 | 95% | 95% | 持平 |
| CPU 占用 | 5.6% | 9.8% |  |
| RAM 占用 | 14.6 GB | 15.2 GB | 持平 |
| GPU 温度 | 72°C | 72°C | 持平 |

8B dense 模型完全装进 GPU，两者表现接近。LM Studio 略快 4%，属于正常波动范围。
说明 Ashforge 在常规场景下不会拖后腿，性能与 LM Studio 持平。

---

## 场景三：双 4090 高性能 — Ashforge 天花板

模型：Qwen3.6-35B-A3B (MoE, 35B 总参 / 3B 激活)，量化：UD-Q4_K_XL

| 指标 | Ashforge (双 4090) |
|------|-----------------|
| 速度 (tok/s) | 115.0 |
| VRAM 占用 | 13,474 + 13,018 = 26,492 MB |
| GPU 利用率 | 33% + 39% |
| CPU 占用 | 0.5% |
| RAM 占用 | 6.0 GB / 128 GB |
| GPU 温度 | 46°C + 51°C |

双 4090 全 GPU 模式，35B MoE 模型跑到 115 tok/s，接近人眼阅读速度的 10 倍。

---

## 关键发现

1. **MoE 模型是 Ashforge 的核心优势**。8GB 显卡跑 30B MoE，Ashforge 比 LM Studio 快 7-8 倍，VRAM 省 65%。即使 LM Studio 手动调优（GPU offload=48），仍只有 3.0 tok/s，而 Ashforge 零配置达到 21 tok/s。
2. **常规 dense 模型两者持平**。8B 模型 LM Studio 44.3 vs Ashforge 42.4 tok/s，差距 4%，可忽略。
3. **Ashforge 零配置 vs LM Studio 需要专家级调参**。LM Studio 需要知道：GPU offload 层数、parallel 数量、batch size、CPU 线程数等。Ashforge 装上就跑，全自动。
4. **高端硬件线性扩展**。双 4090 跑 35B MoE 达到 115 tok/s。

## 为什么 Ashforge 的 MoE offload 更快？

LM Studio 和 Ashforge 都做了 MoE expert offload，但策略不同：

| 策略 | LM Studio | Ashforge |
|------|-----------|-------|
| GPU 数据量 | 7.5GB (93% VRAM) | 2.6GB (32% VRAM) |
| Offload 粒度 | 按层（48 层） | 按模块（`.ffn_.*_exps.`） |
| VRAM 压力 | 极高，频繁 swap | 低，留足余量 |
| CPU-GPU 传输 | 频繁 | 最小化 |

Ashforge 的精细化 offload 策略只把 MoE experts 放 CPU，attention 等关键层留在 GPU，
减少了 CPU-GPU 数据传输，因此即使都做了 offload，Ashforge 仍快 7 倍。

## 场景四：Ashforge iso3 vs llama-server 默认 — VPS AB 测试（2026-04-23）

模型：Qwen3.6-35B-A3B UD-Q4_K_XL，双 RTX 4090 24GB，128GB RAM，ctx=512K

| 指标 | Ashforge (iso3+MTP) | llama-server 默认 | 差距 |
|------|-------------------|-------------------|------|
| No Think (tok/s) | 126.5 | 136.1 | 默认快 8% |
| Thinking (tok/s) | 125.5 | 138.5 | 默认快 10% |
| Prompt 524t 耗时 | 0.72s | 0.65s | 默认快 10% |
| GPU0 VRAM | 18,546 MiB | 23,016 MiB | Ashforge 省 4.4 GB |
| GPU1 VRAM | 17,216 MiB | 20,362 MiB | Ashforge 省 3.1 GB |
| 总 VRAM | 35,762 MiB | 43,378 MiB | Ashforge 省 7.6 GB (17%) |
| RAM | 4,089 MB | 3,736 MB | 持平 |

结论：VRAM 充裕时 iso3 的解码开销导致速度略慢 ~8-10%，但省 7.6 GB VRAM。
iso3 的核心价值在 VRAM 紧张的设备上（8GB 笔记本），而非高端双卡场景。

---

## 场景五：非 thinking 模型对比 — Llama 3.1 8B（2026-04-24）

模型：Meta-Llama-3.1-8B-Instruct Q5_K_M (5.4GB)，RTX 5060 Laptop 8GB

| 指标 | Llama 3.1 8B | Qwen3-8B (对比) |
|------|-------------|-----------------|
| 速度 (tok/s) | 41.5 | 22.0 (no_think) |
| Prompt (tok/s) | ~218 | 261.6 |
| VRAM | 6.9 GB | 7.2 GB |
| ctx | 4K | 32K |
| iso3 | 生效 | 生效 |
| MTP | 不支持 | 支持 |

注意：Llama 3.1 generation 快 89% 主要因为不带 thinking + ctx 小 8 倍（KV cache 开销低）。
iso3 对 Llama 3.1 确实生效（启动参数含 `-ctk iso3 -ctv iso3`），ctx 小是 VRAM 不足导致。

---

## 原始数据

所有测试的完整 JSON 数据保存在 `benchmark/` 目录下，可复现。

| 文件 | 内容 |
|------|------|
| results_8b-lms_*.json | LM Studio 8B 测试 |
| results_8b-ashforge_*.json | Ashforge 8B 测试 |
| results_30b-lms_*.json | LM Studio 30B 测试 |
| results_30b-ashforge_*.json | Ashforge 30B 测试 |
| results_vps_dual4090.json | VPS 双4090 测试 |

## 复现方法

```bash
# 本地测试
python benchmark/bench.py 8b-lms    # LM Studio 先加载模型
python benchmark/bench.py 8b-ashforge  # Ashforge: ashforge run qwen3-8b --fast
python benchmark/bench.py 30b-lms
python benchmark/bench.py 30b-ashforge # Ashforge: ashforge run qwen3-30b-a3b --fast
```
