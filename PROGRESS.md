# Ashforge 项目进度

## 项目概述

本地大模型部署器，对标 LM Studio/Ollama，自动硬件探测 + 参数调优，零配置。

- **仓库：** https://github.com/MMMchou/ashforge
- **源码：** `D:\program\ollama\ashforge-release\`
- **配置目录：** `C:\Users\15488\.ashforge\`
- **技术栈：** Go 1.22+ + llama.cpp turboquant fork (iso3 KV cache)

---

## 版本历史

### v0.1.6（2026-04-25）— MoE offload 修复 + 直接路径修复

**修复：**
- MoE offload warmup 全 OOM 问题（根因：`-ot` 正则未生效，所有层上 GPU）
  - 改用 `--cpu-moe` 替代 `-ot ".ffn_.*_exps.=CPU"`，更可靠
  - `SelectKVCacheType` MoE 模式不再猜 modelVRAM_MB，直接信任 `--fit on` 处理层分配
  - warmup 超时从 60s 延长到 180s（MoE 模型加载 ~13GB 需要更长时间）
- 直接 GGUF 路径 bug（`ashforge run /path/to/model.gguf` 实际走下载流程）
  - `DeployProfile` 新增 `LocalPath` 字段
  - `EnsureFile` 优先返回 `LocalPath`，跳过下载

**架构改进：**
- 分工更清晰：`--fit on` 负责层分配（llama.cpp 比 Ashforge 算得准），Ashforge 负责 KV cache 类型选择和速度探测

**实测数据（RTX 5060 Laptop 8GB）：**
- Qwen3-30B-A3B Q3_K_XL：32K ctx · f16 KV · 4.8 GB VRAM · **8.7 tok/s**
- Qwen3-8B Q5_K_M：28K ctx · q8_0 KV · 7.2 GB VRAM · **38.3 tok/s**
- Llama 3.1 8B Q5_K_M：64K ctx · f16 KV · 7.5 GB VRAM · **29.1 tok/s**

---


**修复：**
- iso3 检测结果缓存到磁盘（`~/.ashforge/iso3_support_<mtime>.txt`），同一 binary 只检测一次
- SM 版本感知超时：SM < 75 直接跳过（不支持 iso3），SM 75-119 用 15s，SM 120+ 用 60s
- 去掉 `Start()` 里的冗余 iso3 检测（Preflight 已做过）
- OOM 建议文案动态化：小模型 OOM 不再错误建议"换 MoE offload 模型"
  - 模型 > VRAM 70%：建议换小量化，dense 大模型才建议换 MoE 架构
  - 模型 < VRAM 50%（小模型 OOM）：建议 `--reset` 重新探测
- 小模型（< 2GB）ubatch 从 512 降到 128，修复 `--kv-unified` 预分配 OOM

**新功能：**
- 带宽从 nvidia-smi XML 精确计算：`bus_width_bits/8 × max_mem_clock_MHz × 2 / 1000`
  - 覆盖所有 Kepler+（2012 年及以后）显卡，无需手动维护枚举表
  - 虚拟化环境 / 老驱动读不到时，枚举表兜底
- 低带宽卡（< 200 GB/s，如 GTX 1660 Ti 192 GB/s）warmup 只测 ubatch=128，减少 1-2 分钟等待
- 完整 GPU 带宽枚举表（GTX 10/16/20/30/40/50 + 数据中心 V100/P100/H200）

---

### v0.1.4（2026-04-25）— Blackwell 架构支持修复

**修复：**
- RTX 50 系（SM120 Blackwell）iso3 检测超时：10s → 60s
  - 根因：CUDA 12.4 无 SM120 预编译 kernel，每次启动需 PTX JIT 编译（~30s）
- 检测到 SM120 时打印提示：`⚠ RTX 50 系首次启动需要 JIT 编译 (~30s)`

---

### v0.1.3（2026-04-25）— 混合架构支持 + APEX 量化 + 硬件探测增强

**新功能：**
- APEX 量化三档预设：Quality (q8_0) / Balanced (q5_k_m) / Compact (q4_k_m)
- 混合架构动态检测：iso3 自动禁用 + `--swa-full` 补偿（DeltaNet/SSM 架构）
- 直接 GGUF 路径支持：`ashforge run /path/to/model.gguf`
- `ashforge cache clear` 命令
- `--llama-server` 自定义二进制路径
- Flash Attention 自动启用（SM75+）
- NVLink 自动检测

**修复：**
- nvidia-smi XML 解析替代脆弱的 CSV 解析
- CUDA 13.2 警告（已知低比特量化 bug）
- Tesla/Quadro GPU 识别
- 多卡 VRAM 计算错误

---

### v0.1.2（2026-04-24）— iso3 检测时机修复

**修复：**
- iso3 检测移到 warmup 之前（Preflight 阶段）
  - 根因：warmup 用 iso3 参数启动 llama-server，但 binary 不支持 iso3 → 所有 ctx 探测都失败，误报 OOM
  - 修复后：先检测 → 不支持则回退 q8_0/q4_0 → warmup 正常运行

---

### v0.1.1（2026-04-23）— 基础功能完成

**核心功能：**
- hardware probe：GPU（nvidia-smi）、CPU（sysinfo）、RAM
- model matcher：VRAM 匹配量化，full_gpu / moe_offload 两种模式
- warmup benchmark：oobabooga 公式反解起始 ctx，二分探测最大 ctx，ubatch 实测
- config cache：结果缓存到 `~/.ashforge/profiles/`，第二次启动 2s
- bundled binary：turboquant iso3 llama-server 打包进 release
- OpenAI 兼容 API：`http://localhost:21435/v1`

---

## 核心架构

```
ashforge run <model>
    ↓
[1] probe hardware     → GPU/CPU/RAM 检测
[2] match model        → 选量化 + 模式（full_gpu / moe_offload）
[3] check files        → bundled binary + 模型文件
[4] preflight          → iso3 检测 + VRAM 预检
[5] warmup benchmark   → 探测最优 ctx + ubatch（有缓存则跳过）
[6] start server       → llama-server + OpenAI API
```

**KV cache 选择策略（`model/kv_cache.go`）：**
- 计算 f16 KV cache 占用，装得下就用 f16（最快）
- 装不下降到 q8_0+q4_0（平衡）
- 仍不够用 iso3（最省 VRAM，需 turboquant binary）
- 兜底 q4_0+q4_0

**iso3 检测（`engine/runner.go`）：**
- SM < 75：直接跳过（不支持）
- SM 75-119：`--help` 检测，15s 超时
- SM 120+：`--help` 检测，60s 超时（PTX JIT）
- 结果缓存到磁盘，同一 binary 只检测一次

---

## 实测数据

### RTX 5060 Laptop 8GB（本机）

| 模型 | 量化 | ctx | 速度 | VRAM |
|------|------|-----|------|------|
| Qwen3-8B | Q5_K_M | 32K | 22.0 tok/s | 7.2 GB |
| Llama 3.1 8B | Q5_K_M | 64K | 26.2 tok/s | 7.3 GB |
| Llama 3.1 8B | Q5_K_M | 8K | 51.7 tok/s | 7.0 GB |
| Qwen3-30B-A3B | Q3_K_XL | 4K | 14.2 tok/s | 2.4 GB |

**vs LM Studio（Llama 3.1 8B Q5_K_M, 8K ctx）：**
- Ashforge: 51.7 tok/s，64K ctx 自动
- LM Studio: 46.5 tok/s，8K ctx 默认

### 双 RTX 4090 VPS（183.222.230.89）

| 模型 | 量化 | ctx | 速度 | VRAM |
|------|------|-----|------|------|
| Qwen3.6-35B-A3B | Q4_K_XL | 512K | 126.5 tok/s | 35.8 GB |
| Qwen3.6-35B-A3B | Q5_K_M | 256K | 47.2 tok/s | 8.1 GB |

---

## 已知问题 / 技术债

| 问题 | 状态 | 说明 |
|------|------|------|
| GTX 1660 Ti 小模型 OOM | ✅ v0.1.5 修复 | ubatch 512→128，OOM 建议文案修复 |
| Blackwell iso3 检测超时 | ✅ v0.1.4 修复 | 超时 10s→60s |
| iso3 每次重复检测 | ✅ v0.1.5 修复 | 结果缓存到磁盘 |
| warmup 测速 vs 实际速度差异 | 🔄 待优化 | 64K ctx warmup 28.7 但实际 20 tok/s |
| matcher 本地文件检查每次扫目录 | 🔄 技术债 | 可加缓存 |
| warmup profile 无版本号 | 🔄 技术债 | 参数变更后可能用旧缓存 |
| PagedAttention | ❌ 不做 | 需要 llama.cpp 主线支持，等上游 |

---

## CI/CD

**GitHub Actions：**
- `release.yml`：push tag → 编译 Go binary（Windows/Linux）→ 打包 release
- `build-llama-server.yml`：手动触发 → 编译 turboquant iso3 llama-server

**CUDA 架构：** sm_75/80/86/89（Turing/Ampere/Ada）
- SM120（Blackwell）通过 PTX JIT 运行时编译支持

**MSVC 编译 3 处 fix（已集成到 workflow）：**
1. `ops.cpp`：`extern "C" GGML_API int turbo3_cpu_wht_group_size;` → 改为定义
2. `ggml-turbo-quant.c`：添加 `#define _USE_MATH_DEFINES`
3. `llama-kv-cache.cpp`：添加 `float * g_innerq_scale_inv_host = nullptr;`
