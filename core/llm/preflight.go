package llm

import (
	"fmt"
	"math"

	"github.com/MMMchou/ashforge/core/catalog"
	"github.com/MMMchou/ashforge/core/hw"
)

// PreflightCheck 只拦截明显不可能的情况：模型文件本身装不下。
func PreflightCheck(profile *catalog.DeployProfile, sys *hw.System) error {
	gpu := sys.PrimaryGPU()
	if gpu == nil {
		return nil
	}

	vramGB := float64(sys.TotalVRAM_MB()) / 1024.0 // 多卡总VRAM
	ramGB := float64(sys.RAM.Total_MB) / 1024.0
	totalAvailGB := vramGB + ramGB*0.8

	if profile.Size_GB > totalAvailGB {
		return fmt.Errorf("模型 %.1f GB 超出可用内存 %.1f GB\n"+
			"  建议：选择更小的量化",
			profile.Size_GB, totalAvailGB)
	}

	if profile.Mode == "moe_offload" || profile.Mode == "moe_partial" {
		checkRAMForOffload(profile, sys)
	}

	// KV cache check: ensure at least 4K ctx fits after model weights
	if profile.Mode == "full_gpu" && gpu != nil {
		modelMB := profile.Size_GB * 1024
		kvMinMB := float64(profile.EstimateKVCacheMB(4096, "q4_0") * 2)
		overheadMB := 512.0
		needed := modelMB + kvMinMB + overheadMB
		if needed > float64(sys.TotalVRAM_MB()) {
			fmt.Printf("      ⚠️  VRAM 紧张：模型 %.0f MB + KV cache %.0f MB > %d MB\n",
				modelMB, kvMinMB, sys.TotalVRAM_MB())
		}
	}

	return nil
}

func checkRAMForOffload(profile *catalog.DeployProfile, sys *hw.System) {
	ramNeededGB := profile.Size_GB * 0.9
	ramAvailGB := float64(sys.RAM.Total_MB) / 1024.0
	ramFreeGB := float64(sys.RAM.Free_MB) / 1024.0

	if ramNeededGB > ramAvailGB-3 {
		fmt.Printf("      ⚠️  RAM 可能不足：模型需要 %.1f GB，总共 %.1f GB\n", ramNeededGB, ramAvailGB)
	} else if ramNeededGB > ramFreeGB {
		fmt.Printf("      ⚠️  RAM 当前可用 %.1f GB，模型需要 %.1f GB\n", ramFreeGB, ramNeededGB)
	}
}

// oobabooga 公式（19517 次实测，60 个模型，中位误差 365 MiB）
// 来源：https://oobabooga.github.io/blog/posts/gguf-vram-formula/
//
// vram_mib = (size_per_layer - 17.996 + 0.0000315 * kv_cache_factor)
//            * (gpu_layers + max(0.969, cache_type - (floor(50.778 * emb_per_ctx) + 9.988)))
//            + 1516.523
//
// size_per_layer = size_in_mb / n_layers
// kv_cache_factor = n_kv_heads * cache_type * ctx_size
// embedding_per_context = embedding_dim / ctx_size

// PredictVRAM 用 oobabooga 公式预测 VRAM 占用（MiB）
func PredictVRAM(sizeInMB float64, nLayers, gpuLayers, kvHeads, embeddingDim, ctxSize int, cacheType float64) float64 {
	sizePerLayer := sizeInMB / float64(nLayers)
	kvCacheFactor := float64(kvHeads) * cacheType * float64(ctxSize)
	embPerCtx := float64(embeddingDim) / float64(ctxSize)

	term1 := sizePerLayer - 17.99552795246051 + 3.148552680382576e-05*kvCacheFactor
	term2 := float64(gpuLayers) + math.Max(0.9690636483914102, cacheType-(math.Floor(50.77817218646521*embPerCtx)+9.987899908205632))

	return term1*term2 + 1516.522943869404
}

// SolveMaxCtx 反解 oobabooga 公式：给定可用 VRAM，求最大 ctx_size
// 二分搜索，因为公式对 ctx 不是简单线性关系
func SolveMaxCtx(freeVRAM_MiB float64, sizeInMB float64, nLayers, gpuLayers, kvHeads, embeddingDim int, cacheType float64) int {
	lo, hi := 512, 524288 // 搜索范围 512 ~ 512K

	for lo < hi {
		mid := (lo + hi + 1) / 2
		predicted := PredictVRAM(sizeInMB, nLayers, gpuLayers, kvHeads, embeddingDim, mid, cacheType)
		if predicted <= freeVRAM_MiB {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// IdealStartCtx 用 oobabooga 公式计算首次尝试的上下文大小
// 策略：min(模型原生最大ctx, oobabooga公式反解的最大ctx)
// 加 577 MiB safety buffer（95% 置信度）
func IdealStartCtx(profile *catalog.DeployProfile, sys *hw.System) int {
	if profile.CtxOverride > 0 {
		return profile.CtxOverride
	}

	gpu := sys.PrimaryGPU()
	if gpu == nil {
		return 4096
	}

	// 模型原生最大 ctx
	nativeMax := profile.NativeCtx
	if nativeMax <= 0 {
		nativeMax = 131072
	}

	// 可用 VRAM：多卡求和（llama-server 自动分层到所有 GPU）
	var freeVRAM float64
	for _, g := range sys.GPUs {
		if g.VRAMFree_MB > 0 {
			freeVRAM += float64(g.VRAMFree_MB)
		} else {
			freeVRAM += float64(g.VRAM_MB) - 1500
		}
	}

	// 减去 577 MiB safety buffer（oobabooga 95% 置信度）× GPU 数量
	availVRAM := freeVRAM - 577*float64(sys.GPUCount())

	if availVRAM < 1000 {
		return 4096
	}

	// 模型参数
	sizeInMB := profile.Size_GB * 1024
	nLayers := profile.Layers
	if nLayers <= 0 {
		nLayers = 32
	}
	gpuLayers := nLayers // 全部放 GPU
	kvHeads := profile.KVHeads
	if kvHeads <= 0 {
		kvHeads = 8
	}
	embeddingDim := profile.EmbeddingDim
	if embeddingDim <= 0 {
		// 从 head_dim * num_heads 估算，或用常见默认值
		embeddingDim = 4096
	}

	// cache_type: iso3=3, q4_0=4, q8_0=8, f16=16
	cacheType := 8.0 // 默认 q8_0+q4_0 取 K 的值
	if profile.HasIsoQuant {
		cacheType = 3.0
	}

	// 反解最大 ctx
	maxCtx := SolveMaxCtx(availVRAM, sizeInMB, nLayers, gpuLayers, kvHeads, embeddingDim, cacheType)

	// 取 min(nativeMax, maxCtx)，对齐到 2 的幂
	ctx := nativeMax
	if maxCtx < ctx {
		ctx = maxCtx
	}
	return alignToPow2(ctx)
}

// alignToPow2 向下对齐到 2 的幂次
func alignToPow2(n int) int {
	powers := []int{524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096}
	for _, p := range powers {
		if n >= p {
			return p
		}
	}
	return 4096
}
