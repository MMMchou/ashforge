package optimizer

import (
	"github.com/MMMchou/ashforge/internal/engine"
	"github.com/MMMchou/ashforge/internal/hardware"
	"github.com/MMMchou/ashforge/internal/model"
)

// StartingParams holds initial parameter recommendations
type StartingParams struct {
	Mode       string
	OTFlags    string
	BatchSizes []int
	UBatchSize int
	Threads    int
	CtxSize    int
	KVCacheK   string
	KVCacheV   string
}

// DeriveStartingParams applies the three parameter rules from spec
func DeriveStartingParams(hw *hardware.HardwareProbe, profile *model.DeployProfile) StartingParams {
	vramMB := hw.TotalVRAM_MB() // 多卡总VRAM

	isMoE := profile.Arch == "moe"
	modelFits := profile.Size_GB < float64(vramMB/1024)*0.85

	ctxSize := engine.IdealStartCtx(profile, hw)

	// KV cache 类型：基于 VRAM 计算自动选择（同 warmup/runner）
	kvK, kvV := profile.SelectKVCacheType(vramMB, ctxSize)

	if isMoE && !modelFits {
		moeThreads := hw.CPU.Cores / 2
		if moeThreads < 4 {
			moeThreads = 4
		}
		return StartingParams{
			Mode:       "moe_offload",
			OTFlags:    profile.OTFlags,
			BatchSizes: []int{1024, 2048, 4096},
			UBatchSize: 512,
			Threads:    moeThreads,
			CtxSize:    ctxSize,
			KVCacheK:   kvK,
			KVCacheV:   kvV,
		}
	}

	return StartingParams{
		Mode:       "full_gpu",
		BatchSizes: []int{256, 512, 1024},
		UBatchSize: 512, // 默认 512，warmup 会实测 128 vs 512
		Threads:    2,
		CtxSize:    ctxSize,
		KVCacheK:   kvK,
		KVCacheV:   kvV,
	}
}
