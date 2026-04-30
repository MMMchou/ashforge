package llm

import (
	"github.com/MMMchou/ashforge/core/catalog"
	"github.com/MMMchou/ashforge/core/hw"
)

// ThreadsForMode returns the optimal thread count based on inference mode.
// full_gpu: GPU does all the work, CPU just orchestrates — 2 threads suffice.
// moe_offload/moe_partial: CPU handles expert layers, needs real parallelism.
func ThreadsForMode(mode string, sys *hw.System) int {
	if mode == "moe_offload" || mode == "moe_partial" {
		t := sys.CPU.Cores / 2
		if t < 4 {
			t = 4
		}
		return t
	}
	return 2
}

// CacheReuseForCtx returns dynamic cache-reuse size based on context.
// Larger ctx → longer system prompts → more tokens worth caching.
func CacheReuseForCtx(ctxSize int) int {
	switch {
	case ctxSize >= 32768:
		return 1024
	case ctxSize >= 8192:
		return 512
	default:
		return 256
	}
}

// ShouldMlock returns whether mlock should be enabled.
// mlock prevents swapping model pages to disk. Only enabled when
// remaining RAM after model load > 30% of total.
func ShouldMlock(sys *hw.System, profile *catalog.DeployProfile) bool {
	totalMB := float64(sys.RAM.Total_MB)
	freeMB := float64(sys.RAM.Free_MB)
	if totalMB == 0 {
		return false
	}

	var modelRAM_MB float64
	if profile.Mode == "moe_offload" || profile.Mode == "moe_partial" {
		modelRAM_MB = profile.Size_GB * 1024 * 0.9
	} else {
		modelRAM_MB = profile.Size_GB * 1024 * 0.1
	}

	return (freeMB - modelRAM_MB) > totalMB*0.3
}

// ShouldMmap returns whether mmap should be enabled.
// Memory-mapped I/O accelerates model loading. Enabled when
// model size < 70% of available RAM (avoids swap pressure).
func ShouldMmap(sys *hw.System, profile *catalog.DeployProfile) bool {
	freeMB := float64(sys.RAM.Free_MB)
	modelMB := profile.Size_GB * 1024
	return modelMB < freeMB*0.7
}

// ContainsFlag checks if any of the given flags appear in args.
func ContainsFlag(args []string, flags ...string) bool {
	for _, a := range args {
		for _, f := range flags {
			if a == f {
				return true
			}
		}
	}
	return false
}
