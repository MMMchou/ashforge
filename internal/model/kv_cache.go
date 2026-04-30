package model

// EstimateKVCacheMB calculates KV cache VRAM usage in MB for one tensor (K or V).
// Formula: n_layers * n_kv_heads * head_dim * ctx_size * bytes_per_element / (1024*1024)
// Multiply by 2 externally if you need both K+V with the same type.
func (p *DeployProfile) EstimateKVCacheMB(ctxSize int, kvType string) int {
	if p.Layers == 0 || p.KVHeads == 0 || p.HeadDim == 0 {
		// Fallback: can't calculate, conservative estimate
		return ctxSize / 1024 * 2 // ~2MB per 1K ctx per tensor
	}

	bpe := bytesPerKVType(kvType)
	totalBytes := float64(p.Layers) * float64(p.KVHeads) *
		float64(p.HeadDim) * float64(ctxSize) * bpe

	return int(totalBytes / (1024 * 1024))
}

// bytesPerKVType returns bytes per element for different KV cache quantization types
func bytesPerKVType(kvType string) float64 {
	switch kvType {
	case "f16", "bf16":
		return 2.0
	case "q8_0":
		return 1.0
	case "q4_0":
		return 0.5
	case "iso3":
		return 0.375 // ~3 bits per element
	default:
		return 2.0
	}
}

// SelectKVCacheType chooses the fastest KV cache type that fits in available VRAM.
// Strategy: try f16 first (fastest), fall back to q8_0+q4_0 if tight, then iso3, then q4_0+q4_0.
//
// For moe_offload: --fit on handles layer allocation; we only need to fit KV cache.
// Use full vramMB as baseline — llama.cpp will carve out model weights itself.
func (p *DeployProfile) SelectKVCacheType(vramMB int, ctxSize int) (k, v string) {
	var freeAfterModel int
	if p.Mode == "moe_offload" || p.Mode == "moe_partial" {
		// Attention VRAM ≈ total_size * 0.30 (Qwen3-MoE ~28/94 layers, Mixtral ~0.25).
		// Minimum floor 1024MB for small models.
		attentionVRAM := int(p.Size_GB * 1024 * 0.30)
		if attentionVRAM < 1024 {
			attentionVRAM = 1024
		}
		// If warmup measured actual VRAM usage, use that instead (more accurate)
		if p.MeasuredVRAM_MB > 0 {
			attentionVRAM = p.MeasuredVRAM_MB
		}
		freeAfterModel = vramMB - attentionVRAM
	} else {
		modelVRAM_MB := int(p.Size_GB * 1024)
		reserveMB := 1024 // 1GB for activations, overhead, etc.
		freeAfterModel = vramMB - modelVRAM_MB - reserveMB
	}

	// Try f16 first (fastest, same as LM Studio default)
	kvF16_MB := p.EstimateKVCacheMB(ctxSize, "f16") * 2 // K + V
	if freeAfterModel >= kvF16_MB {
		return "f16", "f16"
	}

	// f16 doesn't fit → q8_0 K + q4_0 V (balanced)
	kvMixed_MB := p.EstimateKVCacheMB(ctxSize, "q8_0") + p.EstimateKVCacheMB(ctxSize, "q4_0")
	if freeAfterModel >= kvMixed_MB {
		return "q8_0", "q4_0"
	}

	// Still tight → iso3 if available (most compressed)
	if p.HasIsoQuant {
		kvIso3_MB := p.EstimateKVCacheMB(ctxSize, "iso3") * 2
		if freeAfterModel >= kvIso3_MB {
			return "iso3", "iso3"
		}
	}

	// iso3 不可用或仍不够 → q4_0+q4_0（最小标准量化）
	kvQ4_MB := p.EstimateKVCacheMB(ctxSize, "q4_0") * 2
	if freeAfterModel >= kvQ4_MB {
		return "q4_0", "q4_0"
	}

	// Last resort: q4_0+q4_0 anyway, let OOM handler deal with it
	return "q4_0", "q4_0"
}
