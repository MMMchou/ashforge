package catalog

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/MMMchou/ashforge/core/hw"
)

// ---------------------------------------------------------------------------
// 1. Registry loading
// ---------------------------------------------------------------------------

func TestLoadRegistry(t *testing.T) {
	store, err := Load()
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if len(store.Models) == 0 {
		t.Fatal("Load() returned zero models; embedded registry.yaml is empty or broken")
	}

	// Spot-check: every model must have an ID and at least one quantization.
	for _, m := range store.Models {
		if m.ID == "" {
			t.Errorf("model with empty ID found (display_name=%q)", m.DisplayName)
		}
		if len(m.Quantizations) == 0 {
			t.Errorf("model %q has no quantizations defined", m.ID)
		}
	}
}

func TestStoreGet_ExactMatch(t *testing.T) {
	store, err := Load()
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}

	first := store.Models[0]
	got, err := store.Get(first.ID)
	if err != nil {
		t.Fatalf("Get(%q) failed: %v", first.ID, err)
	}
	if got.ID != first.ID {
		t.Errorf("Get(%q) returned ID=%q", first.ID, got.ID)
	}
}

func TestStoreGet_NotFound(t *testing.T) {
	store, err := Load()
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}

	_, err = store.Get("nonexistent-model-xyz-999")
	if err == nil {
		t.Fatal("Get() should have returned an error for a nonexistent model")
	}
}

func TestStoreGet_FuzzyMatch(t *testing.T) {
	store, err := Load()
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}

	// Use first model's ID with separators stripped — should still match.
	first := store.Models[0]
	mangled := first.ID
	// Remove dashes/underscores/dots to exercise the fuzzy path.
	for _, sep := range []string{"-", "_", "."} {
		mangled = removeAll(mangled, sep)
	}
	got, err := store.Get(mangled)
	if err != nil {
		t.Fatalf("Get(%q) should fuzzy-match %q but failed: %v", mangled, first.ID, err)
	}
	if got.ID != first.ID {
		t.Errorf("Get(%q) returned ID=%q, want %q", mangled, got.ID, first.ID)
	}
}

func removeAll(s, old string) string {
	out := s
	for i := 0; i < len(out); {
		idx := indexOf(out[i:], old)
		if idx < 0 {
			break
		}
		out = out[:i+idx] + out[i+idx+len(old):]
	}
	return out
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// 2. Strategy / Match
// ---------------------------------------------------------------------------

func TestMatch_EmptyQuantizations(t *testing.T) {
	model := &ModelDef{
		ID:            "test-empty",
		DisplayName:   "Empty Quant Model",
		Quantizations: nil, // intentionally empty
	}
	sys := &hw.System{
		GPUs: []hw.GPUInfo{{VRAM_MB: 24576}},
		RAM:  hw.RAMInfo{Total_MB: 65536},
	}

	_, err := Match(model, sys)
	if err == nil {
		t.Fatal("Match() should fail when Quantizations is empty")
	}
}

func TestMatch_FullGPU_DenseModel(t *testing.T) {
	model := &ModelDef{
		ID:          "test-dense",
		DisplayName: "Test Dense 7B",
		Family:      "llama",
		Arch:        "dense",
		Layers:      32,
		Quantizations: []Quantization{
			{
				ID:         "q4-k-m",
				HFRepo:     "test/repo",
				HFFile:     "*Q4_K_M*",
				Size_GB:    4.0,
				MinVRAM_GB: 5.0,
				MinRAM_GB:  8.0,
			},
			{
				ID:         "q8-0",
				HFRepo:     "test/repo",
				HFFile:     "*Q8_0*",
				Size_GB:    7.5,
				MinVRAM_GB: 9.0,
				MinRAM_GB:  12.0,
			},
		},
	}

	// 24 GB VRAM — both quants fit, should pick highest quality (largest Size_GB).
	sys := &hw.System{
		GPUs: []hw.GPUInfo{{VRAM_MB: 24576}},
		RAM:  hw.RAMInfo{Total_MB: 65536},
	}

	profile, err := Match(model, sys)
	if err != nil {
		t.Fatalf("Match() failed: %v", err)
	}
	if profile.Mode != "full_gpu" {
		t.Errorf("Mode = %q, want full_gpu", profile.Mode)
	}
	// Should pick the larger quant since both fit.
	if profile.Quant != "q8-0" {
		t.Errorf("Quant = %q, want q8-0 (higher quality)", profile.Quant)
	}
}

func TestMatch_InsufficientHardware(t *testing.T) {
	model := &ModelDef{
		ID:          "test-big",
		DisplayName: "Test Big 70B",
		Family:      "llama",
		Arch:        "dense",
		Layers:      80,
		Quantizations: []Quantization{
			{
				ID:         "q4-k-m",
				Size_GB:    40.0,
				MinVRAM_GB: 42.0,
				MinRAM_GB:  50.0,
			},
		},
	}

	// Tiny system: 4 GB VRAM, 8 GB RAM — nothing fits.
	sys := &hw.System{
		GPUs: []hw.GPUInfo{{VRAM_MB: 4096}},
		RAM:  hw.RAMInfo{Total_MB: 8192},
	}

	_, err := Match(model, sys)
	if err == nil {
		t.Fatal("Match() should fail with insufficient hardware")
	}
}

func TestMatch_MoEOffload(t *testing.T) {
	model := &ModelDef{
		ID:                 "test-moe",
		DisplayName:        "Test MoE 30B",
		Family:             "qwen3moe",
		Arch:               "moe",
		Layers:             48,
		ExpertsTotal:       128,
		ExpertsActive:      8,
		MoeOffloadTemplate: ".ffn_.*_exps.=CPU",
		Quantizations: []Quantization{
			{
				ID:         "q4-k-m",
				Size_GB:    17.0,
				MinVRAM_GB: 4.0,
				MinRAM_GB:  16.0,
			},
		},
	}

	// 8 GB VRAM: too small for full model (17 GB) but big enough for shared layers.
	sys := &hw.System{
		GPUs: []hw.GPUInfo{{VRAM_MB: 8192}},
		RAM:  hw.RAMInfo{Total_MB: 32768},
	}

	profile, err := Match(model, sys)
	if err != nil {
		t.Fatalf("Match() failed: %v", err)
	}
	if profile.Mode != "moe_offload" && profile.Mode != "moe_partial" {
		t.Errorf("Mode = %q, want moe_offload or moe_partial for MoE model on limited VRAM", profile.Mode)
	}
}

// ---------------------------------------------------------------------------
// 3. GGUF parsing — kvCount sanity check
// ---------------------------------------------------------------------------

// buildMinimalGGUF creates the smallest valid GGUF v3 header with the given kvCount.
func buildMinimalGGUF(kvCount uint64) []byte {
	var buf bytes.Buffer

	buf.Write(ggufMagic[:])                                         // magic
	binary.Write(&buf, binary.LittleEndian, uint32(3))              // version 3
	binary.Write(&buf, binary.LittleEndian, uint64(0))              // tensor_count
	binary.Write(&buf, binary.LittleEndian, kvCount)                // kv_count

	return buf.Bytes()
}

func TestReadGGUFMeta_KVCountSanityCheck(t *testing.T) {
	// Write a GGUF file with kvCount > 10000, which should be rejected.
	dir := t.TempDir()
	path := filepath.Join(dir, "corrupted.gguf")

	data := buildMinimalGGUF(99999)
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadGGUFMeta(path)
	if err == nil {
		t.Fatal("ReadGGUFMeta should reject kvCount > 10000 as corrupted")
	}
	if !contains(err.Error(), "suspicious kv_count") {
		t.Errorf("error message %q should mention 'suspicious kv_count'", err.Error())
	}
}

func TestReadGGUFMeta_ValidSmallFile(t *testing.T) {
	// A GGUF file with kvCount=0 (valid but empty metadata) should succeed.
	dir := t.TempDir()
	path := filepath.Join(dir, "empty_meta.gguf")

	data := buildMinimalGGUF(0)
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}

	meta, err := ReadGGUFMeta(path)
	if err != nil {
		t.Fatalf("ReadGGUFMeta failed on valid 0-kv file: %v", err)
	}
	if meta.Layers != 0 {
		t.Errorf("Layers = %d, want 0 for empty metadata", meta.Layers)
	}
}

func TestReadGGUFMeta_BadMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "not_gguf.bin")

	if err := os.WriteFile(path, []byte("NOT_GGUF_FILE_CONTENTS"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadGGUFMeta(path)
	if err == nil {
		t.Fatal("ReadGGUFMeta should fail on non-GGUF file")
	}
}

func TestReadGGUFMeta_UnsupportedVersion(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v99.gguf")

	var buf bytes.Buffer
	buf.Write(ggufMagic[:])
	binary.Write(&buf, binary.LittleEndian, uint32(99)) // unsupported version
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadGGUFMeta(path)
	if err == nil {
		t.Fatal("ReadGGUFMeta should fail on unsupported GGUF version")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsInner(s, substr))
}

func containsInner(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// 4. KV cache estimation
// ---------------------------------------------------------------------------

func TestEstimateKVCacheMB_ZeroFields(t *testing.T) {
	// When architectural fields are zero, fallback formula kicks in.
	p := &DeployProfile{Layers: 0, KVHeads: 0, HeadDim: 0}
	got := p.EstimateKVCacheMB(8192, "f16")
	// Fallback: ctxSize / 1024 * 2 = 8192/1024*2 = 16
	if got != 16 {
		t.Errorf("EstimateKVCacheMB fallback = %d, want 16", got)
	}
}

func TestEstimateKVCacheMB_KnownValues(t *testing.T) {
	// Llama-3 8B-like: 32 layers, 8 KV heads, 128 head_dim
	p := &DeployProfile{
		Layers:  32,
		KVHeads: 8,
		HeadDim: 128,
	}

	tests := []struct {
		name    string
		ctx     int
		kvType  string
		wantMB  int
	}{
		{
			name:   "f16_8k",
			ctx:    8192,
			kvType: "f16",
			// 32 * 8 * 128 * 8192 * 2.0 / (1024*1024) = 512 MB per tensor
			wantMB: 512,
		},
		{
			name:   "q8_0_8k",
			ctx:    8192,
			kvType: "q8_0",
			// 32 * 8 * 128 * 8192 * 1.0 / (1024*1024) = 256 MB per tensor
			wantMB: 256,
		},
		{
			name:   "q4_0_8k",
			ctx:    8192,
			kvType: "q4_0",
			// 32 * 8 * 128 * 8192 * 0.5 / (1024*1024) = 128 MB per tensor
			wantMB: 128,
		},
		{
			name:   "iso3_8k",
			ctx:    8192,
			kvType: "iso3",
			// 32 * 8 * 128 * 8192 * 0.375 / (1024*1024) = 96 MB per tensor
			wantMB: 96,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := p.EstimateKVCacheMB(tc.ctx, tc.kvType)
			if got != tc.wantMB {
				t.Errorf("EstimateKVCacheMB(%d, %q) = %d, want %d", tc.ctx, tc.kvType, got, tc.wantMB)
			}
		})
	}
}

func TestSelectKVCacheType(t *testing.T) {
	// Dense model, Llama-3 8B-like, 4.0 GB size.
	p := &DeployProfile{
		Mode:        "full_gpu",
		Size_GB:     4.0,
		Layers:      32,
		KVHeads:     8,
		HeadDim:     128,
		HasIsoQuant: true,
	}

	tests := []struct {
		name   string
		vramMB int
		ctx    int
		wantK  string
		wantV  string
	}{
		{
			// 24 GB VRAM, 8K ctx: f16 KV (512*2=1024 MB) easily fits after 4 GB model + 1 GB reserve.
			name:   "plenty_vram_picks_f16",
			vramMB: 24576,
			ctx:    8192,
			wantK:  "f16",
			wantV:  "f16",
		},
		{
			// 6 GB VRAM, 8K ctx: after 4 GB model + 1 GB reserve = 1 GB free.
			// f16 needs 1024 MB — doesn't fit. q8+q4 = 256+128 = 384 MB — fits.
			name:   "tight_vram_picks_mixed",
			vramMB: 6144,
			ctx:    8192,
			wantK:  "q8_0",
			wantV:  "q4_0",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			k, v := p.SelectKVCacheType(tc.vramMB, tc.ctx)
			if k != tc.wantK || v != tc.wantV {
				t.Errorf("SelectKVCacheType(%d, %d) = (%q, %q), want (%q, %q)",
					tc.vramMB, tc.ctx, k, v, tc.wantK, tc.wantV)
			}
		})
	}
}

func TestSelectKVCacheType_MoEOffload(t *testing.T) {
	// MoE offload: attention VRAM = Size_GB * 0.25 * 1024
	p := &DeployProfile{
		Mode:        "moe_offload",
		Size_GB:     17.0,
		Layers:      48,
		KVHeads:     4,
		HeadDim:     128,
		HasIsoQuant: true,
	}

	// attention VRAM = 17*1024*0.25 = 4352 MB
	// free = 8192 - 4352 = 3840 MB
	// f16 per tensor: 48*4*128*8192*2.0/(1024*1024) = 384 MB, ×2 = 768 MB → fits
	k, v := p.SelectKVCacheType(8192, 8192)
	if k != "f16" || v != "f16" {
		t.Errorf("MoE offload: SelectKVCacheType = (%q, %q), want (f16, f16)", k, v)
	}
}

// ---------------------------------------------------------------------------
// 5. MoE ratio calculation (exercised through estimateMinVRAM & calcMoEMode)
// ---------------------------------------------------------------------------

func TestEstimateMinVRAM_MoE(t *testing.T) {
	// MoE model: 17 GB, 128 experts, 8 active.
	// expertRatio = 1.0 - 8/128 = 0.9375
	// sharedGB = 17.0 * (1.0 - 0.9375*0.9) = 17.0 * (1.0 - 0.84375) = 17.0 * 0.15625 = 2.65625
	// minVRAM = 2.65625 + 1.5 = 4.15625, rounded = 4.2
	got := estimateMinVRAM(17.0, true, 128, 8)
	want := 4.2
	if math.Abs(got-want) > 0.1 {
		t.Errorf("estimateMinVRAM(17, moe, 128, 8) = %.2f, want ~%.2f", got, want)
	}
}

func TestEstimateMinVRAM_Dense(t *testing.T) {
	got := estimateMinVRAM(7.5, false, 0, 0)
	want := 9.0 // 7.5 + 1.5
	if math.Abs(got-want) > 0.01 {
		t.Errorf("estimateMinVRAM(7.5, dense) = %.2f, want %.2f", got, want)
	}
}

func TestCalcMoEMode(t *testing.T) {
	tests := []struct {
		name         string
		expertsTotal int
		layers       int
		sizeGB       float64
		vramMB       int
		wantMode     string
	}{
		{
			name:         "zero_experts_fallback",
			expertsTotal: 0,
			layers:       48,
			sizeGB:       17.0,
			vramMB:       8192,
			wantMode:     "moe_offload",
		},
		{
			name:         "zero_layers_fallback",
			expertsTotal: 128,
			layers:       0,
			sizeGB:       17.0,
			vramMB:       8192,
			wantMode:     "moe_offload",
		},
		{
			name:         "huge_vram_fits_all",
			expertsTotal: 128,
			layers:       48,
			sizeGB:       17.0,
			vramMB:       102400, // 100 GB
			wantMode:     "full_gpu",
		},
		{
			name:         "tiny_vram_all_cpu",
			expertsTotal: 128,
			layers:       48,
			sizeGB:       17.0,
			vramMB:       2048, // 2 GB — not even attention fits comfortably
			wantMode:     "moe_offload",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			profile := &DeployProfile{
				ExpertsTotal: tc.expertsTotal,
				Layers:       tc.layers,
				Size_GB:      tc.sizeGB,
			}
			sys := &hw.System{
				GPUs: []hw.GPUInfo{{VRAM_MB: tc.vramMB}},
			}
			mode, _ := calcMoEMode(profile, sys)
			if mode != tc.wantMode {
				t.Errorf("calcMoEMode(%s) = %q, want %q", tc.name, mode, tc.wantMode)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 6. Helper: isHybridArchitecture
// ---------------------------------------------------------------------------

func TestIsHybridArchitecture(t *testing.T) {
	tests := []struct {
		arch string
		want bool
	}{
		{"qwen3_next", true},
		{"Qwen3Next", true},
		{"jamba", true},
		{"hybrid_v2", true},
		{"rwkv6", true},
		{"mamba", true},
		{"llama", false},
		{"qwen2", false},
		{"gemma2", false},
		{"", false},
	}

	for _, tc := range tests {
		t.Run(tc.arch, func(t *testing.T) {
			got := isHybridArchitecture(tc.arch)
			if got != tc.want {
				t.Errorf("isHybridArchitecture(%q) = %v, want %v", tc.arch, got, tc.want)
			}
		})
	}
}
