package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadGGUFMeta(t *testing.T) {
	// This test requires a real GGUF file to work.
	// Skip if no test file is available.
	testPaths := []string{
		filepath.Join("D:", "program", "ollama", "test"),
		filepath.Join("D:", "program", "ollama", "ashforge", "models"),
		filepath.Join(os.Getenv("HOME"), ".ashforge", "models"),
	}

	var testFile string
	for _, dir := range testPaths {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if filepath.Ext(e.Name()) == ".gguf" {
				testFile = filepath.Join(dir, e.Name())
				break
			}
		}
		if testFile != "" {
			break
		}
	}

	if testFile == "" {
		t.Skip("No GGUF file found for testing")
	}

	meta, err := ReadGGUFMeta(testFile)
	if err != nil {
		t.Fatalf("ReadGGUFMeta failed: %v", err)
	}

	t.Logf("File: %s", filepath.Base(testFile))
	t.Logf("Architecture: %s", meta.Architecture)
	t.Logf("Name: %s", meta.Name)
	t.Logf("Layers: %d", meta.Layers)
	t.Logf("Experts: %d total, %d active", meta.ExpertsTotal, meta.ExpertsActive)
	t.Logf("Context: %d", meta.ContextLength)
	t.Logf("FileType: %d (%s)", meta.FileType, fileTypeToQuantName(meta.FileType))
	t.Logf("FileSize: %.2f GB", float64(meta.FileSize)/(1024*1024*1024))

	// Basic sanity checks
	if meta.Architecture == "" {
		t.Error("Architecture should not be empty")
	}
	if meta.Layers == 0 {
		t.Error("Layers should be > 0")
	}
	if meta.FileSize == 0 {
		t.Error("FileSize should be > 0")
	}
}

func TestMetaToModelDef(t *testing.T) {
	// Test dense model
	denseMeta := &GGUFMeta{
		Name:          "Test-7B",
		Architecture:  "llama",
		Layers:        32,
		ExpertsTotal:  0,
		ExpertsActive: 0,
		ContextLength: 8192,
		FileType:      15, // Q4_K_M
		FileSize:      4 * 1024 * 1024 * 1024, // 4 GB
	}

	def := metaToModelDef(denseMeta, "test-7b-q4_k_m.gguf")

	if def.Arch != "dense" {
		t.Errorf("Expected arch=dense, got %s", def.Arch)
	}
	if def.ExpertsTotal != 0 {
		t.Errorf("Expected ExpertsTotal=0, got %d", def.ExpertsTotal)
	}
	if def.MoeOffloadTemplate != "" {
		t.Errorf("Dense model should not have MoeOffloadTemplate")
	}
	if len(def.Quantizations) != 1 {
		t.Errorf("Expected 1 quantization, got %d", len(def.Quantizations))
	}
	if def.Quantizations[0].ID != "Q4_K_M" {
		t.Errorf("Expected quant ID=Q4_K_M, got %s", def.Quantizations[0].ID)
	}
	if def.TotalParams_B == 0 {
		t.Error("TotalParams_B should be estimated")
	}

	t.Logf("Dense model: %s, %.1fB params, %s", def.DisplayName, def.TotalParams_B, def.Quantizations[0].ID)

	// Test MoE model
	moeMeta := &GGUFMeta{
		Name:          "Test-MoE-8x7B",
		Architecture:  "qwen2moe",
		Layers:        28,
		ExpertsTotal:  8,
		ExpertsActive: 2,
		ContextLength: 32768,
		FileType:      17, // Q5_K_M
		FileSize:      15 * 1024 * 1024 * 1024, // 15 GB
	}

	moeDef := metaToModelDef(moeMeta, "test-moe-8x7b-q5_k_m.gguf")

	if moeDef.Arch != "moe" {
		t.Errorf("Expected arch=moe, got %s", moeDef.Arch)
	}
	if moeDef.ExpertsTotal != 8 {
		t.Errorf("Expected ExpertsTotal=8, got %d", moeDef.ExpertsTotal)
	}
	if moeDef.ExpertsActive != 2 {
		t.Errorf("Expected ExpertsActive=2, got %d", moeDef.ExpertsActive)
	}
	if moeDef.MoeOffloadTemplate == "" {
		t.Error("MoE model should have MoeOffloadTemplate")
	}
	if moeDef.ActiveParams_B >= moeDef.TotalParams_B {
		t.Errorf("ActiveParams should be less than TotalParams for MoE")
	}

	t.Logf("MoE model: %s, %.1fB total / %.1fB active, %s",
		moeDef.DisplayName, moeDef.TotalParams_B, moeDef.ActiveParams_B, moeDef.Quantizations[0].ID)
}

func TestStopTokensForArch(t *testing.T) {
	tests := []struct {
		arch     string
		expected []string
	}{
		{"llama", []string{"<|eot_id|>", "<|end_header_id|>"}},
		{"qwen2", []string{"<|im_start|>", "<|im_end|>", "<|endoftext|>"}},
		{"qwen2moe", []string{"<|im_start|>", "<|im_end|>", "<|endoftext|>"}},
		{"gemma", []string{"<end_of_turn>", "<eos>"}},
		{"phi3", []string{"<|endoftext|>", "<|end|>"}},
		{"unknown", []string{"</s>"}},
	}

	for _, tt := range tests {
		tokens := stopTokensForArch(tt.arch)
		if len(tokens) != len(tt.expected) {
			t.Errorf("arch=%s: expected %d tokens, got %d", tt.arch, len(tt.expected), len(tokens))
			continue
		}
		for i, tok := range tokens {
			if tok != tt.expected[i] {
				t.Errorf("arch=%s: token[%d] expected %s, got %s", tt.arch, i, tt.expected[i], tok)
			}
		}
	}
}

func TestEstimateParams(t *testing.T) {
	tests := []struct {
		sizeGB   float64
		fileType int
		minParam float64
		maxParam float64
	}{
		{4.0, 15, 6.0, 8.0},   // Q4_K_M: ~7B
		{8.0, 15, 13.0, 16.0}, // Q4_K_M: ~14B
		{15.0, 17, 20.0, 24.0}, // Q5_K_M: ~22B
	}

	for _, tt := range tests {
		sizeBytes := int64(tt.sizeGB * 1024 * 1024 * 1024)
		params := estimateParams(sizeBytes, tt.fileType)
		if params < tt.minParam || params > tt.maxParam {
			t.Errorf("sizeGB=%.1f, fileType=%d: expected params in [%.1f, %.1f], got %.1f",
				tt.sizeGB, tt.fileType, tt.minParam, tt.maxParam, params)
		}
		t.Logf("%.1f GB @ type %d -> %.1fB params", tt.sizeGB, tt.fileType, params)
	}
}
