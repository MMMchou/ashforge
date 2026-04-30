package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/MMMchou/ashforge/internal/config"
)

// GGUFMeta holds metadata extracted from a GGUF file header.
type GGUFMeta struct {
	Name          string
	Architecture  string
	Layers        int
	KVHeads       int
	HeadDim       int
	EmbeddingDim  int
	ExpertsTotal  int
	ExpertsActive int
	ContextLength int
	FileType      int
	FileSize      int64
	IsHybrid      bool // true if model has recurrent/DeltaNet/SSM layers mixed with attention
}

// GGUF value type constants (GGUF v3 spec)
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// ggufMagic is the GGUF file magic number: "GGUF" in little-endian.
var ggufMagic = [4]byte{0x47, 0x47, 0x55, 0x46}

// ScanLocalModels scans the model directory for .gguf files not in the database
// and returns synthetic ModelDef entries for each unknown file.
func ScanLocalModels(db *ModelDB) []ModelDef {
	modelDir := config.ModelDir()
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return nil
	}

	// Build a set of known filenames from the database for quick lookup.
	known := make(map[string]bool)
	for _, m := range db.Models {
		for _, q := range m.Quantizations {
			if q.HFFile != "" {
				// HFFile may contain wildcards; normalize by stripping them.
				name := strings.ReplaceAll(q.HFFile, "*", "")
				known[strings.ToLower(name)] = true
			}
		}
	}

	var discovered []ModelDef
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			continue
		}

		// Check if this file matches any known model filename.
		lowerName := strings.ToLower(name)
		isKnown := false
		for k := range known {
			if strings.Contains(lowerName, k) || strings.Contains(k, lowerName) {
				isKnown = true
				break
			}
		}
		if isKnown {
			continue
		}

		// Also check by model ID — the filename stem might match a known model ID.
		stem := strings.TrimSuffix(name, filepath.Ext(name))
		stemLower := strings.ToLower(stem)
		idMatch := false
		for _, m := range db.Models {
			if strings.Contains(stemLower, strings.ToLower(m.ID)) {
				idMatch = true
				break
			}
		}
		if idMatch {
			continue
		}

		fullPath := filepath.Join(modelDir, name)
		meta, err := ReadGGUFMeta(fullPath)
		if err != nil {
			continue // skip files we can't parse
		}

		discovered = append(discovered, metaToModelDef(meta, name))
	}

	return discovered
}

// ReadGGUFMeta reads metadata from a GGUF file header.
func ReadGGUFMeta(path string) (*GGUFMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf: %w", err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat gguf: %w", err)
	}

	// Read magic
	var magic [4]byte
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file (magic: %x)", magic)
	}

	// Read version
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	// Read tensor count and metadata KV count
	var tensorCount, kvCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor_count: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &kvCount); err != nil {
		return nil, fmt.Errorf("read kv_count: %w", err)
	}

	meta := &GGUFMeta{
		FileSize: info.Size(),
	}

	// Read KV pairs
	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			return meta, nil // return what we have so far
		}

		var valueType uint32
		if err := binary.Read(f, binary.LittleEndian, &valueType); err != nil {
			return meta, nil
		}

		value, err := readGGUFValue(f, valueType)
		if err != nil {
			return meta, nil
		}

		// Extract fields we care about.
		switch key {
		case "general.architecture":
			if s, ok := value.(string); ok {
				meta.Architecture = s
			}
		case "general.name":
			if s, ok := value.(string); ok {
				meta.Name = s
			}
		case "general.file_type":
			if v, ok := value.(uint32); ok {
				meta.FileType = int(v)
			}
		default:
			// Architecture-specific keys: {arch}.block_count, {arch}.expert_count, etc.
			if meta.Architecture != "" {
				prefix := meta.Architecture + "."
				switch {
				case key == prefix+"block_count":
					if v, ok := value.(uint32); ok {
						meta.Layers = int(v)
					}
				case key == prefix+"expert_count":
					if v, ok := value.(uint32); ok {
						meta.ExpertsTotal = int(v)
					}
				case key == prefix+"expert_used_count":
					if v, ok := value.(uint32); ok {
						meta.ExpertsActive = int(v)
					}
				case key == prefix+"context_length":
					if v, ok := value.(uint32); ok {
						meta.ContextLength = int(v)
					}
				case key == prefix+"attention.head_count_kv":
					if v, ok := value.(uint32); ok {
						meta.KVHeads = int(v)
					}
				case key == prefix+"attention.key_length":
					if v, ok := value.(uint32); ok {
						meta.HeadDim = int(v)
					}
				case key == prefix+"embedding_length":
					if v, ok := value.(uint32); ok {
						meta.EmbeddingDim = int(v)
					}
				}

				// Detect hybrid architecture: tensor keys containing ssm/delta/recurrent
				if strings.Contains(key, ".ssm.") || strings.Contains(key, ".delta") ||
					strings.Contains(key, ".recurrent") || strings.Contains(key, ".mamba") {
					meta.IsHybrid = true
				}
			}
		}
	}

	// Architecture-level hybrid detection (fallback for models without ssm/delta tensor keys)
	if !meta.IsHybrid {
		meta.IsHybrid = isHybridArchitecture(meta.Architecture)
	}

	return meta, nil
}

// isHybridArchitecture returns true for known hybrid (attention + recurrent) architectures.
// These models mix standard attention layers with DeltaNet/SSM/Mamba layers,
// meaning not all layers have KV cache — iso3/TurboQuant breaks on these.
func isHybridArchitecture(arch string) bool {
	a := strings.ToLower(arch)
	// Known hybrid architectures:
	// - qwen3_next / qwen3next: Qwen3.6 with Gated DeltaNet layers
	// - jamba: Jamba (attention + Mamba)
	// - hybrid: generic hybrid marker
	return strings.Contains(a, "qwen3_next") ||
		strings.Contains(a, "qwen3next") ||
		strings.Contains(a, "jamba") ||
		strings.Contains(a, "hybrid") ||
		strings.Contains(a, "rwkv") ||
		strings.Contains(a, "mamba")
}

// readGGUFString reads a GGUF string: uint64 length followed by UTF-8 bytes.
func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // sanity check: 1 MB max string
		return "", fmt.Errorf("gguf string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readGGUFValue reads a GGUF value of the given type.
// Returns the value as an interface{} — callers type-assert what they need.
func readGGUFValue(r io.Reader, vtype uint32) (interface{}, error) {
	switch vtype {
	case ggufTypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case ggufTypeString:
		return readGGUFString(r)
	case ggufTypeArray:
		// Read array element type and count, then skip all elements.
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		// We don't need array values — just skip them.
		for j := uint64(0); j < count; j++ {
			if _, err := readGGUFValue(r, elemType); err != nil {
				return nil, err
			}
		}
		return nil, nil
	default:
		return nil, fmt.Errorf("unknown GGUF value type: %d", vtype)
	}
}

// fileTypeToQuantName maps GGUF file_type enum to human-readable quant names.
func fileTypeToQuantName(ft int) string {
	names := map[int]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
		7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
		10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
		14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
	}
	if name, ok := names[ft]; ok {
		return name
	}
	return fmt.Sprintf("Q%d", ft)
}

// estimateParams estimates total parameters (in billions) from file size and quantization.
func estimateParams(fileSizeBytes int64, fileType int) float64 {
	sizeGB := float64(fileSizeBytes) / (1024 * 1024 * 1024)

	// Bytes per parameter for common quantizations
	var bpp float64
	switch fileType {
	case 12: // Q3_K_M
		bpp = 0.45
	case 14: // Q4_K_S
		bpp = 0.53
	case 2, 15: // Q4_0, Q4_K_M
		bpp = 0.55
	case 16: // Q5_K_S
		bpp = 0.65
	case 8, 9, 17: // Q5_0, Q5_1, Q5_K_M
		bpp = 0.68
	case 18: // Q6_K
		bpp = 0.82
	case 7: // Q8_0
		bpp = 1.05
	case 1: // F16
		bpp = 2.0
	case 0: // F32
		bpp = 4.0
	default:
		bpp = 0.6
	}

	params := sizeGB / bpp
	// Round to one decimal place
	return math.Round(params*10) / 10
}

// stopTokensForArch returns reasonable default stop tokens for a given architecture.
func stopTokensForArch(arch string) []string {
	switch strings.ToLower(arch) {
	case "llama":
		return []string{"<|eot_id|>", "<|end_header_id|>"}
	case "qwen2", "qwen2moe", "qwen3", "qwen3moe":
		return []string{"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
	case "gemma", "gemma2":
		return []string{"<end_of_turn>", "<eos>"}
	case "phi3":
		return []string{"<|endoftext|>", "<|end|>"}
	default:
		return []string{"</s>"}
	}
}

// metaToModelDef converts GGUF metadata into a synthetic ModelDef.
func metaToModelDef(meta *GGUFMeta, filename string) ModelDef {
	// Derive a display name: prefer metadata name, fall back to filename stem.
	displayName := meta.Name
	if displayName == "" {
		displayName = strings.TrimSuffix(filename, filepath.Ext(filename))
	}

	// Build a stable ID from the filename (lowercase, no extension).
	id := strings.ToLower(strings.TrimSuffix(filename, filepath.Ext(filename)))
	id = strings.ReplaceAll(id, " ", "-")

	isMoE := meta.ExpertsTotal > 0
	arch := "dense"
	if isMoE {
		arch = "moe"
	}

	totalParams := estimateParams(meta.FileSize, meta.FileType)
	activeParams := totalParams
	if isMoE && meta.ExpertsTotal > 0 && meta.ExpertsActive > 0 {
		// Rough estimate: active params ≈ total * (active_experts / total_experts)
		// This is a simplification — shared layers aren't scaled, but it's close enough.
		activeParams = math.Round(totalParams*float64(meta.ExpertsActive)/float64(meta.ExpertsTotal)*10) / 10
	}

	quantName := fileTypeToQuantName(meta.FileType)
	sizeGB := float64(meta.FileSize) / (1024 * 1024 * 1024)
	sizeGB = math.Round(sizeGB*100) / 100

	moeTemplate := ""
	if isMoE {
		moeTemplate = ".ffn_.*_exps.=CPU"
	}

	contextLength := meta.ContextLength
	if contextLength == 0 {
		contextLength = 4096 // safe default
	}

	layers := meta.Layers
	if layers == 0 {
		layers = 32 // common default
	}

	def := ModelDef{
		ID:                 id,
		DisplayName:        displayName,
		Family:             meta.Architecture,
		Arch:               arch,
		TotalParams_B:      totalParams,
		ActiveParams_B:     activeParams,
		Layers:             layers,
		ExpertsTotal:       meta.ExpertsTotal,
		ExpertsActive:      meta.ExpertsActive,
		KVHeads:            meta.KVHeads,
		HeadDim:            meta.HeadDim,
		IsHybrid:           meta.IsHybrid,
		MoeOffloadTemplate: moeTemplate,
		StopTokens:         stopTokensForArch(meta.Architecture),
		Quantizations: []Quantization{
			{
				ID:             quantName,
				HFRepo:         "",
				HFFile:         filename,
				Size_GB:        sizeGB,
				QualityLossPct: 0, // unknown
				MinVRAM_GB:     estimateMinVRAM(sizeGB, isMoE, meta.ExpertsTotal, meta.ExpertsActive),
				MinRAM_GB:      estimateMinRAM(sizeGB, isMoE),
			},
		},
	}

	return def
}

// estimateMinVRAM calculates minimum VRAM needed.
// MoE models with offload only need shared layers in VRAM (~10-15% of total),
// dense models need the full model + overhead.
func estimateMinVRAM(sizeGB float64, isMoE bool, expertsTotal, expertsActive int) float64 {
	if isMoE && expertsTotal > 0 {
		// MoE offload: only shared layers + active expert weights in VRAM
		// Shared layers are roughly (1 - expert_ratio) of total size
		// For 128-expert models like Qwen3-30B-A3B, shared layers ~= 5-10% of model
		expertRatio := 1.0 - float64(expertsActive)/float64(expertsTotal)
		sharedGB := sizeGB * (1.0 - expertRatio*0.9) // ~10% shared + active experts
		minVRAM := sharedGB + 1.5                     // + KV cache + overhead
		if minVRAM < 4.0 {
			minVRAM = 4.0
		}
		return math.Round(minVRAM*10) / 10
	}
	return sizeGB + 1.5
}

// estimateMinRAM calculates minimum RAM needed.
// MoE offload puts experts in RAM, so RAM needs to hold the full model.
func estimateMinRAM(sizeGB float64, isMoE bool) float64 {
	if isMoE {
		// MoE offload: full model in RAM + system overhead
		// But RAM requirement is lower than sizeGB+3 because GPU handles shared layers
		return math.Round((sizeGB*0.8+2.0)*10) / 10
	}
	return sizeGB + 3.0
}
