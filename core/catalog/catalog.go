package catalog

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/MMMchou/ashforge/core/config"
	"gopkg.in/yaml.v3"
)

//go:embed registry.yaml
var embeddedRegistry embed.FS

// Store is the model knowledge base
type Store struct {
	Models []ModelDef `yaml:"models"`
}

// ModelDef defines a model in the knowledge base
type ModelDef struct {
	ID                 string         `yaml:"id"`
	DisplayName        string         `yaml:"display_name"`
	Family             string         `yaml:"family"`
	Arch               string         `yaml:"arch"` // "moe" or "dense"
	TotalParams_B      float64        `yaml:"total_params_b"`
	ActiveParams_B     float64        `yaml:"active_params_b"`
	Layers             int            `yaml:"layers"`
	ExpertsTotal       int            `yaml:"experts_total"`
	ExpertsActive      int            `yaml:"experts_active"`
	KVHeads            int            `yaml:"kv_heads"`
	HeadDim            int            `yaml:"head_dim"`
	IsHybrid           bool           `yaml:"is_hybrid"` // hybrid attention+recurrent (DeltaNet/SSM/Mamba)
	NativeMTP          bool           `yaml:"native_mtp"`
	Quantizations      []Quantization `yaml:"quantizations"`
	MoeOffloadTemplate string         `yaml:"moe_offload_template"`
	StopTokens         []string       `yaml:"stop_tokens"`
}

// Quantization defines a quantization variant
type Quantization struct {
	ID             string  `yaml:"id"`
	HFRepo         string  `yaml:"hf_repo"`
	HFFile         string  `yaml:"hf_file"`
	Size_GB        float64 `yaml:"size_gb"`
	QualityLossPct float64 `yaml:"quality_loss_pct"`
	MinVRAM_GB     float64 `yaml:"min_vram_gb"`
	MinRAM_GB      float64 `yaml:"min_ram_gb"`
}

// Load loads the model knowledge base
func Load() (*Store, error) {
	db := &Store{}

	// Load embedded registry
	data, err := embeddedRegistry.ReadFile("registry.yaml")
	if err != nil {
		return nil, fmt.Errorf("failed to read embedded registry.yaml: %w", err)
	}
	if err := yaml.Unmarshal(data, db); err != nil {
		return nil, fmt.Errorf("failed to parse embedded registry.yaml: %w", err)
	}

	// Try to load user override
	userPath := filepath.Join(config.Dir(), "models.yaml")
	if userData, err := os.ReadFile(userPath); err == nil {
		var userDB Store
		if err := yaml.Unmarshal(userData, &userDB); err == nil {
			// Merge: user models override embedded ones by ID
			for _, um := range userDB.Models {
				found := false
				for i, em := range db.Models {
					if em.ID == um.ID {
						db.Models[i] = um
						found = true
						break
					}
				}
				if !found {
					db.Models = append(db.Models, um)
				}
			}
		}
	}

	return db, nil
}

// List returns all models
func (db *Store) List() []ModelDef {
	return db.Models
}

// Get finds a model by name (fuzzy matching)
func (db *Store) Get(name string) (*ModelDef, error) {
	name = strings.ToLower(strings.TrimSpace(name))

	// Exact match first
	for i := range db.Models {
		if strings.ToLower(db.Models[i].ID) == name {
			return &db.Models[i], nil
		}
	}

	// Partial match
	for i := range db.Models {
		if strings.Contains(strings.ToLower(db.Models[i].ID), name) {
			return &db.Models[i], nil
		}
		if strings.Contains(strings.ToLower(db.Models[i].DisplayName), name) {
			return &db.Models[i], nil
		}
	}

	// Fuzzy: remove separators and try again
	normalized := strings.ReplaceAll(name, "-", "")
	normalized = strings.ReplaceAll(normalized, "_", "")
	normalized = strings.ReplaceAll(normalized, ".", "")
	for i := range db.Models {
		id := strings.ReplaceAll(strings.ToLower(db.Models[i].ID), "-", "")
		id = strings.ReplaceAll(id, "_", "")
		id = strings.ReplaceAll(id, ".", "")
		if strings.Contains(id, normalized) {
			return &db.Models[i], nil
		}
	}

	return nil, fmt.Errorf("model '%s' not found. Run 'ashforge list' to see available models", name)
}

// IsMoE returns whether the model is a Mixture of Experts model
func (m *ModelDef) IsMoE() bool {
	return m.Arch == "moe"
}

// GetOrDetect tries the database first, then falls back to GGUF auto-detection.
func (db *Store) GetOrDetect(name string) (*ModelDef, error) {
	// 0. Direct file path: if the input is an existing .gguf file, read it directly
	if strings.HasSuffix(strings.ToLower(name), ".gguf") {
		if info, err := os.Stat(name); err == nil && !info.IsDir() {
			meta, err := ReadGGUFMeta(name)
			if err != nil {
				return nil, fmt.Errorf("failed to read GGUF file: %w", err)
			}
			def := metaToModelDef(meta, filepath.Base(name))
			return &def, nil
		}
		// Also check in model directory
		modelDirPath := filepath.Join(config.ModelDir(), name)
		if info, err := os.Stat(modelDirPath); err == nil && !info.IsDir() {
			meta, err := ReadGGUFMeta(modelDirPath)
			if err != nil {
				return nil, fmt.Errorf("failed to read GGUF file: %w", err)
			}
			def := metaToModelDef(meta, filepath.Base(name))
			return &def, nil
		}
	}

	// 1. Try exact/fuzzy match in database
	if def, err := db.Get(name); err == nil {
		return def, nil
	}

	// 2. Try to find a matching .gguf file in model directories
	normalized := strings.ToLower(strings.TrimSpace(name))

	dirs := config.ModelScanDirs()

	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			fname := entry.Name()
			if !strings.HasSuffix(strings.ToLower(fname), ".gguf") {
				continue
			}
			stem := strings.ToLower(strings.TrimSuffix(fname, filepath.Ext(fname)))
			// Match if name is contained in filename or vice versa
			if strings.Contains(stem, normalized) || strings.Contains(normalized, stem) {
				fullPath := filepath.Join(dir, fname)
				meta, err := ReadGGUFMeta(fullPath)
				if err != nil {
					continue
				}
				def := metaToModelDef(meta, fname)
				return &def, nil
			}
		}
	}

	modelDir := config.ModelDir()
	return nil, fmt.Errorf("model '%s' not found in database or local files.\nRun 'ashforge list' to see available models.\nOr place a .gguf file in %s", name, modelDir)
}

// ListAll returns all models: database + auto-detected local files.
func (db *Store) ListAll() []ModelDef {
	all := make([]ModelDef, len(db.Models))
	copy(all, db.Models)

	// Append auto-detected models not already in the database
	discovered := ScanLocalModels(db)
	all = append(all, discovered...)

	return all
}
