package model

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/MMMchou/ashforge/internal/config"
)

// HFModelInfo holds cached model info from HuggingFace API
type HFModelInfo struct {
	ModelID       string   `json:"model_id"`
	Architecture  string   `json:"architecture"`
	Tags          []string `json:"tags"`
	StopTokens    []string `json:"stop_tokens,omitempty"`
	ContextLength int      `json:"context_length,omitempty"`
	FetchedAt     string   `json:"fetched_at"`
}

// hfCacheDir returns the path to the HF cache directory
func hfCacheDir() string {
	return filepath.Join(config.Dir(), "model_cache")
}

// hfCachePath returns the cache file path for a given model ID
func hfCachePath(modelID string) string {
	// Sanitize model ID for use as filename
	safe := strings.ReplaceAll(modelID, "/", "_")
	safe = strings.ReplaceAll(safe, "\\", "_")
	return filepath.Join(hfCacheDir(), safe+".json")
}

// FetchHFModelInfoAsync fetches model info from HuggingFace API in the background.
// It does not block the caller. Results are cached to disk.
func FetchHFModelInfoAsync(modelID string) {
	go func() {
		_ = fetchAndCache(modelID)
	}()
}

// LoadHFCache loads cached HF model info if available and not expired (7 days).
func LoadHFCache(modelID string) *HFModelInfo {
	path := hfCachePath(modelID)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var info HFModelInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return nil
	}
	// Check expiry (7 days)
	fetched, err := time.Parse(time.RFC3339, info.FetchedAt)
	if err != nil || time.Since(fetched) > 7*24*time.Hour {
		return nil
	}
	return &info
}

// fetchAndCache queries HuggingFace API and saves the result
func fetchAndCache(modelID string) error {
	// Ensure cache directory exists
	os.MkdirAll(hfCacheDir(), 0755)

	cfg, _ := config.Load()
	mirror := cfg.HFMirror
	if mirror == "" {
		mirror = "https://huggingface.co"
	}

	// Try to find the GGUF repo on HuggingFace
	// Common patterns: "unsloth/{Model}-GGUF", "bartowski/{Model}-GGUF"
	repos := guessHFRepos(modelID)

	for _, repo := range repos {
		apiURL := fmt.Sprintf("%s/api/models/%s", mirror, repo)

		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Get(apiURL)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			continue
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			continue
		}

		// Parse HF API response
		var hfResp struct {
			ID     string   `json:"id"`
			Tags   []string `json:"tags"`
			Config struct {
				Architectures         []string `json:"architectures"`
				MaxPositionEmbeddings int      `json:"max_position_embeddings"`
			} `json:"config"`
		}
		if err := json.Unmarshal(body, &hfResp); err != nil {
			continue
		}

		arch := ""
		if len(hfResp.Config.Architectures) > 0 {
			arch = hfResp.Config.Architectures[0]
		}

		info := &HFModelInfo{
			ModelID:       hfResp.ID,
			Architecture:  arch,
			Tags:          hfResp.Tags,
			ContextLength: hfResp.Config.MaxPositionEmbeddings,
			FetchedAt:     time.Now().Format(time.RFC3339),
		}

		// Infer stop tokens from architecture
		info.StopTokens = inferStopTokens(arch)

		// Save to cache
		data, err := json.MarshalIndent(info, "", "  ")
		if err != nil {
			return err
		}
		return os.WriteFile(hfCachePath(modelID), data, 0644)
	}

	return fmt.Errorf("model not found on HuggingFace: %s", modelID)
}

// guessHFRepos generates possible HuggingFace repo names for a model ID
func guessHFRepos(modelID string) []string {
	// Normalize: "qwen3-8b" -> "Qwen3-8B"
	name := modelID

	// Common GGUF repo patterns
	return []string{
		fmt.Sprintf("unsloth/%s-GGUF", name),
		fmt.Sprintf("bartowski/%s-GGUF", name),
		fmt.Sprintf("Qwen/%s-GGUF", name),
		fmt.Sprintf("google/%s-GGUF", name),
		fmt.Sprintf("microsoft/%s-GGUF", name),
		fmt.Sprintf("meta-llama/%s-GGUF", name),
	}
}

// inferStopTokens infers stop tokens from HuggingFace architecture name
func inferStopTokens(arch string) []string {
	archLower := strings.ToLower(arch)
	switch {
	case strings.Contains(archLower, "qwen"):
		return []string{"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
	case strings.Contains(archLower, "llama"):
		return []string{"<|eot_id|>", "<|end_header_id|>"}
	case strings.Contains(archLower, "gemma"):
		return []string{"<end_of_turn>", "<eos>"}
	case strings.Contains(archLower, "phi"):
		return []string{"<|endoftext|>", "<|end|>"}
	case strings.Contains(archLower, "mistral"), strings.Contains(archLower, "mixtral"):
		return []string{"</s>", "[INST]"}
	case strings.Contains(archLower, "deepseek"):
		return []string{"<|end_of_text|>", "<|im_end|>"}
	case strings.Contains(archLower, "command"):
		return []string{"<|END_OF_TURN_TOKEN|>"}
	default:
		return []string{"</s>"}
	}
}
