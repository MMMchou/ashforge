package model

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/MMMchou/ashforge/internal/config"
	"github.com/MMMchou/ashforge/internal/download"
)

// EnsureFile ensures the model GGUF file is available locally
func EnsureFile(profile *DeployProfile) (string, error) {
	// Direct path mode: skip download entirely
	if profile.LocalPath != "" {
		if info, err := os.Stat(profile.LocalPath); err == nil && !info.IsDir() {
			return profile.LocalPath, nil
		}
		return "", fmt.Errorf("local path not found: %s", profile.LocalPath)
	}

	cfg, err := config.Load()
	if err != nil {
		return "", err
	}

	// Determine filename from HFFile pattern
	filename, err := resolveFilename(profile.HFRepo, profile.HFFile, cfg.HFMirror)
	if err != nil {
		return "", fmt.Errorf("failed to resolve model filename: %w", err)
	}

	// Check alternative locations first (avoids downloading when file exists elsewhere)
	altPaths := []string{
		filepath.Join("D:", "program", "ollama", "test", filename),
		filepath.Join("D:", "program", "ollama", "ashforge", "models", filename),
	}
	for _, altPath := range altPaths {
		if info, err := os.Stat(altPath); err == nil && info.Size() > 100*1024*1024 {
			fmt.Printf("      Model found at: %s\n", altPath)
			return altPath, nil
		}
	}

	// Check cache directory
	modelPath := filepath.Join(config.ModelDir(), filename)
	if info, err := os.Stat(modelPath); err == nil && info.Size() > 100*1024*1024 {
		return modelPath, nil
	}

	// Download from HuggingFace
	downloadURL := fmt.Sprintf("%s/%s/resolve/main/%s", cfg.HFMirror, profile.HFRepo, filename)
	fmt.Printf("Downloading model: %s\n", filename)
	fmt.Printf("  From: %s\n", downloadURL)

	if err := download.DownloadFile(downloadURL, modelPath, true); err != nil {
		os.Remove(modelPath)
		return "", fmt.Errorf("failed to download model: %w", err)
	}

	return modelPath, nil
}

// resolveFilename resolves a glob pattern like "*UD-Q5_K_XL*" to an actual filename
func resolveFilename(repo, pattern, mirror string) (string, error) {
	// If pattern doesn't contain wildcards, use it directly
	if !strings.Contains(pattern, "*") {
		return pattern, nil
	}

	// Query HuggingFace API for file listing
	apiURL := fmt.Sprintf("%s/api/models/%s", mirror, repo)
	resp, err := http.Get(apiURL)
	if err != nil {
		// Fallback: construct filename from pattern
		return fallbackFilename(repo, pattern), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fallbackFilename(repo, pattern), nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fallbackFilename(repo, pattern), nil
	}

	// Parse response to find matching file
	var modelInfo struct {
		Siblings []struct {
			Filename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.Unmarshal(body, &modelInfo); err != nil {
		return fallbackFilename(repo, pattern), nil
	}

	// Match pattern against filenames
	searchTerm := strings.ReplaceAll(pattern, "*", "")
	for _, sibling := range modelInfo.Siblings {
		if strings.Contains(sibling.Filename, searchTerm) && strings.HasSuffix(sibling.Filename, ".gguf") {
			return sibling.Filename, nil
		}
	}

	return fallbackFilename(repo, pattern), nil
}

// fallbackFilename constructs a reasonable filename when API is unavailable
func fallbackFilename(repo, pattern string) string {
	// Extract model name from repo: "unsloth/Qwen3.6-35B-A3B-GGUF" -> "Qwen3.6-35B-A3B"
	parts := strings.Split(repo, "/")
	repoName := parts[len(parts)-1]
	modelName := strings.TrimSuffix(repoName, "-GGUF")

	// Extract quant from pattern: "*UD-Q5_K_XL*" -> "UD-Q5_K_XL"
	quant := strings.ReplaceAll(pattern, "*", "")
	quant = strings.Trim(quant, "-_")

	return fmt.Sprintf("%s-%s.gguf", modelName, quant)
}
