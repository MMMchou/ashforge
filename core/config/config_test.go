package config

import (
	"os"
	"testing"
)

func TestLoadDefaults(t *testing.T) {
	// Point HOME at a temp dir so no real config.yaml is found,
	// and clear any env overrides that would interfere.
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)

	envVars := []string{
		"ASHFORGE_HF_MIRROR",
		"ASHFORGE_LLAMA_PORT",
		"ASHFORGE_PROXY_PORT",
		"ASHFORGE_MODEL_DIR",
		"ASHFORGE_LOG_LEVEL",
	}
	for _, e := range envVars {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.HFMirror != "https://hf-mirror.com" {
		t.Errorf("HFMirror = %q, want %q", cfg.HFMirror, "https://hf-mirror.com")
	}
	if cfg.LlamaPort != 21434 {
		t.Errorf("LlamaPort = %d, want %d", cfg.LlamaPort, 21434)
	}
	if cfg.ProxyPort != 21435 {
		t.Errorf("ProxyPort = %d, want %d", cfg.ProxyPort, 21435)
	}
	if cfg.LogLevel != "info" {
		t.Errorf("LogLevel = %q, want %q", cfg.LogLevel, "info")
	}
	if cfg.CacheTTLDays != 30 {
		t.Errorf("CacheTTLDays = %d, want %d", cfg.CacheTTLDays, 30)
	}
	if cfg.ModelDirOverride != "" {
		t.Errorf("ModelDirOverride = %q, want empty", cfg.ModelDirOverride)
	}
}

func TestLoadEnvOverrideProxyPort(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("ASHFORGE_PROXY_PORT", "9999")

	// Clear other env vars to avoid interference.
	for _, e := range []string{"ASHFORGE_HF_MIRROR", "ASHFORGE_LLAMA_PORT", "ASHFORGE_MODEL_DIR", "ASHFORGE_LOG_LEVEL"} {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.ProxyPort != 9999 {
		t.Errorf("ProxyPort = %d, want 9999", cfg.ProxyPort)
	}
}

func TestLoadEnvOverrideLlamaPort(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("ASHFORGE_LLAMA_PORT", "8888")

	for _, e := range []string{"ASHFORGE_HF_MIRROR", "ASHFORGE_PROXY_PORT", "ASHFORGE_MODEL_DIR", "ASHFORGE_LOG_LEVEL"} {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.LlamaPort != 8888 {
		t.Errorf("LlamaPort = %d, want 8888", cfg.LlamaPort)
	}
}

func TestLoadEnvOverrideHFMirror(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("ASHFORGE_HF_MIRROR", "https://custom-mirror.example.com")

	for _, e := range []string{"ASHFORGE_LLAMA_PORT", "ASHFORGE_PROXY_PORT", "ASHFORGE_MODEL_DIR", "ASHFORGE_LOG_LEVEL"} {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.HFMirror != "https://custom-mirror.example.com" {
		t.Errorf("HFMirror = %q, want %q", cfg.HFMirror, "https://custom-mirror.example.com")
	}
}

func TestLoadEnvOverrideLogLevel(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("ASHFORGE_LOG_LEVEL", "debug")

	for _, e := range []string{"ASHFORGE_HF_MIRROR", "ASHFORGE_LLAMA_PORT", "ASHFORGE_PROXY_PORT", "ASHFORGE_MODEL_DIR"} {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.LogLevel != "debug" {
		t.Errorf("LogLevel = %q, want %q", cfg.LogLevel, "debug")
	}
}

func TestLoadEnvOverrideModelDir(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("ASHFORGE_MODEL_DIR", "/custom/models")

	for _, e := range []string{"ASHFORGE_HF_MIRROR", "ASHFORGE_LLAMA_PORT", "ASHFORGE_PROXY_PORT", "ASHFORGE_LOG_LEVEL"} {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.ModelDirOverride != "/custom/models" {
		t.Errorf("ModelDirOverride = %q, want %q", cfg.ModelDirOverride, "/custom/models")
	}
}

func TestLoadEnvInvalidPortFallsBackToDefault(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("ASHFORGE_PROXY_PORT", "not_a_number")

	for _, e := range []string{"ASHFORGE_HF_MIRROR", "ASHFORGE_LLAMA_PORT", "ASHFORGE_MODEL_DIR", "ASHFORGE_LOG_LEVEL"} {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.ProxyPort != 21435 {
		t.Errorf("ProxyPort = %d, want 21435 (default) when env is non-numeric", cfg.ProxyPort)
	}
}

func TestLoadFromYAMLFile(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)

	// Clear all env overrides.
	for _, e := range []string{"ASHFORGE_HF_MIRROR", "ASHFORGE_LLAMA_PORT", "ASHFORGE_PROXY_PORT", "ASHFORGE_MODEL_DIR", "ASHFORGE_LOG_LEVEL"} {
		t.Setenv(e, "")
	}

	// Create a config.yaml in the temp home.
	cfgDir := tmp + "/.ashforge"
	if err := os.MkdirAll(cfgDir, 0755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	yamlContent := []byte(`hf_mirror: "https://yaml-mirror.example.com"
llama_port: 11111
proxy_port: 22222
log_level: "warn"
`)
	if err := os.WriteFile(cfgDir+"/config.yaml", yamlContent, 0644); err != nil {
		t.Fatalf("failed to write config.yaml: %v", err)
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.HFMirror != "https://yaml-mirror.example.com" {
		t.Errorf("HFMirror = %q, want %q", cfg.HFMirror, "https://yaml-mirror.example.com")
	}
	if cfg.LlamaPort != 11111 {
		t.Errorf("LlamaPort = %d, want 11111", cfg.LlamaPort)
	}
	if cfg.ProxyPort != 22222 {
		t.Errorf("ProxyPort = %d, want 22222", cfg.ProxyPort)
	}
	if cfg.LogLevel != "warn" {
		t.Errorf("LogLevel = %q, want %q", cfg.LogLevel, "warn")
	}
}

func TestLoadEnvOverridesYAML(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)

	// Write YAML with one port value, then override with env.
	cfgDir := tmp + "/.ashforge"
	if err := os.MkdirAll(cfgDir, 0755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	yamlContent := []byte(`proxy_port: 22222
`)
	if err := os.WriteFile(cfgDir+"/config.yaml", yamlContent, 0644); err != nil {
		t.Fatalf("failed to write config.yaml: %v", err)
	}

	t.Setenv("ASHFORGE_PROXY_PORT", "33333")
	for _, e := range []string{"ASHFORGE_HF_MIRROR", "ASHFORGE_LLAMA_PORT", "ASHFORGE_MODEL_DIR", "ASHFORGE_LOG_LEVEL"} {
		t.Setenv(e, "")
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.ProxyPort != 33333 {
		t.Errorf("ProxyPort = %d, want 33333 (env should override yaml)", cfg.ProxyPort)
	}
}

func TestLoadMultipleEnvOverrides(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("ASHFORGE_HF_MIRROR", "https://env-mirror.example.com")
	t.Setenv("ASHFORGE_LLAMA_PORT", "7777")
	t.Setenv("ASHFORGE_PROXY_PORT", "8888")
	t.Setenv("ASHFORGE_MODEL_DIR", "/env/models")
	t.Setenv("ASHFORGE_LOG_LEVEL", "trace")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() returned unexpected error: %v", err)
	}

	if cfg.HFMirror != "https://env-mirror.example.com" {
		t.Errorf("HFMirror = %q, want %q", cfg.HFMirror, "https://env-mirror.example.com")
	}
	if cfg.LlamaPort != 7777 {
		t.Errorf("LlamaPort = %d, want 7777", cfg.LlamaPort)
	}
	if cfg.ProxyPort != 8888 {
		t.Errorf("ProxyPort = %d, want 8888", cfg.ProxyPort)
	}
	if cfg.ModelDirOverride != "/env/models" {
		t.Errorf("ModelDirOverride = %q, want %q", cfg.ModelDirOverride, "/env/models")
	}
	if cfg.LogLevel != "trace" {
		t.Errorf("LogLevel = %q, want %q", cfg.LogLevel, "trace")
	}
}
