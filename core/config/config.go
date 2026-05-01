package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"gopkg.in/yaml.v3"
)

type Config struct {
	HFMirror         string   `yaml:"hf_mirror"`
	LlamaPort        int      `yaml:"llama_port"`
	ProxyPort        int      `yaml:"proxy_port"`
	LogLevel         string   `yaml:"log_level"`
	CacheTTLDays     int      `yaml:"cache_ttl_days,omitempty"`
	ModelDirOverride string   `yaml:"model_dir,omitempty"`
	AltModelDirs     []string `yaml:"alt_model_dirs,omitempty"`
	Priority         string   `yaml:"priority,omitempty"`
	APIKey           string   `yaml:"api_key,omitempty"`
}

var defaultConfig = Config{
	HFMirror:     "https://hf-mirror.com",
	LlamaPort:    21434,
	ProxyPort:    21435,
	LogLevel:     "info",
	CacheTTLDays: 30,
}

var (
	cachedCfg *Config
	cacheMu   sync.Mutex
)

func Dir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not determine home directory: %v; falling back to temp dir\n", err)
		home = os.TempDir()
	}
	return filepath.Join(home, ".ashforge")
}

func BinDir() string     { return filepath.Join(Dir(), "bin") }
func ProfileDir() string { return filepath.Join(Dir(), "profiles") }
func LogDir() string     { return filepath.Join(Dir(), "logs") }

// ModelDir returns the model directory, respecting user override
func ModelDir() string {
	cfg, err := Load()
	if err == nil && cfg.ModelDirOverride != "" {
		return cfg.ModelDirOverride
	}
	return filepath.Join(Dir(), "models")
}

// ModelScanDirs returns all directories to scan for model files:
// the primary ModelDir plus any alt_model_dirs from config.
func ModelScanDirs() []string {
	dirs := []string{ModelDir()}
	cfg, err := Load()
	if err == nil && len(cfg.AltModelDirs) > 0 {
		dirs = append(dirs, cfg.AltModelDirs...)
	}
	return dirs
}

func configPath() string {
	return filepath.Join(Dir(), "config.yaml")
}

func EnsureConfigDir() error {
	dirs := []string{Dir(), BinDir(), ModelDir(), ProfileDir(), LogDir()}
	for _, d := range dirs {
		if err := os.MkdirAll(d, 0755); err != nil {
			return err
		}
	}
	return nil
}

func Load() (*Config, error) {
	cacheMu.Lock()
	defer cacheMu.Unlock()

	if cachedCfg != nil {
		cp := *cachedCfg
		return &cp, nil
	}

	cfg := defaultConfig
	data, err := os.ReadFile(configPath())
	if err != nil {
		if !os.IsNotExist(err) {
			return nil, err
		}
	} else {
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, err
		}
	}

	// Environment variable overrides
	if v := os.Getenv("ASHFORGE_HF_MIRROR"); v != "" {
		cfg.HFMirror = v
	}
	if v := os.Getenv("ASHFORGE_LLAMA_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			cfg.LlamaPort = p
		}
	}
	if v := os.Getenv("ASHFORGE_PROXY_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			cfg.ProxyPort = p
		}
	}
	if v := os.Getenv("ASHFORGE_MODEL_DIR"); v != "" {
		cfg.ModelDirOverride = v
	}
	if v := os.Getenv("ASHFORGE_LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if v := os.Getenv("ASHFORGE_API_KEY"); v != "" {
		cfg.APIKey = v
	}

	cachedCfg = &cfg
	cp := *cachedCfg
	return &cp, nil
}

func Save(cfg *Config) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	err = os.WriteFile(configPath(), data, 0644)
	if err != nil {
		return err
	}

	cacheMu.Lock()
	cachedCfg = nil
	cacheMu.Unlock()

	return nil
}
