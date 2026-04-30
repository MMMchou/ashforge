package config

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type Config struct {
	HFMirror  string `yaml:"hf_mirror"`
	LlamaPort int    `yaml:"llama_port"`
	ProxyPort int    `yaml:"proxy_port"`
	LogLevel  string `yaml:"log_level"`
	ModelDirOverride string `yaml:"model_dir,omitempty"` // 自定义模型存储路径，留空则用默认 ~/.ashforge/models/
	Priority string `yaml:"priority,omitempty"` // 模式偏好: speed/balanced/context，默认 balanced
}

var defaultConfig = Config{
	HFMirror:  "https://hf-mirror.com",
	LlamaPort: 21434,
	ProxyPort: 21435,
	LogLevel:  "info",
}

func Dir() string {
	home, _ := os.UserHomeDir()
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
	cfg := defaultConfig
	data, err := os.ReadFile(configPath())
	if err != nil {
		if os.IsNotExist(err) {
			return &cfg, nil
		}
		return nil, err
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func Save(cfg *Config) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(configPath(), data, 0644)
}
