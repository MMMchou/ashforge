package ide

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/BurntSushi/toml"
)

// IDE represents a detected IDE
type IDE struct {
	Name       string
	ConfigPath string
	Detected   bool
}

// Detect finds installed IDEs
func Detect() []IDE {
	return []IDE{
		// detectClaudeCode(), // Skip for now
		detectCodex(),
		detectCursor(),
	}
}

// Inject configures an IDE to use Ashforge
func Inject(ide *IDE, port int, apiKey string) error {
	switch ide.Name {
	case "Claude Code":
		return injectClaudeCode(ide.ConfigPath, port, apiKey)
	case "Codex CLI":
		return injectCodex(ide.ConfigPath, port, apiKey)
	case "Cursor":
		return injectCursor(ide.ConfigPath, port, apiKey)
	default:
		return fmt.Errorf("unsupported IDE: %s", ide.Name)
	}
}

// Undo restores original IDE config from backup
func Undo(ide *IDE) error {
	backupPath := ide.ConfigPath + ".ashforge-backup"
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		return fmt.Errorf("no backup found for %s", ide.Name)
	}
	data, err := os.ReadFile(backupPath)
	if err != nil {
		return err
	}
	return os.WriteFile(ide.ConfigPath, data, 0644)
}

// backup creates a backup of the config file
func backup(path string) error {
	backupPath := path + ".ashforge-backup"
	if _, err := os.Stat(backupPath); err == nil {
		return nil // Backup already exists
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return os.WriteFile(backupPath, data, 0644)
}

// --- Claude Code ---

func detectClaudeCode() IDE {
	home, _ := os.UserHomeDir()
	configPath := filepath.Join(home, ".claude", "settings.json")
	_, err := os.Stat(configPath)
	return IDE{
		Name:       "Claude Code",
		ConfigPath: configPath,
		Detected:   err == nil,
	}
}

func injectClaudeCode(configPath string, port int, apiKey string) error {
	if err := backup(configPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	// Read existing config or create new
	var cfg map[string]interface{}
	if data, err := os.ReadFile(configPath); err == nil {
		json.Unmarshal(data, &cfg)
	}
	if cfg == nil {
		cfg = make(map[string]interface{})
	}

	// Set env vars
	env, ok := cfg["env"].(map[string]interface{})
	if !ok {
		env = make(map[string]interface{})
	}
	env["ANTHROPIC_BASE_URL"] = fmt.Sprintf("http://127.0.0.1:%d", port)
	env["ANTHROPIC_AUTH_TOKEN"] = apiKey
	cfg["env"] = env

	// Write back
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	os.MkdirAll(filepath.Dir(configPath), 0755)
	return os.WriteFile(configPath, data, 0644)
}

// --- Codex CLI ---

func detectCodex() IDE {
	home, _ := os.UserHomeDir()
	configPath := filepath.Join(home, ".codex", "config.toml")
	_, err := os.Stat(configPath)
	return IDE{
		Name:       "Codex CLI",
		ConfigPath: configPath,
		Detected:   err == nil,
	}
}

func injectCodex(configPath string, port int, apiKey string) error {
	if err := backup(configPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	// Read existing config
	var cfg map[string]interface{}
	if data, err := os.ReadFile(configPath); err == nil {
		toml.Unmarshal(data, &cfg)
	}
	if cfg == nil {
		cfg = make(map[string]interface{})
	}

	// Set model provider
	cfg["model_provider"] = "ashforge"

	// Set provider config
	providers, ok := cfg["model_providers"].(map[string]interface{})
	if !ok {
		providers = make(map[string]interface{})
	}
	providers["ashforge"] = map[string]interface{}{
		"base_url": fmt.Sprintf("http://127.0.0.1:%d", port),
		"api_key":  apiKey,
		"wire_api": "responses",
	}
	cfg["model_providers"] = providers

	// Write back as TOML
	var buf strings.Builder
	enc := toml.NewEncoder(&buf)
	if err := enc.Encode(cfg); err != nil {
		return err
	}
	os.MkdirAll(filepath.Dir(configPath), 0755)
	return os.WriteFile(configPath, []byte(buf.String()), 0644)
}

// --- Cursor ---

func detectCursor() IDE {
	var configPath string
	if runtime.GOOS == "windows" {
		appData := os.Getenv("APPDATA")
		configPath = filepath.Join(appData, "Cursor", "User", "settings.json")
	} else {
		home, _ := os.UserHomeDir()
		configPath = filepath.Join(home, ".config", "Cursor", "User", "settings.json")
	}
	_, err := os.Stat(configPath)
	return IDE{
		Name:       "Cursor",
		ConfigPath: configPath,
		Detected:   err == nil,
	}
}

func injectCursor(configPath string, port int, apiKey string) error {
	if err := backup(configPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	// Read existing config
	var cfg map[string]interface{}
	if data, err := os.ReadFile(configPath); err == nil {
		json.Unmarshal(data, &cfg)
	}
	if cfg == nil {
		cfg = make(map[string]interface{})
	}

	// Set OpenAI-compatible base URL
	cfg["openai.apiBase"] = fmt.Sprintf("http://127.0.0.1:%d/v1", port)
	cfg["openai.apiKey"] = apiKey

	// Write back
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(configPath, data, 0644)
}
