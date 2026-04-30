package i18n

import "os"

// Lang represents a supported language
type Lang string

const (
	EN Lang = "en"
	ZH Lang = "zh"
)

// current holds the active language
var current = detectLang()

// SetLang overrides the detected language
func SetLang(l Lang) { current = l }

// Current returns the active language
func Current() Lang { return current }

// detectLang auto-detects language from environment
func detectLang() Lang {
	for _, key := range []string{"LANG", "LC_ALL", "LANGUAGE"} {
		val := os.Getenv(key)
		if val != "" {
			if len(val) >= 2 && (val[:2] == "zh" || val[:2] == "ZH") {
				return ZH
			}
		}
	}
	return EN
}

// messages stores all translatable strings
var messages = map[string]map[Lang]string{
	// === Startup ===
	"subtitle": {
		EN: "Auto-tuned LLM Engine v%s · llama.cpp b8864",
		ZH: "自动调优 LLM 引擎 v%s · llama.cpp b8864",
	},
	"by_author": {
		EN: "by Ashan",
		ZH: "by Ashan",
	},

	// === Commands ===
	"cmd_run": {
		EN: "Run a model with auto-tuned parameters",
		ZH: "以自动调优参数运行模型",
	},
	"cmd_list": {
		EN: "List available models",
		ZH: "列出可用模型",
	},
	"cmd_version": {
		EN: "Print Ashforge version",
		ZH: "打印 Ashforge 版本",
	},

	// === Hardware Probe ===
	"probe_detecting": {
		EN: "Detecting hardware...",
		ZH: "正在检测硬件...",
	},
	"probe_gpu_found": {
		EN: "GPU: %s (%d MB VRAM)",
		ZH: "GPU: %s (%d MB 显存)",
	},
	"probe_no_gpu": {
		EN: "No GPU detected, using CPU mode",
		ZH: "未检测到 GPU，使用 CPU 模式",
	},
	"probe_apple_silicon": {
		EN: "Apple Silicon detected: %s (Metal)",
		ZH: "检测到 Apple Silicon: %s (Metal)",
	},

	// === Model ===
	"model_not_found": {
		EN: "Model not found. Run 'ashforge list' to see available models",
		ZH: "未找到模型。运行 'ashforge list' 查看可用模型",
	},
	"model_dir_hint": {
		EN: "Place .gguf files in this directory, Ashforge will auto-detect them",
		ZH: "将 .gguf 文件放入此目录，Ashforge 自动识别",
	},
	"model_config_hint": {
		EN: "Custom model directory: ashforge config set model_dir=/your/path",
		ZH: "自定义模型目录：ashforge config set model_dir=/your/path",
	},

	// === Proxy ===
	"proxy_started": {
		EN: "Ashforge proxy started (port %d)",
		ZH: "Ashforge 代理已启动 (端口 %d)",
	},
	"proxy_stopped": {
		EN: "Ashforge proxy stopped",
		ZH: "Ashforge 代理已停止",
	},

	// === Stream Detection ===
	"repetition_warning": {
		EN: "\n\n⚠️  Repetition detected, auto-stopped. Try rephrasing or refreshing the conversation.",
		ZH: "\n\n⚠️  检测到重复输出，已自动停止。建议重新提问或刷新对话。",
	},

	// === Exit ===
	"goodbye": {
		EN: "Thanks for using Ashforge · github.com/MMMchou/ashforge",
		ZH: "感谢使用 Ashforge · github.com/MMMchou/ashforge",
	},

	// === Warmup ===
	"warmup_mode_hint": {
		EN: "ashforge run %s --mode speed/balanced/context",
		ZH: "ashforge run %s --mode speed/balanced/context",
	},

	// === IDE ===
	"inject_hint": {
		EN: "Auto-configure IDE: ashforge inject",
		ZH: "自动配置 IDE：ashforge inject",
	},

	// === Errors ===
	"port_in_use": {
		EN: "Port %d is already in use, trying next available port...",
		ZH: "端口 %d 已被占用，正在尝试下一个可用端口...",
	},
}

// T returns the translated string for the given key in the current language.
// Falls back to English if the key or language is missing.
func T(key string) string {
	if m, ok := messages[key]; ok {
		if s, ok := m[current]; ok {
			return s
		}
		if s, ok := m[EN]; ok {
			return s
		}
	}
	return key // fallback: return key itself
}
