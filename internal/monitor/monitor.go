package monitor

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/MMMchou/ashforge/internal/proxy"
)

// MonitorData holds all collected metrics
type MonitorData struct {
	VRAM_Used_MB  int
	VRAM_Total_MB int
	RAM_Used_MB   uint64
	RAM_Total_MB  uint64
	CtxUsed       int
	CtxTotal      int
	TokPerSec     float64
	GPU_Temp_C    int
	GPU_Util_Pct  int     // GPU 利用率百分比
	CPU_Pct       float64
	Alerts        []string
}

// Collect gathers all metrics from GPU, system, and llama-server
func Collect(backendPort int) MonitorData {
	var d MonitorData
	collectGPU(&d)
	collectSystem(&d)
	collectFromMetrics(&d, backendPort)
	d.Alerts = checkAlerts(d)
	return d
}

// collectGPU reads VRAM, temperature, and utilization from nvidia-smi
func collectGPU(d *MonitorData) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return
	}
	// 多卡求和：VRAM 累加，温度/利用率取最大值
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) >= 4 {
			used, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
			total, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
			temp, _ := strconv.Atoi(strings.TrimSpace(parts[2]))
			util, _ := strconv.Atoi(strings.TrimSpace(parts[3]))
			d.VRAM_Used_MB += used
			d.VRAM_Total_MB += total
			if temp > d.GPU_Temp_C {
				d.GPU_Temp_C = temp
			}
			if util > d.GPU_Util_Pct {
				d.GPU_Util_Pct = util
			}
		}
	}
}

// collectFromMetrics reads context usage and speed from llama-server /metrics
func collectFromMetrics(d *MonitorData, port int) {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", port))
	if err != nil {
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "#") {
			continue
		}
		switch {
		case strings.HasPrefix(line, "llamacpp:kv_cache_used_cells "):
			d.CtxUsed = parseMetricInt(line)
		case strings.HasPrefix(line, "llamacpp:kv_cache_tokens "):
			if d.CtxUsed == 0 {
				d.CtxUsed = parseMetricInt(line)
			}
		case strings.HasPrefix(line, "llamacpp:prompt_tokens_seconds "):
			// prompt processing speed (less relevant for user)
		case strings.HasPrefix(line, "llamacpp:tokens_predicted_seconds "):
			d.TokPerSec = parseMetricFloat(line)
		}
	}

	// Get n_ctx from /slots endpoint if not available from metrics
	if d.CtxTotal == 0 {
		d.CtxTotal = fetchCtxFromSlots(port)
	}
}

// fetchCtxFromSlots gets n_ctx from /slots endpoint
func fetchCtxFromSlots(port int) int {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/slots", port))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()

	var slots []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return 0
	}

	if len(slots) > 0 {
		if nCtx, ok := slots[0]["n_ctx"].(float64); ok {
			return int(nCtx)
		}
	}
	return 0
}

func parseMetricInt(line string) int {
	parts := strings.Fields(line)
	if len(parts) >= 2 {
		v, _ := strconv.ParseFloat(parts[len(parts)-1], 64)
		return int(v)
	}
	return 0
}

func parseMetricFloat(line string) float64 {
	parts := strings.Fields(line)
	if len(parts) >= 2 {
		v, _ := strconv.ParseFloat(parts[len(parts)-1], 64)
		return v
	}
	return 0
}

// collectSystem reads RAM and CPU usage (platform-specific)
func collectSystem(d *MonitorData) {
	if runtime.GOOS == "windows" {
		collectSystemWindows(d)
	} else {
		collectSystemLinux(d)
	}
}

func collectSystemWindows(d *MonitorData) {
	// Use wmic for memory
	out, err := exec.Command("wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:csv").Output()
	if err != nil {
		return
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	for _, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) >= 3 {
			free, _ := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64)
			total, _ := strconv.ParseUint(strings.TrimSpace(parts[2]), 10, 64)
			if total > 0 {
				d.RAM_Total_MB = total / 1024
				d.RAM_Used_MB = (total - free) / 1024
			}
		}
	}
}

func collectSystemLinux(d *MonitorData) {
	out, err := exec.Command("cat", "/proc/meminfo").Output()
	if err != nil {
		return
	}
	var total, available uint64
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			total = parseMemInfoKB(line)
		} else if strings.HasPrefix(line, "MemAvailable:") {
			available = parseMemInfoKB(line)
		}
	}
	if total > 0 {
		d.RAM_Total_MB = total / 1024
		d.RAM_Used_MB = (total - available) / 1024
	}
}

func parseMemInfoKB(line string) uint64 {
	parts := strings.Fields(line)
	if len(parts) >= 2 {
		v, _ := strconv.ParseUint(parts[1], 10, 64)
		return v
	}
	return 0
}

// checkAlerts returns active alert messages
func checkAlerts(d MonitorData) []string {
	var alerts []string

	// Alert 1: VRAM > 90%
	if d.VRAM_Total_MB > 0 {
		pct := float64(d.VRAM_Used_MB) / float64(d.VRAM_Total_MB)
		if pct > 0.9 {
			alerts = append(alerts, "⚠️  显存即将用满，推理可能中断")
		}
	}

	// Alert 2: Context > 80%
	if d.CtxTotal > 0 {
		pct := float64(d.CtxUsed) / float64(d.CtxTotal)
		if pct > 0.8 {
			alerts = append(alerts, fmt.Sprintf("⚠️  上下文已用 %.0f%%，建议新开对话", pct*100))
		}
	}

	// Alert 3: GPU temp > 85°C
	if d.GPU_Temp_C > 85 {
		alerts = append(alerts, fmt.Sprintf("⚠️  GPU 温度 %d℃，可能触发降频", d.GPU_Temp_C))
	}

	return alerts
}

// Monitor runs periodic collection and rendering
type Monitor struct {
	backendPort int
	modelName   string
	stopCh      chan struct{}
	running     bool
	ctxTotal    int      // set from config
	lastAlerts  []string // dedup: only print new alerts
	ParamInfo   string   // runtime params summary (e.g., "64K ctx · f16 KV · ub512 · mlock")
}

// NewMonitor creates a new monitor
func NewMonitor(backendPort int, modelName string) *Monitor {
	return &Monitor{
		backendPort: backendPort,
		modelName:   modelName,
		stopCh:      make(chan struct{}),
	}
}

// StartAsync starts the monitor in a background goroutine
func (m *Monitor) StartAsync() {
	m.running = true
	go m.run()
}

// Stop stops the monitor
func (m *Monitor) Stop() {
	if m.running {
		m.running = false
		close(m.stopCh)
	}
}

func (m *Monitor) run() {
	// Initial delay to let server stabilize
	select {
	case <-time.After(3 * time.Second):
	case <-m.stopCh:
		return
	}

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	// 清屏并隐藏光标
	fmt.Print("\033[2J\033[H\033[?25l")

	for {
		select {
		case <-m.stopCh:
			// 恢复光标
			fmt.Print("\033[?25h")
			return
		case <-ticker.C:
			d := Collect(m.backendPort)
			m.renderPanel(d)

			// 只打印新告警
			for _, alert := range d.Alerts {
				isNew := true
				for _, prev := range m.lastAlerts {
					if prev == alert {
						isNew = false
						break
					}
				}
				if isNew {
					fmt.Printf("\n%s", alert)
				}
			}
			m.lastAlerts = d.Alerts
		}
	}
}

// renderPanel 渲染实时监控面板
func (m *Monitor) renderPanel(d MonitorData) {
	// 移动光标到顶部
	fmt.Print("\033[H")

	// 标题行
	fmt.Print("\033[90m─ 实时监控 · ")
	if d.TokPerSec > 0 {
		fmt.Print("推理中")
	} else {
		fmt.Print("空载")
	}
	fmt.Print(" ─────────────────── 每 2s 刷新 ─\033[0m\n")

	// 参数行（如果有）
	if m.ParamInfo != "" {
		fmt.Printf("  \033[36m%s\033[0m\n", m.ParamInfo)
	}

	// 指标行
	fmt.Print("  ")
	fmt.Print("\033[34m速度\033[0m          ")
	fmt.Print("\033[32m显存\033[0m           ")
	fmt.Print("\033[32m内存\033[0m        ")
	fmt.Print("\033[35mGPU\033[0m      ")
	fmt.Print("\033[32m温度\033[0m\n")

	// 数值行
	fmt.Print("  ")

	// 速度
	if d.TokPerSec > 0 {
		fmt.Printf("\033[34m%.1f tok/s\033[0m   ", d.TokPerSec)
	} else {
		fmt.Print("\033[90m— tok/s\033[0m     ")
	}

	// 显存
	if d.VRAM_Total_MB > 0 {
		vramPct := float64(d.VRAM_Used_MB) / float64(d.VRAM_Total_MB)
		vramColor := "\033[32m" // 绿色
		if vramPct > 0.9 {
			vramColor = "\033[31m" // 红色
		} else if vramPct > 0.7 {
			vramColor = "\033[33m" // 黄色
		}
		fmt.Printf("%s%.1f/%.0f GB\033[0m    ", vramColor, float64(d.VRAM_Used_MB)/1024, float64(d.VRAM_Total_MB)/1024)
	} else {
		fmt.Print("\033[90m—\033[0m           ")
	}

	// 内存
	if d.RAM_Total_MB > 0 {
		ramPct := float64(d.RAM_Used_MB) / float64(d.RAM_Total_MB)
		ramColor := "\033[32m"
		if ramPct > 0.9 {
			ramColor = "\033[31m"
		} else if ramPct > 0.7 {
			ramColor = "\033[33m"
		}
		fmt.Printf("%s%.1f/%.0f GB\033[0m   ", ramColor, float64(d.RAM_Used_MB)/1024, float64(d.RAM_Total_MB)/1024)
	} else {
		fmt.Print("\033[90m—\033[0m        ")
	}

	// GPU 利用率
	if d.GPU_Util_Pct >= 0 {
		fmt.Printf("\033[35m%d%%\033[0m      ", d.GPU_Util_Pct)
	} else {
		fmt.Print("\033[90m—\033[0m      ")
	}

	// 温度
	if d.GPU_Temp_C > 0 {
		tempColor := "\033[32m"
		if d.GPU_Temp_C > 85 {
			tempColor = "\033[31m"
		} else if d.GPU_Temp_C > 75 {
			tempColor = "\033[33m"
		}
		fmt.Printf("%s%d°C\033[0m\n", tempColor, d.GPU_Temp_C)
	} else {
		fmt.Print("\033[90m—\033[0m\n")
	}

	// 进度条行
	fmt.Print("  ")

	// 速度进度条
	if d.TokPerSec > 0 {
		speedBar := int(d.TokPerSec / 100 * 10)
		if speedBar > 10 {
			speedBar = 10
		}
		fmt.Printf("\033[34m[%s%s]\033[0m ", strings.Repeat("=", speedBar), strings.Repeat(".", 10-speedBar))
	} else {
		fmt.Print("\033[90m[..........]\033[0m ")
	}

	// 显存进度条
	if d.VRAM_Total_MB > 0 {
		vramPct := float64(d.VRAM_Used_MB) / float64(d.VRAM_Total_MB)
		vramBar := int(vramPct * 10)
		vramColor := "\033[32m"
		if vramPct > 0.9 {
			vramColor = "\033[31m"
		} else if vramPct > 0.7 {
			vramColor = "\033[33m"
		}
		fmt.Printf("%s[%s%s]\033[0m ", vramColor, strings.Repeat("=", vramBar), strings.Repeat(".", 10-vramBar))
	} else {
		fmt.Print("\033[90m[..........]\033[0m ")
	}

	// 内存进度条
	if d.RAM_Total_MB > 0 {
		ramPct := float64(d.RAM_Used_MB) / float64(d.RAM_Total_MB)
		ramBar := int(ramPct * 10)
		ramColor := "\033[32m"
		if ramPct > 0.9 {
			ramColor = "\033[31m"
		} else if ramPct > 0.7 {
			ramColor = "\033[33m"
		}
		fmt.Printf("%s[%s%s]\033[0m ", ramColor, strings.Repeat("=", ramBar), strings.Repeat(".", 10-ramBar))
	} else {
		fmt.Print("\033[90m[..........]\033[0m ")
	}

	// GPU 进度条
	if d.GPU_Util_Pct >= 0 {
		gpuBar := d.GPU_Util_Pct / 10
		fmt.Printf("\033[35m[%s%s]\033[0m ", strings.Repeat("=", gpuBar), strings.Repeat(".", 10-gpuBar))
	} else {
		fmt.Print("\033[90m[..........]\033[0m ")
	}

	// 温度进度条
	if d.GPU_Temp_C > 0 {
		tempBar := d.GPU_Temp_C / 10
		if tempBar > 10 {
			tempBar = 10
		}
		tempColor := "\033[32m"
		if d.GPU_Temp_C > 85 {
			tempColor = "\033[31m"
		} else if d.GPU_Temp_C > 75 {
			tempColor = "\033[33m"
		}
		fmt.Printf("%s[%s%s]\033[0m", tempColor, strings.Repeat("=", tempBar), strings.Repeat(".", 10-tempBar))
	} else {
		fmt.Print("\033[90m[..........]\033[0m")
	}
	fmt.Println()

	fmt.Print("\033[90m─────────────────────────────────────────────────────────\033[0m\n")

	// 上下文进度条（如果有 n_ctx）
	if d.CtxTotal > 0 {
		ctxPct := 0.0
		ctxBar := 0
		if d.CtxUsed > 0 {
			ctxPct = float64(d.CtxUsed) / float64(d.CtxTotal)
			ctxBar = int(ctxPct * 20)
			if ctxBar > 20 {
				ctxBar = 20
			}
		}

		ctxColor := "\033[32m"
		if ctxPct > 0.8 {
			ctxColor = "\033[31m"
		} else if ctxPct > 0.6 {
			ctxColor = "\033[33m"
		}

		// Format as xK/32K
		usedK := float64(d.CtxUsed) / 1024
		totalK := float64(d.CtxTotal) / 1024
		freeK := totalK - usedK
		if freeK < 0 {
			freeK = 0
		}

		compCount := proxy.GlobalCompressStats.Count.Load()
		compSaved := proxy.GlobalCompressStats.TokensSaved.Load()

		fmt.Printf("  \033[36m上下文\033[0m  %s[%s%s]\033[0m  %.1fK / %.0fK  余 %.1fK",
			ctxColor,
			strings.Repeat("=", ctxBar),
			strings.Repeat(".", 20-ctxBar),
			usedK, totalK, freeK)

		if compCount > 0 {
			savedK := float64(compSaved) / 1024
			fmt.Printf("  \033[35m压缩 %d 次 · 省 %.1fK\033[0m", compCount, savedK)
		}
		fmt.Println()
	}
}

// RenderPanel renders the full monitor panel (for `ashforge status --live`)
func RenderPanel(d MonitorData, modelName string) string {
	var b strings.Builder
	width := 53

	b.WriteString(fmt.Sprintf("┌%s┐\n", strings.Repeat("─", width)))

	title := fmt.Sprintf("  Ashforge — %s", modelName)
	if d.TokPerSec > 0 {
		title += fmt.Sprintf(" @ %.1f tok/s", d.TokPerSec)
	}
	padding := width - len([]rune(title))
	if padding < 1 {
		padding = 1
	}
	b.WriteString(fmt.Sprintf("│%s%s│\n", title, strings.Repeat(" ", padding)))
	b.WriteString(fmt.Sprintf("├%s┤\n", strings.Repeat("─", width)))

	// VRAM bar
	if d.VRAM_Total_MB > 0 {
		b.WriteString(renderBar("显存", d.VRAM_Used_MB, d.VRAM_Total_MB, "MB", width))
	}

	// RAM bar
	if d.RAM_Total_MB > 0 {
		b.WriteString(renderBar("内存", int(d.RAM_Used_MB), int(d.RAM_Total_MB), "MB", width))
	}

	// Context bar
	if d.CtxTotal > 0 {
		b.WriteString(renderBar("上下文", d.CtxUsed, d.CtxTotal, "tok", width))
	}

	// Speed
	if d.TokPerSec > 0 {
		speedLine := fmt.Sprintf("  速度  %.1f tok/s", d.TokPerSec)
		pad := width - len([]rune(speedLine))
		if pad < 1 {
			pad = 1
		}
		b.WriteString(fmt.Sprintf("│%s%s│\n", speedLine, strings.Repeat(" ", pad)))
	}

	// Temperature
	if d.GPU_Temp_C > 0 {
		status := "✓ 正常"
		if d.GPU_Temp_C > 85 {
			status = "⚠ 过高"
		} else if d.GPU_Temp_C > 75 {
			status = "~ 偏高"
		}
		tempLine := fmt.Sprintf("  温度  %d℃  %s", d.GPU_Temp_C, status)
		pad := width - len([]rune(tempLine))
		if pad < 1 {
			pad = 1
		}
		b.WriteString(fmt.Sprintf("│%s%s│\n", tempLine, strings.Repeat(" ", pad)))
	}

	b.WriteString(fmt.Sprintf("└%s┘\n", strings.Repeat("─", width)))

	// Alerts
	for _, alert := range d.Alerts {
		b.WriteString(alert + "\n")
	}

	return b.String()
}

func renderBar(label string, used, total int, unit string, width int) string {
	pct := float64(used) / float64(total)
	barLen := 20
	filled := int(pct * float64(barLen))
	if filled > barLen {
		filled = barLen
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barLen-filled)

	var usedStr, totalStr string
	if unit == "MB" && total > 1024 {
		usedStr = fmt.Sprintf("%.1f GB", float64(used)/1024)
		totalStr = fmt.Sprintf("%.0f GB", float64(total)/1024)
	} else {
		usedStr = fmt.Sprintf("%d", used)
		totalStr = fmt.Sprintf("%d", total)
	}

	line := fmt.Sprintf("  %-4s  %s  %s / %s  %.0f%%",
		label, bar, usedStr, totalStr, pct*100)

	pad := width - len([]rune(line))
	if pad < 1 {
		pad = 1
	}
	return fmt.Sprintf("│%s%s│\n", line, strings.Repeat(" ", pad))
}
