package tui

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

	"github.com/MMMchou/ashforge/core/gateway"
)

// ── ANSI helpers ────────────────────────────────────────────────────

// colorForPct returns an ANSI color code based on percentage thresholds.
//
//	< 70%  → green (\033[32m)
//	70-90% → yellow (\033[33m)
//	> 90%  → red (\033[31m)
func colorForPct(pct float64) string {
	switch {
	case pct > 0.9:
		return "\033[31m"
	case pct > 0.7:
		return "\033[33m"
	default:
		return "\033[32m"
	}
}

// renderBar renders a progress bar like [====......] of the given width
// (number of character slots inside the brackets). The fill amount is
// determined by pct (0.0 – 1.0).
func renderBar(pct float64, width int) string {
	filled := int(pct * float64(width))
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}
	return "[" + strings.Repeat("=", filled) + strings.Repeat(".", width-filled) + "]"
}

// ── Metric data ─────────────────────────────────────────────────────

// DisplayData holds all collected metrics.
type DisplayData struct {
	VRAM_Used_MB  int
	VRAM_Total_MB int
	RAM_Used_MB   uint64
	RAM_Total_MB  uint64
	CtxUsed       int
	CtxTotal      int
	TokPerSec     float64
	GPU_Temp_C    int
	GPU_Util_Pct  int     // GPU utilisation percentage
	CPU_Pct       float64
	Alerts        []string
}

// Collect gathers all metrics from GPU, system, and llama-server.
func Collect(backendPort int) DisplayData {
	var d DisplayData
	collectGPU(&d)
	collectSystem(&d)
	collectFromMetrics(&d, backendPort)
	d.Alerts = checkAlerts(d)
	return d
}

// ── GPU collection ──────────────────────────────────────────────────

func collectGPU(d *DisplayData) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return
	}
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

// ── Metrics from llama-server ───────────────────────────────────────

func collectFromMetrics(d *DisplayData, port int) {
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
		case strings.HasPrefix(line, "llamacpp:tokens_predicted_seconds "):
			d.TokPerSec = parseMetricFloat(line)
		}
	}

	if d.CtxTotal == 0 {
		d.CtxTotal = fetchCtxFromSlots(port)
	}
}

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

// parseMetricInt extracts an integer value from a Prometheus metric line.
// Kept as a local copy to avoid circular imports with gateway.
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

// ── System (RAM / CPU) collection ───────────────────────────────────

func collectSystem(d *DisplayData) {
	if runtime.GOOS == "windows" {
		collectSystemWindows(d)
	} else {
		collectSystemLinux(d)
	}
}

func collectSystemWindows(d *DisplayData) {
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

func collectSystemLinux(d *DisplayData) {
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

// ── Alerts ──────────────────────────────────────────────────────────

func checkAlerts(d DisplayData) []string {
	var alerts []string

	if d.VRAM_Total_MB > 0 {
		pct := float64(d.VRAM_Used_MB) / float64(d.VRAM_Total_MB)
		if pct > 0.9 {
			alerts = append(alerts, "⚠️  显存即将用满，推理可能中断")
		}
	}

	if d.CtxTotal > 0 {
		pct := float64(d.CtxUsed) / float64(d.CtxTotal)
		if pct > 0.8 {
			alerts = append(alerts, fmt.Sprintf("⚠️  上下文已用 %.0f%%，建议新开对话", pct*100))
		}
	}

	if d.GPU_Temp_C > 85 {
		alerts = append(alerts, fmt.Sprintf("⚠️  GPU 温度 %d℃，可能触发降频", d.GPU_Temp_C))
	}

	return alerts
}

// ── Display (live TUI) ─────────────────────────────────────────────

// Display runs periodic collection and rendering (renamed from Monitor).
type Display struct {
	backendPort int
	modelName   string
	stopCh      chan struct{}
	running     bool
	ctxTotal    int
	lastAlerts  []string
	ParamInfo   string // runtime params summary (e.g., "64K ctx · f16 KV · ub512 · mlock")
}

// NewDisplay creates a new Display (renamed from NewMonitor).
func NewDisplay(backendPort int, modelName string) *Display {
	return &Display{
		backendPort: backendPort,
		modelName:   modelName,
		stopCh:      make(chan struct{}),
	}
}

// StartAsync starts the display in a background goroutine.
func (d *Display) StartAsync() {
	d.running = true
	go d.run()
}

// Stop stops the display loop.
func (d *Display) Stop() {
	if d.running {
		d.running = false
		close(d.stopCh)
	}
}

func (d *Display) run() {
	// Initial delay to let server stabilise.
	select {
	case <-time.After(3 * time.Second):
	case <-d.stopCh:
		return
	}

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	// Clear screen and hide cursor.
	fmt.Print("\033[2J\033[H\033[?25l")

	for {
		select {
		case <-d.stopCh:
			// Restore cursor.
			fmt.Print("\033[?25h")
			return
		case <-ticker.C:
			data := Collect(d.backendPort)
			d.renderPanel(data)

			// Only print new alerts.
			for _, alert := range data.Alerts {
				isNew := true
				for _, prev := range d.lastAlerts {
					if prev == alert {
						isNew = false
						break
					}
				}
				if isNew {
					fmt.Printf("\n%s", alert)
				}
			}
			d.lastAlerts = data.Alerts
		}
	}
}

// ── Live panel (ANSI TUI) ──────────────────────────────────────────

func (d *Display) renderPanel(data DisplayData) {
	// Move cursor to top-left.
	fmt.Print("\033[H")

	// Title row.
	fmt.Print("\033[90m─ 实时监控 · ")
	if data.TokPerSec > 0 {
		fmt.Print("推理中")
	} else {
		fmt.Print("空载")
	}
	fmt.Print(" ─────────────────── 每 2s 刷新 ─\033[0m\n")

	// Param info row.
	if d.ParamInfo != "" {
		fmt.Printf("  \033[36m%s\033[0m\n", d.ParamInfo)
	}

	// Column headers.
	fmt.Print("  ")
	fmt.Print("\033[34m速度\033[0m          ")
	fmt.Print("\033[32m显存\033[0m           ")
	fmt.Print("\033[32m内存\033[0m        ")
	fmt.Print("\033[35mGPU\033[0m      ")
	fmt.Print("\033[32m温度\033[0m\n")

	// Values row.
	fmt.Print("  ")

	// Speed value.
	if data.TokPerSec > 0 {
		fmt.Printf("\033[34m%.1f tok/s\033[0m   ", data.TokPerSec)
	} else {
		fmt.Print("\033[90m— tok/s\033[0m     ")
	}

	// VRAM value.
	if data.VRAM_Total_MB > 0 {
		pct := float64(data.VRAM_Used_MB) / float64(data.VRAM_Total_MB)
		fmt.Printf("%s%.1f/%.0f GB\033[0m    ", colorForPct(pct),
			float64(data.VRAM_Used_MB)/1024, float64(data.VRAM_Total_MB)/1024)
	} else {
		fmt.Print("\033[90m—\033[0m           ")
	}

	// RAM value.
	if data.RAM_Total_MB > 0 {
		pct := float64(data.RAM_Used_MB) / float64(data.RAM_Total_MB)
		fmt.Printf("%s%.1f/%.0f GB\033[0m   ", colorForPct(pct),
			float64(data.RAM_Used_MB)/1024, float64(data.RAM_Total_MB)/1024)
	} else {
		fmt.Print("\033[90m—\033[0m        ")
	}

	// GPU utilisation value.
	if data.GPU_Util_Pct >= 0 {
		fmt.Printf("\033[35m%d%%\033[0m      ", data.GPU_Util_Pct)
	} else {
		fmt.Print("\033[90m—\033[0m      ")
	}

	// Temperature value.
	if data.GPU_Temp_C > 0 {
		// Temperature uses slightly different thresholds (75/85 instead of 70/90).
		tempColor := "\033[32m"
		if data.GPU_Temp_C > 85 {
			tempColor = "\033[31m"
		} else if data.GPU_Temp_C > 75 {
			tempColor = "\033[33m"
		}
		fmt.Printf("%s%d°C\033[0m\n", tempColor, data.GPU_Temp_C)
	} else {
		fmt.Print("\033[90m—\033[0m\n")
	}

	// ── Progress bar row ────────────────────────────────────────────
	const barWidth = 10
	fmt.Print("  ")

	// Speed bar.
	if data.TokPerSec > 0 {
		speedPct := data.TokPerSec / 100
		if speedPct > 1 {
			speedPct = 1
		}
		fmt.Printf("\033[34m%s\033[0m ", renderBar(speedPct, barWidth))
	} else {
		fmt.Printf("\033[90m%s\033[0m ", renderBar(0, barWidth))
	}

	// VRAM bar.
	if data.VRAM_Total_MB > 0 {
		pct := float64(data.VRAM_Used_MB) / float64(data.VRAM_Total_MB)
		fmt.Printf("%s%s\033[0m ", colorForPct(pct), renderBar(pct, barWidth))
	} else {
		fmt.Printf("\033[90m%s\033[0m ", renderBar(0, barWidth))
	}

	// RAM bar.
	if data.RAM_Total_MB > 0 {
		pct := float64(data.RAM_Used_MB) / float64(data.RAM_Total_MB)
		fmt.Printf("%s%s\033[0m ", colorForPct(pct), renderBar(pct, barWidth))
	} else {
		fmt.Printf("\033[90m%s\033[0m ", renderBar(0, barWidth))
	}

	// GPU utilisation bar.
	if data.GPU_Util_Pct >= 0 {
		pct := float64(data.GPU_Util_Pct) / 100
		fmt.Printf("\033[35m%s\033[0m ", renderBar(pct, barWidth))
	} else {
		fmt.Printf("\033[90m%s\033[0m ", renderBar(0, barWidth))
	}

	// Temperature bar.
	if data.GPU_Temp_C > 0 {
		tempPct := float64(data.GPU_Temp_C) / 100
		if tempPct > 1 {
			tempPct = 1
		}
		tempColor := "\033[32m"
		if data.GPU_Temp_C > 85 {
			tempColor = "\033[31m"
		} else if data.GPU_Temp_C > 75 {
			tempColor = "\033[33m"
		}
		fmt.Printf("%s%s\033[0m", tempColor, renderBar(tempPct, barWidth))
	} else {
		fmt.Printf("\033[90m%s\033[0m", renderBar(0, barWidth))
	}
	fmt.Println()

	fmt.Print("\033[90m─────────────────────────────────────────────────────────\033[0m\n")

	// Context progress bar (wider, 20 chars).
	if data.CtxTotal > 0 {
		const ctxBarWidth = 20
		ctxPct := 0.0
		if data.CtxUsed > 0 {
			ctxPct = float64(data.CtxUsed) / float64(data.CtxTotal)
		}

		ctxColor := "\033[32m"
		if ctxPct > 0.8 {
			ctxColor = "\033[31m"
		} else if ctxPct > 0.6 {
			ctxColor = "\033[33m"
		}

		usedK := float64(data.CtxUsed) / 1024
		totalK := float64(data.CtxTotal) / 1024
		freeK := totalK - usedK
		if freeK < 0 {
			freeK = 0
		}

		compCount := gateway.GlobalCompressStats.Count.Load()
		compSaved := gateway.GlobalCompressStats.TokensSaved.Load()

		fmt.Printf("  \033[36m上下文\033[0m  %s%s\033[0m  %.1fK / %.0fK  余 %.1fK",
			ctxColor,
			renderBar(ctxPct, ctxBarWidth),
			usedK, totalK, freeK)

		if compCount > 0 {
			savedK := float64(compSaved) / 1024
			fmt.Printf("  \033[35m压缩 %d 次 · 省 %.1fK\033[0m", compCount, savedK)
		}
		fmt.Println()
	}
}

// ── Static panel (for `ashforge status`) ────────────────────────────

// RenderPanel renders the full status panel as a string (box-drawing).
func RenderPanel(data DisplayData, modelName string) string {
	var b strings.Builder
	width := 53

	b.WriteString(fmt.Sprintf("┌%s┐\n", strings.Repeat("─", width)))

	title := fmt.Sprintf("  Ashforge — %s", modelName)
	if data.TokPerSec > 0 {
		title += fmt.Sprintf(" @ %.1f tok/s", data.TokPerSec)
	}
	padding := width - len([]rune(title))
	if padding < 1 {
		padding = 1
	}
	b.WriteString(fmt.Sprintf("│%s%s│\n", title, strings.Repeat(" ", padding)))
	b.WriteString(fmt.Sprintf("├%s┤\n", strings.Repeat("─", width)))

	// VRAM bar.
	if data.VRAM_Total_MB > 0 {
		b.WriteString(renderStaticBar("显存", data.VRAM_Used_MB, data.VRAM_Total_MB, "MB", width))
	}

	// RAM bar.
	if data.RAM_Total_MB > 0 {
		b.WriteString(renderStaticBar("内存", int(data.RAM_Used_MB), int(data.RAM_Total_MB), "MB", width))
	}

	// Context bar.
	if data.CtxTotal > 0 {
		b.WriteString(renderStaticBar("上下文", data.CtxUsed, data.CtxTotal, "tok", width))
	}

	// Speed.
	if data.TokPerSec > 0 {
		speedLine := fmt.Sprintf("  速度  %.1f tok/s", data.TokPerSec)
		pad := width - len([]rune(speedLine))
		if pad < 1 {
			pad = 1
		}
		b.WriteString(fmt.Sprintf("│%s%s│\n", speedLine, strings.Repeat(" ", pad)))
	}

	// Temperature.
	if data.GPU_Temp_C > 0 {
		status := "✓ 正常"
		if data.GPU_Temp_C > 85 {
			status = "⚠ 过高"
		} else if data.GPU_Temp_C > 75 {
			status = "~ 偏高"
		}
		tempLine := fmt.Sprintf("  温度  %d℃  %s", data.GPU_Temp_C, status)
		pad := width - len([]rune(tempLine))
		if pad < 1 {
			pad = 1
		}
		b.WriteString(fmt.Sprintf("│%s%s│\n", tempLine, strings.Repeat(" ", pad)))
	}

	b.WriteString(fmt.Sprintf("└%s┘\n", strings.Repeat("─", width)))

	// Alerts.
	for _, alert := range data.Alerts {
		b.WriteString(alert + "\n")
	}

	return b.String()
}

// renderStaticBar renders a box-drawing bar row for the static RenderPanel.
func renderStaticBar(label string, used, total int, unit string, width int) string {
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
