package proxy

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ContextTracker monitors KV cache usage from llama-server
type ContextTracker struct {
	backendPort int
	mu          sync.RWMutex
	lastUsage   ContextUsage
	lastUpdate  time.Time
}

// ContextUsage holds context usage metrics
type ContextUsage struct {
	Used  int     // tokens used
	Total int     // total context size
	Pct   float64 // usage percentage
}

// NewContextTracker creates a context tracker
func NewContextTracker(backendPort int) *ContextTracker {
	ct := &ContextTracker{
		backendPort: backendPort,
	}
	go ct.pollMetrics()
	return ct
}

// GetUsage returns the latest context usage
func (ct *ContextTracker) GetUsage() ContextUsage {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	return ct.lastUsage
}

// pollMetrics periodically fetches context usage from /metrics
func (ct *ContextTracker) pollMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		usage := ct.fetchUsage()
		ct.mu.Lock()
		ct.lastUsage = usage
		ct.lastUpdate = time.Now()
		ct.mu.Unlock()
	}
}

func (ct *ContextTracker) fetchUsage() ContextUsage {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", ct.backendPort))
	if err != nil {
		return ContextUsage{}
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var used, total int

	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "llamacpp:kv_cache_used_cells ") {
			used = parseMetricInt(line)
		} else if strings.HasPrefix(line, "llamacpp:kv_cache_tokens ") {
			if used == 0 {
				used = parseMetricInt(line)
			}
		} else if strings.HasPrefix(line, "llamacpp:n_ctx ") {
			total = parseMetricInt(line)
		}
	}

	var pct float64
	if total > 0 {
		pct = float64(used) / float64(total)
	}

	return ContextUsage{
		Used:  used,
		Total: total,
		Pct:   pct,
	}
}

func parseMetricInt(line string) int {
	parts := strings.Fields(line)
	if len(parts) >= 2 {
		v, _ := strconv.ParseFloat(parts[len(parts)-1], 64)
		return int(v)
	}
	return 0
}

// InjectContextWarning adds context warning headers to response
func (ct *ContextTracker) InjectContextWarning(w http.ResponseWriter) {
	usage := ct.GetUsage()
	if usage.Pct > 0.8 {
		w.Header().Set("X-Ashforge-Context-Warning",
			fmt.Sprintf("context_%.0f_pct", usage.Pct*100))
	}
}

// GetContextFullHint returns a warning message if context is > 90%
func (ct *ContextTracker) GetContextFullHint() string {
	usage := ct.GetUsage()
	if usage.Pct > 0.9 {
		return fmt.Sprintf("\n\n---\n⚠️  上下文已用 %.0f%%（%d/%d tokens）。\n早期对话内容可能被截断，重要信息请置于最近的消息中。",
			usage.Pct*100, usage.Used, usage.Total)
	}
	return ""
}
