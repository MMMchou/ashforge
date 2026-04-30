package dashboard

import (
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"sync"
	"time"
)

// Status holds the current model serving status
type Status struct {
	ModelName    string  `json:"model_name"`
	StartedAt   string  `json:"started_at"`
	UptimeStr   string  `json:"uptime"`
	GPUName      string  `json:"gpu_name"`
	VRAMMB       int     `json:"vram_mb"`
	VRAMUsedMB   int     `json:"vram_used_mb"`
	RAMTotalMB   uint64  `json:"ram_total_mb"`
	RAMUsedMB    uint64  `json:"ram_used_mb"`
	ContextSize  int     `json:"context_size"`
	ContextUsed  int     `json:"context_used"`
	SpeedTPS     float64 `json:"speed_tps"`
	RequestCount int64   `json:"request_count"`
	ProxyPort    int     `json:"proxy_port"`
	BackendPort  int     `json:"backend_port"`
}

// Dashboard serves a web-based monitoring UI
type Dashboard struct {
	mu      sync.RWMutex
	status  Status
	started time.Time
}

// New creates a new Dashboard
func New() *Dashboard {
	return &Dashboard{
		started: time.Now(),
	}
}

// UpdateStatus updates the dashboard status
func (d *Dashboard) UpdateStatus(s Status) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.status = s
	d.status.UptimeStr = time.Since(d.started).Truncate(time.Second).String()
}

// Serve starts the dashboard HTTP server on the given port
func (d *Dashboard) Serve(port int) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/", d.handleIndex)
	mux.HandleFunc("/api/status", d.handleStatus)

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	fmt.Printf("      Dashboard: http://%s\n", addr)

	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}
	return server.ListenAndServe()
}

func (d *Dashboard) handleStatus(w http.ResponseWriter, r *http.Request) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	d.status.UptimeStr = time.Since(d.started).Truncate(time.Second).String()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(d.status)
}

func (d *Dashboard) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tmpl.Execute(w, nil)
}

var tmpl = template.Must(template.New("dashboard").Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ashforge Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { 
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: #0a0a0f; color: #c8c8d0;
    min-height: 100vh; padding: 2rem;
  }
  .header { 
    text-align: center; margin-bottom: 2rem;
    border-bottom: 1px solid #1a1a2e; padding-bottom: 1rem;
  }
  .header h1 { 
    font-size: 1.8rem; color: #ff6b35;
    text-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
  }
  .header .subtitle { color: #666; font-size: 0.85rem; margin-top: 0.3rem; }
  .grid { 
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem; max-width: 1200px; margin: 0 auto;
  }
  .card { 
    background: #12121a; border: 1px solid #1a1a2e;
    border-radius: 8px; padding: 1.2rem;
  }
  .card h3 { color: #ff6b35; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.8rem; }
  .metric { margin-bottom: 0.6rem; }
  .metric .label { color: #666; font-size: 0.8rem; }
  .metric .value { color: #e0e0e8; font-size: 1.4rem; font-weight: bold; }
  .metric .unit { color: #666; font-size: 0.8rem; }
  .bar-bg { background: #1a1a2e; border-radius: 4px; height: 6px; margin-top: 0.3rem; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
  .bar-fill.ok { background: linear-gradient(90deg, #22c55e, #4ade80); }
  .bar-fill.warn { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
  .bar-fill.danger { background: linear-gradient(90deg, #ef4444, #f87171); }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.5rem; }
  .status-dot.online { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
  .footer { text-align: center; margin-top: 2rem; color: #333; font-size: 0.75rem; }
</style>
</head>
<body>
<div class="header">
  <h1>ASHFORGE</h1>
  <div class="subtitle">Auto-tuned LLM Engine</div>
</div>
<div class="grid">
  <div class="card">
    <h3>Model</h3>
    <div class="metric">
      <div class="value"><span class="status-dot online"></span><span id="model">—</span></div>
    </div>
    <div class="metric">
      <div class="label">Uptime</div>
      <div class="value" id="uptime" style="font-size:1rem">—</div>
    </div>
    <div class="metric">
      <div class="label">Requests</div>
      <div class="value" id="requests" style="font-size:1rem">0</div>
    </div>
  </div>
  <div class="card">
    <h3>Performance</h3>
    <div class="metric">
      <div class="label">Speed</div>
      <div class="value"><span id="speed">—</span> <span class="unit">tok/s</span></div>
    </div>
    <div class="metric">
      <div class="label">API Endpoint</div>
      <div class="value" id="endpoint" style="font-size:0.85rem; color:#ff6b35">—</div>
    </div>
  </div>
  <div class="card">
    <h3>GPU / VRAM</h3>
    <div class="metric">
      <div class="label" id="gpu-name">—</div>
      <div class="value"><span id="vram-used">—</span> / <span id="vram-total">—</span> <span class="unit">MB</span></div>
      <div class="bar-bg"><div class="bar-fill ok" id="vram-bar" style="width:0%"></div></div>
    </div>
  </div>
  <div class="card">
    <h3>Context</h3>
    <div class="metric">
      <div class="value"><span id="ctx-used">—</span> / <span id="ctx-total">—</span> <span class="unit">tokens</span></div>
      <div class="bar-bg"><div class="bar-fill ok" id="ctx-bar" style="width:0%"></div></div>
    </div>
  </div>
  <div class="card">
    <h3>System RAM</h3>
    <div class="metric">
      <div class="value"><span id="ram-used">—</span> / <span id="ram-total">—</span> <span class="unit">MB</span></div>
      <div class="bar-bg"><div class="bar-fill ok" id="ram-bar" style="width:0%"></div></div>
    </div>
  </div>
</div>
<div class="footer">Ashforge Dashboard · by Ashan</div>
<script>
function barClass(pct) { return pct > 90 ? 'danger' : pct > 70 ? 'warn' : 'ok'; }
function update() {
  fetch('/api/status').then(r => r.json()).then(d => {
    document.getElementById('model').textContent = d.model_name || '—';
    document.getElementById('uptime').textContent = d.uptime || '—';
    document.getElementById('speed').textContent = d.speed_tps ? d.speed_tps.toFixed(1) : '—';
    document.getElementById('requests').textContent = d.request_count || 0;
    document.getElementById('endpoint').textContent = 'http://localhost:' + (d.proxy_port || '—') + '/v1';
    document.getElementById('gpu-name').textContent = d.gpu_name || 'No GPU';
    document.getElementById('vram-used').textContent = d.vram_used_mb || 0;
    document.getElementById('vram-total').textContent = d.vram_mb || 0;
    var vp = d.vram_mb ? (d.vram_used_mb / d.vram_mb * 100) : 0;
    var vb = document.getElementById('vram-bar');
    vb.style.width = vp + '%';
    vb.className = 'bar-fill ' + barClass(vp);
    document.getElementById('ctx-used').textContent = d.context_used || 0;
    document.getElementById('ctx-total').textContent = d.context_size || 0;
    var cp = d.context_size ? (d.context_used / d.context_size * 100) : 0;
    var cb = document.getElementById('ctx-bar');
    cb.style.width = cp + '%';
    cb.className = 'bar-fill ' + barClass(cp);
    document.getElementById('ram-used').textContent = d.ram_used_mb || 0;
    document.getElementById('ram-total').textContent = d.ram_total_mb || 0;
    var rp = d.ram_total_mb ? (d.ram_used_mb / d.ram_total_mb * 100) : 0;
    var rb = document.getElementById('ram-bar');
    rb.style.width = rp + '%';
    rb.className = 'bar-fill ' + barClass(rp);
  }).catch(() => {});
}
update();
setInterval(update, 2000);
</script>
</body>
</html>`))
