package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
)

// Gateway is the Ashforge gateway server
type Gateway struct {
	listenPort  int
	listenHost  string // 监听地址，默认 127.0.0.1，用 0.0.0.0 开放局域网
	backendPort int
	modelAlias  string // llama-server 的 model alias
	proxy       *httputil.ReverseProxy
	server      *http.Server
	mu          sync.Mutex
	running     bool
	ctxTracker  *ContextTracker
	compressCfg CompressConfig
}

// New creates a new gateway server
func New(listenPort, backendPort int, modelAlias string, host string) *Gateway {
	if host == "" {
		host = "127.0.0.1"
	}
	target, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", backendPort))

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = -1 // Flush immediately for streaming

	g := &Gateway{
		listenPort:  listenPort,
		listenHost:  host,
		backendPort: backendPort,
		modelAlias:  modelAlias,
		proxy:       proxy,
		ctxTracker:  NewContextTracker(backendPort),
		compressCfg: DefaultCompressConfig(backendPort),
	}

	return g
}

// Start starts the gateway server (blocking)
func (g *Gateway) Start() error {
	mux := http.NewServeMux()

	// /v1/responses — format conversion for Codex CLI
	mux.HandleFunc("/v1/responses", g.handleResponses)
	// /responses — same, without /v1/ prefix (newer clients like Cursor, Claude Code)
	mux.HandleFunc("/responses", g.handleResponses)

	// /v1/chat/completions — streaming-aware proxy with repetition detection
	mux.HandleFunc("/v1/chat/completions", g.handleChatCompletions)

	// All other /v1/ endpoints — transparent reverse proxy
	mux.HandleFunc("/v1/", g.handleTransparent)
	mux.HandleFunc("/health", g.handleTransparent)

	g.server = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", g.listenHost, g.listenPort),
		Handler: mux,
	}

	g.mu.Lock()
	g.running = true
	g.mu.Unlock()

	log.Printf("Ashforge gateway listening on :%d → llama-server :%d", g.listenPort, g.backendPort)
	return g.server.ListenAndServe()
}

// StartAsync starts the gateway server in a goroutine
func (g *Gateway) StartAsync() {
	go func() {
		if err := g.Start(); err != nil && err != http.ErrServerClosed {
			log.Printf("Gateway server error: %v", err)
		}
	}()
}

// Stop stops the gateway server
func (g *Gateway) Stop() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.running {
		return nil
	}
	g.running = false
	if g.ctxTracker != nil {
		g.ctxTracker.Stop()
	}
	if g.server != nil {
		return g.server.Close()
	}
	return nil
}

// handleTransparent proxies requests directly to llama-server
func (g *Gateway) handleTransparent(w http.ResponseWriter, r *http.Request) {
	g.proxy.ServeHTTP(w, r)
}

// handleChatCompletions handles /v1/chat/completions with streaming support and repetition detection
func (g *Gateway) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Read body to check if streaming
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request", http.StatusBadRequest)
		return
	}
	r.Body.Close()

	// Rewrite model field to match llama-server's alias
	body = g.rewriteModelField(body)

	// Apply compression if needed (Module 5: context compression)
	body = g.maybeCompressMessages(body)

	// Check if streaming is requested
	isStream := containsStream(body)

	if !isStream {
		// Non-streaming: transparent proxy
		r.Body = io.NopCloser(io.Reader(newBytesReader(body)))
		r.ContentLength = int64(len(body))
		g.proxy.ServeHTTP(w, r)
		return
	}

	// Streaming: proxy with repetition detection
	g.streamWithDetection(w, r, body)
}

// maybeCompressMessages checks if the request messages exceed the context threshold
// and compresses them if needed. Returns the (possibly modified) body.
func (g *Gateway) maybeCompressMessages(body []byte) []byte {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body
	}

	rawMsgs, ok := req["messages"]
	if !ok {
		return body
	}

	// Convert []interface{} to []map[string]interface{}
	msgSlice, ok := rawMsgs.([]interface{})
	if !ok {
		return body
	}

	messages := make([]map[string]interface{}, 0, len(msgSlice))
	for _, m := range msgSlice {
		if mm, ok := m.(map[string]interface{}); ok {
			messages = append(messages, mm)
		}
	}

	// Get total context from tracker, fallback to 32768
	totalCtx := 32768
	if g.ctxTracker != nil {
		usage := g.ctxTracker.GetUsage()
		if usage.Total > 0 {
			totalCtx = usage.Total
		}
	}

	compressed, changed := CompressMessages(messages, totalCtx, g.compressCfg)
	if !changed {
		return body
	}

	// Replace messages in request
	req["messages"] = compressed
	newBody, err := json.Marshal(req)
	if err != nil {
		log.Printf("[compress] failed to marshal compressed request: %v", err)
		return body
	}

	return newBody
}

// rewriteModelField replaces the model field in the request body with llama-server's alias
func (g *Gateway) rewriteModelField(body []byte) []byte {
	if g.modelAlias == "" {
		return body
	}

	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body
	}

	// Replace model field
	req["model"] = g.modelAlias

	rewritten, err := json.Marshal(req)
	if err != nil {
		return body
	}

	return rewritten
}

func containsStream(body []byte) bool {
	// Simple check for "stream":true or "stream": true
	for i := 0; i < len(body)-10; i++ {
		if body[i] == '"' && i+8 < len(body) {
			if string(body[i:i+8]) == "\"stream\"" {
				// Look for true after colon
				for j := i + 8; j < len(body) && j < i+20; j++ {
					if body[j] == 't' {
						return true
					}
					if body[j] == 'f' {
						return false
					}
				}
			}
		}
	}
	return false
}
