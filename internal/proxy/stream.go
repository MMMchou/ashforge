package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// streamWithDetection proxies a streaming chat completion with repetition detection
// and context usage warnings
func (s *Server) streamWithDetection(w http.ResponseWriter, r *http.Request, body []byte) {
	targetURL := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.backendPort)
	proxyReq, err := http.NewRequest("POST", targetURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "proxy error", http.StatusInternalServerError)
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(proxyReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("backend error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		w.WriteHeader(resp.StatusCode)
		w.Write(respBody)
		return
	}

	// Inject context warning header (Module 4)
	if s.ctxTracker != nil {
		s.ctxTracker.InjectContextWarning(w)
	}

	// Set streaming headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	detector := NewRepetitionDetector(3, 5, 200)
	loopDet := NewLoopDetector()

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			fmt.Fprintf(w, "%s\n", line)
			flusher.Flush()
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			// Append context full hint if > 90% (Module 4)
			if s.ctxTracker != nil {
				if hint := s.ctxTracker.GetContextFullHint(); hint != "" {
					hintChunk := fmt.Sprintf(`{"choices":[{"delta":{"content":%q},"index":0}]}`, hint)
					fmt.Fprintf(w, "data: %s\n\n", hintChunk)
					flusher.Flush()
				}
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			break
		}

		// Parse chunk to extract content for detection
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err == nil {
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				content := chunk.Choices[0].Delta.Content

				// Check both detectors (Module 3: n-gram + pattern loop)
				ngramTriggered := detector.Feed(content)
				loopTriggered := loopDet.Feed(content)

				if ngramTriggered || loopTriggered {
					reason := "repetition"
					if loopTriggered {
						reason = "loop_detected"
					}
					log.Printf("%s detected in stream, injecting stop", reason)
					// Inject warning + stop
					warnChunk := `{"choices":[{"delta":{"content":"\n\n⚠️  Repetition detected, auto-stopped. Try rephrasing or refreshing the conversation.\n检测到重复输出，已自动停止。建议重新提问或刷新对话。"},"index":0}]}`
					fmt.Fprintf(w, "data: %s\n\n", warnChunk)
					stopChunk := fmt.Sprintf(`{"choices":[{"delta":{},"finish_reason":"%s","index":0}]}`, reason)
					fmt.Fprintf(w, "data: %s\n\n", stopChunk)
					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
					return
				}
			}
		}

		// Forward the line as-is
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

// newBytesReader creates a bytes.Reader (helper to avoid import in handler.go)
func newBytesReader(b []byte) *bytes.Reader {
	return bytes.NewReader(b)
}
