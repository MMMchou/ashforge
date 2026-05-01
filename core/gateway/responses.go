package gateway

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

// responsesRequest matches the OpenAI Responses API request format
type responsesRequest struct {
	Model           string      `json:"model"`
	Instructions    string      `json:"instructions"`
	Input           interface{} `json:"input"`
	Stream          bool        `json:"stream"`
	MaxOutputTokens int         `json:"max_output_tokens"`
	Temperature     *float64    `json:"temperature"`
}

// handleResponses converts Responses API format to Chat Completions and back
func (g *Gateway) handleResponses(w http.ResponseWriter, r *http.Request) {
	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		writeResponsesError(w, "read body failed", http.StatusBadRequest)
		return
	}

	var req responsesRequest
	if err := json.Unmarshal(rawBody, &req); err != nil {
		writeResponsesError(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	messages := responsesInputToMessages(req.Instructions, req.Input)
	if len(messages) == 0 {
		writeResponsesError(w, "no valid messages in input", http.StatusBadRequest)
		return
	}

	chatReq := map[string]interface{}{
		"model":    req.Model,
		"messages": messages,
		"stream":   req.Stream,
	}
	if req.MaxOutputTokens > 0 {
		chatReq["max_tokens"] = req.MaxOutputTokens
	}
	if req.Temperature != nil {
		chatReq["temperature"] = *req.Temperature
	}

	chatBody, _ := json.Marshal(chatReq)

	targetURL := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", g.backendPort)
	proxyReq, err := http.NewRequest("POST", targetURL, bytes.NewReader(chatBody))
	if err != nil {
		writeResponsesError(w, "proxy error", http.StatusInternalServerError)
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: backendTimeout}
	resp, err := client.Do(proxyReq)
	if err != nil {
		writeResponsesError(w, fmt.Sprintf("backend error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("Responses: backend returned %d: %s", resp.StatusCode, string(body))
		writeResponsesError(w, fmt.Sprintf("upstream error %d", resp.StatusCode), resp.StatusCode)
		return
	}

	responseID := fmt.Sprintf("resp_%d", time.Now().UnixNano())

	if req.Stream {
		g.handleResponsesStream(w, resp.Body, responseID, req.Model)
	} else {
		g.handleResponsesNonStream(w, resp.Body, responseID, req.Model)
	}
}

// handleResponsesStream converts Chat Completions SSE to Responses API SSE
func (g *Gateway) handleResponsesStream(w http.ResponseWriter, body io.Reader, responseID, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeResponsesError(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	itemID := "item_0"

	// response.created
	emitSSE(w, flusher, "response.created", map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":     responseID,
			"object": "response",
			"status": "in_progress",
			"model":  model,
			"output": []interface{}{},
		},
	})

	// response.output_item.added
	outputItemInProgress := map[string]interface{}{
		"id":      itemID,
		"type":    "message",
		"role":    "assistant",
		"status":  "in_progress",
		"content": []interface{}{},
	}
	emitSSE(w, flusher, "response.output_item.added", map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         outputItemInProgress,
	})

	// response.content_part.added
	emitSSE(w, flusher, "response.content_part.added", map[string]interface{}{
		"type":          "response.content_part.added",
		"item_id":       itemID,
		"output_index":  0,
		"content_index": 0,
		"part": map[string]interface{}{
			"type": "output_text",
			"text": "",
		},
	})

	// Stream text deltas
	var fullText strings.Builder
	var promptTokens, completionTokens int

	detector := NewRepetitionDetector(3, 5, 200)

	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			promptTokens = chunk.Usage.PromptTokens
			completionTokens = chunk.Usage.CompletionTokens
		}

		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			text := chunk.Choices[0].Delta.Content
			fullText.WriteString(text)

			// Check for repetition
			if detector.Feed(text) {
				log.Printf("Repetition detected, stopping stream")
				break
			}

			// response.output_text.delta
			emitSSE(w, flusher, "response.output_text.delta", map[string]interface{}{
				"type":          "response.output_text.delta",
				"item_id":       itemID,
				"output_index":  0,
				"content_index": 0,
				"delta":         text,
			})
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("[responses] stream scan error: %v", err)
	}

	finalText := fullText.String()

	// response.output_text.done
	emitSSE(w, flusher, "response.output_text.done", map[string]interface{}{
		"type":          "response.output_text.done",
		"item_id":       itemID,
		"output_index":  0,
		"content_index": 0,
		"text":          finalText,
	})

	// response.output_item.done
	outputItemDone := map[string]interface{}{
		"id":     itemID,
		"type":   "message",
		"role":   "assistant",
		"status": "completed",
		"content": []interface{}{
			map[string]interface{}{"type": "output_text", "text": finalText},
		},
	}
	emitSSE(w, flusher, "response.output_item.done", map[string]interface{}{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item":         outputItemDone,
	})

	// response.completed
	emitSSE(w, flusher, "response.completed", map[string]interface{}{
		"type": "response.completed",
		"response": map[string]interface{}{
			"id":     responseID,
			"object": "response",
			"status": "completed",
			"model":  model,
			"output": []interface{}{outputItemDone},
			"usage": map[string]interface{}{
				"input_tokens":  promptTokens,
				"output_tokens": completionTokens,
				"total_tokens":  promptTokens + completionTokens,
			},
		},
	})
}

// handleResponsesNonStream converts Chat Completions JSON to Responses API JSON
func (g *Gateway) handleResponsesNonStream(w http.ResponseWriter, body io.Reader, responseID, model string) {
	bodyBytes, err := io.ReadAll(body)
	if err != nil {
		writeResponsesError(w, "read response failed", http.StatusInternalServerError)
		return
	}

	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(bodyBytes, &chatResp); err != nil {
		writeResponsesError(w, "parse response failed", http.StatusInternalServerError)
		return
	}

	content := ""
	if len(chatResp.Choices) > 0 {
		content = chatResp.Choices[0].Message.Content
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":     responseID,
		"object": "response",
		"status": "completed",
		"model":  model,
		"output": []interface{}{
			map[string]interface{}{
				"id":     "item_0",
				"type":   "message",
				"role":   "assistant",
				"status": "completed",
				"content": []interface{}{
					map[string]interface{}{"type": "output_text", "text": content},
				},
			},
		},
		"usage": map[string]interface{}{
			"input_tokens":  chatResp.Usage.PromptTokens,
			"output_tokens": chatResp.Usage.CompletionTokens,
			"total_tokens":  chatResp.Usage.PromptTokens + chatResp.Usage.CompletionTokens,
		},
	})
}

// responsesInputToMessages converts Responses API input to Chat Completions messages
func responsesInputToMessages(instructions string, input interface{}) []map[string]interface{} {
	var messages []map[string]interface{}

	if instructions != "" {
		messages = append(messages, map[string]interface{}{
			"role": "system", "content": instructions,
		})
	}

	items, ok := input.([]interface{})
	if !ok {
		if s, ok := input.(string); ok {
			messages = append(messages, map[string]interface{}{
				"role": "user", "content": s,
			})
		}
		return messages
	}

	for _, item := range items {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		itemType, _ := m["type"].(string)
		role, _ := m["role"].(string)
		if itemType != "message" {
			continue
		}
		chatRole := role
		if role == "developer" {
			chatRole = "system"
		}
		content := extractTextContent(m["content"])
		if content != "" {
			messages = append(messages, map[string]interface{}{
				"role": chatRole, "content": content,
			})
		}
	}

	return messages
}

func extractTextContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				if m["type"] == "input_text" || m["type"] == "output_text" {
					if text, ok := m["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func emitSSE(w http.ResponseWriter, flusher http.Flusher, eventType string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, string(jsonData))
	flusher.Flush()
}

func writeResponsesError(w http.ResponseWriter, message string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"param":   nil,
			"code":    nil,
		},
	})
}
