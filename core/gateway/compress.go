package gateway

import (
	"fmt"
	"log"
	"strings"
	"sync/atomic"
	"unicode/utf8"
)

// CompressStats tracks compression statistics (thread-safe)
type CompressStats struct {
	Count       atomic.Int64 // total compressions performed
	TokensSaved atomic.Int64 // total tokens saved
}

// global stats instance, readable from monitor
var GlobalCompressStats CompressStats

// CompressConfig holds compression parameters
type CompressConfig struct {
	// TriggerPct: compress when messages exceed this % of total context
	TriggerPct float64
	// TailKeepTokens: protect the most recent N tokens from compression
	TailKeepTokens int
	// BackendPort: llama-server port (kept for future use)
	BackendPort int
}

// DefaultCompressConfig returns sensible defaults for 32K context
func DefaultCompressConfig(backendPort int) CompressConfig {
	return CompressConfig{
		TriggerPct:     0.75, // trigger at 75% of ctx
		TailKeepTokens: 8192, // keep last 8K tokens
		BackendPort:    backendPort,
	}
}

// estimateTokens gives a rough token count for a string.
// Chinese: ~1.5 chars/token, English: ~4 chars/token.
func estimateTokens(s string) int {
	cjk := 0
	total := 0
	for _, r := range s {
		total++
		if r >= 0x4E00 && r <= 0x9FFF || // CJK Unified
			r >= 0x3400 && r <= 0x4DBF || // CJK Extension A
			r >= 0xF900 && r <= 0xFAFF { // CJK Compat
			cjk++
		}
	}
	ascii := total - cjk
	tokens := float64(cjk)*0.67 + float64(ascii)*0.25
	if tokens < 1 && total > 0 {
		return 1
	}
	return int(tokens)
}

// estimateMessagesTokens returns total estimated tokens for a message slice
func estimateMessagesTokens(messages []map[string]interface{}) int {
	total := 0
	for _, msg := range messages {
		content, _ := msg["content"].(string)
		total += estimateTokens(content)
		total += 4 // per-message overhead
	}
	return total
}

// CompressMessages checks if messages need compression and compresses if so.
// Uses pure algorithmic compression (no model call) for zero latency.
func CompressMessages(messages []map[string]interface{}, totalCtx int, cfg CompressConfig) ([]map[string]interface{}, bool) {
	totalTokens := estimateMessagesTokens(messages)
	threshold := int(float64(totalCtx) * cfg.TriggerPct)

	if totalTokens <= threshold {
		return messages, false
	}

	log.Printf("[compress] triggered: ~%d tokens > %d threshold (ctx=%d)", totalTokens, threshold, totalCtx)

	headEnd := findHeadEnd(messages)
	tailStart := findTailStart(messages, cfg.TailKeepTokens)

	if tailStart <= headEnd+1 {
		log.Printf("[compress] not enough middle to compress (head=%d, tail=%d, total=%d)", headEnd, tailStart, len(messages))
		return messages, false
	}

	middle := messages[headEnd:tailStart]
	log.Printf("[compress] compressing %d middle messages (%d head, %d tail kept)",
		len(middle), headEnd, len(messages)-tailStart)

	// Pure algorithmic summary: extract key info from each message
	summary := extractiveSummary(middle)

	// Reconstruct: head + summary message + tail
	compressed := make([]map[string]interface{}, 0, headEnd+1+len(messages)-tailStart)
	compressed = append(compressed, messages[:headEnd]...)
	compressed = append(compressed, map[string]interface{}{
		"role":    "system",
		"content": fmt.Sprintf("[Context Summary — %d messages compressed / 对话摘要 — %d 条消息已压缩]\n\n%s", len(middle), len(middle), summary),
	})
	compressed = append(compressed, messages[tailStart:]...)

	newTokens := estimateMessagesTokens(compressed)
	saved := totalTokens - newTokens
	log.Printf("[compress] done: %d → %d tokens (saved %d)", totalTokens, newTokens, saved)

	GlobalCompressStats.Count.Add(1)
	GlobalCompressStats.TokensSaved.Add(int64(saved))

	return compressed, true
}

const maxSummaryRunes = 3000 // ~2K tokens budget for summary

// extractiveSummary builds a summary by keeping the first line of each message
// plus any lines containing code fences, file paths, function names, or commands.
func extractiveSummary(messages []map[string]interface{}) string {
	var sb strings.Builder
	currentRunes := 0

	for _, msg := range messages {
		if currentRunes >= maxSummaryRunes {
			break
		}

		role, _ := msg["role"].(string)
		content, _ := msg["content"].(string)
		if content == "" {
			continue
		}

		lines := strings.Split(content, "\n")
		var kept []string

		for i, line := range lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}

			keep := false
			switch {
			case i == 0: // always keep first line
				keep = true
			case strings.HasPrefix(trimmed, "```"): // code fences
				keep = true
			case strings.Contains(trimmed, "/"): // file paths
				keep = true
			case strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* "): // bullet points
				keep = true
			case strings.HasPrefix(trimmed, "def ") || strings.HasPrefix(trimmed, "func ") ||
				strings.HasPrefix(trimmed, "class ") || strings.HasPrefix(trimmed, "function "): // function defs
				keep = true
			case strings.Contains(trimmed, "TODO") || strings.Contains(trimmed, "FIXME") ||
				strings.Contains(trimmed, "待办") || strings.Contains(trimmed, "问题"): // action items
				keep = true
			case strings.HasPrefix(trimmed, "$") || strings.HasPrefix(trimmed, ">"): // commands
				keep = true
			}

			if keep {
				// Truncate very long lines
				if utf8.RuneCountInString(trimmed) > 120 {
					runes := []rune(trimmed)
					trimmed = string(runes[:120]) + "…"
				}
				kept = append(kept, trimmed)
			}
		}

		if len(kept) == 0 {
			// At minimum keep first line truncated
			runes := []rune(lines[0])
			if len(runes) > 80 {
				runes = runes[:80]
			}
			kept = append(kept, string(runes)+"…")
		}

		entry := fmt.Sprintf("[%s] %s", role, strings.Join(kept, "\n  "))
		entryRunes := utf8.RuneCountInString(entry)
		if currentRunes+entryRunes > maxSummaryRunes {
			// Truncate this entry to fit
			remaining := maxSummaryRunes - currentRunes
			if remaining > 20 {
				runes := []rune(entry)
				if len(runes) > remaining {
					entry = string(runes[:remaining]) + "…"
				}
				sb.WriteString(entry)
				sb.WriteString("\n")
			}
			break
		}

		sb.WriteString(entry)
		sb.WriteString("\n")
		currentRunes += entryRunes + 1
	}

	return sb.String()
}

// findHeadEnd returns the index after the protected head messages.
func findHeadEnd(messages []map[string]interface{}) int {
	idx := 0
	for idx < len(messages) {
		role, _ := messages[idx]["role"].(string)
		if role != "system" {
			break
		}
		idx++
	}
	if idx < len(messages) {
		idx++
	}
	if idx < len(messages) {
		role, _ := messages[idx]["role"].(string)
		if role == "assistant" {
			idx++
		}
	}
	return idx
}

// findTailStart returns the index where the protected tail begins.
func findTailStart(messages []map[string]interface{}, keepTokens int) int {
	accum := 0
	for i := len(messages) - 1; i >= 0; i-- {
		content, _ := messages[i]["content"].(string)
		accum += estimateTokens(content) + 4
		if accum >= keepTokens {
			return i
		}
	}
	return 0
}
