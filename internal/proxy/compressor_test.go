package proxy

import (
	"testing"
)

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		name  string
		input string
		min   int
		max   int
	}{
		{"empty", "", 0, 0},
		{"english short", "hello world", 2, 5},
		{"english sentence", "The quick brown fox jumps over the lazy dog", 8, 15},
		{"chinese short", "你好世界", 2, 4},
		{"chinese sentence", "这是一个用于测试的中文句子，包含一些常见的汉字。", 15, 40},
		{"mixed", "Hello 你好 World 世界", 3, 8},
		{"code", "func main() { fmt.Println(\"hello\") }", 5, 15},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := estimateTokens(tt.input)
			if got < tt.min || got > tt.max {
				t.Errorf("estimateTokens(%q) = %d, want [%d, %d]", tt.input, got, tt.min, tt.max)
			}
		})
	}
}

func TestEstimateMessagesTokens(t *testing.T) {
	msgs := []map[string]interface{}{
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "Hello, how are you?"},
		{"role": "assistant", "content": "I'm doing well, thank you!"},
	}
	total := estimateMessagesTokens(msgs)
	if total < 15 || total > 40 {
		t.Errorf("estimateMessagesTokens = %d, want [15, 40]", total)
	}
}

func TestFindHeadEnd(t *testing.T) {
	msgs := []map[string]interface{}{
		{"role": "system", "content": "system prompt"},
		{"role": "user", "content": "first question"},
		{"role": "assistant", "content": "first answer"},
		{"role": "user", "content": "second question"},
		{"role": "assistant", "content": "second answer"},
	}
	got := findHeadEnd(msgs)
	if got != 3 {
		t.Errorf("findHeadEnd = %d, want 3 (system + first user + first assistant)", got)
	}
}

func TestFindHeadEnd_MultiSystem(t *testing.T) {
	msgs := []map[string]interface{}{
		{"role": "system", "content": "system prompt 1"},
		{"role": "system", "content": "system prompt 2"},
		{"role": "user", "content": "first question"},
		{"role": "assistant", "content": "first answer"},
		{"role": "user", "content": "second question"},
	}
	got := findHeadEnd(msgs)
	if got != 4 {
		t.Errorf("findHeadEnd = %d, want 4 (2 system + first user + first assistant)", got)
	}
}

func TestFindTailStart(t *testing.T) {
	msgs := []map[string]interface{}{
		{"role": "system", "content": "short"},
		{"role": "user", "content": "short"},
		{"role": "assistant", "content": "short"},
		{"role": "user", "content": "short"},
		{"role": "assistant", "content": "This is a longer message that should be in the tail section of the conversation."},
	}
	// With a small keepTokens, tail should start near the end
	got := findTailStart(msgs, 10)
	if got >= len(msgs) || got < 3 {
		t.Errorf("findTailStart(10) = %d, want 3-4", got)
	}
}

func TestCompressMessages_NoCompress(t *testing.T) {
	msgs := []map[string]interface{}{
		{"role": "system", "content": "You are helpful."},
		{"role": "user", "content": "Hi"},
		{"role": "assistant", "content": "Hello!"},
	}
	cfg := CompressConfig{TriggerPct: 0.75, TailKeepTokens: 8192, BackendPort: 0}
	result, changed := CompressMessages(msgs, 32768, cfg)
	if changed {
		t.Error("expected no compression for short conversation")
	}
	if len(result) != len(msgs) {
		t.Errorf("expected %d messages, got %d", len(msgs), len(result))
	}
}

func TestCompressMessages_TriggerButTooFewMiddle(t *testing.T) {
	// Create messages that exceed threshold but have no compressible middle
	longContent := make([]byte, 100000)
	for i := range longContent {
		longContent[i] = 'a'
	}
	msgs := []map[string]interface{}{
		{"role": "system", "content": "system"},
		{"role": "user", "content": string(longContent)},
		{"role": "assistant", "content": string(longContent)},
	}
	cfg := CompressConfig{TriggerPct: 0.75, TailKeepTokens: 8192, BackendPort: 0}
	// totalCtx=100 to force trigger, but head+tail covers everything
	result, changed := CompressMessages(msgs, 100, cfg)
	if changed {
		t.Error("expected no compression when head+tail covers all messages")
	}
	if len(result) != len(msgs) {
		t.Errorf("expected %d messages, got %d", len(msgs), len(result))
	}
}
