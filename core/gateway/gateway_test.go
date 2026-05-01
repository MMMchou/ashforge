package gateway

import (
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Gateway.Stop — nil-safety
// ---------------------------------------------------------------------------

func TestGatewayStop_NilServerAndTracker(t *testing.T) {
	// A zero-value Gateway (server and ctxTracker both nil) must not panic.
	g := &Gateway{running: true}
	if err := g.Stop(); err != nil {
		t.Fatalf("Stop() returned unexpected error: %v", err)
	}
}

func TestGatewayStop_NotRunning(t *testing.T) {
	// Stop on a gateway that was never started should be a no-op.
	g := &Gateway{}
	if err := g.Stop(); err != nil {
		t.Fatalf("Stop() on non-running gateway returned error: %v", err)
	}
}

func TestGatewayStop_DoubleStop(t *testing.T) {
	g := &Gateway{running: true}
	if err := g.Stop(); err != nil {
		t.Fatalf("first Stop() returned error: %v", err)
	}
	if err := g.Stop(); err != nil {
		t.Fatalf("second Stop() returned error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// ContextTracker.Stop — double-close safety & goroutine leak
// ---------------------------------------------------------------------------

func TestContextTrackerStop_DoubleSafe(t *testing.T) {
	ct := &ContextTracker{
		stopCh: make(chan struct{}),
	}
	// First stop should close the channel.
	ct.Stop()
	// Second stop must not panic (select/default guard).
	ct.Stop()
}

func TestContextTrackerStop_GoroutineExits(t *testing.T) {
	// Use NewContextTracker which spawns pollMetrics goroutine.
	// We pass port 0 so the HTTP call fails harmlessly.
	ct := NewContextTracker(0)
	ct.Stop()

	// Give the goroutine a moment to return; if it doesn't, the channel
	// read below will confirm it stopped.
	select {
	case <-ct.stopCh:
		// channel is closed — good
	case <-time.After(2 * time.Second):
		t.Fatal("stopCh was not closed after Stop()")
	}
}

// ---------------------------------------------------------------------------
// ContextTracker.GetUsage — default zero value
// ---------------------------------------------------------------------------

func TestContextTrackerGetUsage_Default(t *testing.T) {
	ct := &ContextTracker{
		stopCh: make(chan struct{}),
	}
	defer ct.Stop()

	u := ct.GetUsage()
	if u.Used != 0 || u.Total != 0 || u.Pct != 0 {
		t.Fatalf("expected zero-value ContextUsage, got %+v", u)
	}
}

// ---------------------------------------------------------------------------
// RepetitionDetector — n-gram threshold
// ---------------------------------------------------------------------------

func TestRepetitionDetector_TriggersOnRepeatedNgram(t *testing.T) {
	// ngramSize=2, threshold=3, windowSize=100
	d := NewRepetitionDetector(2, 3, 100)

	// Feed the 2-gram ("hello","world") three times → triggers on 3rd occurrence.
	tokens := []string{"hello", "world", "hello", "world", "hello", "world"}
	var triggeredAt int
	for i, tok := range tokens {
		if d.Feed(tok) {
			triggeredAt = i + 1 // 1-based token count
			break
		}
	}
	if triggeredAt == 0 {
		t.Fatal("expected repetition detection, but it never triggered")
	}
	if !d.IsTriggered() {
		t.Fatal("IsTriggered() should be true after detection")
	}
}

func TestRepetitionDetector_NoFalsePositive(t *testing.T) {
	d := NewRepetitionDetector(3, 5, 200)

	// Feed unique tokens — should never trigger.
	for i := 0; i < 100; i++ {
		tok := string(rune('A' + (i % 26)))
		if i > 25 {
			tok += string(rune('a' + (i % 26)))
		}
		if d.Feed(tok) {
			t.Fatalf("false positive at token %d", i)
		}
	}
}

func TestRepetitionDetector_SpecialTokenRepeat(t *testing.T) {
	d := NewRepetitionDetector(3, 100, 200) // high ngram threshold so only special-token path fires

	// 3 consecutive identical special tokens should trigger.
	for i := 0; i < 2; i++ {
		if d.Feed("<|end|>") {
			t.Fatalf("triggered too early at token %d", i)
		}
	}
	if !d.Feed("<|end|>") {
		t.Fatal("expected trigger on 3rd consecutive special token")
	}
}

func TestRepetitionDetector_StaysTriggeredAfterDetection(t *testing.T) {
	d := NewRepetitionDetector(2, 2, 50)

	// Trigger it
	d.Feed("a")
	d.Feed("b")
	d.Feed("a")
	d.Feed("b") // 2nd occurrence of "a|b"

	if !d.IsTriggered() {
		t.Fatal("should be triggered")
	}

	// Subsequent feeds should still return true.
	if !d.Feed("anything") {
		t.Fatal("Feed should return true once triggered")
	}
}

// ---------------------------------------------------------------------------
// RepetitionDetector — sliding window eviction
// ---------------------------------------------------------------------------

func TestRepetitionDetector_WindowEviction(t *testing.T) {
	// Small window: only 6 tokens kept. Ngram size=2, threshold=3.
	d := NewRepetitionDetector(2, 3, 6)

	// Feed "a","b" twice then push them out of window with filler.
	d.Feed("a")
	d.Feed("b") // ngram "a|b" count=1
	d.Feed("a")
	d.Feed("b") // ngram "a|b" count=2
	// Now fill with unique tokens to evict the early ngrams.
	d.Feed("x")
	d.Feed("y")
	d.Feed("z")
	d.Feed("w")

	// The old "a|b" ngrams should have been evicted, so feeding "a","b"
	// again should NOT immediately trigger (count should be < threshold).
	if d.Feed("a") {
		t.Fatal("should not trigger after window eviction")
	}
	if d.Feed("b") {
		t.Fatal("should not trigger — evicted ngrams should not count")
	}
}

// ---------------------------------------------------------------------------
// LoopDetector
// ---------------------------------------------------------------------------

func TestLoopDetector_ConsecutiveSame(t *testing.T) {
	d := NewLoopDetector()

	// Fill 50 tokens of the same value — should trigger consecutive-same.
	for i := 0; i < 49; i++ {
		if d.Feed("x") {
			return // triggered before window full — acceptable
		}
	}
	if !d.Feed("x") {
		t.Fatal("expected loop detection on 50 identical tokens")
	}
}

func TestLoopDetector_PatternRepeat(t *testing.T) {
	d := NewLoopDetector()

	// Repeat a short pattern many times to fill the window.
	pattern := []string{"A", "B", "C"}
	for i := 0; i < 50; i++ {
		tok := pattern[i%len(pattern)]
		if d.Feed(tok) {
			return // detected — pass
		}
	}
	t.Fatal("expected pattern-repeat detection for A-B-C cycle")
}

func TestLoopDetector_NormalTextNoTrigger(t *testing.T) {
	d := NewLoopDetector()

	// Feed 50 unique tokens — should not trigger.
	for i := 0; i < 50; i++ {
		tok := string(rune('A' + i))
		if d.Feed(tok) {
			t.Fatalf("false positive at token %d", i)
		}
	}
}

// ---------------------------------------------------------------------------
// containsStream helper
// ---------------------------------------------------------------------------

func TestContainsStream(t *testing.T) {
	tests := []struct {
		name string
		body string
		want bool
	}{
		{"stream true", `{"model":"m","stream":true}`, true},
		{"stream false", `{"model":"m","stream":false}`, false},
		{"stream true spaced", `{"stream": true}`, true},
		{"no stream field", `{"model":"m"}`, false},
		{"empty body", `{}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := containsStream([]byte(tt.body))
			if got != tt.want {
				t.Errorf("containsStream(%q) = %v, want %v", tt.body, got, tt.want)
			}
		})
	}
}
