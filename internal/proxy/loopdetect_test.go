package proxy

import (
	"testing"
)

func TestLoopDetector_ConsecutiveSame(t *testing.T) {
	d := NewLoopDetector()
	// Feed 50 identical tokens — should trigger at 60% threshold
	for i := 0; i < 50; i++ {
		if d.Feed("hello") {
			t.Logf("Triggered at token %d", i+1)
			return
		}
	}
	t.Error("Expected loop detection to trigger on 50 identical tokens")
}

func TestLoopDetector_PatternRepeat(t *testing.T) {
	d := NewLoopDetector()
	// Feed "A B C" pattern repeated many times
	pattern := []string{"A", "B", "C"}
	for i := 0; i < 60; i++ {
		tok := pattern[i%3]
		if d.Feed(tok) {
			t.Logf("Pattern loop triggered at token %d", i+1)
			return
		}
	}
	t.Error("Expected pattern loop detection to trigger on ABCABC repeat")
}

func TestLoopDetector_NormalOutput(t *testing.T) {
	d := NewLoopDetector()
	// Feed varied tokens — should NOT trigger
	tokens := []string{
		"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
		".", "This", "is", "a", "test", "of", "normal", "output", "that",
		"should", "not", "trigger", "the", "loop", "detector", "because",
		"it", "has", "sufficient", "variety", "in", "its", "token", "stream",
		"and", "does", "not", "repeat", "any", "particular", "pattern",
		"more", "than", "twice", "which", "means", "everything", "works",
		"as", "expected", "here",
	}
	for i, tok := range tokens {
		if d.Feed(tok) {
			t.Errorf("False positive at token %d (%s)", i+1, tok)
			return
		}
	}
}

func TestRepetitionDetector_NGram(t *testing.T) {
	d := NewRepetitionDetector(3, 5, 200)
	// Feed a 3-gram repeated 5+ times
	pattern := []string{"hello", "world", "!"}
	for round := 0; round < 10; round++ {
		for _, tok := range pattern {
			if d.Feed(tok) {
				t.Logf("N-gram repetition triggered at round %d", round+1)
				return
			}
		}
	}
	t.Error("Expected n-gram repetition to trigger")
}

func TestRepetitionDetector_NormalOutput(t *testing.T) {
	d := NewRepetitionDetector(3, 5, 200)
	tokens := []string{
		"def", "quicksort", "(", "arr", ")", ":", "\n",
		"if", "len", "(", "arr", ")", "<=", "1", ":", "\n",
		"return", "arr", "\n",
		"pivot", "=", "arr", "[", "0", "]", "\n",
		"left", "=", "[", "x", "for", "x", "in", "arr", "if", "x", "<", "pivot", "]", "\n",
		"right", "=", "[", "x", "for", "x", "in", "arr", "if", "x", ">", "pivot", "]", "\n",
	}
	for i, tok := range tokens {
		if d.Feed(tok) {
			t.Errorf("False positive at token %d (%s)", i+1, tok)
			return
		}
	}
}
