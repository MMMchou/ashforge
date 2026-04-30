package proxy

import (
	"strings"
	"sync"
)

// RepetitionDetector detects repeated token patterns in streaming output.
type RepetitionDetector struct {
	mu           sync.Mutex
	window       []string
	windowSize   int
	ngramCounts  map[string]int
	ngramSize    int
	threshold    int
	totalTokens  int
	triggered    bool
	triggerToken string
}

// NewRepetitionDetector creates a detector.
// ngramSize=3 means it tracks 3-token sequences.
// threshold=5 means 5 repeats of the same 3-gram triggers detection.
func NewRepetitionDetector(ngramSize, threshold, windowSize int) *RepetitionDetector {
	return &RepetitionDetector{
		window:      make([]string, 0, windowSize),
		windowSize:  windowSize,
		ngramCounts: make(map[string]int),
		ngramSize:   ngramSize,
		threshold:   threshold,
	}
}

// Feed adds a token and returns true if repetition is detected.
func (d *RepetitionDetector) Feed(token string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.triggered {
		return true
	}

	d.totalTokens++

	// Special token detection: consecutive identical special tokens
	trimmed := strings.TrimSpace(token)
	if isSpecialToken(trimmed) {
		if len(d.window) > 0 && strings.TrimSpace(d.window[len(d.window)-1]) == trimmed {
			count := 1
			for i := len(d.window) - 1; i >= 0; i-- {
				if strings.TrimSpace(d.window[i]) == trimmed {
					count++
				} else {
					break
				}
			}
			if count >= 3 {
				d.triggered = true
				d.triggerToken = trimmed
				return true
			}
		}
	}

	// Add to sliding window
	d.window = append(d.window, token)
	if len(d.window) > d.windowSize {
		d.removeOldestNgram()
		d.window = d.window[1:]
	}

	// Check ngram if we have enough tokens
	if len(d.window) >= d.ngramSize {
		ngram := d.buildNgram(d.window[len(d.window)-d.ngramSize:])
		d.ngramCounts[ngram]++
		if d.ngramCounts[ngram] >= d.threshold {
			d.triggered = true
			d.triggerToken = ngram
			return true
		}
	}

	return false
}

// IsTriggered returns whether repetition was detected.
func (d *RepetitionDetector) IsTriggered() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.triggered
}

func (d *RepetitionDetector) removeOldestNgram() {
	if len(d.window) >= d.ngramSize {
		ngram := d.buildNgram(d.window[:d.ngramSize])
		if d.ngramCounts[ngram] > 0 {
			d.ngramCounts[ngram]--
			if d.ngramCounts[ngram] == 0 {
				delete(d.ngramCounts, ngram)
			}
		}
	}
}

func (d *RepetitionDetector) buildNgram(tokens []string) string {
	return strings.Join(tokens, "|")
}

func isSpecialToken(s string) bool {
	if s == "" {
		return false
	}
	if strings.HasPrefix(s, "<|") && strings.HasSuffix(s, "|>") {
		return true
	}
	if strings.HasPrefix(s, "<") && strings.HasSuffix(s, ">") && len(s) <= 20 {
		return true
	}
	return false
}
