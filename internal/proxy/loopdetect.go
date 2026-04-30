package proxy

// LoopDetector detects repeating patterns in streaming output.
// Complements RepetitionDetector by catching "ABCABC" style loops
// and consecutive identical token runs (spec Module 3).
type LoopDetector struct {
	window     []string
	windowSize int
	threshold  float64 // consecutive-same ratio to trigger
}

// NewLoopDetector creates a detector with spec defaults:
// 50-token window, 60% consecutive-same threshold.
func NewLoopDetector() *LoopDetector {
	return &LoopDetector{
		windowSize: 50,
		threshold:  0.6,
	}
}

// Feed adds a token and returns true if a loop is detected.
func (d *LoopDetector) Feed(token string) bool {
	d.window = append(d.window, token)
	if len(d.window) > d.windowSize {
		d.window = d.window[1:]
	}

	if len(d.window) < d.windowSize {
		return false
	}

	return d.detectConsecutiveSame() || d.detectPatternRepeat()
}

// detectConsecutiveSame checks if > threshold of adjacent tokens are identical.
func (d *LoopDetector) detectConsecutiveSame() bool {
	same := 0
	for i := 1; i < len(d.window); i++ {
		if d.window[i] == d.window[i-1] {
			same++
		}
	}
	return float64(same)/float64(len(d.window)) > d.threshold
}

// detectPatternRepeat checks for short repeating patterns (2-10 tokens).
// e.g. "A B C A B C A B C" with patternLen=3.
func (d *LoopDetector) detectPatternRepeat() bool {
	for pLen := 2; pLen <= 10; pLen++ {
		if d.hasPattern(pLen) {
			return true
		}
	}
	return false
}

func (d *LoopDetector) hasPattern(pLen int) bool {
	n := len(d.window)
	if n < pLen*3 {
		return false // need at least 3 repetitions
	}

	// Take the last pLen tokens as the candidate pattern
	pattern := d.window[n-pLen:]
	matches := 0

	// Check up to 5 preceding windows of the same length
	limit := n - pLen*6
	if limit < 0 {
		limit = 0
	}
	for i := n - pLen*2; i >= limit; i -= pLen {
		if i+pLen > n {
			continue
		}
		if slicesEqual(d.window[i:i+pLen], pattern) {
			matches++
		} else {
			break // pattern broken, stop checking
		}
	}

	return matches >= 2 // pattern repeated 3+ times total
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
