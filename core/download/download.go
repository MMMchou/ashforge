package download

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/schollz/progressbar/v3"
)

// DownloadFile downloads a file with progress bar and resume support
func DownloadFile(downloadURL, dest string, resume bool) error {
	// Check if file exists and get size for resume
	var existingSize int64
	if resume {
		if info, err := os.Stat(dest); err == nil {
			existingSize = info.Size()
		}
	}

	// Create HTTP request with range header for resume
	req, err := http.NewRequest("GET", downloadURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	// Create HTTP client with proxy support
	client := &http.Client{
		Timeout: 0, // No timeout for large downloads
		Transport: &http.Transport{
			Proxy: func(r *http.Request) (*url.URL, error) {
				// Check environment variables for proxy
				if proxy := os.Getenv("HTTP_PROXY"); proxy != "" {
					return url.Parse(proxy)
				}
				if proxy := os.Getenv("http_proxy"); proxy != "" {
					return url.Parse(proxy)
				}
				return nil, nil
			},
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	// Check status
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	// Get total size
	totalSize := resp.ContentLength
	if resp.StatusCode == http.StatusPartialContent {
		totalSize += existingSize
	}

	// Open file for writing
	flag := os.O_CREATE | os.O_WRONLY
	if existingSize > 0 && resp.StatusCode == http.StatusPartialContent {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
		existingSize = 0
	}

	out, err := os.OpenFile(dest, flag, 0644)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	// Create progress bar
	bar := progressbar.NewOptions64(
		totalSize,
		progressbar.OptionSetDescription("Downloading"),
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionShowBytes(true),
		progressbar.OptionSetWidth(40),
		progressbar.OptionThrottle(100*time.Millisecond),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() {
			fmt.Fprint(os.Stderr, "\n")
		}),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionFullWidth(),
		progressbar.OptionSetRenderBlankState(true),
	)

	if existingSize > 0 {
		bar.Set64(existingSize)
	}

	// Copy with progress
	_, err = io.Copy(io.MultiWriter(out, bar), resp.Body)
	if err != nil {
		return fmt.Errorf("download interrupted: %w", err)
	}

	return nil
}
