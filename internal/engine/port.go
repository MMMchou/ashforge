package engine

import (
	"fmt"
	"net"
	"time"
)

// waitForPort polls a TCP port until it's ready or timeout
func waitForPort(host string, port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	addr := fmt.Sprintf("%s:%d", host, port)

	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			conn.Close()
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timeout waiting for port %d", port)
}

// findFreePort finds an available port starting from the given port
func findFreePort(start int) int {
	for port := start; port < start+10; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err == nil {
			ln.Close()
			return port
		}
	}
	return start
}

// FindAvailablePort finds an available port, starting from the preferred port.
// If the preferred port is taken, it tries up to 100 ports ahead.
// Returns the available port and whether it differs from the preferred one.
func FindAvailablePort(preferred int) (int, bool) {
	// First try the preferred port
	if isPortAvailable(preferred) {
		return preferred, false
	}

	// Scan for next available port
	for offset := 1; offset <= 100; offset++ {
		port := preferred + offset
		if port > 65535 {
			break
		}
		if isPortAvailable(port) {
			return port, true
		}
	}

	// Last resort: let OS assign a port
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return preferred, false // give up, return preferred
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	return port, true
}

// isPortAvailable checks if a TCP port is available for binding
func isPortAvailable(port int) bool {
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	ln.Close()
	return true
}
