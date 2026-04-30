//go:build !windows && !linux

package llm

import (
	"os"
	"os/exec"
)

// StartProcess starts a process (fallback for unsupported platforms)
func StartProcess(binaryPath string, args []string) *os.Process {
	cmd := exec.Command(binaryPath, args...)
	if err := cmd.Start(); err != nil {
		return nil
	}
	return cmd.Process
}
