//go:build linux

package engine

import (
	"os"
	"os/exec"
	"syscall"
)

// StartProcess starts a process with Linux-specific flags
func StartProcess(binaryPath string, args []string) *os.Process {
	cmd := exec.Command(binaryPath, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
	if err := cmd.Start(); err != nil {
		return nil
	}
	return cmd.Process
}
