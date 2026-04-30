//go:build linux

package engine

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"
)

func setProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
}

// SetProcAttr is the exported version for use by other packages
func SetProcAttr(cmd *exec.Cmd) {
	setProcAttr(cmd)
}

func killProcess(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return fmt.Errorf("failed to find process %d: %w", pid, err)
	}
	if err := process.Signal(syscall.SIGTERM); err != nil {
		return fmt.Errorf("failed to send SIGTERM: %w", err)
	}
	time.Sleep(2 * time.Second)
	if isProcessAlive(pid) {
		process.Kill()
	}
	return nil
}

func isProcessAlive(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return process.Signal(syscall.Signal(0)) == nil
}
