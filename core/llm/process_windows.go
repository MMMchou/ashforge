//go:build windows

package llm

import (
	"os"
	"os/exec"
	"syscall"
)

// StartProcess starts a process with Windows-specific flags
func StartProcess(binaryPath string, args []string) *os.Process {
	cmd := exec.Command(binaryPath, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
	if err := cmd.Start(); err != nil {
		return nil
	}
	return cmd.Process
}
