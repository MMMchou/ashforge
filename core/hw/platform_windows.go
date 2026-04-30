//go:build windows

package hw

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

// detectAMD uses DXGI to detect AMD GPUs on Windows
func detectAMD() ([]GPUInfo, error) {
	// TODO: Implement DXGI enumeration for AMD GPUs
	// For now, return empty to indicate no AMD GPUs detected
	return nil, fmt.Errorf("AMD GPU detection not yet implemented on Windows")
}

// detectCPU detects CPU information on Windows
func detectCPU() (CPUInfo, error) {
	cpu := CPUInfo{
		Model:   getCPUModel(),
		Cores:   getCPUCores(),
		Threads: getCPUCores(),
		HasAVX2: true, // Assume modern CPUs have AVX2
	}

	return cpu, nil
}

// getCPUCores gets the number of logical processors
func getCPUCores() int {
	cmd := exec.Command("wmic", "cpu", "get", "NumberOfLogicalProcessors")
	output, err := cmd.Output()
	if err != nil {
		return 4 // fallback
	}
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if n, err := strconv.Atoi(line); err == nil && n > 0 {
			return n
		}
	}
	return 4
}

// getCPUModel reads CPU model from registry
func getCPUModel() string {
	// Try to read from registry
	cmd := exec.Command("wmic", "cpu", "get", "name")
	output, err := cmd.Output()
	if err != nil {
		return "Unknown CPU"
	}

	lines := strings.Split(string(output), "\n")
	if len(lines) > 1 {
		return strings.TrimSpace(lines[1])
	}
	return "Unknown CPU"
}

// detectRAM detects RAM information on Windows
func detectRAM() (RAMInfo, error) {
	totalMB, usedMB := getSystemRAM()
	if totalMB == 0 {
		return RAMInfo{}, fmt.Errorf("failed to detect RAM")
	}

	ramType := detectRAMType()

	return RAMInfo{
		Total_MB: totalMB,
		Used_MB:  usedMB,
		Free_MB:  totalMB - usedMB,
		Type:     ramType,
	}, nil
}

// getSystemRAM uses GlobalMemoryStatusEx to get RAM info
func getSystemRAM() (totalMB uint64, usedMB uint64) {
	kernel32 := windows.NewLazySystemDLL("kernel32.dll")
	globalMemoryStatusEx := kernel32.NewProc("GlobalMemoryStatusEx")

	type memoryStatusEx struct {
		Length               uint32
		MemoryLoad           uint32
		TotalPhys            uint64
		AvailPhys            uint64
		TotalPageFile        uint64
		AvailPageFile        uint64
		TotalVirtual         uint64
		AvailVirtual         uint64
		AvailExtendedVirtual uint64
	}

	var memStatus memoryStatusEx
	memStatus.Length = uint32(unsafe.Sizeof(memStatus))

	ret, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&memStatus)))
	if ret == 0 {
		return 0, 0
	}

	totalMB = memStatus.TotalPhys / (1024 * 1024)
	usedMB = (memStatus.TotalPhys - memStatus.AvailPhys) / (1024 * 1024)
	return
}

// detectRAMType tries to detect DDR type via WMI
func detectRAMType() string {
	cmd := exec.Command("wmic", "memorychip", "get", "SMBIOSMemoryType")
	output, err := cmd.Output()
	if err != nil {
		return "unknown"
	}

	// SMBIOSMemoryType values: 26=DDR4, 34=DDR5
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "34" {
			return "ddr5"
		} else if line == "26" {
			return "ddr4"
		}
	}

	return "unknown"
}

// detectOS detects OS information
func detectOS() OSInfo {
	return OSInfo{
		Platform: "windows",
		Arch:     "amd64",
		Version:  getWindowsVersion(),
	}
}

// getWindowsVersion gets Windows version string
func getWindowsVersion() string {
	cmd := exec.Command("cmd", "/c", "ver")
	output, err := cmd.Output()
	if err != nil {
		return "Windows (unknown version)"
	}
	return strings.TrimSpace(string(output))
}
