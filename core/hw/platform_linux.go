//go:build linux

package hw

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// detectAMD detects AMD GPUs via sysfs on Linux
func detectAMD() ([]GPUInfo, error) {
	matches, err := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	if err != nil || len(matches) == 0 {
		return nil, fmt.Errorf("no AMD GPU found")
	}

	var gpus []GPUInfo
	for i, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		vramBytes, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
		vramMB := int(vramBytes / (1024 * 1024))

		// Try to get GPU name
		nameFile := filepath.Join(filepath.Dir(path), "product_name")
		name := "AMD GPU"
		if nameData, err := os.ReadFile(nameFile); err == nil {
			name = strings.TrimSpace(string(nameData))
		}

		gpus = append(gpus, GPUInfo{
			Index:   i,
			Name:    name,
			VRAM_MB: vramMB,
		})
	}

	return gpus, nil
}

// detectCPU detects CPU information on Linux
func detectCPU() (CPUInfo, error) {
	cpu := CPUInfo{
		Cores:   runtime.NumCPU(),
		Threads: runtime.NumCPU(),
	}

	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return cpu, nil
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "model name") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				cpu.Model = strings.TrimSpace(parts[1])
			}
		}
		if strings.Contains(line, "avx2") {
			cpu.HasAVX2 = true
		}
		if strings.Contains(line, "avx512") {
			cpu.HasAVX512 = true
		}
	}

	return cpu, nil
}

// detectRAM detects RAM information on Linux
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

// getSystemRAM reads /proc/meminfo
func getSystemRAM() (totalMB uint64, usedMB uint64) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	var total, available uint64
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		val, _ := strconv.ParseUint(fields[1], 10, 64)
		switch fields[0] {
		case "MemTotal:":
			total = val // kB
		case "MemAvailable:":
			available = val // kB
		}
	}
	totalMB = total / 1024
	usedMB = (total - available) / 1024
	return
}

// detectRAMType tries to detect DDR type via dmidecode
func detectRAMType() string {
	cmd := exec.Command("dmidecode", "-t", "memory")
	output, err := cmd.Output()
	if err != nil {
		return "unknown"
	}

	if strings.Contains(string(output), "DDR5") {
		return "ddr5"
	} else if strings.Contains(string(output), "DDR4") {
		return "ddr4"
	}
	return "unknown"
}

// detectOS detects OS information
func detectOS() OSInfo {
	version := "Linux"
	if data, err := os.ReadFile("/etc/os-release"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "PRETTY_NAME=") {
				version = strings.Trim(strings.TrimPrefix(line, "PRETTY_NAME="), "\"")
				break
			}
		}
	}

	return OSInfo{
		Platform: "linux",
		Arch:     runtime.GOARCH,
		Version:  version,
	}
}
