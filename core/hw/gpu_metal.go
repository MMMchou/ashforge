//go:build darwin

package hw

import (
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

func init() {
	detectMetalGPU = func() *GPUInfo {
		gpu, err := detectMetal()
		if err != nil {
			return nil
		}
		return gpu
	}
}

// detectAMD is not applicable on macOS
func detectAMD() ([]GPUInfo, error) {
	return nil, fmt.Errorf("AMD GPU detection not supported on macOS")
}

// detectMetal detects Apple Silicon GPU via system_profiler
func detectMetal() (*GPUInfo, error) {
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return nil, fmt.Errorf("failed to detect Apple Silicon: %w", err)
	}

	chipName := strings.TrimSpace(string(out))
	if !strings.Contains(chipName, "Apple") {
		return nil, fmt.Errorf("not Apple Silicon: %s", chipName)
	}

	// Get unified memory size (Apple Silicon shares memory between CPU and GPU)
	memOut, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return nil, fmt.Errorf("failed to detect memory: %w", err)
	}

	memBytes, _ := strconv.ParseUint(strings.TrimSpace(string(memOut)), 10, 64)
	totalMB := int(memBytes / (1024 * 1024))

	// Apple Silicon uses unified memory — GPU can use ~75% of total RAM
	gpuVRAM := totalMB * 3 / 4

	// Detect chip variant for bandwidth estimation
	bandwidth := estimateAppleBandwidth(chipName)

	return &GPUInfo{
		Index:            0,
		Name:             chipName + " GPU (Metal)",
		VRAM_MB:          gpuVRAM,
		VRAMFree_MB:      gpuVRAM / 2, // conservative estimate
		MemBandwidth_GBs: bandwidth,
	}, nil
}

// estimateAppleBandwidth returns estimated memory bandwidth in GB/s for Apple chips
func estimateAppleBandwidth(chipName string) float64 {
	switch {
	case strings.Contains(chipName, "M4 Ultra"):
		return 819.2
	case strings.Contains(chipName, "M4 Max"):
		return 546.0
	case strings.Contains(chipName, "M4 Pro"):
		return 273.0
	case strings.Contains(chipName, "M4"):
		return 120.0
	case strings.Contains(chipName, "M3 Ultra"):
		return 819.2
	case strings.Contains(chipName, "M3 Max"):
		return 408.0
	case strings.Contains(chipName, "M3 Pro"):
		return 150.0
	case strings.Contains(chipName, "M3"):
		return 100.0
	case strings.Contains(chipName, "M2 Ultra"):
		return 819.2
	case strings.Contains(chipName, "M2 Max"):
		return 408.0
	case strings.Contains(chipName, "M2 Pro"):
		return 200.0
	case strings.Contains(chipName, "M2"):
		return 100.0
	case strings.Contains(chipName, "M1 Ultra"):
		return 819.2
	case strings.Contains(chipName, "M1 Max"):
		return 408.0
	case strings.Contains(chipName, "M1 Pro"):
		return 200.0
	case strings.Contains(chipName, "M1"):
		return 68.25
	default:
		return 100.0 // conservative fallback
	}
}

// detectCPU detects CPU information on macOS
func detectCPU() (CPUInfo, error) {
	cpu := CPUInfo{
		Cores:   runtime.NumCPU(),
		Threads: runtime.NumCPU(),
	}

	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err == nil {
		cpu.Model = strings.TrimSpace(string(out))
	}

	// Apple Silicon doesn't have AVX, but has NEON
	if runtime.GOARCH == "arm64" {
		cpu.HasAVX2 = false
		cpu.HasAVX512 = false
	} else {
		// Intel Mac
		featOut, err := exec.Command("sysctl", "-n", "machdep.cpu.leaf7_features").Output()
		if err == nil {
			features := string(featOut)
			cpu.HasAVX2 = strings.Contains(features, "AVX2")
			cpu.HasAVX512 = strings.Contains(features, "AVX512")
		}
	}

	return cpu, nil
}

// detectRAM detects RAM information on macOS
func detectRAM() (RAMInfo, error) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return RAMInfo{}, fmt.Errorf("failed to detect RAM: %w", err)
	}

	memBytes, _ := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	totalMB := memBytes / (1024 * 1024)

	// Get used memory via vm_stat
	usedMB := getUsedMemoryDarwin()

	ramType := "lpddr" // Apple Silicon always uses LPDDR
	if runtime.GOARCH != "arm64" {
		ramType = "unknown" // Intel Mac
	}

	return RAMInfo{
		Total_MB: totalMB,
		Used_MB:  usedMB,
		Free_MB:  totalMB - usedMB,
		Type:     ramType,
	}, nil
}

// getUsedMemoryDarwin parses vm_stat output to estimate used memory
func getUsedMemoryDarwin() uint64 {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0
	}

	var pageSize uint64 = 16384 // default for Apple Silicon
	var activePages, wiredPages uint64

	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "page size of") {
			fmt.Sscanf(line, "Mach Virtual Memory Statistics: (page size of %d bytes)", &pageSize)
		}
		if strings.HasPrefix(line, "Pages active:") {
			val := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "Pages active:"), "."))
			activePages, _ = strconv.ParseUint(val, 10, 64)
		}
		if strings.HasPrefix(line, "Pages wired down:") {
			val := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "Pages wired down:"), "."))
			wiredPages, _ = strconv.ParseUint(val, 10, 64)
		}
	}

	return (activePages + wiredPages) * pageSize / (1024 * 1024)
}

// detectOS detects OS information on macOS
func detectOS() OSInfo {
	version := "macOS"
	out, err := exec.Command("sw_vers", "-productVersion").Output()
	if err == nil {
		version = "macOS " + strings.TrimSpace(string(out))
	}

	return OSInfo{
		Platform: "darwin",
		Arch:     runtime.GOARCH,
		Version:  version,
	}
}
