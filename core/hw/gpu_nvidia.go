package hw

import (
	"encoding/xml"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// nvidiaSMILog is the top-level XML structure from nvidia-smi -q -x
type nvidiaSMILog struct {
	DriverVersion string       `xml:"driver_version"`
	CUDAVersion   string       `xml:"cuda_version"`
	GPUs          []nvidiaSMIG `xml:"gpu"`
}

type nvidiaSMIG struct {
	ProductName    string `xml:"product_name"`
	MemBusWidth    string `xml:"memory_bus_width"`
	FBMemory       struct {
		Total string `xml:"total"`
		Used  string `xml:"used"`
		Free  string `xml:"free"`
	} `xml:"fb_memory_usage"`
	MaxClocks struct {
		MemClock string `xml:"mem_clock"`
	} `xml:"max_clocks"`
}

// detectNVIDIA detects NVIDIA GPUs using nvidia-smi XML output + CSV fallback.
// XML provides name, memory, bandwidth; CSV provides compute_cap and VRAM fallback.
func detectNVIDIA() ([]GPUInfo, error) {
	// Step 1: XML for name, memory, driver, bandwidth
	xmlOut, err := exec.Command("nvidia-smi", "-q", "-x").Output()
	if err != nil {
		return nil, err
	}

	var smiLog nvidiaSMILog
	if err := xml.Unmarshal(xmlOut, &smiLog); err != nil {
		return nil, err
	}

	if len(smiLog.GPUs) == 0 {
		return nil, nil
	}

	// Step 2: CSV for compute_cap + VRAM fallback (stable across all driver versions)
	computeCaps := queryComputeCaps(len(smiLog.GPUs))
	csvVRAM := queryCSVMemory(len(smiLog.GPUs))

	gpus := make([]GPUInfo, 0, len(smiLog.GPUs))
	for i, g := range smiLog.GPUs {
		vramTotal := parseMemValue(g.FBMemory.Total)
		vramUsed := parseMemValue(g.FBMemory.Used)
		vramFree := parseMemValue(g.FBMemory.Free)

		// CSV fallback: if XML fb_memory_usage is empty/zero (schema change in newer drivers)
		if vramTotal == 0 && i < len(csvVRAM) && csvVRAM[i] > 0 {
			vramTotal = csvVRAM[i]
			vramFree = vramTotal - vramUsed
			if vramFree < 0 {
				vramFree = 0
			}
		}

		cc := ""
		if i < len(computeCaps) {
			cc = computeCaps[i]
		}

		isBlackwell := strings.HasPrefix(cc, "12")

		gpus = append(gpus, GPUInfo{
			Index:            i,
			Name:             strings.TrimSpace(g.ProductName),
			VRAM_MB:          vramTotal,
			VRAMUsed_MB:      vramUsed,
			VRAMFree_MB:      vramFree,
			ComputeCap:       cc,
			CUDADriver:       smiLog.CUDAVersion,
			MemBandwidth_GBs: calcBandwidth(g, strings.TrimSpace(g.ProductName)),
			IsBlackwell:      isBlackwell,
		})
	}

	// Debug: warn if any GPU has 0 VRAM (helps diagnose driver issues)
	for _, g := range gpus {
		if g.VRAM_MB == 0 {
			fmt.Printf("      ⚠  GPU %d (%s): VRAM=0, nvidia-smi XML may have changed. Report this to https://github.com/MMMchou/ashforge/issues\n", g.Index, g.Name)
		}
	}

	return gpus, nil
}

// calcBandwidth calculates memory bandwidth from XML fields (bus width + max clock).
// Falls back to estimateBandwidth() for virtualized environments or old drivers.
func calcBandwidth(g nvidiaSMIG, name string) float64 {
	// Parse bus width: "192 bit" → 192
	busWidth := 0
	bwStr := strings.TrimSpace(g.MemBusWidth)
	bwStr = strings.TrimSuffix(bwStr, " bit")
	bwStr = strings.TrimSpace(bwStr)
	if v, err := strconv.Atoi(bwStr); err == nil && v > 0 {
		busWidth = v
	}

	// Parse max mem clock: "7501 MHz" → 7501
	maxClockMHz := 0
	clkStr := strings.TrimSpace(g.MaxClocks.MemClock)
	clkStr = strings.TrimSuffix(clkStr, " MHz")
	clkStr = strings.TrimSpace(clkStr)
	if v, err := strconv.Atoi(clkStr); err == nil && v > 0 {
		maxClockMHz = v
	}

	if busWidth > 0 && maxClockMHz > 0 {
		// bandwidth (GB/s) = bus_width_bits/8 * freq_MHz * 2(DDR) / 1000
		return float64(busWidth) / 8 * float64(maxClockMHz) * 2 / 1000
	}

	// Fallback: name-based estimate (virtualized envs, old drivers)
	return estimateBandwidth(name)
}

// queryComputeCaps reads compute capability via CSV (simple, stable format for this one field)
func queryComputeCaps(gpuCount int) []string {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=compute_cap",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return make([]string, gpuCount)
	}

	caps := make([]string, 0, gpuCount)
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			caps = append(caps, line)
		}
	}
	return caps
}

// parseMemValue extracts integer MiB from strings like "24564 MiB", "24564 MB", "24,564 MiB", or "24564".
// Handles unit variations across nvidia-smi driver versions.
func parseMemValue(s string) int {
	s = strings.TrimSpace(s)
	// Remove known units
	for _, suffix := range []string{" MiB", " MB", " mib", " mb"} {
		s = strings.TrimSuffix(s, suffix)
	}
	s = strings.TrimSpace(s)
	// Remove commas (e.g. "16,384")
	s = strings.ReplaceAll(s, ",", "")
	v, _ := strconv.Atoi(s)
	return v
}

// queryCSVMemory reads total VRAM via CSV as fallback when XML fb_memory_usage is empty.
// This handles nvidia-smi XML schema changes across driver versions.
func queryCSVMemory(gpuCount int) []int {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.total",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return make([]int, gpuCount)
	}

	vrams := make([]int, 0, gpuCount)
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		line = strings.ReplaceAll(line, ",", "")
		if v, err := strconv.Atoi(line); err == nil {
			vrams = append(vrams, v)
		}
	}
	return vrams
}

// detectNVLink checks if NVLink is present between GPUs via nvidia-smi
func detectNVLink() bool {
	out, err := exec.Command("nvidia-smi", "nvlink", "--status").Output()
	if err != nil {
		return false
	}
	return strings.Contains(strings.ToLower(string(out)), "active")
}

// estimateBandwidth estimates memory bandwidth based on GPU name
func estimateBandwidth(name string) float64 {
	n := strings.ToLower(name)
	switch {
	// RTX 50 series (Blackwell)
	case strings.Contains(n, "5090"):
		return 1792.0
	case strings.Contains(n, "5080"):
		return 960.0
	case strings.Contains(n, "5070 ti"):
		return 896.0
	case strings.Contains(n, "5070"):
		return 672.0
	case strings.Contains(n, "5060 ti"):
		return 448.0
	case strings.Contains(n, "5060"):
		return 448.0
	// RTX 40 series (Ada)
	case strings.Contains(n, "4090"):
		return 1008.0
	case strings.Contains(n, "4080 super"):
		return 736.0
	case strings.Contains(n, "4080"):
		return 717.0
	case strings.Contains(n, "4070 ti super"):
		return 672.0
	case strings.Contains(n, "4070 ti"):
		return 504.0
	case strings.Contains(n, "4070 super"):
		return 504.0
	case strings.Contains(n, "4070"):
		return 504.0
	case strings.Contains(n, "4060 ti"):
		return 288.0
	case strings.Contains(n, "4060"):
		return 272.0
	// RTX 30 series (Ampere)
	case strings.Contains(n, "3090 ti"):
		return 1008.0
	case strings.Contains(n, "3090"):
		return 936.0
	case strings.Contains(n, "3080 ti"):
		return 912.0
	case strings.Contains(n, "3080"):
		return 760.0
	case strings.Contains(n, "3070 ti"):
		return 608.0
	case strings.Contains(n, "3070"):
		return 448.0
	case strings.Contains(n, "3060 ti"):
		return 448.0
	case strings.Contains(n, "3060"):
		return 360.0
	// RTX 20 series (Turing)
	case strings.Contains(n, "2080 ti"):
		return 616.0
	case strings.Contains(n, "2080 super"):
		return 496.0
	case strings.Contains(n, "2080"):
		return 448.0
	case strings.Contains(n, "2070 super"):
		return 448.0
	case strings.Contains(n, "2070"):
		return 448.0
	case strings.Contains(n, "2060 super"):
		return 448.0
	case strings.Contains(n, "2060"):
		return 336.0
	// GTX 16 series (Turing, no RT cores)
	case strings.Contains(n, "1660 ti"):
		return 288.0
	case strings.Contains(n, "1660 super"):
		return 336.0
	case strings.Contains(n, "1660"):
		return 192.0
	case strings.Contains(n, "1650 super"):
		return 192.0
	case strings.Contains(n, "1650"):
		return 128.0
	// GTX 10 series (Pascal)
	case strings.Contains(n, "1080 ti"):
		return 484.0
	case strings.Contains(n, "1080"):
		return 320.0
	case strings.Contains(n, "1070 ti"):
		return 256.0
	case strings.Contains(n, "1070"):
		return 256.0
	case strings.Contains(n, "1060"):
		return 192.0
	// Data center
	case strings.Contains(n, "a100"):
		return 2039.0
	case strings.Contains(n, "h100"):
		return 3350.0
	case strings.Contains(n, "h200"):
		return 4800.0
	case strings.Contains(n, "p40"):
		return 346.0
	case strings.Contains(n, "p100"):
		return 732.0
	case strings.Contains(n, "v100"):
		return 900.0
	default:
		return 0.0
	}
}
