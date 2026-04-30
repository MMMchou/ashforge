package hardware

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// detectMetalGPU is set by probe_darwin.go on macOS to detect Apple Silicon GPU.
// On other platforms it remains nil.
var detectMetalGPU func() *GPUInfo

// HardwareProbe contains detected hardware information
type HardwareProbe struct {
	GPUs []GPUInfo `json:"gpus"`
	CPU  CPUInfo   `json:"cpu"`
	RAM  RAMInfo   `json:"ram"`
	OS   OSInfo    `json:"os"`
}

// GPUInfo holds per-GPU information
type GPUInfo struct {
	Index            int     `json:"index"`
	Name             string  `json:"name"`
	VRAM_MB          int     `json:"vram_mb"`
	VRAMUsed_MB      int     `json:"vram_used_mb"`
	VRAMFree_MB      int     `json:"vram_free_mb"`
	ComputeCap       string  `json:"compute_cap"`        // "8.9" for SM89
	CUDADriver       string  `json:"cuda_driver"`        // "12.8"
	MemBandwidth_GBs float64 `json:"mem_bandwidth_gbs"`  // 1008.0
	IsBlackwell      bool    `json:"is_blackwell"`       // SM120x
}

// CPUInfo holds CPU information
type CPUInfo struct {
	Model       string `json:"model"`
	Cores       int    `json:"cores"`
	Threads     int    `json:"threads"`
	HasAVX2     bool   `json:"has_avx2"`
	HasAVX512   bool   `json:"has_avx512"`
}

// RAMInfo holds RAM information
type RAMInfo struct {
	Total_MB uint64 `json:"total_mb"`
	Used_MB  uint64 `json:"used_mb"`
	Free_MB  uint64 `json:"free_mb"`
	Type     string `json:"type"` // "ddr4", "ddr5", "unknown"
}

// OSInfo holds OS information
type OSInfo struct {
	Platform string `json:"platform"` // "windows", "linux", "darwin"
	Arch     string `json:"arch"`     // "amd64", "arm64"
	Version  string `json:"version"`  // OS version string
}

// Probe detects hardware and returns a HardwareProbe
func Probe() (*HardwareProbe, error) {
	probe := &HardwareProbe{}

	// Detect GPUs (NVIDIA first, then AMD, then Apple Silicon Metal)
	gpus, err := detectNVIDIA()
	if err == nil && len(gpus) > 0 {
		probe.GPUs = gpus
	} else {
		gpus, err = detectAMD()
		if err == nil && len(gpus) > 0 {
			probe.GPUs = gpus
		} else if detectMetalGPU != nil {
			if metalGPU := detectMetalGPU(); metalGPU != nil {
				probe.GPUs = []GPUInfo{*metalGPU}
			}
		}
	}

	// Detect CPU
	cpu, err := detectCPU()
	if err != nil {
		return nil, fmt.Errorf("CPU detection failed: %w", err)
	}
	probe.CPU = cpu

	// Detect RAM
	ram, err := detectRAM()
	if err != nil {
		return nil, fmt.Errorf("RAM detection failed: %w", err)
	}
	probe.RAM = ram

	// Detect OS
	probe.OS = detectOS()

	return probe, nil
}

// PrimaryGPU returns the strongest GPU (highest bandwidth, then VRAM, then SM).
// Used for display/logging only — capability decisions use ClusterCaps().
func (p *HardwareProbe) PrimaryGPU() *GPUInfo {
	if len(p.GPUs) == 0 {
		return nil
	}
	primary := &p.GPUs[0]
	for i := range p.GPUs {
		g := &p.GPUs[i]
		if g.MemBandwidth_GBs > primary.MemBandwidth_GBs {
			primary = g
		} else if g.MemBandwidth_GBs == primary.MemBandwidth_GBs && g.VRAM_MB > primary.VRAM_MB {
			primary = g
		}
	}
	return primary
}

// TotalVRAM_MB returns the sum of VRAM across all GPUs
func (p *HardwareProbe) TotalVRAM_MB() int {
	total := 0
	for _, g := range p.GPUs {
		total += g.VRAM_MB
	}
	return total
}

// GPUCount returns the number of GPUs
func (p *HardwareProbe) GPUCount() int {
	return len(p.GPUs)
}

// SMVersion returns the SM version as an integer (e.g. "7.5" → 75, "12.0" → 120).
// Returns 0 if unknown. Uses PrimaryGPU — for multi-GPU min SM, use ClusterCaps().
func (p *HardwareProbe) SMVersion() int {
	gpu := p.PrimaryGPU()
	return gpuSMVersion(gpu)
}

// gpuSMVersion parses SM version from a single GPU's ComputeCap string.
func gpuSMVersion(gpu *GPUInfo) int {
	if gpu == nil || gpu.ComputeCap == "" {
		return 0
	}
	parts := strings.SplitN(gpu.ComputeCap, ".", 2)
	if len(parts) == 0 {
		return 0
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0
	}
	minor := 0
	if len(parts) == 2 {
		minor, _ = strconv.Atoi(parts[1])
	}
	return major*10 + minor
}

// ClusterCapabilities aggregates multi-GPU capabilities.
// Resources (VRAM, bandwidth) are summed; capabilities (SM, iso3, FA) take the intersection.
type ClusterCapabilities struct {
	TotalVRAM_MB   int
	TotalBandwidth float64
	MinSM          int      // lowest SM across all GPUs
	MinBandwidth   float64  // lowest bandwidth across all GPUs
	SupportsIso3   bool     // all GPUs SM >= 80
	SupportsFA     bool     // all GPUs SM >= 75
	HasBlackwell   bool     // any GPU is Blackwell (SM120+)
	Primary        *GPUInfo // strongest GPU (for display)
}

// ClusterCaps computes aggregated capabilities across all GPUs.
func (p *HardwareProbe) ClusterCaps() ClusterCapabilities {
	caps := ClusterCapabilities{
		MinSM:        999,
		MinBandwidth: 1e9,
		SupportsIso3: true,
		SupportsFA:   true,
	}

	if len(p.GPUs) == 0 {
		caps.MinSM = 0
		caps.MinBandwidth = 0
		caps.SupportsIso3 = false
		caps.SupportsFA = false
		return caps
	}

	for i := range p.GPUs {
		g := &p.GPUs[i]
		caps.TotalVRAM_MB += g.VRAM_MB
		caps.TotalBandwidth += g.MemBandwidth_GBs

		sm := gpuSMVersion(g)
		if sm < caps.MinSM {
			caps.MinSM = sm
		}
		if g.MemBandwidth_GBs < caps.MinBandwidth {
			caps.MinBandwidth = g.MemBandwidth_GBs
		}
		if sm < 80 {
			caps.SupportsIso3 = false
		}
		if sm < 75 {
			caps.SupportsFA = false
		}
		if g.IsBlackwell {
			caps.HasBlackwell = true
		}
	}

	caps.Primary = p.PrimaryGPU()
	return caps
}

// SupportsFlashAttn returns true if ALL GPUs support Flash Attention (SM75+).
// Uses ClusterCaps intersection — one weak card disables FA for the whole cluster.
func (p *HardwareProbe) SupportsFlashAttn() bool {
	return p.ClusterCaps().SupportsFA
}

// HasNVLink returns true if NVLink is detected between GPUs.
// Only meaningful when GPUCount() > 1.
func (p *HardwareProbe) HasNVLink() bool {
	if len(p.GPUs) < 2 {
		return false
	}
	return detectNVLink()
}

// TensorSplitArg returns the --tensor-split value for multi-GPU setups.
// Weights are VRAM × bandwidth so that weaker cards (low VRAM or low bandwidth)
// get fewer layers and don't bottleneck the whole system.
// Returns "" for single-GPU setups (no split needed).
func (p *HardwareProbe) TensorSplitArg() string {
	if len(p.GPUs) < 2 {
		return ""
	}

	weights := make([]float64, len(p.GPUs))
	for i, g := range p.GPUs {
		bw := g.MemBandwidth_GBs
		if bw <= 0 {
			bw = 200 // conservative fallback
		}
		weights[i] = float64(g.VRAM_MB) * bw
	}

	// Normalize to percentages and format
	total := 0.0
	for _, w := range weights {
		total += w
	}
	if total == 0 {
		return ""
	}

	parts := make([]string, len(weights))
	for i, w := range weights {
		pct := w / total * 100
		parts[i] = fmt.Sprintf("%.0f", pct)
	}
	return strings.Join(parts, ",")
}

// Fingerprint returns a unique hardware fingerprint for profile caching
// Format: "sm89_24576mb_ddr5" (single GPU) or "sm89_2x24576mb_ddr5" (multi GPU)
func (p *HardwareProbe) Fingerprint() string {
	gpu := p.PrimaryGPU()
	if gpu == nil {
		return fmt.Sprintf("cpu_%dmb_%s", p.RAM.Total_MB, p.RAM.Type)
	}

	// Remove dots from compute cap: "8.9" -> "89", "6.1" -> "61"
	cc := strings.ReplaceAll(gpu.ComputeCap, ".", "")

	if len(p.GPUs) > 1 {
		return fmt.Sprintf("sm%s_%dx%dmb_%s", cc, len(p.GPUs), gpu.VRAM_MB, p.RAM.Type)
	}
	return fmt.Sprintf("sm%s_%dmb_%s", cc, gpu.VRAM_MB, p.RAM.Type)
}

// JSON returns JSON representation
func (p *HardwareProbe) JSON() (string, error) {
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}
