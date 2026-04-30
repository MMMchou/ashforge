package optimizer

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/MMMchou/ashforge/internal/config"
	"github.com/MMMchou/ashforge/internal/engine"
	"github.com/MMMchou/ashforge/internal/hardware"
	"github.com/MMMchou/ashforge/internal/model"
)

// maxUpwardProbes limits how many times we double ctx after the first success.
// Prevents 8K→16K→32K→64K→128K→256K chains on large VRAM (each costs 30-60s).
const maxUpwardProbes = 3

// minContextTPS is the minimum speed for the "context" mode to be usable.
const minContextTPS = 15.0

// ProbeResult stores one warmup probe measurement.
type ProbeResult struct {
	Ctx  int
	TPS  float64
	Args []string
	VRAM int
	OOM  bool
}

// ModeProfile stores one of the three mode options.
type ModeProfile struct {
	Priority string   `json:"priority"` // "speed" / "balanced" / "context"
	CtxSize  int      `json:"ctx_size"`
	TPS      float64  `json:"tps"`
	Args     []string `json:"args"`
}

// OptimizedProfile is the result of warmup benchmark
type OptimizedProfile struct {
	ModelID     string        `json:"model_id"`
	HardwareFP  string        `json:"hardware_fp"`
	Quant       string        `json:"quant"`
	Mode        string        `json:"mode"`
	Priority    string        `json:"priority"`           // current selected mode
	MeasuredTPS float64       `json:"measured_tps"`
	VRAMUsed_MB int           `json:"vram_used_mb"`
	LaunchArgs  []string      `json:"launch_args"`
	Profiles    []ModeProfile `json:"profiles,omitempty"` // three mode options
	CreatedAt   string        `json:"created_at"`
}

// ClearProfileCache deletes the cached profile for a given model+hardware combo.
func ClearProfileCache(modelID string, hw *hardware.HardwareProbe) error {
	fingerprint := hw.Fingerprint()
	profilePath := filepath.Join(config.ProfileDir(), fmt.Sprintf("%s_%s.json", modelID, fingerprint))
	if err := os.Remove(profilePath); err != nil {
		if os.IsNotExist(err) {
			return nil // nothing to clear
		}
		return err
	}
	return nil
}

// Warmup runs the warmup benchmark and returns optimized parameters.
//
// ┌─────────────────────────────────────────────────────────────────┐
// │ Warmup Flow                                                     │
// │                                                                 │
// │ 1. Cache check                                                  │
// │    - CtxOverride > 0 → skip cache                               │
// │    - Valid cache → return immediately (2s startup)               │
// │    - --fast + no cache → error                                  │
// │                                                                 │
// │ 2. Phase 1: Coarse search (power-of-2 steps, max 8 probes)     │
// │    Start from IdealStartCtx×2 (full_gpu) or nativeMax (MoE)    │
// │    a) OOM       → record failedCtx, halve, retry                │
// │    b) Too slow  → record failedCtx, halve, retry                │
// │    c) Success   → record bestCtx, double up (max 3 times)       │
// │    d) Double fails → stop, we have [bestCtx, failedCtx] range   │
// │                                                                 │
// │ 3. Phase 1.5: Fine search (4K precision binary search)          │
// │    Between bestCtx and failedCtx, binary search for real max    │
// │    Only runs when failedCtx > bestCtx (gap to explore)          │
// │                                                                 │
// │ 4. Phase 2: ubatch tuning                                       │
// │    Test 128 vs 512, pick faster. Uses bestCtx from above.       │
// │                                                                 │
// │ 5. Save profile to cache                                        │
// └─────────────────────────────────────────────────────────────────┘
func Warmup(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe, fast bool, modeOverride string) (*OptimizedProfile, error) {
	fingerprint := hw.Fingerprint()
	profilePath := filepath.Join(config.ProfileDir(), fmt.Sprintf("%s_%s.json", profile.ModelID, fingerprint))

	// Determine preferred priority
	cfg, _ := config.Load()
	priority := cfg.Priority
	if priority == "" {
		priority = "balanced"
	}
	if modeOverride != "" {
		priority = modeOverride
	}

	// User specified --ctx-size → skip cache, use their value directly
	if profile.CtxOverride > 0 {
		fmt.Printf("      用户指定 ctx=%d，跳过缓存\n", profile.CtxOverride)
	} else {
		// Check cache (spec: second launch should be 2s)
		if cached, err := loadCachedProfile(profilePath); err == nil {
			if isCacheValid(cached, fingerprint) {
				// If cache has Profiles and user wants a different mode, switch without re-warmup
				if len(cached.Profiles) > 0 && modeOverride != "" {
					for _, mp := range cached.Profiles {
						if mp.Priority == modeOverride {
							cached.Priority = modeOverride
							cached.MeasuredTPS = mp.TPS
							cached.LaunchArgs = mp.Args
							saveProfile(cached, profilePath)
							fmt.Printf("      切换到%s模式（%s ctx · %.1f tok/s）\n", modeName(modeOverride), fmtCtx(mp.CtxSize), mp.TPS)
							return cached, nil
						}
					}
				}
				// Use cached profile as-is
				if len(cached.Profiles) == 0 {
					// Old cache without Profiles → re-warmup
					fmt.Printf("      旧缓存格式，重新探测\n")
				} else {
					created, _ := time.Parse(time.RFC3339, cached.CreatedAt)
					age := time.Since(created)
					ageStr := fmt.Sprintf("%.0f 天前", age.Hours()/24)
					if age.Hours() < 24 {
						ageStr = fmt.Sprintf("%.0f 小时前", age.Hours())
					}
					cachedCtx := extractArgValue(cached.LaunchArgs, "--ctx-size")
					fmt.Printf("      使用上次配置（%s ctx · %.1f tok/s · %s模式 · %s）\n", cachedCtx, cached.MeasuredTPS, modeName(cached.Priority), ageStr)
					fmt.Printf("      提示: ashforge run model --mode speed/balanced/context 切换模式\n")
					return cached, nil
				}
			} else {
				fmt.Printf("      Cache expired, re-running warmup\n")
			}
		}
	}

	// --fast with no cache: skip warmup entirely
	if fast {
		fmt.Printf("      No cached profile, using defaults\n")
		return nil, fmt.Errorf("no cached profile available")
	}

	port := cfg.LlamaPort + 10

	// Blackwell JIT warmup: first run of CUDA 12.4 binary on SM120 needs PTX JIT compilation (~60s).
	// We run llama-server --version once to trigger JIT and populate the CUDA cache.
	// Subsequent launches (all probes + final start) will be fast (~2s).
	if hw.ClusterCaps().HasBlackwell {
		if err := ensureCUDAJITCache(binaryPath); err != nil {
			fmt.Printf("      ⚠  JIT 预热失败: %v\n", err)
		}
	}

	// RAM safety check
	if warning := checkRAMSafety(hw, profile); warning != "" {
		fmt.Printf("      %s\n", warning)
	}

	// --- Phase 1: probe max ctx (speed-aware) ---
	nativeMax := profile.NativeCtx
	if nativeMax <= 0 {
		nativeMax = 131072
	}

	batchSize, ubatchSize := 512, 128
	if profile.Mode == "moe_offload" || profile.Mode == "moe_partial" {
		batchSize, ubatchSize = 4096, 512
		if profile.Mode == "moe_partial" {
			ubatchSize = 4096 // community-tested: -ub 4096 improves prompt processing for partial offload
		}
	}

	// Phase 1 now collects ALL successful probes (no threshold filtering).
	// Three modes are derived from the collected data points after probing.
	// MoE: speed doesn't vary with ctx (PCIe-bound), so just find max ctx.
	var probeResults []ProbeResult

	var bestCtx int
	var bestTPS float64
	var bestArgs []string
	var bestVRAM int

	if profile.CtxOverride > 0 {
		// User override: use exact value, no alignment
		ctxFixed := profile.CtxOverride
		// Clamp to model's native max
		if profile.NativeCtx > 0 && ctxFixed > profile.NativeCtx {
			fmt.Printf("      Warning: ctx %d 超过模型最大值 %d，已截断\n", ctxFixed, profile.NativeCtx)
			ctxFixed = profile.NativeCtx
		}
		fmt.Printf("      User override: ctx=%s ... ", fmtCtx(ctxFixed))
		args := BuildArgs(profile, binaryPath, modelPath, port, hw, ctxFixed, batchSize, ubatchSize)
		tps, vram, err := runBenchmarkRound(binaryPath, args, port)
		if err != nil {
			return nil, fmt.Errorf("user-specified ctx=%s failed to start (OOM?)", fmtCtx(ctxFixed))
		}
		fmt.Printf("%.1f tok/s\n", tps)
		bestCtx = ctxFixed
		bestTPS = tps
		bestArgs = args
		bestVRAM = vram
	} else {
		// --- Phase 1: coarse search (power-of-2 steps, max 8 probes) ---
		// Starting point:
		//   full_gpu:     IdealStartCtx×2 (oobabooga formula + 1 step headroom)
		//   moe_offload:  nativeMax (formula wrong for MoE, GPU only has KV cache)
		var startCtx int
		if profile.Mode == "moe_offload" || profile.Mode == "moe_partial" {
			startCtx = nativeMax
		} else {
			ideal := engine.IdealStartCtx(profile, hw)
			// Blackwell (SM120) + CUDA 12.4 binary: VRAM margins are tight,
			// ideal×2 causes all probes to OOM. Start from ideal directly.
			if hw.ClusterCaps().HasBlackwell {
				startCtx = ideal
			} else {
				startCtx = ideal * 2
			}
			if startCtx > nativeMax {
				startCtx = nativeMax
			}
		}
		startCtx = alignToPow2(startCtx)

		var failedCtx int // smallest ctx that failed (OOM or too slow)
		upwardProbes := 0 // count of upward doublings after first success
		ctxTry := startCtx
		for attempt := 1; attempt <= 8; attempt++ {
			fmt.Printf("      Probe %d: ctx=%s ... ", attempt, fmtCtx(ctxTry))
			args := BuildArgs(profile, binaryPath, modelPath, port, hw, ctxTry, batchSize, ubatchSize)
			tps, vram, err := runBenchmarkRound(binaryPath, args, port)

			if err != nil {
				// OOM: record ceiling, halve and retry downward
				fmt.Printf("OOM\n")
				probeResults = append(probeResults, ProbeResult{Ctx: ctxTry, OOM: true})
				if failedCtx == 0 || ctxTry < failedCtx {
					failedCtx = ctxTry
				}
				ctxTry /= 2
				if ctxTry < 4096 {
					ctxTry = 4096
				}
				if ctxTry == 4096 && attempt > 1 {
					break
				}
				continue
			}

			// Success: record probe result
			fmt.Printf("%.1f tok/s\n", tps)
			probeResults = append(probeResults, ProbeResult{Ctx: ctxTry, TPS: tps, Args: args, VRAM: vram})

			if bestCtx == 0 || ctxTry > bestCtx {
				bestCtx = ctxTry
				bestTPS = tps
				bestArgs = args
				bestVRAM = vram
			}

			// Try doubling to find the ceiling (max 3 upward probes)
			if failedCtx == 0 && ctxTry < nativeMax && upwardProbes < maxUpwardProbes {
				upwardProbes++
				nextCtx := alignToPow2(ctxTry * 2)
				if nextCtx > nativeMax {
					nextCtx = alignToPow2(nativeMax)
				}
				if nextCtx > ctxTry {
					ctxTry = nextCtx
					continue
				}
			}
			break
		}

		// --- Phase 1.5: fine search (4K precision binary search) ---
		// Between bestCtx (works) and failedCtx (OOM), find the real max.
		if bestCtx > 0 && failedCtx > bestCtx {
			lo := bestCtx
			hi := failedCtx
			for hi-lo > 4096 {
				mid := ((lo + hi) / 2 / 1024) * 1024 // align to 1K boundary
				if mid <= lo || mid >= hi {
					break
				}
				fmt.Printf("      Fine: ctx=%s ... ", fmtCtx(mid))
				args := BuildArgs(profile, binaryPath, modelPath, port, hw, mid, batchSize, ubatchSize)
				tps, vram, err := runBenchmarkRound(binaryPath, args, port)
				if err != nil {
					fmt.Printf("OOM\n")
					probeResults = append(probeResults, ProbeResult{Ctx: mid, OOM: true})
					hi = mid
					continue
				}
				fmt.Printf("%.1f tok/s\n", tps)
				probeResults = append(probeResults, ProbeResult{Ctx: mid, TPS: tps, Args: args, VRAM: vram})
				lo = mid
				bestCtx = mid
				bestTPS = tps
				bestArgs = args
				bestVRAM = vram
			}
		}
	}

	if bestCtx == 0 {
		return nil, fmt.Errorf("all ctx probes failed (tried down to 4K)")
	}

	// MoE: warmup 后把实测 VRAM 回写到 profile。
	// SelectKVCacheType 下次调用时会用这个值代替估算值，更准确。
	if (profile.Mode == "moe_offload" || profile.Mode == "moe_partial") && bestVRAM > 0 {
		profile.MeasuredVRAM_MB = bestVRAM
	}

	// --- Phase 2: ubatch 探测 ---
	// 低带宽卡（< 200 GB/s）只测 128，避免大 ubatch 加剧带宽瓶颈
	// 用 ClusterCaps.MinBandwidth：任何一张卡带宽低就限制 ubatch
	ubatchCandidates := []int{128, 512}
	if minBW := hw.ClusterCaps().MinBandwidth; minBW > 0 && minBW < 200 {
		ubatchCandidates = []int{128}
	}
	var ubBestTPS float64
	var ubBestArgs []string
	var ubBestVRAM int
	fmt.Printf("      Tune ubatch: ")
	for _, ub := range ubatchCandidates {
		args2 := BuildArgs(profile, binaryPath, modelPath, port, hw, bestCtx, 512, ub)
		tps2, vram2, err := runBenchmarkRound(binaryPath, args2, port)
		if err != nil {
			fmt.Printf("ub=%d failed; ", ub)
			continue
		}
		fmt.Printf("ub=%d → %.1f tok/s; ", ub, tps2)
		if tps2 > ubBestTPS {
			ubBestTPS = tps2
			ubBestArgs = args2
			ubBestVRAM = vram2
		}
	}
	fmt.Println()
	// Phase 2 结果覆盖 Phase 1（只要有成功的测试）
	if ubBestTPS > 0 {
		bestTPS = ubBestTPS
		bestArgs = ubBestArgs
		bestVRAM = ubBestVRAM
	}

	fmt.Printf("      ✓ %.1f tok/s @ %s ctx\n", bestTPS, fmtCtx(bestCtx))
	if profile.Mode == "moe_offload" || profile.Mode == "moe_partial" {
		fmt.Printf("      ℹ MoE offload · speed limited by PCIe bandwidth, not context size\n")
	}

	// --- Derive three mode profiles from probe results ---
	modes := deriveModeProfiles(probeResults, profile, modelPath, port, hw, batchSize, ubatchSize)

	// MoE or only one distinct ctx → skip selection, use context mode
	if profile.Mode == "moe_offload" || profile.Mode == "moe_partial" || len(modes) <= 1 {
		if len(modes) == 0 {
			modes = []ModeProfile{{Priority: "balanced", CtxSize: bestCtx, TPS: bestTPS, Args: bestArgs}}
		}
		priority = modes[len(modes)-1].Priority // use largest ctx
	} else {
		// Interactive selection
		priority = promptModeSelection(modes, priority)
	}

	// Find selected mode's data
	var selectedMode *ModeProfile
	for i := range modes {
		if modes[i].Priority == priority {
			selectedMode = &modes[i]
			break
		}
	}
	if selectedMode == nil {
		selectedMode = &ModeProfile{Priority: priority, CtxSize: bestCtx, TPS: bestTPS, Args: bestArgs}
	}

	// Save priority to config
	cfg.Priority = priority
	config.Save(cfg)

	// Save profile
	optimized := &OptimizedProfile{
		ModelID:     profile.ModelID,
		HardwareFP:  fingerprint,
		Quant:       profile.Quant,
		Mode:        profile.Mode,
		Priority:    priority,
		MeasuredTPS: selectedMode.TPS,
		VRAMUsed_MB: bestVRAM,
		LaunchArgs:  selectedMode.Args,
		Profiles:    modes,
		CreatedAt:   time.Now().Format(time.RFC3339),
	}

	if err := saveProfile(optimized, profilePath); err != nil {
		fmt.Printf("      Warning: failed to save profile: %v\n", err)
	} else {
		fmt.Printf("      Saved profile: %s\n", profilePath)
	}

	return optimized, nil
}

// alignToPow2 rounds down to the nearest power of 2 (min 4096).
func alignToPow2(n int) int {
	powers := []int{524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096}
	for _, p := range powers {
		if n >= p {
			return p
		}
	}
	return 4096
}

// BuildArgs constructs llama-server arguments with explicit ctx size and batch sizes.
func BuildArgs(profile *model.DeployProfile, binaryPath, modelPath string, port int, hw *hardware.HardwareProbe, ctxSize, batchSize, ubatchSize int) []string {
	vramMB := hw.TotalVRAM_MB() // 多卡总VRAM

	// KV cache 类型：基于 VRAM 计算自动选择最快的类型
	// 优先 f16（最快），装不下就降到 q8_0+q4_0，再不行用 iso3
	kvK, kvV := profile.SelectKVCacheType(vramMB, ctxSize)

	// Parallel slots: 单用户场景用 1 slot（速度优先）
	// 多 slot 会预分配多份 KV cache，压缩可用 ctx，本地单人场景得不偿失
	parallel := 1

	useMlock := shouldMlock(hw, profile)

	args := []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--n-gpu-layers", "999",
		"--parallel", strconv.Itoa(parallel),
		"--cont-batching",
		"--metrics",
		"--no-webui",
		"--cache-reuse", strconv.Itoa(cacheReuseForCtx(ctxSize)),
		"-ctk", kvK,
		"-ctv", kvV,
		"--ctx-size", strconv.Itoa(ctxSize),
		"--threads", strconv.Itoa(threadsForMode(profile.Mode, hw)),
		"--batch-size", strconv.Itoa(batchSize),
		"--ubatch-size", strconv.Itoa(ubatchSize),
	}

	// --kv-unified: pre-allocate KV cache as one contiguous block.
	// Skip on: (1) Blackwell — CUDA 12.4 binary on 13.x driver over-allocates
	//          (2) Multi-GPU — unified KV lands entirely on GPU 0, causing OOM
	//              even when total VRAM across all cards is sufficient
	// Without this flag, llama.cpp uses paged KV allocation (grows on demand, spreads across devices).
	if !hw.ClusterCaps().HasBlackwell && hw.GPUCount() <= 1 {
		args = append(args, "--kv-unified")
	}

	// MoE offload modes:
	// moe_offload: all expert layers on CPU (--cpu-moe)
	// moe_partial: some expert layers on CPU (--n-cpu-moe N), rest on GPU
	// --fit on CANNOT be combined with --cpu-moe or --n-cpu-moe (ik_llama.cpp docs).
	// Only use --fit on for full_gpu (dense) models.
	if profile.Mode == "moe_offload" {
		args = append(args, "--cpu-moe")
	} else if profile.Mode == "moe_partial" && profile.NCpuMoe > 0 {
		args = append(args, "--n-cpu-moe", strconv.Itoa(profile.NCpuMoe))
	} else {
		args = append(args, "--fit", "on")
	}

	// Flash Attention: SM75+ (Turing and newer)
	if hw.SupportsFlashAttn() {
		args = append(args, "--flash-attn", "on")
	}

	// Multi-GPU: NVLink + graph support → graph split mode; otherwise → weighted tensor-split
	// MoE models: skip -sm graph (tensor parallel causes expert buffer explosion), use layer split
	if hw.GPUCount() > 1 {
		if hw.HasNVLink() && engine.SupportsGraphSplit(binaryPath) && profile.Arch != "moe" {
			args = append(args, "-sm", "graph")
		} else if ts := hw.TensorSplitArg(); ts != "" {
			args = append(args, "--tensor-split", ts)
		}
	}

	// Hybrid architecture (DeltaNet/SSM): full SWA KV cache for correct long-context behavior
	if profile.IsHybrid {
		args = append(args, "--swa-full")
	}

	// mlock: lock model in RAM when headroom is sufficient
	if useMlock {
		args = append(args, "--mlock")
	}

	// mmap: accelerate model loading via memory-mapped I/O
	// Skip when mlock is active (mlock+mmap can conflict on some systems)
	if !useMlock && shouldMmap(hw, profile) {
		args = append(args, "--mmap")
	}

	// MTP speculative decoding: Qwen3.6 models have native MTP heads, 40-80% speed boost
	if profile.NativeMTP {
		args = append(args, "--num-speculative-tokens", "3")
	}

	// N-gram lookup: zero-cost speculative decoding for models without MTP
	if !profile.NativeMTP {
		args = append(args, "--lookup", "8")
	}

	// KV cache defragmentation: auto-compact when fragmentation > 10%
	args = append(args, "--defrag-thold", "0.1")

	return args
}

// runBenchmarkRound starts llama-server, runs a warmup prompt, measures tok/s
func runBenchmarkRound(binaryPath string, args []string, port int) (tps float64, vramMB int, err error) {
	// Start llama-server
	eng, err := startBenchServer(binaryPath, args, port)
	if err != nil {
		return 0, 0, err
	}
	defer stopBenchServer(eng)

	// Send warmup prompt and measure
	tps, err = measureTPS(port)
	if err != nil {
		return 0, 0, err
	}

	// Get VRAM usage from metrics
	vramMB = getVRAMFromMetrics(port)

	return tps, vramMB, nil
}

// startBenchServer starts llama-server for benchmarking
func startBenchServer(binaryPath string, args []string, port int) (*os.Process, error) {
	// Create log file for debugging
	logPath := filepath.Join(config.LogDir(), fmt.Sprintf("warmup-%d.log", time.Now().Unix()))
	logFile, err := os.Create(logPath)
	if err != nil {
		logFile = nil
	}

	cmd := exec.Command(binaryPath, args...)
	// Set LD_LIBRARY_PATH to binary's directory so libmtmd.so/libllama.so are found
	binaryDir := filepath.Dir(binaryPath)
	cmd.Env = append(os.Environ(),
		"LD_LIBRARY_PATH="+binaryDir+":"+os.Getenv("LD_LIBRARY_PATH"),
	)
	// MoE + multi-GPU: disable CUDA graph capture to avoid memory leak (llama.cpp #20315)
	if containsFlag(args, "--n-cpu-moe", "--cpu-moe") {
		cmd.Env = append(cmd.Env, "GGML_CUDA_DISABLE_GRAPHS=1")
	}
	if logFile != nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	engine.SetProcAttr(cmd)

	if err := cmd.Start(); err != nil {
		if logFile != nil {
			logFile.Close()
		}
		return nil, fmt.Errorf("failed to start benchmark server: %w", err)
	}

	// Close log file when process exits
	go func() {
		cmd.Wait()
		if logFile != nil {
			logFile.Close()
		}
	}()

	// Wait for health endpoint
	deadline := time.Now().Add(180 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return cmd.Process, nil
			}
		}
		time.Sleep(1 * time.Second)
	}

	cmd.Process.Kill()
	return nil, fmt.Errorf("benchmark server failed to start within 60s (log: %s)", logPath)
}

// stopBenchServer stops the benchmark server
func stopBenchServer(proc *os.Process) {
	if proc != nil {
		proc.Kill()
		proc.Wait()
	}
}

// measureTPS sends a prompt and measures decode tokens per second
func measureTPS(port int) (float64, error) {
	// Use pure ASCII prompt to avoid UTF-8 encoding issues on Windows bash
	reqBody := `{
		"model": "test",
		"messages": [{"role": "user", "content": "Write a Python quicksort implementation with detailed comments and edge case handling."}],
		"max_tokens": 200,
		"stream": false
	}`

	start := time.Now()
	resp, err := http.Post(
		fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
		"application/json",
		strings.NewReader(reqBody),
	)
	if err != nil {
		return 0, fmt.Errorf("benchmark request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read response: %w", err)
	}
	elapsed := time.Since(start)

	// Parse response to get token count
	var result struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("failed to parse response: %w", err)
	}

	tokens := result.Usage.CompletionTokens
	if tokens == 0 {
		return 0, fmt.Errorf("no tokens generated")
	}

	tps := float64(tokens) / elapsed.Seconds()
	return tps, nil
}

// getVRAMFromMetrics reads VRAM usage from llama-server metrics endpoint
func getVRAMFromMetrics(port int) int {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", port))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	// Look for VRAM metric in Prometheus format
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "llama_vram_usage_bytes") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				val, _ := strconv.ParseFloat(parts[1], 64)
				return int(val / (1024 * 1024))
			}
		}
	}
	return 0
}

func loadCachedProfile(path string) (*OptimizedProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var profile OptimizedProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

// isCacheValid checks if cached profile is still valid (fingerprint match + < 30 days old)
func isCacheValid(cached *OptimizedProfile, currentFingerprint string) bool {
	// Check fingerprint match
	if cached.HardwareFP != currentFingerprint {
		return false
	}
	// Check age (30 days)
	created, err := time.Parse(time.RFC3339, cached.CreatedAt)
	if err != nil {
		return false
	}
	return time.Since(created) < 30*24*time.Hour
}

func saveProfile(profile *OptimizedProfile, path string) error {
	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// threadsForMode returns the optimal thread count based on inference mode.
// full_gpu: GPU does all the work, CPU just orchestrates — 2 threads suffice.
// moe_offload/moe_partial: CPU handles expert layers, needs real parallelism.
func threadsForMode(mode string, hw *hardware.HardwareProbe) int {
	if mode == "moe_offload" || mode == "moe_partial" {
		t := hw.CPU.Cores / 2
		if t < 4 {
			t = 4
		}
		return t
	}
	return 2
}

// checkRAMSafety warns when free RAM is too low for comfortable operation.
// Reserve = max(Total_MB * 20%, 2048 MB) for system stability.
func checkRAMSafety(hw *hardware.HardwareProbe, profile *model.DeployProfile) string {
	totalMB := hw.RAM.Total_MB
	freeMB := hw.RAM.Free_MB

	reserveMB := totalMB / 5
	if reserveMB < 2048 {
		reserveMB = 2048
	}

	var modelRAM_MB uint64
	if profile.Mode == "moe_offload" || profile.Mode == "moe_partial" {
		modelRAM_MB = uint64(profile.Size_GB * 1024 * 0.9)
	} else {
		modelRAM_MB = uint64(profile.Size_GB * 1024 * 0.1)
	}

	needed := modelRAM_MB + reserveMB
	if freeMB < needed {
		return fmt.Sprintf("Warning: RAM tight: %.1f GB free, need ~%.1f GB (%.1f GB model + %.1f GB reserve)",
			float64(freeMB)/1024,
			float64(needed)/1024,
			float64(modelRAM_MB)/1024,
			float64(reserveMB)/1024)
	}
	return ""
}

// shouldMlock returns true when there's enough RAM headroom to safely lock
// the model in memory. mlock prevents swapping model pages to disk.
// Only enabled when remaining RAM after model load > 30% of total.
func shouldMlock(hw *hardware.HardwareProbe, profile *model.DeployProfile) bool {
	totalMB := float64(hw.RAM.Total_MB)
	freeMB := float64(hw.RAM.Free_MB)
	if totalMB == 0 {
		return false
	}

	var modelRAM_MB float64
	if profile.Mode == "moe_offload" {
		modelRAM_MB = profile.Size_GB * 1024 * 0.9
	} else {
		modelRAM_MB = profile.Size_GB * 1024 * 0.1
	}

	remainingAfterLoad := freeMB - modelRAM_MB
	return remainingAfterLoad > totalMB*0.3
}

// cacheReuseForCtx returns dynamic cache-reuse size based on context.
// Larger ctx → longer system prompts → more tokens worth caching.
func cacheReuseForCtx(ctxSize int) int {
	switch {
	case ctxSize >= 32768:
		return 1024
	case ctxSize >= 8192:
		return 512
	default:
		return 256
	}
}

// shouldMmap returns true when memory-mapped I/O should be used for faster model loading.
// Enabled when model size < 70% of available RAM (avoids swap pressure).
func shouldMmap(hw *hardware.HardwareProbe, profile *model.DeployProfile) bool {
	freeMB := float64(hw.RAM.Free_MB)
	modelMB := profile.Size_GB * 1024
	return modelMB < freeMB*0.7
}

// containsFlag checks if any of the given flags appear in args.
func containsFlag(args []string, flags ...string) bool {
	for _, a := range args {
		for _, f := range flags {
			if a == f {
				return true
			}
		}
	}
	return false
}

// fmtCtx formats a ctx size for human display: "8K", "12K", "65536" etc.
func fmtCtx(ctx int) string {
	if ctx >= 1024 && ctx%1024 == 0 {
		return fmt.Sprintf("%dK", ctx/1024)
	}
	return strconv.Itoa(ctx)
}

// ensureCUDAJITCache runs llama-server --version to trigger PTX JIT compilation on Blackwell.
// CUDA driver caches compiled kernels to disk (%APPDATA%\NVIDIA\ComputeCache on Windows,
// ~/.nv/ComputeCache on Linux). After this completes, all subsequent launches are fast (~2s).
// Only needed once per binary — the cache persists across reboots.
func ensureCUDAJITCache(binaryPath string) error {
	// Check if JIT cache marker exists (skip if already warmed)
	markerPath := filepath.Join(config.Dir(), "jit_warmed_"+filepath.Base(binaryPath))
	if _, err := os.Stat(markerPath); err == nil {
		return nil // already warmed
	}

	fmt.Printf("      RTX 50 系首次运行，正在编译 CUDA 内核（约 60s，仅需一次）...\n")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, binaryPath, "--version")
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Errorf("JIT compilation timed out (120s)")
	}

	// Write marker so we don't repeat this
	os.WriteFile(markerPath, []byte("1"), 0644)
	fmt.Printf("      ✓ CUDA 内核编译完成，后续启动将秒开\n")
	return err
}

// extractArgValue extracts the value of a flag from args (e.g., "--ctx-size" → "8K")
func extractArgValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			val := args[i+1]
			if flag == "--ctx-size" {
				if n, err := strconv.Atoi(val); err == nil {
					return fmtCtx(n)
				}
			}
			return val
		}
	}
	return ""
}

// deriveModeProfiles computes three mode options from probe results.
// speed: smallest successful ctx (fastest TPS)
// balanced: largest ctx where TPS >= peakTPS * 0.7
// context: largest ctx where TPS >= 15 tok/s
func deriveModeProfiles(results []ProbeResult, profile *model.DeployProfile, modelPath string, port int, hw *hardware.HardwareProbe, batchSize, ubatchSize int) []ModeProfile {
	// Filter successful probes, sort by ctx ascending
	var successes []ProbeResult
	for _, r := range results {
		if !r.OOM && r.TPS > 0 {
			successes = append(successes, r)
		}
	}
	if len(successes) == 0 {
		return nil
	}

	// Sort by ctx ascending (smallest first)
	for i := 0; i < len(successes)-1; i++ {
		for j := i + 1; j < len(successes); j++ {
			if successes[j].Ctx < successes[i].Ctx {
				successes[i], successes[j] = successes[j], successes[i]
			}
		}
	}

	// Find peak TPS (smallest ctx = fastest)
	peakTPS := successes[0].TPS
	for _, r := range successes {
		if r.TPS > peakTPS {
			peakTPS = r.TPS
		}
	}

	// Speed mode: highest TPS probe
	speedProbe := successes[0]
	for _, r := range successes {
		if r.TPS > speedProbe.TPS {
			speedProbe = r
		}
	}

	// Balanced mode: largest ctx where TPS >= peakTPS * 0.7
	balancedThreshold := peakTPS * 0.7
	balancedProbe := speedProbe // fallback to speed
	for _, r := range successes {
		if r.TPS >= balancedThreshold && r.Ctx > balancedProbe.Ctx {
			balancedProbe = r
		}
	}

	// Context mode: largest ctx where TPS >= 15 tok/s
	var contextProbe *ProbeResult
	for i := len(successes) - 1; i >= 0; i-- {
		if successes[i].TPS >= minContextTPS {
			contextProbe = &successes[i]
			break
		}
	}

	// Build modes list, dedup by ctx
	var modes []ModeProfile
	seen := make(map[int]bool)

	// Always add speed
	modes = append(modes, ModeProfile{
		Priority: "speed",
		CtxSize:  speedProbe.Ctx,
		TPS:      speedProbe.TPS,
		Args:     speedProbe.Args,
	})
	seen[speedProbe.Ctx] = true

	// Add balanced if different from speed
	if !seen[balancedProbe.Ctx] {
		modes = append(modes, ModeProfile{
			Priority: "balanced",
			CtxSize:  balancedProbe.Ctx,
			TPS:      balancedProbe.TPS,
			Args:     balancedProbe.Args,
		})
		seen[balancedProbe.Ctx] = true
	} else {
		// balanced same as speed, mark speed as balanced too
		modes[0].Priority = "balanced"
	}

	// Add context if different and exists
	if contextProbe != nil && !seen[contextProbe.Ctx] {
		modes = append(modes, ModeProfile{
			Priority: "context",
			CtxSize:  contextProbe.Ctx,
			TPS:      contextProbe.TPS,
			Args:     contextProbe.Args,
		})
	}

	return modes
}

// promptModeSelection displays the three modes and asks user to choose.
// Returns the selected priority string. Defaults to defaultPriority after 10s.
func promptModeSelection(modes []ModeProfile, defaultPriority string) string {
	fmt.Println()
	fmt.Println("      ┌─ 选择模式 ─────────────────────────────────────┐")
	for i, m := range modes {
		rec := "  "
		if m.Priority == defaultPriority || (defaultPriority == "balanced" && m.Priority == "balanced") {
			rec = " ←"
		}
		fmt.Printf("      │  [%d] %-8s  %5s ctx · %5.1f tok/s %s      │\n",
			i+1, modeName(m.Priority), fmtCtx(m.CtxSize), m.TPS, rec)
	}
	fmt.Println("      └────────────────────────────────────────────────┘")

	// Find default index
	defaultIdx := 0
	for i, m := range modes {
		if m.Priority == defaultPriority {
			defaultIdx = i
			break
		}
	}

	fmt.Printf("      选择 (1-%d，10s 后默认 %d): ", len(modes), defaultIdx+1)

	// Read with timeout
	resultCh := make(chan int, 1)
	go func() {
		var input string
		fmt.Scanln(&input)
		input = strings.TrimSpace(input)
		if n, err := strconv.Atoi(input); err == nil && n >= 1 && n <= len(modes) {
			resultCh <- n - 1
		} else {
			resultCh <- defaultIdx
		}
	}()

	select {
	case idx := <-resultCh:
		fmt.Printf("      ✓ 已选择: %s\n", modeName(modes[idx].Priority))
		return modes[idx].Priority
	case <-time.After(10 * time.Second):
		fmt.Printf("\n      ✓ 默认: %s\n", modeName(modes[defaultIdx].Priority))
		return modes[defaultIdx].Priority
	}
}

// modeName returns Chinese display name for a priority.
func modeName(priority string) string {
	switch priority {
	case "speed":
		return "速度优先"
	case "balanced":
		return "均衡"
	case "context":
		return "上下文优先"
	default:
		return priority
	}
}
