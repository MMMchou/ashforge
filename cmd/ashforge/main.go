package main

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/fatih/color"
	"github.com/spf13/cobra"

	"github.com/MMMchou/ashforge/internal/config"
	"github.com/MMMchou/ashforge/internal/engine"
	"github.com/MMMchou/ashforge/internal/hardware"
	"github.com/MMMchou/ashforge/internal/ide"
	"github.com/MMMchou/ashforge/internal/model"
	"github.com/MMMchou/ashforge/internal/monitor"
	"github.com/MMMchou/ashforge/internal/optimizer"
	"github.com/MMMchou/ashforge/internal/proxy"
)

var version = "dev"

func main() {
	rootCmd := &cobra.Command{
		Use:   "ashforge",
		Short: "Ashforge — deploy local LLMs faster than LM Studio / Ollama",
		Long:  "Ashforge is a CLI tool that automatically optimizes local LLM deployment.\nSame model, faster speed, zero manual tuning.",
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			return config.EnsureConfigDir()
		},
		Run: func(cmd *cobra.Command, args []string) {
			showMainMenu()
		},
	}

	rootCmd.AddCommand(
		newRunCmd(),
		newStopCmd(),
		newStatusCmd(),
		newProbeCmd(),
		newInjectCmd(),
		newBenchCmd(),
		newListCmd(),
		newConfigCmd(),
		newCacheCmd(),
		newVersionCmd(),
	)

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print Ashforge version",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Ashforge %s\n", version)
		},
	}
}

func newRunCmd() *cobra.Command {
	var fast bool
	var bench bool
	var ctxSize int
	var reset bool
	var llamaServer string
	var host string
	var mode string
	cmd := &cobra.Command{
		Use:   "run <model>",
		Short: "Deploy and start a model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runModel(args[0], fast, bench, ctxSize, reset, llamaServer, host, mode)
		},
	}
	cmd.Flags().BoolVar(&fast, "fast", false, "Skip warmup, use cached profile")
	cmd.Flags().BoolVar(&bench, "bench", false, "Run benchmark after starting")
	cmd.Flags().BoolVar(&reset, "reset", false, "清除缓存，重新 warmup 探测最优参数")
	// 微调模式：手动指定上下文大小（覆盖自动计算）
	// 建议值：4096, 8192, 16384, 32768, 65536, 131072
	// 越大上下文越长但越慢，越小速度越快但上下文短
	// 0 = 自动模式（根据 VRAM 和模型大小动态计算最优值）
	cmd.Flags().IntVar(&ctxSize, "ctx-size", 0, "手动指定上下文大小（0=自动）")
	cmd.Flags().StringVar(&llamaServer, "llama-server", "", "使用自定义 llama-server 二进制（完整路径）")
	cmd.Flags().StringVar(&host, "host", "127.0.0.1", "监听地址（默认 127.0.0.1，用 0.0.0.0 开放局域网）")
	cmd.Flags().StringVar(&mode, "mode", "", "模式选择: speed/balanced/context（默认用上次选择）")
	return cmd
}

func newStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stop",
		Short: "Stop the running model",
		RunE: func(cmd *cobra.Command, args []string) error {
			return stopModel()
		},
	}
}

func newStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Show running model status",
		RunE: func(cmd *cobra.Command, args []string) error {
			return showStatus()
		},
	}
}

func newProbeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "probe",
		Short: "Output hardware fingerprint",
		RunE: func(cmd *cobra.Command, args []string) error {
			return probeHardware()
		},
	}
}

func newInjectCmd() *cobra.Command {
	var ide string
	var undo bool
	cmd := &cobra.Command{
		Use:   "inject",
		Short: "Inject IDE configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			return injectIDE(ide, undo)
		},
	}
	cmd.Flags().StringVar(&ide, "ide", "", "Target IDE: all, cc, codex, cursor")
	cmd.Flags().BoolVar(&undo, "undo", false, "Restore original IDE configs")
	return cmd
}

func newBenchCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "bench",
		Short: "Benchmark the running model",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runBench()
		},
	}
}

func newListCmd() *cobra.Command {
	var installed bool
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List supported models",
		RunE: func(cmd *cobra.Command, args []string) error {
			return listModels(installed)
		},
	}
	cmd.Flags().BoolVar(&installed, "installed", false, "Only show downloaded models")
	return cmd
}

func newCacheCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cache",
		Short: "Manage warmup cache",
	}
	cmd.AddCommand(&cobra.Command{
		Use:   "clear [model]",
		Short: "清除 warmup 缓存（不指定模型则清除全部）",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return clearCache(args)
		},
	})
	return cmd
}

func newConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Manage configuration",
	}
	cmd.AddCommand(&cobra.Command{
		Use:   "show",
		Short: "Show current configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			return showConfig()
		},
	})
	cmd.AddCommand(&cobra.Command{
		Use:   "set <key=value>",
		Short: "Set a configuration value",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return setConfig(args[0])
		},
	})
	return cmd
}

// ── TUI helpers ──────────────────────────────────────────────────────

var dim = color.New(color.FgHiBlack) // 灰色，用于次要信息和导流文字

// displayWidth 计算字符串在终端的显示宽度（中文占2列，ASCII占1列）
func displayWidth(s string) int {
	width := 0
	for _, r := range s {
		if r < 128 {
			width++ // ASCII 字符占1列
		} else {
			width += 2 // 中文等宽字符占2列
		}
	}
	return width
}

func printLogo() {
	blue := color.New(color.FgBlue)
	blue.Print(`
 █████╗ ███████╗██╗  ██╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔══██╗██╔════╝██║  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
███████║███████╗███████║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██╔══██║╚════██║██╔══██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██║  ██║███████║██║  ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
`)
	fmt.Printf("Auto-tuned LLM Engine v%s · llama.cpp b8864\n", version)
	dim.Println("by Ashan")
}

// modelDirDisplay returns the display path for the model directory
func modelDirDisplay() string {
	if runtime.GOOS == "windows" {
		return config.ModelDir() // Windows: 展开完整路径
	}
	return "~/.ashforge/models/"
}

func showMainMenu() {
	printLogo()

	// ── 命令列表 ──
	fmt.Println()
	dim.Println("── 命令 ──────────────────────────────────────────")
	cmdBlue := color.New(color.FgBlue)
	tag := color.New(color.FgBlue)

	cmds := []struct{ name, desc, extra string }{
		{"ashforge run <model>", "启动模型服务，自动调参", "[推荐]"},
		{"ashforge list", "列出已下载的模型", ""},
		{"ashforge bench <model>", "重新测速并更新参数缓存", ""},
		{"ashforge inject", "注入 IDE 配置（CC/Cursor/Codex）", ""},
		{"ashforge stop", "停止当前运行的服务", ""},
		{"ashforge status", "查看当前服务状态", ""},
	}
	for _, c := range cmds {
		cmdBlue.Printf("  %-25s", c.name)
		dim.Printf(" %s", c.desc)
		if c.extra != "" {
			tag.Printf("  %s", c.extra)
		}
		fmt.Println()
	}

	// ── 已下载模型 ──
	fmt.Println()
	dim.Println("── 本地模型 ────────────────────────────────────────")
	db, err := model.LoadStore()
	if err == nil {
		green := color.New(color.FgGreen)
		for _, m := range db.ListAll() {
			green.Print("  ● ")
			fmt.Printf("%s", m.DisplayName)
			if len(m.Quantizations) > 0 {
				q := m.Quantizations[0]
				dim.Printf("  %s · %.1f GB", q.ID, q.Size_GB)
			}
			if m.Arch != "" {
				dim.Printf(" · %s", m.Arch)
			}
			fmt.Println()
		}
	}

	// 模型文件夹路径 + 换目录提示
	fmt.Println()
	dim.Printf("  模型文件夹：%s\n", modelDirDisplay())
	dim.Println("  将 .gguf 文件放入此目录，Ashforge 自动识别")
	// 提示用户如何更换模型文件夹，避免 C 盘空间不足
	// 运行: ashforge config set model_dir=D:\models
	// 即可将模型存储迁移到其他盘，之后手动移动已有 .gguf 文件即可
	dim.Println("  如需更换文件夹: ashforge config set model_dir=D:\\your\\path")

	fmt.Println()
	fmt.Print("  ")
	color.New(color.FgGreen).Print("→")
	fmt.Print(" 运行 ")
	color.New(color.FgBlue).Print("ashforge run qwen3-30b")
	fmt.Println(" 开始使用")

	dim.Println("  → github.com/MMMchou/ashforge")
	fmt.Println()
}

func runModel(modelName string, fast, bench bool, ctxSize int, reset bool, llamaServer string, host string, mode string) error {
	printLogo()
	fmt.Println()

	// [1/5] Probe hardware
	fmt.Printf("[1/6] Probing hardware...\n")
	hw, err := hardware.Probe()
	if err != nil {
		return fmt.Errorf("hardware probe failed: %w", err)
	}
	gpu := hw.PrimaryGPU()
	if gpu != nil {
		if hw.GPUCount() > 1 {
			fmt.Printf("      GPU: %d cards, %d MB total VRAM\n", hw.GPUCount(), hw.TotalVRAM_MB())
			for i, g := range hw.GPUs {
				fmt.Printf("        #%d %s (SM%s, %d MB, %.0f GB/s)\n",
					i, g.Name, strings.ReplaceAll(g.ComputeCap, ".", ""), g.VRAM_MB, g.MemBandwidth_GBs)
			}
			if ts := hw.TensorSplitArg(); ts != "" {
				fmt.Printf("      Split: %s (VRAM×BW weighted)\n", ts)
			}
		} else {
			fmt.Printf("      GPU: %s (SM%s, %d MB VRAM, %.0f GB/s)\n",
				gpu.Name, strings.ReplaceAll(gpu.ComputeCap, ".", ""), gpu.VRAM_MB, gpu.MemBandwidth_GBs)
		}
	}
	fmt.Printf("      RAM: %d GB %s\n", hw.RAM.Total_MB/1024, strings.ToUpper(hw.RAM.Type))
	fmt.Printf("      OS:  %s %s\n", hw.OS.Platform, hw.OS.Arch)

	// Validate CUDA version for Blackwell
	if err := engine.ValidateCUDAVersion(hw); err != nil {
		return err
	}

	// [2/5] Select configuration
	fmt.Printf("\n[2/6] Selecting configuration...\n")
	db, err := model.LoadStore()
	if err != nil {
		return fmt.Errorf("failed to load model database: %w", err)
	}
	modelDef, err := db.GetOrDetect(modelName)
	if err != nil {
		return err
	}
	profile, err := model.Match(modelDef, hw)
	if err != nil {
		return err
	}
	// 直接路径模式：绝对路径传入时跳过下载
	if strings.HasSuffix(strings.ToLower(modelName), ".gguf") {
		if absPath, err := filepath.Abs(modelName); err == nil {
			if info, err := os.Stat(absPath); err == nil && !info.IsDir() {
				profile.LocalPath = absPath
			}
		}
	}
	// 微调模式：用户手动指定 ctx
	if ctxSize > 0 {
		profile.CtxOverride = ctxSize
	}
	fmt.Printf("      Model:  %s (%s, %.0fB", profile.DisplayName, profile.Arch, modelDef.TotalParams_B)
	if modelDef.IsMoE() {
		fmt.Printf(" total / %.0fB active", modelDef.ActiveParams_B)
	}
	fmt.Printf(")\n")
	fmt.Printf("      Quant:  %s (%.1f GB)\n", profile.Quant, profile.Size_GB)
	fmt.Printf("      Mode:   %s", profile.Mode)
	if profile.Mode == "moe_offload" {
		fmt.Printf(" (experts on CPU)")
	}
	fmt.Printf("\n")
	accel := []string{}
	if hw.SupportsFlashAttn() {
		accel = append(accel, "Flash Attention")
	}
	if profile.NativeMTP {
		accel = append(accel, "MTP (native)")
	}
	if hw.GPUCount() > 1 && hw.HasNVLink() {
		accel = append(accel, "NVLink")
	} else if hw.GPUCount() > 1 {
		accel = append(accel, fmt.Sprintf("Tensor Split (%s)", hw.TensorSplitArg()))
	}
	if profile.IsHybrid {
		accel = append(accel, "SWA-Full (hybrid arch)")
	}
	if len(accel) > 0 {
		fmt.Printf("      Accel:  %s\n", strings.Join(accel, " + "))
	}

	// [3/5] Check files
	fmt.Printf("\n[3/6] Checking files...\n")
	var binaryPath string
	var isTurboQuant bool
	if llamaServer != "" {
		if _, err := os.Stat(llamaServer); err != nil {
			return fmt.Errorf("指定的 llama-server 不存在: %s", llamaServer)
		}
		binaryPath = llamaServer
		fmt.Printf("      Binary: %s [user-specified]\n", filepath.Base(binaryPath))
	} else {
		var err error
		binaryPath, isTurboQuant, err = engine.EnsureBinary(hw)
		if err != nil {
			return fmt.Errorf("failed to ensure binary: %w", err)
		}
		fmt.Printf("      Binary: %s [cached]\n", filepath.Base(binaryPath))
	}
	engine.VerifyBackend(binaryPath, hw)

	modelPath, err := model.EnsureFile(profile)
	if err != nil {
		return fmt.Errorf("failed to ensure model file: %w", err)
	}
	fmt.Printf("      Model:  %s [cached]\n", filepath.Base(modelPath))

	// [4/6] OOM preflight check
	fmt.Printf("\n[4/6] Preflight check...\n")
	// iso3: static check (bundled turboquant binary + all GPUs SM>=80)
	caps := hw.ClusterCaps()
	if profile.HasIsoQuant && !engine.ShouldUseIso3(isTurboQuant, caps.MinSM) {
		fmt.Printf("      iso3 不可用（MinSM%d 或非 turboquant binary），回退到 q8_0/q4_0\n", caps.MinSM)
		profile.HasIsoQuant = false
	}
	if err := engine.PreflightCheck(profile, hw); err != nil {
		return err
	}
	fmt.Printf("      ✓ VRAM sufficient\n")

	// [5/6] Warmup benchmark
	fmt.Printf("\n[5/6] Warmup benchmark...\n")
	if reset {
		if err := optimizer.ClearProfileCache(profile.ModelID, hw); err != nil {
			fmt.Printf("      ⚠️  清除缓存失败: %v\n", err)
		} else {
			fmt.Printf("      已清除缓存，重新探测\n")
		}
	}
	optimized, err := optimizer.Warmup(profile, binaryPath, modelPath, hw, fast, mode)
	if err != nil {
		fmt.Printf("      ⚠️  Warmup failed: %v\n", err)
		fmt.Printf("      Using default parameters\n")
	} else {
		fmt.Printf("      ✓ %.1f tok/s\n", optimized.MeasuredTPS)
	}

	// [6/6] Start server + proxy
	fmt.Printf("\n[6/6] Starting server...\n")
	cfg, _ := config.Load()

	// Use warmup-optimized args if available
	var optimizedArgs []string
	if optimized != nil {
		optimizedArgs = optimized.LaunchArgs
	}
	eng, err := engine.StartWithArgs(profile, binaryPath, modelPath, hw, optimizedArgs, host)
	if err != nil {
		return fmt.Errorf("failed to start llama-server: %w", err)
	}
	fmt.Printf("      llama-server started (PID %d, port %d)\n", eng.PID, eng.Port)

	// Start proxy
	proxyServer := proxy.NewServer(cfg.ProxyPort, eng.Port, profile.ModelID, host)
	proxyServer.StartAsync()
	fmt.Printf("      Ashforge proxy started (port %d)\n", cfg.ProxyPort)

	// Start monitor
	mon := monitor.NewMonitor(eng.Port, profile.DisplayName)
	if optimized != nil {
		mon.ParamInfo = buildParamSummary(optimized.LaunchArgs)
	}
	mon.StartAsync()

	tpsStr := ""
	if optimized != nil {
		tpsStr = fmt.Sprintf(" @ %.1f tok/s", optimized.MeasuredTPS)
	}

	// Ready 框
	fmt.Println()
	boxWidth := 51 // 框内容宽度（不含边框）
	fmt.Println("  ┌─────────────────────────────────────────────────┐")

	// 第一行：Ready 状态
	readyText := fmt.Sprintf("Ready — %s%s", profile.DisplayName, tpsStr)
	readyWidth := displayWidth(readyText)
	pad := boxWidth - readyWidth
	if pad < 1 {
		pad = 1
	}
	color.Green("  │  %s%s│\n", readyText, strings.Repeat(" ", pad))

	// 第二行：API 地址
	apiHost := host
	if apiHost == "" || apiHost == "0.0.0.0" {
		apiHost = "127.0.0.1" // 显示时用 127.0.0.1，但实际监听 0.0.0.0
	}
	apiText := fmt.Sprintf("API: http://%s:%d/v1/chat/completions", apiHost, cfg.ProxyPort)
	apiWidth := displayWidth(apiText)
	pad2 := boxWidth - apiWidth
	if pad2 < 1 {
		pad2 = 1
	}
	fmt.Printf("  │  %s%s│\n", apiText, strings.Repeat(" ", pad2))

	// 第三行：模型文件夹
	modelDirText := fmt.Sprintf("模型文件夹: %s", modelDirDisplay())
	modelDirWidth := displayWidth(modelDirText)
	pad3 := boxWidth - modelDirWidth
	if pad3 < 1 {
		pad3 = 1
	}
	dim.Printf("  │  %s%s│\n", modelDirText, strings.Repeat(" ", pad3))

	fmt.Println("  └─────────────────────────────────────────────────┘")

	fmt.Println()
	fmt.Printf("  运行 ")
	color.New(color.FgBlue).Print("ashforge inject")
	fmt.Println(" 接入 IDE · Ctrl+C 停止")
	fmt.Println()

	// Block until interrupted
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	<-sigCh

	fmt.Println("\n正在停止服务...")
	mon.Stop()
	proxyServer.Stop()
	engine.Stop()
	color.Green("✓ llama-server 已停止\n")
	color.Green("✓ Ashforge proxy 已停止\n")
	fmt.Println()
	dim.Println("Thanks for using Ashforge · github.com/MMMchou/ashforge")

	return nil
}

// buildParamSummary extracts key params from launch args into a human-readable string.
func buildParamSummary(args []string) string {
	var parts []string
	for i, a := range args {
		if i+1 >= len(args) {
			break
		}
		v := args[i+1]
		switch a {
		case "--ctx-size":
			if n, err := strconv.Atoi(v); err == nil {
				parts = append(parts, fmt.Sprintf("%dK ctx", n/1024))
			}
		case "-ctk":
			parts = append(parts, "KV:"+v)
		case "--ubatch-size":
			parts = append(parts, "ub"+v)
		case "--cache-reuse":
			parts = append(parts, "reuse:"+v)
		}
	}
	// Check for flags without values
	for _, a := range args {
		switch a {
		case "--mlock":
			parts = append(parts, "mlock")
		case "--mmap":
			parts = append(parts, "mmap")
		}
	}
	return strings.Join(parts, " · ")
}

func stopModel() error {
	fmt.Println("Stopping model...")
	if err := engine.Stop(); err != nil {
		return err
	}
	color.Green("✓ Model stopped\n")
	return nil
}

func showStatus() error {
	eng, err := engine.Status()
	if err != nil {
		return err
	}
	if eng == nil {
		fmt.Println("No model running.")
		return nil
	}
	fmt.Printf("Model running:\n")
	fmt.Printf("  PID:  %d\n", eng.PID)
	fmt.Printf("  Port: %d\n", eng.Port)
	return nil
}

func probeHardware() error {
	hw, err := hardware.Probe()
	if err != nil {
		return fmt.Errorf("hardware probe failed: %w", err)
	}

	jsonStr, err := hw.JSON()
	if err != nil {
		return err
	}

	fmt.Println(jsonStr)
	fmt.Printf("\nFingerprint: %s\n", hw.Fingerprint())
	return nil
}

func injectIDE(ideName string, undo bool) error {
	cfg, _ := config.Load()
	apiKey := "af-demo-key"

	if undo {
		ides := ide.Detect()
		for _, i := range ides {
			if !i.Detected {
				continue
			}
			if ideName != "" && ideName != "all" {
				if !matchesIDEName(i.Name, ideName) {
					continue
				}
			}
			fmt.Printf("Restoring %s...\n", i.Name)
			if err := ide.Undo(&i); err != nil {
				fmt.Printf("  ✗ Failed: %v\n", err)
			} else {
				color.Green("  ✓ Restored\n")
			}
		}
		return nil
	}

	fmt.Println("\nDetecting installed IDEs...")
	ides := ide.Detect()
	detected := []ide.IDE{}
	for _, i := range ides {
		if i.Detected {
			detected = append(detected, i)
			color.Green("  ✓ %s\n", i.Name)
		} else {
			fmt.Printf("  ✗ %s (not installed)\n", i.Name)
		}
	}

	if len(detected) == 0 {
		fmt.Println("\nNo supported IDEs detected.")
		return nil
	}

	fmt.Println("\nInjecting configuration...")
	for _, i := range detected {
		if ideName != "" && ideName != "all" {
			if !matchesIDEName(i.Name, ideName) {
				continue
			}
		}
		fmt.Printf("  %s → ", i.Name)
		if err := ide.Inject(&i, cfg.ProxyPort, apiKey); err != nil {
			color.Red("Failed: %v\n", err)
		} else {
			color.Green("✓\n")
		}
	}

	fmt.Println("\nRestart your IDE to apply changes.")
	fmt.Printf("Model endpoint: http://127.0.0.1:%d\n", cfg.ProxyPort)
	return nil
}

func matchesIDEName(fullName, shortName string) bool {
	shortName = strings.ToLower(shortName)
	fullName = strings.ToLower(fullName)
	switch shortName {
	case "cc", "claude", "claude-code":
		return strings.Contains(fullName, "claude")
	case "codex":
		return strings.Contains(fullName, "codex")
	case "cursor":
		return strings.Contains(fullName, "cursor")
	default:
		return false
	}
}

func runBench() error {
	fmt.Println("Benchmark (placeholder)")
	return nil
}

func listModels(installed bool) error {
	db, err := model.LoadStore()
	if err != nil {
		return err
	}

	dim.Println("\n── 本地模型 ────────────────────────────────────────")
	fmt.Println()

	for _, m := range db.ListAll() {
		color.New(color.FgGreen).Print("  ● ")
		fmt.Printf("%s\n", color.CyanString(m.DisplayName))
		fmt.Printf("    ID:     %s\n", m.ID)
		fmt.Printf("    Arch:   %s", m.Arch)
		if m.IsMoE() {
			fmt.Printf(" (%.0fB total / %.0fB active)\n", m.TotalParams_B, m.ActiveParams_B)
		} else {
			fmt.Printf(" (%.0fB params)\n", m.TotalParams_B)
		}
		fmt.Printf("    Quants: ")
		for i, q := range m.Quantizations {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%s (%.1fGB)", q.ID, q.Size_GB)
		}
		fmt.Println()
		fmt.Println()
	}

	dim.Printf("  模型文件夹：%s\n", modelDirDisplay())
	dim.Println("  将 .gguf 文件放入此目录，Ashforge 自动识别")
	dim.Println("  如需更换文件夹: ashforge config set model_dir=D:\\your\\path")
	fmt.Println()
	dim.Println("  → github.com/MMMchou/ashforge")
	fmt.Println()

	return nil
}

func clearCache(args []string) error {
	profileDir := config.ProfileDir()

	if len(args) == 0 {
		// 清除全部缓存
		entries, err := os.ReadDir(profileDir)
		if err != nil {
			if os.IsNotExist(err) {
				fmt.Println("没有缓存可清除")
				return nil
			}
			return err
		}
		count := 0
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".json") {
				os.Remove(filepath.Join(profileDir, e.Name()))
				count++
			}
		}
		if count == 0 {
			fmt.Println("没有缓存可清除")
		} else {
			color.Green("✓ 已清除 %d 个模型缓存\n", count)
		}
		return nil
	}

	// 清除指定模型的缓存
	modelName := strings.ToLower(args[0])
	entries, err := os.ReadDir(profileDir)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println("没有缓存可清除")
			return nil
		}
		return err
	}
	count := 0
	for _, e := range entries {
		if strings.Contains(strings.ToLower(e.Name()), modelName) {
			os.Remove(filepath.Join(profileDir, e.Name()))
			count++
		}
	}
	if count == 0 {
		fmt.Printf("未找到模型 '%s' 的缓存\n", args[0])
	} else {
		color.Green("✓ 已清除 %d 个缓存文件\n", count)
	}
	return nil
}

func showConfig() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	fmt.Printf("HF Mirror:   %s\n", cfg.HFMirror)
	fmt.Printf("Llama Port:  %d\n", cfg.LlamaPort)
	fmt.Printf("Proxy Port:  %d\n", cfg.ProxyPort)
	fmt.Printf("Model Dir:   %s\n", config.ModelDir())
	fmt.Printf("Config Dir:  %s\n", config.Dir())
	return nil
}

func setConfig(kv string) error {
	parts := strings.SplitN(kv, "=", 2)
	if len(parts) != 2 {
		return fmt.Errorf("invalid format, use: key=value")
	}
	key := strings.TrimSpace(parts[0])
	value := strings.TrimSpace(parts[1])

	cfg, err := config.Load()
	if err != nil {
		return err
	}

	switch key {
	case "hf_mirror":
		cfg.HFMirror = value
	case "llama_port":
		port, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("invalid port number: %v", err)
		}
		cfg.LlamaPort = port
	case "proxy_port":
		port, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("invalid port number: %v", err)
		}
		cfg.ProxyPort = port
	case "log_level":
		cfg.LogLevel = value
	case "model_dir":
		// 空值表示重置为默认
		if value == "" {
			cfg.ModelDirOverride = ""
			color.Green("✓ 模型目录已重置为默认: %s\n", filepath.Join(config.Dir(), "models"))
		} else {
			// 验证路径是否存在或可创建
			if err := os.MkdirAll(value, 0755); err != nil {
				return fmt.Errorf("无法创建模型目录 %s: %v", value, err)
			}
			cfg.ModelDirOverride = value
			color.Green("✓ 模型目录已设置为: %s\n", value)
			dim.Println("  提示：请手动将现有 .gguf 文件移动到新目录")
		}
	default:
		return fmt.Errorf("unknown config key: %s (available: hf_mirror, llama_port, proxy_port, log_level, model_dir)", key)
	}

	if err := config.Save(cfg); err != nil {
		return fmt.Errorf("failed to save config: %v", err)
	}

	if key != "model_dir" {
		color.Green("✓ Config updated: %s = %s\n", key, value)
	}
	return nil
}
