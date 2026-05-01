package llm

import (
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/MMMchou/ashforge/core/catalog"
	"github.com/MMMchou/ashforge/core/config"
	"github.com/MMMchou/ashforge/core/hw"
)

// RunningEngine represents a running llama-server instance
type RunningEngine struct {
	PID        int
	Port       int
	ModelID    string
	BinaryPath string
	LogPath    string
	CtxSize    int      // 实际使用的上下文大小
	logFile    *os.File // 保持日志文件句柄，进程退出时再关
}

// Start starts llama-server with probe-and-retry strategy.
// 探测式启动：尝试最优 ctx → OOM 就减半重试 → 最多 3 次。
// 注意：iso3 检测已在 main.go [4/6] Preflight 阶段完成，profile.HasIsoQuant 已更新。
func Start(profile *catalog.DeployProfile, binaryPath, modelPath string, sys *hw.System, host string) (*RunningEngine, error) {
	ctxSize := IdealStartCtx(profile, sys)

	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			fmt.Printf("      ⚠️  显存不足，降低上下文至 %dK 重试...\n", ctxSize/1024)
		}

		eng, err := startOnce(profile, binaryPath, modelPath, sys, ctxSize, host)
		if err == nil {
			eng.CtxSize = ctxSize
			return eng, nil
		}

		// 判断是否 OOM（进程启动后很快退出 = 大概率 OOM）
		if !isLikelyOOM(err) {
			return nil, err
		}

		// 清理失败的进程
		Stop()

		// ctx 减半
		ctxSize = ctxSize / 2
		if ctxSize < 4096 {
			ctxSize = 4096
		}

		// 已经是最小了还失败 → 给出详细的 VRAM 分析
		if ctxSize == 4096 && attempt > 0 {
			return nil, buildOOMError(profile, sys, attempt+1)
		}
	}

	return nil, fmt.Errorf("3 次启动均失败，建议选择更小的模型")
}

// StartWithArgs starts llama-server with pre-optimized args from warmup.
func StartWithArgs(profile *catalog.DeployProfile, binaryPath, modelPath string, sys *hw.System, optimizedArgs []string, host string) (*RunningEngine, error) {
	if len(optimizedArgs) == 0 {
		return Start(profile, binaryPath, modelPath, sys, host)
	}

	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}

	actualPort := findFreePort(cfg.LlamaPort)
	if actualPort != cfg.LlamaPort {
		fmt.Printf("Port %d in use, using %d instead\n", cfg.LlamaPort, actualPort)
	}

	// Patch port and host in optimized args
	args := make([]string, len(optimizedArgs))
	copy(args, optimizedArgs)
	for i, a := range args {
		if a == "--port" && i+1 < len(args) {
			args[i+1] = strconv.Itoa(actualPort)
		}
		if a == "--host" && i+1 < len(args) && host != "" {
			args[i+1] = host
		}
	}

	return launchProcess(profile, binaryPath, args, actualPort, sys.ClusterCaps().HasBlackwell)
}

// startOnce 单次启动尝试
func startOnce(profile *catalog.DeployProfile, binaryPath, modelPath string, sys *hw.System, ctxSize int, host string) (*RunningEngine, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}

	actualPort := findFreePort(cfg.LlamaPort)
	if actualPort != cfg.LlamaPort {
		fmt.Printf("Port %d in use, using %d instead\n", cfg.LlamaPort, actualPort)
	}

	args := BuildArgs(profile, binaryPath, modelPath, actualPort, sys, ctxSize, 512, 512)
	if host != "" && host != "127.0.0.1" {
		args = append(args, "--host", host)
	} else {
		args = append(args, "--host", "127.0.0.1")
	}
	return launchProcess(profile, binaryPath, args, actualPort, sys.ClusterCaps().HasBlackwell)
}

// launchProcess 启动 llama-server 进程并等待就绪
func launchProcess(profile *catalog.DeployProfile, binaryPath string, args []string, port int, isBlackwell bool) (*RunningEngine, error) {
	logPath := filepath.Join(config.LogDir(), fmt.Sprintf("llama-server-%d.log", time.Now().Unix()))
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}

	cmd := exec.Command(binaryPath, args...)
	// Remove CUDA_VISIBLE_DEVICES so llama-server discovers all GPUs.
	// Set LD_LIBRARY_PATH to binary's directory so libmtmd.so/libllama.so are found.
	binaryDir := filepath.Dir(binaryPath)
	existingLD := os.Getenv("LD_LIBRARY_PATH")
	ldPath := binaryDir
	if existingLD != "" {
		ldPath = binaryDir + ":" + existingLD
	}
	cmd.Env = append(
		filterEnvVar(os.Environ(), "CUDA_VISIBLE_DEVICES"),
		"LD_LIBRARY_PATH="+ldPath,
	)
	// MoE + multi-GPU: disable CUDA graph capture to avoid memory leak (llama.cpp #20315)
	if isBlackwell || (len(args) > 0 && ContainsFlag(args, "--n-cpu-moe", "--cpu-moe")) {
		cmd.Env = append(cmd.Env, "GGML_CUDA_DISABLE_GRAPHS=1")
	}
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	setProcAttr(cmd)

	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, fmt.Errorf("failed to start llama-server: %w", err)
	}

	// 监控进程退出
	exitCh := make(chan error, 1)
	go func() {
		exitCh <- cmd.Wait()
	}()

	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(cmd.Process.Pid)), 0644); err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("failed to write PID file: %w", err)
	}

	eng := &RunningEngine{
		PID:        cmd.Process.Pid,
		Port:       port,
		ModelID:    profile.ModelID,
		BinaryPath: binaryPath,
		LogPath:    logPath,
		logFile:    logFile,
	}

	fmt.Printf("Waiting for llama-server to be ready (port %d)...\n", port)

	// 等待就绪，同时监控进程是否提前退出（OOM 等）
	// Blackwell (SM120) 首次启动需要 PTX JIT 编译 30-60s，延长超时避免误判 OOM
	timeout := 90 * time.Second
	if isBlackwell {
		timeout = 180 * time.Second
	}
	deadline := time.After(timeout)
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()

	for {
		select {
		case err := <-exitCh:
			// 进程退出了 = 启动失败（大概率 OOM）
			_ = err
			logContent := readLastLines(logPath, 20)
			return nil, fmt.Errorf("llama-server exited during startup:\n%s", logContent)
		case <-deadline:
			Stop()
			if isBlackwell {
				return nil, fmt.Errorf("llama-server startup timeout (%s) — RTX 50系首次启动需JIT编译(~60s)，请重试", timeout)
			}
			return nil, fmt.Errorf("llama-server failed to start within %s", timeout)
		case <-tick.C:
			if isPortReady("127.0.0.1", port) {
				return eng, nil
			}
		}
	}
}

// isLikelyOOM 判断启动失败是否可能是 OOM
func isLikelyOOM(err error) bool {
	msg := err.Error()
	// Parameter errors: binary doesn't support the flag, not OOM
	if strings.Contains(msg, "invalid value") ||
		strings.Contains(msg, "error while handling argument") ||
		strings.Contains(msg, "unknown argument") ||
		strings.Contains(msg, "unrecognized option") {
		return false
	}
	// Timeout: not OOM (JIT compilation, slow load, etc.)
	if strings.Contains(msg, "startup timeout") {
		return false
	}
	// Missing shared library: not OOM, binary can't load dependencies
	if strings.Contains(msg, "error while loading shared libraries") ||
		strings.Contains(msg, "cannot open shared object file") {
		return false
	}
	return strings.Contains(msg, "exited during startup") ||
		strings.Contains(msg, "CUDA out of memory") ||
		strings.Contains(msg, "ggml_cuda") ||
		strings.Contains(msg, "not enough memory") ||
		strings.Contains(msg, "alloc")
}

// readLastLines 读取文件最后 n 行
func readLastLines(path string, n int) string {
	f, err := os.Open(path)
	if err != nil {
		return "(无法读取日志)"
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return "(无法读取日志)"
	}

	// Read last 8KB (enough for ~20 lines)
	size := stat.Size()
	readSize := int64(8192)
	if readSize > size {
		readSize = size
	}

	buf := make([]byte, readSize)
	if _, err := f.ReadAt(buf, size-readSize); err != nil && err != io.EOF {
		return "(无法读取日志)"
	}

	lines := strings.Split(string(buf), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// CleanOldLogs removes log files older than maxAge from the logs directory.
func CleanOldLogs(maxAge time.Duration) {
	logDir := config.LogDir()
	entries, err := os.ReadDir(logDir)
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-maxAge)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".log") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			os.Remove(filepath.Join(logDir, e.Name()))
		}
	}
}

// filterEnvVar removes a specific environment variable from the env slice.
// Used to strip CUDA_VISIBLE_DEVICES so llama-server discovers all GPUs.
func filterEnvVar(env []string, key string) []string {
	prefix := key + "="
	result := make([]string, 0, len(env))
	for _, e := range env {
		if !strings.HasPrefix(e, prefix) {
			result = append(result, e)
		}
	}
	return result
}

// isPortReady 检查端口是否可连接
func isPortReady(host string, port int) bool {
	addr := fmt.Sprintf("%s:%d", host, port)
	conn, err := (&net.Dialer{Timeout: 500 * time.Millisecond}).Dial("tcp", addr)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// Stop stops the running llama-server
func Stop() error {
	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	data, err := os.ReadFile(pidPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no running model found")
		}
		return fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return fmt.Errorf("invalid PID in file: %w", err)
	}

	if err := killProcess(pid); err != nil {
		return err
	}

	os.Remove(pidPath)
	return nil
}

// Status returns the status of the running engine
func Status() (*RunningEngine, error) {
	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	data, err := os.ReadFile(pidPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return nil, fmt.Errorf("invalid PID in file: %w", err)
	}

	if !isProcessAlive(pid) {
		os.Remove(pidPath)
		return nil, nil
	}

	cfg, err := config.Load()
	if err != nil {
		return &RunningEngine{PID: pid, Port: 0}, nil
	}
	return &RunningEngine{
		PID:  pid,
		Port: cfg.LlamaPort,
	}, nil
}

// buildOOMError 构建详细的 OOM 错误信息，建议根据实际情况动态生成
func buildOOMError(profile *catalog.DeployProfile, sys *hw.System, attempts int) error {
	vramMB := sys.TotalVRAM_MB()
	modelMB := int(profile.Size_GB * 1024)
	kvMB := profile.EstimateKVCacheMB(4096, "q4_0") * 2 // 最小 KV cache
	needMB := modelMB + kvMB + 1024                      // 模型 + KV + overhead

	gpu := sys.PrimaryGPU()
	gpuName := "GPU"
	if gpu != nil {
		gpuName = gpu.Name
	}

	msg := fmt.Sprintf("连续 %d 次启动失败，即使最小上下文(4K)也无法运行\n\n", attempts)
	msg += fmt.Sprintf("  %s: %d MB VRAM\n", gpuName, vramMB)
	msg += fmt.Sprintf("  模型 %s: ~%d MB\n", profile.DisplayName, modelMB)
	msg += fmt.Sprintf("  KV cache (4K, q4_0): ~%d MB\n", kvMB)
	msg += fmt.Sprintf("  预估总需: ~%d MB\n\n", needMB)

	if needMB > vramMB {
		msg += fmt.Sprintf("  差额: %d MB\n\n", needMB-vramMB)
	}

	msg += "  建议:\n"
	vramGB := float64(vramMB) / 1024.0

	if profile.Size_GB > vramGB*0.7 {
		// 模型本身偏大
		msg += "  1. 选择更小的量化 (Q4_K_M 或 Q2_K)\n"
		if profile.Arch != "moe" && profile.Size_GB > 10 {
			// dense 大模型，建议换 MoE 架构
			msg += "  2. 换用 MoE 架构模型（如 Qwen3-30B-A3B），expert 层自动放 CPU RAM\n"
		} else {
			msg += "  2. 选择更小的模型\n"
		}
	} else {
		// 模型小但还是 OOM → 参数配置问题
		msg += "  1. 运行 ashforge run " + profile.ModelID + " --reset 重新探测参数\n"
		msg += "  2. 模型较小但仍 OOM，可能是参数配置问题，请升级到最新版本\n"
	}

	return fmt.Errorf("%s", msg)
}
