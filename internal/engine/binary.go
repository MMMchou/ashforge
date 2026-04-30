package engine

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/MMMchou/ashforge/internal/config"
	"github.com/MMMchou/ashforge/internal/download"
	"github.com/MMMchou/ashforge/internal/hardware"
)

const releaseTag = "b8864"

// selectBinary returns the local binary name based on hardware
func selectBinary(hw *hardware.HardwareProbe) string {
	gpu := hw.PrimaryGPU()
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}

	if gpu == nil {
		return "llama-server" + ext
	}

	// Apple Silicon Metal
	if runtime.GOOS == "darwin" && strings.Contains(gpu.Name, "Metal") {
		return "llama-server" + ext // macOS Metal build is the default binary
	}

	// GPU detected via nvidia-smi → CUDA; otherwise Vulkan
	if gpu.ComputeCap != "" {
		return "llama-server-cuda" + ext
	}

	return "llama-server-vulkan" + ext
}

// downloadURL returns the correct release asset URL
func downloadURL(hw *hardware.HardwareProbe) string {
	base := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s", releaseTag)
	gpu := hw.PrimaryGPU()

	isNVIDIA := gpu != nil && gpu.ComputeCap != ""

	if runtime.GOOS == "windows" {
		if isNVIDIA {
			return fmt.Sprintf("%s/llama-%s-bin-win-cuda-12.4-x64.zip", base, releaseTag)
		}
		if gpu != nil {
			return fmt.Sprintf("%s/llama-%s-bin-win-vulkan-x64.zip", base, releaseTag)
		}
		return fmt.Sprintf("%s/llama-%s-bin-win-cpu-x64.zip", base, releaseTag)
	}

	// macOS (Metal backend for Apple Silicon, CPU for Intel)
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return fmt.Sprintf("%s/llama-%s-bin-macos-arm64.zip", base, releaseTag)
		}
		return fmt.Sprintf("%s/llama-%s-bin-macos-x64.zip", base, releaseTag)
	}

	// Linux
	if isNVIDIA {
		return fmt.Sprintf("%s/llama-%s-bin-ubuntu-x64.tar.gz", base, releaseTag)
	}
	if gpu != nil {
		return fmt.Sprintf("%s/llama-%s-bin-ubuntu-vulkan-x64.tar.gz", base, releaseTag)
	}
	return fmt.Sprintf("%s/llama-%s-bin-ubuntu-x64.tar.gz", base, releaseTag)
}

// EnsureBinary ensures the correct llama-server binary is available.
// Priority: bundled iso3 binary > cached binary > download official release.
// Returns (path, isTurboQuant, error). isTurboQuant=true means the binary supports iso3.
func EnsureBinary(hw *hardware.HardwareProbe) (string, bool, error) {
	binaryName := selectBinary(hw)

	// 1. Check for bundled iso3 binary (shipped with ashforge release)
	bundledPath := findBundledBinary(binaryName)
	if bundledPath != "" {
		fmt.Printf("      Using bundled iso3 binary: %s\n", filepath.Base(bundledPath))
		return bundledPath, true, nil
	}

	// 2. Check cached binary in ~/.ashforge/bin/
	binaryPath := filepath.Join(config.BinDir(), binaryName)
	if _, err := os.Stat(binaryPath); err == nil {
		return binaryPath, false, nil
	}

	// 3. Download official release as fallback
	url := downloadURL(hw)
	fmt.Printf("      Downloading: %s\n", filepath.Base(url))

	archivePath := filepath.Join(config.BinDir(), filepath.Base(url))
	if err := download.DownloadFile(url, archivePath, true); err != nil {
		return "", false, fmt.Errorf("download failed: %w", err)
	}

	fmt.Printf("      Extracting llama-server...\n")
	if err := extractLlamaServer(archivePath, binaryPath); err != nil {
		os.Remove(archivePath)
		return "", false, fmt.Errorf("extraction failed: %w", err)
	}

	os.Remove(archivePath)

	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		os.Chmod(binaryPath, 0755)
	}

	return binaryPath, false, nil
}

// VerifyBackend checks that the binary actually supports the expected backend.
// Runs llama-server --version and checks output for "CUDA" when NVIDIA GPU is present.
func VerifyBackend(binaryPath string, hw *hardware.HardwareProbe) {
	gpu := hw.PrimaryGPU()
	if gpu == nil {
		return
	}

	isNVIDIA := gpu.ComputeCap != ""
	out, err := exec.Command(binaryPath, "--version").CombinedOutput()
	if err != nil {
		return // can't verify, proceed anyway
	}

	output := strings.ToLower(string(out))
	hasCUDA := strings.Contains(output, "cuda")

	if isNVIDIA && !hasCUDA {
		fmt.Println("      Warning: NVIDIA GPU detected but binary lacks CUDA support")
		fmt.Println("      Performance may be degraded. Consider re-downloading with: ashforge update")
	}
}

// findBundledBinary looks for a bundled llama-server shipped alongside ashforge.
// Searches: same directory as ashforge executable (resolving symlinks), ~/.ashforge/bin/, then ./bin/.
func findBundledBinary(binaryName string) string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}

	// Resolve symlinks (install.sh creates symlink in /usr/local/bin → ~/.ashforge/bin/)
	realExe, err := filepath.EvalSymlinks(exe)
	if err != nil {
		realExe = exe
	}
	realDir := filepath.Dir(realExe)
	exeDir := filepath.Dir(exe)

	candidates := []string{
		filepath.Join(realDir, binaryName),
		filepath.Join(config.BinDir(), binaryName),
		filepath.Join(exeDir, binaryName),
		filepath.Join(exeDir, "bin", binaryName),
	}

	seen := make(map[string]bool)
	for _, path := range candidates {
		if seen[path] {
			continue
		}
		seen[path] = true
		if info, err := os.Stat(path); err == nil && !info.IsDir() && info.Size() > 1024*1024 {
			return path
		}
	}
	return ""
}

// extractLlamaServer extracts llama-server binary from zip or tar.gz
func extractLlamaServer(archivePath, destPath string) error {
	if strings.HasSuffix(archivePath, ".zip") {
		return extractFromZip(archivePath, destPath)
	}
	if strings.HasSuffix(archivePath, ".tar.gz") || strings.HasSuffix(archivePath, ".tgz") {
		return extractFromTarGz(archivePath, destPath)
	}
	return fmt.Errorf("unsupported archive format: %s", archivePath)
}

func extractFromZip(archivePath, destPath string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	target := "llama-server"
	if runtime.GOOS == "windows" {
		target = "llama-server.exe"
	}

	// Also extract CUDA runtime DLLs if present
	dllsToExtract := map[string]bool{
		"cublas64_12.dll":    true,
		"cublasLt64_12.dll":  true,
		"cudart64_12.dll":    true,
		"ggml-cuda.dll":      true,
	}

	binDir := filepath.Dir(destPath)
	found := false

	for _, f := range r.File {
		name := filepath.Base(f.Name)

		if name == target {
			if err := extractZipFile(f, destPath); err != nil {
				return err
			}
			found = true
		} else if dllsToExtract[name] {
			dllPath := filepath.Join(binDir, name)
			if err := extractZipFile(f, dllPath); err != nil {
				fmt.Printf("      Warning: failed to extract %s: %v\n", name, err)
			}
		}
	}

	if !found {
		return fmt.Errorf("llama-server not found in archive")
	}
	return nil
}

func extractZipFile(f *zip.File, destPath string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, rc)
	return err
}

func extractFromTarGz(archivePath, destPath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	target := "llama-server"

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		name := filepath.Base(hdr.Name)
		if name == target {
			out, err := os.Create(destPath)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
			os.Chmod(destPath, 0755)
			return nil
		}
	}

	return fmt.Errorf("llama-server not found in archive")
}

// ShouldUseIso3 determines iso3 support via static checks — no runtime detection.
// Two conditions: (1) SM >= 80 (Ampere+), (2) binary is turboquant build.
func ShouldUseIso3(isTurboQuant bool, sm int) bool {
	return isTurboQuant && sm >= 80
}

// SupportsGraphSplit checks if the binary supports -sm graph (turboquant fork feature).
// Thread-safe, only runs --help once per session via sync.Once.
var (
	graphSplitOnce   sync.Once
	graphSplitResult bool
)

func SupportsGraphSplit(binaryPath string) bool {
	graphSplitOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, binaryPath, "--help").CombinedOutput()
		graphSplitResult = err == nil && strings.Contains(string(out), "graph")
	})
	return graphSplitResult
}

// ValidateCUDAVersion checks for known CUDA driver issues
func ValidateCUDAVersion(hw *hardware.HardwareProbe) error {
	gpu := hw.PrimaryGPU()
	if gpu == nil {
		return nil
	}

	// CUDA 13.2 has a confirmed bug affecting low-bit quantization inference
	// (garbled output, repeated tokens). Downgrade to 13.1 resolves it.
	if gpu.CUDADriver == "13.2" {
		fmt.Printf("      ⚠️  CUDA %s detected — known bug with low-bit quantization\n", gpu.CUDADriver)
		fmt.Println("      If you see garbled output, downgrade driver to CUDA 13.1")
	}

	// Blackwell + CUDA 13.x: use CUDA 12.4 binary for stability
	if gpu.IsBlackwell && strings.HasPrefix(gpu.CUDADriver, "13.") {
		fmt.Printf("      Warning: RTX 50 series with CUDA %s detected\n", gpu.CUDADriver)
		fmt.Println("      Ashforge will use CUDA 12.4 binary for stability.")
	}

	return nil
}
