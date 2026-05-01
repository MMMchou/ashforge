package llm

import (
	"testing"

	"github.com/MMMchou/ashforge/core/hw"
)

func TestThreadsForMode(t *testing.T) {
	tests := []struct {
		name string
		mode string
		sys  *hw.System
		want int
	}{
		{
			name: "full_gpu returns 2",
			mode: "full_gpu",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 16}},
			want: 2,
		},
		{
			name: "unknown mode returns 2",
			mode: "something_else",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 16}},
			want: 2,
		},
		{
			name: "empty mode returns 2",
			mode: "",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 16}},
			want: 2,
		},
		{
			name: "moe_offload uses half cores",
			mode: "moe_offload",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 16}},
			want: 8,
		},
		{
			name: "moe_partial uses half cores",
			mode: "moe_partial",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 24}},
			want: 12,
		},
		{
			name: "moe_offload clamps to minimum 4",
			mode: "moe_offload",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 4}},
			want: 4,
		},
		{
			name: "moe_partial clamps to minimum 4 with 2 cores",
			mode: "moe_partial",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 2}},
			want: 4,
		},
		{
			name: "moe_offload with 0 cores clamps to 4",
			mode: "moe_offload",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 0}},
			want: 4,
		},
		{
			name: "moe_offload with 8 cores returns 4",
			mode: "moe_offload",
			sys:  &hw.System{CPU: hw.CPUInfo{Cores: 8}},
			want: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ThreadsForMode(tt.mode, tt.sys)
			if got != tt.want {
				t.Errorf("ThreadsForMode(%q, cores=%d) = %d, want %d",
					tt.mode, tt.sys.CPU.Cores, got, tt.want)
			}
		})
	}
}

func TestCacheReuseForCtx(t *testing.T) {
	tests := []struct {
		name    string
		ctxSize int
		want    int
	}{
		{name: "large ctx 32768", ctxSize: 32768, want: 1024},
		{name: "large ctx above 32768", ctxSize: 65536, want: 1024},
		{name: "medium ctx 8192", ctxSize: 8192, want: 512},
		{name: "medium ctx 16384", ctxSize: 16384, want: 512},
		{name: "small ctx 4096", ctxSize: 4096, want: 256},
		{name: "small ctx 2048", ctxSize: 2048, want: 256},
		{name: "zero ctx", ctxSize: 0, want: 256},
		{name: "boundary just below 8192", ctxSize: 8191, want: 256},
		{name: "boundary just below 32768", ctxSize: 32767, want: 512},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CacheReuseForCtx(tt.ctxSize)
			if got != tt.want {
				t.Errorf("CacheReuseForCtx(%d) = %d, want %d",
					tt.ctxSize, got, tt.want)
			}
		})
	}
}

func TestContainsFlag(t *testing.T) {
	tests := []struct {
		name  string
		args  []string
		flags []string
		want  bool
	}{
		{
			name:  "flag present",
			args:  []string{"--verbose", "--debug", "--output", "file.txt"},
			flags: []string{"--debug"},
			want:  true,
		},
		{
			name:  "flag absent",
			args:  []string{"--verbose", "--output", "file.txt"},
			flags: []string{"--debug"},
			want:  false,
		},
		{
			name:  "multiple flags first matches",
			args:  []string{"--verbose", "--debug"},
			flags: []string{"--verbose", "--quiet"},
			want:  true,
		},
		{
			name:  "multiple flags second matches",
			args:  []string{"--quiet", "--debug"},
			flags: []string{"--verbose", "--quiet"},
			want:  true,
		},
		{
			name:  "no flags to check",
			args:  []string{"--verbose"},
			flags: []string{},
			want:  false,
		},
		{
			name:  "empty args",
			args:  []string{},
			flags: []string{"--debug"},
			want:  false,
		},
		{
			name:  "both empty",
			args:  []string{},
			flags: []string{},
			want:  false,
		},
		{
			name:  "nil args",
			args:  nil,
			flags: []string{"--debug"},
			want:  false,
		},
		{
			name:  "partial match is not a match",
			args:  []string{"--debug-mode"},
			flags: []string{"--debug"},
			want:  false,
		},
		{
			name:  "exact match required",
			args:  []string{"-v"},
			flags: []string{"-v"},
			want:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ContainsFlag(tt.args, tt.flags...)
			if got != tt.want {
				t.Errorf("ContainsFlag(%v, %v) = %v, want %v",
					tt.args, tt.flags, got, tt.want)
			}
		})
	}
}
