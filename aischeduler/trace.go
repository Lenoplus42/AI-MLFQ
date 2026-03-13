package aischeduler

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// traceSnapshotInterval is the number of simulation ticks between successive
// TraceSnapshot captures. 200 ticks keeps the JSON payload manageable for
// multi-hundred-thousand-tick simulations while still providing enough
// resolution to observe queue dynamics and VRAM pressure trends.
const traceSnapshotInterval = 200

// TraceSnapshot records a point-in-time system state for the web visualizer.
type TraceSnapshot struct {
	Tick      int `json:"tick"`
	VRAMUsage int `json:"vramUsage"`
	Q0Len     int `json:"q0Len"`
	Q1Len     int `json:"q1Len"`
	Q2Len     int `json:"q2Len"`
}

// TraceFile is the root structure written to disk. Embedding MaxKVCacheTokens
// here lets the frontend draw the VRAM threshold line without hardcoding it.
type TraceFile struct {
	MaxKVCacheTokens int              `json:"maxKVCacheTokens"`
	Snapshots        []TraceSnapshot  `json:"snapshots"`
}

// writeTrace serialises tf to path as compact JSON, creating any required
// parent directories first. All errors are wrapped with context.
func writeTrace(tf TraceFile, path string) error {
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create trace dir: %w", err)
		}
	}
	data, err := json.Marshal(tf)
	if err != nil {
		return fmt.Errorf("marshal trace: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write trace file: %w", err)
	}
	return nil
}

// safeIdx returns s[i] when i is a valid index, 0 otherwise. Used to read
// queue length slices of unknown size without panicking.
func safeIdx(s []int, i int) int {
	if i < len(s) {
		return s[i]
	}
	return 0
}
