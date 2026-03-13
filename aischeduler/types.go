package aischeduler

import (
	"sync"
	"time"
)

// TaskState defines the current status of an Agent task within our system.
type TaskState int

const (
	StateReady   TaskState = iota // Waiting in a queue to be processed.
	StateRunning                  // Currently consuming the simulated LLM resource.
	StateBlocked                  // Waiting for external I/O (e.g., PCIe page-in after VRAM eviction).
	StateDone                     // Execution complete.
)

// TaskPhase models the two-stage LLM inference pipeline within a single request.
type TaskPhase int

const (
	// PhasePrefill is the compute-bound phase where the full prompt is
	// processed in parallel. One scheduler tick consumes PrefillTokensPerTick
	// tokens and builds the initial KV cache.
	PhasePrefill TaskPhase = iota

	// PhaseDecode is the memory-bandwidth-bound autoregressive phase. One
	// scheduler tick generates DecodeTokensPerTick output tokens.
	PhaseDecode
)

// AgentTask represents a single lifecycle of an LLM request from an agent.
type AgentTask struct {
	TaskID          string
	PriorityLevel   int       // 0 (Top VIP), 1 (Middle), 2 (Bottom FCFS), etc.
	TimeQuantumUsed int       // Ticks consumed at the current queue level.
	State           TaskState
	Phase           TaskPhase // Current execution phase; starts at PhasePrefill.

	// ArrivalTick is the pre-calculated simulation tick at which this task
	// enters the system, derived from a Poisson inter-arrival process during
	// workload generation. RunSimulation submits the task when tick >= ArrivalTick.
	ArrivalTick int

	PrefillTokensRequired  int // Total prompt tokens to process.
	PrefillTokensProcessed int // Prompt tokens consumed so far.
	DecodeTokensRequired   int // Total output tokens to generate.
	DecodeTokensProcessed  int // Output tokens generated so far.

	// Metrics for the benchmark report.
	ArrivalTime    time.Time
	FirstResponse  time.Time
	CompletionTime time.Time
}

// TotalKVTokens returns the current KV-cache footprint in VRAM: the sum of all
// prompt tokens already processed plus all output tokens generated. This is the
// value the VRAM eviction logic tracks against MaxKVCacheTokens.
func (t *AgentTask) TotalKVTokens() int {
	return t.PrefillTokensProcessed + t.DecodeTokensProcessed
}

// MLFQQueue represents a single tier in our multi-level feedback system.
// It MUST be thread-safe as tasks will be enqueued and dequeued concurrently.
type MLFQQueue struct {
	Level   int
	Quantum int // The time slice allocated for this specific level.

	mu    sync.Mutex   // Guards tasks against concurrent access.
	tasks []*AgentTask // Slice acting as the underlying FIFO structure.
}

// Supervisor interface defines the core contract for our MLFQ scheduler.
type Supervisor interface {
	SubmitTask(task *AgentTask) // Entry point for new agents.
	Tick()                      // The global timer interrupt pulse that drives scheduling.
	PromoteAll()                // Starvation prevention: moves all tasks to Queue 0.
}
