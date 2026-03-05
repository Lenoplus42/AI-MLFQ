package aischeduler

import (
	"sync"
	"time"
)

// TaskState defines the current status of an Agent task within our system.
type TaskState int
const (
	StateReady TaskState = iota // Waiting in a queue to be processed.
	StateRunning                // Currently consuming the simulated LLM resource
	StateBlocked                // Waiting for external I/O (e.g., API rate limit)
	StateDone                   // Execution complete
)

// AgentTask represents a single lifecycle of an LLM request from an agent.
type AgentTask struct {
	TaskID          string
	PriorityLevel   int       // 0 (Top VIP), 1 (Middle), 2 (Bottom FCFS) etc.
	TokensRequired  int       // Simulated total tokens needed for this task
	TokensProcessed int       
	TimeQuantumUsed int       // Time slices consumed in the current queue level
	State           TaskState

	// Metrics for our final stress test benchmark report
	ArrivalTime    time.Time
	FirstResponse  time.Time
	CompletionTime time.Time
}

// MLFQQueue represents a single tier in our multi-level feedback system.
// It MUST be thread-safe as tasks will be enqueued and dequeued concurrently.
type MLFQQueue struct {
	Level   int
	Quantum int            // The time slice allocated for this specific level
	
	mu      sync.Mutex     // Mutex lock to prevent data races from concurrent goroutines
	tasks   []*AgentTask   // Slice acting as the underlying queue structure
}

// Supervisor interface defines the core contract for our MLFQ scheduler.
type Supervisor interface {
	SubmitTask(task *AgentTask) // Entry point for new agents
	Tick()                      // The global timer interrupt pulse that drives scheduling
	PromoteAll()                // Starvation prevention: moves all tasks to Queue 0
}