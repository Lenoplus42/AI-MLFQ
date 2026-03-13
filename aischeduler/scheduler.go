package aischeduler

import (
	"fmt"
	"time"
)

// MLFQScheduler is a fully dynamic, configurable multi-level feedback queue
// scheduler for AI agents.
type MLFQScheduler struct {
	queues               []*MLFQQueue
	prefillTokensPerTick int // tokens processed per tick during PhasePrefill (chunked)
	decodeTokensPerTick  int // tokens generated per tick during PhaseDecode (autoregressive)
}

// NewMLFQScheduler builds an MLFQ with len(quantums) priority levels.
// Each entry in quantums sets the time-slice for that level; a negative or zero
// value is treated as infinite (pure FCFS), which is the conventional behaviour
// for the lowest-priority tier. The cfg argument supplies the per-tick token
// rates for both inference phases.
func NewMLFQScheduler(cfg SimConfig, quantums []int) *MLFQScheduler {
	queues := make([]*MLFQQueue, len(quantums))
	for i, q := range quantums {
		queues[i] = &MLFQQueue{
			Level:   i,
			Quantum: q,
			tasks:   make([]*AgentTask, 0),
		}
	}
	return &MLFQScheduler{
		queues:               queues,
		prefillTokensPerTick: cfg.PrefillTokensPerTick,
		decodeTokensPerTick:  cfg.DecodeTokensPerTick,
	}
}

// SubmitTask enqueues a new task into the highest-priority queue (Level 0).
func (s *MLFQScheduler) SubmitTask(task *AgentTask) {
	task.PriorityLevel = 0
	task.State = StateReady
	task.ArrivalTime = time.Now()
	s.queues[0].Enqueue(task)
}

// Tick simulates one timer interrupt in AI-MLFQ. It selects the highest-priority
// non-empty queue, runs the head task for one time unit through the appropriate
// inference phase, then re-schedules or retires it according to MLFQ policy.
func (s *MLFQScheduler) Tick() {
	var currentQueue *MLFQQueue
	for _, q := range s.queues {
		if q.Len() > 0 {
			currentQueue = q
			break
		}
	}
	if currentQueue == nil {
		return
	}

	task := currentQueue.Dequeue()

	if task.FirstResponse.IsZero() {
		task.FirstResponse = time.Now()
	}

	task.TimeQuantumUsed++
	task.State = StateRunning

	done := s.advancePhase(task)
	if done {
		task.State = StateDone
		task.CompletionTime = time.Now()
		return
	}

	// Condition B — quantum exhausted: demote one level (infinite-quantum queues
	// are exempt). Demotion is the core mechanism that prevents long-running tasks
	// from monopolising high-priority queues.
	isFiniteQuantum := currentQueue.Quantum > 0
	if isFiniteQuantum && task.TimeQuantumUsed >= currentQueue.Quantum {
		task.TimeQuantumUsed = 0
		if task.PriorityLevel < len(s.queues)-1 {
			task.PriorityLevel++
		}
		task.State = StateReady
		s.queues[task.PriorityLevel].Enqueue(task)
		return
	}

	// Condition C — still has work remaining and quantum is not exhausted.
	// Re-insert at the front so this task holds the CPU for its full quantum
	// without yielding to peers after every single tick.
	task.State = StateReady
	currentQueue.EnqueueFront(task)
}

// advancePhase executes one tick of work for the task's current inference phase
// and returns true when the task has fully completed both phases.
func (s *MLFQScheduler) advancePhase(task *AgentTask) bool {
	switch task.Phase {
	case PhasePrefill:
		remaining := task.PrefillTokensRequired - task.PrefillTokensProcessed
		chunk := s.prefillTokensPerTick
		if remaining < chunk {
			chunk = remaining
		}
		task.PrefillTokensProcessed += chunk

		if task.PrefillTokensProcessed >= task.PrefillTokensRequired {
			// Cap to avoid over-counting VRAM and transition immediately so
			// the next tick begins decoding without a wasted scheduling round.
			task.PrefillTokensProcessed = task.PrefillTokensRequired
			task.Phase = PhaseDecode
		}
		return false

	default: // PhaseDecode
		task.DecodeTokensProcessed += s.decodeTokensPerTick
		return task.DecodeTokensProcessed >= task.DecodeTokensRequired
	}
}

// Evict removes a specific task from whichever queue currently holds it.
// Returns true if the task was found and removed. The caller is responsible
// for updating the task's State and tracking the eviction penalty.
func (s *MLFQScheduler) Evict(taskID string) bool {
	for _, q := range s.queues {
		if q.Remove(taskID) {
			return true
		}
	}
	return false
}

// ReEnqueue places a previously evicted task back into the queue corresponding
// to its current PriorityLevel, preserving any demotion that occurred before
// the eviction. Unlike SubmitTask, it does not reset the task to Q0.
func (s *MLFQScheduler) ReEnqueue(task *AgentTask) {
	task.State = StateReady
	s.queues[task.PriorityLevel].Enqueue(task)
}

// PrintStatus logs the current depth of every queue level.
func (s *MLFQScheduler) PrintStatus() {
	fmt.Println("=== MLFQ Status ===")
	for _, q := range s.queues {
		label := fmt.Sprintf("Q%d (quantum=%d)", q.Level, q.Quantum)
		if q.Quantum <= 0 {
			label = fmt.Sprintf("Q%d (FCFS)", q.Level)
		}
		fmt.Printf("  %-20s tasks: %d\n", label, q.Len())
	}
	fmt.Println("===================")
}
