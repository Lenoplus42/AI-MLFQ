package aischeduler

import (
	"fmt"
	"time"
)

// MLFQScheduler is a fully dynamic, configurable multi-level feedback queue
// scheduler for AI agents. 
type MLFQScheduler struct {
	queues []*MLFQQueue
}

// NewMLFQScheduler builds an MLFQ with len(quantums) priority levels.
// Each entry in quantums sets the time-slice for that level.
// A quantum of 0 or negative is treated as infinite (pure FCFS), which is
// the conventional behaviour for the lowest-priority queue.
func NewMLFQScheduler(quantums []int) *MLFQScheduler {
	queues := make([]*MLFQQueue, len(quantums))
	for i, q := range quantums {
		queues[i] = &MLFQQueue{
			Level:   i,
			Quantum: q,
			tasks:   make([]*AgentTask, 0),
		}
	}
	return &MLFQScheduler{queues: queues}
}

// SubmitTask enqueues a new task into the highest-priority queue (Level 0).
// Note: TokensRequired is unknown in real life, but is needed in our benchmark simulator.
func (s *MLFQScheduler) SubmitTask(task *AgentTask) {
	task.PriorityLevel = 0
	task.State = StateReady
	task.ArrivalTime = time.Now()
	s.queues[0].Enqueue(task)
}

// Tick simulates one timer interrupt in AI-MLFQ. Each call selects one task
// from the highest non-empty queue, runs it for one time unit, then
// re-schedules or retires it according to MLFQ policy.
func (s *MLFQScheduler) Tick() {
	// Scan top-down: highest priority (starting from Level 0) wins resource.
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

	task.TokensProcessed++
	task.TimeQuantumUsed++
	task.State = StateRunning

	// Condition A — task has consumed all tokens it needs.
	if task.TokensProcessed >= task.TokensRequired {
		task.State = StateDone
		task.CompletionTime = time.Now()
		return
	}

	// Condition B — quantum exhausted: demote one level (FCFS queues are exempt).
	// Demotion is the core mechanism that prevents tasks from
	// monopolising high-priority queues.
	isFiniteQuantum := currentQueue.Quantum > 0
	if isFiniteQuantum && task.TimeQuantumUsed >= currentQueue.Quantum {
		task.TimeQuantumUsed = 0
		if task.PriorityLevel < len(s.queues) - 1 {
			task.PriorityLevel++
		}
		task.State = StateReady
		s.queues[task.PriorityLevel].Enqueue(task)
		return
	}

	// Condition C — still has tokens left and quantum is not exhausted; continue
	// in the same queue.
	task.State = StateReady
	currentQueue.Enqueue(task)
}

// PrintStatus logs the current depth of every queue level.
// For debugging and observing scheduler behaviour under load.
func (s *MLFQScheduler) PrintStatus() {
	fmt.Println("=== MLFQ Status ===")
	for _, q := range s.queues {
		label := fmt.Sprintf("Q%d (quantum=%d)", q.Level, q.Quantum)
		if q.Quantum <= 0 {
			label = fmt.Sprintf("Q%d (FCFS)", q.Level)
		}
		fmt.Printf("  %-20s tasks: %d\n", label, q.Len()) // Nice little formatting trick to align the text.
	}
	fmt.Println("===================")
}
