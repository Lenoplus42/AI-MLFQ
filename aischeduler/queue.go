package aischeduler

// Enqueue appends a task to the back of this queue level.
// Safe for concurrent use by multiple goroutines.
func (q *MLFQQueue) Enqueue(task *AgentTask) {
	// Note: We use a simple Mutex + Slice here.
	// A Lock-free RingBuffer would be more efficient for high-throughput, but 
	// at, for example, 10k concurrent agents, the bottleneck could be LLM network I/O, 
	// not lock contention. Avoid premature optimization here, will profile later and see if we need to change.
	q.mu.Lock()
	defer q.mu.Unlock() // Auto unlocks when function exits
	q.tasks = append(q.tasks, task) // "In MLFQ... Job starts in highest priority queue." -- CS162
}

// EnqueueFront inserts a task at the head of the queue, giving it priority
// over all other waiting tasks at this level. Used by Condition C to let a
// running task hold the CPU for its full quantum without interleaving.
func (q *MLFQQueue) EnqueueFront(task *AgentTask) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.tasks = append([]*AgentTask{task}, q.tasks...)
}

// Dequeue removes and returns the task at the front of this queue level.
// Returns nil when the queue is empty.
func (q *MLFQQueue) Dequeue() *AgentTask {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.tasks) == 0 {
		return nil
	}

	task := q.tasks[0] // same as var task *AgentTask = q.tasks[0]. Short Variable Declaration.
	q.tasks = q.tasks[1:]
	return task
}

// Remove finds and removes the first task matching taskID. Returns true if a
// task was found and excised, false if the ID was not present in this level.
func (q *MLFQQueue) Remove(taskID string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i, t := range q.tasks {
		if t.TaskID == taskID {
			q.tasks = append(q.tasks[:i], q.tasks[i+1:]...)
			return true
		}
	}
	return false
}

// Concurrency Safe method that returns the number of tasks currently in this queue.
func (q *MLFQQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.tasks)
}
