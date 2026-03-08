package aischeduler

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"time"
)

const (
	interactiveRatio   = 0.80
	interactiveMinToks = 10
	interactiveMaxToks = 50
	heavyMinToks       = 500
	heavyMaxToks       = 1000
)

// taskClass tags a task's workload category for metric segregation.
type taskClass int

const (
	classInteractive taskClass = iota
	classHeavy
)

// taskRecord pairs an AgentTask with its tick-based timing observations.
type taskRecord struct {
	task         *AgentTask
	class        taskClass
	arrivalTick  int
	firstRspTick int
	doneTick     int
}

// SimResult holds the raw observations from a single simulation run.
type SimResult struct {
	name        string
	totalTicks  int
	interactive []*taskRecord
	heavy       []*taskRecord
}

// GenerateWorkload produces numTasks AgentTasks with an 80/20 interactive-to-heavy
// split. It returns two independent deep copies so FIFO and MLFQ runs are identical.
func GenerateWorkload(numTasks int) ([]*AgentTask, []*AgentTask) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	original := make([]*AgentTask, numTasks)
	for i := range original {
		tokens := interactiveMinToks + rng.Intn(interactiveMaxToks-interactiveMinToks+1)
		if float64(i)/float64(numTasks) >= interactiveRatio {
			tokens = heavyMinToks + rng.Intn(heavyMaxToks-heavyMinToks+1)
		}
		original[i] = &AgentTask{
			TaskID:         fmt.Sprintf("task-%04d", i),
			TokensRequired: tokens,
		}
	}

	// Shuffle so heavy tasks are not clustered at the tail.
	rng.Shuffle(len(original), func(i, j int) { original[i], original[j] = original[j], original[i] })
	
	// Deep copy to keep task list exact same, for precise comparison.
	fifo := deepCopyTasks(original)
	mlfq := deepCopyTasks(original)
	return fifo, mlfq
}

func deepCopyTasks(src []*AgentTask) []*AgentTask {
	dst := make([]*AgentTask, len(src))
	for i, t := range src {
		cp := *t
		dst[i] = &cp
	}
	return dst
}

// classifyTask returns classInteractive when TokensRequired falls within the
// interactive band, classHeavy otherwise.
func classifyTask(t *AgentTask) taskClass {
	if t.TokensRequired <= interactiveMaxToks {
		return classInteractive
	}
	return classHeavy
}

// RunSimulation submits all tasks to scheduler, drives Tick() until every task
// reaches StateDone, and returns a SimResult with per-task tick observations.
func RunSimulation(name string, scheduler *MLFQScheduler, tasks []*AgentTask) SimResult {
	records := make([]*taskRecord, len(tasks))
	for i, t := range tasks {
		records[i] = &taskRecord{
			task:  t,
			class: classifyTask(t),
		}
	}

	// Snapshot arrival tick before submitting (SubmitTask overwrites ArrivalTime
	// with wall-clock; we capture the tick count here instead).
	tick := 0
	for _, r := range records {
		r.arrivalTick = tick
		scheduler.SubmitTask(r.task)
	}

	pending := len(tasks)
	// Track which tasks have been observed as done each tick to capture doneTick.
	doneSet := make(map[string]bool, len(tasks))
	recordByID := make(map[string]*taskRecord, len(records))
	for _, r := range records {
		recordByID[r.task.TaskID] = r
	}

	// firstRspSeen tracks whether we've already captured the first-response tick.
	firstRspSeen := make(map[string]bool, len(tasks))

	for pending > 0 {
		tick++
		scheduler.Tick()

		for _, r := range records {
			t := r.task
			if !firstRspSeen[t.TaskID] && !t.FirstResponse.IsZero() {
				r.firstRspTick = tick
				firstRspSeen[t.TaskID] = true
			}
			if !doneSet[t.TaskID] && t.State == StateDone {
				r.doneTick = tick
				doneSet[t.TaskID] = true
				pending--
			}
		}
	}

	result := SimResult{name: name, totalTicks: tick}
	for _, r := range recordByID {
		switch r.class {
		case classInteractive:
			result.interactive = append(result.interactive, r)
		case classHeavy:
			result.heavy = append(result.heavy, r)
		}
	}
	return result
}

// metrics holds the computed summary statistics for one task class.
type metrics struct {
	count          int
	avgTurnaround  float64
	p99FirstRspLat float64
}

func computeMetrics(records []*taskRecord) metrics {
	if len(records) == 0 {
		return metrics{}
	}

	latencies := make([]float64, 0, len(records))
	var totalTurnaround float64

	for _, r := range records {
		latencies = append(latencies, float64(r.firstRspTick-r.arrivalTick))
		totalTurnaround += float64(r.doneTick - r.arrivalTick)
	}

	sort.Float64s(latencies)
	p99Idx := int(float64(len(latencies)-1) * 0.99)

	return metrics{
		count:          len(records),
		avgTurnaround:  totalTurnaround / float64(len(records)),
		p99FirstRspLat: latencies[p99Idx],
	}
}

// PrintBenchmarkReport prints a structured comparison table between two SimResults.
// Assumes results[0] is the baseline (FIFO) and results[1] is the challenger (MLFQ).
func PrintBenchmarkReport(results [2]SimResult) {
	fifo := results[0]
	mlfq := results[1]

	fifoI := computeMetrics(fifo.interactive)
	fifoH := computeMetrics(fifo.heavy)
	mlfqI := computeMetrics(mlfq.interactive)
	mlfqH := computeMetrics(mlfq.heavy)

	sep := strings.Repeat("─", 72)
	hdr := strings.Repeat("═", 72)

	fmt.Println()
	fmt.Println(hdr)
	fmt.Printf("  %-38s  %12s  %12s\n", "METRIC", fifo.name, mlfq.name)
	fmt.Println(hdr)

	printRow := func(label string, a, b float64, unit string) {
		delta := ""
		if a != 0 {
			pct := (b - a) / a * 100
			sign := "+"
			if pct < 0 {
				sign = ""
			}
			delta = fmt.Sprintf("  (%s%.1f%%)", sign, pct)
		}
		fmt.Printf("  %-38s  %11.2f%s  %11.2f%s%s\n", label, a, unit, b, unit, delta)
	}

	fmt.Printf("  %-38s  %12d  %12d\n", "Total Ticks to Drain", fifo.totalTicks, mlfq.totalTicks)
	fmt.Println(sep)

	fmt.Printf("  %-38s\n", "INTERACTIVE TASKS (tokens 10–50)")
	fmt.Printf("  %-38s  %12d  %12d\n", "  Count", fifoI.count, mlfqI.count)
	printRow("  Avg Turnaround Time", fifoI.avgTurnaround, mlfqI.avgTurnaround, "t")
	printRow("  P99 First-Response Latency", fifoI.p99FirstRspLat, mlfqI.p99FirstRspLat, "t")
	fmt.Println(sep)

	fmt.Printf("  %-38s\n", "HEAVY BATCH TASKS (tokens 500–1000)")
	fmt.Printf("  %-38s  %12d  %12d\n", "  Count", fifoH.count, mlfqH.count)
	printRow("  Avg Turnaround Time", fifoH.avgTurnaround, mlfqH.avgTurnaround, "t")
	printRow("  P99 First-Response Latency", fifoH.p99FirstRspLat, mlfqH.p99FirstRspLat, "t")
	fmt.Println(hdr)
	fmt.Println()
}
