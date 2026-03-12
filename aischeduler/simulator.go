package aischeduler

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

// SimConfig is the single source of truth for every tunable parameter in a
// simulation run. Pass it by value; zero values are intentionally invalid so
// callers must be explicit.
type SimConfig struct {
	NumTasks         int
	InteractiveRatio float64
	InteractiveToks  [2]int // [min, max] tokens for interactive tasks
	HeavyToks        [2]int // [min, max] tokens for heavy-batch tasks

	ArrivalIntervalTicks int // ticks between successive task arrivals

	// Physical GPU constraints — scaffolding for the VRAM eviction layer.
	MaxKVCacheTokens          int // total VRAM capacity expressed in KV-cache tokens
	ContextSwitchPenaltyTicks int // PCIe page-out + page-in cost when evicting a task

	// If non-empty, the benchmark report is written to this path in addition
	// to being printed to stdout.
	ReportFilepath string
}

// taskClass tags a task's workload category for metric segregation.
type taskClass int

const (
	classInteractive taskClass = iota
	classHeavy
)

// taskRecord pairs an AgentTask with its tick-based timing observations and
// any simulator-level eviction state. These fields are intentionally kept out
// of AgentTask to avoid polluting the core scheduling types.
type taskRecord struct {
	task         *AgentTask
	class        taskClass
	arrivalTick  int
	firstRspTick int
	doneTick     int

	// VRAM eviction state — managed exclusively by the simulator loop.
	isEvicted        bool
	penaltyRemaining int // ticks until PCIe page-in completes
}

// SimResult holds the raw observations from a single simulation run.
type SimResult struct {
	name        string
	totalTicks  int
	interactive []*taskRecord
	heavy       []*taskRecord
}

// GenerateWorkload produces cfg.NumTasks AgentTasks according to the
// interactive/heavy split defined in cfg. Returns two independent deep copies
// so FIFO and MLFQ runs operate on identical but unshared data.
func GenerateWorkload(cfg SimConfig) ([]*AgentTask, []*AgentTask) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	interMin, interMax := cfg.InteractiveToks[0], cfg.InteractiveToks[1]
	heavyMin, heavyMax := cfg.HeavyToks[0], cfg.HeavyToks[1]

	original := make([]*AgentTask, cfg.NumTasks)
	for i := range original {
		tokens := interMin + rng.Intn(interMax-interMin+1)
		if float64(i)/float64(cfg.NumTasks) >= cfg.InteractiveRatio {
			tokens = heavyMin + rng.Intn(heavyMax-heavyMin+1)
		}
		original[i] = &AgentTask{
			TaskID:         fmt.Sprintf("task-%04d", i),
			TokensRequired: tokens,
		}
	}

	// Scatter heavy tasks throughout the slice so they are not clustered at
	// the tail, which would artificially favour MLFQ.
	rng.Shuffle(len(original), func(i, j int) { original[i], original[j] = original[j], original[i] })

	return deepCopyTasks(original), deepCopyTasks(original)
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
// interactive band defined by cfg, classHeavy otherwise.
func classifyTask(t *AgentTask, cfg SimConfig) taskClass {
	if t.TokensRequired <= cfg.InteractiveToks[1] {
		return classInteractive
	}
	return classHeavy
}

// RunSimulation drives the scheduler until every submitted task reaches
// StateDone, feeding tasks one-at-a-time at cfg.ArrivalIntervalTicks cadence
// to replicate realistic Poisson-ish traffic rather than a thundering herd.
//
// When cfg.MaxKVCacheTokens > 0 the simulator enforces a VRAM budget: after
// each scheduler tick it sums the KV-cache footprint (TokensProcessed) of all
// active tasks and, if the budget is breached, evicts the lowest-priority tasks
// first. Each evicted task incurs cfg.ContextSwitchPenaltyTicks of PCIe I/O
// delay before it is eligible to re-enter the scheduler.
func RunSimulation(name string, scheduler *MLFQScheduler, tasks []*AgentTask, cfg SimConfig) SimResult {
	records := make([]*taskRecord, len(tasks))
	for i, t := range tasks {
		records[i] = &taskRecord{
			task:  t,
			class: classifyTask(t, cfg),
		}
	}

	recordByID := make(map[string]*taskRecord, len(records))
	for _, r := range records {
		recordByID[r.task.TaskID] = r
	}

	doneSet := make(map[string]bool, len(tasks))
	firstRspSeen := make(map[string]bool, len(tasks))

	tick := 0
	nextSubmitIdx := 0
	pending := len(tasks)

	for pending > 0 {
		tick++

		// Phase 1 — Admit: one new task enters every ArrivalIntervalTicks.
		// Tasks that have not yet arrived are invisible to the scheduler, so a
		// short task that arrives while a heavy task is running immediately
		// becomes the highest-priority work item at Q0.
		if nextSubmitIdx < len(tasks) && tick%cfg.ArrivalIntervalTicks == 0 {
			r := records[nextSubmitIdx]
			r.arrivalTick = tick
			scheduler.SubmitTask(r.task)
			nextSubmitIdx++
		}

		// Phase 2 — Page-In: count down PCIe transfer timers. Tasks whose
		// penalty expires are re-enqueued before scheduler.Tick() runs, so they
		// are eligible to be scheduled in this very tick.
		for _, r := range records {
			if !r.isEvicted {
				continue
			}
			if r.penaltyRemaining > 0 {
				r.penaltyRemaining--
			} else {
				// penaltyRemaining == 0: page-in complete.
				r.isEvicted = false
				scheduler.ReEnqueue(r.task)
			}
		}

		// Phase 3 — Schedule: advance the simulation by one timer interrupt.
		scheduler.Tick()

		// Phase 4 — Observe: capture first-response and completion ticks.
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

		// Phase 5 — VRAM eviction: enforce the KV-cache budget.
		// Runs after observation so tasks that completed this tick are already
		// StateDone and excluded from both the footprint sum and victim pool.
		if cfg.MaxKVCacheTokens <= 0 {
			continue
		}

		vramUsed := 0
		for _, r := range records {
			if r.task.TokensProcessed > 0 && !r.isEvicted && r.task.State != StateDone {
				vramUsed += r.task.TokensProcessed
			}
		}

		if vramUsed <= cfg.MaxKVCacheTokens {
			continue
		}

		// Build the eviction candidate list: active tasks sitting in a queue,
		// sorted by PriorityLevel descending so the lowest-priority work (the
		// largest level number) is evicted first. Q0 tasks are touched only when
		// no lower-priority alternatives remain.
		candidates := make([]*taskRecord, 0)
		for _, r := range records {
			if !r.isEvicted && r.task.State == StateReady && r.task.TokensProcessed > 0 {
				candidates = append(candidates, r)
			}
		}
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].task.PriorityLevel > candidates[j].task.PriorityLevel
		})

		for _, victim := range candidates {
			if vramUsed <= cfg.MaxKVCacheTokens {
				break
			}
			if !scheduler.Evict(victim.task.TaskID) {
				continue
			}
			vramUsed -= victim.task.TokensProcessed
			victim.isEvicted = true
			victim.penaltyRemaining = cfg.ContextSwitchPenaltyTicks
			victim.task.State = StateBlocked
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

// PrintBenchmarkReport renders a structured comparison table to stdout. When
// cfg.ReportFilepath is non-empty, the identical ASCII output is also persisted
// to that path, creating or truncating the file as needed.
func PrintBenchmarkReport(results [2]SimResult, cfg SimConfig) {
	fifo := results[0]
	mlfq := results[1]

	fifoI := computeMetrics(fifo.interactive)
	fifoH := computeMetrics(fifo.heavy)
	mlfqI := computeMetrics(mlfq.interactive)
	mlfqH := computeMetrics(mlfq.heavy)

	writers := []io.Writer{os.Stdout}
	if cfg.ReportFilepath != "" {
		f, err := os.Create(cfg.ReportFilepath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warn: could not create report file %q: %v\n", cfg.ReportFilepath, err)
		} else {
			defer f.Close()
			writers = append(writers, f)
		}
	}
	out := io.MultiWriter(writers...)

	interLabel := fmt.Sprintf("INTERACTIVE TASKS (tokens %d–%d)", cfg.InteractiveToks[0], cfg.InteractiveToks[1])
	heavyLabel := fmt.Sprintf("HEAVY BATCH TASKS (tokens %d–%d)", cfg.HeavyToks[0], cfg.HeavyToks[1])

	sep := strings.Repeat("─", 72)
	hdr := strings.Repeat("═", 72)

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
		fmt.Fprintf(out, "  %-38s  %11.2f%s  %11.2f%s%s\n", label, a, unit, b, unit, delta)
	}

	fmt.Fprintln(out)
	fmt.Fprintln(out, hdr)
	fmt.Fprintf(out, "  %-38s  %12s  %12s\n", "METRIC", fifo.name, mlfq.name)
	fmt.Fprintln(out, hdr)
	fmt.Fprintf(out, "  %-38s  %12d  %12d\n", "Total Ticks to Drain", fifo.totalTicks, mlfq.totalTicks)
	fmt.Fprintln(out, sep)

	fmt.Fprintf(out, "  %-38s\n", interLabel)
	fmt.Fprintf(out, "  %-38s  %12d  %12d\n", "  Count", fifoI.count, mlfqI.count)
	printRow("  Avg Turnaround Time", fifoI.avgTurnaround, mlfqI.avgTurnaround, "t")
	printRow("  P99 First-Response Latency", fifoI.p99FirstRspLat, mlfqI.p99FirstRspLat, "t")
	fmt.Fprintln(out, sep)

	fmt.Fprintf(out, "  %-38s\n", heavyLabel)
	fmt.Fprintf(out, "  %-38s  %12d  %12d\n", "  Count", fifoH.count, mlfqH.count)
	printRow("  Avg Turnaround Time", fifoH.avgTurnaround, mlfqH.avgTurnaround, "t")
	printRow("  P99 First-Response Latency", fifoH.p99FirstRspLat, mlfqH.p99FirstRspLat, "t")
	fmt.Fprintln(out, hdr)
	fmt.Fprintln(out)
}
