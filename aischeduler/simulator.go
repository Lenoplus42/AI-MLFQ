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
	NumTasks    int
	InteractiveRatio float64
	InteractiveToks  [2]int // [min, max] tokens for interactive tasks
	HeavyToks        [2]int // [min, max] tokens for heavy-batch tasks

	ArrivalIntervalTicks int // ticks between successive task arrivals

	// Physical GPU constraints — scaffolding for the VRAM eviction layer.
	MaxKVCacheTokens         int // total VRAM capacity expressed in KV-cache tokens
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

		// Admit one task every ArrivalIntervalTicks. Tasks that have not yet
		// arrived are invisible to the scheduler, so a short task arriving after
		// a heavy one is immediately eligible to preempt it at Q0.
		if nextSubmitIdx < len(tasks) && tick%cfg.ArrivalIntervalTicks == 0 {
			r := records[nextSubmitIdx]
			r.arrivalTick = tick
			scheduler.SubmitTask(r.task)
			nextSubmitIdx++
		}

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
