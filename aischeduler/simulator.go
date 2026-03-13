package aischeduler

import (
	"fmt"
	"io"
	"math"
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

	// Per-class token ranges for each inference phase.
	InteractivePrefillToks [2]int // [min, max] prompt tokens for interactive tasks
	InteractiveDecodeToks  [2]int // [min, max] output tokens for interactive tasks
	HeavyPrefillToks       [2]int // [min, max] prompt tokens for heavy-batch tasks
	HeavyDecodeToks        [2]int // [min, max] output tokens for heavy-batch tasks

	// Structural tick-rate constants for each inference phase.
	// PrefillTokensPerTick models the parallelism of the attention kernel over the
	// full prompt (chunked prefill). DecodeTokensPerTick models autoregressive
	// single-token generation, which is always 1.
	PrefillTokensPerTick int
	DecodeTokensPerTick  int

	ArrivalIntervalTicks int // ticks between successive task arrivals

	// Physical GPU constraints.
	MaxKVCacheTokens          int // total VRAM budget expressed in KV-cache tokens
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

	randInRange := func(r [2]int) int { return r[0] + rng.Intn(r[1]-r[0]+1) }

	original := make([]*AgentTask, cfg.NumTasks)
	for i := range original {
		var prefill, decode int
		if float64(i)/float64(cfg.NumTasks) < cfg.InteractiveRatio {
			prefill = randInRange(cfg.InteractivePrefillToks)
			decode = randInRange(cfg.InteractiveDecodeToks)
		} else {
			prefill = randInRange(cfg.HeavyPrefillToks)
			decode = randInRange(cfg.HeavyDecodeToks)
		}
		original[i] = &AgentTask{
			TaskID:                fmt.Sprintf("task-%04d", i),
			Phase:                 PhasePrefill,
			PrefillTokensRequired: prefill,
			DecodeTokensRequired:  decode,
		}
	}

	// Scatter heavy tasks so they are not clustered at the tail, which would
	// artificially favour MLFQ by front-loading interactive work.
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

// classifyTask returns classInteractive when the task's prefill size falls
// within the interactive prefill band. The prefill range is the reliable
// discriminator since interactive and heavy ranges are non-overlapping.
func classifyTask(t *AgentTask, cfg SimConfig) taskClass {
	if t.PrefillTokensRequired <= cfg.InteractivePrefillToks[1] {
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
		// firstRspTick is recorded strictly when the task is in PhaseDecode with
		// at least one decode token generated. This correctly models TTFT as the
		// elapsed time from arrival to the moment the first output token appears,
		// which includes the full prefill cost.
		for _, r := range records {
			t := r.task
			if !firstRspSeen[t.TaskID] && t.Phase == PhaseDecode && t.DecodeTokensProcessed >= 1 {
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
			if r.task.TotalKVTokens() > 0 && !r.isEvicted && r.task.State != StateDone {
				vramUsed += r.task.TotalKVTokens()
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
			if !r.isEvicted && r.task.State == StateReady && r.task.TotalKVTokens() > 0 {
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
			vramUsed -= victim.task.TotalKVTokens()
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
	avgTBT         float64 // average time between consecutive decode tokens
}

func computeMetrics(records []*taskRecord) metrics {
	if len(records) == 0 {
		return metrics{}
	}

	latencies := make([]float64, 0, len(records))
	var totalTurnaround, totalTBT float64

	for _, r := range records {
		latencies = append(latencies, float64(r.firstRspTick-r.arrivalTick))
		totalTurnaround += float64(r.doneTick - r.arrivalTick)

		// TBT is the average gap between consecutive output tokens.
		// (doneTick - firstRspTick) spans the decode phase from the first token
		// to the last. N output tokens produce N-1 inter-token intervals, so the
		// denominator is max(1, DecodeTokensProcessed-1) to avoid division by
		// zero for single-token tasks while keeping the unit consistent (ticks).
		decodeSpan := float64(r.doneTick - r.firstRspTick)
		intervals := math.Max(1.0, float64(r.task.DecodeTokensProcessed-1))
		totalTBT += decodeSpan / intervals
	}

	sort.Float64s(latencies)
	p99Idx := int(float64(len(latencies)-1) * 0.99)
	n := float64(len(records))

	return metrics{
		count:          len(records),
		avgTurnaround:  totalTurnaround / n,
		p99FirstRspLat: latencies[p99Idx],
		avgTBT:         totalTBT / n,
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

	interLabel := fmt.Sprintf("INTERACTIVE  prefill %d–%d  decode %d–%d",
		cfg.InteractivePrefillToks[0], cfg.InteractivePrefillToks[1],
		cfg.InteractiveDecodeToks[0], cfg.InteractiveDecodeToks[1])
	heavyLabel := fmt.Sprintf("HEAVY BATCH  prefill %d–%d  decode %d–%d",
		cfg.HeavyPrefillToks[0], cfg.HeavyPrefillToks[1],
		cfg.HeavyDecodeToks[0], cfg.HeavyDecodeToks[1])

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
	printRow("  Avg Time Between Tokens (TBT)", fifoI.avgTBT, mlfqI.avgTBT, "t")
	fmt.Fprintln(out, sep)

	fmt.Fprintf(out, "  %-38s\n", heavyLabel)
	fmt.Fprintf(out, "  %-38s  %12d  %12d\n", "  Count", fifoH.count, mlfqH.count)
	printRow("  Avg Turnaround Time", fifoH.avgTurnaround, mlfqH.avgTurnaround, "t")
	printRow("  P99 First-Response Latency", fifoH.p99FirstRspLat, mlfqH.p99FirstRspLat, "t")
	printRow("  Avg Time Between Tokens (TBT)", fifoH.avgTBT, mlfqH.avgTBT, "t")
	fmt.Fprintln(out, hdr)
	fmt.Fprintln(out)
}
