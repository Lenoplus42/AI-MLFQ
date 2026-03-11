package main

import (
	"fmt"
	"time"

	"github.com/Lenoplus42/AI-MLFQ/aischeduler"
)

func main() {
	cfg := aischeduler.SimConfig{
		NumTasks:         5000,
		InteractiveRatio: 0.85,
		InteractiveToks:  [2]int{10, 50},
		HeavyToks:        [2]int{10000, 30000},

		ArrivalIntervalTicks: 5,

		MaxKVCacheTokens:          200000,
		ContextSwitchPenaltyTicks: 15,

		ReportFilepath: "benchmark_report/benchmark_report_h100_disaster.txt",
	}

	fmt.Println("Booting AIOS MLFQ Benchmark Simulator...")
	fmt.Printf(
		"Generating %d Agent Tasks (%.0f%% Interactive, %.0f%% Heavy Batch)...\n",
		cfg.NumTasks, cfg.InteractiveRatio*100, (1-cfg.InteractiveRatio)*100,
	)

	start := time.Now()
	fifoWorkload, mlfqWorkload := aischeduler.GenerateWorkload(cfg)
	fmt.Printf("Workload generated in %v. Deep copies ready.\n\n", time.Since(start))

	fmt.Println("⏳ [1/2] Running Baseline: Single-Level FIFO Queue...")
	fifoScheduler := aischeduler.NewMLFQScheduler([]int{-1})
	fifoResult := aischeduler.RunSimulation("FIFO", fifoScheduler, fifoWorkload, cfg)

	fmt.Println("⚡ [2/2] Running Challenger: 3-Level MLFQ (Q0=10, Q1=50, Q2=FCFS)...")
	mlfqScheduler := aischeduler.NewMLFQScheduler([]int{10, 50, -1})
	mlfqResult := aischeduler.RunSimulation("MLFQ", mlfqScheduler, mlfqWorkload, cfg)

	aischeduler.PrintBenchmarkReport([2]aischeduler.SimResult{fifoResult, mlfqResult}, cfg)

	if cfg.ReportFilepath != "" {
		fmt.Printf("Report saved to: %s\n", cfg.ReportFilepath)
	}
}
