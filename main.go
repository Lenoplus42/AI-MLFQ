package main

import (
	"fmt"
	"time"
	"github.com/Lenoplus42/AI-MLFQ/aischeduler"
)

func main() {
	fmt.Println("Booting AIOS MLFQ Benchmark Simulator...")
	
	numTasks := 1000
	fmt.Printf("Generating %d concurrent Agent Tasks (80%% Interactive, 20%% Heavy Batch)...\n", numTasks)
	
	start := time.Now()
	fifoWorkload, mlfqWorkload := aischeduler.GenerateWorkload(numTasks)
	fmt.Printf("Workload generated in %v. Deep copies ready.\n\n", time.Since(start))

	// Running Baseline: Single-Level FIFO Queue
	fmt.Println("⏳ [1/2] Running Baseline: Single-Level FIFO Queue...")
	// Pass -1 to enforce downgrading into FIFO
	fifoScheduler := aischeduler.NewMLFQScheduler([]int{-1})
	fifoResult := aischeduler.RunSimulation("FIFO", fifoScheduler, fifoWorkload)

	// Running Challenger: 3-Level MLFQ (Q0=10, Q1=50, Q2=FCFS)
	fmt.Println("⚡ [2/2] Running Challenger: 3-Level MLFQ (Q0=10, Q1=50, Q2=FCFS)...")
	// Pass [10, 50, -1] to activate multilevel priority feature
	mlfqScheduler := aischeduler.NewMLFQScheduler([]int{10, 50, -1})
	mlfqResult := aischeduler.RunSimulation("MLFQ", mlfqScheduler, mlfqWorkload)

	// The ultimate dual
	aischeduler.PrintBenchmarkReport([2]aischeduler.SimResult{fifoResult, mlfqResult})
}