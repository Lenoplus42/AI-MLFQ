# 📄 AIOS-MLFQ: Technical Design Document

**Title:** AIOS-MLFQ
**Author:** Lennox (Siyuan Fu)
**Status:** MVP Validated (Phase 1) towards FIFO tail latency
**Date:** March 2026

## 1. Abstract

The "Head-of-Line Blocking" problem in LLM inference occurs when long-context batch tasks monopolize the GPU's Decode phase, causing interactive agent latency to skyrocket. This project demonstrates a memory-aware Multi-Level Feedback Queue (MLFQ) scheduler that exploits **Compute-I/O Overlap** to achieve a **99.8% reduction in P99 tail latency** for interactive workloads, with zero degradation to total system throughput.

## 2. The Physical Mapping (Bridging Go to Hardware)

To ensure this simulation accurately reflects Data Center physics, we define the following constants:

- **The Atomic Unit (1 Tick):** Corresponds to one Autoregressive Decode Step (generating exactly 1 token). In modern hardware, this takes roughly **15ms**.
- **VRAM Capacity (`MaxKVCacheTokens`):** Represents the HBM (High Bandwidth Memory) limit after model weights are loaded. For an **NVIDIA H100 (80GB)** running an 8B model, we allocate **200,000 tokens** for the KV Cache pool.
- **The PCIe Penalty:** * Micro-switches (SRAM ↔ HBM) take nanoseconds and are treated as 0 Ticks.
    - Macro-switches (HBM ↔ PCIe ↔ Host RAM) are the true bottleneck. Moving a 20k token context (~6GB) over a **PCIe 5.0 x16** bus takes ~200ms. We simulate this as `ContextSwitchPenaltyTicks = 15`.

## 3. Architecture & Contracts

The system strictly decouples the scheduling logic from the physical simulation.

### 3.1 The Scheduler (`MLFQScheduler`)

Acts purely as a state machine. It handles priority demotion but remains "blind" to physical memory constraints.

```go
// Supervisor interface defines the core contract for our MLFQ scheduler.
type Supervisor interface {
	SubmitTask(task *AgentTask) // Entry point for new agents
	Tick()                      // The global timer interrupt pulse
	PromoteAll()                // Future: Starvation prevention
}
```

### 3.2 The Simulator Engine

Acts as the "Motherboard." It observes VRAM footprint at every tick and enforces physical limits.

```go
// SimConfig is the single source of truth for the physical simulation.
type SimConfig struct {
	NumTasks                  int
	InteractiveRatio          float64
	InteractiveToks           [2]int 
	HeavyToks                 [2]int 
	ArrivalIntervalTicks      int 
	MaxKVCacheTokens          int 
	ContextSwitchPenaltyTicks int 
}
```

### 3.3 The Eviction Policy (Phase 5)

When `currentVRAMTokens > MaxKVCacheTokens`, the simulator triggers a Page Out. It sorts active tasks by `PriorityLevel` descending, ruthlessly sacrificing heavy tasks in the lowest priority queues (Q2/Q1) to protect the interactive tasks in Q0.

- Evicted tasks enter `StateBlocked` and receive a `penaltyRemaining` countdown.
- **Compute-I/O Overlap:** While evicted tasks serve their PCIe penalty, the global `Tick` continues, allowing the GPU's SMs to instantly process the next Q0 task.

## 4. Benchmark Results (The H100 Disaster Scenario)

### **4.1 Key Metrics Defined**

We evaluate system performance through two adversarial metrics.

**P99 First-Response Latency (Time-To-First-Token or TTFT)** measures the delay experienced by the slowest 1% of tasks before generating their first word. In interactive AI, a near-zero TTFT is the ultimate UX baseline; it prevents users from assuming the system has crashed. Conversely,

**Average Turnaround Time** measures the total lifespan of a task from submission to the final token. We use Turnaround as our "cost" metric to transparently show how much heavy-batch tasks are delayed to subsidize the instant responses of interactive agents.

**Setup:** A flash crowd of 5,000 concurrent agents (85% interactive, 15% long-context 10k-30k tokens) flooding the system, intentionally triggering severe VRAM thrashing.

- **The Interactive Rescue (-99.8% P99 Latency):** The P99 time-to-first-token for interactive tasks plummeted from a catastrophic 14,727,545 ticks (FIFO) to just **24,730 ticks** (MLFQ).
- **Robin Hood Economics (+1.1% Turnaround):** The total turnaround time for massive 30k-token batch tasks increased by a mere 1.1%. A mathematically sound trade-off to subsidize immediate interactive responses.
- **Conservation of Compute:** The total ticks to drain the entire workload (14,954,224) was identical for both FIFO and MLFQ. This proves the asynchronous DMA simulation: the GPU was kept at 100% utilization.

## 5. Critical Real-World Limitations

1. **PCIe Bus Contention:** Our DES assumes a flat 15-tick penalty. In a real cluster, simultaneous evictions saturate PCIe bandwidth, causing exponential latency spikes.
2. **The Prefill Phase:** We only simulate the autoregressive Decode phase. Real LLM serving includes a compute-bound Prefill phase that complicates predictable time-quantums.
3. **The "Stuttering" Blindspot (Time Between Tokens):** > While our P99 TTFT metric successfully proves that interactive tasks *start* immediately, it masks potential generation stuttering. If an interactive task receives its first token in Q0 but is subsequently evicted to Host RAM due to VRAM overflow, the delay before the second token (Time Between Tokens, or TBT) will spike catastrophically. Our current DES tracks initial and final timestamps but abstracts away TBT, which remains a critical, unmeasured UX bottleneck in this MVP.

## 6. Future Work

- **Priority Boost (`PromoteAll`):** Implement a global starvation-prevention mechanism.
- **Watermark-Aware Eviction:** Defer PCIe Page Outs until VRAM hits a critical watermark (e.g., 90%), allowing heavy batch tasks to "freeload" during low-traffic periods.
