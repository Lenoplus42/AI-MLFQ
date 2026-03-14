# 📄 AIOS-MLFQ: Technical Design Document

**Title:** AIOS-MLFQ
**Author:** Lennox (Siyuan Fu)
**Status:** MVP Validated (Phase 1) towards FIFO tail latency
**Date:** March 2026

## 1. Abstract

The "Head-of-Line Blocking" problem in LLM inference occurs when long-context batch tasks monopolize the GPU's Decode phase, causing interactive agent latency to skyrocket. This project demonstrates a memory-aware Multi-Level Feedback Queue (MLFQ) scheduler that exploits **Compute-I/O Overlap** to achieve a **99.8% reduction in P99 tail latency** for interactive workloads, with zero degradation to total system throughput.

## 2. The Physical Mapping (Bridging Go to Hardware)

To ensure this simulation accurately reflects Data Center physics, we define the following constants:

* **The Atomic Unit (1 Tick)**: Represents the minimal non-preemptible compute window. We define 1 Tick $\approx$ 15ms. In this window, the SMs process either a Chunked Prefill block (512 tokens) or a single Autoregressive Decode step (1 token).
* **VRAM Capacity (MaxKVCacheTokens)**: For an 8B model (using GQA， FP16), a single KV cache token consumes roughly 128 KB. We strictly constrain the available KV cache pool to 200,000 tokens ($\approx$ 25.6 GB). This represents a highly congested multi-tenant VRAM partition, intentionally forcing the system to rely heavily on PCIe swapping.
* **The PCIe Penalty**: Moving a 20k token context over a PCIe 5.0 x16 bus takes time. We simulate this macro-switch (HBM ↔ Host DRAM) as ContextSwitchPenaltyTicks = 15.
* **Poisson Arrivals ($\lambda = 0.02$)**: We model traffic using a true Poisson process. $\lambda = 0.02$ means, on average, 1 new request arrives every 50 ticks (750ms). 

**Understanding the Disaster Load ($\rho$):** 
Our system takes ~718k ticks to drain 5000 tasks, meaning the physical service rate ($\mu$) is 1 task per 143 ticks. With an arrival rate of 1 task per 50 ticks, the Traffic Intensity ($\rho = \lambda / \mu$) is roughly 2.8. The system is receiving 280% more traffic than it can physically process, creating a brutal bottleneck that perfectly exposes the scheduler's behavior.

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

We evaluate system performance through three adversarial metrics:
* P99 First-Response Latency (Time-To-First-Token or TTFT): The delay experienced by the slowest 1% of tasks before generating their first word. 
* Average Time Between Tokens (TBT): Measures generation stuttering. Calculated as the time from the first decode token to the final token, divided by the decode length.
* Average Turnaround Time: The total lifespan of a task. Used as our "cost" metric.

Setup: A $\lambda = 0.02$ Poisson arrival disaster simulation (average 1 task per 750ms) with 5,000 agents, intentionally triggering severe VRAM thrashing.

* The Interactive Rescue (-100% P99 TTFT): Interactive TTFT plummeted from a catastrophic ~462k ticks (FIFO) to just 16 ticks (MLFQ). The AI feels completely instantaneous.
* The TBT Trade-off (Human-Perceptible vs. Machine Wait): Because Q0 tasks are time-sharing, MLFQ Interactive TBT increased from 1.00t to 3.89t ($\approx$ 58ms per token). At ~17 tokens per second, this is still incredibly fluent for human reading. However, the Heavy Batch tasks absorbed the true physics tax, with their TBT skyrocketing to 492 ticks ($\approx$ 7.3 seconds between tokens) as they were constantly evicted to Host RAM.
* Conservation of Compute: Total ticks to drain (718,323) was identical for both policies, proving our Compute-I/O Overlap keeps the GPU at 100% utilization despite the massive context switching.

## 5. Critical Real-World Limitations

1. **PCIe Bus Contention:** Our DES assumes a flat 15-tick penalty. In a real cluster, simultaneous evictions saturate PCIe bandwidth, causing exponential latency spikes.
2. **The Prefill Phase:** We only simulate the autoregressive Decode phase. Real LLM serving includes a compute-bound Prefill phase that complicates predictable time-quantums.
3. **The "Stuttering" Blindspot (Time Between Tokens):** > While our P99 TTFT metric successfully proves that interactive tasks *start* immediately, it masks potential generation stuttering. If an interactive task receives its first token in Q0 but is subsequently evicted to Host RAM due to VRAM overflow, the delay before the second token (Time Between Tokens, or TBT) will spike catastrophically. Our current DES tracks initial and final timestamps but abstracts away TBT, which remains a critical, unmeasured UX bottleneck in this MVP.

## 6. Future Work

- **Priority Boost (`PromoteAll`):** Implement a global starvation-prevention mechanism.
- **Watermark-Aware Eviction:** Defer PCIe Page Outs until VRAM hits a critical watermark (e.g., 90%), allowing heavy batch tasks to "freeload" during low-traffic periods.
