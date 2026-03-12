# ⚡ AI-MLFQ: Memory-Aware LLM Scheduler

[![Go Report Card](https://goreportcard.com/badge/github.com/Lenoplus42/AI-MLFQ)](https://goreportcard.com/report/github.com/Lenoplus42/AI-MLFQ)

*A discrete-event simulator proving that OS-level MLFQ scheduling can obliterate LLM tail latency in multi-agent environments.*

## 📊 The Benchmark (H100 Disaster Simulation)
We simulated a flash crowd of 5,000 concurrent agents (85% short interactive chats, 15% massive 30k-token RAG batches) hitting a strictly constrained 200k-token VRAM budget. 

By aggressively preempting heavy tasks and simulating realistic PCIe 4.0/5.0 Page-Out penalties, **Interactive P99 Time-to-First-Token (TTFT) dropped by 99.8%** with zero loss to overall system throughput.

~~~text
════════════════════════════════════════════════════════════════════════
  METRIC                                          FIFO          MLFQ
════════════════════════════════════════════════════════════════════════
  Total Ticks to Drain                        14954224      14954224
────────────────────────────────────────────────────────────────────────
  INTERACTIVE TASKS (tokens 10–50)  
    Count                                         4250          4250
    Avg Turnaround Time                    7144425.36t       96281.74t  (-98.7%)
    P99 First-Response Latency            14727545.00t       24730.00t  (-99.8%)
────────────────────────────────────────────────────────────────────────
  HEAVY BATCH TASKS (tokens 10000–30000)
    Count                                          750           750
    Avg Turnaround Time                    7377582.64t     7462272.47t  (+1.1%)
    P99 First-Response Latency            14740188.00t       24850.00t  (-99.8%)
════════════════════════════════════════════════════════════════════════
~~~

## 🧠 Motivation
In concurrent multi-agent environments, treating an LLM API call as an indivisible, blocking request leads to catastrophic **Head-of-Line (HoL) blocking**. When compute-heavy background tasks monopolize the inference engine, latency-sensitive interactive agents suffer from severe P99 tail latency spikes. Standard HTTP-level FIFO dispatchers fail entirely at isolating these bimodal workloads.

## 🏗️ Architecture & Feasibility
AI-MLFQ brings OS-level CPU scheduling primitives to LLM resource management. Assuming modern inference backends (e.g., vLLM) that support *Continuous Batching* and *Paged KV Cache*, we abstract a single LLM forward pass (generating "one" token) as an atomic compute quantum.

* **Preemption:** Long-running generation tasks are preempted once they exhaust their high-priority time quantum.
* **Memory-Aware Eviction:** Unlike traditional OS threads where context switching is cheap (SRAM $\rightarrow$ HBM), LLM context switching requires massive PCIe transfers (HBM $\leftrightarrow$ Host DRAM). Our simulator actively tracks active KV Cache tokens and enforces PCIe I/O penalties when VRAM bounds are breached.
* **Compute-I/O Overlap:** While evicted tasks serve their PCIe penalty, the GPU's Streaming Multiprocessors (SMs) immediately execute the next interactive task, keeping utilization at 100%.

> **📖 Deep Dive:** For a rigorous breakdown of how our variables map to real-world hardware (Tick duration, H100 VRAM limits, and PCIe bandwidth maths), read the [ARCHITECTURE.md](ARCHITECTURE.md).

## 🛠️ Usage
To run the simulator and generate the benchmark report yourself:

~~~bash
go run main.go
~~~
