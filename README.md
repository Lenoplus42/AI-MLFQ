# AI-LMFQ: MLFQ-based LLM Scheduler
## Motivation
In concurrent multi-agent environments, treating an LLM API call as an indivisible, blocking request leads to catastrophic **Head-of-Line (HoL) blocking**. When compute-heavy background tasks (e.g., large codebase analysis or web scraping) monopolize the inference engine, latency-sensitive interactive agents suffer from severe P99 tail latency spikes. 

Standard FIFO or unmanaged concurrent dispatchers fail to isolate these heterogeneous workloads.

## Architecture & Feasibility 
AI-MLFQ tries to bring OS-level CPU scheduling primitives to LLM resource management. 

Instead of routing opaque HTTP requests, this scheduler operates on the assumption of modern inference backends (e.g., vLLM) that support **Continuous Batching** and **KV Cache Paging**. We abstract a single LLM forward pass (generating "one" token) as an atomic compute quantum. 

By implementing a **Multi-Level Feedback Queue (MLFQ)**:
1. **Preemption:** Long-running generation tasks are preempted once they exhaust their high-priority time quantum and are demoted to lower-priority queues.
2. **Zero-Waste Context Switching:** Because modern inference engines retain or page out the KV cache, preempted tasks can be suspended and resumed later without redundant prefill recomputation.
3. **Latency Isolation:** Interactive agents remain in the top-tier queue, receiving immediate compute resources, while batch-processing agents utilize system idle cycles.

This is an ongoing project building by Lennox, purely for fun.
