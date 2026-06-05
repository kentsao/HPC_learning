# Day 3: MPI Distributed Memory Programming & Pi Estimator

Today we implemented a distributed Monte Carlo Pi estimator using MPI (Message Passing Interface) and ran it with multiple parallel processes.

## 1. Benchmark Execution Results

*   **Total Samples (Throws)**: 100,000,000 (100 Million)
*   **Target Architectures**: 1 Process vs. 4 Processes (running locally)

| Configuration | Samples/Process | Execution Time | Speedup | Parallel Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| 1 Process | 100,000,000 | **1.023 seconds** | 1.00x (ref) | - |
| 4 Processes | 25,000,000 | **0.272 seconds** | **3.76x** | **94.0%** |

### Analysis:
- **Correctness Check**: Passed! The estimate of $\pi$ converged with a tiny absolute error, and the results aggregated correctly on Rank 0.
- **Speedup Insights**: Spawning 4 processes yielded a massive **3.76x speedup**, equivalent to **94% parallel efficiency**. Because Monte Carlo simulation is an "embarrassingly parallel" problem (processes don't need to coordinate or share data during the heavy calculation loop), the scaling is nearly linear. The slight drop to 94% is due to standard process spawning overhead and the final collective `MPI_Reduce` communication.

---

## 2. MPI vs. OpenMP: Key Architectural Differences

| Dimension | OpenMP (Day 2) | MPI (Day 3) |
| :--- | :--- | :--- |
| **Model** | Shared Memory | Distributed Memory (Shared Nothing) |
| **Concurrency unit** | Thread (lightweight, runs inside same process) | Process (heavyweight, completely isolated memory) |
| **Memory Access** | Direct access to all variables (shared scope) | Explicit message passing via networks / buffers |
| **Scale Limit** | Single physical machine (limited by CPU cores) | Virtually unlimited (thousands of server nodes) |
| **MLSys Application** | Single-GPU thread pooling / CPU preprocessing | Multi-node GPU clusters (PyTorch DDP, DeepSpeed) |

---

## 3. Tomorrow's Target (Day 4): Cache Locality & Cache Tiling (Loop Blocking)

Tomorrow, we return to single-thread optimization, but we look at how C++ compilers and CPU architectures interact at the hardware level.
- You will learn about **L1/L2/L3 Cache lines** and how memory is fetched.
- We will look at why a standard matrix multiplication suffers from massive cache misses (since matrices are stored in memory row-by-row, but we access matrix B column-by-column).
- You will implement **Loop Tiling (Blocking)** to ensure data remains inside the ultra-fast L1/L2 cache during matrix multiplication.
