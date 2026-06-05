# Day 2: OpenMP Shared Memory Programming & Parallel GEMM

Today we successfully parallelized General Matrix Multiplication (GEMM) using OpenMP, achieving massive speedups on the Apple M1 multithreaded architecture.

## 1. Benchmark Execution Results

*   **Matrix Dimensions**: 512 x 512 double precision (~134 million operations)
*   **Target Core Count**: 8 Threads (Max capacity of Apple M1)

| Implementation | Execution Time | Speedup | Efficiency |
| :--- | :--- | :--- | :--- |
| Sequential Baseline | **186.37 ms** | 1.00x (ref) | - |
| Parallel OpenMP | **39.28 ms** | **4.74x** | 59.3% |

### Analysis:
- **Correctness Check**: Passed! Output values of the parallel matrix matched the sequential baseline exactly.
- **Speedup Insights**: Spawning 8 threads yielded a speedup of **4.74x**. While theoretical max speedup is 8.0x, in practice, scheduling overhead, thread context switching, and shared memory bandwidth limitations (as all 8 cores compete to fetch matrix data from the L2/L3 cache and main memory) degrade linear scaling. This is a common performance bottleneck in memory-intensive HPC operations.

---

## 2. OpenMP Directives Under the Hood

You implemented the parallel GEMM outer-loop parallelization using:
```cpp
#pragma omp parallel for shared(A, B, C, M, N, K) private(i, j, k)
```

### Breakdown:
1.  `#pragma omp parallel`: Spawns a team of worker threads (defaulting to the number of logical CPU cores).
2.  `for`: Splits the iterations of the immediate next loop (the `i` loop) across the active threads.
3.  `shared(A, B, C, M, N, K)`: Tells the compiler that these matrices and dimension size integers reside in shared memory space. Only one copy exists in RAM, and all threads read/write directly to it.
4.  `private(i, j, k)`: Ensures that each thread has its own local stack variables for the loop counters. If `j` and `k` were shared, one thread would overwrite the loop progress of another, causing race conditions and catastrophic mathematical errors.

---

## 3. Tomorrow's Target (Day 3): MPI Distributed Memory Programming

Tomorrow, we step away from **Shared Memory** (where all cores share a single RAM block) to **Distributed Memory** (using MPI — Message Passing Interface).
- This is the standard programming model for large computing clusters and multi-node GPU clusters (like those used to train LLMs across thousands of GPUs).
- You will learn how to compile and run multi-process code where processes have isolated memory spaces and must communicate explicitly by sending and receiving messages.
