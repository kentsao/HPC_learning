# Day 5: SIMD Vectorization using ARM Neon Intrinsics

Today we vectorized our General Matrix Multiplication (GEMM) using **ARM Neon Intrinsics**, processing 4 floating-point numbers simultaneously in 128-bit vector registers. We achieved a **14x–16x speedup** over the baseline!

## 1. Benchmark Execution Results

*   **Matrix Dimensions**: 1024 x 1024 single precision (float) (~2.14 billion operations)
*   **Vector Engine**: ARM Neon (128-bit registers, 4 floats per vector)

| Implementation | Execution Time | Throughput | Speedup | Performance Note |
| :--- | :--- | :--- | :--- | :--- |
| Baseline IJK | **1365.45 ms** | 1.57 GFLOPs | 1.00x (ref) | Standard sequential |
| Optimized IKJ | **82.43 ms** | **26.05 GFLOPs** | **16.57x** | Compiler Autovectorized |
| Neon SIMD | **96.32 ms** | **22.30 GFLOPs** | **14.18x** | Hand-Written Intrinsics |

### Analysis:
- **Correctness Check**: Passed! The results of our Neon vectorized matrix multiplication matched the reference baseline exactly.
- **Why is `gemm_ikj` slightly faster than `gemm_neon`?**
  When compiling with `-O3`, the compiler (Clang 17) automatically performs **Autovectorization** on clean, stream-aligned loops like `gemm_ikj`. 
  In our hand-written `gemm_neon`, we write explicit memory loads and stores for matrix C in the innermost loop (`vld1q_f32` and `vst1q_f32`). 
  The compiler, however, is smarter: it optimizes register allocation, unrolls the loops (e.g., accumulating into multiple registers simultaneously to hide instruction latency), and keeps intermediate sums in registers longer without storing them back to RAM on every iteration.
- **The Core Lesson**: Modern compilers are incredibly powerful. In HPC, the general rule of thumb is: **first write clean, cache-friendly code (like IKJ) that the compiler can easily autovectorize**. Only write manual intrinsics or assembly when the compiler fails to vectorize or when optimizing highly specialized kernels.

---

## 2. 🌟 Week 1 Summary: HPC CPU Foundations Complete!

We have successfully finished your first week of HPC foundations! Here is what you have built and learned:

1.  **Day 1 (Toolchain & Memory Models)**: Set up the environment, git, and compilers. Learned about C++ memory layout (stack vs. heap) and why passing by value copies RAM, while passing by reference is zero-copy.
2.  **Day 2 (OpenMP Shared Memory)**: Leveraged M1's multiple cores to parallelize GEMM, achieving **4.7x speedup** on 8 threads. Learned about race conditions and private variable scoping.
3.  **Day 3 (MPI Distributed Memory)**: Spelled out processes with separate RAM architectures. Built a distributed Monte Carlo Pi estimator with **94% parallel efficiency** using `MPI_Reduce`.
4.  **Day 4 (Cache Locality & Tiling)**: Reordered loop structures to IKJ, aligning access with contiguous cache lines. Achieved **8x speedup** through spatial locality.
5.  **Day 5 (SIMD Vectorization)**: Utilized ARM Neon vector registers to perform operations on 4 elements in parallel, achieving **14x–16x speedup**.

---

## 3. 📂 What's Next? Week 2: GPU Architecture & MLSys Deep Learning Systems

Next week, we move from CPUs to **GPUs and Deep Learning Systems**!
*   **Day 6**: GPU Hardware Architecture (SMs, warps, threads, shared vs. global memory, latency hiding).
*   **Day 7**: Writing custom C++/MPS extensions for PyTorch.
*   **Day 8**: Writing a local PyTorch Distributed Data Parallel (DDP) training loop from scratch.
