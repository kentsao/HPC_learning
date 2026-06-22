# Day 4: Cache Locality, Loop Reordering & Loop Tiling

Today we optimized sequential General Matrix Multiplication (GEMM) by adjusting memory access patterns to align with physical CPU cache hierarchies. We achieved an **8x performance improvement** without any multi-threading!

## 1. Benchmark Execution Results

*   **Matrix Dimensions**: 1024 x 1024 double precision (~2.14 billion operations)
*   **System Cache Size**: Apple M1 has a large 128KB L1 cache and 12MB L2 cache.

| Implementation | Execution Time | Throughput | Speedup | Cache Locality |
| :--- | :--- | :--- | :--- | :--- |
| Baseline IJK | **1.419 seconds** | 1.51 GFLOPs | 1.00x (ref) | Poor (Strided access on B) |
| Optimized IKJ | **0.176 seconds** | **12.19 GFLOPs** | **8.06x** | High (Contiguous row access) |
| Tiled GEMM (Block=64) | **0.177 seconds** | **12.13 GFLOPs** | **8.02x** | High (Cache-sized sub-blocks) |

### Analysis:
- **Correctness Check**: Passed! Both optimized versions matched the baseline output exactly.
- **The Loop Reordering (IKJ) Magic**: In standard `IJK`, the innermost loop accesses $B(k, j)$ along a column. Because matrices are stored row-by-row in RAM, jumping down columns triggers a cache miss on almost every loop iteration. In `IKJ`, the innermost loop accesses $B(k, j)$ along a row, which is contiguous. The CPU loads 8 double-precision numbers in a single cache line (64 bytes), resulting in 7 cache hits for every 1 cache miss!
- **IKJ vs. Tiling at N=1024**: At $1024 \times 1024$, the total memory footprint for the inputs is about 16MB. Since the M1 Mac has a massive, high-bandwidth 12MB/16MB shared L2 cache, the matrices almost fit inside the hardware caches. Therefore, IKJ streaming is already near-optimal. If we scale to $SIZE \ge 2048$ (exceeding cache sizes), Tiled GEMM would begin to outperform IKJ because it prevents tiles from being evicted to main RAM.

---

## 2. Hardware Insights for MLSys
In deep learning systems:
1.  **Memory Bound vs. Compute Bound**: Standard GEMM is naturally compute-bound if caches are utilized, but becomes memory-bound if cache misses are frequent. By optimizing cache hits, we increased arithmetic intensity from 1.5 GFLOPs to 12 GFLOPs.
2.  **GPU Shared Memory**: In Week 3, when you write CUDA/Triton kernels, you will see that **Triton uses the exact same concept of Loop Tiling**. Triton blocks threads into SRAM tiles to avoid reading/writing to global GPU High Bandwidth Memory (HBM)! Today's lesson is the mathematical foundation of FlashAttention and GPU programming.

---

## 3. Tomorrow's Target (Day 5): SIMD Vectorization (Neon Intrinsics)

Tomorrow, we squeeze the absolute maximum performance out of a single CPU core.
- You will learn about **SIMD (Single Instruction Multiple Data)**.
- You will learn how the Apple M1's **Neon vector engine** can execute calculations on multiple elements simultaneously.
- You will write custom Neon vector intrinsics in C++ to compute our tiled matrix multiplication.
