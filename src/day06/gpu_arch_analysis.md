# Day 6: GPU Hardware Architecture & Latency Hiding Analysis

Welcome to the start of **Week 2: GPU Architecture & Deep Learning Systems**! Today is a theoretical research and documentation day. Understanding how GPU hardware functions is critical before writing custom CUDA or Triton kernels, because GPU programming is heavily constrained by physical chip physics.

Your challenge today is to read the resources below, research the core concepts, and **fill in the answers to the 3 questions at the bottom of this file**.

---

## 1. 📚 Required Reading List

Take 15–20 minutes to read through these highly-regarded articles:
1.  **[How GPUs Work (NVIDIA Developer Blog)](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)**: Introduces the basic CUDA programming model, Grids, Blocks, Threads, and how they map to hardware.
2.  **[GPU Architecture: A Developer's Perspective](https://fabiensanglard.net/cuda/)**: A fantastic, visual deep-dive explaining Streaming Multiprocessors (SMs), Warps, and memory hierarchy.
3.  **[Warp Divergence & Execution (GeeksforGeeks)](https://www.geeksforgeeks.org/cuda-warp-divergence/)**: Explains why `if-else` branches can severely degrade GPU performance.

---

## 2. 🧠 Core Theoretical Concepts

*   **Latency-Optimized (CPU) vs. Throughput-Optimized (GPU)**: 
    *   A CPU is designed to execute a single thread of instructions as fast as possible (minimizing latency) using huge caches, aggressive out-of-order execution, and branch prediction.
    *   A GPU is designed to execute millions of threads simultaneously (maximizing throughput) using massive arrays of ALUs (Arithmetic Logic Units) and simple instruction decoders.
*   **SIMT (Single Instruction, Multiple Threads)**: The execution model of GPUs. Threads are grouped into physical units of **32 threads called a Warp**. All 32 threads in a warp execute the exact same instruction at the exact same clock cycle, but on different data lanes.
*   **Latency Hiding**: While a CPU uses large caches to avoid memory stalls, a GPU has virtually no out-of-order execution logic. Instead, if a warp stalls waiting for memory to load from global VRAM, the hardware scheduler instantly context-switches to another warp that is ready to compute. Since switching warps takes **zero clock cycles** (all warp states are stored in registers on the chip), GPUs hide latency through massive thread concurrency.

---

## ✍️ Your Assignment: Research & Document

Please edit this file (or write your answers directly in your response) to answer the following three questions based on your readings:

### Question 1: CPU vs. GPU Architecture Comparison
Fill in the comparison table below:

| Feature | CPU | GPU |
| :--- | :--- | :--- |
| **Primary Design Goal** | Minimize Latency (Run 1 thread fast) | Maximize Throughput (Run millions of threads) |
| **Core Count** | 4-16 cores | thousands of cores |
| **ALU vs. Cache/Control Ratio** | High cache/control, low ALU | Low cache/control, high ALU |
| **Memory Latency Hiding Method** | Large caches, Prefetching | Massive multithreading warp switching |

### Question 2: Warp Divergence
*   **Define Warp Divergence in your own words**:
    When encounter the divergence, the warp will split into two groups, and each group will execute the instruction in parallel. But the performance will degrade because the warp will wait for the other group to finish. In other words, the warp will execute the instruction in serial.
*   **Why does a simple `if-else` condition inside a GPU kernel cause warp divergence, and what is its performance impact?**:
    As I mentioned, the warp will split into two groups and execute the instruction in parallel. But the performance will degrade because the warp will wait for the other group to finish. In other words, the warp will execute the instruction in serial. It is the same as the CPU's branch prediction but it is more worse.

### Question 3: GPU Memory Hierarchy & Bandwidth
*   **Fill in the typical bandwidth/latency characteristics of the hierarchy below**:
    1.  **Registers**: Stored frequently used variables and data. Low latency, high bandwidth.
    2.  **Shared Memory (SRAM)**: Extremely high bandwidth, low latency. Used to store data that is frequently accessed by threads in a warp. 
    3.  **Global Memory (HBM/GDDR)**: The data is loaded from the global memory to the shared memory and then used by the threads in a warp. High latency, low bandwidth.
*   **Why do specialized deep learning kernels (like Triton or FlashAttention) focus heavily on tiling data from Global Memory into Shared Memory?**:
    Because the global memory is slow, so we need to load the data from the global memory to the shared memory and then use the data in the shared memory. 
