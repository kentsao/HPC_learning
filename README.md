# HPC & MLSys Learning Portfolio

Welcome to my portfolio designed for applying to a PhD in High-Performance Computing (HPC) and Machine Learning Systems (MLSys), specifically targeting top universities in Taiwan (NTU, NYCU, NTHU).

This repository serves as a tracked, day-by-day record of my theoretical learning, hands-on side projects, and open-source contributions. 

## 🛠️ Environment Configuration
All local CPU-based parallel algorithms are developed and benchmarked on an **Apple Silicon M1 Mac** (arm64, macOS).
- **Conda Environment**: `hpc_learn` (Python 3.10, PyTorch MPS, NumPy, Pytest, Jupyter)
- **Compiler Suite**: Apple Clang 17 (with Homebrew `libomp` for OpenMP and `open-mpi` for MPI)
- **GPU Kernel Acceleration**: OpenAI Triton kernels configured to run on **Google Colab GPUs**.

---

## 📅 Progress Tracker & Syllabus

### 📂 Week 1: Core HPC & CPU Foundations (Local M1 Mac)
- [x] **[Day 1: Env & C++ Timer Utility](file:///Users/kentsao/Desktop/HPC_learning/src/day01/README.md)**
  - Initialized Git, Conda env (`hpc_learn`), compilers, and Homebrew support.
  - Implemented high-resolution C++ Timer class: [timer.hpp](file:///Users/kentsao/Desktop/HPC_learning/src/utils/timer.hpp).
  - Benchmarked Pass-by-Value vs. Pass-by-Reference memory copy overhead: [test_timer.cpp](file:///Users/kentsao/Desktop/HPC_learning/src/day01/test_timer.cpp).
- [x] **[Day 2: OpenMP Shared Memory Programming](file:///Users/kentsao/Desktop/HPC_learning/src/day02/README.md)** (Multi-threaded matrix multiplication).
- [ ] **Day 3: MPI Distributed Memory Programming** (Message passing, Pi estimation).
- [ ] **Day 4: Cache Locality & Cache Tiling** (Loop blocking optimizations for GEMM).
- [ ] **Day 5: SIMD Neon Vectorization** (Hand-written assembly/intrinsics vectorization).

### 📂 Week 2: GPU Architecture & High-Performance Deep Learning Systems
- [ ] **Day 6: GPU Hardware Architecture** (SMs, warps, global vs. shared memory).
- [ ] **Day 7: PyTorch custom C++ / Metal Extensions**.
- [ ] **Day 8: PyTorch Distributed Data Parallel (DDP) Simulation** (All-reduce under the hood).
- [ ] **Day 9: Memory Optimization Foundations** (FSDP, DeepSpeed ZeRO 1/2/3, 3D Parallelism).
- [ ] **Day 10: Mixed Precision (FP16/BF16/FP8) & Activation Checkpointing**.

### 📂 Week 3: Kernel Optimization & Triton Programming (MLSys Core)
- [ ] **Day 11: Triton vector add kernel**.
- [ ] **Day 12: Triton high-performance GEMM kernel** (Block-level SRAM caching).
- [ ] **Day 13: FlashAttention Math & Computational Complexities** (IO Bottleneck Analysis).
- [ ] **Day 14: Implementing custom FlashAttention in Triton**.
- [ ] **Day 15: Weight-only model quantization (INT4/INT8 dequantization kernels)**.

### 📂 Week 4: Capstone Project & Open-Source Contributions
- [ ] **Day 16-17: Capstone Project** (Fused SwiGLU / RMSNorm Triton Kernel & Roofline profiling).
- [ ] **Day 18-20: Open-Source Contribution** (Bug fix / optimization for target repo).
- [ ] **Day 21-30: Portfolio Optimization, Blog Writing, & PhD Application Assembly**.
