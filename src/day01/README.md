# Day 1: High-Resolution Timing, Compilers, and C++ Memory Model

Welcome to Day 1 of your High-Performance Computing (HPC) and MLSys journey! Today, we laid down the foundational pillars: setting up the environment and understanding the core differences between Python and C++ when it comes to performance, compilation, and memory management.

## 1. C++ vs. Python: The Compilation Paradigm

In Python, the code is executed by an interpreter (the `python` program) line-by-line. This dynamic nature is great for rapid prototyping but slow for execution because the interpreter must inspect every variable's type and structure at runtime.

In **C++**, code is **statically compiled**:
- **Source Code (`.cpp`, `.hpp`)** is translated by a compiler (`clang++` or `g++`) into **machine-native binary instructions** beforehand.
- The compiler performs global analysis, optimizing loop structures, variable lifetimes, and instruction ordering before the code even runs.
- **Header Files (`.hpp` or `.h`)** contain declarations (interfaces), while **Source Files (`.cpp`)** contain definitions (implementations).

### How to Compile C++ Manually:
We compiled our test suite with:
```bash
clang++ -O3 -std=c++17 src/day01/test_timer.cpp -o src/day01/test_timer
```
- `-O3`: Enables high-level optimization. The compiler will apply aggressive parallelization heuristics, loop unrolling, and vectorization optimizations.
- `-std=c++17`: Specifies the C++17 standard to use modern language features.
- `-o src/day01/test_timer`: Specifies the output binary name.

---

## 2. The C++ Memory Model: Stack vs. Heap

In Python, everything is a dynamic object allocated on the **Heap**. Memory is automatically managed by a garbage collector.

In C++, you have fine-grained control over where variables live:

1. **Stack Memory**:
   - Extremely fast allocation and deallocation.
   - Automatically managed. When a function finishes, all variables declared on its stack are immediately destroyed.
   - **Limit**: Very small size (typically 8MB). Creating large arrays on the stack will cause a **Stack Overflow**.
   - Example: `double x = 3.14;` or `hpc::Timer timer;`

2. **Heap Memory**:
   - Large capacity (virtually all available RAM).
   - Manually managed. You allocate with `new` and must release it with `delete` to avoid **memory leaks**.
   - Slower allocation/deallocation overhead.
   - Example: `double* arr = new double[1000000]; ... delete[] arr;` or `std::vector<double> vec;` (the `std::vector` class manages heap allocation under the hood).

---

## 3. Pointers and References: The Foundations of Zero-Copy

Data movement is the primary bottleneck in HPC. Moving a 10 GB tensor across memory channels takes significantly more time than actually performing arithmetic on it. To build high-performance systems, we must master **Zero-Copy passing**.

Let's examine how our Day 1 benchmark results demonstrated this:

### The Three Modes of Passing Data:

### A. Pass-by-Value (Memory Copying)
```cpp
double sum_by_value(std::vector<double> vec) { ... }
```
- **What happens**: C++ copies the *entire* vector of 50 million doubles (~400 MB) into a new location in memory specifically for this function.
- **Performance**: **~154 ms**
- **HPC Verdict**: ❌ **Terrible.** We wasted CPU cycles copying data that we only wanted to read.

### B. Pass-by-Const-Reference (Standard HPC Read Pattern)
```cpp
double sum_by_reference(const std::vector<double>& vec) { ... }
```
- **What happens**: The `&` symbol denotes a **Reference**. Instead of copying the vector, we pass its *memory address*. The `const` keyword ensures the function cannot modify the data.
- **Performance**: **~48 ms (3x faster!)**
- **HPC Verdict**:  **Gold Standard.** Zero-copy reading of large datasets.

### C. Pass-by-Pointer (Explicit Address Passing)
```cpp
double sum_by_pointer(const std::vector<double>* vec_ptr) { ... }
```
- **What happens**: The `*` denotes a **Pointer** variable that stores the memory address of the vector. We must access it using the arrow `->` or dereference operator `*vec_ptr`.
- **Performance**: **~47 ms**
- **HPC Verdict**:  **Excellent.** Historically used in C. In modern C++, references are generally preferred for cleaner syntax, but pointers are essential when writing raw CUDA/GPU kernels where you work with raw device pointers (`float*`).

---

## 4. Today's Code Artifacts

- **[timer.hpp](file:///Users/kentsao/Desktop/HPC_learning/src/utils/timer.hpp)**: A clean, header-only high-resolution timer.
- **[test_timer.cpp](file:///Users/kentsao/Desktop/HPC_learning/src/day01/test_timer.cpp)**: The benchmark executable verifying stack/heap allocation and the speed differential of pass-by-value vs. reference.

---

## 5. What's Next? (Day 2)

Tomorrow, we dive into **OpenMP Shared Memory Programming**. You will learn how to take advantage of multiple CPU cores on your M1 Mac to run calculations in parallel, handle race conditions, and write your first multi-threaded algorithm!
