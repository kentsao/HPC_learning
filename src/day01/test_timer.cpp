#include "../utils/timer.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// ============================================================================
// C++ LESSON FOR PYTHON DEVELOPERS: Pass-by-Value vs. Pass-by-Reference
// ============================================================================
// In Python, everything is passed by object reference (which behaves like a pointer).
// In C++, objects are passed by VALUE (copied) by default!
// For high-performance computing, copying large datasets is a critical bottleneck.
//
// Below, we define three functions to demonstrate this core concept.

// 1. Pass-by-value: This copies the ENTIRE vector of numbers (extremely slow!).
double sum_by_value(std::vector<double> vec) {
    double total = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        total += vec[i];
    }
    return total;
}

// 2. Pass-by-const-reference: This passes a read-only reference (address) without copying.
// The `&` denotes reference, and `const` guarantees we won't modify the original data.
// In HPC, this is the standard way to pass large matrices, tensors, and datasets!
double sum_by_reference(const std::vector<double>& vec) {
    double total = 0.0;
    // Modern C++ range-based for loop (similar to "for val in vec:" in Python)
    for (double val : vec) {
        total += val;
    }
    return total;
}

// 3. Pass-by-pointer: Passes the memory address explicitly using `*`.
// We access methods or elements using the arrow operator `->` or dereferencing `*`.
double sum_by_pointer(const std::vector<double>* vec_ptr) {
    // Check if the pointer is null to avoid segmentation faults (unauthorized memory access)
    if (vec_ptr == nullptr) return 0.0;
    
    double total = 0.0;
    // vec_ptr->size() is shorthand for (*vec_ptr).size()
    for (size_t i = 0; i < vec_ptr->size(); ++i) {
        total += (*vec_ptr)[i]; // Dereference and access index
    }
    return total;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "      HPC Day 1: C++ High-Resolution Timer & Pointers    " << std::endl;
    std::cout << "=========================================================" << std::endl;

    // 1. Array allocation (std::vector is a dynamic array, like Python list)
    // Allocating 50 million elements (about 400 MB in memory)
    const size_t NUM_ELEMENTS = 50000000;
    std::cout << "[1/4] Allocating " << NUM_ELEMENTS << " doubles (approx. 400MB)..." << std::endl;
    
    // Benchmarking allocation
    hpc::Timer timer(true); // Create and start timer immediately
    std::vector<double> data(NUM_ELEMENTS, 1.23);
    timer.stop();
    timer.print("Allocation Time");

    // 2. Benchmark Pass-by-Value (Copying 400MB of data!)
    std::cout << "\n[2/4] Benchmarking Pass-by-Value (Copying data)..." << std::endl;
    timer.start();
    double total_val = sum_by_value(data);
    timer.stop();
    std::cout << "Result (Value): " << total_val << std::endl;
    timer.print("Pass-by-Value Time");

    // 3. Benchmark Pass-by-Const-Reference (Passing address, NO COPYING!)
    std::cout << "\n[3/4] Benchmarking Pass-by-Reference (Zero copy)..." << std::endl;
    timer.start();
    double total_ref = sum_by_reference(data);
    timer.stop();
    std::cout << "Result (Reference): " << total_ref << std::endl;
    timer.print("Pass-by-Reference Time");

    // 4. Benchmark Pass-by-Pointer (Explicit address passing)
    std::cout << "\n[4/4] Benchmarking Pass-by-Pointer (Zero copy, pointer access)..." << std::endl;
    timer.start();
    // The `&` operator gets the memory address of the object (pointer)
    double total_ptr = sum_by_pointer(&data);
    timer.stop();
    std::cout << "Result (Pointer): " << total_ptr << std::endl;
    timer.print("Pass-by-Pointer Time");

    std::cout << "\n=========================================================" << std::endl;
    std::cout << "Key takeaway: Pass-by-Reference/Pointer avoids overhead of copying memory." << std::endl;
    std::cout << "This is fundamental in HPC where data movements are highly expensive!" << std::endl;
    std::cout << "=========================================================" << std::endl;

    return 0;
}
