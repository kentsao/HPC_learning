#include "../utils/timer.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
// Fake functions if OpenMP is not enabled (to allow compilation)
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif

// A simple matrix class using a flat std::vector for contiguous 1D memory
// allocation In HPC, we ALWAYS represent 2D matrices as a flat 1D array in
// memory to ensure cache-coherency and spatial locality.
class Matrix {
public:
  int rows;
  int cols;
  std::vector<double> data;

  Matrix(int r, int c, double val = 0.0) : rows(r), cols(c), data(r * c, val) {}

  // Operator to access elements using matrix(row, col) syntax
  inline double &operator()(int r, int c) { return data[r * cols + c]; }

  inline const double &operator()(int r, int c) const {
    return data[r * cols + c];
  }
};

// Initialize matrix with random values between 0.0 and 1.0
void initialize_matrix(Matrix &mat) {
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<> dis(0.0, 1.0);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      mat(i, j) = dis(gen);
    }
  }
}

// Helper to verify if two matrices are approximately equal
bool verify_results(const Matrix &C_ref, const Matrix &C_test,
                    double epsilon = 1e-6) {
  if (C_ref.rows != C_test.rows || C_ref.cols != C_test.cols)
    return false;
  for (int i = 0; i < C_ref.rows; ++i) {
    for (int j = 0; j < C_ref.cols; ++j) {
      if (std::abs(C_ref(i, j) - C_test(i, j)) > epsilon) {
        std::cout << "Mismatch at (" << i << ", " << j
                  << "): Ref = " << C_ref(i, j) << ", Test = " << C_test(i, j)
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

// ============================================================================
// TODO 1: SEQUENTIAL MATRIX MULTIPLICATION (GEMM)
// ============================================================================
// Compute C = A * B sequentially using a basic 3-nested loop (i, j, k).
// Recall:
// For each row 'i' of matrix A and each col 'j' of matrix B,
// C(i, j) is the dot product of A's row 'i' and B's col 'j':
// C(i, j) = sum_{k=0}^{N-1} A(i, k) * B(k, j)
void gemm_sequential(const Matrix &A, const Matrix &B, Matrix &C) {
  // Zero out C first
  std::fill(C.data.begin(), C.data.end(), 0.0);

  int M = A.rows;
  int N = B.cols;
  int K = A.cols; // which is B.rows

  // --- YOUR CODE HERE (TODO 1) ---
  // Hint: Write three nested loops.
  // Loop 1: i from 0 to M - 1
  //   Loop 2: j from 0 to N - 1
  //     Loop 3: k from 0 to K - 1
  //       C(i, j) += A(i, k) * B(k, j)
  // --------------------------------
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

// ============================================================================
// TODO 2: PARALLEL MATRIX MULTIPLICATION WITH OPENMP
// ============================================================================
// Compute C = A * B in parallel using OpenMP.
//
// Concepts to apply:
// 1. Use the compiler directive: #pragma omp parallel for
// 2. Decide which variables are shared across threads, and which are private.
//    - Matrices A, B, and C are large read/write arrays: they MUST be 'shared'.
//    - Loop counters 'i', 'j', 'k' belong to the iteration space of a single
//    thread:
//      if they are not private, threads will overwrite each other's counters
//      (RACE CONDITION!). By default, in OpenMP, the loop variable of the
//      parallelized 'for' is private. However, inner loop variables (like 'j'
//      and 'k') must be declared private.
void gemm_parallel(const Matrix &A, const Matrix &B, Matrix &C) {
  // Zero out C first
  std::fill(C.data.begin(), C.data.end(), 0.0);

  int M = A.rows;
  int N = B.cols;
  int K = A.cols;

  // --- YOUR CODE HERE (TODO 2) ---
  // Hint:
  // Add the OpenMP compiler directive right above the outer loop.
  // It should look something like:
  // #pragma omp parallel for shared(A, B, C, M, N, K) private(j, k)
  // (Note: the outer loop variable 'i' is automatically made private by
  // OpenMP).
  //
  // Write your outer loop 'i' and inner loops 'j' and 'k' here.
  // --------------------------------
  int i, j, k;
#pragma omp parallel for shared(A, B, C, M, N, K) private(i, j, k)
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      for (k = 0; k < K; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

int main() {
  std::cout << "========================================================="
            << std::endl;
  std::cout << "      HPC Day 2: OpenMP Shared Memory Parallel GEMM      "
            << std::endl;
  std::cout << "========================================================="
            << std::endl;

  // Matrix size N x N
  const int SIZE = 512; // 512 x 512 matrix multiply takes ~134M operations
  std::cout << "Matrix Size: " << SIZE << " x " << SIZE << std::endl;
  std::cout << "OpenMP Max Threads Available: " << omp_get_max_threads()
            << std::endl;

  Matrix A(SIZE, SIZE);
  Matrix B(SIZE, SIZE);
  Matrix C_seq(SIZE, SIZE);
  Matrix C_par(SIZE, SIZE);

  initialize_matrix(A);
  initialize_matrix(B);

  hpc::Timer timer;

  // 1. Benchmark Sequential GEMM
  std::cout << "\n[1/3] Running Sequential Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_sequential(A, B, C_seq);
  timer.stop();
  double seq_time = timer.elapsed_seconds();
  timer.print("Sequential Time");

  // 2. Benchmark Parallel GEMM
  std::cout << "\n[2/3] Running Parallel OpenMP Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_parallel(A, B, C_par);
  timer.stop();
  double par_time = timer.elapsed_seconds();
  timer.print("Parallel Time");

  // 3. Verify Correctness & Calculate Speedup
  std::cout << "\n[3/3] Verifying Correctness & Speedup..." << std::endl;
  bool correct = verify_results(C_seq, C_par);
  if (correct) {
    std::cout << "✅ Correctness Check Passed! Parallel output matches "
                 "sequential baseline."
              << std::endl;
    double speedup = seq_time / par_time;
    std::cout << "🔥 Speedup: " << speedup << "x" << std::endl;
    std::cout << "💡 Theoretical Max Speedup: " << omp_get_max_threads() << "x"
              << std::endl;
  } else {
    std::cout << "❌ Correctness Check Failed! Please review your loops and "
                 "OpenMP private scope."
              << std::endl;
  }

  std::cout << "========================================================="
            << std::endl;
  return 0;
}
