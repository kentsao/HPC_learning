#include "../utils/timer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// A flat 1D matrix class representing 2D layouts in row-major order
class Matrix {
public:
  int rows;
  int cols;
  std::vector<double> data;

  Matrix(int r, int c, double val = 0.0) : rows(r), cols(c), data(r * c, val) {}

  inline double &operator()(int r, int c) { return data[r * cols + c]; }

  inline const double &operator()(int r, int c) const {
    return data[r * cols + c];
  }
};

void initialize_matrix(Matrix &mat) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(0.0, 1.0);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      mat(i, j) = dis(gen);
    }
  }
}

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

// 1. Baseline GEMM (IJK Layout)
// This is the basic sequential implementation.
// As 'k' increases, B(k, j) jumps in memory by cols elements. This causes
// severe cache misses!
void gemm_baseline(const Matrix &A, const Matrix &B, Matrix &C) {
  std::fill(C.data.begin(), C.data.end(), 0.0);
  int M = A.rows;
  int N = B.cols;
  int K = A.cols;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

// ============================================================================
// TODO 1: OPTIMIZED LOOP ORDER (IKJ Layout)
// ============================================================================
// Swap the inner loops to make memory access in the innermost loop contiguous!
//
// Hint:
// Loop order:
// For i = 0 to M - 1:
//   For k = 0 to K - 1:
//     double r = A(i, k); // Read once into a register
//     For j = 0 to N - 1:
//       C(i, j) += r * B(k, j);
//
// Why this works: As 'j' increments in the innermost loop, B(k, j) and C(i, j)
// are accessed contiguously in memory. This results in nearly 100% Cache Hits!
void gemm_ikj(const Matrix &A, const Matrix &B, Matrix &C) {
  std::fill(C.data.begin(), C.data.end(), 0.0);
  int M = A.rows;
  int N = B.cols;
  int K = A.cols;

  // --- YOUR CODE HERE (TODO 1) ---
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      double r = A(i, k); // Read once into a register
      for (int j = 0; j < N; ++j) {
        C(i, j) += r * B(k, j);
      }
    }
  }

  // --------------------------------
}

// ============================================================================
// TODO 2: LOOP TILING (BLOCKING) GEMM
// ============================================================================
// Loop Tiling splits the loop range into smaller blocks (tiles) of size
// BLOCK_SIZE. We load tiles of A and B that fit entirely inside the L1/L2
// cache, perform matrix multiplication inside those tiles, and move to the next
// tile.
//
// Hint:
// Use 6 nested loops:
// - Outer 3 loops iterate over the tiles (steps of BLOCK_SIZE).
// - Inner 3 loops perform matrix multiplication within the current tiles.
//
// Structure:
// For ii = 0 to M - 1 (step BLOCK_SIZE):
//   For kk = 0 to K - 1 (step BLOCK_SIZE):
//     For jj = 0 to N - 1 (step BLOCK_SIZE):
//       // Micro-kernel inside the tile
//       For i = ii to min(ii + BLOCK_SIZE, M):
//         For k = kk to min(kk + BLOCK_SIZE, K):
//           double r = A(i, k);
//           For j = jj to min(jj + BLOCK_SIZE, N):
//             C(i, j) += r * B(k, j);
void gemm_tiled(const Matrix &A, const Matrix &B, Matrix &C, int block_size) {
  std::fill(C.data.begin(), C.data.end(), 0.0);
  int M = A.rows;
  int N = B.cols;
  int K = A.cols;

  // --- YOUR CODE HERE (TODO 2) ---
  for (int ii = 0; ii < M; ii += block_size) {
    for (int kk = 0; kk < K; kk += block_size) {
      for (int jj = 0; jj < N; jj += block_size) {
        // Micro-kernel inside the tile
        for (int i = ii; i < std::min(ii + block_size, M); i++) {
          for (int k = kk; k < std::min(kk + block_size, K); k++) {
            double r = A(i, k);
            for (int j = jj; j < std::min(jj + block_size, N); j++) {
              C(i, j) += r * B(k, j);
            }
          }
        }
      }
    }
  }

  // --------------------------------
}

int main() {
  std::cout << "========================================================="
            << std::endl;
  std::cout << "      HPC Day 4: Cache Locality & Loop Tiling GEMM       "
            << std::endl;
  std::cout << "========================================================="
            << std::endl;

  // We increase matrix size to 1024 x 1024 to make cache effects more
  // prominent. 1024 x 1024 matrices take 8MB of memory each (exceeding standard
  // M1 L1 Cache size)
  const int SIZE = 1024;
  const int BLOCK_SIZE = 64; // Block size that fits well inside L2 Cache
  std::cout << "Matrix Size: " << SIZE << " x " << SIZE << std::endl;
  std::cout << "Tiling Block Size: " << BLOCK_SIZE << std::endl;

  Matrix A(SIZE, SIZE);
  Matrix B(SIZE, SIZE);
  Matrix C_ref(SIZE, SIZE);
  Matrix C_ikj(SIZE, SIZE);
  Matrix C_tiled(SIZE, SIZE);

  initialize_matrix(A);
  initialize_matrix(B);

  hpc::Timer timer;

  // 1. Benchmark Baseline IJK
  std::cout << "\n[1/4] Running Baseline IJK Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_baseline(A, B, C_ref);
  timer.stop();
  double base_time = timer.elapsed_seconds();
  timer.print("Baseline IJK Time");

  // 2. Benchmark Optimized IKJ
  std::cout << "\n[2/4] Running Optimized IKJ Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_ikj(A, B, C_ikj);
  timer.stop();
  double ikj_time = timer.elapsed_seconds();
  timer.print("Optimized IKJ Time");

  // Verify IKJ
  if (!verify_results(C_ref, C_ikj)) {
    std::cout << "❌ Correctness Check Failed for IKJ!" << std::endl;
    return -1;
  }
  std::cout << "✅ Correctness Check Passed for IKJ. Speedup: "
            << base_time / ikj_time << "x" << std::endl;

  // 3. Benchmark Tiled (Blocked) GEMM
  std::cout << "\n[3/4] Running Tiled (Blocked) Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_tiled(A, B, C_tiled, BLOCK_SIZE);
  timer.stop();
  double tiled_time = timer.elapsed_seconds();
  timer.print("Tiled Time");

  // Verify Tiled
  if (!verify_results(C_ref, C_tiled)) {
    std::cout << "❌ Correctness Check Failed for Tiled!" << std::endl;
    return -1;
  }
  std::cout << "✅ Correctness Check Passed for Tiled. Speedup: "
            << base_time / tiled_time << "x" << std::endl;

  std::cout << "\n[4/4] Performance Summary:" << std::endl;
  std::cout << "Baseline IJK : " << base_time << " s ("
            << (2.0 * SIZE * SIZE * SIZE) / (base_time * 1e9) << " GFLOPs)"
            << std::endl;
  std::cout << "Optimized IKJ: " << ikj_time << " s ("
            << (2.0 * SIZE * SIZE * SIZE) / (ikj_time * 1e9) << " GFLOPs)"
            << std::endl;
  std::cout << "Tiled GEMM   : " << tiled_time << " s ("
            << (2.0 * SIZE * SIZE * SIZE) / (tiled_time * 1e9) << " GFLOPs)"
            << std::endl;
  std::cout << "========================================================="
            << std::endl;

  return 0;
}
