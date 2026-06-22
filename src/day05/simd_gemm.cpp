#include "../utils/timer.hpp"
#include <arm_neon.h> // Header for ARM Neon intrinsics
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// A flat 1D matrix class representing 2D layouts using single-precision floats
// (float) We use float because Neon 128-bit vector registers can process 4
// floats simultaneously, which is standard for deep learning (FP32) workloads.
class Matrix {
public:
  int rows;
  int cols;
  std::vector<float> data;

  Matrix(int r, int c, float val = 0.0f) : rows(r), cols(c), data(r * c, val) {}

  inline float &operator()(int r, int c) { return data[r * cols + c]; }

  inline const float &operator()(int r, int c) const {
    return data[r * cols + c];
  }
};

void initialize_matrix(Matrix &mat) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      mat(i, j) = dis(gen);
    }
  }
}

bool verify_results(const Matrix &C_ref, const Matrix &C_test,
                    float epsilon = 1e-4f) {
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

// 1. Baseline GEMM (IJK Layout) - Single Precision
void gemm_baseline(const Matrix &A, const Matrix &B, Matrix &C) {
  std::fill(C.data.begin(), C.data.end(), 0.0f);
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

// 2. Optimized IKJ Layout - Single Precision
void gemm_ikj(const Matrix &A, const Matrix &B, Matrix &C) {
  std::fill(C.data.begin(), C.data.end(), 0.0f);
  int M = A.rows;
  int N = B.cols;
  int K = A.cols;

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float r = A(i, k);
      for (int j = 0; j < N; ++j) {
        C(i, j) += r * B(k, j);
      }
    }
  }
}

// ============================================================================
// TODO: NEON SIMD VECTORIZED MATRIX MULTIPLICATION
// ============================================================================
// Optimize the IKJ layout using ARM Neon vector intrinsics.
// We will process 4 float elements at a time in the innermost loop.
//
// Neon Intrinsics to use:
// 1. float32x4_t: The data type representing a 128-bit register holding 4
// floats.
// 2. vdupq_n_f32(val): Broadcasts/replicates a scalar float to all 4 lanes of a
// vector.
// 3. vld1q_f32(ptr): Loads 4 contiguous floats from memory address 'ptr' into a
// vector.
// 4. vfmaq_f32(c_vec, a_vec, b_vec): Fused Multiply-Accumulate instruction.
//    Computes: c_vec + (a_vec * b_vec) and returns the result vector.
// 5. vst1q_f32(ptr, c_vec): Stores the 4 floats of 'c_vec' back into memory
// address 'ptr'.
void gemm_neon(const Matrix &A, const Matrix &B, Matrix &C) {
  std::fill(C.data.begin(), C.data.end(), 0.0f);
  int M = A.rows;
  int N = B.cols;
  int K = A.cols;

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      // --- YOUR CODE HERE (TODO) ---
      // 1. Load A(i, k) and replicate it 4 times into a vector.
      //    Example: float32x4_t a_vec = vdupq_n_f32(A(i, k));

      // 2. Loop over columns 'j' with a stride of 4 (j += 4):
      //      a. Load 4 floats from B(k, j) using vld1q_f32.
      //         Address hint: &B(k, j)
      //      b. Load 4 floats from C(i, j) using vld1q_f32.
      //      c. Multiply-accumulate using vfmaq_f32.
      //      d. Store the result back to C(i, j) using vst1q_f32.
      // -----------------------------
      float32x4_t a_vec = vdupq_n_f32(A(i, k));

      for (int j = 0; j < N; j += 4) {
        float32x4_t b_vec = vld1q_f32(&B(k, j));
        float32x4_t c_vec = vld1q_f32(&C(i, j));
        c_vec = vfmaq_f32(c_vec, a_vec, b_vec);
        vst1q_f32(&C(i, j), c_vec);
      }
    }
  }
}

int main() {
  std::cout << "========================================================="
            << std::endl;
  std::cout << "      HPC Day 5: ARM Neon SIMD Vectorized GEMM           "
            << std::endl;
  std::cout << "========================================================="
            << std::endl;

  const int SIZE =
      1024; // Ensure size is a multiple of 4 for easy vector alignment
  std::cout << "Matrix Size: " << SIZE << " x " << SIZE << std::endl;

  Matrix A(SIZE, SIZE);
  Matrix B(SIZE, SIZE);
  Matrix C_ref(SIZE, SIZE);
  Matrix C_ikj(SIZE, SIZE);
  Matrix C_neon(SIZE, SIZE);

  initialize_matrix(A);
  initialize_matrix(B);

  hpc::Timer timer;

  // 1. Baseline GEMM (IJK)
  std::cout << "\n[1/4] Running Baseline IJK Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_baseline(A, B, C_ref);
  timer.stop();
  double base_time = timer.elapsed_seconds();
  timer.print("Baseline Time");

  // 2. Optimized GEMM (IKJ)
  std::cout << "\n[2/4] Running Optimized IKJ Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_ikj(A, B, C_ikj);
  timer.stop();
  double ikj_time = timer.elapsed_seconds();
  timer.print("IKJ Time");

  if (!verify_results(C_ref, C_ikj)) {
    std::cout << "❌ Correctness Check Failed for IKJ!" << std::endl;
    return -1;
  }
  std::cout << "✅ Correctness Check Passed for IKJ. Speedup: "
            << base_time / ikj_time << "x" << std::endl;

  // 3. Neon Vectorized GEMM
  std::cout << "\n[3/4] Running Neon Vectorized Matrix Multiplication..."
            << std::endl;
  timer.start();
  gemm_neon(A, B, C_neon);
  timer.stop();
  double neon_time = timer.elapsed_seconds();
  timer.print("Neon SIMD Time");

  if (!verify_results(C_ref, C_neon)) {
    std::cout << "❌ Correctness Check Failed for Neon SIMD!" << std::endl;
    return -1;
  }
  std::cout << "✅ Correctness Check Passed for Neon SIMD. Speedup: "
            << base_time / neon_time << "x" << std::endl;

  std::cout << "\n[4/4] Performance Summary:" << std::endl;
  std::cout << "Baseline IJK : " << base_time << " s ("
            << (2.0 * SIZE * SIZE * SIZE) / (base_time * 1e9) << " GFLOPs)"
            << std::endl;
  std::cout << "Optimized IKJ: " << ikj_time << " s ("
            << (2.0 * SIZE * SIZE * SIZE) / (ikj_time * 1e9) << " GFLOPs)"
            << std::endl;
  std::cout << "Neon SIMD    : " << neon_time << " s ("
            << (2.0 * SIZE * SIZE * SIZE) / (neon_time * 1e9) << " GFLOPs)"
            << std::endl;
  std::cout << "========================================================="
            << std::endl;

  return 0;
}
