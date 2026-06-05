#include <chrono>
#include <iostream>
#include <mpi.h>
#include <random>

// ============================================================================
// C++ & MPI LESSON FOR PYTHON DEVELOPERS: Distributed Memory Parallelism
// ============================================================================
// Unlike OpenMP where multiple threads share the same variables in RAM, in MPI
// (Message Passing Interface), we run MULTIPLE INDEPENDENT PROCESSES.
//
// Each process:
// 1. Has its own separate memory space (no shared variables!).
// 2. Can reside on a different computer node in a server room.
// 3. Identifies itself using a unique ID called a "rank" (0, 1, 2, ...).
// 4. Must communicate with other processes explicitly using network messages
//    or shared-memory buffers.
//
// Monte Carlo Pi Estimation:
// Imagine a square of size 2x2 containing a circle of radius 1.
// - Area of Square = 4
// - Area of Circle = pi * r^2 = pi
// - Ratio of Circle area to Square area = pi / 4
// - If we throw N random points into the square, the fraction of points that
//   fall inside the circle is approximately pi / 4.
// - pi approx 4 * (points_inside / total_points)

int main(int argc, char *argv[]) {
  // --- TODO 1: Initialize MPI environment ---
  // Initialize MPI, get the rank of this process, and the total number of
  // processes.
  int rank = 0;
  int size = 1;

  // Hint:
  // Call: MPI_Init(&argc, &argv);
  // Call: MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Call: MPI_Comm_size(MPI_COMM_WORLD, &size);
  // ------------------------------------------
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Total throws across all processes
  const long long total_throws = 100000000LL; // 100 million throws

  // Each process does a fraction of the total throws
  long long local_throws = total_throws / size;

  // Record start time on Rank 0
  double start_time = 0.0;
  if (rank == 0) {
    start_time = MPI_Wtime(); // MPI high-resolution wall timer
    std::cout << "========================================================="
              << std::endl;
    std::cout << "      HPC Day 3: MPI Distributed Pi Estimation           "
              << std::endl;
    std::cout << "========================================================="
              << std::endl;
    std::cout << "Total processes: " << size << std::endl;
    std::cout << "Total throws: " << total_throws << " (" << local_throws
              << " per process)" << std::endl;
  }

  // Initialize random number generator with a unique seed for each process!
  // If all processes used the same seed, they would generate the exact same
  // numbers, making our parallel efforts completely redundant.
  std::mt19937 gen(42 + rank);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  long long local_inside_count = 0;

  // --- TODO 2: Monte Carlo Loop ---
  // Loop local_throws times.
  // 1. Generate random x and y between -1.0 and 1.0.
  // 2. Check if the point falls inside the unit circle (x^2 + y^2 <= 1.0).
  // 3. If yes, increment local_inside_count.
  // ---------------------------------
  // (Write your loop here)
  for (long long i = 0; i < local_throws; ++i) {
    double x = dis(gen);
    double y = dis(gen);
    if (x * x + y * y <= 1.0) {
      local_inside_count++;
    }
  }

  // --- TODO 3: Aggregate Results ---
  // Collect all local counts from all processes and sum them up on Process 0.
  long long global_inside_count = 0;

  // Hint:
  // Use MPI_Reduce. The parameters are:
  // MPI_Reduce(&sendbuf, &recvbuf, count, datatype, op, root, comm)
  // - sendbuf: &local_inside_count
  // - recvbuf: &global_inside_count
  // - count: 1
  // - datatype: MPI_LONG_LONG
  // - op: MPI_SUM (reduces by summing)
  // - root: 0 (the rank that will receive the final sum)
  // - comm: MPI_COMM_WORLD
  // ---------------------------------
  MPI_Reduce(&local_inside_count, &global_inside_count, 1, MPI_LONG_LONG,
             MPI_SUM, 0, MPI_COMM_WORLD);

  // Print results on Rank 0
  if (rank == 0) {
    double end_time = MPI_Wtime();
    double pi_estimate =
        4.0 * (double)global_inside_count / (double)total_throws;
    double error = std::abs(pi_estimate - 3.14159265358979323846);

    std::cout << "\nResults:" << std::endl;
    std::cout << "Estimated Pi : " << pi_estimate << std::endl;
    std::cout << "Actual Pi    : 3.141592653589793" << std::endl;
    std::cout << "Absolute Error: " << error << std::endl;
    std::cout << "Execution Time: " << (end_time - start_time) << " seconds"
              << std::endl;
    std::cout << "========================================================="
              << std::endl;
  }

  // --- TODO 4: Finalize MPI ---
  // Clean up and finalize the MPI environment.
  // Hint: Call MPI_Finalize();
  // -----------------------------
  MPI_Finalize();

  return 0;
}
