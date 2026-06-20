//
// C++ threads implementation for the One-Shot project:
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is a parallel counterpart of the sequential reference
// (iterative_SpMV.cpp). It preserves the same algorithmic structure and the
// same matrix generation / vector initialization, so that results can be
// compared against the sequential baseline within a numerical tolerance.
//
// Parallelization strategy (first implementation)
// ------------------------------------------------
//   * Persistent worker threads (std::thread), spawned once for the whole
//     timed iterative loop.
//   * Static work distribution by a NNZ-BALANCED partition of the SOURCE rows:
//     each worker owns a contiguous range of source rows carrying roughly the
//     same number of nonzeros. Because the kernel iterates over source rows
//     (and writes to the shifted output position), the per-worker load depends
//     only on the nonzeros of its source rows and is therefore CONSTANT across
//     epochs. The matrix evolution (row shift) becomes a cheap change of the
//     output index mapping, with no data movement and no repartitioning.
//   * Per iteration the computation is a Map (shifted SpMV) + a global
//     reduction (L2 norm) + a Map (scaling). The partial sum-of-squares is
//     fused into the SpMV map. A reusable mutex + condition_variable barrier
//     provides the two synchronization points per iteration.
//   * Two ping-pong buffers replace the explicit x/y swap, so no serial
//     section is needed inside the loop (the logical row shift is derived
//     locally by each worker from the iteration index).
//
// Command line (superset of the sequential reference):
//   -n  N        matrix size, NxN
//   -nz K        total number of nonzeros
//   -m  mode     regular | irregular
//   -s  seed     optional seed, default 111
//   -t  T        optional number of threads (default: hardware concurrency)
//   --dump-vector FILE
//                 optional dump of the final normalized vector
//
// Build:
//   g++ -O3 -std=c++20 -pthread -I . -Wall iterative_SpMV_threads.cpp -o threads
//
// Examples:
//   ./threads -n 500000 -nz 20000000 -m irregular -t 8
//   ./threads -n 5000 -nz 20000 -m irregular -t 4 --dump-vector thr_vec.dump
//

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"


// number of iterations
static constexpr std::uint32_t NUM_ITERS = 500;
// number of iterations between two matrix-evolution steps
static constexpr std::uint32_t EPOCH_LEN = 25;


// Vector operations (used only for the serial vector initialization, exactly
// as in the sequential reference).

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static double l2_norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}

static void normalize(std::vector<double>& x) {
    const double nrm = l2_norm(x);
    const double inv = 1.0 / nrm;

    for (double& v : x) {
        v *= inv;
    }
}


// Computes the epoch parameter (identical to the sequential reference).
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}


// Synchronization uses std::barrier (C++20), which is reusable across the
// iterative loop's two barrier points per iteration.

// Cache-line-padded double: each worker's reduction slot lives on its own
// 64-byte cache line, so the per-iteration writes to partial[tid] do not
// invalidate neighbouring slots in other cores' caches (false sharing).
struct alignas(64) PaddedDouble {
    double v = 0.0;
};


// NNZ-balanced contiguous partition of the source rows [0, n).
//
// boundary[t] .. boundary[t+1] is the source-row range owned by worker t.
// Boundaries are chosen so that each range carries approximately the same
// number of nonzeros, using the CSR row_ptr array as a prefix sum of nnz.
static std::vector<std::size_t> nnz_balanced_partition(const CSRMatrix& A, unsigned P) {
    const std::size_t n = A.n;
    const std::uint64_t total_nnz = A.row_ptr[n];

    std::vector<std::size_t> boundary(P + 1);
    boundary[0] = 0;
    boundary[P] = n;

    for (unsigned t = 1; t < P; ++t) {
        // Target cumulative nnz at the start of worker t.
        const std::uint64_t target =
            static_cast<std::uint64_t>((static_cast<__uint128_t>(total_nnz) * t) / P);

        // First row whose cumulative nnz reaches the target.
        const auto it = std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), target);
        std::size_t row = static_cast<std::size_t>(it - A.row_ptr.begin());

        // Keep boundaries monotonic and within [0, n].
        if (row < boundary[t - 1]) row = boundary[t - 1];
        if (row > n) row = n;
        boundary[t] = row;
    }

    // Ensure monotonicity at the tail as well.
    for (unsigned t = 1; t <= P; ++t) {
        if (boundary[t] < boundary[t - 1]) boundary[t] = boundary[t - 1];
    }
    boundary[P] = n;
    return boundary;
}


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};


// Per-worker body: runs the whole timed iterative computation for its own
// source-row range [s0, s1).
static void worker_body(unsigned tid,
                        unsigned P,
                        const CSRMatrix& A,
                        std::size_t shift_rows,
                        std::size_t s0,
                        std::size_t s1,
                        std::vector<double>& buf0,
                        std::vector<double>& buf1,
                        std::vector<PaddedDouble>& partial,
                        std::vector<PaddedDouble>& partial_dot,
                        std::barrier<>& sync,
                        double& rayleigh_out) {
    const std::size_t n = A.n;
    const std::uint64_t* rp = A.row_ptr.data();
    const std::uint32_t* ci = A.col_idx.data();
    const double*        va = A.values.data();

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        // Logical row shift for this iteration, derived locally (no shared
        // state). It is equivalent to the cumulative shift of the sequential
        // reference: row_shift updated by shift_rows every EPOCH_LEN iters.
        const std::size_t epochs = iter / EPOCH_LEN;
        const std::size_t shift  = (shift_rows * epochs) % n;

        // Ping-pong buffers: read from one, write to the other.
        const double* xr;
        double*       yw;
        if ((iter & 1u) == 0u) {
            xr = buf0.data();
            yw = buf1.data();
        } else {
            xr = buf1.data();
            yw = buf0.data();
        }

        // Phase A: shifted SpMV map over the owned source rows, fused with the
        // local sum-of-squares accumulation for the L2 norm.
        double local_ss = 0.0;
        for (std::size_t src = s0; src < s1; ++src) {
            double sum = 0.0;
            for (std::uint64_t p = rp[src]; p < rp[src + 1]; ++p) {
                sum += va[p] * xr[ci[p]];
            }
            const std::size_t out = (src + shift) % n;
            yw[out] = sum;
            local_ss += sum * sum;
        }
        partial[tid].v = local_ss;

        // Global reduction point: every worker waits for all partial sums.
        sync.arrive_and_wait();

        // Each worker reduces the partial sums in the same fixed order, so the
        // resulting inverse norm is identical across workers (no need to
        // publish a shared value).
        double total = 0.0;
        for (unsigned k = 0; k < P; ++k) {
            total += partial[k].v;
        }
        const double inv = 1.0 / std::sqrt(total);

        // Phase B: scale the entries this worker produced.
        for (std::size_t src = s0; src < s1; ++src) {
            const std::size_t out = (src + shift) % n;
            yw[out] *= inv;
        }

        // End-of-iteration barrier: the written buffer becomes the read buffer
        // of the next iteration, so all writes must be visible first.
        sync.arrive_and_wait();
    }

    // Final diagnostics, inside the timed region as in the sequential code.
    // After NUM_ITERS iterations the final vector lives in buf0 (NUM_ITERS is
    // even); the extra SpMV uses the final shift and is written into buf1.
    const std::size_t epochs_final = (NUM_ITERS - 1) / EPOCH_LEN;
    const std::size_t shift_final  = (shift_rows * epochs_final) % n;

    const double* xf = buf0.data();   // final x
    double*       ye = buf1.data();   // extra y = A_shifted * x

    double local_dot = 0.0;
    for (std::size_t src = s0; src < s1; ++src) {
        double sum = 0.0;
        for (std::uint64_t p = rp[src]; p < rp[src + 1]; ++p) {
            sum += va[p] * xf[ci[p]];
        }
        const std::size_t out = (src + shift_final) % n;
        ye[out] = sum;
        local_dot += xf[out] * sum;   // contribution to dot(x, y)
    }
    partial_dot[tid].v = local_dot;

    // Combine the Rayleigh-like value (rayleigh = dot(x, y)).
    sync.arrive_and_wait();
    if (tid == 0) {
        double total = 0.0;
        for (unsigned k = 0; k < P; ++k) {
            total += partial_dot[k].v;
        }
        rayleigh_out = total;
    }
}


static IterativeResult iterative_spmv_evolving_threads(const CSRMatrix& A,
                                                       std::uint64_t seed,
                                                       unsigned num_threads,
                                                       std::vector<double>* final_vector) {
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // Phase 1: serial vector initialization, identical to the sequential
    // reference, so that the initial vector (and thus the result) matches.
    std::vector<double> buf0(n);
    std::vector<double> buf1(n, 0.0);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : buf0) {
        v = rng.next_unit();
    }
    normalize(buf0);

    // Clamp the number of workers to something meaningful.
    unsigned P = num_threads;
    if (P == 0) P = 1;
    if (P > n) P = static_cast<unsigned>(n);

    const std::vector<std::size_t> boundary = nnz_balanced_partition(A, P);

    std::vector<PaddedDouble> partial(P);
    std::vector<PaddedDouble> partial_dot(P);
    std::barrier<> sync(P);
    double rayleigh = 0.0;

    // Phase 2 + 3: spawn the persistent workers for the whole iterative loop
    // and the final diagnostics.
    std::vector<std::thread> threads;
    threads.reserve(P);
    for (unsigned t = 0; t < P; ++t) {
        threads.emplace_back(worker_body,
                             t, P, std::cref(A), shift_rows,
                             boundary[t], boundary[t + 1],
                             std::ref(buf0), std::ref(buf1),
                             std::ref(partial), std::ref(partial_dot),
                             std::ref(sync), std::ref(rayleigh));
    }
    for (std::thread& th : threads) {
        th.join();
    }

    // The final vector is in buf0. The checksum (order independent) is computed
    // here, still inside the timed region as in the sequential reference.
    const std::uint64_t checksum = checksum_vector(buf0);

    if (final_vector != nullptr) {
        *final_vector = std::move(buf0);
    }

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = (shift_rows * ((NUM_ITERS - 1) / EPOCH_LEN)) % n
    };
}


int main(int argc, char** argv) {
    // Phase 0: read problem size, sparsity mode, seed, threads, dump path.
    std::uint64_t n64     = 0;
    std::uint64_t nz      = 0;
    std::uint64_t seed    = 111;
    std::uint64_t threads = 0;
    std::string mode;
    std::string dump_vector_path;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode)) {
        usage(argv[0]);
        std::cerr << "  -t   Optional number of threads (default: hardware concurrency)\n";
        return 1;
    }

    // Optional arguments.
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);
    if (!read_arg_u64(argc, argv, "-t", threads)) {
        (void)read_arg_u64(argc, argv, "--threads", threads);
    }

    unsigned num_threads = static_cast<unsigned>(threads);
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }

    const std::size_t n = static_cast<std::size_t>(n64);
    std::cout << "SPARSE_ITERATION_CPP_THREADS\n";

    try {
        // Phase 1: input construction (not included in computation time).
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();

        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "threads=" << num_threads << "\n";
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed iterative computation.
        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result =
            iterative_spmv_evolving_threads(G.A, seed, num_threads, final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now();

        const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << computation_sec << "\n";

        // Phase 3: optional correctness support, outside the timed region.
        if (!dump_vector_path.empty()) {
            dump_vector(dump_vector_path, final_vector);
            std::cout << "vector_dump=" << dump_vector_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
