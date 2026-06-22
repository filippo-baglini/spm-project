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
//     fused into the SpMV map. A C++20 std::barrier provides the two
//     synchronization points per iteration; its completion function performs
//     the norm reduction once and publishes the inverse norm to all workers.
//   * Two ping-pong buffers replace the explicit x/y swap, so no serial
//     section is needed inside the loop (the logical row shift is derived
//     locally by each worker from the iteration index).
//
// Optimizations grounded in SPM NOTES / the course example code (Code/):
//   A  false-sharing padding of the reduction slots (PaddedDouble).
//   B  std::barrier instead of a hand-rolled mutex+condition_variable barrier.
//   C  thread affinity: each worker is pinned to a fixed logical CPU. The
//      default policy is compact (core == tid); set the AFFINITY environment
//      variable to a CPU list (e.g. "0,2,4-10:2") to override it.
//   D  __restrict__ on the CSR arrays and ping-pong buffers (no aliasing).
//   E  the std::barrier completion function reduces the norm once per iteration.
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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "matrix_generation.hpp"
#include "utils.hpp"


// Optimization C: thread affinity / pinning (SPM NOTES p.95, p.130; course
// example Code/spmcode7). Pinning each worker to a fixed logical CPU stops the
// scheduler from migrating it and keeps it on the NUMA node that first-touched
// its data, which is the dominant cost on this memory-bound kernel at high
// thread counts.

// Best-effort pin of the calling thread to a single logical CPU. A no-op on
// non-Linux platforms or if the syscall is rejected (e.g. CPU not in cpuset).
#if defined(__linux__)
static void pin_to_core(unsigned core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#else
static void pin_to_core(unsigned) {}
#endif

// Parse a CPU-list string such as "0,2,4-10:2" -> {0,2,4,6,8,10}. Mirrors the
// course helper Affinity.hpp::parse_cpu_list. Used only to let the AFFINITY
// environment variable override the default compact (core == tid) policy, so
// compact/scatter placements can be benchmarked without recompiling.
static std::vector<unsigned> parse_cpu_list(const std::string& slist) {
    std::vector<unsigned> cpus;
    std::stringstream ss(slist);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim surrounding whitespace
        const auto b = tok.find_first_not_of(" \t");
        if (b == std::string::npos) continue;
        const auto e = tok.find_last_not_of(" \t");
        tok = tok.substr(b, e - b + 1);

        const auto dash = tok.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(static_cast<unsigned>(std::stoul(tok, nullptr, 0)));
        } else {
            const auto col = tok.find(':');
            unsigned a = static_cast<unsigned>(std::stoul(tok.substr(0, dash), nullptr, 0));
            unsigned hi, step = 1;
            if (col == std::string::npos) {
                hi = static_cast<unsigned>(std::stoul(tok.substr(dash + 1), nullptr, 0));
            } else {
                hi   = static_cast<unsigned>(std::stoul(tok.substr(dash + 1, col - (dash + 1)), nullptr, 0));
                step = static_cast<unsigned>(std::stoul(tok.substr(col + 1), nullptr, 0));
                if (step == 0) step = 1;
            }
            if (a > hi) std::swap(a, hi);
            for (unsigned c = a; c <= hi; c += step) cpus.push_back(c);
        }
    }
    return cpus;
}

// Resolve the logical CPU that worker `tid` should pin to. With AFFINITY set,
// worker tid takes the tid-th entry of the parsed list (wrapping if shorter);
// otherwise the compact default core == tid is used.
static unsigned core_for_worker(unsigned tid) {
    static const std::vector<unsigned> cpus = [] {
        const char* env = std::getenv("AFFINITY");
        return (env && *env) ? parse_cpu_list(env) : std::vector<unsigned>{};
    }();
    if (!cpus.empty()) return cpus[tid % cpus.size()];
    return tid;
}


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


// Optimization E: std::barrier completion function (SPM NOTES p.83; course
// example Code/spmcode6/spinbarrier-wait.cpp). Instead of every worker
// redundantly reducing the partial sum-of-squares (P*P adds per iteration),
// the last thread to reach the norm barrier runs this functor once (P adds),
// in the same fixed order as before, and publishes the inverse L2 norm that all
// workers then read. operator() must be noexcept to satisfy std::barrier's
// CompletionFunction requirement.
struct NormReducer {
    const PaddedDouble* partial = nullptr;
    unsigned            P       = 0;
    double*             inv_norm = nullptr;

    void operator()() noexcept {
        double total = 0.0;
        for (unsigned k = 0; k < P; ++k) {
            total += partial[k].v;
        }
        *inv_norm = 1.0 / std::sqrt(total);
    }
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
                        std::barrier<NormReducer>& sync,
                        const double& inv_norm,
                        double& rayleigh_out) {
    // Optimization C: pin this worker to a fixed logical CPU for the whole run.
    pin_to_core(core_for_worker(tid));

    const std::size_t n = A.n;
    // Optimization D: __restrict__ tells the compiler these arrays and the
    // ping-pong buffers below do not alias, enabling tighter scheduling and
    // vectorization of the scale/norm loops.
    const std::uint64_t* __restrict__ rp = A.row_ptr.data();
    const std::uint32_t* __restrict__ ci = A.col_idx.data();
    const double*        __restrict__ va = A.values.data();

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        // Logical row shift for this iteration, derived locally (no shared
        // state). It is equivalent to the cumulative shift of the sequential
        // reference: row_shift updated by shift_rows every EPOCH_LEN iters.
        const std::size_t epochs = iter / EPOCH_LEN;
        const std::size_t shift  = (shift_rows * epochs) % n;

        // Ping-pong buffers: read from one, write to the other.
        const double* __restrict__ xr;
        double*       __restrict__ yw;
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

        // Global reduction point: every worker waits for all partial sums. The
        // barrier's completion function (NormReducer) reduces them once into the
        // shared inv_norm; after crossing the barrier every worker reads the
        // same published value. (Optimization E.)
        sync.arrive_and_wait();
        const double inv = inv_norm;

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

    // Shared inverse L2 norm, written once per norm barrier by the completion
    // function and read by all workers (Optimization E).
    double inv_norm = 0.0;
    std::barrier<NormReducer> sync(P, NormReducer{partial.data(), P, &inv_norm});
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
                             std::ref(sync), std::cref(inv_norm), std::ref(rayleigh));
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
