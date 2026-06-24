//
// C++ threads + thread-pool implementation for the One-Shot project:
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is a COMPARISON VARIANT of the C++ threads deliverable. The primary
// threads version (iterative_SpMV_threads.cpp) uses a *static*, nnz-balanced
// partition of the source rows: it spawns the workers once and each owns a
// fixed range for the whole run. This variant instead balances the work
// *dynamically* by driving the professor's thread pool (Code/spmcode7):
// every iteration it submits the row range as many small tasks, and the pool's
// idle workers pull whatever task is next.
//
// The point of the experiment is the three-way contrast:
//   * static-partition threads   (iterative_SpMV_threads.cpp)
//   * dynamic-pool   threads     (this file)
//   * dynamic OpenMP tasks       (iterative_SpMV_omp.cpp)
// On this memory-bound, well-balanced kernel the dynamic pool is expected to be
// SLOWER than the static partition: a dynamic scheme must re-submit a batch of
// tasks on every one of the NUM_ITERS iterations, and each submit takes the
// pool's single global queue mutex, notifies a worker, and (on C++20) heap-
// allocates a packaged_task/future. Quantifying that scheduling overhead -- and
// explaining why static partitioning wins here -- is the deliverable.
//
// Optimization parity with the static-threads version (iterative_SpMV_threads.cpp,
// opts A-E): A (false-sharing padding, per-chunk PaddedDouble) and C (affinity, via
// the pool's AFFINITY-driven pinning) are present; E (reduce the norm once) is done
// by the coordinator after wait_completion; D (__restrict__ on the CSR arrays and
// buffers) is applied inside each task lambda below. B (std::barrier) does not apply
// -- the pool provides its own queue-based barrier (wait_completion).
//
// About the pool (include/threadPool.hpp): it is a SHARED-FIFO-QUEUE pool (one
// std::deque + one mutex; workers pop the front), with futures and a cooperative
// "helping" wait (wait_future). It is NOT a distributed work-stealing scheduler
// (no per-worker deques, no victim selection), but the shared queue still gives
// dynamic load balancing. Here we use submit() + wait_completion() as a per-phase
// barrier; the wait_future() helping path is unnecessary because our tasks do not
// recursively spawn sub-tasks.
//
// The algorithm, matrix generation and serial vector initialization are kept
// IDENTICAL to the sequential reference, so the result matches the sequential
// baseline within a numerical tolerance (the strict bitwise checksum may differ
// because the reduction order differs -- expected and acceptable).
//
// Command line (same as the sequential / threads versions):
//   -n  N        matrix size, NxN
//   -nz K        total number of nonzeros
//   -m  mode     regular | irregular
//   -s  seed     optional seed, default 111
//   -t  T        optional number of pool workers (default: hardware concurrency)
//   --dump-vector FILE
//                 optional dump of the final normalized vector
//
// Thread affinity reuses the same AFFINITY environment convention as the threads
// version (e.g. AFFINITY=0-31): the string is parsed by the pool and worker i is
// pinned to the i-th CPU in the list (compact when AFFINITY="0..NCPU-1").
//
// Build:
//   g++ -O3 -std=c++20 -pthread -I include
//       src/threads/iterative_SpMV_pool.cpp -o bin/pool
//   (or: make pool)
//
// Examples:
//   ./bin/pool -n 500000 -nz 20000000 -m irregular -t 8
//   AFFINITY=0-31 ./bin/pool -n 500000 -nz 20000000 -t 16
//   GRAIN=4096 ./bin/pool -n 100000 -nz 4000000 -m irregular -t 8
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"
#include "threadPool.hpp"   // pulls in taskFactory.hpp and Affinity.hpp


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


// Resolve the task grainsize: GRAIN env override, else a value that yields
// roughly (workers * 8) tasks so the pool can load-balance the irregular
// nonzero distribution dynamically. Same policy/knob as the OpenMP version.
static std::size_t resolve_grain(std::size_t n, unsigned P) {
    if (const char* env = std::getenv("GRAIN")) {
        if (*env) {
            const long g = std::strtol(env, nullptr, 10);
            if (g > 0) return static_cast<std::size_t>(g);
        }
    }
    std::size_t grain = n / (static_cast<std::size_t>(P) * 8);
    if (grain == 0) grain = 1;
    return grain;
}


// Cache-line-padded double: each chunk's reduction slot lives on its own 64-byte
// cache line, so concurrent writes to neighbouring slots by different workers do
// not invalidate each other (false sharing).
struct alignas(64) PaddedDouble {
    double v = 0.0;
};


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};


static IterativeResult iterative_spmv_evolving_pool(const CSRMatrix& A,
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

    // CSR arrays, hoisted out of the loop (captured by value -- as pointers --
    // into each task; they reference the shared, read-only matrix).
    const std::uint64_t* rp = A.row_ptr.data();
    const std::uint32_t* ci = A.col_idx.data();
    const double*        va = A.values.data();

    // Static decomposition of the row range [0, n) into fixed chunks of `grain`
    // rows. The chunks are the unit of dynamic scheduling: the pool decides
    // which worker runs which chunk, but the chunk boundaries never move.
    const std::size_t grain   = resolve_grain(n, num_threads);
    const std::size_t nchunks = (n + grain - 1) / grain;

    // One padded reduction slot per chunk for the fused sum-of-squares / dot.
    std::vector<PaddedDouble> partial(nchunks);
    PaddedDouble* part = partial.data();

    double rayleigh = 0.0;

    // Build the pool (P workers) ONCE for the whole computation -- the analogue
    // of the threads version spawning its workers once. The main thread acts as
    // the coordinator: it submits each phase's tasks and then wait_completion()s
    // (sleeping until the pool is idle), so it does not itself compute. With
    // -t P this gives ~P active compute threads. Pool construction/teardown is
    // inside the timed region, mirroring thread spawn/join in the threads version.
    const std::string aff = affinity::get_affinity_from_env();   // AFFINITY env
    threadPool pool(num_threads, aff);

    // Phase 2: timed iterative loop.
    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        // Logical row shift for this iteration (same cumulative shift as the
        // sequential reference: updated every EPOCH_LEN iterations).
        const std::size_t epochs = iter / EPOCH_LEN;
        const std::size_t shift  = (shift_rows * epochs) % n;

        // Ping-pong buffers: read from one, write to the other.
        const double* xr = ((iter & 1u) == 0u) ? buf0.data() : buf1.data();
        double*       yw = ((iter & 1u) == 0u) ? buf1.data() : buf0.data();

        // Phase A: shifted SpMV map over the source rows, fused with the local
        // sum-of-squares, submitted as one task per chunk. Disjoint source-row
        // ranges + the (src+shift)%n bijection mean every yw entry is written by
        // exactly one task (no races); each task writes only its own part[c].
        for (std::size_t c = 0; c < nchunks; ++c) {
            const std::size_t lo = c * grain;
            const std::size_t hi = std::min(lo + grain, n);
            pool.submit([rp, ci, va, xr, yw, n, shift, lo, hi, c, part]() {
                // Optimization D: alias the captured pointers to local __restrict__
                // pointers (you cannot qualify a lambda capture directly). The
                // no-alias promise lets the compiler vectorize the scale/store as in
                // the static-threads version.
                const std::uint64_t* __restrict__ RP = rp;
                const std::uint32_t* __restrict__ CI = ci;
                const double*        __restrict__ VA = va;
                const double*        __restrict__ XR = xr;
                double*              __restrict__ YW = yw;
                double ss = 0.0;
                for (std::size_t src = lo; src < hi; ++src) {
                    double sum = 0.0;
                    for (std::uint64_t p = RP[src]; p < RP[src + 1]; ++p) {
                        sum += VA[p] * XR[CI[p]];
                    }
                    const std::size_t out = (src + shift) % n;
                    YW[out] = sum;
                    ss += sum * sum;
                }
                part[c].v = ss;
            });
        }
        pool.wait_completion();   // per-phase barrier: all SpMV tasks finished

        // Global reduction over the per-chunk partials (fixed chunk order).
        double ss = 0.0;
        for (std::size_t c = 0; c < nchunks; ++c) ss += part[c].v;
        const double inv = 1.0 / std::sqrt(ss);

        // Phase B: scale the entries produced this iteration. The output indices
        // (src + shift) % n form a permutation of [0, n), so this scales the
        // whole written buffer.
        for (std::size_t c = 0; c < nchunks; ++c) {
            const std::size_t lo = c * grain;
            const std::size_t hi = std::min(lo + grain, n);
            pool.submit([yw, n, shift, lo, hi, inv]() {
                double* __restrict__ YW = yw;   // Optimization D (no aliasing)
                for (std::size_t src = lo; src < hi; ++src) {
                    const std::size_t out = (src + shift) % n;
                    YW[out] *= inv;
                }
            });
        }
        pool.wait_completion();   // writes visible before the next iteration reads
    }

    // Phase 3: final diagnostics, inside the timed region as in the sequential
    // code. After NUM_ITERS (even) iterations the final vector is in buf0; the
    // extra SpMV uses the final shift and is written into buf1.
    const std::size_t epochs_final = (NUM_ITERS - 1) / EPOCH_LEN;
    const std::size_t shift_final  = (shift_rows * epochs_final) % n;

    const double* xf = buf0.data();   // final x
    double*       ye = buf1.data();   // extra y = A_shifted * x

    for (std::size_t c = 0; c < nchunks; ++c) {
        const std::size_t lo = c * grain;
        const std::size_t hi = std::min(lo + grain, n);
        pool.submit([rp, ci, va, xf, ye, n, shift_final, lo, hi, c, part]() {
            const std::uint64_t* __restrict__ RP = rp;   // Optimization D (no aliasing)
            const std::uint32_t* __restrict__ CI = ci;
            const double*        __restrict__ VA = va;
            const double*        __restrict__ XF = xf;
            double*              __restrict__ YE = ye;
            double local_dot = 0.0;
            for (std::size_t src = lo; src < hi; ++src) {
                double sum = 0.0;
                for (std::uint64_t p = RP[src]; p < RP[src + 1]; ++p) {
                    sum += VA[p] * XF[CI[p]];
                }
                const std::size_t out = (src + shift_final) % n;
                YE[out] = sum;
                local_dot += XF[out] * sum;   // contribution to dot(x, y)
            }
            part[c].v = local_dot;
        });
    }
    pool.wait_completion();

    double dot_xy = 0.0;
    for (std::size_t c = 0; c < nchunks; ++c) dot_xy += part[c].v;
    rayleigh = dot_xy;

    // The final vector is in buf0. The checksum (order independent) is computed
    // here, still inside the timed region as in the sequential reference.
    const std::uint64_t checksum = checksum_vector(buf0);

    if (final_vector != nullptr) {
        *final_vector = std::move(buf0);
    }

    // The pool is destroyed at function return (wait_completion + join), inside
    // the timed region -- the analogue of joining the std::threads.
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
        std::cerr << "  -t   Optional number of pool workers (default: hardware concurrency)\n";
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
    std::cout << "SPARSE_ITERATION_CPP_POOL\n";

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
            iterative_spmv_evolving_pool(G.A, seed, num_threads, final_vector_out);
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
