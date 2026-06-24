//
// C++ threads implementation for the One-Shot project — NUMA LOCALITY VARIANT:
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is a locality-optimized counterpart of the main threads version
// (iterative_SpMV_threads.cpp). It keeps the same algorithm, the same
// nnz-balanced static partition, and all optimizations A-E, and adds a
// MEMORY-LOCALITY preprocessing pass on the CSR matrix:
//
//   F  NUMA-aware parallel first-touch of a private replica of the CSR arrays.
//      The provided generator fills the matrix SERIALLY, so every page of
//      row_ptr/col_idx/values is first-touched by the master thread and, under
//      the Linux first-touch policy, lands on a single NUMA node. On a
//      multi-socket node the pinned workers on the *other* socket then stream
//      the matrix across the interconnect every iteration. Since the roofline
//      study showed K1 is bound by *streaming the matrix from DRAM*, this remote
//      traffic is the dominant avoidable cost. Here each (pinned) worker
//      first-touches and copies ITS OWN nnz-balanced slice of a replica, so the
//      matrix stream becomes node-local AND uses both memory controllers. Because
//      this worker is the same one that later reads that slice (static partition,
//      same pinning), the placement is near-perfect. The replica is bit-identical
//      to the generated matrix, so this is correctness-neutral. (SPM NOTES p.95:
//      first touch is "particularly beneficial for memory-bound and irregular
//      workloads".)
//
// The replica build is an UNTIMED preprocessing step (matrix preparation, like
// the generation itself); the timed region is identical in scope to the main
// threads version (buffer init + partition + iterative loop). x and y are
// L3-resident (0.8-4 MB), so their NUMA placement is negligible and they are
// left exactly as in the baseline.
//
// Tag line SPARSE_ITERATION_CPP_THREADS_NUMA; same CLI/stdout as the baseline so
// scripts/analyze_all.sh parses it unchanged.
//
// Build:
//   g++ -O3 -std=c++20 -pthread -I include -Wall iterative_SpMV_threads_numa.cpp -o threads_numa
//
// Env knobs (no recompile):
//   AFFINITY   CPU list, e.g. "0-31" or "0,2,4-10:2" (default compact core==tid)
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
#include <memory>
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


// Optimization C: thread affinity / pinning (SPM NOTES p.95; course example
// Code/spmcode7). Pinning each worker to a fixed logical CPU stops migration and
// keeps it on the NUMA node that first-touched its data — which is exactly what
// the new locality pass (F) relies on.
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

// Parse a CPU-list string such as "0,2,4-10:2" -> {0,2,4,6,8,10}.
static std::vector<unsigned> parse_cpu_list(const std::string& slist) {
    std::vector<unsigned> cpus;
    std::stringstream ss(slist);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
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

// Resolve the logical CPU that worker `tid` pins to (AFFINITY list, else compact
// core == tid). Used by BOTH the replica builders and the compute workers so
// each worker first-touches and later reads its matrix slice on the same core.
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


// Vector operations (used only for the serial vector initialization).
static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
static double l2_norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}
static void normalize(std::vector<double>& x) {
    const double inv = 1.0 / l2_norm(x);
    for (double& v : x) v *= inv;
}


// Computes the epoch parameter (identical to the sequential reference).
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}


// Cache-line-padded double (Optimization A): keeps each worker's reduction slot
// on its own 64-byte line, so per-iteration writes do not falsely share lines.
struct alignas(64) PaddedDouble {
    double v = 0.0;
};

// Optimization E: std::barrier completion function — the last worker to reach
// the norm barrier reduces the partial sum-of-squares once and publishes the
// inverse L2 norm. noexcept as required by std::barrier.
struct NormReducer {
    const PaddedDouble* partial = nullptr;
    unsigned            P       = 0;
    double*             inv_norm = nullptr;

    void operator()() noexcept {
        double total = 0.0;
        for (unsigned k = 0; k < P; ++k) total += partial[k].v;
        *inv_norm = 1.0 / std::sqrt(total);
    }
};


// NUMA-local replica of the CSR arrays (Optimization F). Allocated UNTOUCHED
// (raw new[], not std::vector — a std::vector ctor would zero-init serially on
// the master thread and defeat the per-worker first touch) and filled in
// parallel by the replica builders below.
struct LocalCSR {
    std::size_t   n   = 0;
    std::uint64_t nnz = 0;
    std::unique_ptr<std::uint64_t[]> row_ptr;   // n + 1 entries
    std::unique_ptr<std::uint32_t[]> col_idx;   // nnz entries
    std::unique_ptr<double[]>        values;    // nnz entries
};


// NNZ-balanced contiguous partition of the source rows [0, n) over P workers,
// from a raw row_ptr prefix-sum array. boundary[t]..boundary[t+1] is worker t's
// source-row range, each carrying roughly equal nonzeros. Same logic as the main
// threads version, on a raw pointer (the replica uses unique_ptr, not vector).
static std::vector<std::size_t> nnz_balanced_partition(const std::uint64_t* row_ptr,
                                                       std::size_t n, unsigned P) {
    const std::uint64_t total_nnz = row_ptr[n];

    std::vector<std::size_t> boundary(P + 1);
    boundary[0] = 0;
    boundary[P] = n;

    for (unsigned t = 1; t < P; ++t) {
        const std::uint64_t target =
            static_cast<std::uint64_t>((static_cast<__uint128_t>(total_nnz) * t) / P);
        const auto it = std::lower_bound(row_ptr, row_ptr + (n + 1), target);
        std::size_t row = static_cast<std::size_t>(it - row_ptr);
        if (row < boundary[t - 1]) row = boundary[t - 1];
        if (row > n) row = n;
        boundary[t] = row;
    }
    for (unsigned t = 1; t <= P; ++t) {
        if (boundary[t] < boundary[t - 1]) boundary[t] = boundary[t - 1];
    }
    boundary[P] = n;
    return boundary;
}


// Optimization F: build a NUMA-local replica of the matrix. Each of the P
// workers pins to its compute core and first-touches/copies exactly its own
// nnz-balanced slice, so the replica's pages land on the node that worker will
// later read from. UNTIMED (matrix preparation, like generation).
static LocalCSR build_numa_local_replica(const CSRMatrix& A,
                                         const std::vector<std::size_t>& boundary,
                                         unsigned P) {
    LocalCSR L;
    L.n   = A.n;
    L.nnz = A.row_ptr[A.n];
    L.row_ptr = std::unique_ptr<std::uint64_t[]>(new std::uint64_t[L.n + 1]);
    L.col_idx = std::unique_ptr<std::uint32_t[]>(new std::uint32_t[L.nnz]);
    L.values  = std::unique_ptr<double[]>(new double[L.nnz]);

    const std::uint64_t* src_rp = A.row_ptr.data();
    const std::uint32_t* src_ci = A.col_idx.data();
    const double*        src_va = A.values.data();
    std::uint64_t* dst_rp = L.row_ptr.get();
    std::uint32_t* dst_ci = L.col_idx.get();
    double*        dst_va = L.values.get();
    const std::size_t n = L.n;

    std::vector<std::thread> builders;
    builders.reserve(P);
    for (unsigned t = 0; t < P; ++t) {
        builders.emplace_back([&, t]() {
            pin_to_core(core_for_worker(t));
            const std::size_t s0 = boundary[t];
            const std::size_t s1 = boundary[t + 1];

            // First-touch this worker's row_ptr slice [s0, s1); the last worker
            // also writes the tail entry row_ptr[n]. Each index written once.
            for (std::size_t i = s0; i < s1; ++i) dst_rp[i] = src_rp[i];
            if (t + 1 == P) dst_rp[n] = src_rp[n];

            // First-touch + copy the owned nnz range in one contiguous pass.
            const std::uint64_t b = src_rp[s0];
            const std::uint64_t e = src_rp[s1];
            for (std::uint64_t p = b; p < e; ++p) {
                dst_ci[p] = src_ci[p];
                dst_va[p] = src_va[p];
            }
        });
    }
    for (std::thread& th : builders) th.join();
    return L;
}


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};


// Per-worker body: runs the whole timed iterative computation for its own
// source-row range [s0, s1). Reads the NUMA-local replica via raw pointers.
static void worker_body(unsigned tid,
                        unsigned P,
                        const std::uint64_t* rp_in,
                        const std::uint32_t* ci_in,
                        const double* va_in,
                        std::size_t n,
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
    // Optimization C: pin to the same core that first-touched this slice (F).
    pin_to_core(core_for_worker(tid));

    // Optimization D: __restrict__ — these arrays and the ping-pong buffers do
    // not alias, enabling tighter scheduling / vectorization.
    const std::uint64_t* __restrict__ rp = rp_in;
    const std::uint32_t* __restrict__ ci = ci_in;
    const double*        __restrict__ va = va_in;

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        const std::size_t epochs = iter / EPOCH_LEN;
        const std::size_t shift  = (shift_rows * epochs) % n;

        const double* __restrict__ xr;
        double*       __restrict__ yw;
        if ((iter & 1u) == 0u) {
            xr = buf0.data();
            yw = buf1.data();
        } else {
            xr = buf1.data();
            yw = buf0.data();
        }

        // Phase A: shifted SpMV map fused with the local sum-of-squares.
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

        // Global reduction point; the completion function publishes inv_norm.
        sync.arrive_and_wait();
        const double inv = inv_norm;

        // Phase B: scale the entries this worker produced.
        for (std::size_t src = s0; src < s1; ++src) {
            const std::size_t out = (src + shift) % n;
            yw[out] *= inv;
        }

        sync.arrive_and_wait();
    }

    // Final diagnostics, inside the timed region as in the sequential code.
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
        local_dot += xf[out] * sum;
    }
    partial_dot[tid].v = local_dot;

    sync.arrive_and_wait();
    if (tid == 0) {
        double total = 0.0;
        for (unsigned k = 0; k < P; ++k) total += partial_dot[k].v;
        rayleigh_out = total;
    }
}


// Timed iterative computation over the NUMA-local replica (raw pointers). Scope
// of the timed region matches the main threads version: serial vector init,
// nnz partition, the iterative loop, and the final diagnostics.
static IterativeResult iterative_spmv_evolving_threads(const std::uint64_t* rp,
                                                       const std::uint32_t* ci,
                                                       const double* va,
                                                       std::size_t n,
                                                       std::uint64_t seed,
                                                       unsigned num_threads,
                                                       std::vector<double>* final_vector) {
    const std::size_t shift_rows = compute_shift_rows(n);

    // Phase 1: serial vector initialization, identical to the sequential
    // reference (so the initial vector, and thus the result, matches).
    std::vector<double> buf0(n);
    std::vector<double> buf1(n, 0.0);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : buf0) v = rng.next_unit();
    normalize(buf0);

    unsigned P = num_threads;
    if (P == 0) P = 1;
    if (P > n) P = static_cast<unsigned>(n);

    const std::vector<std::size_t> boundary = nnz_balanced_partition(rp, n, P);

    std::vector<PaddedDouble> partial(P);
    std::vector<PaddedDouble> partial_dot(P);

    double inv_norm = 0.0;
    std::barrier<NormReducer> sync(P, NormReducer{partial.data(), P, &inv_norm});
    double rayleigh = 0.0;

    std::vector<std::thread> threads;
    threads.reserve(P);
    for (unsigned t = 0; t < P; ++t) {
        threads.emplace_back(worker_body,
                             t, P, rp, ci, va, n, shift_rows,
                             boundary[t], boundary[t + 1],
                             std::ref(buf0), std::ref(buf1),
                             std::ref(partial), std::ref(partial_dot),
                             std::ref(sync), std::cref(inv_norm), std::ref(rayleigh));
    }
    for (std::thread& th : threads) th.join();

    const std::uint64_t checksum = checksum_vector(buf0);

    if (final_vector != nullptr) *final_vector = std::move(buf0);

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = (shift_rows * ((NUM_ITERS - 1) / EPOCH_LEN)) % n
    };
}


int main(int argc, char** argv) {
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
    std::cout << "SPARSE_ITERATION_CPP_THREADS_NUMA\n";

    try {
        // Phase 1: input construction (NOT timed).
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();
        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        // Optimization F: NUMA-local replica via parallel first-touch. Also NOT
        // timed — matrix preparation, like the generation above.
        unsigned P = num_threads;
        if (P == 0) P = 1;
        if (P > n) P = static_cast<unsigned>(n);

        const auto tp0 = std::chrono::steady_clock::now();
        const std::vector<std::size_t> boundary =
            nnz_balanced_partition(G.A.row_ptr.data(), n, P);
        const LocalCSR L = build_numa_local_replica(G.A, boundary, P);
        const auto tp1 = std::chrono::steady_clock::now();
        const double prep_sec = std::chrono::duration<double>(tp1 - tp0).count();

        print_matrix_stats(G);
        std::cout << "threads=" << num_threads << "\n";
        std::cout << "generation_time_sec=" << generation_sec << "\n";
        std::cout << "numa_prep_time_sec=" << prep_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed iterative computation over the local replica.
        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result =
            iterative_spmv_evolving_threads(L.row_ptr.get(), L.col_idx.get(), L.values.get(),
                                            n, seed, num_threads, final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now();
        const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << computation_sec << "\n";

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
