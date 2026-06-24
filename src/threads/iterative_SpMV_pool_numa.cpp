//
// C++ thread-pool implementation — NUMA LOCALITY VARIANT:
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is the NUMA counterpart of the thread-pool comparison variant
// (iterative_SpMV_pool.cpp). It keeps the dynamic, shared-queue scheduling
// unchanged and adds the same memory-locality preprocessing pass as the static
// threads NUMA variant (iterative_SpMV_threads_numa.cpp):
//
//   F  NUMA-aware parallel first-touch of a private CSR replica. The provided
//      generator fills the matrix SERIALLY, so all its pages first-touch on one
//      NUMA node; on a multi-socket node the workers on the other socket stream
//      the matrix remotely. Here P pinned helper threads first-touch + copy
//      contiguous nnz-balanced slices of a replica, spreading the matrix across
//      BOTH NUMA nodes' memory controllers. For a bandwidth-bound kernel that
//      alone roughly doubles the available DRAM bandwidth (the STREAM lesson).
//
//      CAVEAT vs the static version: the pool schedules chunks DYNAMICALLY, so a
//      worker does not necessarily process the slice first-touched on its own
//      node. The replica therefore gives the *spread* benefit (use both memory
//      controllers) but not perfect per-worker locality — expect a smaller gain
//      than the static threads_numa variant. The helper threads are pinned to
//      the same AFFINITY cores the pool uses, which is the best generic placement
//      a dynamic scheduler admits.
//
// The replica build is UNTIMED (matrix preparation, like the generation). The
// replica is bit-identical, so the result is correctness-neutral. Everything
// else (algorithm, init, output) is identical to iterative_SpMV_pool.cpp.
//
// Tag line SPARSE_ITERATION_CPP_POOL_NUMA; same CLI/stdout as the baseline.
//
// Build:
//   g++ -O3 -std=c++20 -pthread -I include
//       src/threads/iterative_SpMV_pool_numa.cpp -o bin/pool_numa
//   (or: make pool_numa)
//

#include <algorithm>
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
#include "threadPool.hpp"   // pulls in taskFactory.hpp and Affinity.hpp


// number of iterations
static constexpr std::uint32_t NUM_ITERS = 500;
// number of iterations between two matrix-evolution steps
static constexpr std::uint32_t EPOCH_LEN = 25;


// --- pinning helpers (for the NUMA first-touch builders) ---------------------
// The pool itself pins its workers from the AFFINITY env string; here we pin the
// first-touch helper threads to the SAME cores so the replica's pages are placed
// across the sockets the pool's workers run on.
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

static unsigned core_for_worker(unsigned tid) {
    static const std::vector<unsigned> cpus = [] {
        const char* env = std::getenv("AFFINITY");
        return (env && *env) ? parse_cpu_list(env) : std::vector<unsigned>{};
    }();
    if (!cpus.empty()) return cpus[tid % cpus.size()];
    return tid;
}


// Vector operations (used only for the serial vector initialization).
static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
static double l2_norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}
static void normalize(std::vector<double>& x) {
    const double nrm = l2_norm(x);
    const double inv = 1.0 / nrm;
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


// Resolve the task grainsize: GRAIN env override, else ~(workers * 8) tasks.
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


// Cache-line-padded double (Optimization A).
struct alignas(64) PaddedDouble {
    double v = 0.0;
};


// NUMA replica of the CSR arrays (Optimization F), allocated untouched and
// first-touched in parallel by pinned helper threads (see header note).
struct LocalCSR {
    std::size_t   n   = 0;
    std::uint64_t nnz = 0;
    std::unique_ptr<std::uint64_t[]> row_ptr;
    std::unique_ptr<std::uint32_t[]> col_idx;
    std::unique_ptr<double[]>        values;
};

// Contiguous nnz-balanced partition of [0, n) over P helpers (raw row_ptr).
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
    for (unsigned t = 1; t <= P; ++t)
        if (boundary[t] < boundary[t - 1]) boundary[t] = boundary[t - 1];
    boundary[P] = n;
    return boundary;
}

static LocalCSR build_numa_replica(const CSRMatrix& A,
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
            for (std::size_t i = s0; i < s1; ++i) dst_rp[i] = src_rp[i];
            if (t + 1 == P) dst_rp[n] = src_rp[n];
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


// Timed iterative computation over the NUMA replica (raw pointers), driven by
// the dynamic thread pool. Identical scheduling to iterative_SpMV_pool.cpp.
static IterativeResult iterative_spmv_evolving_pool(const std::uint64_t* rp,
                                                    const std::uint32_t* ci,
                                                    const double* va,
                                                    std::size_t n,
                                                    std::uint64_t seed,
                                                    unsigned num_threads,
                                                    std::vector<double>* final_vector) {
    const std::size_t shift_rows = compute_shift_rows(n);

    // Phase 1: serial vector initialization, identical to the sequential ref.
    std::vector<double> buf0(n);
    std::vector<double> buf1(n, 0.0);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : buf0) v = rng.next_unit();
    normalize(buf0);

    const std::size_t grain   = resolve_grain(n, num_threads);
    const std::size_t nchunks = (n + grain - 1) / grain;

    std::vector<PaddedDouble> partial(nchunks);
    PaddedDouble* part = partial.data();

    double rayleigh = 0.0;

    const std::string aff = affinity::get_affinity_from_env();   // AFFINITY env
    threadPool pool(num_threads, aff);

    // Phase 2: timed iterative loop.
    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        const std::size_t epochs = iter / EPOCH_LEN;
        const std::size_t shift  = (shift_rows * epochs) % n;

        const double* xr = ((iter & 1u) == 0u) ? buf0.data() : buf1.data();
        double*       yw = ((iter & 1u) == 0u) ? buf1.data() : buf0.data();

        for (std::size_t c = 0; c < nchunks; ++c) {
            const std::size_t lo = c * grain;
            const std::size_t hi = std::min(lo + grain, n);
            pool.submit([rp, ci, va, xr, yw, n, shift, lo, hi, c, part]() {
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
        pool.wait_completion();

        double ss = 0.0;
        for (std::size_t c = 0; c < nchunks; ++c) ss += part[c].v;
        const double inv = 1.0 / std::sqrt(ss);

        for (std::size_t c = 0; c < nchunks; ++c) {
            const std::size_t lo = c * grain;
            const std::size_t hi = std::min(lo + grain, n);
            pool.submit([yw, n, shift, lo, hi, inv]() {
                double* __restrict__ YW = yw;
                for (std::size_t src = lo; src < hi; ++src) {
                    const std::size_t out = (src + shift) % n;
                    YW[out] *= inv;
                }
            });
        }
        pool.wait_completion();
    }

    // Phase 3: final diagnostics, inside the timed region.
    const std::size_t epochs_final = (NUM_ITERS - 1) / EPOCH_LEN;
    const std::size_t shift_final  = (shift_rows * epochs_final) % n;

    const double* xf = buf0.data();
    double*       ye = buf1.data();

    for (std::size_t c = 0; c < nchunks; ++c) {
        const std::size_t lo = c * grain;
        const std::size_t hi = std::min(lo + grain, n);
        pool.submit([rp, ci, va, xf, ye, n, shift_final, lo, hi, c, part]() {
            const std::uint64_t* __restrict__ RP = rp;
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
                local_dot += XF[out] * sum;
            }
            part[c].v = local_dot;
        });
    }
    pool.wait_completion();

    double dot_xy = 0.0;
    for (std::size_t c = 0; c < nchunks; ++c) dot_xy += part[c].v;
    rayleigh = dot_xy;

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
        std::cerr << "  -t   Optional number of pool workers (default: hardware concurrency)\n";
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
    std::cout << "SPARSE_ITERATION_CPP_POOL_NUMA\n";

    try {
        // Phase 1: input construction (NOT timed).
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();
        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        // Optimization F: NUMA replica via parallel first-touch (NOT timed).
        unsigned P = num_threads;
        if (P == 0) P = 1;
        if (P > n) P = static_cast<unsigned>(n);

        const auto tp0 = std::chrono::steady_clock::now();
        const std::vector<std::size_t> boundary =
            nnz_balanced_partition(G.A.row_ptr.data(), n, P);
        const LocalCSR L = build_numa_replica(G.A, boundary, P);
        const auto tp1 = std::chrono::steady_clock::now();
        const double prep_sec = std::chrono::duration<double>(tp1 - tp0).count();

        print_matrix_stats(G);
        std::cout << "threads=" << num_threads << "\n";
        std::cout << "generation_time_sec=" << generation_sec << "\n";
        std::cout << "numa_prep_time_sec=" << prep_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed iterative computation over the replica.
        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result =
            iterative_spmv_evolving_pool(L.row_ptr.get(), L.col_idx.get(), L.values.get(),
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
