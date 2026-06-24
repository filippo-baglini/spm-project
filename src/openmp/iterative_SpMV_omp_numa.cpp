//
// OpenMP (task-based) implementation — NUMA LOCALITY VARIANT:
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is the NUMA counterpart of the OpenMP-tasks deliverable
// (iterative_SpMV_omp.cpp). The taskloop scheduling is unchanged; it adds the
// same memory-locality preprocessing pass as the threads/pool NUMA variants:
//
//   F  NUMA-aware parallel first-touch of a private CSR replica. The provided
//      generator fills the matrix SERIALLY, so all its pages first-touch on one
//      NUMA node; on a multi-socket node the threads on the other socket then
//      stream the matrix remotely every iteration. Here the replica is copied by
//      an `#pragma omp parallel for schedule(static)` so each thread faults a
//      contiguous chunk, spreading the matrix across BOTH NUMA nodes' memory
//      controllers. For a bandwidth-bound kernel that alone roughly doubles the
//      available DRAM bandwidth (the STREAM lesson). Pin the team with
//      OMP_PROC_BIND=close OMP_PLACES=cores so the first-touch placement sticks.
//
//      CAVEAT vs the static threads version: taskloop schedules chunks
//      DYNAMICALLY, so a thread does not necessarily process the chunk it
//      first-touched. The replica gives the *spread* benefit (both memory
//      controllers) but not perfect per-thread locality — expect a smaller gain
//      than the static threads_numa variant.
//
// The replica build is UNTIMED (matrix preparation, like the generation). The
// replica is bit-identical, so the result is correctness-neutral. Everything
// else (algorithm, init, output) is identical to iterative_SpMV_omp.cpp.
//
// Tag line SPARSE_ITERATION_OMP_TASKS_NUMA; same CLI/stdout as the baseline.
//
// Build:
//   g++ -O3 -std=c++20 -pthread -fopenmp -I include
//       src/openmp/iterative_SpMV_omp_numa.cpp -o bin/omp_numa
//   (or: make omp_numa)
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
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "matrix_generation.hpp"
#include "utils.hpp"


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


// Resolve the taskloop grainsize: GRAIN env override, else ~(threads * 8) tasks.
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


// NUMA replica of the CSR arrays (Optimization F), allocated untouched and
// first-touched in parallel by a static-scheduled OpenMP loop (see header note).
struct LocalCSR {
    std::size_t   n   = 0;
    std::uint64_t nnz = 0;
    std::unique_ptr<std::uint64_t[]> row_ptr;
    std::unique_ptr<std::uint32_t[]> col_idx;
    std::unique_ptr<double[]>        values;
};

static LocalCSR build_numa_replica_omp(const CSRMatrix& A, unsigned P) {
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
    const std::size_t   n   = L.n;
    const std::uint64_t nnz = L.nnz;

    // First-touch (and copy) in parallel; schedule(static) gives each thread a
    // contiguous chunk so pages are spread across the team's NUMA nodes. The
    // copy order matches the kernel's read order (row_ptr then the nnz arrays).
    #pragma omp parallel num_threads(P) default(none) \
        shared(dst_rp, src_rp, dst_ci, src_ci, dst_va, src_va, n, nnz)
    {
        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i <= n; ++i) dst_rp[i] = src_rp[i];

        #pragma omp for schedule(static)
        for (std::uint64_t p = 0; p < nnz; ++p) {
            dst_ci[p] = src_ci[p];
            dst_va[p] = src_va[p];
        }
    }
    return L;
}


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};


// Timed iterative computation over the NUMA replica (raw pointers), driven by
// OpenMP taskloops. Identical scheduling to iterative_SpMV_omp.cpp.
static IterativeResult iterative_spmv_evolving_omp(const std::uint64_t* rp,
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

    const std::size_t grain = resolve_grain(n, num_threads);

    double rayleigh = 0.0;

    // Phase 2: timed iterative loop. One parallel region for the whole loop.
    #pragma omp parallel num_threads(num_threads) \
        shared(buf0, buf1, rayleigh)
    {
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            #pragma omp single
            {
                const std::size_t epochs = iter / EPOCH_LEN;
                const std::size_t shift  = (shift_rows * epochs) % n;

                const double* xr = ((iter & 1u) == 0u) ? buf0.data() : buf1.data();
                double*       yw = ((iter & 1u) == 0u) ? buf1.data() : buf0.data();

                double ss = 0.0;
                #pragma omp taskloop grainsize(grain) reduction(+: ss) \
                    default(none) shared(rp, ci, va, xr, yw, n, shift)
                for (std::size_t src = 0; src < n; ++src) {
                    double sum = 0.0;
                    for (std::uint64_t p = rp[src]; p < rp[src + 1]; ++p) {
                        sum += va[p] * xr[ci[p]];
                    }
                    const std::size_t out = (src + shift) % n;
                    yw[out] = sum;
                    ss += sum * sum;
                }

                const double inv = 1.0 / std::sqrt(ss);

                #pragma omp taskloop grainsize(grain) \
                    default(none) shared(yw, n, shift, inv)
                for (std::size_t src = 0; src < n; ++src) {
                    const std::size_t out = (src + shift) % n;
                    yw[out] *= inv;
                }
            } // implicit barrier
        }

        #pragma omp single
        {
            const std::size_t epochs_final = (NUM_ITERS - 1) / EPOCH_LEN;
            const std::size_t shift_final  = (shift_rows * epochs_final) % n;

            const double* xf = buf0.data();
            double*       ye = buf1.data();

            double dot_xy = 0.0;
            #pragma omp taskloop grainsize(grain) reduction(+: dot_xy) \
                default(none) shared(rp, ci, va, xf, ye, n, shift_final)
            for (std::size_t src = 0; src < n; ++src) {
                double sum = 0.0;
                for (std::uint64_t p = rp[src]; p < rp[src + 1]; ++p) {
                    sum += va[p] * xf[ci[p]];
                }
                const std::size_t out = (src + shift_final) % n;
                ye[out] = sum;
                dot_xy += xf[out] * sum;
            }
            rayleigh = dot_xy;
        }
    }

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
        std::cerr << "  -t   Optional number of threads (default: OMP runtime default)\n";
        return 1;
    }

    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);
    if (!read_arg_u64(argc, argv, "-t", threads)) {
        (void)read_arg_u64(argc, argv, "--threads", threads);
    }

    unsigned num_threads = static_cast<unsigned>(threads);
    if (num_threads == 0) {
        num_threads = static_cast<unsigned>(omp_get_max_threads());
        if (num_threads == 0) num_threads = 1;
    }

    const std::size_t n = static_cast<std::size_t>(n64);
    std::cout << "SPARSE_ITERATION_OMP_TASKS_NUMA\n";

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
        const LocalCSR L = build_numa_replica_omp(G.A, P);
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
            iterative_spmv_evolving_omp(L.row_ptr.get(), L.col_idx.get(), L.values.get(),
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
