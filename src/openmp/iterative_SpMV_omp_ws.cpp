//
// OpenMP (work-sharing) implementation for the One-Shot project:
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is the OPTIONAL OpenMP variant the assignment allows ("an additional
// OpenMP implementation based on work-sharing constructs"), provided ONLY as a
// comparison point for the required task-based version
// (src/openmp/iterative_SpMV_omp.cpp). It is a parallel counterpart of the
// sequential reference and preserves the same algorithmic structure, matrix
// generation and vector initialization, so results match the sequential baseline
// within a numerical tolerance (the strict bitwise checksum may differ because
// the parallel reduction combines partial sums in a different order; expected and
// acceptable per the project's tolerance rule, exactly as for the task version).
//
// Parallelization strategy (OpenMP work-sharing)
// ----------------------------------------------
//   * One persistent parallel region (#pragma omp parallel) wraps the whole timed
//     iterative loop, so the thread team is created once -- identical to the
//     task-based version, for a fair head-to-head comparison.
//   * Each iteration's two phases are WORK-SHARING loops (#pragma omp for) rather
//     than `taskloop` tasks:
//       - Phase A: the shifted SpMV map fused with the L2-norm sum-of-squares,
//         expressed as `#pragma omp for reduction(+: ss)`.
//       - Phase B: the scaling map, a plain `#pragma omp for`.
//     The implicit barrier at the end of each `omp for` synchronizes the team
//     (so the buffer written this iteration is visible before the next reads it).
//   * Load balancing / granularity is controlled by the OpenMP `schedule` clause.
//     We use `schedule(runtime)` so the policy (static | dynamic[,chunk] | guided)
//     is selected at run time via the OMP_SCHEDULE environment variable, with NO
//     recompile -- the work-sharing analogue of the task version's GRAIN knob, and
//     the natural axis for the report's granularity analysis. The resolved
//     schedule is printed (omp_get_schedule) so each run is self-documenting.
//
//     NOTE: `schedule(static)` splits the loop by equal ROW COUNT, which is
//     imbalanced on the irregular matrix (heavy rows cluster at the top); the
//     irregular workload is exactly why `dynamic`/`guided` (or the task version's
//     dynamic taskloop) matter. The C++ threads version solves this differently,
//     with a one-time nnz-balanced STATIC partition.
//
//   * Two ping-pong buffers replace the explicit x/y swap; the logical row shift
//     of the evolving matrix is derived locally from the iteration index. The
//     per-iteration reduction variable `ss` is shared in the team and reset by a
//     `#pragma omp single` before each Phase A (a `+` reduction combines into the
//     variable's existing value, so it must start at 0).
//
// Thread affinity is expressed through the standard OpenMP environment, e.g.
//   OMP_PROC_BIND=close OMP_PLACES=cores
//
// Command line (same as the sequential / threads / task versions):
//   -n N  -nz K  -m regular|irregular  [-s seed]  [-t threads]  [--dump-vector FILE]
//
// Build:
//   g++ -O3 -std=c++20 -pthread -fopenmp -I include
//       src/openmp/iterative_SpMV_omp_ws.cpp -o bin/omp_ws
//   (or: make omp_ws)
//
// Examples:
//   OMP_SCHEDULE=static       ./bin/omp_ws -n 500000 -nz 20000000 -m irregular -t 16
//   OMP_SCHEDULE=dynamic,2048 ./bin/omp_ws -n 500000 -nz 20000000 -m irregular -t 16
//   OMP_SCHEDULE=guided       ./bin/omp_ws -n 100000 -nz 4000000  -m irregular -t 8
//

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
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


// Human-readable form of the OpenMP runtime schedule (set via OMP_SCHEDULE), for
// the report: "<kind>,<chunk>". The monotonic/nonmonotonic modifier bit is masked
// off before classifying the kind.
static std::string describe_schedule() {
    omp_sched_t kind;
    int chunk = 0;
    omp_get_schedule(&kind, &chunk);
    const int base = static_cast<int>(kind) & ~static_cast<int>(omp_sched_monotonic);
    std::string name;
    switch (base) {
        case omp_sched_static:  name = "static";  break;
        case omp_sched_dynamic: name = "dynamic"; break;
        case omp_sched_guided:  name = "guided";  break;
        case omp_sched_auto:    name = "auto";    break;
        default:                name = "unknown(" + std::to_string(base) + ")"; break;
    }
    return name + "," + std::to_string(chunk);
}


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};


static IterativeResult iterative_spmv_evolving_omp_ws(const CSRMatrix& A,
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

    // CSR arrays, hoisted out of the loop.
    const std::uint64_t* rp = A.row_ptr.data();
    const std::uint32_t* ci = A.col_idx.data();
    const double*        va = A.values.data();

    // Shared reduction targets (a work-sharing-loop reduction requires the list
    // item to be shared in the enclosing parallel region).
    double ss       = 0.0;
    double dot_xy   = 0.0;
    double rayleigh = 0.0;

    // Phase 2: timed iterative loop. One parallel region for the whole loop;
    // every thread participates in each iteration's work-sharing loops.
    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(buf0, buf1, rp, ci, va, n, shift_rows, ss, dot_xy, rayleigh)
    {
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            // Logical row shift for this iteration (same cumulative shift as the
            // sequential reference). Derived independently by every thread from
            // the shared loop index -- identical values, read-only, no race.
            const std::size_t epochs = iter / EPOCH_LEN;
            const std::size_t shift  = (shift_rows * epochs) % n;

            // Ping-pong buffers: read from one, write to the other.
            const double* xr = ((iter & 1u) == 0u) ? buf0.data() : buf1.data();
            double*       yw = ((iter & 1u) == 0u) ? buf1.data() : buf0.data();

            // Reset the shared sum-of-squares before the reduction (a `+`
            // reduction adds into the existing value). single => implicit barrier.
            #pragma omp single
            {
                ss = 0.0;
            }

            // Phase A: shifted SpMV map over source rows, fused with the L2-norm
            // sum-of-squares, as a work-sharing reduction. Implicit barrier at the
            // end makes `ss` final and the writes to `yw` visible.
            #pragma omp for schedule(runtime) reduction(+: ss)
            for (std::size_t src = 0; src < n; ++src) {
                double sum = 0.0;
                for (std::uint64_t p = rp[src]; p < rp[src + 1]; ++p) {
                    sum += va[p] * xr[ci[p]];
                }
                const std::size_t out = (src + shift) % n;
                yw[out] = sum;
                ss += sum * sum;
            }

            // Every thread reads the now-final shared `ss` (post-barrier) and
            // derives the identical scale factor.
            const double inv = 1.0 / std::sqrt(ss);

            // Phase B: scale the entries produced this iteration. The output
            // indices (src + shift) % n form a permutation of [0, n), so this
            // scales the whole written buffer. Implicit barrier before next iter.
            #pragma omp for schedule(runtime)
            for (std::size_t src = 0; src < n; ++src) {
                const std::size_t out = (src + shift) % n;
                yw[out] *= inv;
            }
        }

        // Phase 3: final diagnostics, inside the timed region as in the sequential
        // code. After NUM_ITERS (even) iterations the final vector is in buf0; the
        // extra SpMV uses the final shift and is written into buf1.
        const std::size_t epochs_final = (NUM_ITERS - 1) / EPOCH_LEN;
        const std::size_t shift_final  = (shift_rows * epochs_final) % n;

        const double* xf = buf0.data();   // final x
        double*       ye = buf1.data();   // extra y = A_shifted * x

        #pragma omp single
        {
            dot_xy = 0.0;
        }
        #pragma omp for schedule(runtime) reduction(+: dot_xy)
        for (std::size_t src = 0; src < n; ++src) {
            double sum = 0.0;
            for (std::uint64_t p = rp[src]; p < rp[src + 1]; ++p) {
                sum += va[p] * xf[ci[p]];
            }
            const std::size_t out = (src + shift_final) % n;
            ye[out] = sum;
            dot_xy += xf[out] * sum;   // contribution to dot(x, y)
        }
        #pragma omp single
        {
            rayleigh = dot_xy;
        }
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
        std::cerr << "  -t   Optional number of threads (default: OMP runtime default)\n";
        std::cerr << "  Schedule/granularity is set via OMP_SCHEDULE (e.g. static, dynamic,2048, guided)\n";
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
        num_threads = static_cast<unsigned>(omp_get_max_threads());
        if (num_threads == 0) num_threads = 1;
    }

    const std::size_t n = static_cast<std::size_t>(n64);
    std::cout << "SPARSE_ITERATION_OMP_WORKSHARING\n";

    try {
        // Phase 1: input construction (not included in computation time).
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();

        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "threads=" << num_threads << "\n";
        std::cout << "schedule=" << describe_schedule() << "\n";
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed iterative computation.
        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result =
            iterative_spmv_evolving_omp_ws(G.A, seed, num_threads, final_vector_out);
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
