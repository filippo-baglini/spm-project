//
// Empirical roofline harness for the One-Shot project SpMV kernels.
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is an ANALYSIS tool (not one of the three required deliverables). It
// answers a single question: are the iterative-SpMV kernels actually bound by
// memory bandwidth, or are they sitting well below the roofline (i.e. poor, with
// headroom from NUMA placement / gather latency / vectorization)?
//
// It builds an EMPIRICAL roofline from three self-contained measurements (no
// external tools, no hardware counters):
//
//   (a) peak compute  -- a register-resident FMA microbench (horizontal roof);
//   (b) peak bandwidth -- a STREAM triad over arrays >> LLC, first-touched in
//       parallel so the FULL aggregate (all-NUMA-node) bandwidth is measured
//       (the sloped roof);
//   (c) kernel points  -- the two shared kernels of the iterative loop, timed in
//       isolation with the SAME nnz-balanced static partition the threads version
//       uses, so the achieved (arithmetic-intensity, GFLOP/s) point is faithful:
//         K1: fused shifted SpMV + L2 sum-of-squares   (Phase A, O(nnz))
//         K2: the scaling pass  y *= inv               (Phase B, O(n), streaming)
//
// The matrix arrays / x / y are first-touched SERIALLY here, exactly as the
// sequential and threads versions initialise them -- so the gap between a
// kernel's achieved bandwidth and the (parallel-first-touch) STREAM peak directly
// exposes any NUMA placement penalty.
//
// FLOP / byte model (documented in docs/ROOFLINE.md):
//   K1  FLOPs = R*(2*nnz + 2*n)      Bytes = R*(20*nnz + 16*n)
//       per nonzero: 8 (values) + 4 (col_idx) + 8 (x gather);
//       per row:     8 (row_ptr, amortised) + 8 (write y).
//   K2  FLOPs = R*n                  Bytes = R*(16*n)   (read 8 + write 8)
//   Arithmetic intensity AI = FLOPs / Bytes (x-gather counted uncached: a lower
//   bound on AI; if x is cache-resident the true AI is higher).
//
// Build:
//   g++ -O3 -std=c++20 -march=native -pthread -fopenmp -I include
//       src/bench/roofline_bench.cpp -o bin/roofline
//   (or: make roofline)
//
// Examples:
//   OMP_PROC_BIND=close OMP_PLACES=cores ./bin/roofline -n 500000 -nz 20000000 -t 32
//   ./bin/roofline -n 200000 -nz 4000000 -m irregular -t 4 -R 50
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <omp.h>
#include <unistd.h>   // sysconf cache sizes

#include "matrix_generation.hpp"
#include "utils.hpp"


// utils.hpp provides several `static` helpers (checksum_vector, dump_vector,
// usage) that this analysis tool does not call. Referencing their addresses marks
// them used, so the immutable header compiles without -Wunused-function noise.
[[maybe_unused]] static const void* const g_utils_unused[] = {
    reinterpret_cast<const void*>(&checksum_vector),
    reinterpret_cast<const void*>(&dump_vector),
    reinterpret_cast<const void*>(&usage),
};

// Volatile sink: reading/accumulating results here stops the optimizer from
// deleting the benchmarked loops as dead code.
static volatile double g_sink = 0.0;

using clk = std::chrono::steady_clock;
static double secs_since(const clk::time_point& t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}


// nnz-balanced contiguous partition of source rows [0, n) into P ranges of
// approximately equal nonzero count (identical to the threads version).
static std::vector<std::size_t> nnz_balanced_partition(const CSRMatrix& A, unsigned P) {
    const std::size_t n = A.n;
    const std::uint64_t total_nnz = A.row_ptr[n];
    std::vector<std::size_t> boundary(P + 1);
    boundary[0] = 0;
    boundary[P] = n;
    for (unsigned t = 1; t < P; ++t) {
        const std::uint64_t target =
            static_cast<std::uint64_t>((static_cast<__uint128_t>(total_nnz) * t) / P);
        const auto it = std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), target);
        std::size_t row = static_cast<std::size_t>(it - A.row_ptr.begin());
        if (row < boundary[t - 1]) row = boundary[t - 1];
        if (row > n) row = n;
        boundary[t] = row;
    }
    for (unsigned t = 1; t <= P; ++t)
        if (boundary[t] < boundary[t - 1]) boundary[t] = boundary[t - 1];
    boundary[P] = n;
    return boundary;
}


// (a) Peak compute: independent FMA chains, fully register-resident, no memory
// traffic. Enough chains to saturate the FMA units' throughput. b/c are runtime
// values so the recurrence cannot be constant-folded away. Returns GFLOP/s.
static double bench_peak_fma(unsigned P, long iters, double b, double c) {
    // 64 independent chains: each a[k] = a[k]*b + c is a latency chain (~4-5 cyc),
    // so we need many in flight (>= FMA_latency * FMA_units * SIMD width) to
    // saturate throughput. This is an empirical LOWER BOUND on peak FLOP/s; it is
    // not the binding roof here anyway, since both kernels' AI is far below the
    // ridge -- they are bandwidth-bound regardless of the exact compute peak.
    constexpr int NACC = 64;
    const auto t0 = clk::now();
    double grand = 0.0;
    #pragma omp parallel num_threads(P) reduction(+: grand)
    {
        double a[NACC];
        for (int k = 0; k < NACC; ++k) a[k] = 0.5 + 0.01 * k;
        for (long it = 0; it < iters; ++it) {
            #pragma omp simd
            for (int k = 0; k < NACC; ++k) a[k] = a[k] * b + c;
        }
        double s = 0.0;
        for (int k = 0; k < NACC; ++k) s += a[k];
        grand += s;
    }
    const double t = secs_since(t0);
    g_sink += grand;
    // 2 FLOPs (1 mul + 1 add) per chain per iteration, NACC chains, P threads.
    return 2.0 * static_cast<double>(NACC) * static_cast<double>(iters) *
           static_cast<double>(P) / t / 1e9;
}


// (b) Peak bandwidth: STREAM triad a = b + s*c, arrays >> LLC, parallel
// first-touch (NUMA-correct => full aggregate bandwidth). Returns GB/s
// (STREAM convention: 24 bytes/element = read b + read c + write a).
//
// CRITICAL for NUMA: the arrays MUST be allocated without being touched (so no
// pages are faulted yet) and then first-touched IN PARALLEL by the same static
// schedule the triad uses -- only then does each page land on the node of the
// thread that will use it. Using std::vector here would be a bug: its ctor
// value-initializes (zeroes) the whole array on the master thread, faulting every
// page onto one node and capping the measured bandwidth at a single socket.
static double bench_stream_triad(unsigned P, std::size_t N, int reps) {
    // new double[N] (no parens) default-initializes scalars => leaves the memory
    // UNtouched (no pages faulted) until the parallel first-touch below.
    std::unique_ptr<double[]> A(new double[N]);
    std::unique_ptr<double[]> B(new double[N]);
    std::unique_ptr<double[]> C(new double[N]);
    double* __restrict__ a = A.get();
    double* __restrict__ b = B.get();
    double* __restrict__ c = C.get();
    const double scalar = 3.0;

    #pragma omp parallel for num_threads(P) schedule(static)
    for (std::size_t i = 0; i < N; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }

    const auto t0 = clk::now();
    for (int r = 0; r < reps; ++r) {
        #pragma omp parallel for num_threads(P) schedule(static)
        for (std::size_t i = 0; i < N; ++i) a[i] = b[i] + scalar * c[i];
    }
    const double t = secs_since(t0);
    g_sink += a[N / 2];
    return 24.0 * static_cast<double>(N) * static_cast<double>(reps) / t / 1e9;
}


// Print the CPU data-cache sizes (glibc sysconf; "n/a" where unsupported) so the
// sweep plateaus below can be labelled L1/L2/L3.
static void print_cache_sizes() {
    const long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long l3 = -1;
#if defined(_SC_LEVEL3_CACHE_SIZE)
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    std::printf("detected data caches (per-core L1/L2, shared L3):");
    if (l1 > 0) std::printf("  L1d=%ld KB", l1 / 1024); else std::printf("  L1d=n/a");
    if (l2 > 0) std::printf("  L2=%ld KB",  l2 / 1024); else std::printf("  L2=n/a");
    if (l3 > 0) std::printf("  L3=%ld KB",  l3 / 1024); else std::printf("  L3=n/a");
    std::printf("\n\n");
}


// (b') Cache-hierarchy bandwidth sweep: run the triad over PER-THREAD private
// buffers of growing size S. When 3*S fits in a core's private cache the triad
// streams from that level; as S grows it spills L1 -> L2 -> L3 -> DRAM, so the
// achieved bandwidth plateaus reveal each level's bandwidth. These plateaus are
// the sloped ceilings of the cache-aware (hierarchical) roofline. Prints a table.
//
// Each thread allocates and first-touches its OWN buffers inside the parallel
// region (so placement is per-core/per-node correct), then times only the triad
// passes. The triad scalar varies per pass so the compiler cannot hoist the
// (otherwise identical) passes out of the loop.
static void bench_cache_sweep(unsigned P) {
    static const std::size_t sizes[] = {   // per-thread elements (doubles): 2 KB .. 32 MB/buffer
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304
    };
    const double TARGET = 3.0e8;   // ~triad element-updates per thread per size (bounds time)

    std::printf("=== cache-hierarchy bandwidth sweep (threads=%u) ===\n", P);
    std::printf("%12s %12s %12s\n", "buf/thread", "footprint", "GB/s");
    std::printf("------------ ------------ ------------\n");
    for (std::size_t S : sizes) {
        std::size_t passes = static_cast<std::size_t>(TARGET / static_cast<double>(S));
        if (passes < 1) passes = 1;

        double max_t = 0.0;
        #pragma omp parallel num_threads(P) reduction(max: max_t)
        {
            std::vector<double> A(S), B(S), C(S);   // per-thread; first-touched on this thread
            for (std::size_t i = 0; i < S; ++i) { A[i] = 0.0; B[i] = 1.0; C[i] = 2.0; }
            double*       __restrict__ a = A.data();
            const double* __restrict__ b = B.data();
            const double* __restrict__ c = C.data();
            #pragma omp barrier                      // all threads start the timed region together
            const auto t0 = clk::now();
            for (std::size_t p = 0; p < passes; ++p) {
                const double s = 3.0 + static_cast<double>(p) * 1e-15;  // vary => no hoist
                for (std::size_t i = 0; i < S; ++i) a[i] = b[i] + s * c[i];
            }
            max_t = secs_since(t0);
            #pragma omp atomic
            g_sink += a[S / 2];
        }

        const double gb = 24.0 * static_cast<double>(S) * static_cast<double>(passes) *
                          static_cast<double>(P) / max_t / 1e9;
        const double buf_kb  = static_cast<double>(S) * 8.0 / 1024.0;
        const double foot_mb = 3.0 * static_cast<double>(S) * 8.0 *
                               static_cast<double>(P) / 1048576.0;
        std::printf("%9.1f KB %9.1f MB %12.2f\n", buf_kb, foot_mb, gb);
    }
    std::printf("\n");
}


struct KernelResult {
    double seconds = 0.0;
    double gflops  = 0.0;
    double gbytes  = 0.0;
    double ai      = 0.0;
};

// (c) K1: fused shifted SpMV + sum-of-squares, R reps, nnz-balanced ranges,
// __restrict__ (Opt D), pinned via OMP_PROC_BIND. Serial first-touch of x/y/A.
static KernelResult bench_spmv(const CSRMatrix& A,
                               const std::vector<double>& x,
                               std::vector<double>& y,
                               const std::vector<std::size_t>& bnd,
                               unsigned P, int R) {
    const std::size_t n = A.n;
    const std::uint64_t nnz = A.row_ptr[n];
    const std::size_t shift = 0;  // pattern is shift-invariant; fix for the bench

    const auto t0 = clk::now();
    for (int r = 0; r < R; ++r) {
        #pragma omp parallel num_threads(P)
        {
            const int tid = omp_get_thread_num();
            const std::size_t s0 = bnd[tid], s1 = bnd[tid + 1];
            const std::uint64_t* __restrict__ RP = A.row_ptr.data();
            const std::uint32_t* __restrict__ CI = A.col_idx.data();
            const double*        __restrict__ VA = A.values.data();
            const double*        __restrict__ X  = x.data();
            double*              __restrict__ Y  = y.data();
            double ss = 0.0;
            for (std::size_t src = s0; src < s1; ++src) {
                double sum = 0.0;
                for (std::uint64_t p = RP[src]; p < RP[src + 1]; ++p)
                    sum += VA[p] * X[CI[p]];
                const std::size_t out = (src + shift) % n;
                Y[out] = sum;
                ss += sum * sum;
            }
            #pragma omp atomic
            g_sink += ss;
        }
    }
    const double t = secs_since(t0);
    KernelResult kr;
    kr.seconds = t;
    const double flops = static_cast<double>(R) * (2.0 * nnz + 2.0 * n);
    const double bytes = static_cast<double>(R) * (20.0 * nnz + 16.0 * n);
    kr.gflops = flops / t / 1e9;
    kr.gbytes = bytes / t / 1e9;
    kr.ai     = flops / bytes;
    return kr;
}

// (c) K2: the scaling pass y *= inv, R reps, same nnz-balanced ranges.
static KernelResult bench_scale(const CSRMatrix& A,
                                std::vector<double>& y,
                                const std::vector<std::size_t>& bnd,
                                unsigned P, int R) {
    const std::size_t n = A.n;
    const std::size_t shift = 0;
    const double inv = 0.9999999;

    const auto t0 = clk::now();
    for (int r = 0; r < R; ++r) {
        #pragma omp parallel num_threads(P)
        {
            const int tid = omp_get_thread_num();
            const std::size_t s0 = bnd[tid], s1 = bnd[tid + 1];
            double* __restrict__ Y = y.data();
            for (std::size_t src = s0; src < s1; ++src) {
                const std::size_t out = (src + shift) % n;
                Y[out] *= inv;
            }
        }
    }
    const double t = secs_since(t0);
    g_sink += y[n / 2];
    KernelResult kr;
    kr.seconds = t;
    const double flops = static_cast<double>(R) * static_cast<double>(n);
    const double bytes = static_cast<double>(R) * 16.0 * static_cast<double>(n);
    kr.gflops = flops / t / 1e9;
    kr.gbytes = bytes / t / 1e9;
    kr.ai     = flops / bytes;
    return kr;
}


static void print_kernel_row(const char* name, const KernelResult& kr,
                             double peakF, double peakBW) {
    // Attainable performance at this kernel's AI: min(peak compute, BW*AI).
    const double roof = std::min(peakF, peakBW * kr.ai);
    const double pct  = (roof > 0.0) ? 100.0 * kr.gflops / roof : 0.0;
    std::printf("%-4s %8.4f %10.4f %9.2f %9.2f %12.2f %7.1f%%\n",
                name, kr.ai, kr.gflops, kr.gbytes, kr.seconds, roof, pct);
}


int main(int argc, char** argv) {
    std::uint64_t n64 = 0, nz = 0, seed = 111, threads = 0, reps = 0, streamN = 0, fmaIters = 0;
    std::string mode;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode)) {
        std::fprintf(stderr,
            "Usage: %s -n N -nz K -m regular|irregular [-s seed] [-t threads]\n"
            "          [-R kernel_reps] [--stream-n ELEMS] [--fma-iters ITERS]\n", argv[0]);
        return 1;
    }
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_u64(argc, argv, "-t", threads);
    (void)read_arg_u64(argc, argv, "-R", reps);
    (void)read_arg_u64(argc, argv, "--stream-n", streamN);
    (void)read_arg_u64(argc, argv, "--fma-iters", fmaIters);

    bool do_cache_sweep = false;   // bare flag: scan argv directly
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--cache-sweep") do_cache_sweep = true;

    unsigned P = static_cast<unsigned>(threads);
    if (P == 0) P = static_cast<unsigned>(omp_get_max_threads());
    if (P == 0) P = 1;
    const int    R    = (reps     > 0) ? static_cast<int>(reps)     : 50;
    const std::size_t SN = (streamN > 0) ? static_cast<std::size_t>(streamN)
                                         : (std::size_t{1} << 25);   // ~33.5M doubles
    const long FI = (fmaIters > 0) ? static_cast<long>(fmaIters) : 200000000L;

    const std::size_t n = static_cast<std::size_t>(n64);

    std::printf("ROOFLINE_BENCH\n");
    std::printf("n=%zu  nz=%llu  mode=%s  seed=%llu  threads=%u  kernel_reps=%d\n",
                n, (unsigned long long)nz, mode.c_str(), (unsigned long long)seed, P, R);

    try {
        // Matrix + vectors: built/initialised SERIALLY (untimed), exactly like the
        // sequential and threads versions (so NUMA placement matches them).
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const CSRMatrix& A = G.A;
        const std::uint64_t nnz = A.row_ptr[n];
        print_matrix_stats(G);
        std::printf("nnz_per_row_avg=%.2f\n\n", static_cast<double>(nnz) / static_cast<double>(n));

        std::vector<double> x(n), y(n, 0.0);
        SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
        for (double& v : x) v = rng.next_unit();

        // (a) + (b): node ceilings.
        const double peakF  = bench_peak_fma(P, FI, 1.0000001, 0.0000001);
        const double peakBW = bench_stream_triad(P, SN, 20);
        const double ridge  = (peakBW > 0.0) ? peakF / peakBW : 0.0;

        std::printf("=== node ceilings (threads=%u) ===\n", P);
        std::printf("peak compute  = %8.2f GFLOP/s   (register FMA microbench)\n", peakF);
        std::printf("peak bandwidth= %8.2f GB/s      (STREAM triad, parallel first-touch)\n", peakBW);
        std::printf("ridge point AI= %8.3f FLOP/byte (peak compute / peak bandwidth)\n\n", ridge);

        // (c): kernels.
        const std::vector<std::size_t> bnd = nnz_balanced_partition(A, P);
        const KernelResult k1 = bench_spmv(A, x, y, bnd, P, R);
        const KernelResult k2 = bench_scale(A, y, bnd, P, R);

        std::printf("=== kernels (achieved vs roofline) ===\n");
        std::printf("%-4s %8s %10s %9s %9s %12s %8s\n",
                    "krn", "AI(F/B)", "GFLOP/s", "GB/s", "time(s)", "roofGFLOP/s", "%roof");
        std::printf("---- -------- ---------- --------- --------- ------------ --------\n");
        print_kernel_row("K1", k1, peakF, peakBW);
        print_kernel_row("K2", k2, peakF, peakBW);

        // K1 arithmetic intensity is a BAND: the table row above counts the x
        // gather as uncached (20 B/nonzero) => lowest AI, highest implied DRAM
        // traffic. The vector x has only n elements (<= a few MB) and is heavily
        // reused, so in practice it is largely cache-resident; then the real DRAM
        // traffic is ~12 B/nonzero and the effective AI is higher. The truth lies
        // between these bounds; report the cached-x end too.
        {
            const double flops    = static_cast<double>(R) * (2.0 * nnz + 2.0 * n);
            const double bytes_hi = static_cast<double>(R) * (12.0 * nnz + 16.0 * n); // x cached
            const double ai_hi    = flops / bytes_hi;
            const double gb_hi    = bytes_hi / k1.seconds / 1e9;          // real DRAM est.
            const double roof_hi  = std::min(peakF, peakBW * ai_hi);
            const double pct_hi   = (roof_hi > 0.0) ? 100.0 * k1.gflops / roof_hi : 0.0;
            std::printf("K1 AI band: uncached-x AI=%.4f (row above) .. cached-x AI=%.4f "
                        "=> est. DRAM %.2f GB/s, %.1f%% of roof\n",
                        k1.ai, ai_hi, gb_hi, pct_hi);
        }

        std::printf("\nK1 = fused shifted SpMV + L2 sum-of-squares (Phase A, dominant)\n");
        std::printf("K2 = scaling pass y*=inv (Phase B, pure streaming)\n");
        std::printf("roofGFLOP/s = min(peak compute, peak BW * AI);  "
                    "%%roof = achieved / roofGFLOP/s\n");
        std::printf("Both AIs are far left of the ridge => bandwidth-bound regime; "
                    "%%roof near 100%% => at the memory wall, well below => headroom.\n");
        // g_sink is volatile, so every accumulation above is retained (no DCE).

        // Optional: cache-hierarchy bandwidth sweep for the cache-aware roofline.
        if (do_cache_sweep) {
            std::printf("\n");
            print_cache_sizes();
            bench_cache_sweep(P);
            std::printf("(largest-footprint GB/s should match the STREAM peak above; "
                        "small buffers expose L1/L2/L3 bandwidth)\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
