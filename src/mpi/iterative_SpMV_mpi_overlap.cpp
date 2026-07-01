//
// Hybrid MPI + OpenMP implementation -- OVERLAP variant (deliverable #3, b).
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// This is the communication/computation-OVERLAPPED counterpart of
// src/mpi/iterative_SpMV_mpi.cpp. The algorithm, the matrix distribution, the
// shift-as-rotation, and the distributed-Allreduce norm are IDENTICAL -- the ONLY
// difference is how the per-iteration result vector is gathered:
//
//   * Blocking baseline (iterative_SpMV_mpi.cpp):
//       compute ALL local rows  ->  one blocking MPI_Allgatherv.
//     The whole SpMV runs, THEN the whole O(n) gather runs: the two largest
//     costs are strictly serial.
//
//   * Overlap variant (THIS file):
//       split the local rows into CHUNKS; as soon as a chunk's z_local is
//       computed, fire a NON-BLOCKING MPI_Iallgatherv for it and keep computing
//       the next chunk while that gather is in flight. A final MPI_Waitall
//       drains the pipeline. The SpMV of chunk c overlaps the communication of
//       chunks < c, so the per-iteration cost trends toward max(compute, comm)
//       instead of compute + comm.
//
// Why this is the only overlap available here: the iteration is a strict
// dependency chain (SpMV -> gather -> rotate -> normalize -> next SpMV), so
// iteration k's gather cannot overlap iteration k+1's SpMV. All overlap must be
// INTRA-iteration, between the SpMV compute and the gather comm -- which is
// exactly what the chunked pipeline does. The classic "local vs remote column"
// SpMV overlap does not help: the matrix has random columns, so almost every
// local row needs remote x entries (nothing local to hide behind).
//
// MPI progress under MPI_THREAD_FUNNELED: only the main thread calls MPI, and it
// is busy inside the OpenMP region while a chunk computes, so a background
// Iallgatherv may not advance until the next MPI call. We DO NOT spin up a
// progress thread (that needs MPI_THREAD_MULTIPLE, which the SPM notes do not
// cover); instead we poll with MPI_Testall between chunks (notes p.139), which
// nudges progress at every chunk boundary. The realistic win is therefore the
// eager/early portion of each transfer overlapping the next chunk's compute --
// largest in the moderate-rank regime where compute >= comm, and tapering off at
// high node counts where comm dominates (the Allgatherv strong-scaling floor).
//
// Grounding (SPM NOTES): "All collectives have a non-blocking version" (p.141,
// MPI_Ibarrier shown); non-blocking calls "let you overlap communication and
// computation" + MPI_Wait/MPI_Test(all) (p.138-139); double buffering / async to
// "maximize the overlap of computation and communication" (p.140). Same templates
// as the baseline: Code/spmcode11/mpi_power_iteration_partitioned.cpp,
// Code/spmcode12/mpi_trapezoid+omp.cpp.
//
// Correctness: the chunked Iallgatherv reassembles the SAME natural-order vector
// z into the SAME positions of zf, so the result is bitwise identical to the
// blocking baseline for any rank x thread x node x chunk count -- CHUNKS is a
// tuning knob only. (Like the baseline it matches the sequential within tolerance,
// not bit-for-bit, since the L2 norm is an MPI_Allreduce, not a canonical-order
// sum -- see the baseline's header for why.)
//
// Command line (baseline args + one extra):
//   -n N  -nz K  -m regular|irregular  [-s seed]  [-t threads_per_rank]
//   [-c chunks]            (#pipeline chunks per gather; default 4; 1 == blocking-like)
//   [--dump-vector FILE]
//
// Build (cluster):
//   mpicxx -O3 -std=c++20 -fopenmp -I include
//          src/mpi/iterative_SpMV_mpi_overlap.cpp -o bin/mpi_overlap
//   (or: make mpi_overlap)
//
// Run:
//   mpirun -np 4 ./bin/mpi_overlap -n 500000 -nz 20000000 -m irregular -t 8 -c 4
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "matrix_generation.hpp"
#include "utils.hpp"


// number of iterations
static constexpr std::uint32_t NUM_ITERS = 500;
// number of iterations between two matrix-evolution steps
static constexpr std::uint32_t EPOCH_LEN = 25;


// Abort on any MPI error (course helper, Code/spmcode11).
static void check_mpi(int err, const std::string& where) {
    if (err == MPI_SUCCESS) return;
    char msg[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(err, msg, &len);
    std::cerr << "MPI error in " << where << ": " << std::string(msg, len) << '\n';
    MPI_Abort(MPI_COMM_WORLD, err);
}

template <typename T>
static T* buffer_ptr(std::vector<T>& v) { return v.empty() ? nullptr : v.data(); }
template <typename T>
static const T* buffer_ptr(const std::vector<T>& v) { return v.empty() ? nullptr : v.data(); }


// Serial vector ops for the initial-vector setup (identical to the sequential
// reference, so the initial vector matches bit-for-bit on every rank).
static double l2_norm(const std::vector<double>& x) {
    return std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
}
static void normalize(std::vector<double>& x) {
    const double inv = 1.0 / l2_norm(x);
    for (double& v : x) v *= inv;
}

// Epoch parameter (identical to the sequential reference).
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// NNZ-balanced contiguous partition of source rows [0, n) over `size` ranks,
// from a raw row_ptr prefix-sum array. boundary[r]..boundary[r+1] is rank r's
// row block. Same logic as the threads/NUMA and blocking-MPI versions.
static std::vector<std::size_t> nnz_balanced_partition(const std::uint64_t* row_ptr,
                                                       std::size_t n, int size) {
    const std::uint64_t total_nnz = row_ptr[n];
    std::vector<std::size_t> boundary(size + 1);
    boundary[0] = 0;
    boundary[size] = n;
    for (int r = 1; r < size; ++r) {
        const std::uint64_t target =
            static_cast<std::uint64_t>((static_cast<__uint128_t>(total_nnz) * r) / size);
        const auto it = std::lower_bound(row_ptr, row_ptr + (n + 1), target);
        std::size_t row = static_cast<std::size_t>(it - row_ptr);
        if (row < boundary[r - 1]) row = boundary[r - 1];
        if (row > n) row = n;
        boundary[r] = row;
    }
    for (int r = 1; r <= size; ++r)
        if (boundary[r] < boundary[r - 1]) boundary[r] = boundary[r - 1];
    boundary[size] = n;
    return boundary;
}


// This rank's distributed slice of the matrix, plus the row counts/displs used
// to reassemble the full result vector with the (I)Allgatherv.
struct DistCSR {
    std::size_t r0 = 0;           // first global source row owned by this rank
    int local_rows = 0;           // number of source rows owned
    std::vector<std::uint64_t> lrp;   // local row_ptr, rebased to 0 (local_rows+1)
    std::vector<std::uint32_t> lci;   // local col_idx (GLOBAL column indices)
    std::vector<double>        lva;   // local values
    std::vector<int> rcounts;     // per-rank row counts (for Allgatherv of z)
    std::vector<int> rdispls;     // per-rank row displacements
};


// Rank 0 generates the full matrix, computes the nnz-balanced partition, and
// MPI_Scatterv distributes the CSR. Returns this rank's local slice. UNTIMED
// (matrix preparation). Reports generation/scatter wall times via the out-params.
// (Identical to the blocking baseline.)
static DistCSR setup_local_csr(std::size_t n, std::uint64_t nz, std::uint64_t seed,
                               const std::string& mode, int rank, int size,
                               double& gen_sec, double& scatter_sec) {
    gen_sec = 0.0;
    scatter_sec = 0.0;

    GeneratedMatrix G;                 // full matrix, rank 0 only
    std::vector<std::size_t> boundary; // size+1, broadcast to all
    if (rank == 0) {
        const double t0 = MPI_Wtime();
        G = generate_matrix(n, nz, seed, mode);
        gen_sec = MPI_Wtime() - t0;
        print_matrix_stats(G);
        boundary = nnz_balanced_partition(G.A.row_ptr.data(), n, size);
    } else {
        boundary.resize(size + 1);
    }

    const double ts0 = MPI_Wtime();

    {
        std::vector<long long> b(size + 1);
        if (rank == 0) for (int r = 0; r <= size; ++r) b[r] = static_cast<long long>(boundary[r]);
        check_mpi(MPI_Bcast(b.data(), size + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD), "Bcast(boundary)");
        if (rank != 0) { boundary.resize(size + 1); for (int r = 0; r <= size; ++r) boundary[r] = static_cast<std::size_t>(b[r]); }
    }

    DistCSR D;
    D.r0 = boundary[rank];
    D.local_rows = static_cast<int>(boundary[rank + 1] - boundary[rank]);

    D.rcounts.resize(size);
    D.rdispls.resize(size);
    for (int r = 0; r < size; ++r) {
        D.rcounts[r] = static_cast<int>(boundary[r + 1] - boundary[r]);
        D.rdispls[r] = static_cast<int>(boundary[r]);
    }

    std::vector<int> nnz_per_row;       // rank 0 only, length n
    if (rank == 0) {
        nnz_per_row.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            nnz_per_row[i] = static_cast<int>(G.A.row_ptr[i + 1] - G.A.row_ptr[i]);
    }
    std::vector<int> lnnz(D.local_rows);
    check_mpi(MPI_Scatterv(rank == 0 ? nnz_per_row.data() : nullptr,
                           D.rcounts.data(), D.rdispls.data(), MPI_INT,
                           buffer_ptr(lnnz), D.local_rows, MPI_INT,
                           0, MPI_COMM_WORLD), "Scatterv(nnz_per_row)");

    D.lrp.resize(D.local_rows + 1);
    D.lrp[0] = 0;
    for (int i = 0; i < D.local_rows; ++i) D.lrp[i + 1] = D.lrp[i] + static_cast<std::uint64_t>(lnnz[i]);
    const std::uint64_t local_nnz = D.lrp[D.local_rows];

    std::vector<int> ncounts, ndispls; // rank 0 only
    if (rank == 0) {
        ncounts.resize(size);
        ndispls.resize(size);
        for (int r = 0; r < size; ++r) {
            const std::uint64_t lo = G.A.row_ptr[boundary[r]];
            const std::uint64_t hi = G.A.row_ptr[boundary[r + 1]];
            if (hi - lo > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                hi > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                std::cerr << "nnz per rank or offset exceeds int range for MPI_Scatterv\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            ncounts[r] = static_cast<int>(hi - lo);
            ndispls[r] = static_cast<int>(lo);
        }
    }
    D.lva.resize(local_nnz);
    D.lci.resize(local_nnz);
    check_mpi(MPI_Scatterv(rank == 0 ? G.A.values.data() : nullptr,
                           rank == 0 ? ncounts.data() : nullptr,
                           rank == 0 ? ndispls.data() : nullptr, MPI_DOUBLE,
                           buffer_ptr(D.lva), static_cast<int>(local_nnz), MPI_DOUBLE,
                           0, MPI_COMM_WORLD), "Scatterv(values)");
    check_mpi(MPI_Scatterv(rank == 0 ? G.A.col_idx.data() : nullptr,
                           rank == 0 ? ncounts.data() : nullptr,
                           rank == 0 ? ndispls.data() : nullptr, MPI_UINT32_T,
                           buffer_ptr(D.lci), static_cast<int>(local_nnz), MPI_UINT32_T,
                           0, MPI_COMM_WORLD), "Scatterv(col_idx)");

    scatter_sec = MPI_Wtime() - ts0;
    return D;
}


// Per-chunk gather metadata, precomputed once. Every rank deterministically
// splits EVERY rank's local row block into `C` contiguous sub-blocks (same rule
// everywhere), so for each chunk c we know the per-rank counts/displacements that
// drive the c-th Iallgatherv. (The split must be a global function of the row
// counts because Iallgatherv is a collective -- all ranks issue the same C calls
// in the same order.)
struct ChunkPlan {
    int C = 1;
    std::vector<std::vector<int>> counts;   // counts[c][r] : rows of rank r in chunk c
    std::vector<std::vector<int>> displs;   // displs[c][r] : global zf offset of that block
    std::vector<int> my_lo;                 // local start (within z_local) of this rank's chunk c
    std::vector<int> my_cnt;                // this rank's row count in chunk c
};

static ChunkPlan build_chunk_plan(const DistCSR& D, int rank, int size, int C) {
    if (C < 1) C = 1;
    ChunkPlan P;
    P.C = C;
    P.counts.assign(C, std::vector<int>(size, 0));
    P.displs.assign(C, std::vector<int>(size, 0));
    P.my_lo.assign(C, 0);
    P.my_cnt.assign(C, 0);
    for (int r = 0; r < size; ++r) {
        const int rc   = D.rcounts[r];
        const int base = rc / C;
        const int rem  = rc % C;
        int off = 0;                                   // local offset within rank r's block
        for (int c = 0; c < C; ++c) {
            const int sz = base + (c < rem ? 1 : 0);   // front-loaded remainder
            P.counts[c][r] = sz;
            P.displs[c][r] = D.rdispls[r] + off;       // global offset into zf
            off += sz;
        }
    }
    for (int c = 0; c < C; ++c) {
        P.my_cnt[c] = P.counts[c][rank];
        P.my_lo[c]  = P.displs[c][rank] - static_cast<int>(D.r0);  // == local offset
    }
    return P;
}


int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::cerr << "MPI_THREAD_FUNNELED not provided\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::uint64_t n64 = 0, nz = 0, seed = 111, threads = 0, chunks = 4;
    std::string mode, dump_vector_path;
    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode)) {
        if (rank == 0) {
            usage(argv[0]);
            std::cerr << "  -t   Optional OpenMP threads per rank (default: omp_get_max_threads)\n";
            std::cerr << "  -c   Optional #pipeline chunks per gather (default: 4; 1 = no overlap)\n";
        }
        MPI_Finalize();
        return 1;
    }
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);
    if (!read_arg_u64(argc, argv, "-t", threads)) (void)read_arg_u64(argc, argv, "--threads", threads);
    (void)read_arg_u64(argc, argv, "-c", chunks);

    int nthreads = static_cast<int>(threads);
    if (nthreads <= 0) nthreads = omp_get_max_threads();
    if (nthreads <= 0) nthreads = 1;

    int C = static_cast<int>(chunks);
    if (C < 1) C = 1;

    const std::size_t n = static_cast<std::size_t>(n64);
    if (rank == 0) std::cout << "SPARSE_ITERATION_MPI_OMP_OVERLAP\n";

    try {
        // --- Setup (UNTIMED): generate on rank 0 + Scatterv the CSR ---
        double gen_sec = 0.0, scatter_sec = 0.0;
        const DistCSR D = setup_local_csr(n, nz, seed, mode, rank, size, gen_sec, scatter_sec);
        const ChunkPlan plan = build_chunk_plan(D, rank, size, C);

        if (rank == 0) {
            std::cout << "ranks=" << size << "\n";
            std::cout << "threads_per_rank=" << nthreads << "\n";
            std::cout << "overlap_chunks=" << C << "\n";
            std::cout << "generation_time_sec=" << gen_sec << "\n";
            std::cout << "scatter_time_sec=" << scatter_sec << "\n\n";
        }

        const std::size_t shift_rows = compute_shift_rows(n);

        std::vector<double> x(n);
        std::vector<double> zf(n);

        SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
        for (double& v : x) v = rng.next_unit();
        normalize(x);

        const std::uint64_t* lrp = buffer_ptr(D.lrp);
        const std::uint32_t* lci = buffer_ptr(D.lci);
        const double*        lva = buffer_ptr(D.lva);
        std::vector<double> z_local(D.local_rows);

        // Pipelined SpMV + gather: for each chunk, compute its z_local rows then
        // launch a non-blocking Iallgatherv; the next chunk's SpMV overlaps the
        // in-flight gather. MPI_Testall between chunks nudges progress (FUNNELED:
        // main thread only). On return, zf holds the full natural-order vector.
        std::vector<MPI_Request> reqs(C, MPI_REQUEST_NULL);
        auto gather_pipelined = [&]() {
            for (int c = 0; c < C; ++c) {
                const int lo  = plan.my_lo[c];
                const int cnt = plan.my_cnt[c];

                // Compute this rank's chunk-c rows of z_local (OpenMP).
                #pragma omp parallel for schedule(static) num_threads(nthreads)
                for (int i = lo; i < lo + cnt; ++i) {
                    double sum = 0.0;
                    for (std::uint64_t p = lrp[i]; p < lrp[i + 1]; ++p)
                        sum += lva[p] * x[lci[p]];
                    z_local[i] = sum;
                }

                // Non-blocking gather of chunk c (every rank contributes its
                // chunk-c block; count may be 0 but the collective is still issued).
                const double* sbuf = (cnt > 0) ? (z_local.data() + lo) : nullptr;
                check_mpi(MPI_Iallgatherv(sbuf, cnt, MPI_DOUBLE,
                                          buffer_ptr(zf),
                                          plan.counts[c].data(), plan.displs[c].data(),
                                          MPI_DOUBLE, MPI_COMM_WORLD, &reqs[c]),
                          "Iallgatherv(chunk)");

                // Poll: advance the earlier in-flight gathers (notes p.139).
                int flag = 0;
                MPI_Testall(c + 1, reqs.data(), &flag, MPI_STATUSES_IGNORE);
            }
            MPI_Waitall(C, reqs.data(), MPI_STATUSES_IGNORE);
        };

        // --- Timed iterative loop (the only thing measured) ---
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        std::size_t row_shift = 0;
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            if (iter > 0 && (iter % EPOCH_LEN) == 0)
                row_shift = (row_shift + shift_rows) % n;

            gather_pipelined();    // SpMV + overlapped Allgatherv -> zf (and z_local)

            // Global L2 norm as a DISTRIBUTED two-level reduction (per-rank partial
            // over z_local + MPI_Allreduce), exactly as in the blocking baseline.
            // The shift is a permutation, so ||y|| == ||z||; each rank reduces only
            // its n/size local rows and no rank repeats a full-vector serial pass.
            // MPI_Allreduce gives every rank the same ss, keeping the replicated x
            // byte-identical across ranks. Result now matches the sequential within
            // tolerance (checksum DIFFERS, like the threads/omp versions).
            double partial_ss = 0.0;
            #pragma omp parallel for reduction(+: partial_ss) \
                schedule(static) num_threads(nthreads)
            for (int i = 0; i < D.local_rows; ++i) partial_ss += z_local[i] * z_local[i];
            double ss = 0.0;
            check_mpi(MPI_Allreduce(&partial_ss, &ss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                      "Allreduce(ss)");
            const double inv = 1.0 / std::sqrt(ss);

            // Shift (logical, local rotation) + scale into x in one pass:
            // x[i] = z[(i - shift) mod n] / ||y||.
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t src = i + n - row_shift;
                if (src >= n) src -= n;          // (i - shift) mod n, shift in [0,n)
                x[i] = inv * zf[src];
            }
        }

        // Final diagnostics (inside the timed region, as in the sequential code):
        // one extra shifted SpMV, rayleigh = dot(x, y) with y the shifted result.
        gather_pipelined();
        double rayleigh = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t src = i + n - row_shift;
            if (src >= n) src -= n;
            rayleigh += x[i] * zf[src];   // x[i] * y[i], y[i] = z[(i-shift) mod n]
        }

        const double local_elapsed = MPI_Wtime() - t0;
        double elapsed = 0.0;
        check_mpi(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
                  "Reduce(time)");

        const std::uint64_t checksum = checksum_vector(x);   // identical on all ranks

        if (rank == 0) {
            std::cout << std::setprecision(15);
            std::cout << "rayleigh=" << rayleigh << "\n";
            std::cout << "checksum=0x" << std::hex << checksum << std::dec << "\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Time (sec) = " << elapsed << "\n";

            if (!dump_vector_path.empty()) {
                dump_vector(dump_vector_path, x);
                std::cout << "vector_dump=" << dump_vector_path << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error (rank " << rank << "): " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
