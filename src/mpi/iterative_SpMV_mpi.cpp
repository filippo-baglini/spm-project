//
// Hybrid MPI + OpenMP implementation for the One-Shot project (deliverable #3):
//
//   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
// Parallelization strategy
// ------------------------
//   * MPI distributes the matrix by SOURCE ROWS: rank 0 generates the full CSR
//     matrix (untimed, as in every version) and MPI_Scatterv distributes a
//     contiguous, nnz-balanced block of rows to each rank. The partition is
//     FIXED for the whole run: the matrix evolution is a circular row shift,
//     which we apply as a cheap LOCAL VECTOR ROTATION, so no matrix data ever
//     moves between ranks (the distributed analogue of the threads version's
//     epoch-invariant static partition).
//   * OpenMP parallelizes each rank's local SpMV across its cores.
//   * The dense vector x is REPLICATED on every rank (it is only O(n) and the
//     irregular column gather needs every entry of x). The big O(nnz) data --
//     the matrix -- is what is distributed.
//
// Per iteration (the only collective is one MPI_Allgatherv):
//   1. local fused SpMV in natural source order: z_local[i] = A_row(r0+i) . x
//      (OpenMP parallel-for; one row per thread, so the per-row accumulation
//       order is preserved, exactly as in the sequential reference).
//   2. MPI_Allgatherv -> the full natural-order vector z on every rank.
//   3. apply the shift as a local rotation: y[i] = z[(i - shift) mod n], which
//      is exactly the sequential's y[i] = A_row((i-shift) mod n) . x.
//   4. normalize y (global L2): the sum-of-squares is done in canonical index
//      order, so the result is bit-for-bit identical to the sequential for ANY
//      rank x thread x node count (checksum MATCHES the sequential, unlike the
//      threads/omp versions). x_{k+1} = y / ||y||.
//
// (An MPI_Allreduce of the local sum-of-squares would be the more "collective"
//  way to do the norm, but it combines partials in a nondeterministic order and
//  so matches only within tolerance. We already own the full vector after the
//  Allgatherv, so we reduce locally in canonical order for the bitwise match and
//  a single collective per iteration.)
//
// Hybrid init: MPI_Init_thread(MPI_THREAD_FUNNELED) -- only the main thread calls
// MPI; OpenMP regions only compute (course example Code/spmcode12/mpi_trapezoid+omp.cpp).
//
// Timing: matrix generation AND the Scatterv distribution are SETUP and are NOT
// timed (consistent with the project rule that generation is untimed and with
// the sequential reference, whose timed region is the iterative loop only). They
// are reported separately. The timed region is the iterative loop, measured with
// MPI_Wtime around an MPI_Barrier; the reported time is the max over ranks.
//
// Command line (same as the other versions):
//   -n N  -nz K  -m regular|irregular  [-s seed]  [-t threads_per_rank]
//   [--dump-vector FILE]   (rank 0 dumps the final full vector)
//
// Build (cluster):
//   mpicxx -O3 -std=c++20 -fopenmp -I include
//          src/mpi/iterative_SpMV_mpi.cpp -o bin/mpi
//   (or: make mpi)
//
// Run:
//   mpirun -np 4 ./bin/mpi -n 500000 -nz 20000000 -m irregular -t 8
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
// row block. Same logic as the threads/NUMA versions.
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
// to reassemble the full result vector with MPI_Allgatherv.
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
static DistCSR setup_local_csr(std::size_t n, std::uint64_t nz, std::uint64_t seed,
                               const std::string& mode, int rank, int size,
                               double& gen_sec, double& scatter_sec) {
    gen_sec = 0.0;
    scatter_sec = 0.0;

    // --- rank 0: generate + partition ---
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

    // Broadcast the partition so every rank knows all row blocks.
    {
        std::vector<long long> b(size + 1);
        if (rank == 0) for (int r = 0; r <= size; ++r) b[r] = static_cast<long long>(boundary[r]);
        check_mpi(MPI_Bcast(b.data(), size + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD), "Bcast(boundary)");
        if (rank != 0) { boundary.resize(size + 1); for (int r = 0; r <= size; ++r) boundary[r] = static_cast<std::size_t>(b[r]); }
    }

    DistCSR D;
    D.r0 = boundary[rank];
    D.local_rows = static_cast<int>(boundary[rank + 1] - boundary[rank]);

    // Row counts/displs (used both for the per-row-nnz scatter and, later, the
    // Allgatherv of the result vector z).
    D.rcounts.resize(size);
    D.rdispls.resize(size);
    for (int r = 0; r < size; ++r) {
        D.rcounts[r] = static_cast<int>(boundary[r + 1] - boundary[r]);
        D.rdispls[r] = static_cast<int>(boundary[r]);
    }

    // 1) Scatter the per-row nnz counts (disjoint, by row blocks), then each rank
    //    prefix-sums them into its rebased local row_ptr.
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

    // 2) Scatter values and col_idx by nnz blocks. For the project sizes the
    //    global nnz (<= 2e7) fits in int; guard otherwise.
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
    // rank 0's full matrix (G) is freed here as the function returns.
    return D;
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

    // Parse args on every rank (argv is replicated by the launcher; deterministic
    // so no broadcast of parameters is needed).
    std::uint64_t n64 = 0, nz = 0, seed = 111, threads = 0;
    std::string mode, dump_vector_path;
    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode)) {
        if (rank == 0) {
            usage(argv[0]);
            std::cerr << "  -t   Optional OpenMP threads per rank (default: omp_get_max_threads)\n";
        }
        MPI_Finalize();
        return 1;
    }
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);
    if (!read_arg_u64(argc, argv, "-t", threads)) (void)read_arg_u64(argc, argv, "--threads", threads);

    int nthreads = static_cast<int>(threads);
    if (nthreads <= 0) nthreads = omp_get_max_threads();
    if (nthreads <= 0) nthreads = 1;

    const std::size_t n = static_cast<std::size_t>(n64);
    if (rank == 0) std::cout << "SPARSE_ITERATION_MPI_OMP\n";

    try {
        // --- Setup (UNTIMED): generate on rank 0 + Scatterv the CSR ---
        double gen_sec = 0.0, scatter_sec = 0.0;
        const DistCSR D = setup_local_csr(n, nz, seed, mode, rank, size, gen_sec, scatter_sec);

        if (rank == 0) {
            std::cout << "ranks=" << size << "\n";
            std::cout << "threads_per_rank=" << nthreads << "\n";
            std::cout << "generation_time_sec=" << gen_sec << "\n";
            std::cout << "scatter_time_sec=" << scatter_sec << "\n\n";
        }

        const std::size_t shift_rows = compute_shift_rows(n);

        // Replicated full vectors. x: current iterate; zf: gathered SpMV result.
        std::vector<double> x(n);
        std::vector<double> zf(n);

        // Initial vector, identical to the sequential reference, on every rank.
        SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
        for (double& v : x) v = rng.next_unit();
        normalize(x);

        // Local CSR (raw aliases).
        const std::uint64_t* lrp = buffer_ptr(D.lrp);
        const std::uint32_t* lci = buffer_ptr(D.lci);
        const double*        lva = buffer_ptr(D.lva);
        std::vector<double> z_local(D.local_rows);

        // One local SpMV phase (fused: writes z_local). OpenMP over local rows.
        auto local_spmv = [&]() {
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (int i = 0; i < D.local_rows; ++i) {
                double sum = 0.0;
                for (std::uint64_t p = lrp[i]; p < lrp[i + 1]; ++p)
                    sum += lva[p] * x[lci[p]];
                z_local[i] = sum;
            }
        };

        // --- Timed iterative loop (the only thing measured) ---
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        std::size_t row_shift = 0;
        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            if (iter > 0 && (iter % EPOCH_LEN) == 0)
                row_shift = (row_shift + shift_rows) % n;

            local_spmv();

            check_mpi(MPI_Allgatherv(buffer_ptr(z_local), D.local_rows, MPI_DOUBLE,
                                     buffer_ptr(zf), D.rcounts.data(), D.rdispls.data(),
                                     MPI_DOUBLE, MPI_COMM_WORLD), "Allgatherv(z)");

            // y[i] = z[(i - shift) mod n], written back into x (free after SpMV).
            // Per-element and independent, so OpenMP-safe (values bit-identical).
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t src = i + n - row_shift;
                if (src >= n) src -= n;          // (i - shift) mod n, shift in [0,n)
                x[i] = zf[src];
            }

            // Normalize in canonical index order (matches the sequential exactly).
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) ss += x[i] * x[i];
            const double inv = 1.0 / std::sqrt(ss);
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
        }

        // Final diagnostics (inside the timed region, as in the sequential code):
        // one extra shifted SpMV, rayleigh = dot(x, y) with y the shifted result.
        local_spmv();
        check_mpi(MPI_Allgatherv(buffer_ptr(z_local), D.local_rows, MPI_DOUBLE,
                                 buffer_ptr(zf), D.rcounts.data(), D.rdispls.data(),
                                 MPI_DOUBLE, MPI_COMM_WORLD), "Allgatherv(z_final)");
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
