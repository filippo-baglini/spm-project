# Hybrid MPI + OpenMP — Deep Dive (Deliverable #3)

**Project:** Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
**Files explained here, line by line:**
`src/mpi/iterative_SpMV_mpi.cpp` (blocking `Allgatherv` baseline) and
`src/mpi/iterative_SpMV_mpi_overlap.cpp` (non-blocking `Iallgatherv`, communication/computation overlap).

This is the distributed-memory companion to `PARALLEL_IMPLEMENTATIONS.md` (which covers the
shared-memory threads / OpenMP / pool versions) and `DOCUMENTATION.md` (the project-wide survey).
The two files documented here are the *third deliverable*: the same power-iteration SpMV, now spread
across **processes that may live on different nodes**, with OpenMP parallelizing each process's share.

If you have not read it yet, skim `PARALLEL_IMPLEMENTATIONS.md §0` first — the **"split source rows,
not output rows"** idea and the **shift-as-vector-rotation** trick carry over verbatim, and everything
below assumes them.

---

## 0. The two levels of parallelism (and why "threads per rank" is a thing)

The single most important mental model for this deliverable:

| | **MPI rank** | **OpenMP thread** |
|---|---|---|
| What it is | a whole **OS process** | a thread spawned **inside** one rank |
| Memory | **private** (own matrix slice, own `x`) | **shared** with its rank's other threads |
| Identity | an integer **rank**, fixed at `MPI_Init` for the whole run | **no rank** — MPI doesn't know it exists |
| Lifetime | the entire program | born at `#pragma omp parallel`, dies at its end (×100s) |
| Communication | only by **messages** (`Allgatherv`, …) | by **shared memory** (no messages) |

So a rank with `threads_per_rank = 4` is **one** process that, whenever it does its local SpMV, fans
that loop across 4 cores, then collapses back to a single thread to make the one MPI call. The geometry
knobs and the hard per-node rule:

```
ranks_per_node (RPN) × threads_per_rank (TPR)  ≤  16 physical cores / node
total_ranks   =  nodes × RPN
total_cores   =  total_ranks × TPR            ( = nodes × 16 when a node is full )
```

The same 16 cores of a node can be filled as `16×1` (flat MPI — 16 message-passing processes),
`4×4` (hybrid — 4 processes, 4 shared-memory threads each), or `1×16` (all OpenMP, no on-node MPI).
They differ only in **how many of the cores are separate message-passing processes vs. shared-memory
threads inside a process** — which is exactly what changes how many participants the per-iteration
collective has to coordinate. (See `compare_mpi.sh` output and `ROOFLINE.md` for the measured effect:
hybrid `4×4` beats flat `16×1` because fewer ranks → a cheaper `Allgatherv`.)

`MPI_Init_thread(MPI_THREAD_FUNNELED)` declares the contract that makes this safe: **only the main
thread of each rank ever calls MPI**; the OpenMP regions purely compute. We verify
`provided >= MPI_THREAD_FUNNELED` at startup and abort otherwise
(`iterative_SpMV_mpi.cpp:268`, course idiom from `Code/spmcode12/mpi_trapezoid+omp.cpp`).

---

## 1. What is distributed, what is replicated

This is the design decision the whole implementation rests on.

- **The matrix (O(nnz)) is distributed by *source* rows.** Each rank owns a contiguous, nnz-balanced
  block of rows and *only* that block's `values`/`col_idx`/`row_ptr`. After rank 0 hands the blocks
  out, **no rank holds the full matrix** — that is the real memory scaling.
- **The dense vector `x` (O(n)) is replicated** — a full copy on every rank. `x` is at most a few MB,
  and the **irregular column gather** `x[col_idx[p]]` can touch *any* entry, so each rank needs *all*
  of `x` to do its local SpMV. Distributing `x` would just force a gather of (almost) all of it back
  anyway (the columns are random — no locality to exploit).

**Why the matrix partition never moves.** The "evolving matrix" is a circular **row shift** every
`EPOCH_LEN = 25` iterations. We apply that shift as a **local rotation of the result vector**, not by
moving matrix rows between ranks. So the nnz-balanced block a rank owns is **fixed for all 500
iterations** — computed once, balanced forever (same insight as the threads version's epoch-invariant
static partition). This is the concrete answer to the sequential header's note that the evolution
"must be reflected consistently in the distributed data layout": it is reflected as a vector rotation,
so the *layout* of the matrix is invariant.

---

## 2. Setup — rank 0 generates, `MPI_Scatterv` distributes (UNTIMED)

`setup_local_csr` (`iterative_SpMV_mpi.cpp:161`) builds each rank's slice. It is **not timed** — matrix
preparation is setup, consistent with the project rule that generation is untimed and with the
sequential reference, whose timed region is the loop only. Generation and scatter wall-times are
reported separately (`generation_time_sec=`, `scatter_time_sec=`).

Steps:

1. **Generate (rank 0 only).** Rank 0 calls the provided `generate_matrix` (deterministic, byte-for-byte
   the immutable helper) and computes the **nnz-balanced partition** `boundary[0..size]` with
   `nnz_balanced_partition` (`:123`) — the very routine the threads/NUMA versions use: binary-search the
   CSR `row_ptr` prefix-sum for the row where cumulative nnz crosses `total_nnz · r / size`, then clamp
   monotonic. `__uint128_t` guards the `total_nnz · r` product from 64-bit overflow.
2. **Broadcast the partition** (`:183`). `boundary[]` goes to every rank via `MPI_Bcast`, so each rank
   knows *all* row blocks — needed both to receive its own slice and to drive the later `Allgatherv`
   counts/displacements.
3. **Scatter the per-row nnz counts** (`MPI_Scatterv`, `:212`). Disjoint by row blocks; each rank
   prefix-sums its received counts into a **rebased local `row_ptr`** (`lrp`, starting at 0).
4. **Scatter `values` and `col_idx`** by **nnz blocks** (`:242`, `:247`). The per-rank nnz count is
   `row_ptr[boundary[r+1]] − row_ptr[boundary[r]]`. **Column indices stay global** — the gather is
   against the full replicated `x`, so no remapping is needed. (An overflow guard aborts if a rank's
   nnz or offset exceeds `INT_MAX`, since `Scatterv` counts are `int`; the project's ≤2·10⁷ nnz fits.)
5. **Rank 0 frees the full matrix** as `setup_local_csr` returns — from here, memory is truly distributed.

The reusable helpers `check_mpi` (`:86`, abort-with-message on any non-`MPI_SUCCESS`) and the
`buffer_ptr` overloads (`:95`, return `nullptr` for empty vectors so zero-count MPI calls are valid)
come straight from the `Code/spmcode11` templates.

`DistCSR` (`:147`) is the per-rank result: `r0` (first global row), `local_rows`, the local CSR arrays,
and `rcounts`/`rdispls` (per-rank **row** counts/displacements) reused every iteration as the
`Allgatherv` recipe.

---

## 3. The per-iteration dataflow

Both files run the identical algorithm; only step 2 (the gather) differs. The loop
(`iterative_SpMV_mpi.cpp:343`) is:

```
every EPOCH_LEN iters:  row_shift = (row_shift + shift_rows) mod n        // matrix "evolves"

1. local fused SpMV (OpenMP):  z_local[i] = Σ_p  lva[p] · x[lci[p]]       // this rank's rows, natural order
2. distributed norm:  partial = Σ_local z_local[i]² (OpenMP) ;  MPI_Allreduce(SUM) → ss
                       inv = 1/√ss            // ‖y‖ = ‖z‖ (shift is a permutation), so reduce z_local
3. MPI_Allgatherv(z_local)  →  full z (natural source order) on every rank
4. shift (LOCAL rotation) + scale in one pass:  x[i] = z[(i − row_shift) mod n] · inv
```

Then **diagnostics inside the timed region** (`:372`): one extra SpMV + gather, and
`rayleigh = Σ x[i]·y[i]` — matching the sequential reference within tolerance.

### 3.1 Why the `Allgatherv` is unavoidable *every* iteration

The iteration is a strict dependency chain:

```
SpMV(x_k) ─► Allgatherv ─► rotate ─► normalize ─► SpMV(x_{k+1}) ─► …
 (big compute)  (big comm)  (cheap)   (cheap)
```

Each rank computes only **its slice** of the new vector, but the **next** SpMV reads the **whole**
vector (random columns reach everywhere). So every iteration must rebuild the full vector on every
rank — that is the `Allgatherv`. You **cannot** defer it to the epoch boundary: there is a fresh SpMV
between every pair of normalizations, and that SpMV needs the full vector. (Deferring would only be
valid for a block-diagonal matrix.) The normalization *is* global but only needs a **scalar** reduction
— it is **not** what forces the O(n) gather; the SpMV is.

### 3.2 The norm: a distributed two-level reduction

The norm is a genuine distributed reduction — the MPI analogue of the threads version's *per-worker
partial + barrier combine* (`PARALLEL_IMPLEMENTATIONS.md §1`): each rank sums the squares of its **own
`z_local` slice** with an OpenMP `reduction(+:ss)`, then a single **`MPI_Allreduce(SUM)`** combines the
per-rank partials into a global `ss` that is **identical on every rank**. Two facts make this clean:

- **The shift is a permutation, so ‖y‖ = ‖z‖.** We can reduce `z_local` *directly* — each rank touches
  only its `n/size` entries — instead of reducing the full rotated vector. No rank repeats a serial
  O(n) pass; the reduction work is actually distributed.
- **`MPI_Allreduce` guarantees the identical result on all ranks**, which is *required*: the replicated
  `x` is the input to the next SpMV's `x[global_column]` gather, so every rank must scale by the same
  `inv` or the replicated vector would diverge. (This is the role the old canonical-order serial sum
  played; `Allreduce` does it without the serial bottleneck.)

**Correctness ⇒ tolerance, not bit-for-bit.** Because the reduction order now differs from the
sequential's, the result matches the sequential **within tolerance** (~1e-13), and the `checksum`
*differs* — exactly like the threads/OpenMP versions, and acceptable under the project's tolerance
rule. The cost is **one extra (scalar) collective per iteration** on top of the `Allgatherv`; the
payoff is removing the per-rank serial O(n) norm, which most helps the high-thread-per-rank hybrid
geometries (2:8, 4:8) where that serial loop was competing with 8 threads.

> Earlier design (kept in git history): reduce `Σ x[i]²` *locally in canonical index order* on the
> already-gathered full vector. That gave a **bit-for-bit** match with the sequential and one fewer
> collective, but at the price of a **serial O(n)** reduction repeated on every rank every iteration.
> Since the project only requires tolerance-level correctness, we traded the bitwise guarantee for the
> distributed (parallel) reduction. To restore the bitwise match, swap the `Allreduce` back for the
> canonical-order local sum.

### 3.3 What is timed

`MPI_Wtime` brackets the **iterative loop only**, after an `MPI_Barrier` to align all ranks (`:339`).
The reported time is the **max over ranks** (`MPI_Reduce(MPI_MAX)`, `:385`) — the honest wall time of
the slowest rank. Generation and `Scatterv` are setup and excluded; `--dump-vector` runs after the
timer. This mirrors exactly what the sequential reference times (loop + diagnostics + checksum).

---

## 4. The overlap variant — `iterative_SpMV_mpi_overlap.cpp`

Everything in §1–§3 is unchanged. The **only** difference is *how* step 2 gathers the result vector.

### 4.1 The idea: pipeline the gather against the SpMV

The blocking baseline computes **all** local rows, **then** does one `Allgatherv` — the two largest
costs run back to back. But the SpMV produces `z_local` row by row, and the gather just moves those
bytes, so we can **software-pipeline** them:

```
[ compute chunk 0 ]
[ compute chunk 1 ][ Iallgatherv chunk 0 ]
[ compute chunk 2 ][ Iallgatherv chunk 1 ]
        …
                   [ Iallgatherv chunk C-1 ]
[ MPI_Waitall ]
```

Split each rank's local row block into `C` contiguous **chunks**. As soon as chunk `c`'s `z_local`
rows are computed, fire a **non-blocking `MPI_Iallgatherv`** for that chunk and keep the OpenMP threads
computing chunk `c+1` while the gather is in flight. A final `MPI_Waitall` drains the pipeline. The
per-iteration cost trends from `T_compute + T_comm` toward `max(T_compute, T_comm) + one_chunk`.

This is the **only** overlap available here: the inter-iteration dependency chain (§3.1) forbids
overlapping iteration `k`'s gather with iteration `k+1`'s SpMV, so all overlap must be **intra-iteration**,
between this iteration's SpMV and its own gather. (The textbook *local-vs-remote-column* SpMV overlap
does **not** help: random columns mean almost every local row needs remote `x` entries — there is no
"local-only" work to hide the comm behind. Same reason CA-Krylov / matrix-powers can't apply.)

### 4.2 `ChunkPlan` — making a collective out of per-rank chunks (`:line build_chunk_plan`)

`MPI_Iallgatherv` is a **collective**: every rank must issue the **same `C` calls in the same order**,
and for call `c` it must supply the per-rank counts/displacements of *everyone's* chunk `c`. Ranks have
**different** local row counts (nnz-balanced), so "chunk c" must be a **global, deterministic function**
of the row counts, computable on every rank.

`build_chunk_plan` does exactly that: every rank splits **every** rank's block into `C` contiguous
sub-blocks with the same front-loaded-remainder rule (`base = rc/C`, first `rc%C` chunks get one
extra). For each chunk `c` it precomputes:

- `counts[c][r]` — rows of rank `r` in chunk `c`,
- `displs[c][r]` — that block's **global** offset into the full vector `zf`,
- `my_lo[c]`, `my_cnt[c]` — this rank's local start (within `z_local`) and count for chunk `c`.

Because the split is the same function everywhere, the `C` collectives are consistent across ranks even
though each rank only physically computes/sends its own sub-blocks. (`C` is global, taken from `-c`; it
is **not** clamped to a per-rank value — that would break collective agreement.)

### 4.3 The pipelined gather (`gather_pipelined`)

```cpp
for (int c = 0; c < C; ++c) {
    // compute this rank's chunk-c rows of z_local (OpenMP, FUNNELED-safe: no MPI inside)
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = lo; i < lo + cnt; ++i) { … z_local[i] = Σ lva[p]·x[lci[p]]; }

    // non-blocking gather of chunk c (every rank issues it, even if cnt == 0)
    MPI_Iallgatherv(sbuf, cnt, MPI_DOUBLE, zf.data(),
                    plan.counts[c].data(), plan.displs[c].data(),
                    MPI_DOUBLE, MPI_COMM_WORLD, &reqs[c]);

    // poll: advance the earlier in-flight gathers (notes p.139)
    int flag = 0; MPI_Testall(c + 1, reqs.data(), &flag, MPI_STATUSES_IGNORE);
}
MPI_Waitall(C, reqs.data(), MPI_STATUSES_IGNORE);
```

Notes on the mechanics:

- **The MPI calls are outside the OpenMP region.** The `#pragma omp parallel for` completes (all
  threads join) before the main thread issues `Iallgatherv` — so only the main thread ever calls MPI,
  honoring `MPI_THREAD_FUNNELED`.
- **A zero-count chunk still issues the collective.** If `C` exceeds some rank's row count, that rank's
  `cnt` for the trailing chunks is 0, but it must *still* call `Iallgatherv` (count 0, `sbuf = nullptr`)
  so the collective stays matched across ranks.
- **`reqs` initialized to `MPI_REQUEST_NULL`** so `MPI_Testall(c+1, …)` safely tests the
  not-yet-issued slots (they count as complete).

### 4.4 MPI progress — the honest caveat (and why no progress thread)

Under `MPI_THREAD_FUNNELED`, the main thread is the *only* one allowed to call MPI, and it spends a
chunk's compute time inside the OpenMP region making **no** MPI calls. Many MPI implementations only
advance a non-blocking collective when the application calls MPI, so a background `Iallgatherv` may not
progress *during* the compute — the realistic overlap is the **eager / early portion** of each transfer
proceeding, plus whatever the `MPI_Testall` poll at each chunk boundary nudges forward.

We deliberately do **not** spin up a dedicated progress/communication thread: that needs
`MPI_THREAD_MULTIPLE`, which the **SPM notes do not cover**, so it would violate the project's
"everything from the notes" constraint. The `MPI_Test`/`MPI_Waitall` polling approach we use *is* in the
notes (p.139). The takeaway for the report: overlap delivers a **constant-factor** win in the regime
where `T_compute ≳ T_comm` (moderate rank counts — 1–4 nodes here), and **tapers off** at high node
counts where the O(n) `Allgatherv` already dominates and there is no compute left to hide it behind —
i.e. it softens, but does not move, the `Allgatherv` strong-scaling floor from `ROOFLINE.md`.

### 4.5 Correctness is untouched

The chunked `Iallgatherv` reassembles the **same** natural-order vector into the **same** positions of
`zf` — only the *order in which bytes move* changes, never the arithmetic. The per-row dot-product order
and the distributed-`Allreduce` norm are bitwise identical to the blocking baseline, so the result is
**invariant to rank × thread × node × chunk count** (it matches the sequential within tolerance, like
the baseline — see §3.2). `C` (the `-c` flag) is a pure tuning knob:
`-c 1` collapses to a single `Iallgatherv` (a no-overlap control ≈ the blocking baseline); larger `C`
trades more collective-launch overhead for more overlap. A handful (4–8) is the usual sweet spot.

### 4.6 Grounding in the SPM notes

| Technique | Notes / Code origin |
|---|---|
| "All collectives have a non-blocking version" (so `MPI_Iallgatherv` exists) | SPM NOTES p.141 (`MPI_Ibarrier` shown) |
| Non-blocking calls "let you overlap communication and computation"; `MPI_Wait`/`MPI_Test` + `*all` variants | SPM NOTES p.138–139 |
| Double buffering / async to "maximize the overlap of computation and communication" | SPM NOTES p.140 (farm skeleton) |
| `MPI_Init_thread(FUNNELED)` + OpenMP compute | SPM NOTES p.140; `Code/spmcode12/mpi_trapezoid+omp.cpp` |
| Partitioned matrix + `Allgatherv` power iteration | `Code/spmcode11/mpi_power_iteration_partitioned.cpp` |
| `MPI_Scatterv` block distribution | `Code/spmcode11/mpi_vectorsumv.cpp`, `Code/spmcode12/mpi_jacobi_1d_blk.cpp` |

---

## 5. Build, run, and compare

Both binaries are built with `mpicxx … -fopenmp` and are **kept out of `make all`** (a plain `make`
must work on machines without an MPI toolchain):

```bash
make mpi            # -> bin/mpi          (blocking Allgatherv baseline)
make mpi_overlap    # -> bin/mpi_overlap  (non-blocking Iallgatherv overlap)
```

Same CLI for both, plus `-c` on the overlap build:

```
-n N  -nz K  -m regular|irregular  [-s seed]  [-t threads_per_rank]
[--dump-vector FILE]                 # both
[-c chunks]                          # overlap only: #pipeline chunks/gather (default 4; 1 = no overlap)
```

Run (one node, hybrid 4×4; or 8 nodes, 32×4):

```bash
mpirun -np 4  --map-by ppr:4:node:pe=4   --bind-to core ./bin/mpi          -n 500000 -nz 20000000 -m irregular -t 4
mpirun -np 4  --map-by ppr:4:node:pe=4   --bind-to core ./bin/mpi_overlap  -n 500000 -nz 20000000 -m irregular -t 4 -c 4
```

**Cluster sweep / comparison** (no sequential rerun — its time is pooled from `results_numa`). All
sweep jobs build and time **both** variants (`VARIANTS="block overlap"`, `CHUNKS` = `-c` list):

- `scripts/run_mpi.sbatch` — one geometry per job (reads the SLURM allocation); builds both binaries + `bin/seq`.
- `scripts/run_mpi_quick.sbatch` — slim, no-seq; `CONFIGS="ranks:rpn:tpr …"`; **resumable** (skips
  complete cells; stops cleanly at `TIME_BUDGET_SEC`).
- `scripts/submit_mpi_sweep.sh` — one slim job per node count (1/2/4/8) × a couple of splits (quick look).
- `scripts/submit_mpi_full.sh` — the **comprehensive** sweep: every meaningful geometry of both variants.
  Axes: nodes {1,2,4,8} × the full per-node rank/thread spectrum `1:16 2:8 4:4 8:2 16:1` (rpn:tpr,
  product = 16 phys cores; `2:8` = one rank per **socket**, the NUMA-optimal point — see
  `docs/`/topology) × {block, overlap} × chunks {1,2,4} × all four project matrix sizes (the two
  40-nnz/row canonical sizes plus the 200- and 8-nnz/row density extremes). One
  resumable job per (node-count, matrix-pair) = 16 jobs; the driver pre-seeds an authoritative
  `config.env` so concurrent jobs don't race on it.
- `scripts/compare_mpi.sh` — per-(n,nz) table over **both variants** (columns `variant`, `chk`, ranks,
  thr/rk, cores): median loop time, speedup vs the pooled sequential baseline, efficiency vs total
  cores, approx aggregate matrix-stream `GB/s`, and the `checksum vs seq` cross-check. With the
  distributed-`Allreduce` norm (§3.2) the checksum now *differs* from the sequential within tolerance
  (like threads/omp); the meaningful invariant is that it is **identical across every MPI geometry**
  (rank × thread × node × chunk), which the table makes easy to eyeball.

The overlap win is largest at 1–2 nodes with a hybrid split and shrinks (then reverses) at higher node
counts, where splitting one `Allgatherv` into `C` collectives multiplies the rank-scaling collective
latency (§4.4); `-c 1` is the built-in no-overlap control.

---

## 6. Correctness checklist

1. **Build:** `make mpi` and `make mpi_overlap` compile clean with `mpicxx -fopenmp` (cluster). Locally
   they need no MPI toolchain; a `-fsyntax-only` check against a stub `mpi.h` is the local smoke test.
2. **Correctness (tolerance):** for both binaries, small case (`-n 5000 -nz 20000 -m irregular`) under
   `mpirun -np {1,2,4}` × a few `-t` (and `-c` for overlap), `--dump-vector` is within ~1e-12 of the
   sequential dump. The `checksum` is **identical across all geometries** (ranks/threads/nodes/chunks)
   — proving the distributed reduction is deterministic — but **differs from the sequential** in the
   low bits (different reduction order, §3.2), exactly like the threads/omp versions.
3. **Scaling:** the sweep (above) shows speedup/efficiency and the `Allgatherv`-bound trend across
   1→8 nodes (ties back to the memory-bandwidth roofline in `ROOFLINE.md`).
4. **No regression:** the non-MPI `make` targets (seq/threads/omp/pool/numa) are untouched — `mpi` and
   `mpi_overlap` are separate, non-`all` targets.
