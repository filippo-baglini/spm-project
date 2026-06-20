# Code Documentation

**Project:** Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
**Files documented:** `utils.hpp`, `matrix_generation.hpp`, `iterative_SpMV.cpp` (sequential reference), `iterative_SpMV_threads.cpp` (C++ threads), `Makefile`.

---

## 1. The problem at a glance

Given a fixed sparse matrix `A ∈ ℝ^{N×N}` (stored in CSR format) and an initial vector `x₀`, the program repeatedly applies

```
y = A_shifted · x        (sparse matrix-vector product, SpMV)
x = y / ‖y‖₂             (L2 normalization — a global reduction)
```

for a fixed number of iterations (`NUM_ITERS = 500`). Every `EPOCH_LEN = 25` iterations the matrix **evolves**: its rows are *circularly shifted* by a fixed amount. The matrix itself never changes — the shift is represented logically by an integer `row_shift`, and "logical row `i` reads from source row `(i − row_shift) mod N`".

The iteration is a power-iteration-like scheme: it converges towards the dominant eigenvector, and the final **Rayleigh-like value** `xᵀ(Ax)` plus a **checksum** are emitted for correctness checking.

The whole thing is structured as a **Map + Reduce** loop:

| Per-iteration step | Pattern | Cost |
|---|---|---|
| `y = A_shifted · x` | Map over rows | O(nnz) |
| `‖y‖₂` | Global reduction | O(N) |
| `y ← y / ‖y‖₂` | Map (scaling) | O(N) |

---

## 2. `utils.hpp` — generic helpers

Header-only collection of utilities shared by every implementation. All functions are `static` (internal linkage) so the header can be included in multiple translation units without ODR violations.

### 2.1 `class SplitMix64` — deterministic PRNG

A minimal, fast, **reproducible** pseudo-random generator. Reproducibility is essential: every implementation must build the *same* matrix and the *same* initial vector from the same seed so results can be compared.

- `SplitMix64(seed)` — seeds the 64-bit internal `state`.
- `next_u64()` — advances the state by the SplitMix64 constant `0x9e3779b97f4a7c15` (the golden-ratio increment) and runs it through `mix()`.
- `next_unit()` — returns a `double` in `[0, 1)`. It takes the top 53 bits of a 64-bit word (`x >> 11`) and divides by `2^53` (`9007199254740992.0`), which is the standard way to produce a uniform double without bias.
- `mix(x)` *(static)* — the SplitMix64 finalizer: a sequence of xor-shift / multiply steps that thoroughly scrambles the bits. Used both internally and externally (matrix generation and the checksum both call it directly).

Because the generator is fully deterministic, two runs with the same seed produce identical streams — this is what makes cross-implementation correctness checks meaningful.

### 2.2 Command-line parsing

- `read_arg_u64(argc, argv, name, out)` — scans `argv` for a flag `name` and parses the following token as an unsigned 64-bit integer (`strtoull`). Returns `true` if found. Used for `-n`, `-nz`, `-s`, and (in the threaded version) `-t`.
- `read_arg_str(argc, argv, name, out)` — same, but stores the following token as a `std::string`. Used for `-m` and `--dump-vector`.

Both iterate `i` while `i + 1 < argc` so that a flag always has a value token after it.

### 2.3 `checksum_vector(x)` — order-independent fingerprint

Computes a 64-bit fingerprint of a vector:

```
acc ^= mix( bits(x[i]) ^ mix(i) )      for every i
```

- `bits(x[i])` is the raw IEEE-754 bit pattern of the double (obtained via `memcpy`, the well-defined way to type-pun).
- Mixing the **index** `i` into the value prevents two swapped elements from cancelling out.
- The accumulation uses **XOR**, which is commutative and associative — so the checksum does **not** depend on the order in which elements are visited. This is useful for parallel code, but note it only matches the sequential checksum when the final vector is *bitwise identical*; floating-point reordering in a parallel reduction will generally change it (see §6.4).

### 2.4 `dump_vector(path, x)`

Writes the vector to a text file, one value per line, at `setprecision(17)` (enough decimal digits to round-trip a `double` exactly). Used by the optional `--dump-vector` mode for element-wise correctness comparison on small inputs. Throws on I/O failure.

### 2.5 `usage(prog)`

Prints the command-line synopsis to `stderr`.

---

## 3. `matrix_generation.hpp` — building the CSR matrix

Generates a sparse matrix in **CSR (Compressed Sparse Row)** format. Generation is deterministic (seeded) and is **never** part of the measured computation time.

### 3.1 Data structures

```cpp
struct CSRMatrix {
    std::size_t n;                       // matrix is n x n
    std::vector<std::uint64_t> row_ptr;  // size n+1; row i spans [row_ptr[i], row_ptr[i+1])
    std::vector<std::uint32_t> col_idx;  // size nnz; column index of each stored value
    std::vector<double>        values;   // size nnz; the stored values
};
```

- `row_ptr` is a **prefix sum**: `row_ptr[i+1] - row_ptr[i]` is the number of nonzeros (`nnz`) in row `i`, and `row_ptr[n]` is the total `nnz`. (The threaded partitioner exploits this prefix-sum property directly — see §6.2.)
- `MatrixStats` holds total nnz and the min/max nnz per row; `GeneratedMatrix` bundles the matrix with its stats.

### 3.2 Sparsity patterns: `regular` vs `irregular`

The number of nonzeros per row is decided **first**, then the columns are filled.

- **`regular_nnz_pattern`** — distributes `nz_total` nonzeros as evenly as possible: every row gets `nz_total / n`, and the first `nz_total % n` rows get one extra. Nearly uniform load → a *balanced* workload.

- **`irregular_weight(i, n)`** — assigns each row a weight depending on which band it falls in:
  - rows `[0, n/10)`: weight **40** (and ×4 for some rows where `i % 16 < 2`, i.e. up to **160**),
  - rows `[n/10, n/4)`: weight **10** (×2 for `i % 16 == 0`),
  - rows `[n/4, 3n/5)`: weight **3**,
  - the rest: weight **1**.

  So the **heavy rows are concentrated in a contiguous block at the top** of the matrix. This is the load-imbalance scenario the project is about.

- **`weighted_nnz_pattern`** — turns those weights into an exact nonzero count per row. Every row starts with 1 nonzero (the diagonal), then the remaining `nz_total − n` "extras" are distributed proportionally to the weights. It uses the classic *largest-remainder* method: take the floor of each ideal share, then hand out the leftover units to the rows with the biggest fractional remainders (sorted), capping any row at `n` (a row can't have more than `n` columns). A final sweep guarantees every requested nonzero is placed.

- **`make_nnz_pattern`** — validates the inputs (`0 < n ≤ uint32 max`, `n ≤ nz_total ≤ n²`) and dispatches to the regular or irregular variant.

### 3.3 Filling columns: `generate_matrix`

For each row `i` (with `row_nnz` nonzeros):

1. **Diagonal first.** `col_idx[begin] = i` with value `1.1 + 0.9·rand` (so the diagonal is always present and dominant, ≈ `[1.1, 2.0)`). A strong diagonal keeps the power iteration well-behaved.
2. **Off-diagonal columns** are chosen by walking the column space with a **coprime stride** (`make_coprime_stride`): starting at a random column, repeatedly add a stride that is coprime to `n`. Because the stride is coprime to `n`, the walk visits distinct columns and never repeats before covering them all — this guarantees `row_nnz` *distinct* columns without needing a set/dedup. The diagonal column is skipped if the walk lands on it.
3. **Row normalization.** Each row is scaled so its values have unit L2 norm (`values[p] *= 1/√(Σ values²)`). This bounds the spectral properties of `A` and keeps the iteration numerically stable.

Each row uses an **independent RNG** seeded from `seed ^ mix(i+1)`, so generation is deterministic and, importantly, **per-row independent** (a property a future parallel generator could exploit).

`print_matrix_stats` reports rows, total/min/max/avg nnz per row — for the irregular matrix the min/max gap is large, quantifying the imbalance.

---

## 4. `iterative_SpMV.cpp` — sequential reference

This is the **ground truth** for both correctness and algorithm structure. The parallel versions must reproduce its numerical result (within tolerance) and keep its structure recognizable.

### 4.1 Constants

- `NUM_ITERS = 500` — total iterations.
- `EPOCH_LEN = 25` — iterations between two matrix-evolution (row-shift) steps → 20 epochs total.

### 4.2 Vector operations

- `dot(a, b)` — `std::inner_product` (sequential left-to-right summation; this fixed order is the FP reference).
- `l2_norm(x)` — `√dot(x, x)`.
- `normalize(x)` — divides every element by its L2 norm. This is the **global reduction** of each iteration.

### 4.3 `compute_shift_rows(n)`

Computes the per-epoch shift amount: `s = n/16 + 17`, forced odd, reduced mod `n`, and never zero. A fixed, deterministic function of `n`.

### 4.4 `spmv_csr_shifted_rows(A, row_shift, x, y)` — the kernel

Computes `y = A_shifted · x`:

```cpp
for i in [0, n):
    src_row = (i + n - row_shift) % n     // logical row i reads source row (i - shift) mod n
    y[i] = Σ_{p in row src_row} A.values[p] * x[A.col_idx[p]]
```

The matrix data is **never moved**. The evolution is expressed purely by remapping which physical CSR row feeds logical output row `i`. `y` is fully overwritten each call (`y.assign(n, 0.0)`).

### 4.5 `iterative_spmv_evolving(A, seed, final_vector)`

The timed core. Three phases:

- **Phase 1 — initialization.** Build `x` from `SplitMix64(seed ^ 0x123456789abcdef0)` and `normalize` it. (Parallel versions must reproduce this exact vector.)
- **Phase 2 — the iterative loop** (500 iters):
  - at each epoch boundary (`iter > 0 && iter % EPOCH_LEN == 0`), advance `row_shift = (row_shift + shift_rows) % n`;
  - `spmv_csr_shifted_rows` → `normalize(y)` → `x.swap(y)`.
- **Phase 3 — diagnostics.** One extra SpMV, then `rayleigh = dot(x, y)` and `checksum = checksum_vector(x)`. Optionally keep the final vector for dumping.

Returns `{rayleigh, checksum, final_row_shift}`.

### 4.6 `main`

Parses `-n -nz -m [-s] [--dump-vector]`, generates the matrix (timed separately, **not** counted), prints stats, then runs and times **only** `iterative_spmv_evolving` with `steady_clock`. Prints `rayleigh`, `checksum`, and `Time (sec)`. The optional vector dump happens **after** the timer stops.

---

## 5. `iterative_SpMV_threads.cpp` — C++ threads implementation

A parallel counterpart that **preserves the algorithm and the initialization** of the sequential reference, so its output is comparable within a numerical tolerance. It uses only concurrency primitives from the course: `std::thread`, `std::mutex`, `std::condition_variable`, and move/`ref` semantics.

### 5.1 The central design idea — partition the *source* rows

The cost of producing output row `i` equals `nnz(src_row)` where `src_row = (i − shift) mod n`. In the irregular matrix the heavy rows form a contiguous block at the top; after a shift, that heavy block lands at output positions `[shift, shift + heavy)` — **which moves every epoch**. So partitioning *output* rows would give each thread a load that swings wildly from epoch to epoch.

Instead, the kernel **iterates over source rows** and writes to the shifted output position `(src + shift) mod n`. Each thread is given a fixed set of *source* rows, so its load is `Σ nnz(src)` over that set — **constant across all epochs**. Because `src → (src+shift) mod n` is a bijection, every `y` entry is written by exactly one thread (no data races, no atomics on `y`). Consequence: on a single shared-memory node the matrix evolution costs nothing but an integer index change — no data movement, no repartitioning.

### 5.2 `class Barrier` — reusable synchronization

A **generation-counter barrier** built from a `mutex` + `condition_variable` (there is no barrier in the C++ standard library that the course covers, so it's hand-built):

- each thread calls `arrive_and_wait()` and decrements `count_`;
- the **last** arriver resets `count_`, bumps `generation_`, and `notify_all()`;
- the others `wait` on a **predicate** (`gen != generation_`), which correctly handles spurious wake-ups (a point the course stresses).

The generation counter is what makes the barrier *reusable* across all 500 iterations.

### 5.3 `nnz_balanced_partition(A, P)` — static load balancing

Splits the source rows `[0, n)` into `P` contiguous ranges of approximately equal **nonzero** count. It exploits the fact that `row_ptr` already *is* the prefix sum of nnz: for worker `t` the target cumulative nnz is `total_nnz · t / P` (computed in 128-bit to avoid overflow), and `std::lower_bound` finds the first row reaching that target. Boundaries are clamped to stay monotonic and within `[0, n]`. This is done **once**, before the loop — and thanks to §5.1 it stays balanced for every epoch.

### 5.4 `worker_body(...)` — what each thread runs

Each worker owns source-row range `[s0, s1)` and runs the **entire** computation (loop + diagnostics) for that range. Two **ping-pong buffers** (`buf0`, `buf1`) replace the explicit `x/y` swap: on even iterations read `buf0`/write `buf1`, on odd ones the reverse. This removes any serial swap step from the loop.

Per iteration:

1. **Local shift derivation.** `shift = (shift_rows · (iter / EPOCH_LEN)) % n`. This is computed independently by each thread and is *mathematically identical* to the sequential cumulative `row_shift` — so no shared shift state and no serialization is needed.
2. **Phase A (Map + local reduce).** For each owned source row: compute the dot product into `sum`, write `yw[(src+shift)%n] = sum`, and accumulate `local_ss += sum²`. Store `local_ss` in `partial[tid]`. The sum-of-squares is **fused** into the SpMV so the norm needs no extra pass.
3. **Barrier #1** — wait for all partial sums.
4. **Global reduce.** Each thread sums `partial[0..P)` itself, in the same fixed order, giving an identical `inv = 1/√total` everywhere (no shared value to publish, no extra race).
5. **Phase B (Map).** Scale the entries this thread produced by `inv`.
6. **Barrier #2** — the buffer just written becomes next iteration's read buffer, so all writes must be visible first.

After the loop, the final vector is in `buf0`. Each worker then does the **extra SpMV** for diagnostics (using the final shift), writing into `buf1` and accumulating its slice of `dot(x, y)` into `partial_dot[tid]`. One last barrier, then thread 0 reduces `partial_dot` into the shared `rayleigh`.

### 5.5 `iterative_spmv_evolving_threads(...)`

The driver:

- builds the initial vector **serially**, byte-for-byte as the sequential reference (`buf0`);
- clamps the worker count `P` to `[1, n]`;
- computes the nnz-balanced partition;
- allocates `partial` / `partial_dot` arrays and the `Barrier`;
- spawns `P` `std::thread`s (handles kept in a `vector`, created with `emplace_back`; arguments passed with `std::ref` / `std::cref` because threads copy their arguments by default);
- `join`s them all;
- computes the order-independent `checksum` of the final `buf0` (still inside the timed region, matching the sequential reference);
- returns `{rayleigh, checksum, final_row_shift}`.

### 5.6 `main`

Same flags as the sequential reference **plus** `-t` / `--threads` (defaults to `std::thread::hardware_concurrency()`). Prints the tag `SPARSE_ITERATION_CPP_THREADS`, the chosen `threads` count, generation time (untimed), then times **only** the parallel computation. The optional dump happens after the timer stops.

### 5.7 What is and isn't in the timed region

To stay comparable with the sequential baseline, the timer wraps exactly what the sequential one does: serial initialization + the iterative loop + the final extra SpMV + Rayleigh + checksum. Thread spawn/join overhead is therefore *included* (an honest cost of the parallel approach), while matrix generation and the optional dump are *excluded*.

---

## 6. Correctness notes

- **Same initial vector.** Phase-1 initialization is byte-identical to the sequential reference, so both start from the same `x₀`.
- **Same logical evolution.** The per-iteration `shift` derived in the worker equals the sequential cumulative `row_shift`.
- **Floating-point differences are expected.** The parallel norm/Rayleigh reductions sum in a different order than the sequential `inner_product`, so results are **not bitwise identical** (the project explicitly allows this). Verify by:
  - small inputs: `--dump-vector` from both, compare element-wise within a tolerance;
  - larger inputs: compare the **Rayleigh value** within a relative tolerance.
- The XOR-based `checksum` will generally **differ** between sequential and threaded once the final vectors differ in their low bits — use it as a quick self-consistency signal (e.g. identical across thread counts only when the FP result is identical), not as a strict equality test against the sequential run.

---

## 7. `Makefile`

- `make` / `make all` — builds both `seq` and `threads`.
- `make seq`, `make threads` — build individually.
- `make clean` — remove the executables.

Optimization flags: `-O3 -march=native -funroll-loops -DNDEBUG`, `-std=c++20`, `-Wall -Wextra`; the threaded target adds `-pthread`. Both targets depend on `matrix_generation.hpp` and `utils.hpp`, so a header change triggers a rebuild.

> **Portability caveat:** `-march=native` tunes for the *compiling* machine's CPU. Compile on the same node type you run on (e.g. directly on the cluster node); otherwise replace it with `-mtune=native` or drop it to avoid illegal-instruction faults on a different micro-architecture.
