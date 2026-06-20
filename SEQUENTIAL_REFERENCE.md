# Understanding the Sequential SpMV Reference

A guided, expanded explanation of the professor-provided sequential reference for the
One-Shot project *"Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices"*.

Files covered:
- `iterative_SpMV.cpp` — the algorithm and command-line driver
- `matrix_generation.hpp` — sparse-matrix construction in CSR format
- `utils.hpp` — generic helpers (RNG, argument parsing, checksum, vector dump)

---

## 1. What the program actually computes

Stripped to its essence, the program runs a **power-iteration-like loop**:

```
x <- random unit vector
repeat NUM_ITERS times:
    y <- A . x          (sparse matrix-vector product)
    y <- y / ||y||      (normalize to unit length)
    swap(x, y)
```

This is the classic *power method*: repeatedly applying a matrix to a vector and
renormalizing makes the vector converge toward the dominant eigenvector of `A`, and the
quantity `xᵀ A x` (computed at the end as `rayleigh`) converges toward the dominant
eigenvalue. You do **not** need to care about the eigenvalue theory to parallelize the code —
what matters is that the loop body is a **sparse matrix-vector product (SpMV)** followed by a
**normalization** (which hides a global reduction).

The twist that makes this project interesting is **evolution**: the matrix is not static.
Every `EPOCH_LEN` iterations the rows are circularly shifted, so the operator changes over
time. The sequential code simulates this cheaply with an integer offset (Section 5).

Constants (top of `iterative_SpMV.cpp`):

| Constant     | Value | Meaning                                            |
|--------------|-------|----------------------------------------------------|
| `NUM_ITERS`  | 500   | total iterations of the loop                       |
| `EPOCH_LEN`  | 25    | iterations between two evolution steps (20 epochs) |

---

## 2. CSR format — the data structure at the center of everything

The matrix is stored as **Compressed Sparse Row (CSR)** — `struct CSRMatrix`
(`matrix_generation.hpp`). Only nonzeros are stored, in three parallel arrays:

- `row_ptr` — size `n + 1`. `row_ptr[i]` is the position where row `i`'s entries start;
  row `i` occupies the half-open range `[row_ptr[i], row_ptr[i+1])`.
- `col_idx` — size `nnz`. The column index of each stored nonzero.
- `values`  — size `nnz`. The numeric value of each stored nonzero.

### Concrete example

For the 3×3 matrix
```
[ 5  0  8 ]
[ 0  3  0 ]
[ 0  6  7 ]
```
CSR is:
```
row_ptr = [0, 2, 3, 5]     // row0: [0,2), row1: [2,3), row2: [3,5)
col_idx = [0, 2, 1, 1, 2]
values  = [5, 8, 3, 6, 7]
```
`row_ptr[i+1] - row_ptr[i]` is the number of nonzeros in row `i` (e.g. row 0 has 2).

### The one loop you must internalize

Iterating row `i` (and the SpMV inner loop) always looks like:
```cpp
for (p = row_ptr[i]; p < row_ptr[i+1]; ++p) {
    // nonzero at column col_idx[p] with value values[p]
    sum += values[p] * x[col_idx[p]];
}
```
This is the kernel that dominates runtime and is what every parallel version optimizes.

---

## 3. `utils.hpp` — generic helpers

### `SplitMix64` (PRNG)
A small, fast, fully deterministic random generator.
- `next_u64()` — advance state by the golden-ratio constant `0x9e3779b9...`, return a mixed
  64-bit value.
- `next_unit()` — a `double` in `[0, 1)` (top 53 bits scaled).
- `static mix(x)` — a 64-bit avalanche hash; reused by the checksum and by per-row seeding.

**Why it matters:** determinism is the backbone of correctness checking. Given the same
seed, the matrix and the initial vector are byte-for-byte identical, so a parallel version
can be compared against the sequential one exactly.

### Command-line parsing
- `read_arg_u64(argc, argv, name, out)` — find flag `name`, parse the following token as an
  unsigned 64-bit integer; returns `false` if the flag is absent (used to detect missing
  required arguments).
- `read_arg_str(...)` — same, but returns the raw string.

### Correctness helpers
- `checksum_vector(x)` — folds the **raw bit pattern** of each element (via `memcpy` into a
  `uint64_t`, then `mix`, XOR-combined with the mixed index) into a single 64-bit value. This
  is the primary equivalence test: two runs are considered equal iff their checksums match.
  Comparing bits (not approximate floats) makes it a strict check.
- `dump_vector(path, x)` — writes every element at 17 significant digits (enough to round-trip
  a `double`) to a text file. Enabled by `--dump-vector`; lets you diff full vectors when a
  checksum mismatch needs investigating.

### `usage(prog)`
Prints the command-line synopsis to `stderr`.

---

## 4. `matrix_generation.hpp` — building the matrix

The goal: deterministically build a CSR matrix with **exactly** `nz` nonzeros, distributed
across rows by a chosen pattern, with row values normalized. Read top-down.

### Step A — decide how many nonzeros each row gets

- `irregular_weight(i, n)` — assigns a *weight* to each row based on its position:
  first `n/10` rows → weight 40 (every ~16th even higher), next up to `n/4` → 10, up to
  `3n/5` → 3, the rest → 1. These weights are what create **dense regions** in `irregular`
  mode — the uneven distribution that makes static, equal-row partitioning perform badly and
  motivates dynamic load balancing in the parallel versions.
- `regular_nnz_pattern(n, nz)` — `regular` mode: each row gets `nz/n` nonzeros, the remainder
  spread one-each over the first rows. Nearly uniform.
- `weighted_nnz_pattern(n, nz)` — `irregular` mode: every row starts with 1 (the diagonal),
  the remaining `nz - n` nonzeros are distributed proportionally to the weights using the
  **largest-remainder method**: floor each row's ideal share, then sort the fractional
  leftovers and hand out the remaining slots so the total is **exactly** `nz`. Capacity caps
  prevent any row from exceeding `n` columns.
- `make_nnz_pattern(...)` — validates (`n>0`, `n` fits in `uint32_t`, `n ≤ nz ≤ n*n`) and
  dispatches to regular/irregular; throws on an unknown mode.

### Step B — place the nonzeros

- `make_coprime_stride(raw, n)` — returns a stride that is **coprime with `n`** (odd, and
  `gcd(stride, n) == 1`). Walking columns as `start, start+stride, start+2·stride, ...`
  modulo `n` then visits every column exactly once before repeating — so within a row no two
  generated entries collide on the same column.
- `generate_matrix(n, nz, seed, mode)` — the assembler:
  1. Build `row_ptr` as a prefix sum of the nnz pattern; assert the total equals `nz`.
  2. For each row `i`, seeded by `seed ^ mix(i+1)` (independent, reproducible per row):
     - place the **diagonal** entry first (column `i`, value `1.1 + 0.9·rand` ⇒ in `[1.1, 2.0)`,
       making the diagonal dominant);
     - fill the remaining `row_nnz - 1` entries by striding through columns (skipping the
       diagonal), each with value in `[0.1, 1.0)`;
     - **normalize the row** to unit L2 norm. Row normalization bounds the operator and keeps
       the iteration numerically well-behaved.
  3. Record `MatrixStats` (min / max / total nnz per row).
- `print_matrix_stats(G)` — prints `rows`, `total_nnz`, `min/max/avg nnz per row`.

---

## 5. `iterative_SpMV.cpp` — the algorithm step by step

### Vector helpers
- `dot(a, b)` — inner product via `std::inner_product`.
- `l2_norm(x)` — `sqrt(dot(x, x))`.
- `normalize(x)` — divide by the L2 norm. **Note:** the norm needs every element, so this is
  a **global reduction** — in parallel code it is the synchronization point each iteration.

### The evolution mechanism (the conceptual core)
- `compute_shift_rows(n)` — the fixed shift amount `s` used at each epoch (≈ `n/16 + 17`,
  forced odd and reduced mod `n`). This is *how far* the rows move per evolution step.
- `spmv_csr_shifted_rows(A, row_shift, x, y)` — SpMV with evolution folded in **logically**.
  The matrix is never physically moved. Output row `i` is computed from **source row**
  `(i + n - row_shift) % n` of the fixed CSR matrix:
  ```cpp
  for (i = 0; i < n; ++i) {
      src = (i + n - row_shift) % n;
      sum = 0;
      for (p = row_ptr[src]; p < row_ptr[src+1]; ++p)
          sum += values[p] * x[col_idx[p]];
      y[i] = sum;
  }
  ```
  **Key insight:** "evolution" is just the integer `row_shift`. The sequential version gets it
  almost for free. A *distributed* version, however, must keep this remapping consistent with
  how rows are physically partitioned across processes — that is the genuinely hard part the
  header comment warns about.

### `iterative_spmv_evolving(A, seed, final_vector?)` — the heart
- **Phase 1 — initialize:** allocate `x`, `y`; fill `x` from a seeded `SplitMix64`
  (`seed ^ 0x1234...`); `normalize(x)`. A parallel version must reproduce *this exact* initial
  vector (same seed, same fill order) to match the checksum.
- **Phase 2 — timed loop** (`iter = 0 .. NUM_ITERS-1`):
  - at each epoch boundary (`iter > 0 && iter % EPOCH_LEN == 0`) advance
    `row_shift = (row_shift + s) % n`;
  - `spmv_csr_shifted_rows(...)` → `normalize(y)` → `x.swap(y)`.
- **Phase 3 — diagnostics:** one extra SpMV, then `rayleigh = dot(x, y)` (a Rayleigh-quotient-
  like scalar) and `checksum = checksum_vector(x)`; optionally move out the final vector for
  dumping.

### `main(argc, argv)` — the driver
- **Phase 0 — parse:** required `-n`, `-nz`, `-m regular|irregular`; optional `-s seed`
  (default 111) and `--dump-vector FILE`. Missing required args → `usage` and exit 1.
- **Phase 1 — generate (timed separately):** `generate_matrix(...)`, print stats and
  `generation_time_sec`. **Generation is deliberately excluded from the measured computation.**
- **Phase 2 — compute (timed):** `iterative_spmv_evolving(...)`; print `rayleigh`, `checksum`
  (hex), and `Time (sec)`.
- **Phase 3 — optional dump (untimed):** if `--dump-vector` was given, write the final vector.

---

## 6. A worked matrix-vector multiplication (following the flow)

A small but faithful trace of `main → iterative_spmv_evolving → spmv_csr_shifted_rows →
normalize → swap`, on a 4×4 matrix. *(The real generator L2-normalizes each row and starts
from a random vector; clean integers are used here so the arithmetic is readable — the control
flow is identical.)*

### Setup: a 4×4 matrix in CSR

```
        col0 col1 col2 col3
row0 [   2    0    1    0  ]
row1 [   0    3    0    0  ]
row2 [   1    0    4    1  ]
row3 [   0    2    0    5  ]
```

nnz per row = `[2, 1, 3, 2]`, total = 8, so:

```
row_ptr = [0, 2, 3, 6, 8]
col_idx = [0, 2,   1,   0, 2, 3,   1, 3]
values  = [2, 1,   3,   1, 4, 1,   2, 5]
          └ row0 ┘└row1┘└─ row2 ─┘└row3┘
```

### Phase 1 — the starting vector

`iterative_spmv_evolving` fills `x` from the seeded RNG and normalizes it. Pretend that
produced (simplified):

```
x = [1, 2, 3, 4]
```

### Iteration 0 — `spmv_csr_shifted_rows` with `row_shift = 0`

`row_shift` starts at 0, so `src_row = (i + n - 0) % n = i` (logical row `i` reads matrix row
`i`). Each output element is the CSR inner loop over that row:

```
y[0] = 2*x[0] + 1*x[2]            = 2*1 + 1*3       = 5
y[1] = 3*x[1]                     = 3*2             = 6
y[2] = 1*x[0] + 4*x[2] + 1*x[3]   = 1*1 + 4*3 + 1*4 = 17
y[3] = 2*x[1] + 5*x[3]            = 2*2 + 5*4       = 24
```

`y = [5, 6, 17, 24]`.

**normalize(y):**

```
||y|| = sqrt(5² + 6² + 17² + 24²) = sqrt(926) ≈ 30.4302
y  ←  [5, 6, 17, 24] / 30.4302 ≈ [0.1643, 0.1972, 0.5586, 0.7887]
```

**swap(x, y):** that normalized vector becomes `x` for iteration 1. Iterations 1…24 repeat
SpMV → normalize → swap, still with `row_shift = 0`.

### Iteration 25 — an epoch boundary fires the evolution

At the top of the loop `25 % EPOCH_LEN(25) == 0`, so `row_shift = (row_shift + s) % n`.
`compute_shift_rows(4) = 4/16 + 17 = 17`, odd, `17 % 4 = 1` → `s = 1`, so `row_shift` becomes
`1`. Now `src_row = (i + 4 - 1) % 4 = (i + 3) % 4`:

```
logical row 0 → src_row 3
logical row 1 → src_row 0
logical row 2 → src_row 1
logical row 3 → src_row 2
```

Re-running the SpMV on `x = [1,2,3,4]` with `row_shift = 1` to show the effect:

```
y[0] uses matrix row 3: 2*x[1] + 5*x[3]          = 24
y[1] uses matrix row 0: 2*x[0] + 1*x[2]          = 5
y[2] uses matrix row 1: 3*x[1]                   = 6
y[3] uses matrix row 2: 1*x[0] + 4*x[2] + 1*x[3] = 17
```

`y = [24, 5, 6, 17]` — the same per-row results as before, just **cyclically rotated by one
position**. That is exactly "the matrix evolved by a circular row shift", achieved with zero
data movement: only the integer `row_shift` changed. (At iter 50 it becomes 2, at 75 → 3, …)

### Phase 3 — final diagnostics

After all 500 iterations the code does one more SpMV into `y` (with the final `row_shift`),
then:

```
rayleigh = dot(x, y)            // x · (A_shifted · x), a single scalar
checksum = checksum_vector(x)   // strict bitwise fold of the final x
```

Those two values are printed, and the `checksum` is what any parallel version must reproduce
for identical `-n -nz -m -s`.

### Takeaways the example shows

1. **SpMV = one CSR inner loop per output row**, summing `values[p] * x[col_idx[p]]`.
2. **normalize is global** — it touches all of `y` (the parallel synchronization point).
3. **swap** makes this iteration's output the next iteration's input.
4. **evolution is just `src_row = (i + n - row_shift) % n`** — a remap, not a real shuffle.

---

## 7. How the pieces fit together

```
main (iterative_SpMV.cpp)
 ├─ read_arg_* / usage            (utils.hpp)      parse CLI
 ├─ generate_matrix               (matrix_generation.hpp)  build fixed CSR matrix [untimed]
 │    └─ make_nnz_pattern, make_coprime_stride, SplitMix64 (utils.hpp)
 ├─ iterative_spmv_evolving       (iterative_SpMV.cpp)     500 normalized SpMV steps [timed]
 │    ├─ compute_shift_rows                              evolution amount
 │    ├─ spmv_csr_shifted_rows                           the SpMV kernel (+ logical shift)
 │    └─ normalize / dot / l2_norm                       global reductions
 └─ checksum_vector / dump_vector (utils.hpp)      correctness outputs
```

One sentence: build a fixed, row-normalized CSR matrix once, then run 500 normalized SpMV
steps whose operator "evolves" via an integer row-shift, and summarize the result with a
Rayleigh value and a strict bitwise checksum.

---

## 8. Implications for the parallel versions

- **Optimize Phase 2 only.** Generation and dumping are outside the timed region.
- **The SpMV is embarrassingly parallel over output rows** — but irregular mode gives wildly
  uneven row lengths, so equal-row partitioning is poorly balanced; balance by nonzeros or use
  dynamic scheduling.
- **`normalize` is a barrier.** Each iteration ends with a global reduction (the norm), so the
  iterations are inherently serialized at that point.
- **Reproduce initialization exactly** (seed, fill order) or the checksum won't match.
- **The shift is cheap here, not in a distributed layout.** Re-deriving `src_row` per row is
  trivial shared-memory; across processes the row remapping must match the data distribution.
- **Validate with the checksum** (and `--dump-vector` for deep diffs) for identical
  `-n -nz -m -s`.

---

## 9. Build and run

```bash
g++ -O3 -std=c++20 -I . -Wall iterative_SpMV.cpp -o seq

# small run with a vector dump for inspection
./seq -n 5000 -nz 20000 -m irregular --dump-vector seq_vec.dump

# the real workloads
./seq -n 500000 -nz 20000000 -m irregular
./seq -n 500000 -nz 20000000 -m regular
```

The `checksum=` line is the value any parallel version must reproduce for the same arguments.
Reference outputs already live in `results/` (e.g. `seq_n100000_nz4000000.txt`).
