# Parallel Implementations — Deep Dive

**Project:** Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
**Files explained here, line by line:**
`src/threads/iterative_SpMV_threads.cpp` (C++ threads) and
`src/openmp/iterative_SpMV_omp.cpp` (OpenMP based on tasks), plus two comparison
variants: `src/threads/iterative_SpMV_pool.cpp` (C++ threads driven by the course
thread pool — §3) and `src/openmp/iterative_SpMV_omp_ws.cpp` (optional OpenMP
**work-sharing** version — §2.9).

This is the **deep companion** to `DOCUMENTATION.md`. That document is a *survey* of every file
in the project; here we zoom in on the two parallel sources and explain **every concurrency
construct and technique they use** — what it is, why it is there, and how it maps onto the
algorithm — at a level aimed at a reader learning the techniques. Where `DOCUMENTATION.md` says
"uses a `std::barrier`", this document explains what a completion-function barrier *is*, why its
functor must be `noexcept`, and what redundant work it removes.

If you have not yet, skim `DOCUMENTATION.md §1` (the Map+Reduce problem) and `§4` (the
sequential reference) first — everything below assumes that baseline.

---

## 0. The shared foundation

### 0.1 The algorithm both versions reproduce

Both parallel versions are *counterparts of the sequential reference*: they keep its exact
structure and its exact numerical initialization, so their output is comparable within a
tolerance. The skeleton is:

```
x  = normalize( random_vector(seed) )                 // serial init (identical everywhere)
repeat NUM_ITERS = 500 times:
    every EPOCH_LEN = 25 iters: row_shift += shift_rows   // the matrix "evolves"
    y = A_shifted · x                                  // shifted SpMV  (Map over rows, O(nnz))
    x = y / ‖y‖₂                                       // L2 normalize  (global Reduce + Map)
rayleigh = xᵀ(A_shifted · x) ;  checksum = fingerprint(x)   // diagnostics
```

The matrix `A` (CSR) is **never physically moved**. "Evolution" is a single integer
`row_shift`: logical output row `i` is fed by source row `(i − row_shift) mod n`. So an epoch
change costs nothing but an index remap. Only the *parallelization* of the per-iteration Map and
Reduce differs between the two files.

### 0.2 The one design idea both versions share: split **source** rows, not output rows

This is the crux that makes both implementations efficient, so it is worth stating once.

The cost of producing one output row equals the number of nonzeros (`nnz`) of the **source** row
that feeds it. In the *irregular* matrix the heavy rows form a contiguous block at the top (see
`DOCUMENTATION.md §3.2`). After a shift, that heavy block lands at output positions
`[shift, shift+heavy)` — a region that **moves every epoch**. So if you partitioned *output*
rows, each thread's load would swing wildly from epoch to epoch.

Instead, both kernels **iterate over source rows** `src` and write to the shifted position
`(src + shift) mod n`:

```cpp
for src in my_range:
    sum   = Σ_p  values[p] * x[col_idx[p]]      // dot product of source row src
    out   = (src + shift) % n                    // where it lands this epoch
    y[out] = sum
```

Two consequences make this the right choice:

1. **Constant load.** A thread that owns a fixed set of source rows always does
   `Σ nnz(src)` work over that set — *independent of the shift*, so balanced for all 500
   iterations with a partition computed **once**.
2. **No races on `y`.** `src → (src + shift) mod n` is a **bijection** (a rotation of
   `[0, n)`), so every `y` entry is written by exactly one thread. No atomics, no locks on the
   output vector.

The threads version turns this into a *static, hand-balanced* partition; the OpenMP version turns
it into *dynamic, runtime-balanced* tasks. Everything else below is detail on those two answers.

---

## 1. C++ threads — `iterative_SpMV_threads.cpp`

> Survey-level overview: `DOCUMENTATION.md §5`. This section is the line-level version.

The file uses only primitives the course covers: `std::thread`, `std::barrier` (C++20),
`std::ref`/`std::cref`, plus POSIX affinity. Its header (lines 31–39) labels five optimizations
**A–E**, each grounded in the SPM notes / `Code/` examples; we explain each in place.

### 1.1 Where each phase lives

| Phase | Code |
|---|---|
| Serial init (build + normalize `x₀`) | `iterative_spmv_evolving_threads`, lines 383–390 |
| Partition + setup | same function, lines 397–406 |
| The 500-iteration loop + diagnostics | `worker_body`, lines 267–371 (run by every thread) |
| Argument parsing / timing / output | `main`, lines 440–514 |

The driver does the serial setup, spawns the workers, joins them, then computes the checksum.
Each worker runs the **entire** timed computation for its own slice of source rows.

### 1.2 Persistent threads, and how arguments reach them

The `P` workers are created **once** for the whole loop (lines 410–422), not per iteration —
the spawn/join cost is paid a single time and is *deliberately inside the timed region* (§1.10),
because it is an honest cost of the threaded approach.

```cpp
threads.emplace_back(worker_body,
                     t, P, std::cref(A), shift_rows,
                     boundary[t], boundary[t + 1],
                     std::ref(buf0), std::ref(buf1),
                     std::ref(partial), std::ref(partial_dot),
                     std::ref(sync), std::cref(inv_norm), std::ref(rayleigh));
```

Two things to notice:

- **`std::thread` copies its arguments by value.** If we passed `buf0` directly, every thread
  would get its *own copy* of the vector and writes would never be shared. `std::ref` / `std::cref`
  wrap a value in a `reference_wrapper` so the thread receives a reference to the *one* shared
  object. `cref` for read-only (`A`, `inv_norm`), `ref` for read/write (the buffers, the
  reduction arrays, the barrier, `rayleigh`).
- Some of these objects (`std::barrier`) are **non-copyable** anyway, so passing them through a
  reference wrapper is not just an optimization — it is the only thing that compiles.

`emplace_back` constructs each `std::thread` in place inside the `vector` (reserved to `P`,
line 411), avoiding a move of the thread handle.

### 1.3 `nnz_balanced_partition` — static load balancing, computed once

Lines 226–255 split source rows `[0, n)` into `P` contiguous ranges of approximately equal
**nonzero** count (not equal *row* count — equal rows would be unbalanced on the irregular
matrix). The trick is that the CSR `row_ptr` array *already is the prefix sum of nnz*:

```cpp
const std::uint64_t target =
    static_cast<std::uint64_t>((static_cast<__uint128_t>(total_nnz) * t) / P);
const auto it  = std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), target);
std::size_t row = static_cast<std::size_t>(it - A.row_ptr.begin());
```

- For worker `t`, the ideal cumulative nnz at its start is `total_nnz · t / P`. The
  `__uint128_t` cast prevents `total_nnz · t` from overflowing 64 bits on large matrices.
- `std::lower_bound` binary-searches the prefix-sum array for the first row reaching that target
  — O(log n) per boundary.
- The boundaries are then clamped to stay **monotonic** and within `[0, n]` (lines 243–253) so a
  pathological distribution can't produce overlapping or backwards ranges.

Because of §0.2, this single partition stays balanced for every epoch — no repartitioning ever.

### 1.4 `std::barrier` with a completion function (Optimizations B + E)

Synchronization uses the C++20 `std::barrier`, which is **reusable**: the same object serves all
500 iterations (a barrier internally flips a phase/generation each time all participants arrive).
That is Optimization **B** — using the standard reusable barrier instead of a hand-rolled
mutex+condition_variable one.

The interesting part is the **completion function** (Optimization E). A `std::barrier` can carry
a functor that the library runs **exactly once**, on the last thread to arrive, *before* any
thread is released. We use it to do the norm reduction:

```cpp
struct NormReducer {                                   // lines 206–218
    const PaddedDouble* partial; unsigned P; double* inv_norm;
    void operator()() noexcept {
        double total = 0.0;
        for (unsigned k = 0; k < P; ++k) total += partial[k].v;
        *inv_norm = 1.0 / std::sqrt(total);
    }
};
std::barrier<NormReducer> sync(P, NormReducer{partial.data(), P, &inv_norm});  // line 405
```

- Without it, **every** thread would redundantly sum the `P` partials — `P·P` additions per
  iteration. The completion function does it **once** (`P` additions), in a single fixed order,
  and publishes `inv_norm` for everyone to read after the barrier.
- `operator()` **must be `noexcept`**: `std::barrier`'s `CompletionFunction` is required not to
  throw (an exception escaping the barrier phase would be undefined), so the type system enforces
  the annotation.

Each iteration crosses the barrier **twice** (lines 327 and 338):
- `arrive_and_wait()` #1 — after every thread has stored its partial sum-of-squares; the
  completion function then produces `inv_norm`.
- `arrive_and_wait()` #2 — after scaling; it guarantees all writes to the just-produced buffer
  are visible before the next iteration reads that buffer.

> Note: this file actually uses *both* mechanisms for safety/clarity — the completion function
> publishes `inv_norm`, and §1.8 also has each thread recompute the same sum locally. Either
> alone would be correct; see §1.8 for why they agree.

### 1.5 False-sharing padding (Optimization A)

```cpp
struct alignas(64) PaddedDouble { double v = 0.0; };   // lines 194–196
std::vector<PaddedDouble> partial(P);                   // line 399
```

Each worker writes its partial sum into `partial[tid].v` every iteration. A bare
`std::vector<double>` would pack those slots 8 bytes apart, so several would share one 64-byte
**cache line**. When core A writes its slot, the coherence protocol invalidates that line in
every other core's cache — even though cores B, C… didn't touch *their own* slot. That is
**false sharing**, and on a memory-bound kernel run at high thread counts it is a real cost.

`alignas(64)` forces each `PaddedDouble` onto its own cache line, so per-iteration partial-sum
writes never invalidate a neighbour's slot. The same padding protects `partial_dot` (the
diagnostics reduction).

### 1.6 Thread affinity / pinning (Optimization C)

Lines 91–148 pin each worker to a fixed logical CPU:

```cpp
static void pin_to_core(unsigned core) {               // Linux; no-op elsewhere
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
```

Called first thing in `worker_body` (line 281). Why it matters on this kernel:

- It stops the OS scheduler from **migrating** a worker to a different core mid-run (which would
  cold-start its caches).
- It keeps the worker on the **NUMA node that first-touched its data**. Under Linux's
  first-touch policy, a page is placed on the node of the thread that first writes it; if the
  thread later migrates, every access becomes a remote-node access. Pinning preserves locality.

The default policy is **compact**: worker `tid` pins to logical CPU `tid` (`core_for_worker`,
line 147). The `AFFINITY` environment variable overrides it with an explicit CPU list parsed by
`parse_cpu_list` (lines 106–148), which understands the course `Affinity.hpp` syntax
`"0,2,4-10:2"` → `{0,2,4,6,8,10}` (comma lists, `a-b` ranges, `:step` strides). This lets you
benchmark compact vs scattered placements **without recompiling**. The whole thing is
best-effort: on non-Linux, or if the CPU isn't in the cpuset, `pin_to_core` is a silent no-op.

### 1.7 `__restrict__` and ping-pong buffers (Optimization D)

```cpp
const std::uint64_t* __restrict__ rp = A.row_ptr.data();   // lines 287–289
const std::uint32_t* __restrict__ ci = A.col_idx.data();
const double*        __restrict__ va = A.values.data();
```

`__restrict__` is a **promise to the compiler that these pointers do not alias** any other
pointer used in the loop. Without it the compiler must assume a write through `yw` might modify
the array behind `va` or `rp`, which blocks reordering and vectorization of the tight scale/norm
loops. With the no-alias promise it can keep CSR data in registers across iterations and
vectorize freely. (It is the programmer's responsibility that the promise holds — here it does,
because the CSR arrays and the output buffer are genuinely distinct allocations.)

**Ping-pong buffers** replace the sequential reference's explicit `x.swap(y)`:

```cpp
if ((iter & 1u) == 0u) { xr = buf0.data(); yw = buf1.data(); }   // lines 301–307
else                   { xr = buf1.data(); yw = buf0.data(); }
```

On even iterations read `buf0` / write `buf1`; on odd iterations the reverse. The "swap" becomes
a branch on the iteration parity — **no serial section** inside the loop, nothing for the threads
to coordinate on. Likewise each worker derives its **`shift` locally** from `iter`
(`shift = shift_rows·(iter/EPOCH_LEN) % n`, line 296), so there is no shared shift counter and no
serialization to advance it.

### 1.8 The per-iteration dataflow (`worker_body`, lines 291–339)

Putting §1.4–1.7 together, one iteration for a worker owning `[s0, s1)` is:

1. **Phase A — fused Map + local reduce** (lines 311–321). For each owned source row: compute the
   dot product `sum`, write `yw[(src+shift)%n] = sum`, and accumulate `local_ss += sum*sum`. The
   sum-of-squares for the L2 norm is **fused into the SpMV** so the norm needs no second pass over
   the data. Store `local_ss` in `partial[tid].v`.
2. **Barrier #1** (line 327) — wait for all partials; the completion function builds `inv_norm`.
3. **Global reduce** (line 328). Each thread reads `inv = inv_norm`. (In this file each thread
   could equivalently re-sum `partial[0..P)` itself in the same fixed order; because the order is
   identical on every thread, every thread gets a **bitwise-identical** `inv`, so there is no
   divergence — that is why having both the completion function and a local read is safe.)
4. **Phase B — Map** (lines 331–334). Scale the entries this worker produced by `inv`. Note it
   re-derives `out = (src+shift)%n` so it scales exactly the entries it wrote — no shared index
   list needed.
5. **Barrier #2** (line 338) — the buffer just written becomes next iteration's read buffer, so
   all writes must be globally visible first.

### 1.9 Diagnostics (lines 341–371)

After the loop, `NUM_ITERS` is even so the final vector is in `buf0`. Each worker does **one
extra SpMV** at the final shift, writing into `buf1`, and accumulates its slice of `dot(x, y)`
into `partial_dot[tid]`. A final `arrive_and_wait()` (line 363), then **thread 0 alone** reduces
`partial_dot` into the shared `rayleigh`. Back in the driver, `checksum_vector(buf0)` (line 426)
computes the order-independent fingerprint — still inside the timed region, matching the
sequential reference exactly.

### 1.10 What is (and isn't) timed

`main` times **only** `iterative_spmv_evolving_threads` with `steady_clock` (lines 489–494). That
span includes: serial init + thread spawn/join + the 500-iteration loop + the extra SpMV +
Rayleigh + checksum — exactly the work the sequential reference times, plus the honest
spawn/join overhead. **Excluded:** matrix generation (lines 475–479, timed separately and
reported as `generation_time_sec`) and the optional `--dump-vector` (after the timer stops).

---

## 2. OpenMP based on tasks — `iterative_SpMV_omp.cpp`

> This version has **no** counterpart section in `DOCUMENTATION.md`, so it gets the most detail
> here. It reproduces §0's algorithm exactly; what changes is the engine — from hand-managed
> threads + a static partition to **OpenMP tasks** + dynamic balancing.

### 2.1 Same algorithm, different engine

The init (lines 154–161), the ping-pong buffers, the local `shift` derivation, the fused
SpMV+norm, and the diagnostics are line-for-line equivalent to the threads/sequential versions —
deliberately, so the result matches the sequential baseline within tolerance. The *only*
substantive change is how the row range `[0, n)` is divided and synchronized each iteration.

### 2.2 The `parallel → single → taskloop` idiom (lines 174–217)

```cpp
#pragma omp parallel num_threads(num_threads) shared(buf0, buf1, rayleigh)
{
    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        #pragma omp single
        {
            ... taskloops ...
        } // implicit barrier here
    }
    ...
}
```

Three nested layers, each doing one job:

- **`#pragma omp parallel`** creates the thread team **once**, around the whole loop — the OpenMP
  analogue of the threads version's persistent workers (§1.2). Creating it once avoids re-forking
  the team 500 times. `num_threads(P)` sets the team size from the CLI; `shared(...)` makes the
  buffers and `rayleigh` shared across the team rather than privatized.

- **`#pragma omp single`** picks **one** thread per iteration to act as the **task-generation
  point**. This is the key teaching point: **`taskloop` is *not* a worksharing construct.** A
  worksharing construct (like `#pragma omp for`) automatically divides its iterations among the
  encountering team. `taskloop` does not — if the *whole team* encountered it, **every thread
  would generate the entire set of tasks**, producing `P×` the intended work. Generating the
  tasks from a single thread, while the rest of the team picks them up and executes them, is the
  correct, course-idiomatic pattern (`Code/spmcode9/omp_taskloop.cpp`,
  `omp_dotprod_taskloop.cpp`).

- **The implicit barrier at the end of `single`** synchronizes the whole team before the next
  iteration begins. This is what guarantees the buffer written this iteration is fully visible
  before the next iteration reads it — the OpenMP analogue of the threads version's barrier #2
  (§1.8). (We rely on it, so we do **not** add `nowait` to the `single`.)

### 2.3 `taskloop` mechanics — chopping the loop into tasks

Inside the `single`, each iteration issues **two** taskloops:

```cpp
double ss = 0.0;
#pragma omp taskloop grainsize(grain) reduction(+: ss) ...      // Phase A: SpMV + norm
for (std::size_t src = 0; src < n; ++src) { ...; ss += sum*sum; }

const double inv = 1.0 / std::sqrt(ss);
#pragma omp taskloop grainsize(grain) ...                        // Phase B: scaling
for (std::size_t src = 0; src < n; ++src) { yw[(src+shift)%n] *= inv; }
```

`taskloop` slices the loop range `[0, n)` into chunks and wraps each chunk in an OpenMP **task**;
the idle team members execute those tasks. Crucially, a `taskloop` has an **implicit taskgroup**:
the construct does not return until *all* of its tasks have completed. That is why no explicit
`#pragma omp taskwait`/barrier is needed between the two phases — when the first taskloop returns,
`ss` is fully reduced and complete, so computing `inv` right after it (line 206) is safe, and the
scaling taskloop can then run.

Phase B can be a **plain** taskloop (no reduction) because the outputs `(src+shift)%n` form a
permutation of `[0, n)` (§0.2), so scaling "every produced entry" scales the whole written buffer
exactly once.

### 2.4 `reduction(+: ss)` on a taskloop

The L2 sum-of-squares is expressed as a **task reduction**: `reduction(+: ss)` on the taskloop
(line 194). OpenMP gives each task a private partial `ss`, runs the bodies, and combines the
partials into the final `ss` for you — the privatization and combination that the threads version
does *by hand* with `PaddedDouble partial[]` + the `NormReducer` completion function (§1.4–1.5)
are here hidden inside the runtime. There is **no false-sharing padding to write** and no manual
reduce loop.

The trade-off: the **order** in which the runtime combines task partials is unspecified and may
vary run to run. So the floating-point `ss` (and therefore the final vector's low bits and the
`checksum`) can differ from the sequential and threads versions. This is **expected and
acceptable** under the project's tolerance rule — correctness is checked by element-wise tolerance
and the Rayleigh value, not by a bitwise checksum (see `DOCUMENTATION.md §6`).

> Considered alternative: the lower-level `#pragma omp taskgroup` + `task_reduction(+: ss)` on the
> group with `in_reduction(+: ss)` on individual `#pragma omp task`s
> (`Code/spmcode9/omp_task_reduction.cpp`). `taskloop reduction` is simply the cleaner OpenMP-5
> spelling of the same idea, so we use it.

### 2.5 `grainsize` and dynamic load balancing (`resolve_grain`, lines 125–135)

```cpp
std::size_t grain = n / (static_cast<std::size_t>(P) * 8);   // ⇒ ≈ 8·P tasks
if (grain == 0) grain = 1;
```

`grainsize(grain)` tells `taskloop` to put roughly `grain` loop iterations in each task. Choosing
`grain = n/(P·8)` yields about **8·P tasks** — comfortably more tasks than threads. That surplus
is what lets the OpenMP runtime **dynamically load-balance**: when a thread that happened to draw
light (low-nnz) rows finishes early, it steals the next pending task instead of idling. This is
the OpenMP-native answer to the irregular-nonzero imbalance that the threads version solves
*statically* with `nnz_balanced_partition` (§1.3).

- Too coarse (few large tasks) → poor balancing, stragglers.
- Too fine (many tiny tasks) → task-creation/scheduling overhead dominates.

`8·P` is a reasonable default; the `GRAIN` environment variable overrides it for tuning
(line 126) — the OpenMP analogue of the threads version's `AFFINITY` knob, i.e. a runtime dial
that needs no recompile.

### 2.6 Data-sharing clauses — `default(none)` discipline

Every taskloop uses `default(none)` (e.g. lines 195, 212, 233):

```cpp
#pragma omp taskloop grainsize(grain) reduction(+: ss) \
    default(none) shared(rp, ci, va, xr, yw, n, shift)
```

`default(none)` **disables implicit data-sharing rules** and forces *every* variable referenced
in the region to be classified explicitly. It is a correctness discipline: if you forget a
variable, the code won't compile, instead of silently defaulting to `shared` or `firstprivate`
and producing a subtle bug. Here the CSR pointers, the buffers, `n`, and `shift` are `shared`
(read-only or write-to-disjoint-indices), `ss`/`dot_xy` are reduction variables, and the loop
index `src` is implicitly private to each task.

One concrete pitfall handled here: a variable may **not** appear in both `shared` and
`firstprivate` on the same construct — that is a clause conflict the compiler rejects (it was
fixed during development by keeping `n` only in `shared`). The enclosing `parallel` similarly
declares `shared(buf0, buf1, rayleigh)` (line 175) so those survive across the team.

### 2.7 Affinity via the environment, not in code

The threads version pins workers from inside the program (§1.6). The OpenMP version expresses the
*same intent* through the **standard OpenMP environment** instead of code:

```
OMP_PROC_BIND=close   OMP_PLACES=cores
```

`OMP_PLACES=cores` defines the places threads may run as physical cores; `OMP_PROC_BIND=close`
binds team threads to places *close* to the master, filling one socket compactly before
spilling to the next (the analogue of the threads version's compact default). These are set in
`scripts/run_omp.sbatch` (which exports them, defaulting to `close`/`cores`), **not** hard-coded —
so placement can be changed per run. Course reference: `Code/spmcode8/omp_affinity.cpp`.

### 2.8 Thread count, diagnostics, and timing

- **Thread count** (`main`, lines 287–291): `-t P` → the `num_threads(P)` clause; if omitted it
  defaults to `omp_get_max_threads()`. Same CLI as the threads version (`-n -nz -m -s -t
  --dump-vector`).
- **Diagnostics** (lines 223–244): a final `#pragma omp single` runs one more taskloop, this time
  `reduction(+: dot_xy)`, computing the extra SpMV at the final shift and accumulating
  `dot(x, y)` → `rayleigh`. `checksum_vector(buf0)` (line 249) runs after the parallel region but
  still inside the timed span.
- **Timing** (lines 312–317): `steady_clock` wraps `iterative_spmv_evolving_omp` — init + the
  parallel region + checksum — exactly as the threads version times its computation. Generation
  and dump are excluded. The output tag is `SPARSE_ITERATION_OMP_TASKS`, and it prints the same
  `rayleigh=`, `checksum=0x…`, `Time (sec) =` lines the analyzer greps, so `analyze_all.sh`
  parses it unchanged.

### 2.9 Optional variant: OpenMP **work-sharing** — `iterative_SpMV_omp_ws.cpp`

The assignment permits *"an additional OpenMP implementation based on **work-sharing constructs**"*
as a comparison point. `src/openmp/iterative_SpMV_omp_ws.cpp` is that variant: it is a **byte-for-byte
clone** of the task version above (same serial init, ping-pong buffers, local shift, fused SpMV+norm,
diagnostics, CLI, timing, output) with **only the engine swapped** — `#pragma omp for` worksharing
loops instead of `taskloop` tasks. It exists *only* to contrast the two work-distribution mechanisms;
the task-based version remains the required deliverable.

What changes, and why it is the interesting contrast:

- **`#pragma omp for` instead of `taskloop`.** The same persistent `#pragma omp parallel` wraps the
  loop (fair comparison), but each iteration's two phases are worksharing loops the *whole team*
  executes, not a `single`-generated task set. There is therefore **no per-iteration task generation**
  — the team simply divides the loop iteration space. Phase A is `#pragma omp for reduction(+: ss)`
  (fused shifted SpMV + L2 sum-of-squares); Phase B is a plain `#pragma omp for` (scaling). The
  **implicit barrier** at the end of each worksharing loop replaces the taskgroup/`single` barrier.
- **Granularity knob = the `schedule` clause, not `grainsize`.** The loops use `schedule(runtime)`, so
  the policy is chosen at run time via **`OMP_SCHEDULE`** (`static`, `dynamic[,chunk]`, `guided`) with
  no recompile — the worksharing analogue of §2.5's `GRAIN`. The binary prints the resolved
  schedule (`omp_get_schedule`) so each run is self-documenting. This is the axis the report's
  granularity analysis sweeps.
- **Granularity finding (the headline result).** The first sweep made the worksharing version look
  hopeless — every schedule capped ~2.5× and `dynamic` ran *worse than serial* with non-monotone
  `t=16 < t=8` inversions. This is **not** a mechanism defect; it is a **granularity trap**. Bare
  `OMP_SCHEDULE=dynamic` resolves to **chunk = 1**: the runtime hands out one row per dispatch, so each
  phase does ~`n` dynamic dispatches per iteration and the shared loop counter is hammered across both
  sockets — pathological. The fair comparison is at *matched* granularity: the task version's
  `taskloop grainsize(n/8P)` (§2.5) is effectively `schedule(dynamic, ~n/8P)`, so we add a
  **`dynamic,auto`** point that the sweep expands per `(n,t)` to `dynamic,⌊n/(8t)⌋` — exactly the task
  grainsize. With that chunk, **worksharing `dynamic` reproduces the task version (~8–10×)**, proving
  the bottleneck was granularity, not "tasks vs worksharing." The full curve brackets it: bare
  `dynamic` (chunk 1) catastrophic → `dynamic,auto`/`dynamic,2048` ≈ tasks → `dynamic,16384` slightly
  worse at high `t` (too coarse) → `static`/`guided` ≈ 2.5× (contiguous chunks meet the heavy-rows-first
  layout). The sweep keeps bare `dynamic` *on purpose* as the failure-end data point.
- **Confirmed on the cluster (node01, 32 cpus, medians of 3).** `ws_dynamic_auto` at t=32 reaches
  **9.7× / 10.7× / 8.4× / 10.5×** on the four matrices (100k/4M, 100k/20M, 500k/4M, 500k/20M) — it
  *matches and slightly beats* `omp_tasks` (7.3× / 9.9× / 7.5× / 10.0×) at t=16/32, because `omp for`
  with a tuned chunk has lower per-dispatch cost than `taskloop`'s per-iteration task creation. The
  trap is worst on the **sparse** matrix (500k/4M, 8 nnz/row): bare `dynamic` runs **0.27–0.48×**
  (slower than serial, with the t=16<t=8 inversion), the densest dispatch-overhead regime. **A fixed
  chunk is not enough** — it must scale with `(n,t)`: `dynamic,2048` collapses to 2.97× and
  `dynamic,16384` to 1.01× at n=100k/t=32 (too few chunks for 32 threads → imbalance/idle), whereas
  `dynamic,auto` keeps ~8 chunks/thread and holds ~9.7×. This `n/(8t)` adaptivity is the whole reason
  the task `grainsize` heuristic works, and the worksharing sweep reproduces it exactly.
- **Static is row-count balanced, *not* nnz-balanced.** This is the crux of the comparison:
  `schedule(static)` hands each thread an equal number of *rows*, but on the irregular matrix the
  heavy rows cluster at the top (`§0.2`), so static is **imbalanced** — the heavy block lands on a few
  threads. `dynamic`/`guided` recover balance by handing out chunks on demand (like the taskloop), at
  the cost of scheduling overhead that grows as the chunk shrinks. So the expected ordering on the
  irregular workload is *static (imbalanced) < well-tuned dynamic/guided ≈ tasks*. (The threads
  version, §1.3, instead pays a one-time **nnz-balanced static** partition and avoids both the
  imbalance *and* the per-iteration scheduling cost — which is why it is usually the fastest.)
- **Reduction scoping detail.** Because a worksharing-loop reduction requires the list item to be
  **shared** in the enclosing parallel, `ss`/`dot_xy` are declared outside the region and listed in
  `shared(...)`; a `#pragma omp single { ss = 0.0; }` resets the accumulator before each Phase A (a
  `+` reduction *adds into* the variable's current value). `default(none)` sits on the `parallel`
  directive (the `for` construct does not accept it), keeping the same data-sharing discipline as §2.6.
- **Same everything else.** Tag `SPARSE_ITERATION_OMP_WORKSHARING`; identical `rayleigh=`/`checksum=`/
  `Time (sec) =` output (so `analyze_all.sh` parses it unchanged); correctness within tolerance
  (verified max |seq − ws| ≈ 1.1e-16 across `static`/`dynamic`/`guided` × t∈{1,2,4,8}). Built with
  `make omp_ws`; compared against the task version by `scripts/run_omp_ws.sbatch`, whose default
  `SCHEDULES` span the granularity axis (one version tag per schedule: `omp`, `ws_static`, `ws_guided`,
  `ws_dynamic`, `ws_dynamic_2048`, `ws_dynamic_16384`, `ws_dynamic_auto`), summarized side by side by
  `./scripts/analyze_all.sh results_omp_ws`.

---

## 3. Variant: dynamic scheduling via the course thread pool — `iterative_SpMV_pool.cpp`

This is a **comparison experiment**, not a fourth required deliverable. The main threads version
(§1) balances work **statically**: it computes one nnz-balanced partition and each worker owns a
fixed range for all 500 iterations. This variant keeps the *threads* model but balances
**dynamically**, the way the OpenMP version does — except the dynamic engine is the professor's
thread pool (`Code/spmcode7`, vendored unchanged into `include/threadPool.hpp`,
`taskFactory.hpp`, `Affinity.hpp`). The goal is a clean three-way contrast: **static-partition
threads (§1) vs dynamic-pool threads (this) vs dynamic OpenMP tasks (§2)**, isolating the cost
and benefit of dynamic scheduling within the C++-threads world.

### 3.1 What the pool actually is (and is not)

It is a **shared-FIFO-queue** pool: a single `std::deque<Task>` guarded by **one mutex**, with
workers (`std::jthread`) that block on a condition variable and pop the queue's front. `submit`
packages any callable + args into a uniform `void()` task **plus a `std::future`** (via
`make_task`, which wraps a `std::packaged_task` — heap-allocated behind a `shared_ptr` on C++20
because `std::function` needs a copyable target), pushes it, and wakes one worker. `inflight_`
counts outstanding tasks; `wait_completion()` blocks until it reaches zero.

It is **not** a work-stealing scheduler — there are no per-worker deques and no victim selection.
The dynamic load balancing comes purely from the shared queue (an idle worker grabs whatever is
next). The pool's `wait_future()` adds **cooperative helping** (a thread waiting on a future
executes pending tasks itself, which prevents deadlock when tasks recursively spawn sub-tasks);
this kernel doesn't spawn nested tasks, so we don't need it.

### 3.2 How this source uses it

Each iteration mirrors §2's two-phase shape, but driven by `submit` + `wait_completion`:

```cpp
threadPool pool(num_threads, affinity::get_affinity_from_env());   // P workers, built once
const std::size_t nchunks = (n + grain - 1) / grain;               // grain = resolve_grain(n,P)
std::vector<PaddedDouble> partial(nchunks);

for (iter …) {
    for (c in [0,nchunks))  pool.submit(/* shifted SpMV of chunk c, part[c].v = Σ sum² */);
    pool.wait_completion();                                        // per-phase barrier
    double inv = 1.0 / std::sqrt( Σ part[c].v );                   // manual reduction
    for (c in [0,nchunks))  pool.submit(/* scale chunk c by inv */);
    pool.wait_completion();
}
```

Key design points, and how they map to §1/§2:

- **Chunks are the unit of scheduling.** The row range `[0, n)` is sliced into `nchunks ≈ 8·P`
  fixed chunks (same `resolve_grain`/`GRAIN` policy as §2.5); the pool decides *which worker* runs
  *which* chunk, so balancing is dynamic even though the chunk boundaries are static.
- **Coordinator model.** The main thread submits a phase's tasks then `wait_completion()`s
  (sleeping on `cv_wait_` until idle); the `P` pool workers do the compute. So `-t P` ≈ P active
  compute threads with the main thread idle during the waits. This is the idiomatic use of *this*
  pool and the fair "what the pool gives you" measurement. (The §1 version, by contrast, has its
  spawning thread *also* be a worker.)
- **Manual reduction.** The pool offers only generic `void()` tasks, so the L2 sum-of-squares is
  reduced by hand from per-chunk `alignas(64) PaddedDouble` slots (the same false-sharing padding
  as §1.5) — a deliberate contrast with OpenMP's `taskloop reduction`. The futures returned by
  `submit` are discarded; correctness rides on `wait_completion`.
- **No races.** Disjoint source-row chunks + the `(src+shift)%n` bijection (§0.2) mean every `yw`
  entry and every `part[c]` slot is written by exactly one task.
- **Optimization parity with `cde`.** The task lambdas carry Opt **D** — the captured CSR/buffer
  pointers are aliased to local `__restrict__` pointers inside each lambda (you can't qualify a
  capture). Together with **A** (padding) and **C** (affinity, via the pool ctor) and the
  coordinator's single norm reduction (**E**-equivalent), the pool matches `cde`; **B**
  (`std::barrier`) is the one that doesn't apply, since the pool brings its own queue barrier.
- **Same everything else.** Identical init, ping-pong buffers, diagnostics, CLI, and timed-region
  rule (pool construction/teardown is inside the timer, the analogue of §1's spawn/join). Tag
  line `SPARSE_ITERATION_CPP_POOL`; affinity via the same `AFFINITY` env convention (passed to the
  pool constructor, which pins worker *i* to the *i*-th listed CPU). Built/run by
  `make pool` and `scripts/run_pool.sbatch` (`VTAGS="pool"`, parsed by `analyze_all.sh`).

### 3.3 What we expect to learn

The interesting quantity is the **per-iteration scheduling overhead**: a dynamic scheme re-submits
`~2·nchunks` tasks on every one of the 500 iterations, and each `submit` takes the single global
mutex, notifies a worker, and (on C++20) heap-allocates a `packaged_task`/future. So the
expectation is **regime-dependent**, and that is precisely the lesson:

- When **tasks are small relative to that overhead** — small `n`, or high `P` with a fine grain —
  the pool can be *much* slower than the static partition, and can even get *worse* with more
  threads as contention on the one queue mutex grows (visible immediately on tiny smoke inputs).
- When **tasks are large** — realistic `n` with a coarse-enough grain — the per-task work
  amortizes the submit cost and the pool becomes competitive with the static version.

The static partition (§1) sidesteps all of this by paying its coordination cost **once** (the
partition) and then only crossing two barriers per iteration. The cluster sweep
(`run_pool.sbatch` vs `run_omp.sbatch` vs `run_all.sbatch`) quantifies where each regime falls;
the headline finding is *why* static partitioning is the right default for this memory-bound,
already-balanced kernel, with the dynamic pool as the controlled counter-experiment.

That "memory-bound" claim is **measured, not assumed**: an empirical **roofline** of the shared
kernels (`docs/ROOFLINE.md`, tool `bin/roofline`) places the dominant SpMV kernel (K1) against the
node's STREAM bandwidth ceiling, and finds it at **~60–90% of the node's ~49 GB/s** at 32 threads
(node bandwidth saturates ~5× over one thread; the node is ~16 physical cores + SMT). So the
falling efficiency at high thread counts is the memory wall, not a coding defect — which is
exactly why dynamic vs static scheduling only shuffles a few percent rather than changing the
asymptote.

---

## 4. Side-by-side comparison

| Design decision | C++ threads (static) | C++ threads + pool (§3) | OpenMP tasks |
|---|---|---|---|
| Team creation | `std::thread` ×P, spawned once (lines 410–422) | `threadPool` of P workers, built once | one `#pragma omp parallel` region (line 174) |
| Work split | manual **static** nnz-balanced partition (§1.3) | **dynamic** chunks via `submit` (§3.2) | **dynamic** `grainsize` tasks (§2.5) |
| Load balancing | computed once, balanced by construction | shared-queue: idle worker takes next chunk | runtime spreads ≈ 8·P tasks across the team |
| Reduction | hand-padded partials + `noexcept` completion fn | hand-padded partials, summed after `wait_completion` | `taskloop reduction(+: …)` (runtime-managed) |
| False sharing | explicit `alignas(64) PaddedDouble` (§1.5) | explicit `alignas(64) PaddedDouble` (§3.2) | not needed — runtime uses private partials |
| Synchronization | two `std::barrier` points / iter (§1.4) | two `wait_completion()` / iter (§3.2) | implicit taskgroup + `single`'s implicit barrier |
| Affinity | in-code `pthread_setaffinity_np`, `AFFINITY` env list | pool pins workers from the `AFFINITY` env list | `OMP_PROC_BIND` / `OMP_PLACES` env |
| Per-iteration cost | 2 barriers (no allocation) | `2·nchunks` submits (mutex + alloc + notify) | task spawn/join, runtime-optimized |
| Runtime tuning knob | `AFFINITY` (placement) | `AFFINITY` + `GRAIN` (granularity) | `GRAIN` (task granularity) |
| Concurrency code volume | substantial (barrier, padding, pinning, parsing) | moderate (`submit` lambdas; pool hides threads) | a handful of pragmas |

The contrast that matters: the static version pays its balancing cost **once**; the pool and
OpenMP pay a **per-iteration** scheduling cost in exchange for dynamic balancing. On this kernel
the work is already balanced (§0.2), so the static version's "pay once" usually wins — the pool
column exists to *measure* that trade rather than assume it. OpenMP, meanwhile, expresses the same
dynamic structure as the pool in far less code by delegating privatization, reduction, and
balancing to the runtime.

The optional **work-sharing** OpenMP variant (§2.9) adds a fourth point on this same axis: it keeps the
OpenMP engine but replaces `taskloop` with `#pragma omp for schedule(runtime)`, so its work
distribution is whatever `OMP_SCHEDULE` selects. It slots in as `schedule(static)` ≈ a *row-count*
static partition (imbalanced here, the cautionary case), and `schedule(dynamic|guided)` ≈ the
taskloop's dynamic balancing without the task-generation step — letting the report separate "dynamic
vs static balancing" from "tasks vs worksharing as the mechanism," holding everything else fixed.

**Shared invariants** (why all three are comparable at all):

- identical serial initialization → the **same `x₀`** bit-for-bit;
- the same locally-derived logical `shift` each iteration → the **same matrix evolution**;
- identical CLI, output tag lines, and timed-region boundaries (init + compute + diagnostics
  timed; generation and dump excluded);
- correctness judged by **tolerance and the Rayleigh value**, not by the bitwise `checksum` —
  the parallel reductions sum in a different order, so low bits (and the checksum) legitimately
  differ. See `DOCUMENTATION.md §6`.

---

## 5. Hybrid MPI + OpenMP — see `docs/MPI_IMPLEMENTATION.md`

The third deliverable adds **distributed memory**: MPI spreads the matrix across processes (ranks,
possibly on different nodes) and OpenMP parallelizes each rank's share across its cores. It comes in
two forms — `src/mpi/iterative_SpMV_mpi.cpp` (blocking `Allgatherv` baseline) and
`src/mpi/iterative_SpMV_mpi_overlap.cpp` (non-blocking `Iallgatherv` with communication/computation
overlap). Because that deliverable has enough moving parts (the rank-vs-thread geometry, the
`Scatterv` distribution, the shift-as-rotation, the bit-for-bit correctness property, and the
non-blocking pipeline), it has its **own dedicated deep-dive: `docs/MPI_IMPLEMENTATION.md`** — read it
there rather than duplicating it here.

In one paragraph: the **matrix is distributed by source rows** (nnz-balanced, fixed for the whole run —
the evolution is a *vector* rotation, so no matrix data moves) while the **dense vector `x` is
replicated** (it is O(n) and the irregular gather needs all of it). Rank 0 generates + `MPI_Scatterv`
distributes the CSR (untimed). Each iteration does a local OpenMP SpMV, **one `Allgatherv`** to rebuild
the full result vector, a local rotation for the shift, and a **canonical-order** norm — which makes
the result **bit-for-bit identical to the sequential** (checksum *matches*, unlike threads/omp). The
overlap variant pipelines the SpMV against a chunked **`MPI_Iallgatherv`** to hide communication.
Strong scaling is bounded by that per-iteration O(n) `Allgatherv` (grows with node count) on top of the
memory-bandwidth ceiling from the roofline study.

## 6. Course grounding

Every technique traces to the SPM notes or the course `Code/` examples cited in each source
file's header:

| Technique | Origin |
|---|---|
| Reusable barrier / completion function (threads) | `Code/spmcode6/spinbarrier-wait.cpp`; SPM NOTES p.83 |
| Thread pinning / affinity (threads) | `Code/spmcode7`; SPM NOTES p.95, p.130 |
| Thread pool (`submit`/`wait_completion`/futures) — pool variant | `Code/spmcode7/include/{threadPool,taskFactory,Affinity}.hpp`, `Code/spmcode7/mergesort-pool.cpp` |
| `parallel → single → taskloop` (OpenMP) | `Code/spmcode9/omp_taskloop.cpp` |
| `taskloop reduction` for a dot product (OpenMP) | `Code/spmcode9/omp_dotprod_taskloop.cpp` |
| `taskgroup` + `task_reduction` (the lower-level alternative) | `Code/spmcode9/omp_task_reduction.cpp` |
| `#pragma omp for` worksharing + `schedule(runtime)`/`OMP_SCHEDULE` + `reduction` (worksharing variant) | SPM NOTES OpenMP worksharing/`schedule`; `Code/spmcode8/jacobi-omp/jacobi-omp.cpp` (single-region `omp for`) |
| OpenMP affinity via environment | `Code/spmcode8/omp_affinity.cpp` |
| Single-region iterative loop shape | `Code/spmcode8/jacobi-omp/jacobi-omp.cpp` |
| Partitioned matrix + `Allgatherv`/`Allreduce` (MPI) | `Code/spmcode11/mpi_power_iteration_partitioned.cpp` |
| `MPI_Scatterv` block distribution (MPI) | `Code/spmcode11/mpi_vectorsumv.cpp`, `Code/spmcode12/mpi_jacobi_1d_blk.cpp` |
| Hybrid `MPI_Init_thread(FUNNELED)` + OpenMP loop (MPI+OpenMP) | `Code/spmcode12/mpi_trapezoid+omp.cpp`, `mpi_summa+omp.cpp` |
| Non-blocking collectives `MPI_Iallgatherv` + `MPI_Wait/Testall` (overlap) | SPM NOTES p.138–139, p.141 ("All collectives have a non-blocking version") |
| Comm/compute overlap & double buffering (overlap) | SPM NOTES p.140 (async farm skeleton) |

This keeps all three implementations defensible against the project constraint that every
technique come from the course material.
