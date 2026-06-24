#!/bin/bash
#SBATCH --job-name=spmv_numa
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results_numa_%j.log
#SBATCH --error=errors_numa_%j.log

# =============================================================================
# NUMA-locality study across ALL THREE parallel models. For each model we run a
# baseline and a NUMA variant (private CSR replica built by parallel first-touch,
# so the matrix is spread across both sockets instead of the generator's serial
# single-node placement). Versions, all into results_numa/ (analyze_all.sh-able):
#
#   seq        sequential reference                          (speedup baseline)
#   cde        static nnz-partition threads (opts A-E)       baseline
#   cde_numa   + NUMA first-touch replica                    iterative_SpMV_threads_numa
#   pool       dynamic thread-pool                           baseline
#   pool_numa  + NUMA first-touch replica                    iterative_SpMV_pool_numa
#   omp        OpenMP taskloop                               baseline
#   omp_numa   + NUMA first-touch replica                    iterative_SpMV_omp_numa
#
# The NUMA replica is UNTIMED (matrix prep). cde gets near-perfect locality
# (static partition matches placement); pool/omp are dynamically scheduled, so
# they get the cross-socket *bandwidth spread* but not perfect per-worker
# locality -> expect a smaller gain. compare_numa.sh prints the three
# baseline-vs-NUMA pairs.
#
# RESUMABLE + soft time-budget (this is 7 binaries over the full grid, so it will
# very likely need several resubmits; completed cells are skipped).
# =============================================================================

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

RESULTS_DIR="${RESULTS_DIR:-results_numa}"
REPS="${REPS:-3}"
MATRIX_SIZES="${MATRIX_SIZES:-100000 500000}"
NONZEROS="${NONZEROS:-4000000 20000000}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8 16 32}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"
TIME_BUDGET_SEC="${TIME_BUDGET_SEC:-1500}"

SEQ_SRC="${SEQ_SRC:-src/sequential/iterative_SpMV.cpp}"
CDE_SRC="${CDE_SRC:-src/threads/iterative_SpMV_threads.cpp}"
CDE_NUMA_SRC="${CDE_NUMA_SRC:-src/threads/iterative_SpMV_threads_numa.cpp}"
POOL_SRC="${POOL_SRC:-src/threads/iterative_SpMV_pool.cpp}"
POOL_NUMA_SRC="${POOL_NUMA_SRC:-src/threads/iterative_SpMV_pool_numa.cpp}"
OMP_SRC="${OMP_SRC:-src/openmp/iterative_SpMV_omp.cpp}"
OMP_NUMA_SRC="${OMP_NUMA_SRC:-src/openmp/iterative_SpMV_omp_numa.cpp}"

BASE_CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra"

BUILD="$RESULTS_DIR/build"
RAW="$RESULTS_DIR/raw"
mkdir -p "$BUILD" "$RAW"

NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
CAPPED=""
for t in $THREAD_COUNTS; do
    if [ "$t" -le "$NCPU" ]; then CAPPED="$CAPPED $t"; fi
done
THREAD_COUNTS="${CAPPED# }"

# Pinning: AFFINITY (threads/pool) + OMP_PROC_BIND/PLACES (OpenMP). Compact list
# 0..NCPU-1 spans both sockets, which is what makes the first-touch spread work.
export AFFINITY="${AFFINITY:-0-$((NCPU - 1))}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
echo "=== Building binaries (host: $(hostname), cpus: $NCPU) ==="
for src in "$SEQ_SRC" "$CDE_SRC" "$CDE_NUMA_SRC" "$POOL_SRC" "$POOL_NUMA_SRC" "$OMP_SRC" "$OMP_NUMA_SRC"; do
    if [ ! -f "$src" ]; then echo "ERROR: source not found: $src" >&2; exit 1; fi
done
g++ $BASE_CXXFLAGS -I include          "$SEQ_SRC"       -o "$BUILD/seq"       || { echo "build failed: seq"       >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread  "$CDE_SRC"       -o "$BUILD/cde"       || { echo "build failed: cde"       >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread  "$CDE_NUMA_SRC"  -o "$BUILD/cde_numa"  || { echo "build failed: cde_numa"  >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread  "$POOL_SRC"      -o "$BUILD/pool"      || { echo "build failed: pool"      >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread  "$POOL_NUMA_SRC" -o "$BUILD/pool_numa" || { echo "build failed: pool_numa" >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread -fopenmp "$OMP_SRC"      -o "$BUILD/omp"      || { echo "build failed: omp"      >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread -fopenmp "$OMP_NUMA_SRC" -o "$BUILD/omp_numa" || { echo "build failed: omp_numa" >&2; exit 1; }

echo ""
echo "=== Configuration ==="
echo "sizes=$MATRIX_SIZES  nonzeros=$NONZEROS  threads=$THREAD_COUNTS"
echo "mode=$MODE  seed=$SEED  reps=$REPS  output=$RESULTS_DIR"
echo "AFFINITY=$AFFINITY  OMP_PROC_BIND=$OMP_PROC_BIND  OMP_PLACES=$OMP_PLACES  budget=${TIME_BUDGET_SEC}s"
echo ""

merge_list() { echo "$1 $2" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }

ALL_SIZES="$MATRIX_SIZES"; ALL_NZ="$NONZEROS"; ALL_T="$THREAD_COUNTS"
if [ -f "$RESULTS_DIR/config.env" ]; then
    OLD_SIZES=$(sed -n 's/^MATRIX_SIZES="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    OLD_NZ=$(sed -n 's/^NONZEROS="\(.*\)"/\1/p'     "$RESULTS_DIR/config.env")
    OLD_T=$(sed -n 's/^THREAD_COUNTS="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    ALL_SIZES=$(merge_list "$OLD_SIZES" "$MATRIX_SIZES")
    ALL_NZ=$(merge_list "$OLD_NZ" "$NONZEROS")
    ALL_T=$(merge_list "$OLD_T" "$THREAD_COUNTS")
fi

{
    echo "MATRIX_SIZES=\"$ALL_SIZES\""
    echo "NONZEROS=\"$ALL_NZ\""
    echo "THREAD_COUNTS=\"$ALL_T\""
    echo "VTAGS=\"cde cde_numa pool pool_numa omp omp_numa\""
    echo "VNAMES=\"threads_cde threads_numa pool pool_numa omp omp_numa\""
    echo "MODE=\"$MODE\""
    echo "SEED=\"$SEED\""
    echo "REPS=\"$REPS\""
} > "$RESULTS_DIR/config.env"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
run_one() {
    local out tm cs
    out=$("$@" 2>/dev/null)
    tm=$(echo "$out" | grep "Time (sec)" | awk '{print $NF}')
    cs=$(echo "$out" | grep "checksum=" | head -1 | cut -d= -f2)
    echo "${tm:-NA} ${cs:-NA}"
}
complete() { [ -f "$1" ] && [ "$(grep -c . "$1")" -ge "$REPS" ]; }
over_budget() { [ "$SECONDS" -ge "$TIME_BUDGET_SEC" ]; }
stop_early() {
    echo ""
    echo "*** TIME BUDGET (${TIME_BUDGET_SEC}s) reached after ${SECONDS}s; stopping cleanly."
    echo "*** Re-submit to resume; completed cells are skipped."
    echo "=== Partial results in $RESULTS_DIR. Compare with: ./scripts/compare_numa.sh ==="
    exit 0
}

# Run one threaded version over the thread grid. Args: tag binary
sweep_threaded() {
    local tag="$1"; local bin="$2"
    echo ""
    echo "=== Variant: $tag ==="
    for n in $MATRIX_SIZES; do
      for nz in $NONZEROS; do
        for t in $THREAD_COUNTS; do
          local f="$RAW/${tag}_n${n}_nz${nz}_t${t}.times"
          if complete "$f"; then echo "skip (done) $tag n=$n nz=$nz t=$t"; continue; fi
          over_budget && stop_early
          : > "$f"
          for r in $(seq 1 $REPS); do
            read tm cs < <(run_one "$bin" -n $n -nz $nz -m $MODE -s $SEED -t $t)
            echo "$tm" >> "$f"
            echo "$cs" >  "$RAW/${tag}_n${n}_nz${nz}_t${t}.cksum"
            echo "$tag n=$n nz=$nz t=$t rep=$r time=$tm"
          done
        done
      done
    done
}

# ----------------------------------------------------------------------------
# Sequential reference (REPS reps per (n,nz)) — the speedup baseline.
# ----------------------------------------------------------------------------
echo "=== Sequential reference ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    if complete "$RAW/seq_n${n}_nz${nz}.times"; then echo "skip (done) seq n=$n nz=$nz"; continue; fi
    over_budget && stop_early
    : > "$RAW/seq_n${n}_nz${nz}.times"
    for r in $(seq 1 $REPS); do
      read tm cs < <(run_one "$BUILD/seq" -n $n -nz $nz -m $MODE -s $SEED)
      echo "$tm" >> "$RAW/seq_n${n}_nz${nz}.times"
      echo "$cs" >  "$RAW/seq_n${n}_nz${nz}.cksum"
      echo "seq n=$n nz=$nz rep=$r time=$tm"
    done
  done
done

# Baseline-then-NUMA per model, so a partial run still yields complete pairs.
sweep_threaded "cde"       "$BUILD/cde"
sweep_threaded "cde_numa"  "$BUILD/cde_numa"
sweep_threaded "pool"      "$BUILD/pool"
sweep_threaded "pool_numa" "$BUILD/pool_numa"
sweep_threaded "omp"       "$BUILD/omp"
sweep_threaded "omp_numa"  "$BUILD/omp_numa"

echo ""
echo "=== Sweep done in ${SECONDS}s. Compare with: ./scripts/compare_numa.sh ==="
