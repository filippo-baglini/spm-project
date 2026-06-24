#!/bin/bash
#SBATCH --job-name=spmv_pool
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results_pool_%j.log
#SBATCH --error=errors_pool_%j.log

# =============================================================================
# Benchmark ONLY the threadpool comparison variant against the sequential
# reference, on the same grid as run_all.sbatch / run_omp.sbatch.
#
# Versions measured:
#   seq  : src/sequential/iterative_SpMV.cpp        (sequential reference)
#   pool : src/threads/iterative_SpMV_pool.cpp        (dynamic scheduling via the
#                                                      course thread pool)
#
# Output is written in the SAME layout the others use, so the existing analyzer
# works unchanged:   ./scripts/analyze_all.sh results_pool
#
# This is the "dynamic-pool threads" point of the static-vs-pool-vs-OpenMP
# comparison. Pair it with results from run_omp.sbatch and run_all.sbatch.
#
# Staying under the 30:00 wall clock of partition 'normal':
#   * Only 2 binaries are measured (seq + pool), so this is light.
#   * RESUMABLE: each (n,nz[,t]) cell is skipped if already complete on disk.
#   * SOFT TIME BUDGET: before each cell it checks elapsed wall time against
#     TIME_BUDGET_SEC (default 1500s = 25 min) and stops CLEANLY if exceeded.
#
# Matrix generation is NOT timed (only the iterative computation is).
# =============================================================================

# Run from the project root so src/... paths and the output dir resolve.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# All knobs overridable from the environment (e.g. a quick smoke test:
# REPS=1 MATRIX_SIZES=2000 NONZEROS=8000 THREAD_COUNTS="1 2 4" sbatch ...).
RESULTS_DIR="${RESULTS_DIR:-results_pool}"
REPS="${REPS:-3}"                        # repetitions per cell (median of these)
MATRIX_SIZES="${MATRIX_SIZES:-100000 500000}"
NONZEROS="${NONZEROS:-4000000 20000000}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8 16 32}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"
TIME_BUDGET_SEC="${TIME_BUDGET_SEC:-1500}"   # stop starting new cells past this

SEQ_SRC="${SEQ_SRC:-src/sequential/iterative_SpMV.cpp}"
POOL_SRC="${POOL_SRC:-src/threads/iterative_SpMV_pool.cpp}"

BASE_CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra"

BUILD="$RESULTS_DIR/build"
RAW="$RESULTS_DIR/raw"
mkdir -p "$BUILD" "$RAW"

# Cap thread counts to the CPUs SLURM actually granted (avoid oversubscription).
NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
CAPPED=""
for t in $THREAD_COUNTS; do
    if [ "$t" -le "$NCPU" ]; then CAPPED="$CAPPED $t"; fi
done
THREAD_COUNTS="${CAPPED# }"

# Thread affinity for the pool: the AFFINITY env convention the pool understands.
# A compact list 0..NCPU-1 pins worker i to CPU i (the analogue of the threads
# version's compact default). Overridable.
export AFFINITY="${AFFINITY:-0-$((NCPU - 1))}"

# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
echo "=== Building binaries (host: $(hostname), cpus: $NCPU) ==="
for src in "$SEQ_SRC" "$POOL_SRC"; do
    if [ ! -f "$src" ]; then echo "ERROR: source not found: $src" >&2; exit 1; fi
done
echo "  seq  <- $SEQ_SRC"
g++ $BASE_CXXFLAGS -I include "$SEQ_SRC" -o "$BUILD/seq" \
    || { echo "build failed: $SEQ_SRC" >&2; exit 1; }
echo "  pool <- $POOL_SRC"
g++ $BASE_CXXFLAGS -I include -pthread "$POOL_SRC" -o "$BUILD/pool" \
    || { echo "build failed: $POOL_SRC" >&2; exit 1; }

SEQ="$BUILD/seq"
POOL="$BUILD/pool"

echo ""
echo "=== Configuration ==="
echo "sizes=$MATRIX_SIZES  nonzeros=$NONZEROS  threads=$THREAD_COUNTS"
echo "mode=$MODE  seed=$SEED  reps=$REPS  output=$RESULTS_DIR"
echo "AFFINITY=$AFFINITY  budget=${TIME_BUDGET_SEC}s"
echo ""

# Sorted-unique numeric union of two space-separated lists (for resumable config).
merge_list() { echo "$1 $2" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }

# Union this run's grid with any saved one so the analyzer sees the full sweep.
ALL_SIZES="$MATRIX_SIZES"; ALL_NZ="$NONZEROS"; ALL_T="$THREAD_COUNTS"
if [ -f "$RESULTS_DIR/config.env" ]; then
    OLD_SIZES=$(sed -n 's/^MATRIX_SIZES="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    OLD_NZ=$(sed -n 's/^NONZEROS="\(.*\)"/\1/p'     "$RESULTS_DIR/config.env")
    OLD_T=$(sed -n 's/^THREAD_COUNTS="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    ALL_SIZES=$(merge_list "$OLD_SIZES" "$MATRIX_SIZES")
    ALL_NZ=$(merge_list "$OLD_NZ" "$NONZEROS")
    ALL_T=$(merge_list "$OLD_T" "$THREAD_COUNTS")
fi

# Record the (cumulative) grid so the analyzer needs no arguments. VTAGS/VNAMES
# match what analyze_all.sh expects; here there is a single measured version.
{
    echo "MATRIX_SIZES=\"$ALL_SIZES\""
    echo "NONZEROS=\"$ALL_NZ\""
    echo "THREAD_COUNTS=\"$ALL_T\""
    echo "VTAGS=\"pool\""
    echo "VNAMES=\"pool_threads\""
    echo "MODE=\"$MODE\""
    echo "SEED=\"$SEED\""
    echo "REPS=\"$REPS\""
} > "$RESULTS_DIR/config.env"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# Run a binary once, echo "<time> <checksum>".
run_one() {
    local out tm cs
    out=$("$@" 2>/dev/null)
    tm=$(echo "$out" | grep "Time (sec)" | awk '{print $NF}')
    cs=$(echo "$out" | grep "checksum=" | head -1 | cut -d= -f2)
    echo "${tm:-NA} ${cs:-NA}"
}

# A cell is complete if its .times file already holds >= REPS non-empty lines.
complete() { [ -f "$1" ] && [ "$(grep -c . "$1")" -ge "$REPS" ]; }

# Soft time-budget guard: returns 0 (true => stop) once elapsed wall time
# ($SECONDS, set by bash) exceeds the budget.
over_budget() { [ "$SECONDS" -ge "$TIME_BUDGET_SEC" ]; }
stop_early() {
    echo ""
    echo "*** TIME BUDGET (${TIME_BUDGET_SEC}s) reached after ${SECONDS}s; stopping cleanly."
    echo "*** Re-submit this script to resume; completed cells are skipped."
    echo "=== Partial results in $RESULTS_DIR. Analyze with: ./scripts/analyze_all.sh $RESULTS_DIR ==="
    exit 0
}

# ----------------------------------------------------------------------------
# Sequential reference (once per (n,nz); REPS reps) — the speedup baseline.
# ----------------------------------------------------------------------------
echo "=== Sequential reference ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    if complete "$RAW/seq_n${n}_nz${nz}.times"; then
      echo "skip (done) seq n=$n nz=$nz"; continue
    fi
    over_budget && stop_early
    : > "$RAW/seq_n${n}_nz${nz}.times"
    for r in $(seq 1 $REPS); do
      read tm cs < <(run_one "$SEQ" -n $n -nz $nz -m $MODE -s $SEED)
      echo "$tm" >> "$RAW/seq_n${n}_nz${nz}.times"
      echo "$cs" >  "$RAW/seq_n${n}_nz${nz}.cksum"
      echo "seq n=$n nz=$nz rep=$r time=$tm"
    done
  done
done

# ----------------------------------------------------------------------------
# Threadpool variant, over the thread grid.
# ----------------------------------------------------------------------------
echo ""
echo "=== Threadpool variant ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    for t in $THREAD_COUNTS; do
      if complete "$RAW/pool_n${n}_nz${nz}_t${t}.times"; then
        echo "skip (done) pool n=$n nz=$nz t=$t"; continue
      fi
      over_budget && stop_early
      : > "$RAW/pool_n${n}_nz${nz}_t${t}.times"
      for r in $(seq 1 $REPS); do
        read tm cs < <(run_one "$POOL" -n $n -nz $nz -m $MODE -s $SEED -t $t)
        echo "$tm" >> "$RAW/pool_n${n}_nz${nz}_t${t}.times"
        echo "$cs" >  "$RAW/pool_n${n}_nz${nz}_t${t}.cksum"
        echo "pool n=$n nz=$nz t=$t rep=$r time=$tm"
      done
    done
  done
done

echo ""
echo "=== Sweep done in ${SECONDS}s. Analyze with: ./scripts/analyze_all.sh $RESULTS_DIR ==="
