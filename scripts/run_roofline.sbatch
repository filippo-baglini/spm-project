#!/bin/bash
#SBATCH --job-name=spmv_roofline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results_roofline_%j.log
#SBATCH --error=errors_roofline_%j.log

# =============================================================================
# Empirical roofline characterization of the SpMV kernels on one compute node.
#
# For each (n, nz) on the usual grid it runs bin/roofline at 1 thread and at the
# full node, so we see:
#   * peak compute (register FMA microbench) and peak bandwidth (STREAM triad),
#     at 1 and P threads (the latter is the all-NUMA-node aggregate);
#   * the achieved (arithmetic intensity, GFLOP/s, GB/s, %roofline) of the two
#     shared kernels K1 (fused SpMV+norm) and K2 (scaling pass).
#
# This is light (microbenches + a few timed kernels), so it stays well under the
# 30:00 wall. All output is captured to results_roofline/roofline.txt; paste the
# numbers into docs/ROOFLINE.md and write the verdict (memory-bound vs headroom).
#
# Matrix generation is NOT timed (consistent with the rest of the project).
# =============================================================================

# Run from the project root so src/... paths and the output dir resolve.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

RESULTS_DIR="${RESULTS_DIR:-results_roofline}"
MATRIX_SIZES="${MATRIX_SIZES:-100000 500000}"
NONZEROS="${NONZEROS:-4000000 20000000}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"
REPS="${REPS:-50}"                       # kernel repetitions per measurement
SRC="${SRC:-src/bench/roofline_bench.cpp}"

BASE_CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra"

# Thread affinity: compact placement so STREAM/kernels see the real memory system.
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

BUILD="$RESULTS_DIR/build"
mkdir -p "$BUILD"
OUT="$RESULTS_DIR/roofline.txt"

NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
# Thread counts to characterize: 1 (single-core baseline) and the full node.
THREAD_COUNTS="${THREAD_COUNTS:-1 $NCPU}"

echo "=== Building roofline tool (host: $(hostname), cpus: $NCPU) ==="
if [ ! -f "$SRC" ]; then echo "ERROR: source not found: $SRC" >&2; exit 1; fi
g++ $BASE_CXXFLAGS -I include -pthread -fopenmp "$SRC" -o "$BUILD/roofline" \
    || { echo "build failed: $SRC" >&2; exit 1; }
ROOF="$BUILD/roofline"

{
    echo "############################################################################"
    echo "# Empirical roofline   host=$(hostname)  cpus=$NCPU  date=$(date -Is)"
    echo "# mode=$MODE seed=$SEED kernel_reps=$REPS"
    echo "# OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
    echo "############################################################################"
} | tee "$OUT"

for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    for t in $THREAD_COUNTS; do
      [ "$t" -gt "$NCPU" ] && continue
      {
        echo ""
        echo "============================================================================"
        echo ">>> n=$n nz=$nz threads=$t"
        echo "============================================================================"
        "$ROOF" -n "$n" -nz "$nz" -m "$MODE" -s "$SEED" -t "$t" -R "$REPS"
      } | tee -a "$OUT"
    done
  done
done

# Cache-hierarchy bandwidth sweep (for the cache-aware/hierarchical roofline):
# once at 1 thread (clean per-core L1/L2/L3/DRAM plateaus) and once at the full
# node (aggregate plateaus = the ceilings for the 32-thread hierarchical plot).
SWEEP_N="${SWEEP_N:-500000}"
SWEEP_NZ="${SWEEP_NZ:-20000000}"
for t in 1 "$NCPU"; do
  {
    echo ""
    echo "============================================================================"
    echo ">>> CACHE SWEEP  n=$SWEEP_N nz=$SWEEP_NZ threads=$t"
    echo "============================================================================"
    "$ROOF" --cache-sweep -n "$SWEEP_N" -nz "$SWEEP_NZ" -m "$MODE" -s "$SEED" -t "$t" -R "$REPS"
  } | tee -a "$OUT"
done

echo ""
echo "=== Roofline sweep done. Full output in $OUT ==="
