#!/bin/bash
#SBATCH --job-name=spmv_mpi
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results_mpi_%j.log
#SBATCH --error=errors_mpi_%j.log

# =============================================================================
# Hybrid MPI + OpenMP benchmark (deliverable #3) for ONE process geometry.
#
# The geometry is whatever SLURM granted this job:
#   nodes            = $SLURM_NNODES
#   ranks (total)    = $SLURM_NTASKS         (= nodes * ntasks-per-node)
#   threads per rank = $SLURM_CPUS_PER_TASK
#
# Override the geometry at submit time, e.g.:
#   sbatch --nodes=2 --ntasks-per-node=8 --cpus-per-task=2 scripts/run_mpi.sbatch
#
# A full strong-scaling sweep = several submissions with different geometry.
# Use scripts/submit_mpi_sweep.sh to fire the recommended set across 1..8 nodes.
#
# Results go to results_mpi/raw, keyed by (n, nz, total_ranks, threads_per_rank),
# so submissions accumulate into one comparable dataset. A sequential baseline is
# run once per (n,nz) for the speedup denominator. RESUMABLE + time-budget guard.
# Matrix generation AND the Scatterv distribution are NOT timed (only the loop).
# =============================================================================

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

RESULTS_DIR="${RESULTS_DIR:-results_mpi}"
REPS="${REPS:-3}"
MATRIX_SIZES="${MATRIX_SIZES:-100000 500000}"
NONZEROS="${NONZEROS:-4000000 20000000}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"
TIME_BUDGET_SEC="${TIME_BUDGET_SEC:-1500}"

MPI_SRC="${MPI_SRC:-src/mpi/iterative_SpMV_mpi.cpp}"
SEQ_SRC="${SEQ_SRC:-src/sequential/iterative_SpMV.cpp}"
MPICXX="${MPICXX:-mpicxx}"
BASE_CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra"

BUILD="$RESULTS_DIR/build"
RAW="$RESULTS_DIR/raw"
mkdir -p "$BUILD" "$RAW"

NODES="${SLURM_NNODES:-1}"
RANKS="${SLURM_NTASKS:-1}"
RPN="${SLURM_NTASKS_PER_NODE:-$RANKS}"
TPR="${SLURM_CPUS_PER_TASK:-1}"

# Hybrid placement: TPR OpenMP threads per rank, RPN ranks per node, each rank
# bound to its own set of cores. (OpenMPI 4 mapping; see Code/spmcode11 Makefile.)
export OMP_NUM_THREADS="$TPR"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
MAP="${MAP:---map-by ppr:${RPN}:node:pe=${TPR} --bind-to core}"

echo "=== Building (host: $(hostname)) ==="
[ -f "$MPI_SRC" ] || { echo "ERROR: $MPI_SRC not found" >&2; exit 1; }
$MPICXX $BASE_CXXFLAGS -I include -fopenmp "$MPI_SRC" -o "$BUILD/mpi" \
    || { echo "build failed: $MPI_SRC (is mpicxx available?)" >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include "$SEQ_SRC" -o "$BUILD/seq" \
    || { echo "build failed: $SEQ_SRC" >&2; exit 1; }

echo ""
echo "=== Geometry: nodes=$NODES ranks=$RANKS ranks/node=$RPN threads/rank=$TPR (total cores=$((RANKS*TPR))) ==="
echo "MAP=$MAP  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "sizes=$MATRIX_SIZES nonzeros=$NONZEROS mode=$MODE seed=$SEED reps=$REPS"
echo ""

# Record/merge the swept grid + the set of geometries seen, so compare_mpi.sh
# needs no arguments.
merge_list() { echo "$1 $2" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }
ALL_SIZES="$MATRIX_SIZES"; ALL_NZ="$NONZEROS"
GEO="r${RANKS}_t${TPR}"
ALL_GEO="$GEO"
if [ -f "$RESULTS_DIR/config.env" ]; then
    OLD_SIZES=$(sed -n 's/^MATRIX_SIZES="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    OLD_NZ=$(sed -n 's/^NONZEROS="\(.*\)"/\1/p'     "$RESULTS_DIR/config.env")
    OLD_GEO=$(sed -n 's/^GEOMETRIES="\(.*\)"/\1/p'  "$RESULTS_DIR/config.env")
    ALL_SIZES=$(merge_list "$OLD_SIZES" "$MATRIX_SIZES")
    ALL_NZ=$(merge_list "$OLD_NZ" "$NONZEROS")
    ALL_GEO=$(echo "$OLD_GEO $GEO" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ' | sed 's/ *$//')
fi
{
    echo "MATRIX_SIZES=\"$ALL_SIZES\""
    echo "NONZEROS=\"$ALL_NZ\""
    echo "GEOMETRIES=\"$ALL_GEO\""
    echo "MODE=\"$MODE\""
    echo "SEED=\"$SEED\""
    echo "REPS=\"$REPS\""
} > "$RESULTS_DIR/config.env"

run_one() {   # echo "<time> <checksum>"
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
    echo "*** TIME BUDGET reached after ${SECONDS}s; stopping cleanly. Re-submit to resume."
    exit 0
}

# Sequential baseline (once per (n,nz), shared across geometries/submissions).
echo "=== Sequential baseline ==="
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

# Hybrid MPI+OpenMP for THIS geometry.
echo ""
echo "=== MPI+OpenMP  geometry $GEO (ranks=$RANKS, threads/rank=$TPR) ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    f="$RAW/mpi_n${n}_nz${nz}_${GEO}.times"
    if complete "$f"; then echo "skip (done) mpi n=$n nz=$nz $GEO"; continue; fi
    over_budget && stop_early
    : > "$f"
    for r in $(seq 1 $REPS); do
      read tm cs < <(run_one mpirun -np "$RANKS" $MAP "$BUILD/mpi" -n $n -nz $nz -m $MODE -s $SEED -t $TPR)
      echo "$tm" >> "$f"
      echo "$cs" >  "$RAW/mpi_n${n}_nz${nz}_${GEO}.cksum"
      echo "mpi n=$n nz=$nz $GEO rep=$r time=$tm"
    done
  done
done

echo ""
echo "=== Done geometry $GEO in ${SECONDS}s. Compare with: ./scripts/compare_mpi.sh ==="
