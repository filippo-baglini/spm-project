#!/bin/bash
#SBATCH --job-name=spmv_mpi_quick
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=results_mpi_quick_%j.log
#SBATCH --error=errors_mpi_quick_%j.log

# =============================================================================
# QUICK MPI+OpenMP check -- a handful of geometries, NO sequential baseline,
# small footprint. Use this to sanity-check the hybrid build/timings without
# tying up all 8 nodes for 30 min. (For the full strong-scaling study use
# scripts/submit_mpi_sweep.sh.)
#
#   * Allocates at most --nodes nodes (default 2) for <= 10 min.
#   * Skips the sequential reference entirely (its time is already measured and
#     stable -- e.g. in results_numa/). compare_mpi.sh pools that seq baseline.
#   * Runs only ONE matrix size by default and a few process geometries.
#
# Submit:
#   sbatch scripts/run_mpi_quick.sbatch                 # 2 nodes, default configs
#   sbatch --nodes=1 scripts/run_mpi_quick.sbatch       # single node only
#   CONFIGS="4:4:4 8:4:4" MATRIX_SIZES=100000 NONZEROS=4000000 \
#       sbatch scripts/run_mpi_quick.sbatch
#
# Compare (pull the already-measured sequential time from the NUMA run):
#   DIR=results_mpi SEQ_DIRS="results_mpi results_numa" ./scripts/compare_mpi.sh
# =============================================================================

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

RESULTS_DIR="${RESULTS_DIR:-results_mpi}"      # shares the dataset compare_mpi.sh reads
REPS="${REPS:-2}"
MATRIX_SIZES="${MATRIX_SIZES:-100000}"
NONZEROS="${NONZEROS:-4000000}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"

# Each config is  TOTAL_RANKS:RANKS_PER_NODE:THREADS_PER_RANK
# Defaults probe 1-node vs 2-node at a hybrid (t=4) and a flat (t=1) split.
CONFIGS="${CONFIGS:-4:4:4 8:4:4 16:16:1 32:16:1}"

MPI_SRC="${MPI_SRC:-src/mpi/iterative_SpMV_mpi.cpp}"
MPICXX="${MPICXX:-mpicxx}"
BASE_CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra"

BUILD="$RESULTS_DIR/build"; RAW="$RESULTS_DIR/raw"; mkdir -p "$BUILD" "$RAW"
ALLOC_NODES="${SLURM_NNODES:-1}"

export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "=== Building MPI binary only (host: $(hostname), allocated nodes: $ALLOC_NODES) ==="
[ -f "$MPI_SRC" ] || { echo "ERROR: $MPI_SRC not found" >&2; exit 1; }
$MPICXX $BASE_CXXFLAGS -I include -fopenmp "$MPI_SRC" -o "$BUILD/mpi" \
    || { echo "build failed: $MPI_SRC (is mpicxx available?)" >&2; exit 1; }

echo ""
echo "=== Quick configs: $CONFIGS   sizes=$MATRIX_SIZES nonzeros=$NONZEROS reps=$REPS ==="
echo ""

run_one() {   # echo "<time> <checksum>"
    local out tm cs
    out=$("$@" 2>/dev/null)
    tm=$(echo "$out" | grep "Time (sec)" | awk '{print $NF}')
    cs=$(echo "$out" | grep "checksum=" | head -1 | cut -d= -f2)
    echo "${tm:-NA} ${cs:-NA}"
}

# Merge this run's grid + geometries into config.env so compare_mpi.sh works.
merge_list() { echo "$1 $2" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }
GEO_TAGS=""
for cfg in $CONFIGS; do
    RANKS="${cfg%%:*}"; TPR="${cfg##*:}"
    GEO_TAGS="$GEO_TAGS r${RANKS}_t${TPR}"
done
ALL_SIZES="$MATRIX_SIZES"; ALL_NZ="$NONZEROS"; ALL_GEO="$GEO_TAGS"
if [ -f "$RESULTS_DIR/config.env" ]; then
    OLD_SIZES=$(sed -n 's/^MATRIX_SIZES="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    OLD_NZ=$(sed -n 's/^NONZEROS="\(.*\)"/\1/p'     "$RESULTS_DIR/config.env")
    OLD_GEO=$(sed -n 's/^GEOMETRIES="\(.*\)"/\1/p'  "$RESULTS_DIR/config.env")
    ALL_SIZES=$(merge_list "$OLD_SIZES" "$MATRIX_SIZES")
    ALL_NZ=$(merge_list "$OLD_NZ" "$NONZEROS")
    ALL_GEO="$OLD_GEO $GEO_TAGS"
fi
ALL_GEO=$(echo "$ALL_GEO" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ' | sed 's/ *$//')
{
    echo "MATRIX_SIZES=\"$ALL_SIZES\""
    echo "NONZEROS=\"$ALL_NZ\""
    echo "GEOMETRIES=\"$ALL_GEO\""
    echo "MODE=\"$MODE\""
    echo "SEED=\"$SEED\""
    echo "REPS=\"$REPS\""
} > "$RESULTS_DIR/config.env"

for cfg in $CONFIGS; do
    RANKS="${cfg%%:*}"; rest="${cfg#*:}"; RPN="${rest%%:*}"; TPR="${rest##*:}"
    nodes_needed=$(( (RANKS + RPN - 1) / RPN ))
    if [ "$nodes_needed" -gt "$ALLOC_NODES" ]; then
        echo "skip cfg $cfg : needs $nodes_needed nodes but only $ALLOC_NODES allocated"
        continue
    fi
    export OMP_NUM_THREADS="$TPR"
    MAP="--map-by ppr:${RPN}:node:pe=${TPR} --bind-to core"
    GEO="r${RANKS}_t${TPR}"
    echo "--- $GEO : ranks=$RANKS ranks/node=$RPN threads/rank=$TPR nodes=$nodes_needed ---"
    for n in $MATRIX_SIZES; do
      for nz in $NONZEROS; do
        f="$RAW/mpi_n${n}_nz${nz}_${GEO}.times"
        : > "$f"
        for r in $(seq 1 $REPS); do
          read tm cs < <(run_one mpirun -np "$RANKS" $MAP "$BUILD/mpi" -n $n -nz $nz -m $MODE -s $SEED -t $TPR)
          echo "$tm" >> "$f"
          echo "$cs" >  "$RAW/mpi_n${n}_nz${nz}_${GEO}.cksum"
          echo "mpi n=$n nz=$nz $GEO rep=$r time=$tm checksum=$cs"
        done
      done
    done
done

echo ""
echo "=== Quick run done in ${SECONDS}s. Compare with:"
echo "    DIR=$RESULTS_DIR SEQ_DIRS=\"$RESULTS_DIR results_numa\" ./scripts/compare_mpi.sh ==="
