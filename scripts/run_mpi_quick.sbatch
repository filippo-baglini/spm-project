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
#   * Times BOTH MPI variants by default (VARIANTS="block overlap"): the blocking
#     MPI_Allgatherv baseline (bin/mpi -> mpi_*) and the non-blocking overlap
#     (bin/mpi_overlap -> mpiov_*), the latter once per -c value in CHUNKS.
#   * Runs only ONE matrix size by default and a few process geometries.
#
# Submit:
#   sbatch scripts/run_mpi_quick.sbatch                 # 2 nodes, default configs
#   sbatch --nodes=1 scripts/run_mpi_quick.sbatch       # single node only
#   VARIANTS="overlap" CHUNKS="1 2 4 8" sbatch scripts/run_mpi_quick.sbatch  # tune -c
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
# RESUMABLE: result cells already holding >=REPS samples are skipped, so a job
# killed at the SLURM time limit (or stopped by the budget below) just resumes on
# resubmit. TIME_BUDGET_SEC stops cleanly before the wall limit; default is
# effectively off so the plain quick run is unchanged (the comprehensive driver
# sets it). 0/empty also disables it.
TIME_BUDGET_SEC="${TIME_BUDGET_SEC:-0}"

# Each config is  TOTAL_RANKS:RANKS_PER_NODE:THREADS_PER_RANK
# Defaults probe 1-node vs 2-node at a hybrid (t=4) and a flat (t=1) split.
CONFIGS="${CONFIGS:-4:4:4 8:4:4 16:16:1 32:16:1}"

# Which MPI variants to time, and (for the overlap one) how many pipeline chunks.
#   block   = blocking MPI_Allgatherv baseline      (bin/mpi)
#   overlap = non-blocking MPI_Iallgatherv pipeline (bin/mpi_overlap), one run per -c value
VARIANTS="${VARIANTS:-block overlap}"
CHUNKS="${CHUNKS:-4}"                   # space-separated list, e.g. "1 2 4 8"

MPI_SRC="${MPI_SRC:-src/mpi/iterative_SpMV_mpi.cpp}"
MPI_OVL_SRC="${MPI_OVL_SRC:-src/mpi/iterative_SpMV_mpi_overlap.cpp}"
MPICXX="${MPICXX:-mpicxx}"
BASE_CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra"

BUILD="$RESULTS_DIR/build"; RAW="$RESULTS_DIR/raw"; mkdir -p "$BUILD" "$RAW"
ALLOC_NODES="${SLURM_NNODES:-1}"

export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "=== Building MPI binaries (host: $(hostname), allocated nodes: $ALLOC_NODES) ==="
case " $VARIANTS " in *" block "*)
    [ -f "$MPI_SRC" ] || { echo "ERROR: $MPI_SRC not found" >&2; exit 1; }
    $MPICXX $BASE_CXXFLAGS -I include -fopenmp "$MPI_SRC" -o "$BUILD/mpi" \
        || { echo "build failed: $MPI_SRC (is mpicxx available?)" >&2; exit 1; }
    echo "built bin: $BUILD/mpi (block)" ;;
esac
case " $VARIANTS " in *" overlap "*)
    [ -f "$MPI_OVL_SRC" ] || { echo "ERROR: $MPI_OVL_SRC not found" >&2; exit 1; }
    $MPICXX $BASE_CXXFLAGS -I include -fopenmp "$MPI_OVL_SRC" -o "$BUILD/mpi_overlap" \
        || { echo "build failed: $MPI_OVL_SRC (is mpicxx available?)" >&2; exit 1; }
    echo "built bin: $BUILD/mpi_overlap (overlap, chunks=$CHUNKS)" ;;
esac

echo ""
echo "=== Quick configs: $CONFIGS   variants=[$VARIANTS] chunks=[$CHUNKS]"
echo "    sizes=$MATRIX_SIZES nonzeros=$NONZEROS reps=$REPS ==="
echo ""

run_one() {   # echo "<time> <checksum>"
    local out tm cs
    out=$("$@" 2>/dev/null)
    tm=$(echo "$out" | grep "Time (sec)" | awk '{print $NF}')
    cs=$(echo "$out" | grep "checksum=" | head -1 | cut -d= -f2)
    echo "${tm:-NA} ${cs:-NA}"
}
# A cell is complete only when it has >=REPS NUMERIC samples; lines like "NA"
# (a failed/empty run) do NOT count, so failures are retried on resume.
complete() { [ -f "$1" ] && [ "$(grep -cE '^[0-9]' "$1")" -ge "$REPS" ]; }
over_budget() { [ "${TIME_BUDGET_SEC:-0}" -gt 0 ] && [ "$SECONDS" -ge "$TIME_BUDGET_SEC" ]; }
stop_early() {
    echo ""
    echo "*** TIME BUDGET (${TIME_BUDGET_SEC}s) reached after ${SECONDS}s; stopping cleanly. Re-submit to resume."
    exit 0
}

# Time one variant for the current (n,nz,GEO): $1=label, rest=binary + its extra
# args. fbase (the result file stem) is taken from $BENCH_FBASE. RESUMABLE +
# budget-guarded: already-complete cells are skipped; stop cleanly over budget.
bench_variant() {   # $1=label  $2=binary  [extra args...]
    local label="$1" bin="$2"; shift 2
    local f="$RAW/${BENCH_FBASE}.times"
    if complete "$f"; then echo "skip (done) $label n=$n nz=$nz $GEO ${*:+$*}"; return; fi
    over_budget && stop_early
    : > "$f"
    for r in $(seq 1 $REPS); do
        read tm cs < <(run_one mpirun -np "$RANKS" $MAP "$bin" -n "$n" -nz "$nz" -m "$MODE" -s "$SEED" -t "$TPR" "$@")
        echo "$tm" >> "$f"
        echo "$cs" >  "$RAW/${BENCH_FBASE}.cksum"
        echo "$label n=$n nz=$nz $GEO ${*:+$* }rep=$r time=$tm checksum=$cs"
    done
}

# Merge this run's grid + geometries into config.env so compare_mpi.sh works.
# Skipped when WRITE_CONFIG=0 (the comprehensive driver writes an authoritative
# config.env itself, so concurrent jobs don't clobber it). Written atomically.
merge_list() { echo "$1 $2" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }
if [ "${WRITE_CONFIG:-1}" = 1 ]; then
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
        echo "VARIANTS=\"$VARIANTS\""
        echo "CHUNKS=\"$CHUNKS\""
    } > "$RESULTS_DIR/config.env.tmp.$$" && mv -f "$RESULTS_DIR/config.env.tmp.$$" "$RESULTS_DIR/config.env"
fi

for cfg in $CONFIGS; do
    RANKS="${cfg%%:*}"; rest="${cfg#*:}"; RPN="${rest%%:*}"; TPR="${rest##*:}"
    nodes_needed=$(( (RANKS + RPN - 1) / RPN ))
    if [ "$nodes_needed" -gt "$ALLOC_NODES" ]; then
        echo "skip cfg $cfg : needs $nodes_needed nodes but only $ALLOC_NODES allocated"
        continue
    fi
    export OMP_NUM_THREADS="$TPR"
    if [ "$RPN" -eq 1 ]; then
        # One rank owns the whole node: binding it to pe=<cores/node> spans BOTH
        # sockets, which OpenMPI core-binding rejects. Leave the rank unbound and
        # let OpenMP (OMP_PROC_BIND/PLACES) pin its threads to cores.
        MAP="--map-by ppr:1:node --bind-to none"
    else
        MAP="--map-by ppr:${RPN}:node:pe=${TPR} --bind-to core"
    fi
    GEO="r${RANKS}_t${TPR}"
    echo "--- $GEO : ranks=$RANKS ranks/node=$RPN threads/rank=$TPR nodes=$nodes_needed ---"
    for n in $MATRIX_SIZES; do
      for nz in $NONZEROS; do
        case " $VARIANTS " in *" block "*)
            BENCH_FBASE="mpi_n${n}_nz${nz}_${GEO}"
            bench_variant "block  " "$BUILD/mpi" ;;
        esac
        case " $VARIANTS " in *" overlap "*)
            for C in $CHUNKS; do
                BENCH_FBASE="mpiov_n${n}_nz${nz}_${GEO}_c${C}"
                bench_variant "overlap" "$BUILD/mpi_overlap" -c "$C"
            done ;;
        esac
      done
    done
done

echo ""
echo "=== Quick run done in ${SECONDS}s. Compare with:"
echo "    DIR=$RESULTS_DIR SEQ_DIRS=\"$RESULTS_DIR results_numa\" ./scripts/compare_mpi.sh ==="
