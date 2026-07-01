#!/bin/bash
#SBATCH --job-name=spmv_omp_ws
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results_omp_ws_%j.log
#SBATCH --error=errors_omp_ws_%j.log

# =============================================================================
# Compare the OpenMP TASK-based version against the optional OpenMP WORK-SHARING
# version (the "additional comparison point" the assignment allows), on a single
# node, against the sequential baseline.
#
# Versions measured (one raw-file tag each, all on the same thread grid):
#   omp        : src/openmp/iterative_SpMV_omp.cpp     (tasks; taskloop + GRAIN)
#   ws_<sched> : src/openmp/iterative_SpMV_omp_ws.cpp  (#pragma omp for) run once
#                per schedule in $SCHEDULES, via OMP_SCHEDULE=<sched>.
# The work-sharing version's granularity knob is the OpenMP `schedule` clause, so
# sweeping a few schedules (static / dynamic / guided) is the granularity study.
#
# Output is written in the SAME layout run_all.sbatch / run_omp.sbatch use, so the
# existing analyzer prints all versions side by side:
#   ./scripts/analyze_all.sh results_omp_ws
#
# Examples:
#   sbatch scripts/run_omp_ws.sbatch
#   SCHEDULES="static dynamic,1024 dynamic,8192 guided" sbatch scripts/run_omp_ws.sbatch
#   REPS=1 MATRIX_SIZES=2000 NONZEROS=8000 THREAD_COUNTS="1 2 4" sbatch scripts/run_omp_ws.sbatch
#
# RESUMABLE + soft time-budget (like run_omp.sbatch): completed cells are skipped,
# and it stops cleanly past TIME_BUDGET_SEC so a long run can be re-submitted.
# Matrix generation is NOT timed (only the iterative computation is).
# =============================================================================

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

RESULTS_DIR="${RESULTS_DIR:-results_omp_ws}"
REPS="${REPS:-3}"
MATRIX_SIZES="${MATRIX_SIZES:-100000 500000}"
NONZEROS="${NONZEROS:-4000000 20000000}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8 16 32}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"
TIME_BUDGET_SEC="${TIME_BUDGET_SEC:-1500}"

# Work-sharing schedules to compare (each becomes a version tag ws_<sched>, with
# any comma -> '_' for the filesystem-safe tag). These map to OMP_SCHEDULE.
#
# The default spans the granularity axis so the comparison is fair:
#   static / guided     contiguous-chunk imbalance baselines (heavy rows cluster
#                       at the top, so one thread is swamped) -> cap ~2.5x.
#   dynamic             bare dynamic == chunk 1 (the OpenMP default). Kept ON
#                       PURPOSE as the "trap" data point: ~n per-row dispatches
#                       per phase + cross-socket counter contention -> catastrophic.
#   dynamic,2048        moderate chunk (near the sweet spot).
#   dynamic,16384       coarse chunk (slightly imbalanced at high t).
#   dynamic,auto        per-(n,t) chunk = n/(8t) == the task version's grainsize
#                       (resolve_grain in iterative_SpMV_omp.cpp), i.e. the
#                       apples-to-apples "same granularity as tasks" point. It is
#                       expanded to dynamic,<chunk> per cell in the run loop below.
SCHEDULES="${SCHEDULES:-static guided dynamic dynamic,2048 dynamic,16384 dynamic,auto}"

SEQ_SRC="${SEQ_SRC:-src/sequential/iterative_SpMV.cpp}"
OMP_SRC="${OMP_SRC:-src/openmp/iterative_SpMV_omp.cpp}"
WS_SRC="${WS_SRC:-src/openmp/iterative_SpMV_omp_ws.cpp}"

BASE_CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra"

# Thread affinity (env-based, like run_omp.sbatch). NOTE: we deliberately do NOT
# export OMP_SCHEDULE here -- it is set per work-sharing run below, and the task
# version ignores it.
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

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

# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
echo "=== Building binaries (host: $(hostname), cpus: $NCPU) ==="
for src in "$SEQ_SRC" "$OMP_SRC" "$WS_SRC"; do
    if [ ! -f "$src" ]; then echo "ERROR: source not found: $src" >&2; exit 1; fi
done
g++ $BASE_CXXFLAGS -I include "$SEQ_SRC" -o "$BUILD/seq" \
    || { echo "build failed: $SEQ_SRC" >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread -fopenmp "$OMP_SRC" -o "$BUILD/omp" \
    || { echo "build failed: $OMP_SRC" >&2; exit 1; }
g++ $BASE_CXXFLAGS -I include -pthread -fopenmp "$WS_SRC" -o "$BUILD/omp_ws" \
    || { echo "build failed: $WS_SRC" >&2; exit 1; }

SEQ="$BUILD/seq"

# ----------------------------------------------------------------------------
# Version table: parallel arrays (tag, binary, schedule). The task version has an
# empty schedule (OMP_SCHEDULE not applicable); each work-sharing schedule is its
# own version tag.
# ----------------------------------------------------------------------------
declare -a VTAG_ARR=("omp")
declare -a VBIN_ARR=("$BUILD/omp")
declare -a VSCHED_ARR=("")
declare -a VNAME_ARR=("omp_tasks")
for s in $SCHEDULES; do
    safe=$(echo "$s" | tr ',' '_')
    VTAG_ARR+=("ws_${safe}")
    VBIN_ARR+=("$BUILD/omp_ws")
    VSCHED_ARR+=("$s")
    VNAME_ARR+=("ws_${safe}")
done
VTAGS="${VTAG_ARR[*]}"
VNAMES="${VNAME_ARR[*]}"

echo ""
echo "=== Configuration ==="
echo "sizes=$MATRIX_SIZES  nonzeros=$NONZEROS  threads=$THREAD_COUNTS"
echo "versions: seq, ${VNAMES}"
echo "schedules=$SCHEDULES  mode=$MODE  seed=$SEED  reps=$REPS  output=$RESULTS_DIR"
echo "OMP_PROC_BIND=$OMP_PROC_BIND  OMP_PLACES=$OMP_PLACES  budget=${TIME_BUDGET_SEC}s"
echo ""

# Sorted-unique numeric union of two space-separated lists (for resumable config).
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

# Record the (cumulative) grid + version tags so the analyzer needs no arguments.
{
    echo "MATRIX_SIZES=\"$ALL_SIZES\""
    echo "NONZEROS=\"$ALL_NZ\""
    echo "THREAD_COUNTS=\"$ALL_T\""
    echo "VTAGS=\"$VTAGS\""
    echo "VNAMES=\"$VNAMES\""
    echo "MODE=\"$MODE\""
    echo "SEED=\"$SEED\""
    echo "REPS=\"$REPS\""
    echo "SCHEDULES=\"$SCHEDULES\""
} > "$RESULTS_DIR/config.env"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
run_one() {   # echo "<time> <checksum>"
    local out tm cs
    out=$("$@" 2>/dev/null)
    tm=$(echo "$out" | grep "Time (sec)" | awk '{print $NF}')
    cs=$(echo "$out" | grep "checksum=" | head -1 | cut -d= -f2)
    echo "${tm:-NA} ${cs:-NA}"
}
# Complete only with >=REPS NUMERIC samples; "NA" (failed run) lines don't count.
complete() { [ -f "$1" ] && [ "$(grep -cE '^[0-9]' "$1")" -ge "$REPS" ]; }
over_budget() { [ "$SECONDS" -ge "$TIME_BUDGET_SEC" ]; }
stop_early() {
    echo ""
    echo "*** TIME BUDGET (${TIME_BUDGET_SEC}s) reached after ${SECONDS}s; stopping cleanly."
    echo "*** Re-submit to resume; completed cells are skipped."
    echo "=== Partial results in $RESULTS_DIR. Analyze: ./scripts/analyze_all.sh $RESULTS_DIR ==="
    exit 0
}

# ----------------------------------------------------------------------------
# Sequential reference (once per (n,nz)) — the speedup baseline.
# ----------------------------------------------------------------------------
echo "=== Sequential reference ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    if complete "$RAW/seq_n${n}_nz${nz}.times"; then echo "skip (done) seq n=$n nz=$nz"; continue; fi
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
# Task version + work-sharing schedules, over the thread grid.
# ----------------------------------------------------------------------------
echo ""
echo "=== OpenMP task vs work-sharing versions ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    for t in $THREAD_COUNTS; do
      for i in "${!VTAG_ARR[@]}"; do
        tag="${VTAG_ARR[$i]}"; bin="${VBIN_ARR[$i]}"; sched="${VSCHED_ARR[$i]}"
        f="$RAW/${tag}_n${n}_nz${nz}_t${t}.times"
        if complete "$f"; then echo "skip (done) $tag n=$n nz=$nz t=$t"; continue; fi
        over_budget && stop_early
        # OMP_SCHEDULE only applies to the work-sharing binary (schedule(runtime)).
        # The 'dynamic,auto' pseudo-schedule resolves per (n,t) to dynamic,<chunk>
        # with chunk = n/(8t) -- exactly the task version's grainsize (resolve_grain,
        # iterative_SpMV_omp.cpp) -- so it is the same-granularity-as-tasks point.
        if [ "$sched" = "dynamic,auto" ]; then
            chunk=$(( n / (8 * t) )); [ "$chunk" -lt 1 ] && chunk=1
            export OMP_SCHEDULE="dynamic,$chunk"
        elif [ -n "$sched" ]; then
            export OMP_SCHEDULE="$sched"
        else
            unset OMP_SCHEDULE
        fi
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
done

echo ""
echo "=== Sweep done in ${SECONDS}s. Analyze with: ./scripts/analyze_all.sh $RESULTS_DIR ==="
