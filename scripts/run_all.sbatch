#!/bin/bash
#SBATCH --job-name=spmv_all
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results_all_%j.log
#SBATCH --error=errors_all_%j.log

# =============================================================================
# Full head-to-head sweep: sequential reference vs ALL threaded versions.
#
# Versions measured (single node, as required by the assignment):
#   seq      : iterative_SpMV.cpp                    (sequential reference)
#   baseline : iterative_SpMV_threads_baseline.cpp   (mutex+CV barrier, unpadded)
#   ab       : iterative_SpMV_threads_AB.cpp         (A: padding + B: std::barrier)
#   cde      : iterative_SpMV_threads.cpp            (A+B + C affinity, D __restrict__,
#                                                     E barrier completion fn)
#
# Noise control: for each (n, nz, threads) cell the three threaded versions are
# run INTERLEAVED (baseline, ab, cde back-to-back) and each cell is REPEATED
# REPS times. The analyzer (analyze_all.sh) takes the MEDIAN. Because the three
# versions are always measured a moment apart they share the same machine state,
# so their relative comparison is trustworthy. The node is requested --exclusive
# so thread pinning (Opt C) and NUMA behaviour are measured on a quiet machine.
#
# Cluster (spmcluster) note: partition 'normal' has a 30:00 wall clock limit, so
# --time is set to the maximum. If the sweep risks overrunning, trim NONZEROS,
# THREAD_COUNTS, or REPS below; the analyzer tolerates missing cells (prints NA).
#
# Matrix generation is NOT timed (only the iterative computation is), exactly as
# in the sequential reference.
# =============================================================================

# Run from the project root so the src/... paths below and the output directory
# resolve regardless of where sbatch was invoked from.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# All knobs below are overridable from the environment (handy for a quick smoke
# test, e.g. REPS=1 MATRIX_SIZES=2000 NONZEROS=8000 THREAD_COUNTS="1 2" ...).
RESULTS_DIR="${RESULTS_DIR:-results_all}"
REPS="${REPS:-3}"                        # repetitions per cell (median of these)
MATRIX_SIZES="${MATRIX_SIZES:-100000 500000}"
NONZEROS="${NONZEROS:-4000000 20000000}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8 16 32}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"
CXXFLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -Wall -Wextra -pthread"

# Threaded version sources (override via env if needed).
SEQ_SRC="${SEQ_SRC:-src/sequential/iterative_SpMV.cpp}"
BASE_SRC="${BASE_SRC:-src/threads/iterative_SpMV_threads_baseline.cpp}"
AB_SRC="${AB_SRC:-src/threads/iterative_SpMV_threads_AB.cpp}"
CDE_SRC="${CDE_SRC:-src/threads/iterative_SpMV_threads.cpp}"

BUILD="$RESULTS_DIR/build"
RAW="$RESULTS_DIR/raw"
mkdir -p "$BUILD" "$RAW"

# Cap thread counts to the CPUs SLURM actually granted, so we never oversubscribe
# (which would distort the affinity/scalability picture).
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
for pair in "seq:$SEQ_SRC" "base:$BASE_SRC" "ab:$AB_SRC" "cde:$CDE_SRC"; do
    tag="${pair%%:*}"; src="${pair#*:}"
    if [ ! -f "$src" ]; then
        echo "ERROR: source not found: $src" >&2; exit 1
    fi
    echo "  $tag <- $src"
    g++ $CXXFLAGS -I include "$src" -o "$BUILD/$tag" || { echo "build failed: $src" >&2; exit 1; }
done

SEQ="$BUILD/seq"
declare -a VNAMES=("baseline" "ab" "cde")
declare -a VTAGS=("base" "ab" "cde")     # raw-file prefixes / binary names

echo ""
echo "=== Configuration ==="
echo "sizes=$MATRIX_SIZES  nonzeros=$NONZEROS  threads=$THREAD_COUNTS"
echo "mode=$MODE  seed=$SEED  reps=$REPS  output=$RESULTS_DIR"
echo "versions: seq, ${VNAMES[*]}"
echo ""

# Sorted-unique numeric union of two space-separated lists.
merge_list() { echo "$1 $2" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }

# Resumable runs: if a partial config.env already exists, the analyzer should
# still see the full grid ever swept, so union this run's grid with the saved
# one. (The run loops below only iterate THIS invocation's grid and skip cells
# already on disk, so re-running fills gaps without redoing completed work.)
ALL_SIZES="$MATRIX_SIZES"; ALL_NZ="$NONZEROS"; ALL_T="$THREAD_COUNTS"
if [ -f "$RESULTS_DIR/config.env" ]; then
    OLD_SIZES=$(sed -n 's/^MATRIX_SIZES="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    OLD_NZ=$(sed -n 's/^NONZEROS="\(.*\)"/\1/p'     "$RESULTS_DIR/config.env")
    OLD_T=$(sed -n 's/^THREAD_COUNTS="\(.*\)"/\1/p' "$RESULTS_DIR/config.env")
    ALL_SIZES=$(merge_list "$OLD_SIZES" "$MATRIX_SIZES")
    ALL_NZ=$(merge_list "$OLD_NZ" "$NONZEROS")
    ALL_T=$(merge_list "$OLD_T" "$THREAD_COUNTS")
fi

# Record the (cumulative) grid so the analyzer needs no arguments.
{
    echo "MATRIX_SIZES=\"$ALL_SIZES\""
    echo "NONZEROS=\"$ALL_NZ\""
    echo "THREAD_COUNTS=\"$ALL_T\""
    echo "VTAGS=\"${VTAGS[*]}\""
    echo "VNAMES=\"${VNAMES[*]}\""
    echo "MODE=\"$MODE\""
    echo "SEED=\"$SEED\""
    echo "REPS=\"$REPS\""
} > "$RESULTS_DIR/config.env"

# ----------------------------------------------------------------------------
# Helper: run a binary once, echo "<time> <checksum>".
# ----------------------------------------------------------------------------
run_one() {
    local out tm cs
    out=$("$@" 2>/dev/null)
    tm=$(echo "$out" | grep "Time (sec)" | awk '{print $NF}')
    cs=$(echo "$out" | grep "checksum=" | head -1 | cut -d= -f2)
    echo "${tm:-NA} ${cs:-NA}"
}

# A cell is complete if its .times file already holds >= REPS non-empty lines.
complete() { [ -f "$1" ] && [ "$(grep -c . "$1")" -ge "$REPS" ]; }

# ----------------------------------------------------------------------------
# Sequential reference (once per (n,nz); REPS reps).
# ----------------------------------------------------------------------------
echo "=== Sequential reference ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    if complete "$RAW/seq_n${n}_nz${nz}.times"; then
      echo "skip (done) seq n=$n nz=$nz"; continue
    fi
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
# Threaded versions, interleaved per cell.
# ----------------------------------------------------------------------------
echo ""
echo "=== Threaded versions (interleaved: ${VNAMES[*]}) ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    for t in $THREAD_COUNTS; do
      alldone=1
      for i in "${!VTAGS[@]}"; do
        complete "$RAW/${VTAGS[$i]}_n${n}_nz${nz}_t${t}.times" || alldone=0
      done
      if [ "$alldone" = 1 ]; then echo "skip (done) n=$n nz=$nz t=$t"; continue; fi
      for i in "${!VTAGS[@]}"; do : > "$RAW/${VTAGS[$i]}_n${n}_nz${nz}_t${t}.times"; done
      for r in $(seq 1 $REPS); do
        line="n=$n nz=$nz t=$t rep=$r "
        for i in "${!VTAGS[@]}"; do
          tag="${VTAGS[$i]}"
          read tm cs < <(run_one "$BUILD/$tag" -n $n -nz $nz -m $MODE -s $SEED -t $t)
          echo "$tm" >> "$RAW/${tag}_n${n}_nz${nz}_t${t}.times"
          echo "$cs" >  "$RAW/${tag}_n${n}_nz${nz}_t${t}.cksum"
          line+=" ${tag}=$tm"
        done
        echo "$line"
      done
    done
  done
done

echo ""
echo "=== Sweep done. Analyze with: ./analyze_all.sh $RESULTS_DIR ==="
