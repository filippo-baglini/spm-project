#!/bin/bash
#SBATCH --job-name=spmv_interleaved
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results_interleaved_%j.log
#SBATCH --error=errors_interleaved_%j.log

# Noise-resistant head-to-head: baseline vs optimized threaded SpMV.
#
# Why this script exists: comparing two runs measured at different times is
# unreliable (the node's load/thermal state drifts by several %). Here both
# versions are built and run INTERLEAVED (base then opt, back to back) and each
# cell is REPEATED several times; we report the MEDIAN. Because base and opt are
# always measured a second apart, they share the same machine state, so the
# median ratio is a trustworthy verdict.
#
# Both threaded versions live as separate source files in the repo, so this
# works on any machine WITHOUT needing the git history (just copy the project
# files over):
#   - baseline : iterative_SpMV_threads_baseline.cpp (mutex+CV barrier, unpadded)
#   - optimized: iterative_SpMV_threads.cpp          (padding + std::barrier)
# Override with BASELINE_SRC / OPT_SRC env vars to point at other .cpp files.

# Run from the project root so src/... paths and the output dir resolve.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

RESULTS_DIR="results_interleaved"
REPS=3                                  # repetitions per cell (3-5 recommended)
MATRIX_SIZES="100000 500000"
NONZEROS="4000000 20000000"
THREAD_COUNTS="1 2 4 8 16 32"
MODE="irregular"
SEED="111"
CXXFLAGS="-O3 -march=native -funroll-loops -std=c++20 -Wall -Wextra -pthread"

BUILD="$RESULTS_DIR/build"
RAW="$RESULTS_DIR/raw"
mkdir -p "$BUILD" "$RAW"

# ----------------------------------------------------------------------------
# Build the three binaries
# ----------------------------------------------------------------------------
echo "=== Building binaries ==="

# Sequential reference (unchanged across versions -> use working tree).
g++ $CXXFLAGS -I include src/sequential/iterative_SpMV.cpp -o "$BUILD/seq_ref"

# Resolve baseline / optimized threaded sources (plain files; no git needed).
BASELINE_SRC="${BASELINE_SRC:-src/threads/iterative_SpMV_threads_baseline.cpp}"
OPT_SRC="${OPT_SRC:-src/threads/iterative_SpMV_threads.cpp}"

for src in "$BASELINE_SRC" "$OPT_SRC"; do
    if [ ! -f "$src" ]; then
        echo "ERROR: source file not found: $src" >&2
        echo "  Expected both threaded versions in the project dir, or set" >&2
        echo "  BASELINE_SRC / OPT_SRC to valid .cpp paths." >&2
        exit 1
    fi
done

echo "baseline source : $BASELINE_SRC"
echo "optimized source: $OPT_SRC"
g++ $CXXFLAGS -I include "$BASELINE_SRC" -o "$BUILD/threads_base"
g++ $CXXFLAGS -I include "$OPT_SRC"      -o "$BUILD/threads_opt"

SEQ="$BUILD/seq_ref"
BASE="$BUILD/threads_base"
OPT="$BUILD/threads_opt"

echo ""
echo "=== Configuration ==="
echo "sizes=$MATRIX_SIZES  nonzeros=$NONZEROS  threads=$THREAD_COUNTS"
echo "mode=$MODE  seed=$SEED  reps=$REPS  output=$RESULTS_DIR"
echo ""

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# Run a binary once; echo "<time> <checksum>".
run_one() {
    local out
    out=$("$@" 2>/dev/null)
    local tm cs
    tm=$(echo "$out" | grep "Time (sec)" | awk '{print $NF}')
    cs=$(echo "$out" | grep "checksum=" | head -1 | cut -d= -f2)
    echo "${tm:-NA} ${cs:-NA}"
}

# Median of numbers on stdin (one per line). Average of the two middle values
# for an even count.
median() {
    sort -n | awk '
        { a[NR]=$1 }
        END {
            if (NR==0) { print "NA"; exit }
            if (NR%2)  { printf "%.6f\n", a[(NR+1)/2] }
            else       { printf "%.6f\n", (a[NR/2]+a[NR/2+1])/2 }
        }'
}

# ----------------------------------------------------------------------------
# Sequential baseline (REPS runs per config; median used for speedups)
# ----------------------------------------------------------------------------
echo "=== Sequential reference timing ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
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
# Interleaved threaded runs: base then opt, back-to-back, REPS times.
# ----------------------------------------------------------------------------
echo ""
echo "=== Interleaved threaded timing (base | opt) ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    for t in $THREAD_COUNTS; do
      : > "$RAW/base_n${n}_nz${nz}_t${t}.times"
      : > "$RAW/opt_n${n}_nz${nz}_t${t}.times"
      for r in $(seq 1 $REPS); do
        read tb cb < <(run_one "$BASE" -n $n -nz $nz -m $MODE -s $SEED -t $t)
        read to co < <(run_one "$OPT"  -n $n -nz $nz -m $MODE -s $SEED -t $t)
        echo "$tb" >> "$RAW/base_n${n}_nz${nz}_t${t}.times"
        echo "$to" >> "$RAW/opt_n${n}_nz${nz}_t${t}.times"
        echo "$cb" >  "$RAW/base_n${n}_nz${nz}_t${t}.cksum"
        echo "$co" >  "$RAW/opt_n${n}_nz${nz}_t${t}.cksum"
        echo "n=$n nz=$nz t=$t rep=$r  base=$tb  opt=$to"
      done
    done
  done
done

# ----------------------------------------------------------------------------
# Summary (medians)
# ----------------------------------------------------------------------------
summary="$RESULTS_DIR/summary.txt"
{
  echo "=============================================================="
  echo "Interleaved benchmark summary  (median of $REPS reps)"
  echo "mode=$MODE  seed=$SEED"
  echo "=============================================================="

  for n in $MATRIX_SIZES; do
    for nz in $NONZEROS; do
      seqmed=$(median < "$RAW/seq_n${n}_nz${nz}.times")
      echo ""
      echo "--------------------------------------------------------------"
      printf "n=%s  nz=%s   (seq median = %ss)\n" "$n" "$nz" "$seqmed"
      echo "--------------------------------------------------------------"
      printf "%-8s %-10s %-10s %-9s %-9s %-9s %-9s\n" \
        "Threads" "Base(s)" "Opt(s)" "Improve" "Spd_base" "Spd_opt" "Checksum"
      echo "------- ---------- ---------- -------- -------- -------- --------"
      for t in $THREAD_COUNTS; do
        bmed=$(median < "$RAW/base_n${n}_nz${nz}_t${t}.times")
        omed=$(median < "$RAW/opt_n${n}_nz${nz}_t${t}.times")
        cb=$(cat "$RAW/base_n${n}_nz${nz}_t${t}.cksum" 2>/dev/null)
        co=$(cat "$RAW/opt_n${n}_nz${nz}_t${t}.cksum" 2>/dev/null)

        improve=$(awk "BEGIN{ if(\"$omed\"!=\"NA\" && $omed>0) printf \"%.2fx\", $bmed/$omed; else print \"NA\" }")
        spb=$(awk "BEGIN{ if(\"$seqmed\"!=\"NA\" && $bmed>0) printf \"%.2f\", $seqmed/$bmed; else print \"NA\" }")
        spo=$(awk "BEGIN{ if(\"$seqmed\"!=\"NA\" && $omed>0) printf \"%.2f\", $seqmed/$omed; else print \"NA\" }")
        if [ "$cb" = "$co" ]; then cs="match"; else cs="DIFF"; fi

        printf "%-8s %-10s %-10s %-9s %-9s %-9s %-9s\n" \
          "$t" "$bmed" "$omed" "$improve" "$spb" "$spo" "$cs"
      done
    done
  done

  echo ""
  echo "Improve  = base_median / opt_median   (>1.00x => optimized is faster)"
  echo "Spd_*    = seq_median / threaded_median"
  echo "Checksum = base vs opt; MUST be 'match' (A+B do not change arithmetic)"
} | tee "$summary"

echo ""
echo "=== Done. Summary written to $summary ==="
