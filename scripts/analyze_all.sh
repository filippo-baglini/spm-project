#!/bin/bash
# =============================================================================
# Analyze the results produced by run_all.sbatch in terms of the course metrics
# (SPM NOTES, "Speedup / Efficiency / Scalability definition"):
#
#   Speedup       S(p) = T_seq      / T_version(p)        (vs sequential reference)
#   Efficiency    E(p) = S(p) / p                          (ideal = 1.00)
#   Scalability   R(p) = T_version(1) / T_version(p)       (relative speedup)
#   Karp-Flatt    e(p) = (1/S - 1/p) / (1 - 1/p)           (experimentally
#                 determined serial fraction; a rising e with p reveals growing
#                 parallel overhead rather than a fixed serial section)
#
# All times are the MEDIAN over the repetitions recorded by run_all.sbatch.
# Sequential vs threaded checksums differ by design (different floating-point
# reduction order); the threaded versions should match EACH OTHER bitwise.
#
# Usage: ./analyze_all.sh [results_dir]   (default: results_all)
# =============================================================================

set -u
# Run from the project root so a relative results dir (default: results_all)
# resolves against the project, not the scripts/ folder.
cd "$(dirname "$0")/.." || exit 1
DIR="${1:-results_all}"
RAW="$DIR/raw"
CFG="$DIR/config.env"

if [ ! -d "$RAW" ] || [ ! -f "$CFG" ]; then
    echo "ERROR: '$DIR' is missing raw/ or config.env. Run run_all.sbatch first." >&2
    exit 1
fi
# shellcheck disable=SC1090
. "$CFG"

# Median of numbers on stdin (one per line); ignores NA. Prints NA if empty.
median() {
    grep -v '^NA$' | sort -n | awk '
        { a[NR]=$1 }
        END {
            if (NR==0) { print "NA"; exit }
            if (NR%2)  { printf "%.6f\n", a[(NR+1)/2] }
            else       { printf "%.6f\n", (a[NR/2]+a[NR/2+1])/2 }
        }'
}

med_file() { [ -f "$1" ] && median < "$1" || echo "NA"; }

# Map a version tag -> human name (parallel arrays from config.env).
read -r -a TAGS  <<< "$VTAGS"
read -r -a NAMES <<< "$VNAMES"
name_of() {
    local i
    for i in "${!TAGS[@]}"; do
        if [ "${TAGS[$i]}" = "$1" ]; then echo "${NAMES[$i]}"; return; fi
    done
    echo "$1"
}

summary="$DIR/analysis.txt"
{
echo "=============================================================="
echo " SpMV performance analysis   (medians of $REPS reps)"
echo " mode=$MODE  seed=$SEED   dir=$DIR"
echo " metrics: Speedup, Efficiency, Scalability, Karp-Flatt e"
echo "=============================================================="

for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    seqmed=$(med_file "$RAW/seq_n${n}_nz${nz}.times")

    echo ""
    echo "##############################################################"
    printf "# n=%s  nz=%s    (sequential median = %ss)\n" "$n" "$nz" "$seqmed"
    echo "##############################################################"

    # ---- per-version metric tables ----
    for tag in $VTAGS; do
      vname=$(name_of "$tag")
      echo ""
      echo "--- version: $vname [$tag] -------------------------------------"
      printf "%-8s %-10s %-9s %-10s %-11s %-9s\n" \
             "Threads" "Time(s)" "Speedup" "Effic." "Scalab." "KarpFlatt"
      echo "------- ---------- -------- ---------- ----------- ---------"

      base1=$(med_file "$RAW/${tag}_n${n}_nz${nz}_t1.times")   # T_version(1)
      for t in $THREAD_COUNTS; do
        tm=$(med_file "$RAW/${tag}_n${n}_nz${nz}_t${t}.times")
        awk -v t="$t" -v tm="$tm" -v sq="$seqmed" -v b1="$base1" '
          BEGIN {
            sp = (tm!="NA" && sq!="NA" && tm>0) ? sq/tm : "NA";
            ef = (sp!="NA") ? sp/t : "NA";
            sc = (tm!="NA" && b1!="NA" && tm>0) ? b1/tm : "NA";
            if (sp!="NA" && t>1) kf = (1/sp - 1.0/t) / (1 - 1.0/t); else kf = "NA";
            spS = (sp=="NA")?"NA":sprintf("%.2f", sp);
            efS = (ef=="NA")?"NA":sprintf("%.2f", ef);
            scS = (sc=="NA")?"NA":sprintf("%.2f", sc);
            kfS = (kf=="NA")?"-":sprintf("%.4f", kf);
            printf "%-8s %-10s %-9s %-10s %-11s %-9s\n", t, tm, spS, efS, scS, kfS;
          }'
      done
    done

    # ---- cross-version speedup comparison ----
    echo ""
    echo "--- speedup vs sequential, all versions side by side ----------"
    printf "%-8s" "Threads"; for tag in $VTAGS; do printf " %-10s" "$(name_of "$tag")"; done; echo ""
    printf "%-8s" "-------"; for tag in $VTAGS; do printf " %-10s" "----------"; done; echo ""
    for t in $THREAD_COUNTS; do
      printf "%-8s" "$t"
      for tag in $VTAGS; do
        tm=$(med_file "$RAW/${tag}_n${n}_nz${nz}_t${t}.times")
        sp=$(awk -v tm="$tm" -v sq="$seqmed" 'BEGIN{ if(tm!="NA"&&sq!="NA"&&tm>0) printf "%.2f", sq/tm; else printf "NA" }')
        printf " %-10s" "$sp"
      done
      echo ""
    done

    # ---- checksum consistency across threaded versions ----
    echo ""
    echo "--- checksum consistency (threaded versions vs each other) ----"
    for t in $THREAD_COUNTS; do
      ref=""; ok="match"
      for tag in $VTAGS; do
        cs=$(cat "$RAW/${tag}_n${n}_nz${nz}_t${t}.cksum" 2>/dev/null)
        if [ -z "$ref" ]; then ref="$cs"; elif [ "$cs" != "$ref" ]; then ok="DIFF"; fi
      done
      printf "  t=%-3s %s  (%s)\n" "$t" "$ref" "$ok"
    done
    seqcs=$(cat "$RAW/seq_n${n}_nz${nz}.cksum" 2>/dev/null)
    echo "  seq:   $seqcs  (expected to differ from threaded: different reduction order)"
  done
done

echo ""
echo "Notes:"
echo "  Speedup    = T_seq / T_version(p)        (>1 faster than sequential)"
echo "  Efficiency = Speedup / p                 (1.00 = perfect scaling)"
echo "  Scalability= T_version(1) / T_version(p) (relative speedup of that version)"
echo "  KarpFlatt  = experimentally determined serial fraction; rising with p"
echo "               => parallel overhead (sync/memory) growing, not a fixed serial part"
} | tee "$summary"

echo ""
echo "Analysis written to $summary"
