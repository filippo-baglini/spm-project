#!/bin/bash
# Compare two SpMV result directories (e.g. baseline vs optimized).
#
# Usage:
#   ./compare_runs.sh <old_dir> <new_dir>
#   ./compare_runs.sh results results_optimized
#
# For every (n, nz) configuration and thread count present in both runs it
# prints, side by side: the threaded time, the speedup vs each run's own
# sequential baseline, the time improvement (old/new), and whether the two
# runs produced the same checksum.
#
# Note: the optimizations (cache-line padding + std::barrier) do NOT change the
# arithmetic order, so the old and new threaded checksums MUST match exactly.
# A "DIFF" in the checksum column therefore signals a real problem, not just
# floating-point reordering.

old_dir="${1:-results}"
new_dir="${2:-results_optimized}"

if [ ! -d "$old_dir" ] || [ ! -d "$new_dir" ]; then
    echo "Usage: $0 <old_dir> <new_dir>"
    echo "  one of the directories does not exist: '$old_dir' / '$new_dir'"
    exit 1
fi

get_time()     { grep "Time (sec)" "$1" 2>/dev/null | awk '{print $NF}'; }
get_checksum() { grep "checksum="  "$1" 2>/dev/null | head -1 | cut -d= -f2; }

echo "=============================================================="
echo "Comparison: old='$old_dir'  vs  new='$new_dir'"
echo "=============================================================="

# Discover configurations from the old dir's sequential result files.
configs=$(ls "$old_dir"/seq_n*.txt 2>/dev/null \
    | sed 's#.*/seq_n\([0-9]*\)_nz\([0-9]*\)\.txt#\1 \2#' | sort -u)

if [ -z "$configs" ]; then
    echo "No seq_n*.txt files found in '$old_dir'."
    exit 1
fi

echo "$configs" | while read n nz; do
    echo ""
    echo "--------------------------------------------------------------"
    printf "Configuration: n=%s  nz=%s\n" "$n" "$nz"
    echo "--------------------------------------------------------------"

    seq_old=$(get_time "$old_dir/seq_n${n}_nz${nz}.txt")
    seq_new=$(get_time "$new_dir/seq_n${n}_nz${nz}.txt")
    printf "Sequential baseline:  old=%ss  new=%ss\n\n" "${seq_old:-NA}" "${seq_new:-NA}"

    printf "%-8s %-10s %-10s %-9s %-9s %-9s %-10s\n" \
        "Threads" "Old(s)" "New(s)" "Improve" "Spd_old" "Spd_new" "Checksum"
    echo "------- ---------- ---------- -------- -------- -------- ----------"

    # Thread counts found in the old dir for this config.
    tcounts=$(ls "$old_dir"/thread_n${n}_nz${nz}_t*.txt 2>/dev/null \
        | sed 's#.*_t\([0-9]*\)\.txt#\1#' | sort -n)

    for t in $tcounts; do
        of="$old_dir/thread_n${n}_nz${nz}_t${t}.txt"
        nf="$new_dir/thread_n${n}_nz${nz}_t${t}.txt"

        to=$(get_time "$of"); tn=$(get_time "$nf")
        co=$(get_checksum "$of"); cn=$(get_checksum "$nf")

        if [ -z "$tn" ]; then
            printf "%-8s %-10s %-10s %-9s %-9s %-9s %-10s\n" \
                "$t" "${to:-NA}" "MISSING" "-" "-" "-" "-"
            continue
        fi

        improve=$(awk "BEGIN{ if($tn>0) printf \"%.2fx\", $to/$tn; else print \"NA\" }")
        spo=$(awk "BEGIN{ if(\"$seq_old\"!=\"\" && $to>0) printf \"%.2f\", $seq_old/$to; else print \"NA\" }")
        spn=$(awk "BEGIN{ if(\"$seq_new\"!=\"\" && $tn>0) printf \"%.2f\", $seq_new/$tn; else print \"NA\" }")

        if [ "$co" = "$cn" ]; then cs="match"; else cs="DIFF"; fi

        printf "%-8s %-10s %-10s %-9s %-9s %-9s %-10s\n" \
            "$t" "$to" "$tn" "$improve" "$spo" "$spn" "$cs"
    done
done

echo ""
echo "Improve = old_time / new_time  (>1.00x means the optimized run is faster)"
echo "Spd_*   = sequential_time / threaded_time  within each run"
