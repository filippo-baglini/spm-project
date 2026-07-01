#!/bin/bash
# =============================================================================
# ONE combined metrics report across every results_* sweep we produced, so all
# implemented versions are summarized in a single command instead of running
# analyze_all.sh / compare_mpi.sh per directory by hand.
#
# It does two things:
#   1. For each results directory it runs the matching analyzer and inlines its
#      full output:
#        - shared-memory sweeps (seq / threads / pool / omp tasks / omp_ws /
#          *_numa), thread-keyed raw  ->  scripts/analyze_all.sh
#        - hybrid MPI+OpenMP sweep (block + overlap), rank x thread-keyed raw ->
#          scripts/compare_mpi.sh
#   2. A consolidated LEADERBOARD: for every (n,nz) matrix, the best speedup each
#      version reached and the configuration (threads, or ranks x threads x
#      chunks) where it happened -- every version side by side, sorted fastest
#      first.
#
# The sequential baseline is pooled across all sweeps (the slim MPI path does not
# re-run seq), exactly like compare_mpi.sh does.
#
# Usage:
#   ./scripts/report_all.sh                       # auto-discover results_*
#   ./scripts/report_all.sh results_omp_ws results_numa results_mpi
#   OUT=report.txt ./scripts/report_all.sh
#   SEQ_DIRS="results_numa results_all" ./scripts/report_all.sh
# =============================================================================

set -u
cd "$(dirname "$0")/.." || exit 1

OUT="${OUT:-report_all.txt}"

# Directories: explicit args, else every results_* with a raw/ folder.
if [ "$#" -gt 0 ]; then
    DIRS="$*"
else
    DIRS=$(for d in results_*; do [ -d "$d/raw" ] && echo "$d"; done | tr '\n' ' ')
fi
DIRS=$(echo "$DIRS" | tr '\n' ' ')   # normalize to a single space-separated line

# Pool the (stable) sequential baseline from these dirs for the leaderboard.
SEQ_DIRS="${SEQ_DIRS:-$DIRS results_numa results_all results_pool}"

# ---- small helpers (same metric math as analyze_all.sh / compare_mpi.sh) -----
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
getcfg()   { sed -n "s/^$2=\"\\(.*\\)\"/\\1/p" "$1/config.env" 2>/dev/null; }
has_glob() { compgen -G "$1" >/dev/null 2>&1; }
nonempty() { [ -d "$1/raw" ] && [ -n "$(ls -A "$1/raw" 2>/dev/null)" ]; }
is_mpi()   { has_glob "$1/raw/mpi_*.times" || has_glob "$1/raw/mpiov_*.times"; }
is_shared(){ [ -f "$1/config.env" ] && grep -q '^VTAGS=' "$1/config.env" 2>/dev/null; }

# Pooled sequential median for (n,nz) across SEQ_DIRS.
seq_med() {
    local n="$1" nz="$2" d f files=()
    for d in $SEQ_DIRS; do f="$d/raw/seq_n${n}_nz${nz}.times"; [ -f "$f" ] && files+=("$f"); done
    [ ${#files[@]} -eq 0 ] && { echo "NA"; return; }
    cat "${files[@]}" | median
}

# Speedup = seq/time as a printable string (NA-safe).
sp_str() { awk -v tm="$1" -v sq="$2" 'BEGIN{ if(tm!="NA"&&sq!="NA"&&tm>0) printf "%.2f", sq/tm; else printf "NA" }'; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# =============================================================================
{
echo "##############################################################################"
echo "#  COMBINED METRICS REPORT  -  all implemented versions"
echo "#  directories: $DIRS"
echo "#  seq baseline pooled from: $SEQ_DIRS"
echo "##############################################################################"

# ----------------------------------------------------------------------------
# Part 1: full per-directory analyzer output.
# ----------------------------------------------------------------------------
for d in $DIRS; do
    if ! nonempty "$d"; then
        echo ""; echo ">>> $d : (no raw data, skipped)"; continue
    fi
    echo ""
    echo "=============================================================================="
    if is_mpi "$d"; then
        echo ">>> $d   [hybrid MPI + OpenMP : block + overlap]"
        echo "=============================================================================="
        DIR="$d" SEQ_DIRS="$SEQ_DIRS" OUT="$TMP/mpi_$(basename "$d").txt" bash scripts/compare_mpi.sh 2>&1
    elif is_shared "$d"; then
        echo ">>> $d   [shared-memory : $(getcfg "$d" VNAMES)]"
        echo "=============================================================================="
        bash scripts/analyze_all.sh "$d" 2>&1
    else
        echo ">>> $d : (unrecognized layout, skipped)"
    fi
done

# ----------------------------------------------------------------------------
# Part 2: consolidated leaderboard - best speedup per version per matrix.
# ----------------------------------------------------------------------------
echo ""
echo "##############################################################################"
echo "#  LEADERBOARD  -  best speedup each version reached (vs pooled sequential)"
echo "##############################################################################"

# Union of (n,nz) matrices seen across all directories' configs.
for d in $DIRS; do
    [ -f "$d/config.env" ] || continue
    for n in $(getcfg "$d" MATRIX_SIZES); do
        for nz in $(getcfg "$d" NONZEROS); do echo "$n $nz"; done
    done
done | sort -k1,1n -k2,2n -u > "$TMP/pairs"

while read -r n nz; do
    [ -z "${n:-}" ] && continue
    seqmed=$(seq_med "$n" "$nz")
    echo ""
    echo "------------------------------------------------------------------------------"
    printf "  n=%s  nz=%s   (sequential median = %ss)\n" "$n" "$nz" "$seqmed"
    echo "------------------------------------------------------------------------------"
    printf "  %-34s %-10s %-16s %-10s\n" "version (source)" "speedup" "at config" "time(s)"
    printf "  %-34s %-10s %-16s %-10s\n" "----------------" "-------" "---------" "-------"

    : > "$TMP/rows"

    for d in $DIRS; do
        nonempty "$d" || continue

        # --- shared-memory: best over the dir's THREAD_COUNTS, per version tag.
        if is_shared "$d" && ! is_mpi "$d"; then
            tcs=$(getcfg "$d" THREAD_COUNTS)
            for tag in $(getcfg "$d" VTAGS); do
                best="NA"; bestcfg="-"; besttm="NA"
                for t in $tcs; do
                    tm=$(med_file "$d/raw/${tag}_n${n}_nz${nz}_t${t}.times")
                    [ "$tm" = "NA" ] && continue
                    sp=$(sp_str "$tm" "$seqmed")
                    [ "$sp" = "NA" ] && continue
                    if [ "$best" = "NA" ] || awk -v a="$sp" -v b="$best" 'BEGIN{exit !(a>b)}'; then
                        best="$sp"; bestcfg="t=$t"; besttm="$tm"
                    fi
                done
                [ "$best" = "NA" ] && continue
                printf "%s\t%s (%s)\t%s\t%s\n" "$best" "$tag" "$(basename "$d")" "$bestcfg" "$besttm" >> "$TMP/rows"
            done
        fi

        # --- MPI: best block and best overlap over all geometries (r x t [x c]).
        if is_mpi "$d"; then
            # block: mpi_n{n}_nz{nz}_r{R}_t{T}.times
            best="NA"; bestcfg="-"; besttm="NA"
            for f in "$d"/raw/mpi_n${n}_nz${nz}_r*_t*.times; do
                [ -e "$f" ] || continue
                b=$(basename "$f" .times)
                R=$(echo "$b" | sed -n 's/.*_r\([0-9]*\)_t.*/\1/p')
                T=$(echo "$b" | sed -n 's/.*_t\([0-9]*\)$/\1/p')
                tm=$(med_file "$f"); [ "$tm" = "NA" ] && continue
                sp=$(sp_str "$tm" "$seqmed"); [ "$sp" = "NA" ] && continue
                if [ "$best" = "NA" ] || awk -v a="$sp" -v b="$best" 'BEGIN{exit !(a>b)}'; then
                    best="$sp"; bestcfg="r${R}xt${T}"; besttm="$tm"
                fi
            done
            [ "$best" != "NA" ] && printf "%s\t%s (%s)\t%s\t%s\n" "$best" "mpi_block" "$(basename "$d")" "$bestcfg" "$besttm" >> "$TMP/rows"

            # overlap: mpiov_n{n}_nz{nz}_r{R}_t{T}_c{C}.times
            best="NA"; bestcfg="-"; besttm="NA"
            for f in "$d"/raw/mpiov_n${n}_nz${nz}_r*_t*_c*.times; do
                [ -e "$f" ] || continue
                b=$(basename "$f" .times)
                R=$(echo "$b" | sed -n 's/.*_r\([0-9]*\)_t.*/\1/p')
                T=$(echo "$b" | sed -n 's/.*_t\([0-9]*\)_c.*/\1/p')
                C=$(echo "$b" | sed -n 's/.*_c\([0-9]*\)$/\1/p')
                tm=$(med_file "$f"); [ "$tm" = "NA" ] && continue
                sp=$(sp_str "$tm" "$seqmed"); [ "$sp" = "NA" ] && continue
                if [ "$best" = "NA" ] || awk -v a="$sp" -v b="$best" 'BEGIN{exit !(a>b)}'; then
                    best="$sp"; bestcfg="r${R}xt${T}c${C}"; besttm="$tm"
                fi
            done
            [ "$best" != "NA" ] && printf "%s\t%s (%s)\t%s\t%s\n" "$best" "mpi_overlap" "$(basename "$d")" "$bestcfg" "$besttm" >> "$TMP/rows"
        fi
    done

    # Sort rows by speedup (col 1) descending, then print.
    if [ -s "$TMP/rows" ]; then
        sort -t$'\t' -k1 -nr "$TMP/rows" | while IFS=$'\t' read -r sp who cfg tm; do
            printf "  %-26s %-10s %-16s %-10s\n" "$who" "$sp" "$cfg" "$tm"
        done
    else
        echo "  (no data for this matrix)"
    fi
done < "$TMP/pairs"

echo ""
echo "Notes:"
echo "  Speedup = T_seq / T_version   (>1 faster than sequential; baseline pooled across sweeps)"
echo "  config  = threads (shared-memory) or ranks x threads-per-rank [x chunks] (MPI)"
echo "  Full per-version Efficiency / Scalability / Karp-Flatt tables are in Part 1 above."
} | tee "$OUT"

echo ""
echo "Combined report written to $OUT"
