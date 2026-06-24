#!/bin/bash
# =============================================================================
# Head-to-head comparison of the C++ THREADS (static nnz-partition) version and
# the THREADPOOL (dynamic) variant, from results already produced by the sbatch
# sweeps (run_all.sbatch -> results_all, run_pool.sbatch -> results_pool).
#
# Why a dedicated script: analyze_all.sh analyses ONE results dir at a time. Here
# the two versions live in two dirs AND the sequential baseline was measured once
# per sweep (i.e. TWICE). To avoid letting a single noisy sequential run skew the
# speedups, this script FOLDS ALL sequential repetitions from every results dir
# into one combined baseline (median over the pooled samples) and reports the
# spread between the individual runs so the fluctuation is visible.
#
# Metrics (SPM NOTES): Speedup S=T_seq/T(p), Efficiency E=S/p, plus the direct
# head-to-head ratio pool/threads (>1 => pool slower) and a geometric-mean verdict.
#
# Usage:
#   ./scripts/compare_threads_pool.sh
#   THREADS_DIR=results_all POOL_DIR=results_pool ./scripts/compare_threads_pool.sh
#   SEQ_DIRS="results_all results_pool results_omp" ./scripts/compare_threads_pool.sh
#   THREADS_TAG=cde POOL_TAG=pool OUT=threads_vs_pool.txt ./scripts/compare_threads_pool.sh
# =============================================================================

set -u
cd "$(dirname "$0")/.." || exit 1

THREADS_DIR="${THREADS_DIR:-results_all}"      # has the static-threads version (tag 'cde')
POOL_DIR="${POOL_DIR:-results_pool}"           # has the threadpool version  (tag 'pool')
POOL_TAG="${POOL_TAG:-pool}"
OUT="${OUT:-threads_vs_pool.txt}"
# Sequential samples are pooled from these dirs (the "ran twice" baseline).
SEQ_DIRS="${SEQ_DIRS:-$THREADS_DIR $POOL_DIR}"

# --- sanity --------------------------------------------------------------------
for d in "$THREADS_DIR" "$POOL_DIR"; do
    if [ ! -d "$d/raw" ] || [ ! -f "$d/config.env" ]; then
        echo "ERROR: '$d' is missing raw/ or config.env (run the sbatch sweeps first)." >&2
        exit 1
    fi
done

# Read a quoted value from a config.env without sourcing it (avoids clobbering).
getcfg() { sed -n "s/^$2=\"\\(.*\\)\"/\\1/p" "$1/config.env"; }
# Sorted-unique numeric union of two space-separated lists.
merge_list() { echo "$1 $2" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }

# Grid = union of both sweeps' grids (NA-tolerant for any missing cell).
MATRIX_SIZES=$(merge_list "$(getcfg "$THREADS_DIR" MATRIX_SIZES)" "$(getcfg "$POOL_DIR" MATRIX_SIZES)")
NONZEROS=$(merge_list     "$(getcfg "$THREADS_DIR" NONZEROS)"     "$(getcfg "$POOL_DIR" NONZEROS)")
THREAD_COUNTS=$(merge_list "$(getcfg "$THREADS_DIR" THREAD_COUNTS)" "$(getcfg "$POOL_DIR" THREAD_COUNTS)")
MODE=$(getcfg "$THREADS_DIR" MODE); [ -z "$MODE" ] && MODE=$(getcfg "$POOL_DIR" MODE)
SEED=$(getcfg "$THREADS_DIR" SEED); [ -z "$SEED" ] && SEED=$(getcfg "$POOL_DIR" SEED)

# Auto-detect the threads tag (the optimized static version is 'cde' in run_all).
THREADS_TAG="${THREADS_TAG:-}"
if [ -z "$THREADS_TAG" ]; then
    for cand in cde threads ab base; do
        if ls "$THREADS_DIR"/raw/${cand}_n*_t*.times >/dev/null 2>&1; then THREADS_TAG="$cand"; break; fi
    done
fi
if [ -z "$THREADS_TAG" ]; then
    echo "ERROR: no threaded results (tried cde/threads/ab/base) in '$THREADS_DIR/raw'." >&2
    exit 1
fi

# --- helpers -------------------------------------------------------------------
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

# Combined sequential median for (n,nz): pool every rep from every SEQ_DIRS dir.
seq_combined() {
    local n="$1" nz="$2" d f files=()
    for d in $SEQ_DIRS; do
        f="$d/raw/seq_n${n}_nz${nz}.times"
        [ -f "$f" ] && files+=("$f")
    done
    [ ${#files[@]} -eq 0 ] && { echo "NA"; return; }
    cat "${files[@]}" | median
}

# --- report --------------------------------------------------------------------
{
echo "=============================================================================="
echo " THREADS vs THREADPOOL  comparison"
echo " threads: [$THREADS_TAG] in $THREADS_DIR        pool: [$POOL_TAG] in $POOL_DIR"
echo " mode=$MODE  seed=$SEED   (all per-cell times are MEDIANS over the reps)"
echo " sequential baseline pooled from: $SEQ_DIRS"
echo "=============================================================================="

for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do

    seqmed=$(seq_combined "$n" "$nz")

    # Per-dir sequential medians + spread, so the fluctuation is explicit.
    nsamp=0; for d in $SEQ_DIRS; do f="$d/raw/seq_n${n}_nz${nz}.times"; [ -f "$f" ] && nsamp=$((nsamp + $(grep -c '[0-9]' "$f"))); done

    echo ""
    echo "##############################################################################"
    printf "# n=%s  nz=%s\n" "$n" "$nz"
    echo "##############################################################################"
    printf "Sequential baseline (combined median) = %ss   [pooled from %s reps]\n" "$seqmed" "$nsamp"
    for d in $SEQ_DIRS; do
        m=$(med_file "$d/raw/seq_n${n}_nz${nz}.times")
        printf "    %-14s seq median = %ss\n" "$d:" "$m"
    done
    # spread = max-min of the per-dir medians, relative to the min.
    for d in $SEQ_DIRS; do med_file "$d/raw/seq_n${n}_nz${nz}.times"; done | awk '
        /^NA$/ { next } { v[++k]=$1 }
        END {
            if (k<2) { print "    (only one sequential run available)"; exit }
            mn=v[1]; mx=v[1]; for(i=2;i<=k;i++){ if(v[i]<mn)mn=v[i]; if(v[i]>mx)mx=v[i] }
            if (mn>0) printf "    run-to-run spread = %.1f%% (min=%.4f, max=%.4f)\n", 100*(mx-mn)/mn, mn, mx
        }'

    # Head-to-head table.
    echo ""
    printf "%-8s %-9s %-9s %-9s %-9s %-8s %-8s %-9s\n" \
           "Threads" "thr T(s)" "pool T(s)" "thr Sp" "pool Sp" "thr E" "pool E" "pool/thr"
    echo "------- --------- --------- --------- --------- -------- -------- ---------"

    for t in $THREAD_COUNTS; do
        thr=$(med_file "$THREADS_DIR/raw/${THREADS_TAG}_n${n}_nz${nz}_t${t}.times")
        pool=$(med_file "$POOL_DIR/raw/${POOL_TAG}_n${n}_nz${nz}_t${t}.times")
        echo "$t $thr $pool"
    done | awk -v sq="$seqmed" '
        function f2(x){ return (x=="NA")?"NA":sprintf("%.2f",x) }
        function f4(x){ return (x=="NA")?"NA":sprintf("%.4f",x) }
        {
            t=$1; thr=$2; pool=$3;
            thsp=(thr!="NA"&&sq!="NA"&&thr>0)?sq/thr:"NA";
            posp=(pool!="NA"&&sq!="NA"&&pool>0)?sq/pool:"NA";
            thef=(thsp!="NA")?thsp/t:"NA";
            poef=(posp!="NA")?posp/t:"NA";
            ratio=(thr!="NA"&&pool!="NA"&&thr>0)?pool/thr:"NA";
            printf "%-8s %-9s %-9s %-9s %-9s %-8s %-8s %-9s\n", \
                   t, f4(thr), f4(pool), f2(thsp), f2(posp), f2(thef), f2(poef), \
                   (ratio=="NA"?"NA":sprintf("%.2fx",ratio));
            if (ratio!="NA" && ratio>0) { s+=log(ratio); k++; if(ratio>1)slow++; else if(ratio<1)fast++ }
        }
        END {
            if (k>0) {
                printf "\nVerdict over %d thread counts: pool slower in %d, faster in %d.\n", k, slow+0, fast+0;
                printf "  geometric-mean pool/threads = %.2fx  (>1 => pool slower on average).\n", exp(s/k);
            }
        }'

    # Checksum cross-check (threads vs pool vs seq) at each thread count.
    echo ""
    echo "checksums (threads vs pool; both expected to differ from seq):"
    for t in $THREAD_COUNTS; do
        tcs=$(cat "$THREADS_DIR/raw/${THREADS_TAG}_n${n}_nz${nz}_t${t}.cksum" 2>/dev/null)
        pcs=$(cat "$POOL_DIR/raw/${POOL_TAG}_n${n}_nz${nz}_t${t}.cksum" 2>/dev/null)
        rel="DIFF (ok: different reduction order)"; [ -n "$tcs" ] && [ "$tcs" = "$pcs" ] && rel="match"
        printf "  t=%-3s threads=%-20s pool=%-20s %s\n" "$t" "${tcs:-NA}" "${pcs:-NA}" "$rel"
    done
    scs=""; for d in $SEQ_DIRS; do scs=$(cat "$d/raw/seq_n${n}_nz${nz}.cksum" 2>/dev/null); [ -n "$scs" ] && break; done
    echo "  seq: ${scs:-NA}  (expected to differ from both threaded versions)"
  done
done

echo ""
echo "Legend:"
echo "  thr T / pool T  = median computation time (s) of the threads / pool version"
echo "  Sp = Speedup = T_seq(combined) / T_version(p)      (>1 faster than sequential)"
echo "  E  = Efficiency = Speedup / p                       (1.00 = perfect scaling)"
echo "  pool/thr = pool_time / threads_time at that p       (>1.00x => pool slower)"
echo "  Sequential baseline is the median over ALL pooled reps from $SEQ_DIRS,"
echo "  which dampens the run-to-run fluctuation of the (twice-measured) sequential."
} | tee "$OUT"

echo ""
echo "Comparison written to $OUT"
