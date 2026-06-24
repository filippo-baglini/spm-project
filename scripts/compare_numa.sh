#!/bin/bash
# =============================================================================
# NUMA-locality comparison across the three parallel models, from results
# produced by run_numa.sbatch (-> results_numa/). For each model it prints a
# baseline-vs-NUMA table:
#
#   cde   vs cde_numa   (static threads:   near-perfect locality)
#   pool  vs pool_numa  (dynamic pool:     cross-socket bandwidth spread)
#   omp   vs omp_numa   (OpenMP taskloop:  cross-socket bandwidth spread)
#
# Metrics (SPM NOTES): Speedup S=T_seq/T(p), Efficiency E=S/p, the direct ratio
# numa/base (<1 => NUMA faster), a geometric-mean verdict, and an APPROX
# matrix-stream bandwidth = ITERS*12*nnz / T (12 = 8 B value + 4 B col_idx per
# nonzero streamed from DRAM each iteration; x is L3-resident, excluded). That
# bandwidth should rise toward the STREAM peak as the stream is spread across
# both NUMA nodes.
#
# Usage:
#   ./scripts/compare_numa.sh
#   DIR=results_numa SEQ_DIRS="results_numa results_all" ./scripts/compare_numa.sh
#   ITERS=500 OUT=numa_compare.txt ./scripts/compare_numa.sh
# =============================================================================

set -u
cd "$(dirname "$0")/.." || exit 1

DIR="${DIR:-results_numa}"
ITERS="${ITERS:-500}"
OUT="${OUT:-numa_compare.txt}"
SEQ_DIRS="${SEQ_DIRS:-$DIR}"

# Model pairs:  baseline_tag:numa_tag:label
PAIRS="${PAIRS:-cde:cde_numa:STATIC_THREADS pool:pool_numa:DYNAMIC_POOL omp:omp_numa:OPENMP_TASKLOOP}"

if [ ! -d "$DIR/raw" ] || [ ! -f "$DIR/config.env" ]; then
    echo "ERROR: '$DIR' missing raw/ or config.env (run run_numa.sbatch first)." >&2
    exit 1
fi

getcfg() { sed -n "s/^$2=\"\\(.*\\)\"/\\1/p" "$1/config.env"; }
MATRIX_SIZES=$(getcfg "$DIR" MATRIX_SIZES)
NONZEROS=$(getcfg "$DIR" NONZEROS)
THREAD_COUNTS=$(getcfg "$DIR" THREAD_COUNTS)
MODE=$(getcfg "$DIR" MODE)
SEED=$(getcfg "$DIR" SEED)

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
seq_combined() {
    local n="$1" nz="$2" d f files=()
    for d in $SEQ_DIRS; do f="$d/raw/seq_n${n}_nz${nz}.times"; [ -f "$f" ] && files+=("$f"); done
    [ ${#files[@]} -eq 0 ] && { echo "NA"; return; }
    cat "${files[@]}" | median
}

# Print one baseline-vs-NUMA table for a model at (n,nz). Args: base numa label seqmed n nz
one_pair() {
    local base="$1" numa="$2" label="$3" seqmed="$4" n="$5" nz="$6"
    echo ""
    echo "  -- $label : $base  vs  $numa --"
    printf "  %-5s %-9s %-9s %-7s %-7s %-7s %-8s %-9s\n" \
           "p" "base T" "numa T" "baseSp" "numaSp" "numaE" "GB/s*" "numa/base"
    echo "  ----- --------- --------- ------- ------- ------- -------- ---------"
    for t in $THREAD_COUNTS; do
        b=$(med_file "$DIR/raw/${base}_n${n}_nz${nz}_t${t}.times")
        m=$(med_file "$DIR/raw/${numa}_n${n}_nz${nz}_t${t}.times")
        echo "$t $b $m"
    done | awk -v sq="$seqmed" -v nz="$nz" -v iters="$ITERS" '
        function f2(x){ return (x=="NA")?"NA":sprintf("%.2f",x) }
        function f4(x){ return (x=="NA")?"NA":sprintf("%.4f",x) }
        {
            t=$1; b=$2; m=$3;
            bsp=(b!="NA"&&sq!="NA"&&b>0)?sq/b:"NA";
            msp=(m!="NA"&&sq!="NA"&&m>0)?sq/m:"NA";
            me=(msp!="NA")?msp/t:"NA";
            gbs=(m!="NA"&&m>0)?(iters*12.0*nz)/m/1e9:"NA";
            r=(b!="NA"&&m!="NA"&&b>0)?m/b:"NA";
            printf "  %-5s %-9s %-9s %-7s %-7s %-7s %-8s %-9s\n", \
                   t, f4(b), f4(m), f2(bsp), f2(msp), f2(me), \
                   (gbs=="NA"?"NA":sprintf("%.1f",gbs)), \
                   (r=="NA"?"NA":sprintf("%.2fx",r));
            if (r!="NA"&&r>0){ s+=log(r); k++; if(r<1)fast++; else if(r>1)slow++ }
        }
        END {
            if (k>0)
                printf "  geomean numa/base = %.2fx  (faster in %d/%d, <1 => NUMA wins)\n", \
                       exp(s/k), fast+0, k;
        }'
}

{
echo "=============================================================================="
echo " NUMA-LOCALITY comparison across the three parallel models"
echo " dir=$DIR   mode=$MODE  seed=$SEED   (per-cell times are MEDIANS over reps)"
echo " sequential baseline pooled from: $SEQ_DIRS    ITERS=$ITERS (for GB/s est.)"
echo "=============================================================================="

for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    seqmed=$(seq_combined "$n" "$nz")
    nsamp=0; for d in $SEQ_DIRS; do f="$d/raw/seq_n${n}_nz${nz}.times"; [ -f "$f" ] && nsamp=$((nsamp + $(grep -c '[0-9]' "$f"))); done

    echo ""
    echo "##############################################################################"
    printf "# n=%s  nz=%s   (avg %.1f nnz/row)\n" "$n" "$nz" "$(awk -v a=$nz -v b=$n 'BEGIN{print a/b}')"
    echo "##############################################################################"
    printf "Sequential baseline (combined median) = %ss   [pooled from %s reps]\n" "$seqmed" "$nsamp"

    for spec in $PAIRS; do
        base="${spec%%:*}"; rest="${spec#*:}"; numa="${rest%%:*}"; label="${rest#*:}"
        one_pair "$base" "$numa" "$label" "$seqmed" "$n" "$nz"
    done

    # Checksum cross-check at the largest thread count present: base vs its numa
    # variant (threads/pool should MATCH bit-for-bit; omp may differ by taskloop
    # reduction order — both fine).
    ckt=$(echo "$THREAD_COUNTS" | tr ' ' '\n' | sort -n | tail -1)
    echo ""
    echo "  checksum base-vs-numa at t=$ckt (threads/pool match exactly; omp may differ by reduction order):"
    for spec in $PAIRS; do
        base="${spec%%:*}"; rest="${spec#*:}"; numa="${rest%%:*}"
        bc=$(cat "$DIR/raw/${base}_n${n}_nz${nz}_t${ckt}.cksum" 2>/dev/null)
        mc=$(cat "$DIR/raw/${numa}_n${n}_nz${nz}_t${ckt}.cksum" 2>/dev/null)
        rel="DIFF (reduction order — fine)"; [ -n "$bc" ] && [ "$bc" = "$mc" ] && rel="match"
        printf "    t=%-3s %-10s=%-18s %-10s=%-18s %s\n" "$ckt" "$base" "${bc:-NA}" "$numa" "${mc:-NA}" "$rel"
    done
  done
done

echo ""
echo "Legend:"
echo "  base T / numa T = median time (s) of the baseline / NUMA variant of that model"
echo "  baseSp/numaSp = Speedup = T_seq(combined)/T(p);  numaE = numaSp/p (1.00 = perfect)"
echo "  GB/s* = APPROX matrix-stream bandwidth of the NUMA variant = ITERS*12*nnz / T"
echo "          (x is L3-resident, excluded); rises as the stream is spread across both sockets"
echo "  numa/base = time ratio at that p (<1.00x => NUMA faster than that model's baseline)"
echo "  Static threads get near-perfect locality; pool/omp are dynamically scheduled, so they get"
echo "  the cross-socket bandwidth spread but not per-worker locality (=> typically a smaller win)."
} | tee "$OUT"

echo ""
echo "Comparison written to $OUT"
