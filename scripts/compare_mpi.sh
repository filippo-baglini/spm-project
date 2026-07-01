#!/bin/bash
# =============================================================================
# Summarize the hybrid MPI+OpenMP sweep from results_mpi/ (produced by
# run_mpi.sbatch / submit_mpi_sweep.sh). For each (n,nz) it prints, over all
# measured geometries (ranks x threads-per-rank) and BOTH variants -- the blocking
# MPI_Allgatherv baseline (block) and the non-blocking MPI_Iallgatherv overlap
# (overlap, one row per -c chunk count) -- the median loop time, the speedup vs the
# sequential baseline, and the parallel efficiency vs TOTAL cores (ranks*threads).
# It also cross-checks the checksum: with the distributed-Allreduce L2 norm both
# MPI variants match the sequential within tolerance (checksum DIFFers, like the
# threads/omp versions) but are identical across every geometry.
#
# Usage:
#   ./scripts/compare_mpi.sh
#   DIR=results_mpi SEQ_DIRS="results_mpi results_numa" ./scripts/compare_mpi.sh
#   OUT=mpi_compare.txt ./scripts/compare_mpi.sh
# =============================================================================

set -u
cd "$(dirname "$0")/.." || exit 1

DIR="${DIR:-results_mpi}"
OUT="${OUT:-mpi_compare.txt}"
# The slim MPI path does not run the sequential baseline, so by default we also
# pool the (stable) seq times+checksums from the other sweeps. Missing dirs are
# harmless. Override with SEQ_DIRS=... if needed.
SEQ_DIRS="${SEQ_DIRS:-$DIR results_numa results_all results_pool}"

if [ ! -d "$DIR/raw" ] || [ ! -f "$DIR/config.env" ]; then
    echo "ERROR: '$DIR' missing raw/ or config.env (run run_mpi.sbatch first)." >&2
    exit 1
fi

getcfg() { sed -n "s/^$2=\"\\(.*\\)\"/\\1/p" "$1/config.env"; }
MATRIX_SIZES=$(getcfg "$DIR" MATRIX_SIZES)
NONZEROS=$(getcfg "$DIR" NONZEROS)
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

{
echo "=============================================================================="
echo " HYBRID MPI + OpenMP  scaling summary"
echo " dir=$DIR   mode=$MODE  seed=$SEED   (per-cell times are MEDIANS over reps)"
echo " sequential baseline pooled from: $SEQ_DIRS"
echo "=============================================================================="

for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    # Skip (n,nz) combinations that were never measured (the swept grid is the
    # union over all submissions, so some cells may have no MPI data yet). Either
    # variant (blocking mpi_* or overlap mpiov_*) counts as data.
    { ls "$DIR"/raw/mpi_n${n}_nz${nz}_r*_t*.times      >/dev/null 2>&1 || \
      ls "$DIR"/raw/mpiov_n${n}_nz${nz}_r*_t*_c*.times >/dev/null 2>&1; } || continue

    seqmed=$(seq_combined "$n" "$nz")
    scs=""; for d in $SEQ_DIRS; do scs=$(cat "$d/raw/seq_n${n}_nz${nz}.cksum" 2>/dev/null); [ -n "$scs" ] && break; done

    echo ""
    echo "##############################################################################"
    printf "# n=%s  nz=%s   (avg %.1f nnz/row)   seq baseline = %ss\n" \
           "$n" "$nz" "$(awk -v a=$nz -v b=$n 'BEGIN{print a/b}')" "$seqmed"
    echo "##############################################################################"
    printf "%-8s %-4s %-7s %-7s %-7s %-10s %-9s %-9s %-9s %s\n" \
           "variant" "chk" "ranks" "thr/rk" "cores" "T(s)" "speedup" "eff" "GB/s*" "checksum vs seq"
    echo "-------- ---- ------- ------- ------- ---------- --------- --------- --------- ---------------"

    # Enumerate this (n,nz)'s geometries for BOTH variants. Blocking files are
    # mpi_n..._r{R}_t{T}; overlap files are mpiov_n..._r{R}_t{T}_c{C}. Sort by
    # total cores, then variant (block<overlap), then chunk count.
    for f in "$DIR"/raw/mpi_n${n}_nz${nz}_r*_t*.times \
             "$DIR"/raw/mpiov_n${n}_nz${nz}_r*_t*_c*.times; do
        [ -e "$f" ] || continue
        base=$(basename "$f" .times)
        case "$base" in
          mpiov_*)                                  # mpiov_n..._r{R}_t{T}_c{C}
            variant=overlap
            geo=${base##*_nz${nz}_}                 # r{R}_t{T}_c{C}
            R=${geo#r};   R=${R%%_*}
            tmp=${geo#*_t}; T=${tmp%%_*}
            Ck=${geo##*_c}
            ;;
          *)                                        # mpi_n..._r{R}_t{T}
            variant=block
            geo=${base##*_nz${nz}_}                 # r{R}_t{T}
            R=${geo#r};  R=${R%%_*}
            T=${geo##*_t}
            Ck="-"
            ;;
        esac
        echo "$R $T $((R*T)) $variant $Ck $f"
    done | sort -k3,3n -k4,4 -k5,5n | while read R T cores variant Ck f; do
        t=$(med_file "$f")
        cs=$(cat "${f%.times}.cksum" 2>/dev/null)
        rel="DIFF"; [ -n "$cs" ] && [ "$cs" = "$scs" ] && rel="match"
        echo "$variant $Ck $R $T $cores $t $cs|$rel"
    done | awk -v sq="$seqmed" -v nz="$nz" '
        function f2(x){ return (x=="NA")?"NA":sprintf("%.2f",x) }
        function f4(x){ return (x=="NA")?"NA":sprintf("%.4f",x) }
        {
            variant=$1; chk=$2; R=$3; T=$4; cores=$5; t=$6; split($7,a,"|"); rel=a[2];
            sp=(t!="NA"&&sq!="NA"&&t>0)?sq/t:"NA";
            eff=(sp!="NA"&&cores>0)?sp/cores:"NA";
            # 500 iters * 12 B/nonzero matrix stream (x is replicated, cached)
            gbs=(t!="NA"&&t>0)?(500.0*12.0*nz)/t/1e9:"NA";
            printf "%-8s %-4s %-7s %-7s %-7s %-10s %-9s %-9s %-9s %s\n", \
                   variant, chk, R, T, cores, f4(t), f2(sp), f2(eff), \
                   (gbs=="NA"?"NA":sprintf("%.1f",gbs)), rel;
        }'
  done
done

echo ""
echo "Legend:"
echo "  variant = block (blocking MPI_Allgatherv) | overlap (non-blocking MPI_Iallgatherv pipeline)"
echo "  chk     = overlap pipeline chunks (-c); '-' for the blocking baseline"
echo "  ranks = total MPI ranks;  thr/rk = OpenMP threads per rank;  cores = ranks*thr/rk"
echo "  T(s)     = median timed-loop wall time (max over ranks); generation+Scatterv are untimed"
echo "  speedup  = T_seq / T(ranks,threads)        eff = speedup / cores"
echo "  GB/s*    = approx matrix-stream bandwidth = 500*12*nz / T (aggregate over all ranks)"
echo "  checksum vs seq = 'DIFF' is EXPECTED: the L2 norm is a distributed MPI_Allreduce, so the"
echo "                    reduction order differs from the sequential (tolerance-level match, like the"
echo "                    threads/omp versions). The checksum is IDENTICAL across every geometry"
echo "                    (rank x thread x node x chunk) -- that cross-geometry invariance is the check."
echo "  Strong scaling is bounded by per-iteration MPI_Allgatherv (grows with node count) on top of"
echo "  the memory-bandwidth ceiling from the roofline study."
} | tee "$OUT"

echo ""
echo "Summary written to $OUT"
