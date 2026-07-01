#!/bin/bash
# =============================================================================
# Verify the MPI versions' Rayleigh quotient matches the SEQUENTIAL reference
# within tolerance.
#
# The sweep scripts only persist time + checksum; the `rayleigh=` line the
# binaries print is not captured. The MPI result is invariant across geometries
# (the checksum is identical for every rank x thread x node x chunk -- see
# compare_mpi.sh), so ONE MPI run per matrix is enough: this runs the sequential
# reference and the MPI binary (block, and overlap if present) on the same
# (n, nz, seed), extracts both `rayleigh=` values, and prints the absolute and
# RELATIVE difference, flagging PASS/FAIL against TOL.
#
# Binaries: reuses already-built ones if found, else builds into $BUILD
# (seq with g++, mpi/mpi_overlap with mpicxx). Matrix generation is untimed and
# irrelevant here -- we only read the printed scalar.
#
# Usage (cluster, repo root):
#   ./scripts/check_mpi_rayleigh.sh
#   NP=8 TPR=2 ./scripts/check_mpi_rayleigh.sh                 # different geometry
#   PAIRS="500000:4000000" TOL=1e-11 ./scripts/check_mpi_rayleigh.sh
#   MAP="--map-by ppr:4:node:pe=4 --bind-to core" ./scripts/check_mpi_rayleigh.sh
#
# Run on a COMPUTE NODE (so nothing touches the login node) via the wrapper:
#   sbatch scripts/check_mpi_rayleigh.sbatch        # geometry from the allocation
# Run bare on the login node ONLY for a tiny matrix: PAIRS="5000:20000" ./scripts/check_mpi_rayleigh.sh
# =============================================================================

set -u
cd "$(dirname "$0")/.." || exit 1

PAIRS="${PAIRS:-100000:4000000 100000:20000000 500000:4000000 500000:20000000}"
SEED="${SEED:-111}"
MODE="${MODE:-irregular}"
NP="${NP:-4}"            # MPI ranks for the check
TPR="${TPR:-4}"         # OpenMP threads per rank
CHUNKS="${CHUNKS:-4}"   # overlap pipeline depth (-c)
TOL="${TOL:-1e-9}"      # RELATIVE tolerance for |rayleigh_mpi - rayleigh_seq|
MPIRUN="${MPIRUN:-mpirun}"
MAP="${MAP:-}"          # extra mpirun flags (binding); empty = launcher default
BUILD="${BUILD:-/tmp/rayleigh_check_build}"
CXX="${CXX:-g++}"
MPICXX="${MPICXX:-mpicxx}"
FLAGS="-O3 -march=native -funroll-loops -DNDEBUG -std=c++20 -I include"

mkdir -p "$BUILD"

build_seq() { $CXX $FLAGS src/sequential/iterative_SpMV.cpp -o "$BUILD/seq_bin"; }
build_mpi() { $MPICXX $FLAGS -fopenmp src/mpi/iterative_SpMV_mpi.cpp -o "$BUILD/mpi_bin"; }
build_ovl() { $MPICXX $FLAGS -fopenmp src/mpi/iterative_SpMV_mpi_overlap.cpp -o "$BUILD/ovl_bin"; }

# seq: ALWAYS build fresh from the current source. A leftover bin/seq (or a
# sweep-built one) may predate the `rayleigh=` print line and emit nothing to
# compare; building fresh also guarantees seq and the MPI binaries come from the
# same source generation. The compile is trivial (a few seconds).
build_seq || { echo "ERROR: failed to build seq from src/sequential/iterative_SpMV.cpp" >&2; exit 1; }
SEQ_BIN="$BUILD/seq_bin"

# mpi block + overlap: prefer the sweep's build dir, else build here.
MPI_BIN=""
for p in results_mpi_allreduce/build/mpi results_mpi/build/mpi bin/mpi "$BUILD/mpi_bin"; do
    [ -x "$p" ] && { MPI_BIN="$p"; break; }
done
[ -z "$MPI_BIN" ] && { build_mpi && MPI_BIN="$BUILD/mpi_bin" || exit 1; }

OVL_BIN=""
for p in results_mpi_allreduce/build/mpi_overlap results_mpi/build/mpi_overlap bin/mpi_overlap "$BUILD/ovl_bin"; do
    [ -x "$p" ] && { OVL_BIN="$p"; break; }
done
[ -z "$OVL_BIN" ] && { build_ovl && OVL_BIN="$BUILD/ovl_bin"; }   # optional; ok if it fails

# Extract the `rayleigh=` scalar from a command's stdout (NA if absent).
get_rayleigh() { "$@" 2>/dev/null | sed -n 's/^rayleigh=//p' | head -1; }

echo "=============================================================================="
echo " MPI Rayleigh-quotient correctness check  (relative tolerance TOL=$TOL)"
echo " geometry: -np $NP, -t $TPR  (overlap -c $CHUNKS)   mode=$MODE seed=$SEED"
echo " seq=$SEQ_BIN"
echo " mpi=$MPI_BIN"
[ -n "$OVL_BIN" ] && echo " ovl=$OVL_BIN"
echo "=============================================================================="
printf "%-9s %-9s %-10s %-24s %-24s %-11s %-6s\n" \
       "n" "nz" "variant" "rayleigh_seq" "rayleigh_mpi" "rel.diff" "verdict"
echo "------------------------------------------------------------------------------------------------------------"

fail=0
for pair in $PAIRS; do
    n="${pair%%:*}"; nz="${pair##*:}"

    r_seq=$(get_rayleigh "$SEQ_BIN" -n "$n" -nz "$nz" -m "$MODE" -s "$SEED")
    [ -z "$r_seq" ] && { printf "%-9s %-9s  seq run produced no rayleigh -- skipping\n" "$n" "$nz"; fail=1; continue; }

    # block, then overlap (if available). Same (n,nz,seed); geometry from env.
    declare -a names=("block") bins=("$MPI_BIN") extra=("")
    if [ -n "$OVL_BIN" ]; then names+=("overlap"); bins+=("$OVL_BIN"); extra+=("-c $CHUNKS"); fi

    for i in "${!names[@]}"; do
        # shellcheck disable=SC2086
        r_mpi=$(get_rayleigh $MPIRUN -np "$NP" $MAP "${bins[$i]}" -n "$n" -nz "$nz" -m "$MODE" -s "$SEED" -t "$TPR" ${extra[$i]})
        if [ -z "$r_mpi" ]; then
            printf "%-9s %-9s %-10s %-24s %-24s %-11s %-6s\n" "$n" "$nz" "${names[$i]}" "$r_seq" "NA" "-" "RUNERR"
            fail=1; continue
        fi
        read rel verdict < <(awk -v a="$r_seq" -v b="$r_mpi" -v tol="$TOL" 'BEGIN{
            d = a-b; if (d<0) d=-d;
            den = (a<0?-a:a); if (den<1e-300) den=1e-300;
            rel = d/den;
            printf "%.3e %s", rel, (rel<=tol ? "PASS" : "FAIL");
        }')
        [ "$verdict" = "FAIL" ] && fail=1
        printf "%-9s %-9s %-10s %-24s %-24s %-11s %-6s\n" "$n" "$nz" "${names[$i]}" "$r_seq" "$r_mpi" "$rel" "$verdict"
    done
done

echo "------------------------------------------------------------------------------------------------------------"
if [ "$fail" -eq 0 ]; then
    echo "ALL within tolerance (rel.diff <= $TOL). The distributed-Allreduce norm is correct."
else
    echo "Some checks FAILED or could not run -- see rows above."
fi
exit "$fail"
