#!/bin/bash
# =============================================================================
# COMPREHENSIVE hybrid MPI+OpenMP sweep -- every meaningful configuration of BOTH
# variants, sized to the cluster topology (2 sockets x 8 cores = 16 phys/node).
#
# Axes swept:
#   * nodes        : 1, 2, 4, 8                              (strong scaling)
#   * geometry     : the full rank/thread spectrum per node, product = 16 phys:
#                      1:16  one fat rank/node (straddles both NUMA domains)
#                      2:8   one rank per SOCKET  <-- NUMA-optimal
#                      4:4   two ranks per socket
#                      8:2   four ranks per socket
#                      16:1  flat MPI (one rank per core)
#                    (rpn:tpr; total ranks = nodes * rpn)
#   * variant      : block (blocking MPI_Allgatherv) AND overlap (MPI_Iallgatherv)
#   * chunks (-c)  : 1, 2, 4 for the overlap variant (1 == no-overlap control)
#   * matrix       : all four project sizes (n:nz pairs below) -- the two 40-nnz/row
#                    canonical sizes plus the 200- and 8-nnz/row density extremes
#
# It submits ONE job per (node-count, matrix-pair) -- 4 nodes x 4 pairs = 16 jobs
# -- each packing that cell's whole geometry x variant x chunk grid into a single
# allocation via run_mpi_quick.sbatch. NO sequential baseline (pooled from
# results_numa by compare_mpi.sh). Jobs are RESUMABLE: re-running this script
# skips already-complete cells, and each job stops cleanly at TIME_BUDGET_SEC
# (just under the wall limit) so nothing is lost if it runs long.
#
# This is the heavy counterpart of submit_mpi_sweep.sh; expect the 8-node jobs to
# hold all 8 nodes for up to TIME. Trim with the env vars below if needed.
#
# Usage (cluster login node, repo root):
#   ./scripts/submit_mpi_full.sh                      # the full grid
#   NODES_LIST="1 2 4" ./scripts/submit_mpi_full.sh   # skip the 8-node jobs
#   SPLITS="2:8 4:4" ./scripts/submit_mpi_full.sh     # only the NUMA-friendly hybrids
#   VARIANTS=block ./scripts/submit_mpi_full.sh       # baseline only (no overlap)
#   PAIRS="500000:20000000" ./scripts/submit_mpi_full.sh   # heavy matrix only
#   DRYRUN=1 ./scripts/submit_mpi_full.sh             # print the sbatch lines only
#
# Then summarize (seq baseline pooled from the NUMA run):
#   ./scripts/compare_mpi.sh
# =============================================================================

set -u
cd "$(dirname "$0")/.." || exit 1

NODES_LIST="${NODES_LIST:-1 2 4 8}"
SPLITS="${SPLITS:-1:16 2:8 4:4 8:2 16:1}"          # rpn:tpr, product = 16 phys cores/node
# All four matrices the rest of the project sweeps (the shared-memory analyzers'
# MATRIX_SIZES x NONZEROS grid): the two 40-nnz/row canonical sizes plus the two
# density extremes -- 100000:20000000 = 200 nnz/row (dense rows, tiny x: MPI's
# best case) and 500000:4000000 = 8 nnz/row (sparse, big x: most comm-bound).
PAIRS="${PAIRS:-100000:4000000 100000:20000000 500000:4000000 500000:20000000}"   # n:nz
VARIANTS="${VARIANTS:-block overlap}"
CHUNKS="${CHUNKS:-1 2 4}"                           # overlap pipeline depth (1 = control)
REPS="${REPS:-3}"
MODE="${MODE:-irregular}"
SEED="${SEED:-111}"
TIME="${TIME:-00:30:00}"
TIME_BUDGET_SEC="${TIME_BUDGET_SEC:-2100}"          # stop ~3 min before the 40-min wall
RESULTS_DIR="${RESULTS_DIR:-results_mpi}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/run_mpi_quick.sbatch}"

# Each quick job inherits these (the driver owns config.env, so WRITE_CONFIG=0).
export VARIANTS CHUNKS REPS MODE SEED RESULTS_DIR TIME_BUDGET_SEC
export WRITE_CONFIG=0

# ---- Pre-seed an authoritative config.env (union of the whole grid) so the
#      concurrent jobs never race on it. ----
ALL_N=""; ALL_NZ=""
for pair in $PAIRS; do ALL_N="$ALL_N ${pair%%:*}"; ALL_NZ="$ALL_NZ ${pair##*:}"; done
ALL_GEO=""
for nodes in $NODES_LIST; do
    for split in $SPLITS; do
        rpn="${split%%:*}"; tpr="${split##*:}"
        ALL_GEO="$ALL_GEO r$(( nodes * rpn ))_t${tpr}"
    done
done
uniq_sorted() { echo "$1" | tr ' ' '\n' | grep -v '^$' | sort -n -u | tr '\n' ' ' | sed 's/ *$//'; }
uniq_geo()    { echo "$1" | tr ' ' '\n' | grep -v '^$' | sort   -u | tr '\n' ' ' | sed 's/ *$//'; }
mkdir -p "$RESULTS_DIR/raw"
if [ "${DRYRUN:-0}" != "1" ]; then
    {
        echo "MATRIX_SIZES=\"$(uniq_sorted "$ALL_N")\""
        echo "NONZEROS=\"$(uniq_sorted "$ALL_NZ")\""
        echo "GEOMETRIES=\"$(uniq_geo "$ALL_GEO")\""
        echo "MODE=\"$MODE\""
        echo "SEED=\"$SEED\""
        echo "REPS=\"$REPS\""
        echo "VARIANTS=\"$VARIANTS\""
        echo "CHUNKS=\"$CHUNKS\""
    } > "$RESULTS_DIR/config.env"
fi

# ---- Submit one job per (node-count, matrix-pair). ----
njobs=0
for nodes in $NODES_LIST; do
    cfgs=""
    for split in $SPLITS; do
        rpn="${split%%:*}"; tpr="${split##*:}"
        cfgs="$cfgs $(( nodes * rpn )):${rpn}:${tpr}"
    done
    cfgs="${cfgs# }"
    for pair in $PAIRS; do
        n="${pair%%:*}"; nz="${pair##*:}"
        export CONFIGS="$cfgs" MATRIX_SIZES="$n" NONZEROS="$nz"
        echo "+ sbatch --nodes=$nodes --time=$TIME  n=$n nz=$nz  CONFIGS=\"$cfgs\""
        if [ "${DRYRUN:-0}" != "1" ]; then
            sbatch --nodes="$nodes" --time="$TIME" --export=ALL "$SBATCH_SCRIPT"
        fi
        njobs=$(( njobs + 1 ))
    done
done

echo ""
echo "Comprehensive sweep: $njobs jobs (no sequential run; resumable -- re-run to fill gaps)."
echo "  nodes=[$NODES_LIST]  splits=[$SPLITS]  variants=[$VARIANTS]  chunks=[$CHUNKS]"
echo "  pairs=[$PAIRS]  reps=$REPS  time=$TIME/job"
echo "After they finish:  ./scripts/compare_mpi.sh"
