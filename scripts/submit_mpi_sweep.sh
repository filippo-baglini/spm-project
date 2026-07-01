#!/bin/bash
# =============================================================================
# Submit a SLIM hybrid MPI+OpenMP strong-scaling sweep.
#
# One short job PER NODE COUNT (not per geometry), each based on the lightweight
# scripts/run_mpi_quick.sbatch:
#   * NO sequential baseline (reuse the already-stable seq times in results_numa);
#   * short wall time (default 10 min, not 30);
#   * each job packs that node count's geometries (all SPLITS) into a single
#     allocation via mpirun, so it only holds `nodes` nodes for ~minutes.
#
# Across the node counts this still gives full strong-scaling curves: e.g. split
# 4:4 yields ranks 4,8,16,32 over nodes 1,2,4,8.
#
# Usage (on the cluster login node, from the repo root):
#   ./scripts/submit_mpi_sweep.sh                  # nodes 1,2,4,8 ; splits 4:4 & 16:1
#   NODES_LIST="1 2 4" ./scripts/submit_mpi_sweep.sh        # avoid the 8-node job
#   SPLITS="4:4" MATRIX_SIZES=500000 NONZEROS=20000000 ./scripts/submit_mpi_sweep.sh
#   VARIANTS="overlap" CHUNKS="1 2 4 8" ./scripts/submit_mpi_sweep.sh   # tune -c only
#   DRYRUN=1 ./scripts/submit_mpi_sweep.sh         # print the sbatch lines only
#
# Each job times BOTH MPI variants by default: the blocking MPI_Allgatherv baseline
# (results keyed mpi_*) and the non-blocking overlap (mpiov_*, one per -c in CHUNKS).
#
# Then summarize (seq baseline pooled from the NUMA run):
#   DIR=results_mpi SEQ_DIRS="results_mpi results_numa" ./scripts/compare_mpi.sh
# =============================================================================

set -u
cd "$(dirname "$0")/.." || exit 1

NODES_LIST="${NODES_LIST:-1 2 4 8}"
SPLITS="${SPLITS:-4:4 16:1}"            # ranks-per-node:threads-per-rank (product <= 16 phys)
MATRIX_SIZES="${MATRIX_SIZES:-100000}"  # one small/fast size by default
NONZEROS="${NONZEROS:-4000000}"
REPS="${REPS:-2}"
TIME="${TIME:-00:10:00}"
VARIANTS="${VARIANTS:-block overlap}"   # time both MPI variants by default
CHUNKS="${CHUNKS:-4}"                    # -c list for the overlap variant
RESULTS_DIR="${RESULTS_DIR:-results_mpi}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/run_mpi_quick.sbatch}"

# Exported so each quick job (submitted with --export=ALL) inherits them.
export MATRIX_SIZES NONZEROS REPS VARIANTS CHUNKS RESULTS_DIR

for nodes in $NODES_LIST; do
    cfgs=""
    for split in $SPLITS; do
        rpn="${split%%:*}"; tpr="${split##*:}"
        ranks=$(( nodes * rpn ))
        cfgs="$cfgs ${ranks}:${rpn}:${tpr}"
    done
    cfgs="${cfgs# }"
    export CONFIGS="$cfgs"
    echo "+ sbatch --nodes=$nodes --time=$TIME --export=ALL $SBATCH_SCRIPT   (CONFIGS=\"$cfgs\")"
    if [ "${DRYRUN:-0}" != "1" ]; then
        sbatch --nodes="$nodes" --time="$TIME" --export=ALL "$SBATCH_SCRIPT"
    fi
done

echo ""
echo "Submitted (or previewed) the slim sweep: $(echo $NODES_LIST | wc -w) short jobs, no sequential run."
echo "After they finish:  DIR=$RESULTS_DIR SEQ_DIRS=\"$RESULTS_DIR results_numa\" ./scripts/compare_mpi.sh"
