#!/bin/bash
#SBATCH --job-name=spmv_rayleigh_check
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=rayleigh_check_%j.log
#SBATCH --error=rayleigh_check_%j.log

# =============================================================================
# Compute-node wrapper for scripts/check_mpi_rayleigh.sh, so NOTHING runs on the
# login node: the batch body executes on the allocated node, so the sequential
# reference AND the mpirun ranks all land on compute hardware.
#
# The MPI geometry is derived FROM THE ALLOCATION (SLURM_NTASKS ranks x
# SLURM_CPUS_PER_TASK threads), so to change it just change the sbatch resources:
#   sbatch scripts/check_mpi_rayleigh.sbatch                          # 4 ranks x 4 threads (16 cores, 1 node)
#   sbatch --ntasks=2 --cpus-per-task=8 scripts/check_mpi_rayleigh.sbatch   # 2 ranks x 8 threads (NUMA-friendly)
#   sbatch --nodes=2 --ntasks=8 --cpus-per-task=4 scripts/check_mpi_rayleigh.sbatch  # 2 nodes
#
# Other knobs pass straight through the environment to the script:
#   PAIRS="500000:4000000" TOL=1e-11 sbatch scripts/check_mpi_rayleigh.sbatch
#   CHUNKS=8 sbatch scripts/check_mpi_rayleigh.sbatch
#
# Output (the PASS/FAIL table) goes to rayleigh_check_<jobid>.log.
# =============================================================================

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# Geometry from the allocation (overridable via a pre-set NP/TPR in the env).
export NP="${NP:-${SLURM_NTASKS:-4}}"
export TPR="${TPR:-${SLURM_CPUS_PER_TASK:-4}}"

echo "rayleigh check on $(hostname): NP=$NP ranks x TPR=$TPR threads  (nodes=${SLURM_NNODES:-1})"

# PAIRS / TOL / CHUNKS / MODE / SEED / MAP are read from the environment by the
# script (sbatch exports the submission env by default), so no need to re-pass them.
exec bash scripts/check_mpi_rayleigh.sh
