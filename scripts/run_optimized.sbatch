#!/bin/bash
#SBATCH --job-name=spmv_optimized
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results_optimized_%j.log
#SBATCH --error=errors_optimized_%j.log

# Benchmark the OPTIMIZED threaded SpMV (false-sharing padding + std::barrier).
#
# Writes all output into a SEPARATE directory (results_optimized/) so the
# baseline results/ folder is left untouched and the two runs can be compared
# afterwards with ./compare_runs.sh results results_optimized
#
# Same matrix/thread configurations and seed as run_comparison.sbatch, so the
# numbers are directly comparable to the baseline.

set -e

# Run from the project root so src/... paths and the output dir resolve.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Output directory for this (new) run. Change here if you want another name.
RESULTS_DIR="results_optimized"

# Compile options (match the Makefile / baseline run).
CXXFLAGS="-O3 -march=native -funroll-loops -std=c++20 -Wall -Wextra -pthread"
INCLUDES="-I include"

echo "=== Building Sequential Reference ==="
g++ $CXXFLAGS $INCLUDES -o seq_spmv src/sequential/iterative_SpMV.cpp
echo "Sequential binary: seq_spmv"

echo ""
echo "=== Building Optimized Threaded Version ==="
g++ $CXXFLAGS $INCLUDES -o thread_spmv src/threads/iterative_SpMV_threads.cpp
echo "Threaded binary: thread_spmv"

echo ""
echo "=== Test Configuration ==="
MATRIX_SIZES="100000 500000"
NONZEROS="4000000 20000000"
MODE="irregular"
SEED="111"
THREAD_COUNTS="1 2 4 8 16 32"

echo "Matrix sizes:   $MATRIX_SIZES"
echo "Nonzero counts: $NONZEROS"
echo "Mode:           $MODE"
echo "Seed:           $SEED"
echo "Thread counts:  $THREAD_COUNTS"
echo "Output dir:     $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

echo "=== Running Sequential Baseline ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    echo "Sequential: n=$n, nz=$nz"
    out="$RESULTS_DIR/seq_n${n}_nz${nz}.txt"
    echo "Sequential run: n=$n, nz=$nz" > "$out"
    ./seq_spmv -n $n -nz $nz -m $MODE -s $SEED \
      --dump-vector "$RESULTS_DIR/seq_vec_n${n}_nz${nz}.dump" >> "$out" 2>&1 \
      || echo "Sequential test failed"
    echo "" >> "$out"
  done
done

echo ""
echo "=== Running Optimized Threaded Tests ==="
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    for t in $THREAD_COUNTS; do
      echo "Threaded: n=$n, nz=$nz, threads=$t"
      out="$RESULTS_DIR/thread_n${n}_nz${nz}_t${t}.txt"
      echo "Threaded run: n=$n, nz=$nz, threads=$t" > "$out"
      ./thread_spmv -n $n -nz $nz -m $MODE -s $SEED -t $t \
        --dump-vector "$RESULTS_DIR/thread_vec_n${n}_nz${nz}_t${t}.dump" >> "$out" 2>&1 \
        || echo "Threaded test failed"
      echo "" >> "$out"
    done
  done
done

echo ""
echo "=== Done ==="
echo "New results saved in: $RESULTS_DIR/"
echo "Baseline results untouched in: results/"
echo ""
echo "Compare the two runs with:"
echo "  ./compare_runs.sh results $RESULTS_DIR"
