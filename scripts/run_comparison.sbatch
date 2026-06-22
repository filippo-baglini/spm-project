#!/bin/bash
#SBATCH --job-name=spmv_comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results_%j.log
#SBATCH --error=errors_%j.log

# Comparison benchmark: Sequential vs Threaded SpMV
# Tests on irregular workload with various matrix and thread configurations

set -e

# Run from the project root so src/... paths and the output dir resolve.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Compile options
CXXFLAGS="-O3 -std=c++20 -Wall -pthread"
INCLUDES="-I include"

echo "=== Building Sequential Version ==="
g++ $CXXFLAGS $INCLUDES -o seq_spmv src/sequential/iterative_SpMV.cpp
echo "Sequential binary: seq_spmv"

echo ""
echo "=== Building Threaded Version ==="
g++ $CXXFLAGS $INCLUDES -o thread_spmv src/threads/iterative_SpMV_threads.cpp
echo "Threaded binary: thread_spmv"

echo ""
echo "=== Test Configuration ==="
MATRIX_SIZES="100000 500000"
NONZEROS="4000000 20000000"
MODE="irregular"
SEED="111"

echo "Matrix sizes: $MATRIX_SIZES"
echo "Nonzero counts: $NONZEROS"
echo "Mode: $MODE"
echo "Seed: $SEED"
echo ""

# Create output directory for results
mkdir -p results

echo "=== Running Tests ==="
echo ""

# Sequential baseline runs
echo "--- Sequential Baseline Tests ---"
for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    echo "Testing: n=$n, nz=$nz"
    output_file="results/seq_n${n}_nz${nz}.txt"
    echo "Sequential run: n=$n, nz=$nz" > "$output_file"
    ./seq_spmv -n $n -nz $nz -m $MODE -s $SEED --dump-vector results/seq_vec_n${n}_nz${nz}.dump >> "$output_file" 2>&1 || echo "Sequential test failed"
    echo "" >> "$output_file"
  done
done

echo ""
echo "--- Threaded Tests ---"
# Test with different thread counts
THREAD_COUNTS="1 2 4 8 16 32"

for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    for t in $THREAD_COUNTS; do
      # Skip thread counts greater than available cores
      if [ $t -le 32 ]; then
        echo "Testing: n=$n, nz=$nz, threads=$t"
        output_file="results/thread_n${n}_nz${nz}_t${t}.txt"
        echo "Threaded run: n=$n, nz=$nz, threads=$t" > "$output_file"
        ./thread_spmv -n $n -nz $nz -m $MODE -s $SEED -t $t --dump-vector results/thread_vec_n${n}_nz${nz}_t${t}.dump >> "$output_file" 2>&1 || echo "Threaded test failed"
        echo "" >> "$output_file"
      fi
    done
  done
done

echo ""
echo "=== Correctness Verification ==="

for n in $MATRIX_SIZES; do
  for nz in $NONZEROS; do
    echo "Comparing sequential vs threaded (n=$n, nz=$nz):"
    seq_file="results/seq_n${n}_nz${nz}.txt"
    thread_1_file="results/thread_n${n}_nz${nz}_t1.txt"
    
    if [ -f "$seq_file" ] && [ -f "$thread_1_file" ]; then
      seq_checksum=$(grep "checksum=" "$seq_file" | head -1)
      seq_rayleigh=$(grep "rayleigh=" "$seq_file" | head -1)
      thread_checksum=$(grep "checksum=" "$thread_1_file" | head -1)
      thread_rayleigh=$(grep "rayleigh=" "$thread_1_file" | head -1)
      
      echo "  Sequential: $seq_checksum"
      echo "  Threaded(t=1): $thread_checksum"
      if [ "$seq_checksum" = "$thread_checksum" ]; then
        echo "  ✓ Checksums match"
      else
        echo "  ✗ Checksums differ (possible correctness issue)"
      fi
      
      echo "  Sequential: $seq_rayleigh"
      echo "  Threaded(t=1): $thread_rayleigh"
      echo ""
    fi
  done
done

echo "=== Summary ==="
echo "All results saved in results/ directory"
echo "Test log: $SLURM_JOBID"
echo ""
echo "To analyze results:"
echo "  grep 'Time (sec)' results/*.txt"
echo "  grep 'checksum=' results/*.txt"
echo "  grep 'rayleigh=' results/*.txt"
