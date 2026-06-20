# SpMV Benchmark Testing Guide

## Overview
This directory contains tools to build and compare the sequential and threaded versions of the iterative sparse matrix-vector computation on evolving matrices.

**Analysis Tools:**
- `analyze_results.sh` - Bash script (recommended for HPC clusters, no Python required)
- `analyze_results.py` - Python script (for local/offline analysis if preferred)

## Files
- `run_comparison.sbatch` - Slurm batch script for running comprehensive tests
- `analyze_results.sh` - Bash script to parse and analyze benchmark results (no dependencies)
- `seq_spmv` - Sequential reference binary (built by sbatch script)
- `thread_spmv` - Threaded implementation binary (built by sbatch script)

## Quick Start

### 1. Submit the batch job
```bash
sbatch run_comparison.sbatch
```

This will:
- Compile both sequential and threaded versions
- Run tests with multiple matrix sizes and thread counts
- Save results in `results/` directory
- Verify correctness by comparing checksums

### 2. Monitor job status
```bash
squeue -u $USER
```

### 3. Analyze results
```bash
./analyze_results.sh
```

This generates:
- Correctness verification (checksum and Rayleigh quotient comparison)
- Performance metrics (execution time, speedup, efficiency)
- Organized by matrix configuration
- Uses only standard Unix tools (no Python required)

## Test Parameters

The `run_comparison.sbatch` tests the following configurations:

**Matrix Sizes:**
- n = 100000, 500000

**Nonzero Counts:**
- nz = 4000000, 20000000

**Thread Counts:**
- 1, 2, 4, 8, 16, 32

**Workload:**
- irregular mode (most demanding)
- seed = 111 (deterministic)

## Understanding Results

### Correctness
- Checksum and Rayleigh quotient must match between sequential and all threaded runs
- Mismatches indicate bugs in the threaded implementation

### Performance Metrics
- **Time (sec)**: Wall-clock execution time of the timed iterative loop
- **Speedup**: Sequential time / Threaded time
- **Efficiency**: Speedup / #threads × 100%

### Example Output
```
Configuration: n=500000, nz=20000000
===========================================

--- Correctness Verification ---
Sequential: checksum=0xabcd1234, rayleigh=2.500000000000000e+00
  Threads=1: checksum=0xabcd1234, rayleigh=2.500000000000000e+00 ✓
  Threads=4: checksum=0xabcd1234, rayleigh=2.500000000000000e+00 ✓

--- Performance Analysis ---
Sequential time: 12.5432 sec

Threads    Time (sec)       Speedup      Efficiency
------------------------------------------------
1          12.5421         1.00x        100.0%
4          3.2157          3.90x        97.5%
8          1.6523          7.59x        94.9%
```

## Customizing Tests

Edit `run_comparison.sbatch` to change:

**Matrix parameters:**
```bash
MATRIX_SIZES="100000 500000"
NONZEROS="4000000 20000000"
```

**Thread ranges:**
```bash
THREAD_COUNTS="1 2 4 8 16 32"
```

**Job resources:**
```bash
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
```

## Troubleshooting

### Binary not found
Ensure the build commands in `run_comparison.sbatch` match your source file names:
- Sequential: `iterative_SpMV.cpp`
- Threaded: `iterative_SpMV.cpp` with `-DUSE_THREADS` (or separate file)

### Checksum mismatches
Check that:
1. Both implementations use the same random seed
2. Matrix generation is identical
3. Thread synchronization is correct

### Poor scaling
Check for:
- Load imbalance in row distribution
- Synchronization bottlenecks
- Memory contention
- Insufficient work granularity

## Advanced Analysis

For manual inspection of specific runs:
```bash
# View raw output
cat results/seq_n500000_nz20000000.txt

# Extract timing data
grep "Time (sec)" results/*.txt

# Check correctness
grep "checksum=" results/*.txt

# Run analysis script again
./analyze_results.sh
```

## Notes
- Vector dump files (`.dump`) are generated but not included in timed measurements
- All tests use the irregular sparsity pattern (main workload)
- The sbatch script runs on a single node with up to 32 CPU cores
