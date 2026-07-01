# Makefile for the One-Shot project
#
#   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
#
# Layout:
#   include/        provided helpers (matrix_generation.hpp, utils.hpp)
#   src/sequential/ sequential reference
#   src/threads/    C++ threads implementations
#   src/openmp/     OpenMP-tasks implementation
#   bin/            build outputs (created here; gitignored)
#
# Targets:
#   make             build 'seq', 'threads', 'omp' and 'pool'
#   make seq         sequential reference                 -> bin/seq
#   make threads     C++ threads implementation           -> bin/threads
#   make omp         OpenMP-tasks implementation          -> bin/omp
#   make pool        threadpool comparison variant        -> bin/pool
#   make threads_numa NUMA-locality threads variant        -> bin/threads_numa
#   make pool_numa   NUMA-locality threadpool variant       -> bin/pool_numa
#   make omp_numa    NUMA-locality OpenMP variant           -> bin/omp_numa
#   make mpi         hybrid MPI+OpenMP (deliverable #3)      -> bin/mpi  (needs mpicxx)
#   make roofline    empirical roofline analysis tool      -> bin/roofline
#   make clean       remove the bin/ directory

CXX      ?= g++
MPICXX   ?= mpicxx
CXXSTD   := -std=c++20
OPTFLAGS := -O3 -march=native -funroll-loops -DNDEBUG
WARN     := -Wall -Wextra
CXXFLAGS := $(CXXSTD) $(OPTFLAGS) $(WARN) -I include

# Shared headers: rebuild the executables if any of these change.
HEADERS  := include/matrix_generation.hpp include/utils.hpp

# Vendored thread-pool headers (provided, Code/spmcode7), used by 'pool'.
POOL_HEADERS := include/threadPool.hpp include/taskFactory.hpp include/Affinity.hpp

BIN      := bin

.PHONY: all clean

all: seq threads omp omp_ws pool threads_numa pool_numa omp_numa

# Convenience names so 'make threads' etc. still work; real outputs live in bin/.
.PHONY: seq threads omp omp_ws pool threads_numa pool_numa omp_numa roofline threads_AB threads_baseline
seq:              $(BIN)/seq
threads:          $(BIN)/threads
omp:              $(BIN)/omp
omp_ws:           $(BIN)/omp_ws
pool:             $(BIN)/pool
threads_numa:     $(BIN)/threads_numa
pool_numa:        $(BIN)/pool_numa
omp_numa:         $(BIN)/omp_numa
roofline:         $(BIN)/roofline
threads_AB:       $(BIN)/threads_AB
threads_baseline: $(BIN)/threads_baseline

$(BIN):
	mkdir -p $(BIN)

$(BIN)/seq: src/sequential/iterative_SpMV.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/threads: src/threads/iterative_SpMV_threads.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

# OpenMP-tasks implementation (deliverable #2).
$(BIN)/omp: src/openmp/iterative_SpMV_omp.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread -fopenmp $< -o $@

# OpenMP work-sharing variant (optional comparison point; schedule via OMP_SCHEDULE).
$(BIN)/omp_ws: src/openmp/iterative_SpMV_omp_ws.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread -fopenmp $< -o $@

# Threadpool comparison variant (dynamic scheduling via the course pool).
$(BIN)/pool: src/threads/iterative_SpMV_pool.cpp $(HEADERS) $(POOL_HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

# NUMA-locality threads variant (parallel first-touch CSR replica).
$(BIN)/threads_numa: src/threads/iterative_SpMV_threads_numa.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

# NUMA-locality threadpool variant (first-touch replica + dynamic pool).
$(BIN)/pool_numa: src/threads/iterative_SpMV_pool_numa.cpp $(HEADERS) $(POOL_HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

# NUMA-locality OpenMP variant (first-touch replica + taskloop).
$(BIN)/omp_numa: src/openmp/iterative_SpMV_omp_numa.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread -fopenmp $< -o $@

# Hybrid MPI + OpenMP (deliverable #3). Built with mpicxx (cluster); kept out of
# 'all' so a plain 'make' works on machines without an MPI toolchain.
#   mpi          blocking Allgatherv baseline      -> bin/mpi
#   mpi_overlap  non-blocking Iallgatherv overlap  -> bin/mpi_overlap
.PHONY: mpi mpi_overlap
mpi: $(BIN)/mpi
mpi_overlap: $(BIN)/mpi_overlap
$(BIN)/mpi: src/mpi/iterative_SpMV_mpi.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXSTD) $(OPTFLAGS) $(WARN) -I include -fopenmp $< -o $@
$(BIN)/mpi_overlap: src/mpi/iterative_SpMV_mpi_overlap.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXSTD) $(OPTFLAGS) $(WARN) -I include -fopenmp $< -o $@

# Empirical roofline analysis tool (peak compute + STREAM bandwidth + kernels).
$(BIN)/roofline: src/bench/roofline_bench.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread -fopenmp $< -o $@

# Frozen pre-optimization version, for benchmarking against 'threads'.
$(BIN)/threads_baseline: src/threads/iterative_SpMV_threads_baseline.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

# Frozen A+B snapshot (padding + std::barrier only), to attribute the gains of
# the later C/D/E optimizations (affinity, __restrict__, barrier completion fn).
$(BIN)/threads_AB: src/threads/iterative_SpMV_threads_AB.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

clean:
	$(RM) -r $(BIN)
