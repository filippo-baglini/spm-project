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
#   make             build 'seq', 'threads' and 'omp'
#   make seq         sequential reference                 -> bin/seq
#   make threads     C++ threads implementation           -> bin/threads
#   make omp         OpenMP-tasks implementation          -> bin/omp
#   make clean       remove the bin/ directory

CXX      ?= g++
CXXSTD   := -std=c++20
OPTFLAGS := -O3 -march=native -funroll-loops -DNDEBUG
WARN     := -Wall -Wextra
CXXFLAGS := $(CXXSTD) $(OPTFLAGS) $(WARN) -I include

# Shared headers: rebuild the executables if any of these change.
HEADERS  := include/matrix_generation.hpp include/utils.hpp

BIN      := bin

.PHONY: all clean

all: seq threads omp

# Convenience names so 'make threads' etc. still work; real outputs live in bin/.
.PHONY: seq threads omp threads_AB threads_baseline
seq:              $(BIN)/seq
threads:          $(BIN)/threads
omp:              $(BIN)/omp
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

# Frozen pre-optimization version, for benchmarking against 'threads'.
$(BIN)/threads_baseline: src/threads/iterative_SpMV_threads_baseline.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

# Frozen A+B snapshot (padding + std::barrier only), to attribute the gains of
# the later C/D/E optimizations (affinity, __restrict__, barrier completion fn).
$(BIN)/threads_AB: src/threads/iterative_SpMV_threads_AB.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

clean:
	$(RM) -r $(BIN)
