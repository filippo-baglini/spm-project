# Makefile for the One-Shot project
#
#   Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
#
# Targets:
#   make            build both 'seq' and 'threads'
#   make seq        build the sequential reference            -> ./seq
#   make threads    build the C++ threads implementation      -> ./threads
#   make clean      remove the built executables

CXX      ?= g++
CXXSTD   := -std=c++20
OPTFLAGS := -O3 -march=native -funroll-loops -DNDEBUG
WARN     := -Wall -Wextra
CXXFLAGS := $(CXXSTD) $(OPTFLAGS) $(WARN) -I .

# Shared headers: rebuild the executables if any of these change.
HEADERS  := matrix_generation.hpp utils.hpp

BINARIES := seq threads

.PHONY: all clean

all: $(BINARIES)

seq: iterative_SpMV.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) iterative_SpMV.cpp -o $@

threads: iterative_SpMV_threads.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -pthread iterative_SpMV_threads.cpp -o $@

clean:
	$(RM) $(BINARIES)
