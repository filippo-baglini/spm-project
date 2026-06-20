#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Deterministic PRNG / mixing

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state(seed) {}

    std::uint64_t next_u64() {
        state += 0x9e3779b97f4a7c15ULL;
        return mix(state);
    }

    double next_unit() {
        const std::uint64_t x = next_u64();
        return (x >> 11) * (1.0 / 9007199254740992.0);
    }

    static std::uint64_t mix(std::uint64_t x) {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    }

private:
    std::uint64_t state;
};

// Command-line parsing

static bool read_arg_u64(int argc, char** argv, const std::string& name, std::uint64_t& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            out = std::strtoull(argv[i + 1], nullptr, 10);
            return true;
        }
    }
    return false;
}

static bool read_arg_str(int argc, char** argv, const std::string& name, std::string& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            out = argv[i + 1];
            return true;
        }
    }
    return false;
}


static std::uint64_t checksum_vector(const std::vector<double>& x) {
    std::uint64_t acc = 0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &x[i], sizeof(double));
        acc ^= SplitMix64::mix(bits ^ SplitMix64::mix(i));
    }

    return acc;
}

static void dump_vector(const std::string& path, const std::vector<double>& x) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("could not open vector dump file: " + path);
    }

    out << std::setprecision(17);
    for (const double v : x) {
        out << v << '\n';
    }

    if (!out) {
        throw std::runtime_error("could not write vector dump file: " + path);
    }
}


static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " -n N -nz K -m regular|irregular [-s seed] [--dump-vector FILE]\n\n"
        << "Parameters:\n"
        << "  -n   Matrix size, NxN\n"
        << "  -nz  Total number of nonzeros\n"
        << "  -m   Matrix mode: regular or irregular\n"
        << "  -s   Optional seed, default 111\n"
        << "  --dump-vector FILE\n"
        << "       Optional output file for the final normalized vector\n";
}
