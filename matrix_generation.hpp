#pragma once

#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// CSR matrix

struct CSRMatrix {
    std::size_t n = 0;
    std::vector<std::uint64_t> row_ptr;
    std::vector<std::uint32_t> col_idx;
    std::vector<double> values;
};

struct MatrixStats {
    std::uint64_t total_nnz = 0;
    std::uint32_t min_nnz_per_row = 0;
    std::uint32_t max_nnz_per_row = 0;
};

struct GeneratedMatrix {
    CSRMatrix A;
    MatrixStats stats;
};

// Sparsity pattern

static long double irregular_weight(std::size_t i, std::size_t n) {
    const std::size_t r1_end = n / 10;
    const std::size_t r2_end = n / 4;
    const std::size_t r3_end = (3 * n) / 5;

    long double w = 1.0L;

    if (i < r1_end) {
        w = 40.0L;
        if ((i % 16) < 2) {
            w *= 4.0L;
        }
    } else if (i < r2_end) {
        w = 10.0L;
        if ((i % 16) == 0) {
            w *= 2.0L;
        }
    } else if (i < r3_end) {
        w = 3.0L;
    } else {
        w = 1.0L;
    }

    return w;
}

static std::vector<std::uint32_t> regular_nnz_pattern(std::size_t n, std::uint64_t nz_total) {
    const std::uint64_t base = nz_total / n;
    const std::uint64_t rem = nz_total % n;

    if (base > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("too many nonzeros per row for uint32_t");
    }

    std::vector<std::uint32_t> nnz(n, static_cast<std::uint32_t>(base));
    for (std::size_t i = 0; i < static_cast<std::size_t>(rem); ++i) {
        ++nnz[i];
    }
    return nnz;
}

// Distribute exactly nz_total nonzeros according to row weights.
static std::vector<std::uint32_t> weighted_nnz_pattern(std::size_t n, std::uint64_t nz_total) {
    if (nz_total < n) {
        throw std::runtime_error("nz must be at least n");
    }

    std::vector<long double> weights(n);
    long double sum_w = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        weights[i] = irregular_weight(i, n);
        sum_w += weights[i];
    }

    const std::uint64_t capacity = static_cast<std::uint64_t>(n) - 1;
    const std::uint64_t extras_total = nz_total - n;

    std::vector<std::uint32_t> nnz(n, 1);
    std::vector<std::pair<long double, std::size_t>> remainder;
    remainder.reserve(n);

    std::uint64_t assigned = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const long double ideal = static_cast<long double>(extras_total) * weights[i] / sum_w;
        std::uint64_t extra = static_cast<std::uint64_t>(std::floor(ideal));
        if (extra > capacity) {
            extra = capacity;
        }

        nnz[i] += static_cast<std::uint32_t>(extra);
        assigned += extra;

        if (extra < capacity) {
            remainder.push_back({ideal - std::floor(ideal), i});
        }
    }

    std::uint64_t left = extras_total - assigned;

    std::sort(remainder.begin(), remainder.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });

    for (const auto& [unused, i] : remainder) {
        (void)unused;
        if (left == 0) {
            break;
        }
        if (nnz[i] < n) {
            ++nnz[i];
            --left;
        }
    }

    if (left > 0) {
        for (std::size_t i = 0; i < n && left > 0; ++i) {
            const std::uint64_t room = static_cast<std::uint64_t>(n) - nnz[i];
            const std::uint64_t add = std::min(room, left);
            nnz[i] += static_cast<std::uint32_t>(add);
            left -= add;
        }
    }

    if (left != 0) {
        throw std::runtime_error("internal error: could not distribute all nonzeros");
    }

    return nnz;
}

static std::vector<std::uint32_t> make_nnz_pattern(std::size_t n,
                                                   std::uint64_t nz_total,
                                                   const std::string& mode) {
    if (n == 0) {
        throw std::runtime_error("n must be positive");
    }
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("n exceeds uint32_t column index range");
    }
    if (nz_total < n) {
        throw std::runtime_error("nz must be at least n");
    }
    if (nz_total > static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n)) {
        throw std::runtime_error("nz cannot exceed n*n");
    }

    if (mode == "regular") {
        return regular_nnz_pattern(n, nz_total);
    }
    if (mode == "irregular") {
        return weighted_nnz_pattern(n, nz_total);
    }

    throw std::runtime_error("unknown matrix mode");
}

// Matrix generation

static std::uint64_t make_coprime_stride(std::uint64_t raw, std::size_t n) {
    if (n <= 1) {
        return 1;
    }

    std::uint64_t stride = raw % n;
    if (stride == 0) {
        stride = 1;
    }

    if ((stride % 2) == 0) {
        ++stride;
        if (stride >= n) {
            stride = 1;
        }
    }

    while (std::gcd(stride, static_cast<std::uint64_t>(n)) != 1) {
        stride += 2;
        if (stride >= n) {
            stride = 1;
        }
    }

    return stride;
}

static GeneratedMatrix generate_matrix(std::size_t n,
                                       std::uint64_t nz_total,
                                       std::uint64_t seed,
                                       const std::string& mode) {
    GeneratedMatrix G;
    G.A.n = n;

    const std::vector<std::uint32_t> nnz_per_row = make_nnz_pattern(n, nz_total, mode);

    G.A.row_ptr.resize(n + 1, 0);

    std::uint64_t prefix = 0;
    std::uint32_t min_row_nnz = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_row_nnz = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t row_nnz = nnz_per_row[i];

        prefix += row_nnz;
        G.A.row_ptr[i + 1] = prefix;

        min_row_nnz = std::min(min_row_nnz, row_nnz);
        max_row_nnz = std::max(max_row_nnz, row_nnz);
    }

    if (prefix != nz_total) {
        throw std::runtime_error("internal error: generated nnz does not match requested nz");
    }

    G.A.col_idx.resize(nz_total);
    G.A.values.resize(nz_total);

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t begin = G.A.row_ptr[i];
        const std::uint64_t end = G.A.row_ptr[i + 1];
        const std::uint64_t row_nnz = end - begin;

        SplitMix64 rng(seed ^ SplitMix64::mix(i + 1));

        G.A.col_idx[begin] = static_cast<std::uint32_t>(i);

        // Diagonal value.
        {
            const double v = 1.0 + 0.1 + 0.9 * rng.next_unit();
            G.A.values[begin] = v;
        }

        if (row_nnz > 1) {
            const std::uint64_t start = rng.next_u64() % n;
            const std::uint64_t stride = make_coprime_stride(rng.next_u64(), n);

            std::uint64_t filled = 1;
            std::uint64_t pos = start;

            while (filled < row_nnz) {
                const std::uint32_t col = static_cast<std::uint32_t>(pos);

                if (col != i) {
                    const std::uint64_t idx = begin + filled;
                    G.A.col_idx[idx] = col;

                    const double v = 0.1 + 0.9 * rng.next_unit();
                    G.A.values[idx] = v;

                    ++filled;
                }

                pos += stride;
                if (pos >= n) {
                    pos %= n;
                }
            }
        }

        double row_norm2 = 0.0;
        for (std::uint64_t p = begin; p < end; ++p) {
            row_norm2 += G.A.values[p] * G.A.values[p];
        }

        const double inv = 1.0 / std::sqrt(row_norm2);
        for (std::uint64_t p = begin; p < end; ++p) {
            G.A.values[p] *= inv;
        }
    }

    G.stats.total_nnz = nz_total;
    G.stats.min_nnz_per_row = min_row_nnz;
    G.stats.max_nnz_per_row = max_row_nnz;

    return G;
}

static void print_matrix_stats(const GeneratedMatrix& G) {
    std::cout << "rows=" << G.A.n << "  ";
    std::cout << "total_nnz=" << G.stats.total_nnz << "  ";
    std::cout << "min_nnz_per_row=" << G.stats.min_nnz_per_row << "  ";
    std::cout << "max_nnz_per_row=" << G.stats.max_nnz_per_row << "  ";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "avg_nnz_per_row="
              << static_cast<double>(G.stats.total_nnz) / static_cast<double>(G.A.n)
              << "\n";
}
