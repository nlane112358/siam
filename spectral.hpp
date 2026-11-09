#pragma once

#include "config.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace siam {

struct SpectralDecomposition {
    using Vec = std::array<double, Config::EMBED_D>;
    using Mat = std::array<Vec, Config::EMBED_D>;

    Vec mean{};
    Vec eigenvalues{};       // Sorted descending.
    Mat eigenvectors{};      // Eigenvectors are stored in columns.

    static SpectralDecomposition fit(const std::vector<Embedding>& samples) {
        constexpr int dims = Config::EMBED_D;
        if (samples.empty())
            throw std::runtime_error("siam: cannot fit spectral decomposition to an empty grid");

        SpectralDecomposition result;
        for (const Embedding& sample : samples) {
            for (int d = 0; d < dims; ++d) {
                const double value = static_cast<double>(sample[d]);
                if (!std::isfinite(value))
                    throw std::runtime_error("siam: non-finite value in spectral decomposition");
                result.mean[d] += value;
            }
        }
        const double inv_count = 1.0 / static_cast<double>(samples.size());
        for (double& value : result.mean) value *= inv_count;

        Mat covariance{};
        for (const Embedding& sample : samples) {
            Vec centered{};
            for (int d = 0; d < dims; ++d)
                centered[d] = static_cast<double>(sample[d]) - result.mean[d];
            for (int d = 0; d < dims; ++d)
                for (int e = d; e < dims; ++e)
                    covariance[d][e] += centered[d] * centered[e];
        }
        const double covariance_scale = samples.size() > 1
            ? 1.0 / static_cast<double>(samples.size() - 1) : 0.0;
        for (int d = 0; d < dims; ++d) {
            for (int e = d; e < dims; ++e) {
                covariance[d][e] *= covariance_scale;
                covariance[e][d] = covariance[d][e];
            }
            result.eigenvectors[d][d] = 1.0;
        }

        // Symmetric Jacobi rotations; this is shared by PCA and RX whitening.
        for (int sweep = 0; sweep < 100; ++sweep) {
            double max_off_diagonal = 0.0, diagonal_scale = 0.0;
            for (int d = 0; d < dims; ++d) {
                diagonal_scale = std::max(diagonal_scale, std::abs(covariance[d][d]));
                for (int e = d + 1; e < dims; ++e)
                    max_off_diagonal = std::max(max_off_diagonal, std::abs(covariance[d][e]));
            }
            const double tolerance = std::max(1e-300, diagonal_scale * 1e-13);
            if (max_off_diagonal <= tolerance) break;

            bool rotated = false;
            for (int p = 0; p < dims; ++p) {
                for (int q = p + 1; q < dims; ++q) {
                    const double apq = covariance[p][q];
                    if (std::abs(apq) <= tolerance) continue;
                    const double tau = (covariance[q][q] - covariance[p][p]) / (2.0 * apq);
                    const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                                     (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                    const double c = 1.0 / std::sqrt(1.0 + t * t);
                    const double s = t * c;
                    const double app = covariance[p][p], aqq = covariance[q][q];
                    covariance[p][p] = app - t * apq;
                    covariance[q][q] = aqq + t * apq;
                    covariance[p][q] = covariance[q][p] = 0.0;
                    for (int k = 0; k < dims; ++k) {
                        if (k != p && k != q) {
                            const double akp = covariance[k][p], akq = covariance[k][q];
                            covariance[k][p] = covariance[p][k] = c * akp - s * akq;
                            covariance[k][q] = covariance[q][k] = s * akp + c * akq;
                        }
                        const double vkp = result.eigenvectors[k][p];
                        const double vkq = result.eigenvectors[k][q];
                        result.eigenvectors[k][p] = c * vkp - s * vkq;
                        result.eigenvectors[k][q] = s * vkp + c * vkq;
                    }
                    rotated = true;
                }
            }
            if (!rotated) break;
        }

        Vec unsorted_values{};
        Mat unsorted_vectors = result.eigenvectors;
        std::array<int, dims> order{};
        for (int d = 0; d < dims; ++d) {
            unsorted_values[d] = std::max(0.0, covariance[d][d]);
            order[d] = d;
        }
        std::sort(order.begin(), order.end(), [&unsorted_values](int a, int b) {
            if (unsorted_values[a] == unsorted_values[b]) return a < b;
            return unsorted_values[a] > unsorted_values[b];
        });
        for (int k = 0; k < dims; ++k) {
            result.eigenvalues[k] = unsorted_values[order[k]];
            for (int d = 0; d < dims; ++d)
                result.eigenvectors[d][k] = unsorted_vectors[d][order[k]];
        }
        return result;
    }

    float mahalanobis_sq(const Embedding& x, double reg_eps = 1e-5) const noexcept {
        double dist_sq = 0.0;
        for (int k = 0; k < Config::EMBED_D; ++k) {
            double projection = 0.0;
            for (int d = 0; d < Config::EMBED_D; ++d)
                projection += (static_cast<double>(x[d]) - mean[d]) * eigenvectors[d][k];
            dist_sq += (projection * projection) / std::max(eigenvalues[k] + reg_eps,
                                                             std::numeric_limits<double>::min());
        }
        return static_cast<float>(dist_sq);
    }

    static float chi2_to_zscore(float d_sq, float degrees_of_freedom) noexcept {
        const float k = std::max(1.0f, degrees_of_freedom);
        const float var = 2.0f / (9.0f * k);
        const float cube_root = std::cbrt(std::max(0.0f, d_sq / k));
        return (cube_root - (1.0f - var)) / std::sqrt(var);
    }

    static float chi2_to_zscore(float d_sq) noexcept {
        return chi2_to_zscore(d_sq, static_cast<float>(Config::EMBED_D));
    }
};

} // namespace siam
