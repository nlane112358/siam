#pragma once

#include "network.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace siam {

class ChangeDetector {
public:
    explicit ChangeDetector(const SiameseEncoder& encoder) : encoder_(encoder) {}

    bool infer(const ImagePair& pair, int stride, float tile_thresh, float alert_frac,
               bool upsample, const std::string& out_path,
               const std::string& drop_path, const std::string& increase_path,
               const std::string& embedding_map_path, bool quiet) const {
        const PgmImage& pre = pair.pre;
        const PgmImage& post = pair.post;
        const int grid_h = (pre.height() - Config::PATCH_S) / stride + 1;
        const int grid_w = (pre.width() - Config::PATCH_S) / stride + 1;
        const size_t n_tiles = static_cast<size_t>(grid_h) * grid_w;
        const bool directional = !drop_path.empty() && !increase_path.empty();
        const auto u0 = extract_scene_embeddings_in_rows(pre, stride);
        const auto u1 = extract_scene_embeddings_in_rows(post, stride);
        if (!embedding_map_path.empty())
            save_embedding_pca_ppm(embedding_map_path, u1, grid_w, grid_h);
        const std::vector<float> mean_delta = directional
            ? mean_backscatter_delta(pre, post, grid_w, grid_h, stride)
            : std::vector<float>{};
        PgmImage change_map(grid_w, grid_h);
        PgmImage drop_map, increase_map;
        if (directional) {
            drop_map = PgmImage(grid_w, grid_h);
            increase_map = PgmImage(grid_w, grid_h);
        }
        int anomalous_tiles = 0, drop_tiles = 0, increase_tiles = 0;
        #pragma omp parallel for reduction(+:anomalous_tiles,drop_tiles,increase_tiles)
        for (size_t t = 0; t < n_tiles; ++t) {
            float cos_sim = 0.0f;
            for (int d = 0; d < Config::EMBED_D; ++d) cos_sim += u0[t][d] * u1[t][d];
            const float cos_dist = 0.5f * (1.0f - cos_sim);
            change_map.data()[t] = cos_dist;
            if (cos_dist >= tile_thresh) {
                ++anomalous_tiles;
                if (directional && mean_delta[t] < 0.0f) {
                    drop_map.data()[t] = cos_dist;
                    ++drop_tiles;
                } else if (directional && mean_delta[t] > 0.0f) {
                    increase_map.data()[t] = cos_dist;
                    ++increase_tiles;
                }
            }
        }
        if (upsample) change_map.upsample_bilinear(pre.width(), pre.height(), stride).save(out_path);
        else change_map.save(out_path);
        if (directional) {
            if (upsample) {
                drop_map.upsample_bilinear(pre.width(), pre.height(), stride).save(drop_path);
                increase_map.upsample_bilinear(pre.width(), pre.height(), stride).save(increase_path);
            } else {
                drop_map.save(drop_path);
                increase_map.save(increase_path);
            }
        }
        const float frac = static_cast<float>(anomalous_tiles) / static_cast<float>(n_tiles);
        const bool is_alert = frac >= alert_frac;
        if (!quiet) {
            std::fprintf(stderr, "siam: fraction=%.4f (%d/%zu tiles >= %.2f) status=%s\n",
                         frac, anomalous_tiles, n_tiles, tile_thresh, is_alert ? "ALERT" : "NORMAL");
            if (directional)
                std::fprintf(stderr, "siam: directional anomalous tiles: drop=%d increase=%d\n",
                             drop_tiles, increase_tiles);
        }
        return is_alert;
    }

    void evaluate(const std::vector<ImagePair>& pairs, int stride,
                  float tile_thresh, float alert_frac) const {
        const size_t num_pairs = pairs.size();
        if (num_pairs < 2) throw std::runtime_error("siam: evaluate mode (-e) requires at least 2 image pairs");
        const size_t shift = num_pairs / 2;
        std::vector<size_t> pair_offsets(num_pairs + 1, 0);
        for (size_t p = 0; p < num_pairs; ++p) {
            const size_t q = (p + shift) % num_pairs;
            const int h = std::min(pairs[p].pre.height(), pairs[q].post.height());
            const int w = std::min(pairs[p].pre.width(), pairs[q].post.width());
            const size_t gh = static_cast<size_t>((h - Config::PATCH_S) / stride + 1);
            const size_t gw = static_cast<size_t>((w - Config::PATCH_S) / stride + 1);
            pair_offsets[p + 1] = pair_offsets[p] + gh * gw;
        }
        const size_t total_tiles = pair_offsets[num_pairs];
        std::vector<float> same_dists(total_tiles), diff_dists(total_tiles);
        size_t same_alerts = 0, diff_alerts = 0;
        double same_sum = 0.0, diff_sum = 0.0;
        #pragma omp parallel for reduction(+:same_alerts,diff_alerts,same_sum,diff_sum)
        for (size_t p = 0; p < num_pairs; ++p) {
            const size_t q = (p + shift) % num_pairs;
            const int h = std::min(pairs[p].pre.height(), pairs[q].post.height());
            const int w = std::min(pairs[p].pre.width(), pairs[q].post.width());
            const int gh = (h - Config::PATCH_S) / stride + 1;
            const int gw = (w - Config::PATCH_S) / stride + 1;
            const size_t n_tiles = static_cast<size_t>(gh) * gw, base_idx = pair_offsets[p];
            const auto u_pre = encoder_.extract_scene_embeddings(pairs[p].pre, h, w, stride);
            const auto u_same = encoder_.extract_scene_embeddings(pairs[p].post, h, w, stride);
            const auto u_diff = encoder_.extract_scene_embeddings(pairs[q].post, h, w, stride);
            int anom_same = 0, anom_diff = 0;
            for (size_t t = 0; t < n_tiles; ++t) {
                float sim_same = 0.0f, sim_diff = 0.0f;
                for (int d = 0; d < Config::EMBED_D; ++d) {
                    sim_same += u_pre[t][d] * u_same[t][d];
                    sim_diff += u_pre[t][d] * u_diff[t][d];
                }
                const float dist_same = 0.5f * (1.0f - sim_same);
                const float dist_diff = 0.5f * (1.0f - sim_diff);
                same_dists[base_idx + t] = dist_same; diff_dists[base_idx + t] = dist_diff;
                same_sum += static_cast<double>(dist_same); diff_sum += static_cast<double>(dist_diff);
                if (dist_same >= tile_thresh) ++anom_same;
                if (dist_diff >= tile_thresh) ++anom_diff;
            }
            if (static_cast<float>(anom_same) / static_cast<float>(n_tiles) >= alert_frac) ++same_alerts;
            if (static_cast<float>(anom_diff) / static_cast<float>(n_tiles) >= alert_frac) ++diff_alerts;
        }
        std::sort(same_dists.begin(), same_dists.end());
        std::sort(diff_dists.begin(), diff_dists.end());
        double u_sum = 0.0;
        size_t p_lt = 0, p_le = 0;
        for (size_t k = 0; k < total_tiles; ++k) {
            const float d = diff_dists[k];
            while (p_lt < total_tiles && same_dists[p_lt] < d) ++p_lt;
            if (p_le < p_lt) p_le = p_lt;
            while (p_le < total_tiles && same_dists[p_le] <= d) ++p_le;
            u_sum += static_cast<double>(p_lt) + 0.5 * static_cast<double>(p_le - p_lt);
        }
        const double roc_auc = u_sum / (static_cast<double>(total_tiles) * static_cast<double>(total_tiles));
        const double same_mean = same_sum / static_cast<double>(total_tiles);
        const double diff_mean = diff_sum / static_cast<double>(total_tiles);
        std::printf("siam: evaluated %zu pairs (%zu tiles, stride=%d, thresh=%.2f)\n"
                    "  matched    (pre_i, post_i): mean=%.4f med=%.4f p95=%.4f alerts=%zu/%zu (%.1f%%)\n"
                    "  mismatched (pre_i, post_j): mean=%.4f med=%.4f p95=%.4f alerts=%zu/%zu (%.1f%%)\n"
                    "  separation gap:             %+.4f\n"
                    "  tile ROC-AUC:               %.4f\n",
                    num_pairs, total_tiles, stride, tile_thresh,
                    same_mean, same_dists[total_tiles / 2], same_dists[(total_tiles * 95) / 100],
                    same_alerts, num_pairs, 100.0 * static_cast<double>(same_alerts) / static_cast<double>(num_pairs),
                    diff_mean, diff_dists[total_tiles / 2], diff_dists[(total_tiles * 95) / 100],
                    diff_alerts, num_pairs, 100.0 * static_cast<double>(diff_alerts) / static_cast<double>(num_pairs),
                    diff_mean - same_mean, roc_auc);
    }

    struct CalibStats {
        double mean = 0.0;
        double stddev = 0.0;
        float median = 0.0f;
        float p90 = 0.0f;
        float p95 = 0.0f;
        float p98 = 0.0f;
        float p99 = 0.0f;
        float p995 = 0.0f;
        float p999 = 0.0f;
        float recommended_thresh = 0.0f;
        size_t total_tiles = 0;
    };

    CalibStats calibrate(const std::vector<ImagePair>& pairs, int stride,
                         float target_p_fa, const std::string& out_path,
                         bool quiet) const {
        if (pairs.empty()) {
            throw std::runtime_error("siam: calibrate mode (-C) requires at least one image pair");
        }
        std::vector<float> all_dists;
        double sum = 0.0, sum_sq = 0.0;

        for (const auto& pair : pairs) {
            const PgmImage& pre = pair.pre;
            const PgmImage& post = pair.post;
            const int h = std::min(pre.height(), post.height());
            const int w = std::min(pre.width(), post.width());
            if (h < Config::PATCH_S || w < Config::PATCH_S) continue;
            const int gh = (h - Config::PATCH_S) / stride + 1;
            const int gw = (w - Config::PATCH_S) / stride + 1;
            const size_t n_tiles = static_cast<size_t>(gh) * gw;

            const auto u0 = encoder_.extract_scene_embeddings(pre, h, w, stride);
            const auto u1 = encoder_.extract_scene_embeddings(post, h, w, stride);

            for (size_t t = 0; t < n_tiles; ++t) {
                float cos_sim = 0.0f;
                for (int d = 0; d < Config::EMBED_D; ++d) {
                    cos_sim += u0[t][d] * u1[t][d];
                }
                const float cos_dist = 0.5f * (1.0f - cos_sim);
                all_dists.push_back(cos_dist);
                sum += static_cast<double>(cos_dist);
                sum_sq += static_cast<double>(cos_dist) * static_cast<double>(cos_dist);
            }
        }

        if (all_dists.empty()) {
            throw std::runtime_error("siam: no valid tiles found for calibration");
        }

        const size_t n = all_dists.size();
        std::sort(all_dists.begin(), all_dists.end());

        CalibStats stats;
        stats.total_tiles = n;
        stats.mean = sum / static_cast<double>(n);
        const double var = std::max(0.0, (sum_sq / static_cast<double>(n)) - (stats.mean * stats.mean));
        stats.stddev = std::sqrt(var);

        auto get_p = [&](double p) -> float {
            double idx = p * static_cast<double>(n - 1);
            size_t lo = static_cast<size_t>(std::floor(idx));
            size_t hi = static_cast<size_t>(std::ceil(idx));
            if (hi >= n) hi = n - 1;
            double frac = idx - static_cast<double>(lo);
            return static_cast<float>((1.0 - frac) * all_dists[lo] + frac * all_dists[hi]);
        };

        stats.median = get_p(0.50);
        stats.p90 = get_p(0.90);
        stats.p95 = get_p(0.95);
        stats.p98 = get_p(0.98);
        stats.p99 = get_p(0.99);
        stats.p995 = get_p(0.995);
        stats.p999 = get_p(0.999);

        const double p_target = std::clamp(1.0 - static_cast<double>(target_p_fa), 0.50, 0.9999);
        stats.recommended_thresh = get_p(p_target);

        if (!quiet) {
            std::printf("siam: target ROI calibration across %zu pair(s) (%zu tiles, stride=%d)\n"
                        "  baseline distribution:\n"
                        "    mean=%.4f  std=%.4f  median=%.4f\n"
                        "    p90=%.4f   p95=%.4f   p98=%.4f   p99=%.4f   p99.5=%.4f   p99.9=%.4f\n"
                        "  target false alarm rate P_fa: %.4f (percentile: %.2f%%)\n"
                        "  calibrated tile_thresh:       %.4f\n",
                        pairs.size(), n, stride,
                        stats.mean, stats.stddev, stats.median,
                        stats.p90, stats.p95, stats.p98, stats.p99, stats.p995, stats.p999,
                        target_p_fa, p_target * 100.0, stats.recommended_thresh);
        }

        if (!out_path.empty() && out_path != "-") {
            FileHandle fh(out_path, "w", stdout);
            std::fprintf(fh.get(), "%.4f\n", stats.recommended_thresh);
        } else if (out_path == "-") {
            std::printf("%.4f\n", stats.recommended_thresh);
        }

        return stats;
    }

private:
    std::vector<Embedding> extract_scene_embeddings_in_rows(const PgmImage& img,
                                                              int stride) const {
        const int width = img.width(), height = img.height();
        const int grid_h = (height - Config::PATCH_S) / stride + 1;
        const int grid_w = (width - Config::PATCH_S) / stride + 1;
        if (stride % 4 != 0 || grid_h <= 128)
            return encoder_.extract_scene_embeddings(img, height, width, stride);

        constexpr int tile_rows_per_chunk = 64;
        const size_t n_tiles = static_cast<size_t>(grid_h) * grid_w;
        std::vector<Embedding> embeddings(n_tiles);
        for (int gy0 = 0; gy0 < grid_h; gy0 += tile_rows_per_chunk) {
            const int gy1 = std::min(gy0 + tile_rows_per_chunk, grid_h);
            const int context_gy0 = std::max(0, gy0 - 1);
            const int context_gy1 = std::min(grid_h, gy1 + 1);
            const int crop_y0 = context_gy0 * stride;
            const int crop_h = (context_gy1 - context_gy0 - 1) * stride + Config::PATCH_S;

            PgmImage crop(width, crop_h);
            const size_t crop_pixels = static_cast<size_t>(crop_h) * width;
            std::copy_n(img.data() + static_cast<size_t>(crop_y0) * width,
                        crop_pixels, crop.data());
            const auto chunk = encoder_.extract_scene_embeddings(crop, crop_h, width, stride);
            const int chunk_grid_h = (crop_h - Config::PATCH_S) / stride + 1;
            if (chunk.size() != static_cast<size_t>(chunk_grid_h) * grid_w)
                throw std::runtime_error("siam: invalid chunked embedding grid");

            const int local_gy0 = gy0 - context_gy0;
            for (int gy = gy0; gy < gy1; ++gy) {
                const size_t src = static_cast<size_t>(local_gy0 + gy - gy0) * grid_w;
                const size_t dst = static_cast<size_t>(gy) * grid_w;
                std::copy_n(chunk.data() + src, grid_w, embeddings.data() + dst);
            }
        }
        return embeddings;
    }

    static void save_embedding_pca_ppm(const std::string& path,
                                       const std::vector<Embedding>& embeddings,
                                       int width, int height) {
        constexpr int dims = Config::EMBED_D;
        using Matrix = std::array<std::array<double, dims>, dims>;
        using Vector = std::array<double, dims>;
        const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (width < 1 || height < 1 || embeddings.size() != count)
            throw std::runtime_error("siam: invalid embedding grid for PCA visualization");
        if (count > SIZE_MAX / 3)
            throw std::runtime_error("siam: PCA visualization dimensions overflow");

        Vector mean{};
        Matrix covariance{};
        for (const Embedding& embedding : embeddings) {
            for (int d = 0; d < dims; ++d) {
                const double value = static_cast<double>(embedding[d]);
                if (!std::isfinite(value))
                    throw std::runtime_error("siam: non-finite embedding value in PCA visualization");
                mean[d] += value;
                for (int e = d; e < dims; ++e)
                    covariance[d][e] += value * static_cast<double>(embedding[e]);
            }
        }

        const double inv_count = 1.0 / static_cast<double>(count);
        for (int d = 0; d < dims; ++d) {
            mean[d] *= inv_count;
            for (int e = d; e < dims; ++e) {
                covariance[d][e] = covariance[d][e] * inv_count - mean[d] * mean[e];
                covariance[e][d] = covariance[d][e];
            }
        }

        // Diagonalize the 32x32 covariance matrix with symmetric Jacobi rotations.
        Matrix eigenvectors{};
        for (int d = 0; d < dims; ++d) eigenvectors[d][d] = 1.0;
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
                        const double vkp = eigenvectors[k][p], vkq = eigenvectors[k][q];
                        eigenvectors[k][p] = c * vkp - s * vkq;
                        eigenvectors[k][q] = s * vkp + c * vkq;
                    }
                    rotated = true;
                }
            }
            if (!rotated) break;
        }

        Vector eigenvalues{};
        std::array<int, dims> order{};
        std::iota(order.begin(), order.end(), 0);
        for (int d = 0; d < dims; ++d) eigenvalues[d] = covariance[d][d];
        std::sort(order.begin(), order.end(), [&eigenvalues](int a, int b) {
            if (eigenvalues[a] == eigenvalues[b]) return a < b;
            return eigenvalues[a] > eigenvalues[b];
        });

        const double largest_eigenvalue = std::max(0.0, eigenvalues[order[0]]);
        const double zero_variance_cutoff = largest_eigenvalue * 1e-12;
        std::array<bool, 3> component_enabled{};
        std::array<Vector, 3> component_vectors{};
        for (int component = 0; component < 3; ++component) {
            const int eigen_index = order[component];
            component_enabled[component] = eigenvalues[eigen_index] > zero_variance_cutoff;
            int pivot = 0;
            for (int d = 1; d < dims; ++d) {
                if (std::abs(eigenvectors[d][eigen_index]) >
                    std::abs(eigenvectors[pivot][eigen_index])) pivot = d;
            }
            const double sign = eigenvectors[pivot][eigen_index] < 0.0 ? -1.0 : 1.0;
            for (int d = 0; d < dims; ++d)
                component_vectors[component][d] = eigenvectors[d][eigen_index] * sign;
        }

        std::vector<std::array<double, 3>> scores(count);
        std::array<double, 3> minimum{}, maximum{};
        minimum.fill(std::numeric_limits<double>::infinity());
        maximum.fill(-std::numeric_limits<double>::infinity());
        for (size_t t = 0; t < count; ++t) {
            for (int component = 0; component < 3; ++component) {
                if (!component_enabled[component]) continue;
                double score = 0.0;
                for (int d = 0; d < dims; ++d)
                    score += (static_cast<double>(embeddings[t][d]) - mean[d]) *
                             component_vectors[component][d];
                scores[t][component] = score;
                minimum[component] = std::min(minimum[component], score);
                maximum[component] = std::max(maximum[component], score);
            }
        }

        std::vector<uint8_t> rgb(count * 3, 0);
        for (size_t t = 0; t < count; ++t) {
            for (int component = 0; component < 3; ++component) {
                const double range = maximum[component] - minimum[component];
                if (!component_enabled[component] || !(range > 0.0)) continue;
                const double scaled = 255.0 * (scores[t][component] - minimum[component]) / range;
                rgb[t * 3 + component] = static_cast<uint8_t>(
                    std::clamp(static_cast<int>(std::lround(scaled)), 0, 255));
            }
        }

        FileHandle fh(path, "wb", stdout);
        FILE* fp = fh.get();
        if (std::fprintf(fp, "P6\n%d %d\n255\n", width, height) < 0 ||
            std::fwrite(rgb.data(), 1, rgb.size(), fp) != rgb.size())
            throw std::runtime_error("siam: failed writing PCA visualization to " + path);
    }

    static std::vector<float> mean_backscatter_delta(const PgmImage& pre,
                                                     const PgmImage& post,
                                                     int grid_w, int grid_h,
                                                     int stride) {
        constexpr int patch = Config::PATCH_S;
        const int width = pre.width();
        std::vector<float> means(static_cast<size_t>(grid_w) * grid_h, 0.0f);
        std::vector<double> vertical_sum(grid_w, 0.0);
        std::vector<double> ring(static_cast<size_t>(patch) * grid_w, 0.0);
        std::vector<double> row_sum(grid_w, 0.0);

        // Rolling 32x32 sums keep directional gating O(image pixels) without
        // allocating a full-scene difference or integral image.
        for (int y = 0; y < pre.height(); ++y) {
            const float* pre_row = pre.data() + static_cast<size_t>(y) * width;
            const float* post_row = post.data() + static_cast<size_t>(y) * width;
            double window_sum = 0.0;
            for (int dx = 0; dx < patch; ++dx)
                window_sum += static_cast<double>(post_row[dx] - pre_row[dx]);
            row_sum[0] = window_sum;

            for (int gx = 1; gx < grid_w; ++gx) {
                const int old_x = (gx - 1) * stride;
                if (stride <= patch) {
                    for (int dx = 0; dx < stride; ++dx) {
                        window_sum -= static_cast<double>(post_row[old_x + dx] - pre_row[old_x + dx]);
                        const int add_x = old_x + patch + dx;
                        window_sum += static_cast<double>(post_row[add_x] - pre_row[add_x]);
                    }
                } else {
                    window_sum = 0.0;
                    const int x = gx * stride;
                    for (int dx = 0; dx < patch; ++dx)
                        window_sum += static_cast<double>(post_row[x + dx] - pre_row[x + dx]);
                }
                row_sum[gx] = window_sum;
            }

            const int ring_y = y % patch;
            for (int gx = 0; gx < grid_w; ++gx) {
                const size_t idx = static_cast<size_t>(ring_y) * grid_w + gx;
                vertical_sum[gx] += row_sum[gx] - ring[idx];
                ring[idx] = row_sum[gx];
            }

            if (y >= patch - 1) {
                const int top_y = y - patch + 1;
                if (top_y % stride == 0) {
                    const int gy = top_y / stride;
                    if (gy < grid_h) {
                        const double inv_area = 1.0 / static_cast<double>(patch * patch);
                        for (int gx = 0; gx < grid_w; ++gx)
                            means[static_cast<size_t>(gy) * grid_w + gx] =
                                static_cast<float>(vertical_sum[gx] * inv_area);
                    }
                }
            }
        }
        return means;
    }

    const SiameseEncoder& encoder_;
};

} // namespace siam
