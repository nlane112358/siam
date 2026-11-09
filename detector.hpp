#pragma once

#include "network.hpp"
#include "spectral.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace siam {

class ChangeDetector {
public:
    explicit ChangeDetector(const SiameseEncoder& encoder) : encoder_(encoder) {}

    bool infer(const ImagePair& pair, int stride, double false_alarm_prob,
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
        const std::vector<Embedding> residuals = motion_compensated_residuals(u0, u1, grid_w, grid_h);
        const SpectralDecomposition residual_spectrum = SpectralDecomposition::fit(residuals);
        const std::vector<float> mean_delta = directional
            ? mean_backscatter_delta(pre, post, grid_w, grid_h, stride)
            : std::vector<float>{};
        PgmImage change_map(grid_w, grid_h);
        PgmImage drop_map, increase_map;
        if (directional) {
            drop_map = PgmImage(grid_w, grid_h);
            increase_map = PgmImage(grid_w, grid_h);
        }
        const double tile_probability = false_alarm_prob / static_cast<double>(n_tiles);
        std::vector<uint8_t> anomalous_mask(n_tiles, 0);
        int anomalous_tiles = 0, drop_tiles = 0, increase_tiles = 0;
        #pragma omp parallel for reduction(+:anomalous_tiles,drop_tiles,increase_tiles)
        for (size_t t = 0; t < n_tiles; ++t) {
            const float distance_sq = residual_spectrum.mahalanobis_sq(residuals[t]);
            const double tile_p = normal_survival(SpectralDecomposition::chi2_to_zscore(distance_sq));
            const float score = static_cast<float>(1.0 - tile_p);
            change_map.data()[t] = score;
            if (tile_p <= tile_probability) {
                anomalous_mask[t] = 1;
                ++anomalous_tiles;
                if (directional && mean_delta[t] < 0.0f) {
                    drop_map.data()[t] = score;
                    ++drop_tiles;
                } else if (directional && mean_delta[t] > 0.0f) {
                    increase_map.data()[t] = score;
                    ++increase_tiles;
                }
            }
        }
        std::vector<uint8_t> visited(n_tiles, 0);
        std::vector<size_t> frontier;
        frontier.reserve(static_cast<size_t>(anomalous_tiles));
        size_t blob_count = 0, largest_blob = 0;
        for (size_t root = 0; root < n_tiles; ++root) {
            if (!anomalous_mask[root] || visited[root]) continue;
            ++blob_count;
            size_t blob_size = 0;
            frontier.clear();
            frontier.push_back(root);
            visited[root] = 1;
            while (!frontier.empty()) {
                const size_t tile = frontier.back();
                frontier.pop_back();
                ++blob_size;
                const int gy = static_cast<int>(tile / static_cast<size_t>(grid_w));
                const int gx = static_cast<int>(tile % static_cast<size_t>(grid_w));
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dy == 0 && dx == 0) continue;
                        const int ny = gy + dy, nx = gx + dx;
                        if (ny < 0 || ny >= grid_h || nx < 0 || nx >= grid_w) continue;
                        const size_t neighbor = static_cast<size_t>(ny) * grid_w + nx;
                        if (!anomalous_mask[neighbor] || visited[neighbor]) continue;
                        visited[neighbor] = 1;
                        frontier.push_back(neighbor);
                    }
                }
            }
            largest_blob = std::max(largest_blob, blob_size);
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
        const double frac = static_cast<double>(anomalous_tiles) / static_cast<double>(n_tiles);
        const bool is_alert = anomalous_tiles > 0;
        if (!quiet) {
            std::fprintf(stderr,
                         "siam: RX-CFAR fraction=%.4f (%d/%zu tiles p<=%.2e; image p<=%.2e) blobs=%zu largest=%zu tiles status=%s\n",
                         frac, anomalous_tiles, n_tiles, tile_probability, false_alarm_prob,
                         blob_count, largest_blob,
                         is_alert ? "ALERT" : "NORMAL");
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
private:
    static double normal_survival(float z) noexcept {
        constexpr double inv_sqrt_two = 0.70710678118654752440;
        return 0.5 * std::erfc(static_cast<double>(z) * inv_sqrt_two);
    }

    static float cosine_similarity(const Embedding& a, const Embedding& b) noexcept {
        float similarity = 0.0f;
        for (int d = 0; d < Config::EMBED_D; ++d) similarity += a[d] * b[d];
        return similarity;
    }

    static float similarity_at(const Embedding& reference,
                               const std::vector<Embedding>& grid,
                               int grid_w, int grid_h, int y, int x) noexcept {
        y = std::clamp(y, 0, grid_h - 1);
        x = std::clamp(x, 0, grid_w - 1);
        return cosine_similarity(reference, grid[static_cast<size_t>(y) * grid_w + x]);
    }

    static std::array<float, 2> subpixel_peak_offset(const Embedding& reference,
                                                      const std::vector<Embedding>& grid,
                                                      int grid_w, int grid_h,
                                                      int y, int x) noexcept {
        double f[3][3]{};
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                f[dy + 1][dx + 1] = similarity_at(reference, grid, grid_w, grid_h,
                                                   y + dy, x + dx);

        const double f0 = f[1][1];
        const double hxx = f[1][2] + f[1][0] - 2.0 * f0;
        const double hyy = f[2][1] + f[0][1] - 2.0 * f0;
        const double hxy = 0.25 * (f[2][2] - f[2][0] - f[0][2] + f[0][0]);
        const double gx = 0.5 * (f[1][2] - f[1][0]);
        const double gy = 0.5 * (f[2][1] - f[0][1]);
        const double determinant = hxx * hyy - hxy * hxy;
        if (!(hxx < 0.0 && hyy < 0.0 && determinant > 1e-12)) return {{0.0f, 0.0f}};

        const double dx = (-hyy * gx + hxy * gy) / determinant;
        const double dy = (hxy * gx - hxx * gy) / determinant;
        return {{static_cast<float>(std::clamp(dx, -0.5, 0.5)),
                 static_cast<float>(std::clamp(dy, -0.5, 0.5))}};
    }

    static Embedding interpolate_embedding(const std::vector<Embedding>& grid,
                                           int grid_w, int grid_h,
                                           float y, float x) noexcept {
        y = std::clamp(y, 0.0f, static_cast<float>(grid_h - 1));
        x = std::clamp(x, 0.0f, static_cast<float>(grid_w - 1));
        const int y0 = static_cast<int>(std::floor(y));
        const int x0 = static_cast<int>(std::floor(x));
        const int y1 = std::min(y0 + 1, grid_h - 1);
        const int x1 = std::min(x0 + 1, grid_w - 1);
        const float wy = y - static_cast<float>(y0), wx = x - static_cast<float>(x0);
        const Embedding& e00 = grid[static_cast<size_t>(y0) * grid_w + x0];
        const Embedding& e01 = grid[static_cast<size_t>(y0) * grid_w + x1];
        const Embedding& e10 = grid[static_cast<size_t>(y1) * grid_w + x0];
        const Embedding& e11 = grid[static_cast<size_t>(y1) * grid_w + x1];
        Embedding result{};
        float norm_sq = 1e-12f;
        for (int d = 0; d < Config::EMBED_D; ++d) {
            const float top = e00[d] + wx * (e01[d] - e00[d]);
            const float bottom = e10[d] + wx * (e11[d] - e10[d]);
            result[d] = top + wy * (bottom - top);
            norm_sq += result[d] * result[d];
        }
        const float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (float& value : result) value *= inv_norm;
        return result;
    }

    static std::vector<Embedding> motion_compensated_residuals(
            const std::vector<Embedding>& pre, const std::vector<Embedding>& post,
            int grid_w, int grid_h) {
        constexpr int search_radius = 2;
        const size_t count = static_cast<size_t>(grid_w) * grid_h;
        if (pre.size() != count || post.size() != count)
            throw std::runtime_error("siam: invalid embedding grid for motion compensation");
        std::vector<Embedding> residuals(count);
        #pragma omp parallel for
        for (size_t t = 0; t < count; ++t) {
            const int gy = static_cast<int>(t / static_cast<size_t>(grid_w));
            const int gx = static_cast<int>(t % static_cast<size_t>(grid_w));
            float best_similarity = -std::numeric_limits<float>::infinity();
            int best_y = gy, best_x = gx;
            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                const int ny = gy + dy;
                if (ny < 0 || ny >= grid_h) continue;
                for (int dx = -search_radius; dx <= search_radius; ++dx) {
                    const int nx = gx + dx;
                    if (nx < 0 || nx >= grid_w) continue;
                    const float similarity = cosine_similarity(
                        pre[t], post[static_cast<size_t>(ny) * grid_w + nx]);
                    if (similarity > best_similarity) {
                        best_similarity = similarity;
                        best_y = ny;
                        best_x = nx;
                    }
                }
            }
            const auto offset = subpixel_peak_offset(pre[t], post, grid_w, grid_h, best_y, best_x);
            const Embedding aligned = interpolate_embedding(
                post, grid_w, grid_h, static_cast<float>(best_y) + offset[1],
                static_cast<float>(best_x) + offset[0]);
            for (int d = 0; d < Config::EMBED_D; ++d)
                residuals[t][d] = aligned[d] - pre[t][d];
        }
        return residuals;
    }

    std::vector<Embedding> extract_scene_embeddings_in_rows(const PgmImage& img,
                                                              int stride) const {
        const int width = img.width(), height = img.height();
        const int grid_h = (height - Config::PATCH_S) / stride + 1;
        const int grid_w = (width - Config::PATCH_S) / stride + 1;
        if (stride % 4 != 0 || grid_h <= 128)
            return encoder_.extract_scene_embeddings(img, height, width, stride);

        constexpr int tile_rows_per_chunk = 64;
        const int pyramid_context_rows = (32 * 4 + stride - 1) / stride;
        const size_t n_tiles = static_cast<size_t>(grid_h) * grid_w;
        std::vector<Embedding> embeddings(n_tiles);
        for (int gy0 = 0; gy0 < grid_h; gy0 += tile_rows_per_chunk) {
            const int gy1 = std::min(gy0 + tile_rows_per_chunk, grid_h);
            const int context_gy0 = std::max(0, gy0 - pyramid_context_rows);
            const int context_gy1 = std::min(grid_h, gy1 + pyramid_context_rows);
            const int crop_y0 = context_gy0 * stride;
            const int crop_h = (context_gy1 - context_gy0 - 1) * stride + Config::PATCH_S;

            PgmImage crop(width, crop_h);
            const size_t crop_pixels = static_cast<size_t>(crop_h) * width;
            img.copy_pixels(static_cast<size_t>(crop_y0) * width, crop_pixels, crop.data());
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
        const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (width < 1 || height < 1 || embeddings.size() != count)
            throw std::runtime_error("siam: invalid embedding grid for PCA visualization");
        if (count > SIZE_MAX / 3)
            throw std::runtime_error("siam: PCA visualization dimensions overflow");
        const SpectralDecomposition spectrum = SpectralDecomposition::fit(embeddings);
        const auto& mean = spectrum.mean;
        const double largest_eigenvalue = spectrum.eigenvalues[0];
        const double zero_variance_cutoff = largest_eigenvalue * 1e-12;
        std::array<bool, 3> component_enabled{};
        std::array<SpectralDecomposition::Vec, 3> component_vectors{};
        for (int component = 0; component < 3; ++component) {
            component_enabled[component] = spectrum.eigenvalues[component] > zero_variance_cutoff;
            int pivot = 0;
            for (int d = 1; d < dims; ++d) {
                if (std::abs(spectrum.eigenvectors[d][component]) >
                    std::abs(spectrum.eigenvectors[pivot][component])) pivot = d;
            }
            const double sign = spectrum.eigenvectors[pivot][component] < 0.0 ? -1.0 : 1.0;
            for (int d = 0; d < dims; ++d)
                component_vectors[component][d] = spectrum.eigenvectors[d][component] * sign;
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
            const size_t row_offset = static_cast<size_t>(y) * width;
            double window_sum = 0.0;
            for (int dx = 0; dx < patch; ++dx)
                window_sum += static_cast<double>(post.pixel(row_offset + dx) - pre.pixel(row_offset + dx));
            row_sum[0] = window_sum;

            for (int gx = 1; gx < grid_w; ++gx) {
                const int old_x = (gx - 1) * stride;
                if (stride <= patch) {
                    for (int dx = 0; dx < stride; ++dx) {
                        const size_t remove_index = row_offset + old_x + dx;
                        window_sum -= static_cast<double>(post.pixel(remove_index) - pre.pixel(remove_index));
                        const int add_x = old_x + patch + dx;
                        const size_t add_index = row_offset + add_x;
                        window_sum += static_cast<double>(post.pixel(add_index) - pre.pixel(add_index));
                    }
                } else {
                    window_sum = 0.0;
                    const int x = gx * stride;
                    for (int dx = 0; dx < patch; ++dx)
                        window_sum += static_cast<double>(post.pixel(row_offset + x + dx) -
                                                          pre.pixel(row_offset + x + dx));
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
