#pragma once

#include "network.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace siam {

class ChangeDetector {
public:
    explicit ChangeDetector(const SiameseEncoder& encoder) : encoder_(encoder) {}

    bool infer(const ImagePair& pair, int stride, float tile_thresh, float alert_frac,
               bool upsample, const std::string& out_path, bool quiet) const {
        const PgmImage& pre = pair.pre;
        const PgmImage& post = pair.post;
        const int grid_h = (pre.height() - Config::PATCH_S) / stride + 1;
        const int grid_w = (pre.width() - Config::PATCH_S) / stride + 1;
        const size_t n_tiles = static_cast<size_t>(grid_h) * grid_w;
        const auto u0 = encoder_.extract_scene_embeddings(pre, pre.height(), pre.width(), stride);
        const auto u1 = encoder_.extract_scene_embeddings(post, post.height(), post.width(), stride);
        PgmImage change_map(grid_w, grid_h);
        int anomalous_tiles = 0;
        #pragma omp parallel for reduction(+:anomalous_tiles)
        for (size_t t = 0; t < n_tiles; ++t) {
            float cos_sim = 0.0f;
            for (int d = 0; d < Config::EMBED_D; ++d) cos_sim += u0[t][d] * u1[t][d];
            const float cos_dist = 0.5f * (1.0f - cos_sim);
            change_map.data()[t] = cos_dist;
            if (cos_dist >= tile_thresh) ++anomalous_tiles;
        }
        if (upsample) change_map.upsample_bilinear(pre.width(), pre.height(), stride).save(out_path);
        else change_map.save(out_path);
        const float frac = static_cast<float>(anomalous_tiles) / static_cast<float>(n_tiles);
        const bool is_alert = frac >= alert_frac;
        if (!quiet)
            std::fprintf(stderr, "siam: fraction=%.4f (%d/%zu tiles >= %.2f) status=%s\n",
                         frac, anomalous_tiles, n_tiles, tile_thresh, is_alert ? "ALERT" : "NORMAL");
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
    const SiameseEncoder& encoder_;
};

} // namespace siam
