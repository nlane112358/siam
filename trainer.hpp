#pragma once

#include "network.hpp"
#include "dataset.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace siam {

class ContrastiveTrainer {
public:
    ContrastiveTrainer(SiameseEncoder& encoder, uint32_t seed)
        : encoder_(encoder), rng_(seed), cache_t0_(Config::BATCH_N), cache_t1_(Config::BATCH_N),
          batch_t0_(Config::BATCH_N), batch_t1_(Config::BATCH_N) {}

    void train(std::vector<ImagePair>& pairs, int steps, float base_lr, bool quiet,
               float stability_thresh = Config::DEFAULT_STABILITY_THRESH) {
        if (pairs.empty()) {
            throw std::runtime_error("siam: no image pairs provided for training");
        }

        // 1. Index valid and stable 32x32 patch locations across all pairs
        size_t total_indexed = 0;
        for (auto& pair : pairs) {
            if (pair.valid_locs.empty()) {
                index_valid_stable_patches(pair, stability_thresh);
            }
            total_indexed += pair.valid_locs.size();
        }

        // Fallback: if stability threshold was overly restrictive, index without stability constraint
        if (total_indexed == 0 && stability_thresh < 1.0f) {
            if (!quiet) {
                std::fprintf(stderr, "siam: warning: zero patches met stability threshold %.2f; falling back to valid-pixel mask\n",
                             stability_thresh);
            }
            for (auto& pair : pairs) {
                index_valid_stable_patches(pair, 1.0f);
                total_indexed += pair.valid_locs.size();
            }
        }
        if (total_indexed == 0) {
            throw std::runtime_error("siam: no valid patch locations found across image pairs (all pixels invalid or masked)");
        }

        // 2. Group pairs by geographic location to balance sampling across locations and dates
        struct LocationGroup {
            std::string id;
            std::vector<size_t> pair_indices;
        };
        std::vector<LocationGroup> groups;
        std::unordered_map<std::string, size_t> group_map;
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (pairs[i].valid_locs.empty()) continue;
            const std::string& loc_id = pairs[i].location_id;
            auto it = group_map.find(loc_id);
            if (it == group_map.end()) {
                group_map[loc_id] = groups.size();
                groups.push_back(LocationGroup{loc_id, {i}});
            } else {
                groups[it->second].pair_indices.push_back(i);
            }
        }

        if (!quiet) {
            std::fprintf(stderr, "siam: indexed %zu valid/stable patch locations across %zu pairs and %zu geographic locations\n",
                         total_indexed, pairs.size(), groups.size());
        }

        constexpr float pi = 3.14159265358979323846f;
        float ema_loss = 0.0f;
        std::uniform_real_distribution<float> gain_dist(-0.06f, 0.06f), bias_dist(-0.03f, 0.03f);
        std::uniform_int_distribution<int> jitter_dist(-1, 1), d4_dist(0, 7);
        std::uniform_int_distribution<size_t> loc_dist(0, groups.size() - 1);

        for (int step = 1; step <= steps; ++step) {
            const float progress = (steps > 1) ? static_cast<float>(step - 1) / static_cast<float>(steps - 1) : 0.0f;
            const float cur_lr = base_lr * (0.05f + 0.95f * 0.5f * (1.0f + std::cos(pi * progress)));

            std::array<size_t, Config::BATCH_N> sampled_loc{};
            std::array<size_t, Config::BATCH_N> sampled_p{};
            std::array<int, Config::BATCH_N> sampled_y{}, sampled_x{};

            for (int b = 0; b < Config::BATCH_N; ++b) {
                size_t loc_idx = 0, p = 0;
                int y0 = 0, x0 = 0;
                int attempts = 0;

                do {
                    loc_idx = loc_dist(rng_);
                    const auto& pair_list = groups[loc_idx].pair_indices;
                    std::uniform_int_distribution<size_t> p_dist(0, pair_list.size() - 1);
                    p = pair_list[p_dist(rng_)];

                    const auto& valid_locs = pairs[p].valid_locs;
                    std::uniform_int_distribution<size_t> loc_pick(0, valid_locs.size() - 1);
                    const uint32_t packed = valid_locs[loc_pick(rng_)];
                    y0 = static_cast<int>(packed >> 16);
                    x0 = static_cast<int>(packed & 0xFFFF);

                    // Add small coordinate jitter if within bounds
                    const PgmImage& pre_img = pairs[p].pre;
                    if (y0 + 1 <= pre_img.height() - Config::PATCH_S && (rng_() & 1)) y0 += 1;
                    if (x0 + 1 <= pre_img.width() - Config::PATCH_S && (rng_() & 1)) x0 += 1;

                    // Ensure no overlapping neighboring patches in the batch as negatives
                    bool collision = false;
                    for (int prev = 0; prev < b; ++prev) {
                        if (sampled_loc[prev] == loc_idx) {
                            if (std::abs(sampled_y[prev] - y0) < Config::PATCH_S &&
                                std::abs(sampled_x[prev] - x0) < Config::PATCH_S) {
                                collision = true;
                                break;
                            }
                        }
                    }
                    if (!collision) break;
                } while (++attempts < 32);

                sampled_loc[b] = loc_idx;
                sampled_p[b] = p;
                sampled_y[b] = y0;
                sampled_x[b] = x0;

                const PgmImage& pre_img = pairs[p].pre;
                const PgmImage& post_img = pairs[p].post;
                const int jy = (pre_img.height() > Config::PATCH_S + 1) ? jitter_dist(rng_) : 0;
                const int jx = (pre_img.width() > Config::PATCH_S + 1) ? jitter_dist(rng_) : 0;
                const int d4 = d4_dist(rng_);

                pre_img.extract_patch(y0, x0, batch_t0_[b], 1.0f + gain_dist(rng_), bias_dist(rng_), d4);
                post_img.extract_patch(y0 + jy, x0 + jx, batch_t1_[b], 1.0f + gain_dist(rng_), bias_dist(rng_), d4);
            }

            const float loss = step_batch(cur_lr, 1e-4f);
            ema_loss = (step == 1) ? loss : (0.95f * ema_loss + 0.05f * loss);
            if (!quiet && (step == 1 || step % 25 == 0 || step == steps)) {
                std::fprintf(stderr, "siam: step %4d/%d lr=%.5f loss=%.5f ema=%.5f\n",
                             step, steps, cur_lr, loss, ema_loss);
            }
        }
    }

private:
    float step_batch(float lr, float weight_decay) {
        constexpr int B = Config::BATCH_N;
        #pragma omp parallel for
        for (int i = 0; i < B; ++i) {
            encoder_.forward(batch_t0_[i], cache_t0_[i]);
            encoder_.forward(batch_t1_[i], cache_t1_[i]);
        }
        float logits[B][B], probs_row[B][B], probs_col[B][B];
        constexpr float inv_temp = 1.0f / Config::TEMP;
        float loss_row = 0.0f, loss_col = 0.0f;
        for (int i = 0; i < B; ++i) {
            float max_l = -1e30f;
            for (int j = 0; j < B; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < Config::EMBED_D; ++d) dot += cache_t0_[i].u_hat[d] * cache_t1_[j].u_hat[d];
                logits[i][j] = dot * inv_temp; max_l = std::max(max_l, logits[i][j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < B; ++j) { probs_row[i][j] = std::exp(logits[i][j] - max_l); sum_exp += probs_row[i][j]; }
            for (int j = 0; j < B; ++j) probs_row[i][j] /= sum_exp;
            loss_row += -(logits[i][i] - max_l) + std::log(sum_exp);
        }
        for (int j = 0; j < B; ++j) {
            float max_l = -1e30f;
            for (int i = 0; i < B; ++i) max_l = std::max(max_l, logits[i][j]);
            float sum_exp = 0.0f;
            for (int i = 0; i < B; ++i) { probs_col[i][j] = std::exp(logits[i][j] - max_l); sum_exp += probs_col[i][j]; }
            for (int i = 0; i < B; ++i) probs_col[i][j] /= sum_exp;
            loss_col += -(logits[j][j] - max_l) + std::log(sum_exp);
        }
        std::array<Embedding, B> d_u_hat{}, d_v_hat{};
        constexpr float scale = 0.5f * inv_temp / static_cast<float>(B);
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < B; ++j) {
                const float target = (i == j) ? 2.0f : 0.0f;
                const float g = (probs_row[i][j] + probs_col[i][j] - target) * scale;
                for (int d = 0; d < Config::EMBED_D; ++d) {
                    d_u_hat[i][d] += g * cache_t1_[j].u_hat[d];
                    d_v_hat[j][d] += g * cache_t0_[i].u_hat[d];
                }
            }
        }
        float grad_arr[Config::TOTAL_PARAMS] = {0.0f};
        #pragma omp parallel for reduction(+:grad_arr[:Config::TOTAL_PARAMS])
        for (int i = 0; i < B; ++i) {
            NetworkParams local_g{};
            encoder_.backward(local_g, cache_t0_[i], d_u_hat[i]);
            encoder_.backward(local_g, cache_t1_[i], d_v_hat[i]);
            for (int k = 0; k < Config::TOTAL_PARAMS; ++k) grad_arr[k] += local_g.params[k];
        }
        auto& net_p = encoder_.weights().params;
        for (int k = 0; k < Config::TOTAL_PARAMS; ++k) {
            velocity_.params[k] = 0.9f * velocity_.params[k] + grad_arr[k];
            const float decay = (k < Config::TOTAL_WEIGHTS) ? (1.0f - lr * weight_decay) : 1.0f;
            net_p[k] = net_p[k] * decay - lr * velocity_.params[k];
        }
        return 0.5f * (loss_row + loss_col) / static_cast<float>(B);
    }

    SiameseEncoder& encoder_;
    std::mt19937 rng_;
    NetworkParams velocity_{};
    std::vector<BranchCache> cache_t0_, cache_t1_;
    std::vector<Patch> batch_t0_, batch_t1_;
};

} // namespace siam
