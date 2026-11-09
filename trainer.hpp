#pragma once

#include "network.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
namespace siam {

class BarlowTwinsTrainer {
public:
    BarlowTwinsTrainer(SiameseEncoder& encoder, uint32_t seed)
        : encoder_(encoder), rng_(seed), cache_t0_(Config::BATCH_N), cache_t1_(Config::BATCH_N),
          batch_t0_(Config::BATCH_N), batch_t1_(Config::BATCH_N) {}

    void train(const std::vector<ImagePair>& pairs, int steps, float base_lr, bool quiet) {
        constexpr float pi = 3.14159265358979323846f;
        float ema_loss = 0.0f;
        std::uniform_real_distribution<float> gain_dist(-0.06f, 0.06f), bias_dist(-0.03f, 0.03f);
        std::uniform_int_distribution<int> jitter_dist(-1, 1), d4_dist(0, 7);
        std::uniform_int_distribution<size_t> pair_dist(0, pairs.size() - 1);
        for (int step = 1; step <= steps; ++step) {
            const float progress = (steps > 1) ? static_cast<float>(step - 1) / static_cast<float>(steps - 1) : 0.0f;
            const float cur_lr = base_lr * (0.05f + 0.95f * 0.5f * (1.0f + std::cos(pi * progress)));
            std::array<int, Config::BATCH_N> sampled_p{}, sampled_y{}, sampled_x{};
            for (int b = 0; b < Config::BATCH_N; ++b) {
                size_t p = 0;
                int y0 = 0, x0 = 0, attempts = 0;
                do {
                    p = pair_dist(rng_);
                    const PgmImage& pre_img = pairs[p].pre;
                    y0 = (pre_img.height() > Config::PATCH_S + 1)
                       ? std::uniform_int_distribution<int>(1, pre_img.height() - Config::PATCH_S - 1)(rng_) : 0;
                    x0 = (pre_img.width() > Config::PATCH_S + 1)
                       ? std::uniform_int_distribution<int>(1, pre_img.width() - Config::PATCH_S - 1)(rng_) : 0;
                    bool collision = false;
                    for (int prev = 0; prev < b; ++prev) {
                        if (sampled_p[prev] == static_cast<int>(p) &&
                            std::abs(sampled_y[prev] - y0) < Config::PATCH_S / 2 &&
                            std::abs(sampled_x[prev] - x0) < Config::PATCH_S / 2) {
                            collision = true; break;
                        }
                    }
                    if (!collision) break;
                } while (++attempts < 16);
                sampled_p[b] = static_cast<int>(p); sampled_y[b] = y0; sampled_x[b] = x0;
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
            if (!quiet && (step == 1 || step % 25 == 0 || step == steps))
                std::fprintf(stderr, "siam: step %4d/%d lr=%.5f loss=%.5f ema=%.5f\n", step, steps, cur_lr, loss, ema_loss);
        }
    }

private:
    float step_batch(float lr, float weight_decay) {
        constexpr int B = Config::BATCH_N;
        constexpr int D = Config::EMBED_D;
        constexpr float off_diagonal_weight = 0.005f;
        constexpr float standardize_eps = 1e-4f;
        #pragma omp parallel for
        for (int i = 0; i < B; ++i) {
            encoder_.forward(batch_t0_[i], cache_t0_[i]);
            encoder_.forward(batch_t1_[i], cache_t1_[i]);
        }
        std::array<Embedding, B> z0{}, z1{}, d_z0{}, d_z1{}, d_u_hat{}, d_v_hat{};
        std::array<float, D> mean0{}, mean1{}, inv_std0{}, inv_std1{};
        for (int d = 0; d < D; ++d) {
            for (int b = 0; b < B; ++b) {
                mean0[d] += cache_t0_[b].u_hat[d] / static_cast<float>(B);
                mean1[d] += cache_t1_[b].u_hat[d] / static_cast<float>(B);
            }
            float variance0 = 0.0f, variance1 = 0.0f;
            for (int b = 0; b < B; ++b) {
                const float centered0 = cache_t0_[b].u_hat[d] - mean0[d];
                const float centered1 = cache_t1_[b].u_hat[d] - mean1[d];
                variance0 += centered0 * centered0 / static_cast<float>(B);
                variance1 += centered1 * centered1 / static_cast<float>(B);
            }
            inv_std0[d] = 1.0f / std::sqrt(variance0 + standardize_eps);
            inv_std1[d] = 1.0f / std::sqrt(variance1 + standardize_eps);
            for (int b = 0; b < B; ++b) {
                z0[b][d] = (cache_t0_[b].u_hat[d] - mean0[d]) * inv_std0[d];
                z1[b][d] = (cache_t1_[b].u_hat[d] - mean1[d]) * inv_std1[d];
            }
        }

        std::array<std::array<float, D>, D> correlation{}, d_correlation{};
        float loss = 0.0f;
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                float value = 0.0f;
                for (int b = 0; b < B; ++b) value += z0[b][i] * z1[b][j];
                correlation[i][j] = value / static_cast<float>(B);
                const float target = (i == j) ? 1.0f : 0.0f;
                const float residual = correlation[i][j] - target;
                const float weight = (i == j) ? 1.0f : off_diagonal_weight;
                loss += weight * residual * residual;
                d_correlation[i][j] = 2.0f * weight * residual;
            }
        }

        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    d_z0[b][i] += d_correlation[i][j] * z1[b][j] / static_cast<float>(B);
                    d_z1[b][j] += d_correlation[i][j] * z0[b][i] / static_cast<float>(B);
                }
            }
        }

        for (int d = 0; d < D; ++d) {
            float mean_grad0 = 0.0f, mean_grad1 = 0.0f;
            float mean_grad_z0 = 0.0f, mean_grad_z1 = 0.0f;
            for (int b = 0; b < B; ++b) {
                mean_grad0 += d_z0[b][d] / static_cast<float>(B);
                mean_grad1 += d_z1[b][d] / static_cast<float>(B);
                mean_grad_z0 += d_z0[b][d] * z0[b][d] / static_cast<float>(B);
                mean_grad_z1 += d_z1[b][d] * z1[b][d] / static_cast<float>(B);
            }
            for (int b = 0; b < B; ++b) {
                d_u_hat[b][d] = inv_std0[d] *
                    (d_z0[b][d] - mean_grad0 - z0[b][d] * mean_grad_z0);
                d_v_hat[b][d] = inv_std1[d] *
                    (d_z1[b][d] - mean_grad1 - z1[b][d] * mean_grad_z1);
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
        return loss;
    }

    SiameseEncoder& encoder_;
    std::mt19937 rng_;
    NetworkParams velocity_{};
    std::vector<BranchCache> cache_t0_, cache_t1_;
    std::vector<Patch> batch_t0_, batch_t1_;
};

} // namespace siam
