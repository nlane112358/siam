#pragma once

#include "image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace siam {

struct WeightHeader {
    uint32_t magic;
    uint16_t patch_s, c1, c2, embed_d;
    uint32_t reserved;
};

struct NetworkParams {
    std::array<float, Config::TOTAL_PARAMS> params{};
    static constexpr size_t OFF_W1 = 0;
    static constexpr size_t OFF_W2 = OFF_W1 + Config::NUM_W1;
    static constexpr size_t OFF_W3 = OFF_W2 + Config::NUM_W2;
    static constexpr size_t OFF_B1 = OFF_W3 + Config::NUM_W3;
    static constexpr size_t OFF_B2 = OFF_B1 + Config::C1;
    static constexpr size_t OFF_B3 = OFF_B2 + Config::C2;
    float* w1() noexcept { return params.data() + OFF_W1; }
    float* w2() noexcept { return params.data() + OFF_W2; }
    float* w3() noexcept { return params.data() + OFF_W3; }
    float* b1() noexcept { return params.data() + OFF_B1; }
    float* b2() noexcept { return params.data() + OFF_B2; }
    float* b3() noexcept { return params.data() + OFF_B3; }
    const float* w1() const noexcept { return params.data() + OFF_W1; }
    const float* w2() const noexcept { return params.data() + OFF_W2; }
    const float* w3() const noexcept { return params.data() + OFF_W3; }
    const float* b1() const noexcept { return params.data() + OFF_B1; }
    const float* b2() const noexcept { return params.data() + OFF_B2; }
    const float* b3() const noexcept { return params.data() + OFF_B3; }
};

struct BranchCache {
    std::array<float, (Config::PATCH_S + 2) * (Config::PATCH_S + 2)> in_pad{};
    std::array<float, Config::C1 * Config::H1 * Config::H1> z1{};
    std::array<float, Config::C1 * (Config::H1 + 2) * (Config::H1 + 2)> a1_pad{};
    std::array<float, Config::C2 * Config::H2 * Config::H2> z2{};
    std::array<float, Config::C2 * Config::H2 * Config::H2> a2{};
    std::array<float, Config::C2> gap{};
    Embedding u{};
    Embedding u_hat{};
    float inv_norm = 0.0f;
};

class SiameseEncoder {
public:
    void init_kaiming(std::mt19937& rng) {
        auto uniform = [&rng](float scale) {
            return std::uniform_real_distribution<float>(-scale, scale)(rng);
        };
        weights_.params.fill(0.0f);
        const float s1 = std::sqrt(6.0f / 9.0f);
        for (int i = 0; i < Config::NUM_W1; ++i) weights_.w1()[i] = uniform(s1);
        const float s2 = std::sqrt(6.0f / (Config::C1 * 9.0f));
        for (int i = 0; i < Config::NUM_W2; ++i) weights_.w2()[i] = uniform(s2);
        const float s3 = std::sqrt(6.0f / static_cast<float>(Config::C2));
        for (int i = 0; i < Config::NUM_W3; ++i) weights_.w3()[i] = uniform(s3);
    }

    void save_weights(const std::string& path) const {
        FileHandle fh(path, "wb", nullptr);
        const WeightHeader hdr{Config::MAGIC, Config::PATCH_S, Config::C1,
                               Config::C2, Config::EMBED_D, 0};
        if (std::fwrite(&hdr, sizeof(hdr), 1, fh.get()) != 1 ||
            std::fwrite(weights_.params.data(), sizeof(float), Config::TOTAL_PARAMS, fh.get()) != Config::TOTAL_PARAMS)
            throw std::runtime_error("siam: failed writing weights to " + path);
    }

    void load_weights(const std::string& path) {
        FileHandle fh(path, "rb", nullptr);
        WeightHeader hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, fh.get()) != 1)
            throw std::runtime_error("siam: failed to read weights header from " + path);
        if (hdr.magic != Config::MAGIC || hdr.patch_s != Config::PATCH_S ||
            hdr.c1 != Config::C1 || hdr.c2 != Config::C2 || hdr.embed_d != Config::EMBED_D)
            throw std::runtime_error("siam: incompatible or corrupt weights header in " + path);
        if (std::fread(weights_.params.data(), sizeof(float), Config::TOTAL_PARAMS, fh.get()) != Config::TOTAL_PARAMS)
            throw std::runtime_error("siam: truncated weights payload in " + path);
    }

    void forward(const Patch& patch, BranchCache& c) const noexcept {
        constexpr int P0 = Config::PATCH_S + 2, P1 = Config::H1 + 2;
        c.in_pad.fill(0.0f);
        c.a1_pad.fill(0.0f);
        for (int h = 0; h < Config::PATCH_S; ++h)
            std::memcpy(&c.in_pad[(h + 1) * P0 + 1], &patch[h * Config::PATCH_S], Config::PATCH_S * sizeof(float));

        for (int oc = 0; oc < Config::C1; ++oc) {
            float* z1_oc = &c.z1[oc * Config::H1 * Config::H1];
            float* a1_oc = &c.a1_pad[oc * P1 * P1];
            const float* w_ptr = &weights_.w1()[oc * 9];
            const float b = weights_.b1()[oc];
            for (int oh = 0; oh < Config::H1; ++oh) {
                for (int ow = 0; ow < Config::H1; ++ow) z1_oc[oh * Config::H1 + ow] = b;
                for (int kh = 0; kh < 3; ++kh) {
                    const float* row = &c.in_pad[(oh * 2 + kh) * P0];
                    const float w0 = w_ptr[kh * 3], w1 = w_ptr[kh * 3 + 1], w2 = w_ptr[kh * 3 + 2];
                    for (int ow = 0; ow < Config::H1; ++ow) {
                        const int iw = ow * 2;
                        z1_oc[oh * Config::H1 + ow] += row[iw] * w0 + row[iw + 1] * w1 + row[iw + 2] * w2;
                    }
                }
                for (int ow = 0; ow < Config::H1; ++ow) {
                    const float s = z1_oc[oh * Config::H1 + ow];
                    a1_oc[(oh + 1) * P1 + (ow + 1)] = (s > 0.0f) ? s : 0.1f * s;
                }
            }
        }

        for (int oc = 0; oc < Config::C2; ++oc) {
            float* z2_oc = &c.z2[oc * Config::H2 * Config::H2];
            std::fill_n(z2_oc, Config::H2 * Config::H2, weights_.b2()[oc]);
            for (int ic = 0; ic < Config::C1; ++ic) {
                const float* w_ptr = &weights_.w2()[(oc * Config::C1 + ic) * 9];
                const float* a1_ic = &c.a1_pad[ic * P1 * P1];
                for (int oh = 0; oh < Config::H2; ++oh) {
                    for (int kh = 0; kh < 3; ++kh) {
                        const float* row = &a1_ic[(oh * 2 + kh) * P1];
                        const float w0 = w_ptr[kh * 3], w1 = w_ptr[kh * 3 + 1], w2 = w_ptr[kh * 3 + 2];
                        for (int ow = 0; ow < Config::H2; ++ow) {
                            const int iw = ow * 2;
                            z2_oc[oh * Config::H2 + ow] += row[iw] * w0 + row[iw + 1] * w1 + row[iw + 2] * w2;
                        }
                    }
                }
            }
            for (int i = 0; i < Config::H2 * Config::H2; ++i) {
                const float s = z2_oc[i];
                c.a2[oc * Config::H2 * Config::H2 + i] = (s > 0.0f) ? s : 0.1f * s;
            }
        }

        constexpr float inv_spatial = 1.0f / static_cast<float>(Config::H2 * Config::H2);
        for (int oc = 0; oc < Config::C2; ++oc) {
            float sum = 0.0f;
            for (int i = 0; i < Config::H2 * Config::H2; ++i) sum += c.a2[oc * Config::H2 * Config::H2 + i];
            c.gap[oc] = sum * inv_spatial;
        }

        float norm_sq = 1e-12f;
        for (int d = 0; d < Config::EMBED_D; ++d) {
            float sum = weights_.b3()[d];
            for (int ic = 0; ic < Config::C2; ++ic) sum += weights_.w3()[d * Config::C2 + ic] * c.gap[ic];
            c.u[d] = sum;
            norm_sq += sum * sum;
        }
        c.inv_norm = 1.0f / std::sqrt(norm_sq);
        for (int d = 0; d < Config::EMBED_D; ++d) c.u_hat[d] = c.u[d] * c.inv_norm;
    }

    void backward(NetworkParams& grad, const BranchCache& c, const Embedding& d_u_hat) const noexcept {
        constexpr int P0 = Config::PATCH_S + 2, P1 = Config::H1 + 2;
        float dot = 0.0f;
        for (int d = 0; d < Config::EMBED_D; ++d) dot += c.u_hat[d] * d_u_hat[d];
        Embedding d_u{};
        for (int d = 0; d < Config::EMBED_D; ++d) d_u[d] = c.inv_norm * (d_u_hat[d] - c.u_hat[d] * dot);

        std::array<float, Config::C2> d_gap{};
        for (int d = 0; d < Config::EMBED_D; ++d) {
            grad.b3()[d] += d_u[d];
            for (int ic = 0; ic < Config::C2; ++ic) {
                grad.w3()[d * Config::C2 + ic] += d_u[d] * c.gap[ic];
                d_gap[ic] += weights_.w3()[d * Config::C2 + ic] * d_u[d];
            }
        }

        std::array<float, Config::C2 * Config::H2 * Config::H2> d_z2{};
        constexpr float inv_spatial = 1.0f / static_cast<float>(Config::H2 * Config::H2);
        for (int oc = 0; oc < Config::C2; ++oc) {
            const float g = d_gap[oc] * inv_spatial;
            for (int i = 0; i < Config::H2 * Config::H2; ++i) {
                const int idx = oc * Config::H2 * Config::H2 + i;
                d_z2[idx] = (c.z2[idx] > 0.0f) ? g : 0.1f * g;
            }
        }

        std::array<float, Config::C1 * P1 * P1> d_a1_pad{};
        for (int oc = 0; oc < Config::C2; ++oc) {
            const float* dz_oc = &d_z2[oc * Config::H2 * Config::H2];
            float b_sum = 0.0f;
            for (int i = 0; i < Config::H2 * Config::H2; ++i) b_sum += dz_oc[i];
            grad.b2()[oc] += b_sum;
            for (int ic = 0; ic < Config::C1; ++ic) {
                float* gw_ptr = &grad.w2()[(oc * Config::C1 + ic) * 9];
                const float* w_ptr = &weights_.w2()[(oc * Config::C1 + ic) * 9];
                const float* a1_ic = &c.a1_pad[ic * P1 * P1];
                float* da1_ic = &d_a1_pad[ic * P1 * P1];
                for (int oh = 0; oh < Config::H2; ++oh) {
                    for (int kh = 0; kh < 3; ++kh) {
                        const float* a1_row = &a1_ic[(oh * 2 + kh) * P1];
                        float* da1_row = &da1_ic[(oh * 2 + kh) * P1];
                        float gw0 = 0.0f, gw1 = 0.0f, gw2 = 0.0f;
                        const float w0 = w_ptr[kh * 3], w1 = w_ptr[kh * 3 + 1], w2 = w_ptr[kh * 3 + 2];
                        for (int ow = 0; ow < Config::H2; ++ow) {
                            const float dz = dz_oc[oh * Config::H2 + ow];
                            const int iw = ow * 2;
                            gw0 += dz * a1_row[iw]; gw1 += dz * a1_row[iw + 1]; gw2 += dz * a1_row[iw + 2];
                            da1_row[iw] += dz * w0; da1_row[iw + 1] += dz * w1; da1_row[iw + 2] += dz * w2;
                        }
                        gw_ptr[kh * 3] += gw0; gw_ptr[kh * 3 + 1] += gw1; gw_ptr[kh * 3 + 2] += gw2;
                    }
                }
            }
        }

        for (int oc = 0; oc < Config::C1; ++oc) {
            const float* z1_oc = &c.z1[oc * Config::H1 * Config::H1];
            const float* da1_oc = &d_a1_pad[oc * P1 * P1];
            float* gw_ptr = &grad.w1()[oc * 9];
            float b_sum = 0.0f;
            for (int oh = 0; oh < Config::H1; ++oh) {
                std::array<float, Config::H1> dz_row{};
                for (int ow = 0; ow < Config::H1; ++ow) {
                    const float da = da1_oc[(oh + 1) * P1 + (ow + 1)];
                    const float dz = (z1_oc[oh * Config::H1 + ow] > 0.0f) ? da : 0.1f * da;
                    dz_row[ow] = dz;
                    b_sum += dz;
                }
                for (int kh = 0; kh < 3; ++kh) {
                    const float* in_row = &c.in_pad[(oh * 2 + kh) * P0];
                    float gw0 = 0.0f, gw1 = 0.0f, gw2 = 0.0f;
                    for (int ow = 0; ow < Config::H1; ++ow) {
                        const float dz = dz_row[ow];
                        const int iw = ow * 2;
                        gw0 += dz * in_row[iw]; gw1 += dz * in_row[iw + 1]; gw2 += dz * in_row[iw + 2];
                    }
                    gw_ptr[kh * 3] += gw0; gw_ptr[kh * 3 + 1] += gw1; gw_ptr[kh * 3 + 2] += gw2;
                }
            }
            grad.b1()[oc] += b_sum;
        }
    }

    std::vector<Embedding> extract_scene_embeddings(const PgmImage& img,
                                                     int height, int width, int stride) const {
        const int grid_h = (height - Config::PATCH_S) / stride + 1;
        const int grid_w = (width - Config::PATCH_S) / stride + 1;
        std::vector<Embedding> u_hat_grid(static_cast<size_t>(grid_h) * grid_w);
        if (stride % 4 != 0) {
            #pragma omp parallel for collapse(2)
            for (int gy = 0; gy < grid_h; ++gy) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    Patch p; BranchCache c;
                    img.extract_patch(gy * stride, gx * stride, p);
                    forward(p, c);
                    u_hat_grid[static_cast<size_t>(gy) * grid_w + gx] = c.u_hat;
                }
            }
            return u_hat_grid;
        }

        const int h1 = height / 2, w1_dim = width / 2;
        const int h2 = h1 / 2, w2_dim = w1_dim / 2;
        const int p0_h = height + 2, p0_w = width + 2;
        const int p1_h = h1 + 2, p1_w = w1_dim + 2;
        const int sat_h = h2 + 1, sat_w = w2_dim + 1;
        std::vector<float> in_pad(static_cast<size_t>(p0_h) * p0_w, 0.0f);
        std::vector<float> a1_pad(static_cast<size_t>(Config::C1) * p1_h * p1_w, 0.0f);
        std::vector<float> z2_buf(static_cast<size_t>(Config::C2) * h2 * w2_dim, 0.0f);
        std::vector<double> sat(static_cast<size_t>(Config::C2) * sat_h * sat_w, 0.0);
        for (int y = 0; y < height; ++y) {
            const float* src = &img.data()[static_cast<size_t>(y) * img.width()];
            float* dst = &in_pad[static_cast<size_t>(y + 1) * p0_w + 1];
            for (int x = 0; x < width; ++x) {
                const float v = std::clamp(src[x], 0.0f, 1.0f);
                dst[x] = (v - 0.5f) * 2.0f;
            }
        }

        #pragma omp parallel for
        for (int oc = 0; oc < Config::C1; ++oc) {
            float* a1_oc = &a1_pad[static_cast<size_t>(oc) * p1_h * p1_w];
            const float* w_ptr = &weights_.w1()[oc * 9];
            const float b = weights_.b1()[oc];
            for (int oh = 0; oh < h1; ++oh) {
                float* a1_row = &a1_oc[static_cast<size_t>(oh + 1) * p1_w + 1];
                std::fill_n(a1_row, w1_dim, b);
                for (int kh = 0; kh < 3; ++kh) {
                    const float* in_row = &in_pad[static_cast<size_t>(oh * 2 + kh) * p0_w];
                    const float k0 = w_ptr[kh * 3], k1 = w_ptr[kh * 3 + 1], k2 = w_ptr[kh * 3 + 2];
                    for (int ow = 0; ow < w1_dim; ++ow) {
                        const int iw = ow * 2;
                        a1_row[ow] += in_row[iw] * k0 + in_row[iw + 1] * k1 + in_row[iw + 2] * k2;
                    }
                }
                for (int ow = 0; ow < w1_dim; ++ow) {
                    const float s = a1_row[ow]; a1_row[ow] = (s > 0.0f) ? s : 0.1f * s;
                }
            }
        }

        #pragma omp parallel for
        for (int oc = 0; oc < Config::C2; ++oc) {
            float* z2_oc = &z2_buf[static_cast<size_t>(oc) * h2 * w2_dim];
            double* sat_oc = &sat[static_cast<size_t>(oc) * sat_h * sat_w];
            std::fill_n(z2_oc, h2 * w2_dim, weights_.b2()[oc]);
            for (int ic = 0; ic < Config::C1; ++ic) {
                const float* w_ptr = &weights_.w2()[(oc * Config::C1 + ic) * 9];
                const float* a1_ic = &a1_pad[static_cast<size_t>(ic) * p1_h * p1_w];
                for (int oh = 0; oh < h2; ++oh) {
                    float* z_row = &z2_oc[static_cast<size_t>(oh) * w2_dim];
                    for (int kh = 0; kh < 3; ++kh) {
                        const float* row = &a1_ic[static_cast<size_t>(oh * 2 + kh) * p1_w];
                        const float k0 = w_ptr[kh * 3], k1 = w_ptr[kh * 3 + 1], k2 = w_ptr[kh * 3 + 2];
                        for (int ow = 0; ow < w2_dim; ++ow) {
                            const int iw = ow * 2;
                            z_row[ow] += row[iw] * k0 + row[iw + 1] * k1 + row[iw + 2] * k2;
                        }
                    }
                }
            }
            for (int oh = 0; oh < h2; ++oh) {
                double row_sum = 0.0;
                const float* z_row = &z2_oc[static_cast<size_t>(oh) * w2_dim];
                const double* prev_sat = &sat_oc[static_cast<size_t>(oh) * sat_w];
                double* cur_sat = &sat_oc[static_cast<size_t>(oh + 1) * sat_w];
                for (int ow = 0; ow < w2_dim; ++ow) {
                    const float s = z_row[ow];
                    row_sum += static_cast<double>((s > 0.0f) ? s : 0.1f * s);
                    cur_sat[ow + 1] = prev_sat[ow + 1] + row_sum;
                }
            }
        }

        constexpr double inv_spatial = 1.0 / static_cast<double>(Config::H2 * Config::H2);
        #pragma omp parallel for collapse(2)
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const int y1 = (gy * stride) >> 2, x1 = (gx * stride) >> 2;
                const int y2 = y1 + Config::H2, x2 = x1 + Config::H2;
                std::array<float, Config::C2> gap{};
                for (int oc = 0; oc < Config::C2; ++oc) {
                    const double* sat_oc = &sat[static_cast<size_t>(oc) * sat_h * sat_w];
                    const double sum = sat_oc[static_cast<size_t>(y2) * sat_w + x2]
                                     - sat_oc[static_cast<size_t>(y1) * sat_w + x2]
                                     - sat_oc[static_cast<size_t>(y2) * sat_w + x1]
                                     + sat_oc[static_cast<size_t>(y1) * sat_w + x1];
                    gap[oc] = static_cast<float>(sum * inv_spatial);
                }
                Embedding u{};
                float norm_sq = 1e-12f;
                for (int d = 0; d < Config::EMBED_D; ++d) {
                    float sum = weights_.b3()[d];
                    for (int ic = 0; ic < Config::C2; ++ic) sum += weights_.w3()[d * Config::C2 + ic] * gap[ic];
                    u[d] = sum; norm_sq += sum * sum;
                }
                const float inv_norm = 1.0f / std::sqrt(norm_sq);
                Embedding& u_hat = u_hat_grid[static_cast<size_t>(gy) * grid_w + gx];
                for (int d = 0; d < Config::EMBED_D; ++d) u_hat[d] = u[d] * inv_norm;
            }
        }
        return u_hat_grid;
    }

    NetworkParams& weights() noexcept { return weights_; }
    const NetworkParams& weights() const noexcept { return weights_; }
private:
    NetworkParams weights_{};
};

} // namespace siam
