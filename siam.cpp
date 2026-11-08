/*
 * siam.cpp
 * C++17 Siamese Contrastive (InfoNCE / Cosine) Engine for PGM satellite images
 *
 * Compile:
 *   g++ -std=c++17 -O3 -march=native -Wall -Wextra siam.cpp -lm -o siam
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace siam {

constexpr int EXIT_NORMAL = 0;
constexpr int EXIT_ALERT = 1;
constexpr int EXIT_TROUBLE = 2;

struct Config {
    static constexpr int PATCH_S = 32;
    static constexpr int C1 = 16;
    static constexpr int H1 = 16;
    static constexpr int C2 = 32;
    static constexpr int H2 = 8;
    static constexpr int EMBED_D = 32;
    static constexpr int BATCH_N = 64;
    static constexpr float TEMP = 0.07f;
    static constexpr int NUM_W1 = C1 * 9;
    static constexpr int NUM_W2 = C2 * C1 * 9;
    static constexpr int NUM_W3 = EMBED_D * C2;
    static constexpr int TOTAL_WEIGHTS = NUM_W1 + NUM_W2 + NUM_W3;
    static constexpr int TOTAL_BIASES = C1 + C2 + EMBED_D;
    static constexpr int TOTAL_PARAMS = TOTAL_WEIGHTS + TOTAL_BIASES;
    static constexpr uint32_t MAGIC = 0x5349414Du;
};

using Embedding = std::array<float, Config::EMBED_D>;
using Patch = std::array<float, Config::PATCH_S * Config::PATCH_S>;

class FileHandle {
public:
    FileHandle(const std::string& path, const char* mode, FILE* std_stream)
        : fp_(path == "-" ? std_stream : std::fopen(path.c_str(), mode)), owns_(path != "-") {
        if (!fp_) throw std::runtime_error("siam: failed to open file: " + path);
    }
    ~FileHandle() { if (owns_ && fp_) std::fclose(fp_); }
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    FILE* get() const noexcept { return fp_; }
private:
    FILE* fp_;
    bool owns_;
};

inline void skip_comments_and_ws(FILE* fp) {
    int ch;
    while ((ch = std::fgetc(fp)) != EOF) {
        if (std::isspace(ch)) continue;
        if (ch == '#') {
            while ((ch = std::fgetc(fp)) != EOF && ch != '\n' && ch != '\r') {}
        } else {
            std::ungetc(ch, fp);
            break;
        }
    }
}

class PgmImage {
public:
    PgmImage() = default;
    PgmImage(int width, int height)
        : width_(width), height_(height),
          pixels_(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f) {}

    static PgmImage load(const std::string& path) {
        FileHandle fh(path, "rb", stdin);
        FILE* fp = fh.get();
        char magic[3] = {0};
        if (std::fread(magic, 1, 2, fp) != 2 ||
            (std::strcmp(magic, "P5") != 0 && std::strcmp(magic, "P2") != 0)) {
            throw std::runtime_error("Error: " + path + " is not a valid P5/P2 PGM file");
        }

        int w = 0, h = 0, maxv = 0;
        skip_comments_and_ws(fp);
        if (std::fscanf(fp, "%d", &w) != 1 || w <= 0)
            throw std::runtime_error("Invalid width in " + path);
        skip_comments_and_ws(fp);
        if (std::fscanf(fp, "%d", &h) != 1 || h <= 0)
            throw std::runtime_error("Invalid height in " + path);
        skip_comments_and_ws(fp);
        if (std::fscanf(fp, "%d", &maxv) != 1 || maxv <= 0 || maxv > 65535)
            throw std::runtime_error("Invalid maxval in " + path);

        /* Consume the raster separator; accept CRLF as one line ending. */
        int sep = std::fgetc(fp);
        if (sep == '\r') {
            int next = std::fgetc(fp);
            if (next != '\n' && next != EOF) std::ungetc(next, fp);
        } else if (sep == EOF || !std::isspace(sep)) {
            throw std::runtime_error("Invalid PGM raster separator in " + path);
        }

        const size_t sw = static_cast<size_t>(w), sh = static_cast<size_t>(h);
        if (sw > SIZE_MAX / sh) throw std::runtime_error("PGM dimensions overflow in " + path);
        PgmImage img(w, h);
        const size_t npixels = img.pixels_.size();
        const float inv_max = 1.0f / static_cast<float>(maxv);

        if (std::strcmp(magic, "P5") == 0) {
            const size_t bpp = (maxv < 256) ? 1 : 2;
            if (npixels > SIZE_MAX / bpp) throw std::runtime_error("PGM payload size overflow in " + path);
            std::vector<uint8_t> buf(npixels * bpp);
            if (std::fread(buf.data(), bpp, npixels, fp) != npixels)
                throw std::runtime_error("Truncated P5 payload in " + path);
            if (bpp == 1) {
                for (size_t i = 0; i < npixels; ++i)
                    img.pixels_[i] = static_cast<float>(buf[i]) * inv_max;
            } else {
                for (size_t i = 0; i < npixels; ++i) {
                    const uint16_t val = (static_cast<uint16_t>(buf[2 * i]) << 8) |
                                         static_cast<uint16_t>(buf[2 * i + 1]);
                    img.pixels_[i] = static_cast<float>(val) * inv_max;
                }
            }
        } else {
            for (size_t i = 0; i < npixels; ++i) {
                int val = 0;
                if (std::fscanf(fp, "%d", &val) != 1)
                    throw std::runtime_error("Truncated P2 payload in " + path);
                img.pixels_[i] = static_cast<float>(val) * inv_max;
            }
        }
        return img;
    }

    void save(const std::string& path) const {
        FileHandle fh(path, "wb", stdout);
        FILE* fp = fh.get();
        const size_t npixels = pixels_.size();
        std::vector<uint8_t> buf(npixels);
        for (size_t i = 0; i < npixels; ++i) {
            const float v = std::clamp(pixels_[i], 0.0f, 1.0f);
            buf[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
        std::fprintf(fp, "P5\n%d %d\n255\n", width_, height_);
        if (std::fwrite(buf.data(), 1, npixels, fp) != npixels)
            throw std::runtime_error("siam: failed writing PGM to " + path);
    }

    void extract_patch(int y0, int x0, Patch& out, float gain = 1.0f,
                       float bias = 0.0f, int d4_mode = 0) const noexcept {
        constexpr int S = Config::PATCH_S;
        for (int dy = 0; dy < S; ++dy) {
            for (int dx = 0; dx < S; ++dx) {
                int sy = dy, sx = dx;
                if (d4_mode & 1) sx = S - 1 - sx;
                if (d4_mode & 2) sy = S - 1 - sy;
                if (d4_mode & 4) std::swap(sx, sy);
                float v = pixels_[(y0 + sy) * width_ + (x0 + sx)] * gain + bias;
                v = std::clamp(v, 0.0f, 1.0f);
                out[dy * S + dx] = (v - 0.5f) * 2.0f;
            }
        }
    }

    PgmImage upsample_bilinear(int out_w, int out_h, int stride) const {
        PgmImage dst(out_w, out_h);
        const int grid_h = height_, grid_w = width_;
        const float half_patch = 0.5f * static_cast<float>(Config::PATCH_S - 1);
        const float inv_stride = 1.0f / static_cast<float>(stride);
        #pragma omp parallel for
        for (int y = 0; y < out_h; ++y) {
            const float fy = std::clamp((static_cast<float>(y) - half_patch) * inv_stride,
                                        0.0f, static_cast<float>(grid_h - 1));
            const int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, grid_h - 1);
            const float wy = fy - static_cast<float>(y0);
            const float* row0 = &pixels_[static_cast<size_t>(y0) * grid_w];
            const float* row1 = &pixels_[static_cast<size_t>(y1) * grid_w];
            float* dst_row = &dst.pixels_[static_cast<size_t>(y) * out_w];
            for (int x = 0; x < out_w; ++x) {
                const float fx = std::clamp((static_cast<float>(x) - half_patch) * inv_stride,
                                            0.0f, static_cast<float>(grid_w - 1));
                const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, grid_w - 1);
                const float wx = fx - static_cast<float>(x0);
                const float top = row0[x0] + wx * (row0[x1] - row0[x0]);
                const float bot = row1[x0] + wx * (row1[x1] - row1[x0]);
                dst_row[x] = top + wy * (bot - top);
            }
        }
        return dst;
    }

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    float* data() noexcept { return pixels_.data(); }
    const float* data() const noexcept { return pixels_.data(); }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<float> pixels_;
};

struct ImagePair { PgmImage pre; PgmImage post; };

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

class ContrastiveTrainer {
public:
    ContrastiveTrainer(SiameseEncoder& encoder, uint32_t seed)
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

inline void append_pair(std::vector<ImagePair>& pairs, const std::string& pre_path,
                        const std::string& post_path) {
    if (pre_path == "-" && post_path == "-")
        throw std::runtime_error("siam: cannot stream both pre and post images from stdin");
    PgmImage pre = PgmImage::load(pre_path), post = PgmImage::load(post_path);
    if (pre.width() != post.width() || pre.height() != post.height() ||
        pre.width() < Config::PATCH_S || pre.height() < Config::PATCH_S)
        throw std::runtime_error("siam: dimension mismatch or image smaller than patch size (" + pre_path + ", " + post_path + ")");
    pairs.push_back(ImagePair{std::move(pre), std::move(post)});
}

inline void load_pair_list(const std::string& list_path, std::vector<ImagePair>& pairs) {
    FileHandle fh(list_path, "r", stdin);
    FILE* fp = fh.get();
    char pre_path[4096], post_path[4096];
    while (true) {
        skip_comments_and_ws(fp);
        const int n1 = std::fscanf(fp, "%4095s", pre_path);
        if (n1 == EOF) break;
        skip_comments_and_ws(fp);
        const int n2 = std::fscanf(fp, "%4095s", post_path);
        if (n1 != 1 || n2 != 1)
            throw std::runtime_error("siam: pair list " + list_path + " has an unpaired image path");
        append_pair(pairs, pre_path, post_path);
    }
}

inline std::vector<ImagePair> load_all_pairs(int argc, char** argv, int optind,
                                              const std::string& list_path) {
    std::vector<ImagePair> pairs;
    const int pos_args = argc - optind;
    if (!list_path.empty()) load_pair_list(list_path, pairs);
    if (pos_args == 1 && std::strcmp(argv[optind], "-") == 0) {
        load_pair_list("-", pairs);
    } else if (pos_args > 0) {
        if (pos_args % 2 != 0) throw std::runtime_error("siam: positional image arguments must be paired");
        for (int i = optind; i < argc; i += 2) append_pair(pairs, argv[i], argv[i + 1]);
    }
    if (pairs.empty()) throw std::runtime_error("siam: no valid image pairs provided");
    return pairs;
}

} // namespace siam

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [-tecqu] [-w weights] [-o out.pgm] [-L list.txt] [-n steps] [-l lr]\n"
        "            [-a tile_thresh] [-f alert_frac] [-s stride] [-S seed]\n"
        "            <pre.pgm> <post.pgm> [pre2.pgm post2.pgm ...]\n", prog);
}

int main(int argc, char** argv) {
    using namespace siam;
    bool train_mode = false, eval_mode = false, continue_weights = false;
    bool upsample = false, quiet = false;
    int steps = 200, stride = 8;
    float lr = 0.01f, tile_thresh = 0.35f, alert_frac = 0.05f;
    uint32_t seed = static_cast<uint32_t>(std::time(nullptr)) ^ static_cast<uint32_t>(getpid());
    std::string weights_path = "siam.w", list_path, out_path = "-";
    int opt;
    while ((opt = getopt(argc, argv, "tecquw:L:o:n:l:a:f:s:S:h")) != -1) {
        switch (opt) {
            case 't': train_mode = true; break;
            case 'e': eval_mode = true; break;
            case 'c': continue_weights = true; break;
            case 'u': upsample = true; break;
            case 'q': quiet = true; break;
            case 'w': weights_path = optarg; break;
            case 'L': list_path = optarg; break;
            case 'o': out_path = optarg; break;
            case 'n': steps = std::atoi(optarg); break;
            case 'l': lr = std::strtof(optarg, nullptr); break;
            case 'a': tile_thresh = std::strtof(optarg, nullptr); break;
            case 'f': alert_frac = std::strtof(optarg, nullptr); break;
            case 's': stride = std::atoi(optarg); break;
            case 'S': seed = static_cast<uint32_t>(std::strtoul(optarg, nullptr, 10)); break;
            case 'h': usage(argv[0]); return EXIT_NORMAL;
            default: usage(argv[0]); return EXIT_TROUBLE;
        }
    }
    if ((train_mode && eval_mode) || stride < 1 || steps < 1) {
        usage(argv[0]); return EXIT_TROUBLE;
    }
    try {
        SiameseEncoder encoder;
        if (train_mode) {
            auto pairs = load_all_pairs(argc, argv, optind, list_path);
            if (!quiet && pairs.size() > 1)
                std::fprintf(stderr, "siam: loaded %zu image pairs for training\n", pairs.size());
            std::mt19937 init_rng(seed);
            if (continue_weights) encoder.load_weights(weights_path);
            else encoder.init_kaiming(init_rng);
            ContrastiveTrainer trainer(encoder, seed);
            trainer.train(pairs, steps, lr, quiet);
            encoder.save_weights(weights_path);
            return EXIT_NORMAL;
        }
        if (eval_mode) {
            auto pairs = load_all_pairs(argc, argv, optind, list_path);
            encoder.load_weights(weights_path);
            ChangeDetector detector(encoder);
            detector.evaluate(pairs, stride, tile_thresh, alert_frac);
            return EXIT_NORMAL;
        }
        if (!list_path.empty() || (argc - optind) != 2) {
            usage(argv[0]); return EXIT_TROUBLE;
        }
        std::vector<ImagePair> pairs;
        append_pair(pairs, argv[optind], argv[optind + 1]);
        encoder.load_weights(weights_path);
        ChangeDetector detector(encoder);
        const bool is_alert = detector.infer(pairs[0], stride, tile_thresh, alert_frac,
                                             upsample, out_path, quiet);
        return is_alert ? EXIT_ALERT : EXIT_NORMAL;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return EXIT_TROUBLE;
    }
}
