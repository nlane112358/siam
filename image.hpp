#pragma once

#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace siam {

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

} // namespace siam
