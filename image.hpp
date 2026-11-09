#pragma once

#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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
private:
    struct Mapping {
        Mapping(void* address, size_t length) : address(address), length(length) {}
        ~Mapping() { if (address != MAP_FAILED) munmap(address, length); }
        void* address;
        size_t length;
    };

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
        const size_t npixels = sw * sh;
        const float inv_max = 1.0f / static_cast<float>(maxv);

        if (std::strcmp(magic, "P5") == 0) {
            const size_t bpp = (maxv < 256) ? 1 : 2;
            if (npixels > SIZE_MAX / bpp) throw std::runtime_error("PGM payload size overflow in " + path);

            /* Keep regular-file P5 pixels in the kernel-backed mapping. */
            if (path != "-") {
                const int fd = fileno(fp);
                struct stat st {};
                const off_t payload_offset = ftello(fp);
                if (fd >= 0 && payload_offset >= 0 && fstat(fd, &st) == 0 &&
                    S_ISREG(st.st_mode) && st.st_size >= payload_offset) {
                    const uintmax_t file_size_u = static_cast<uintmax_t>(st.st_size);
                    if (file_size_u > static_cast<uintmax_t>(SIZE_MAX))
                        throw std::runtime_error("PGM file is too large to map: " + path);
                    const size_t file_size = static_cast<size_t>(file_size_u);
                    const size_t offset = static_cast<size_t>(payload_offset);
                    const size_t payload_size = npixels * bpp;
                    if (payload_size > file_size - offset)
                        throw std::runtime_error("Truncated P5 payload in " + path);
                    void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
                    if (base == MAP_FAILED)
                        throw std::runtime_error("siam: failed to map PGM file: " + path);

                    std::shared_ptr<Mapping> mapping;
                    try {
                        mapping = std::make_shared<Mapping>(base, file_size);
                    } catch (...) {
                        munmap(base, file_size);
                        throw;
                    }

                    PgmImage img;
                    img.width_ = w;
                    img.height_ = h;
                    img.mapping_ = std::move(mapping);
                    img.mapped_payload_ = static_cast<const uint8_t*>(base) + offset;
                    img.mapped_bytes_per_pixel_ = bpp;
                    img.mapped_scale_ = inv_max;
                    return img;
                }
            }

            PgmImage img(w, h);
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
            return img;
        } else {
            PgmImage img(w, h);
            for (size_t i = 0; i < npixels; ++i) {
                int val = 0;
                if (std::fscanf(fp, "%d", &val) != 1)
                    throw std::runtime_error("Truncated P2 payload in " + path);
                img.pixels_[i] = static_cast<float>(val) * inv_max;
            }
            return img;
        }
    }

    void save(const std::string& path) const {
        FileHandle fh(path, "wb", stdout);
        FILE* fp = fh.get();
        const size_t npixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        std::vector<uint8_t> buf(npixels);
        for (size_t i = 0; i < npixels; ++i) {
            const float v = std::clamp(pixel(i), 0.0f, 1.0f);
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
                const size_t index = static_cast<size_t>(y0 + sy) * width_ + (x0 + sx);
                float v = pixel(index) * gain + bias;
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
            float* dst_row = &dst.pixels_[static_cast<size_t>(y) * out_w];
            for (int x = 0; x < out_w; ++x) {
                const float fx = std::clamp((static_cast<float>(x) - half_patch) * inv_stride,
                                            0.0f, static_cast<float>(grid_w - 1));
                const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, grid_w - 1);
                const float wx = fx - static_cast<float>(x0);
                const float top0 = pixel(static_cast<size_t>(y0) * grid_w + x0);
                const float top1 = pixel(static_cast<size_t>(y0) * grid_w + x1);
                const float bot0 = pixel(static_cast<size_t>(y1) * grid_w + x0);
                const float bot1 = pixel(static_cast<size_t>(y1) * grid_w + x1);
                const float top = top0 + wx * (top1 - top0);
                const float bot = bot0 + wx * (bot1 - bot0);
                dst_row[x] = top + wy * (bot - top);
            }
        }
        return dst;
    }

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    float* data() { materialize(); return pixels_.data(); }
    const float* data() const { materialize(); return pixels_.data(); }

    float pixel(size_t index) const noexcept {
        if (!mapped_payload_) return pixels_[index];
        if (mapped_bytes_per_pixel_ == 1)
            return static_cast<float>(mapped_payload_[index]) * mapped_scale_;
        const size_t byte_index = index * 2;
        const uint16_t value = (static_cast<uint16_t>(mapped_payload_[byte_index]) << 8) |
                               static_cast<uint16_t>(mapped_payload_[byte_index + 1]);
        return static_cast<float>(value) * mapped_scale_;
    }

    void copy_pixels(size_t offset, size_t count, float* destination) const noexcept {
        for (size_t i = 0; i < count; ++i) destination[i] = pixel(offset + i);
    }

private:
    void materialize() const {
        if (!mapped_payload_) return;
        const size_t count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
        std::vector<float> converted(count);
        for (size_t i = 0; i < count; ++i) converted[i] = pixel(i);
        pixels_.swap(converted);
        mapped_payload_ = nullptr;
        mapped_bytes_per_pixel_ = 0;
        mapping_.reset();
    }

    int width_ = 0;
    int height_ = 0;
    mutable std::vector<float> pixels_;
    mutable std::shared_ptr<Mapping> mapping_;
    mutable const uint8_t* mapped_payload_ = nullptr;
    mutable size_t mapped_bytes_per_pixel_ = 0;
    float mapped_scale_ = 1.0f;
};

struct ImagePair { PgmImage pre; PgmImage post; };

} // namespace siam
