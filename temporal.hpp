#pragma once

#include "config.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace siam {

struct TemporalVmfState {
    int image_w = 0;
    int image_h = 0;
    int stride = 0;
    int grid_w = 0;
    int grid_h = 0;
    std::vector<Embedding> mean;
    std::vector<float> kappa;

    static bool load(const std::string& path, int expected_image_w, int expected_image_h,
                     int expected_stride, int expected_grid_w, int expected_grid_h,
                     TemporalVmfState& state) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            if (errno == ENOENT) return false;
            throw std::runtime_error("siam: failed opening temporal state " + path);
        }

        try {
            char magic[8]{};
            uint32_t version = 0, image_w = 0, image_h = 0, stride = 0;
            uint32_t grid_w = 0, grid_h = 0, embed_d = 0;
            uint64_t tile_count = 0;
            const char expected_magic[8] = {'S', 'I', 'A', 'M', 'V', 'M', 'F', '1'};
            if (std::fread(magic, 1, sizeof(magic), fp) != sizeof(magic) ||
                !read_u32(fp, version) || !read_u32(fp, image_w) || !read_u32(fp, image_h) ||
                !read_u32(fp, stride) || !read_u32(fp, grid_w) || !read_u32(fp, grid_h) ||
                !read_u32(fp, embed_d) || !read_u64(fp, tile_count))
                throw std::runtime_error("siam: truncated temporal state header in " + path);
            if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0 || version != 1 ||
                embed_d != Config::EMBED_D || image_w != static_cast<uint32_t>(expected_image_w) ||
                image_h != static_cast<uint32_t>(expected_image_h) ||
                stride != static_cast<uint32_t>(expected_stride) ||
                grid_w != static_cast<uint32_t>(expected_grid_w) ||
                grid_h != static_cast<uint32_t>(expected_grid_h) ||
                tile_count != static_cast<uint64_t>(expected_grid_w) * expected_grid_h)
                throw std::runtime_error("siam: temporal state geometry or version mismatch in " + path);

            TemporalVmfState loaded;
            loaded.image_w = expected_image_w;
            loaded.image_h = expected_image_h;
            loaded.stride = expected_stride;
            loaded.grid_w = expected_grid_w;
            loaded.grid_h = expected_grid_h;
            loaded.mean.resize(static_cast<size_t>(tile_count));
            loaded.kappa.resize(static_cast<size_t>(tile_count));
            for (size_t tile = 0; tile < loaded.mean.size(); ++tile) {
                for (int d = 0; d < Config::EMBED_D; ++d) {
                    uint16_t encoded = 0;
                    if (!read_u16(fp, encoded))
                        throw std::runtime_error("siam: truncated temporal state payload in " + path);
                    loaded.mean[tile][d] = half_to_float(encoded);
                    if (!std::isfinite(loaded.mean[tile][d]))
                        throw std::runtime_error("siam: non-finite mean in temporal state " + path);
                }
                uint16_t encoded_kappa = 0;
                if (!read_u16(fp, encoded_kappa))
                    throw std::runtime_error("siam: truncated temporal state payload in " + path);
                loaded.kappa[tile] = half_to_float(encoded_kappa);
                if (!std::isfinite(loaded.kappa[tile]) || loaded.kappa[tile] < 0.0f)
                    throw std::runtime_error("siam: invalid concentration in temporal state " + path);

                float norm_sq = 0.0f;
                for (const float value : loaded.mean[tile]) norm_sq += value * value;
                if (!(norm_sq > 1e-12f))
                    throw std::runtime_error("siam: zero mean direction in temporal state " + path);
                const float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (float& value : loaded.mean[tile]) value *= inv_norm;
            }
            if (std::fgetc(fp) != EOF || std::ferror(fp))
                throw std::runtime_error("siam: invalid trailing data in temporal state " + path);
            if (std::fclose(fp) != 0) {
                fp = nullptr;
                throw std::runtime_error("siam: failed closing temporal state " + path);
            }
            fp = nullptr;
            state = std::move(loaded);
            return true;
        } catch (...) {
            if (fp) std::fclose(fp);
            throw;
        }
    }

    void initialize(const std::vector<Embedding>& observations,
                    int width, int height, int tile_stride, int tiles_w, int tiles_h) {
        const size_t count = static_cast<size_t>(tiles_w) * tiles_h;
        if (observations.size() != count)
            throw std::runtime_error("siam: invalid embedding grid for temporal initialization");
        image_w = width;
        image_h = height;
        stride = tile_stride;
        grid_w = tiles_w;
        grid_h = tiles_h;
        mean = observations;
        kappa.assign(count, 1.0f);
        normalize_means();
    }

    void update(const std::vector<Embedding>& observations, float decay = 0.99f) {
        if (observations.size() != mean.size() || kappa.size() != mean.size())
            throw std::runtime_error("siam: temporal observation count does not match state");
        for (size_t tile = 0; tile < mean.size(); ++tile) {
            Embedding natural{};
            const float prior_weight = decay * kappa[tile];
            for (int d = 0; d < Config::EMBED_D; ++d)
                natural[d] = prior_weight * mean[tile][d] + observations[tile][d];
            float norm_sq = 0.0f;
            for (const float value : natural) norm_sq += value * value;
            const float norm = std::sqrt(norm_sq);
            if (!(norm > 1e-8f) || !std::isfinite(norm)) {
                mean[tile] = observations[tile];
                kappa[tile] = 1.0f;
            } else {
                kappa[tile] = norm;
                const float inv_norm = 1.0f / norm;
                for (int d = 0; d < Config::EMBED_D; ++d)
                    mean[tile][d] = natural[d] * inv_norm;
            }
        }
    }

    void save_atomic(const std::string& path) const {
        if (path.empty() || path == "-")
            throw std::runtime_error("siam: temporal state requires a filesystem path");
        const size_t count = static_cast<size_t>(grid_w) * grid_h;
        if (grid_w < 1 || grid_h < 1 || mean.size() != count || kappa.size() != count)
            throw std::runtime_error("siam: invalid temporal state for saving");

        const std::string temp_path = path + ".tmp";
        FILE* fp = std::fopen(temp_path.c_str(), "wb");
        if (!fp) throw std::runtime_error("siam: failed opening temporal state temp file " + temp_path);
        bool ok = true;
        const char magic[8] = {'S', 'I', 'A', 'M', 'V', 'M', 'F', '1'};
        ok = ok && std::fwrite(magic, 1, sizeof(magic), fp) == sizeof(magic);
        ok = ok && write_u32(fp, 1);
        ok = ok && write_u32(fp, static_cast<uint32_t>(image_w));
        ok = ok && write_u32(fp, static_cast<uint32_t>(image_h));
        ok = ok && write_u32(fp, static_cast<uint32_t>(stride));
        ok = ok && write_u32(fp, static_cast<uint32_t>(grid_w));
        ok = ok && write_u32(fp, static_cast<uint32_t>(grid_h));
        ok = ok && write_u32(fp, Config::EMBED_D);
        ok = ok && write_u64(fp, static_cast<uint64_t>(count));
        for (size_t tile = 0; ok && tile < count; ++tile) {
            for (int d = 0; ok && d < Config::EMBED_D; ++d)
                ok = write_u16(fp, float_to_half(mean[tile][d]));
            if (ok) ok = write_u16(fp, float_to_half(kappa[tile]));
        }
        if (std::fflush(fp) != 0 || std::ferror(fp)) ok = false;
        if (std::fclose(fp) != 0) ok = false;
        if (!ok) {
            std::remove(temp_path.c_str());
            throw std::runtime_error("siam: failed writing temporal state " + path);
        }
        if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
            std::remove(temp_path.c_str());
            throw std::runtime_error("siam: failed replacing temporal state " + path);
        }
    }

private:
    static bool read_u16(FILE* fp, uint16_t& value) noexcept {
        unsigned char bytes[2];
        if (std::fread(bytes, 1, 2, fp) != 2) return false;
        value = static_cast<uint16_t>(bytes[0]) |
                static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8);
        return true;
    }

    static bool read_u32(FILE* fp, uint32_t& value) noexcept {
        unsigned char bytes[4];
        if (std::fread(bytes, 1, 4, fp) != 4) return false;
        value = static_cast<uint32_t>(bytes[0]) |
                (static_cast<uint32_t>(bytes[1]) << 8) |
                (static_cast<uint32_t>(bytes[2]) << 16) |
                (static_cast<uint32_t>(bytes[3]) << 24);
        return true;
    }

    static bool read_u64(FILE* fp, uint64_t& value) noexcept {
        uint32_t lo = 0, hi = 0;
        if (!read_u32(fp, lo) || !read_u32(fp, hi)) return false;
        value = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
        return true;
    }

    static bool write_u16(FILE* fp, uint16_t value) noexcept {
        const unsigned char bytes[2] = {
            static_cast<unsigned char>(value & 0xffu),
            static_cast<unsigned char>((value >> 8) & 0xffu)
        };
        return std::fwrite(bytes, 1, 2, fp) == 2;
    }

    static bool write_u32(FILE* fp, uint32_t value) noexcept {
        const unsigned char bytes[4] = {
            static_cast<unsigned char>(value & 0xffu),
            static_cast<unsigned char>((value >> 8) & 0xffu),
            static_cast<unsigned char>((value >> 16) & 0xffu),
            static_cast<unsigned char>((value >> 24) & 0xffu)
        };
        return std::fwrite(bytes, 1, 4, fp) == 4;
    }

    static bool write_u64(FILE* fp, uint64_t value) noexcept {
        return write_u32(fp, static_cast<uint32_t>(value & 0xffffffffu)) &&
               write_u32(fp, static_cast<uint32_t>(value >> 32));
    }

    static uint16_t float_to_half(float value) noexcept {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t sign = (bits >> 16) & 0x8000u;
        const uint32_t exponent = (bits >> 23) & 0xffu;
        uint32_t mantissa = bits & 0x7fffffu;
        if (exponent == 0xffu)
            return static_cast<uint16_t>(sign | (mantissa ? 0x7e00u : 0x7c00u));

        int half_exponent = static_cast<int>(exponent) - 127 + 15;
        if (half_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        if (half_exponent <= 0) {
            if (half_exponent < -10) return static_cast<uint16_t>(sign);
            mantissa |= 0x800000u;
            const unsigned shift = static_cast<unsigned>(14 - half_exponent);
            uint32_t rounded = mantissa >> shift;
            const uint32_t remainder = mantissa & ((1u << shift) - 1u);
            const uint32_t halfway = 1u << (shift - 1u);
            if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
            return static_cast<uint16_t>(sign | rounded);
        }

        uint32_t rounded = mantissa >> 13;
        const uint32_t remainder = mantissa & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
            ++rounded;
            if (rounded == 0x400u) {
                rounded = 0;
                if (++half_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | rounded);
    }

    static float half_to_float(uint16_t value) noexcept {
        const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
        uint32_t exponent = (value >> 10) & 0x1fu;
        uint32_t mantissa = value & 0x03ffu;
        uint32_t bits = 0;
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                int unbiased = -14;
                while ((mantissa & 0x0400u) == 0) {
                    mantissa <<= 1;
                    --unbiased;
                }
                mantissa &= 0x03ffu;
                bits = sign | (static_cast<uint32_t>(unbiased + 127) << 23) |
                       (mantissa << 13);
            }
        } else if (exponent == 0x1fu) {
            bits = sign | 0x7f800000u | (mantissa << 13);
        } else {
            exponent = exponent - 15u + 127u;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
        float result = 0.0f;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    void normalize_means() {
        for (Embedding& direction : mean) {
            float norm_sq = 0.0f;
            for (float value : direction) norm_sq += value * value;
            const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-12f));
            for (float& value : direction) value *= inv_norm;
        }
    }
};

} // namespace siam
