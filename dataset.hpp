#pragma once

#include "image.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace siam {

inline std::string extract_location_id(const std::string& path) {
    std::string stem = path;
    const size_t slash = stem.find_last_of("/\\");
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    const size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);

    // Look for _rXX_cYY pattern (e.g. S1A_..._VV_r01_c02 or tile_r01_c02)
    const size_t r_pos = stem.rfind("_r");
    if (r_pos != std::string::npos) {
        const size_t c_pos = stem.find("_c", r_pos);
        if (c_pos != std::string::npos && c_pos > r_pos) {
            return stem.substr(r_pos + 1); // e.g. "r01_c02"
        }
    }
    return stem;
}

inline PgmImage try_load_mask(const std::string& path) {
    if (path.empty() || path == "-") return PgmImage();
    std::string mask_path;
    const size_t dot = path.find_last_of('.');
    if (dot != std::string::npos) {
        mask_path = path.substr(0, dot) + "_mask.pgm";
    } else {
        mask_path = path + "_mask.pgm";
    }
    if (access(mask_path.c_str(), R_OK) == 0) {
        try {
            return PgmImage::load(mask_path);
        } catch (...) {
            // Fall back if companion mask cannot be parsed
        }
    }
    return PgmImage();
}

inline void index_valid_stable_patches(ImagePair& pair, float stability_thresh = Config::DEFAULT_STABILITY_THRESH) {
    const int w = pair.pre.width(), h = pair.pre.height();
    constexpr int S = Config::PATCH_S; // 32
    if (w < S || h < S) return;

    pair.valid_locs.clear();

    const bool has_mask = (pair.mask.width() == w && pair.mask.height() == h);
    const float* pre_p = pair.pre.data();
    const float* post_p = pair.post.data();
    const float* mask_p = has_mask ? pair.mask.data() : nullptr;

    const size_t stride_i = static_cast<size_t>(w + 1);
    std::vector<int> int_val(stride_i * static_cast<size_t>(h + 1), 0);
    std::vector<float> int_diff(stride_i * static_cast<size_t>(h + 1), 0.0f);

    for (int y = 0; y < h; ++y) {
        int row_val = 0;
        float row_diff = 0.0f;
        const size_t row_offset = static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const size_t idx = row_offset + x;
            bool valid = true;
            if (mask_p) {
                valid = (mask_p[idx] > 0.5f);
            } else {
                valid = (pre_p[idx] > 0.0f && post_p[idx] > 0.0f);
            }
            const float diff = std::abs(post_p[idx] - pre_p[idx]);

            row_val += valid ? 1 : 0;
            row_diff += diff;

            const size_t i_idx = static_cast<size_t>(y + 1) * stride_i + static_cast<size_t>(x + 1);
            const size_t i_above = static_cast<size_t>(y) * stride_i + static_cast<size_t>(x + 1);

            int_val[i_idx] = int_val[i_above] + row_val;
            int_diff[i_idx] = int_diff[i_above] + row_diff;
        }
    }

    constexpr int S2 = S * S; // 1024
    const float inv_s2 = 1.0f / static_cast<float>(S2);

    for (int y = 0; y <= h - S; y += 2) {
        for (int x = 0; x <= w - S; x += 2) {
            const size_t y0 = static_cast<size_t>(y);
            const size_t y1 = static_cast<size_t>(y + S);
            const size_t x0 = static_cast<size_t>(x);
            const size_t x1 = static_cast<size_t>(x + S);

            const int val_sum = int_val[y1 * stride_i + x1]
                              - int_val[y0 * stride_i + x1]
                              - int_val[y1 * stride_i + x0]
                              + int_val[y0 * stride_i + x0];

            if (val_sum != S2) continue;

            const float diff_sum = int_diff[y1 * stride_i + x1]
                                 - int_diff[y0 * stride_i + x1]
                                 - int_diff[y1 * stride_i + x0]
                                 + int_diff[y0 * stride_i + x0];

            const float mean_diff = diff_sum * inv_s2;
            if (mean_diff <= stability_thresh) {
                pair.valid_locs.push_back((static_cast<uint32_t>(y) << 16) | static_cast<uint32_t>(x));
            }
        }
    }
}

inline void append_pair(std::vector<ImagePair>& pairs, const std::string& pre_path,
                        const std::string& post_path) {
    if (pre_path == "-" && post_path == "-")
        throw std::runtime_error("siam: cannot stream both pre and post images from stdin");
    PgmImage pre = PgmImage::load(pre_path), post = PgmImage::load(post_path);
    if (pre.width() != post.width() || pre.height() != post.height() ||
        pre.width() < Config::PATCH_S || pre.height() < Config::PATCH_S)
        throw std::runtime_error("siam: dimension mismatch or image smaller than patch size (" + pre_path + ", " + post_path + ")");
    
    PgmImage mask = try_load_mask(pre_path);
    if (mask.width() == 0) mask = try_load_mask(post_path);
    std::string loc_id = extract_location_id(pre_path);

    pairs.push_back(ImagePair{std::move(pre), std::move(post), std::move(mask), std::move(loc_id), {}});
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
