#pragma once

#include "image.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace siam {

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
