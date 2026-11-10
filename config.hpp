#pragma once

#include <array>
#include <cstdint>
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

} // namespace siam
