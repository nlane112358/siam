/*
 * main.cpp
 * C++17 command-line entry point for the siam change detection engine.
 */

#include "config.hpp"
#include "image.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include "detector.hpp"
#include "dataset.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

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
