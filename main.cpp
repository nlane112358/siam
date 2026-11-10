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

static void usage(FILE* out) {
    std::fprintf(out,
        "NAME\n"
        "    siam - train and run a Siamese image-change detector\n"
        "\n"
        "SYNOPSIS\n"
        "    siam [options] PRE POST\n"
        "    siam -t [options] [PRE POST]...\n"
        "    siam -e [options] [PRE POST]...\n"
        "    siam -C [options] [PRE POST]...\n"
        "    siam -h\n"
        "\n"
        "DESCRIPTION\n"
        "    PRE and POST are P2 or P5 PGM images (8-bit or 16-bit) with matching\n"
        "    dimensions of at least 32x32 pixels. Training, evaluation, and calibration\n"
        "    accept image pairs as operands, from a pair list (-L FILE), or from standard\n"
        "    input (-). Each pair-list entry is PRE POST.\n"
        "\n"
        "OPTIONS\n"
        "    -h              display help and exit\n"
        "    -t              train\n"
        "    -e              evaluate matched and mismatched pairs (at least 2 pairs)\n"
        "    -C              calibrate threshold on baseline/historical pairs\n"
        "    -c              continue training from saved weights\n"
        "    -q              suppress training progress and inference status\n"
        "    -u              upsample the inference map to input dimensions\n"
        "    -w FILE         weights file [siam.w]\n"
        "    -L FILE         read image pairs from FILE; '-' means standard input\n"
        "    -o FILE         write output map or calibrated threshold to FILE; '-' means stdout\n"
        "    -v FILE         write POST tile embeddings as a PCA false-color P6 PPM; '-' means stdout\n"
        "    -D FILE         write anomalous backscatter-drop tiles to FILE\n"
        "    -I FILE         write anomalous backscatter-increase tiles to FILE\n"
        "    -n STEPS        training steps [200]\n"
        "    -l RATE         learning rate [0.01]\n"
        "    -S SEED         random seed [time and process ID]\n"
        "    -a VALUE        tile distance threshold [0.35]\n"
        "    -f VALUE        alert fraction / calibration target P_fa [0.05]\n"
        "    -s PIXELS       sliding-window stride [8]\n"
        "    -T VALUE        temporal stability threshold for positive training patches [0.20]\n"
        "\n"
        "EXIT STATUS\n"
        "    0   normal result or successful training/evaluation\n"
        "    1   anomaly alert\n"
        "    2   invalid arguments or I/O/runtime error\n"
        "\n"
        "EXAMPLES\n"
        "    siam -w siam.w -o change.pgm before.pgm after.pgm\n"
        "    siam -w siam.w -o change.pgm -v embedding.ppm before.pgm after.pgm\n"
        "    siam -w siam.w -o change.pgm -D drop.pgm -I increase.pgm before.pgm after.pgm\n"
        "    siam -t -w siam.w -n 200 -L train_pairs.txt\n"
        "    siam -e -w siam.w -L heldout_pairs.txt\n"
        "    siam -C -w siam.w -L calib_pairs.txt -f 0.005 -o calib_thresh.txt\n");
}

int main(int argc, char** argv) {
    using namespace siam;
    bool train_mode = false, eval_mode = false, calib_mode = false, continue_weights = false;
    bool upsample = false, quiet = false;
    int steps = 200, stride = 8;
    float lr = 0.01f, tile_thresh = 0.35f, alert_frac = 0.05f;
    float stability_thresh = Config::DEFAULT_STABILITY_THRESH;
    uint32_t seed = static_cast<uint32_t>(std::time(nullptr)) ^ static_cast<uint32_t>(getpid());
    std::string weights_path = "siam.w", list_path, out_path = "-", embedding_map_path;
    std::string drop_path, increase_path;
    bool embedding_map_requested = false;
    int opt;
    while ((opt = getopt(argc, argv, "tCeqquv:w:L:o:D:I:n:l:a:f:s:S:T:h")) != -1) {
        switch (opt) {
            case 't': train_mode = true; break;
            case 'C': calib_mode = true; break;
            case 'e': eval_mode = true; break;
            case 'c': continue_weights = true; break;
            case 'u': upsample = true; break;
            case 'q': quiet = true; break;
            case 'v': embedding_map_path = optarg; embedding_map_requested = true; break;
            case 'w': weights_path = optarg; break;
            case 'L': list_path = optarg; break;
            case 'o': out_path = optarg; break;
            case 'D': drop_path = optarg; break;
            case 'I': increase_path = optarg; break;
            case 'n': steps = std::atoi(optarg); break;
            case 'l': lr = std::strtof(optarg, nullptr); break;
            case 'a': tile_thresh = std::strtof(optarg, nullptr); break;
            case 'f': alert_frac = std::strtof(optarg, nullptr); break;
            case 's': stride = std::atoi(optarg); break;
            case 'S': seed = static_cast<uint32_t>(std::strtoul(optarg, nullptr, 10)); break;
            case 'T': stability_thresh = std::strtof(optarg, nullptr); break;
            case 'h': usage(stdout); return EXIT_NORMAL;
            default: usage(stderr); return EXIT_TROUBLE;
        }
    }
    const bool directional = !drop_path.empty() || !increase_path.empty();
    const int mode_count = (train_mode ? 1 : 0) + (eval_mode ? 1 : 0) + (calib_mode ? 1 : 0);
    if (mode_count > 1 || stride < 1 || steps < 1 ||
        (drop_path.empty() != increase_path.empty()) ||
        (directional && (mode_count > 0 || drop_path == "-" || increase_path == "-")) ||
        (directional && (drop_path == increase_path || drop_path == out_path || increase_path == out_path)) ||
        (embedding_map_requested && (embedding_map_path.empty() || mode_count > 0 ||
            embedding_map_path == out_path ||
            (directional && (embedding_map_path == drop_path || embedding_map_path == increase_path))))) {
        usage(stderr); return EXIT_TROUBLE;
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
            trainer.train(pairs, steps, lr, quiet, stability_thresh);
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
        if (calib_mode) {
            auto pairs = load_all_pairs(argc, argv, optind, list_path);
            encoder.load_weights(weights_path);
            ChangeDetector detector(encoder);
            detector.calibrate(pairs, stride, alert_frac, out_path, quiet);
            return EXIT_NORMAL;
        }
        if (!list_path.empty() || (argc - optind) != 2) {
            usage(stderr); return EXIT_TROUBLE;
        }
        std::vector<ImagePair> pairs;
        append_pair(pairs, argv[optind], argv[optind + 1]);
        encoder.load_weights(weights_path);
        ChangeDetector detector(encoder);
        const bool is_alert = detector.infer(pairs[0], stride, tile_thresh, alert_frac,
                                             upsample, out_path, drop_path, increase_path,
                                             embedding_map_path, quiet);
        return is_alert ? EXIT_ALERT : EXIT_NORMAL;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return EXIT_TROUBLE;
    }
}
