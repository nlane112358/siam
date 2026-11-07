/*
 * main.c
 * CLI Entry Point & Mode Dispatcher for pgmcd
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "dataset.h"
#include "detector.h"
#include "image.h"
#include "network.h"
#include "trainer.h"

/* Exit status codes following diff(1) and cmp(1) conventions */
#define EXIT_NORMAL     0 /* No disaster anomaly detected (or training succeeded) */
#define EXIT_ALERT      1 /* Disaster anomaly detected above scene threshold */
#define EXIT_TROUBLE    2 /* Invalid arguments, I/O error, or dimension mismatch */

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [-tecq] [-w weights] [-o out.pgm] [-L list.txt] [-n steps] [-l lr]\n"
        "             [-a tile_thresh] [-f alert_frac] [-s stride] [-S seed]\n"
        "             <pre.pgm> <post.pgm> [pre2.pgm post2.pgm ...]\n"
        "options:\n"
        "  -t            train mode (accepts multiple pairs or '-' stdin pair list)\n"
        "  -e            evaluate mode: matched vs. mismatched tile ROC-AUC (no labels)\n"
        "  -c            continue/fine-tune from existing weights file\n"
        "  -w <file>     weights file path (default: pgmcd.w)\n"
        "  -L <file>     read <pre> <post> path pairs from file ('-' for stdin)\n"
        "  -o <file>     output change map PGM (default: '-' for stdout)\n"
        "  -n <int>      training steps (default: 200)\n"
        "  -l <float>    learning rate (default: 0.01)\n"
        "  -a <float>    tile cosine-distance anomaly threshold [0,1] (default: 0.35)\n"
        "  -f <float>    scene alert fraction threshold [0,1] (default: 0.05)\n"
        "  -s <int>      inference sliding-window stride in pixels (default: 8)\n"
        "  -S <uint>     random seed (default: time-based)\n"
        "  -q            quiet mode (suppress stderr diagnostics)\n"
        "exit status:\n"
        "   0            normal (or training/evaluation complete)\n"
        "   1            disaster anomaly alert triggered\n"
        "   2            trouble (I/O or argument error)\n",
        prog);
}

int main(int argc, char **argv) {
    int train_mode = 0;
    int eval_mode = 0;
    int continue_weights = 0;
    int quiet = 0;
    int steps = 200;
    int stride = 8;
    float lr = 0.01f;
    float tile_thresh = 0.35f;
    float alert_frac = 0.05f;
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    const char *weights_path = "pgmcd.w";
    const char *list_path = NULL;
    const char *out_path = "-"; /* stdout by default */

    int opt;
    while ((opt = getopt(argc, argv, "tecqw:L:o:n:l:a:f:s:S:h")) != -1) {
        switch (opt) {
            case 't': train_mode = 1; break;
            case 'e': eval_mode = 1; break;
            case 'c': continue_weights = 1; break;
            case 'q': quiet = 1; break;
            case 'w': weights_path = optarg; break;
            case 'L': list_path = optarg; break;
            case 'o': out_path = optarg; break;
            case 'n': steps = atoi(optarg); break;
            case 'l': lr = strtof(optarg, NULL); break;
            case 'a': tile_thresh = strtof(optarg, NULL); break;
            case 'f': alert_frac = strtof(optarg, NULL); break;
            case 's': stride = atoi(optarg); break;
            case 'S': seed = (unsigned int)strtoul(optarg, NULL, 10); break;
            case 'h':
                usage(argv[0]);
                return EXIT_NORMAL;
            default:
                usage(argv[0]);
                return EXIT_TROUBLE;
        }
    }

    if (train_mode && eval_mode) {
        fprintf(stderr, "pgmcd: cannot specify both -t (train) and -e (evaluate)\n");
        return EXIT_TROUBLE;
    }
    if (stride < 1 || steps < 1) {
        fprintf(stderr, "pgmcd: stride (-s) and steps (-n) must be >= 1\n");
        return EXIT_TROUBLE;
    }

    srand(seed);

    int pos_args = argc - optind;
    SiameseNet net = {0};
    PairDataset dataset;
    pair_dataset_init(&dataset);

    if (train_mode) {
        if (pair_dataset_load_cli(&dataset, argc, argv, optind, list_path) != 0) {
            usage(argv[0]);
            pair_dataset_destroy(&dataset);
            return EXIT_TROUBLE;
        }

        if (!quiet && dataset.count > 1) {
            fprintf(stderr, "pgmcd: loaded %zu image pairs for training\n", dataset.count);
        }

        if (continue_weights) {
            if (siamese_net_load(&net, weights_path) != 0) {
                pair_dataset_destroy(&dataset);
                return EXIT_TROUBLE;
            }
        } else {
            siamese_net_init(&net);
        }

        TrainerConfig tcfg = {
            .steps = steps,
            .lr = lr,
            .weight_decay = 1e-4f,
            .quiet = quiet
        };
        ContrastiveTrainer trainer;
        if (trainer_init(&trainer, tcfg) != 0) {
            pair_dataset_destroy(&dataset);
            return EXIT_TROUBLE;
        }

        trainer_fit(&trainer, &net, &dataset);
        int save_rc = siamese_net_save(&net, weights_path);

        trainer_destroy(&trainer);
        pair_dataset_destroy(&dataset);
        return (save_rc == 0) ? EXIT_NORMAL : EXIT_TROUBLE;
    }

    DetectorConfig dcfg = {
        .stride = stride,
        .tile_thresh = tile_thresh,
        .alert_frac = alert_frac,
        .quiet = quiet
    };
    ChangeDetector detector;

    if (eval_mode) {
        if (pair_dataset_load_cli(&dataset, argc, argv, optind, list_path) != 0 ||
            dataset.count < 2) {
            fprintf(stderr, "pgmcd: evaluate mode (-e) requires at least 2 image pairs\n");
            pair_dataset_destroy(&dataset);
            return EXIT_TROUBLE;
        }

        if (siamese_net_load(&net, weights_path) != 0) {
            pair_dataset_destroy(&dataset);
            return EXIT_TROUBLE;
        }
        change_detector_init(&detector, &net, dcfg);

        EvalMetrics metrics;
        if (change_detector_evaluate(&detector, &dataset, &metrics) != 0) {
            pair_dataset_destroy(&dataset);
            return EXIT_TROUBLE;
        }

        eval_metrics_print(&metrics, &dcfg);
        pair_dataset_destroy(&dataset);
        return EXIT_NORMAL;
    }

    /* Default mode: Inference (expects exactly <pre.pgm> <post.pgm>) */
    if (list_path || pos_args != 2) {
        usage(argv[0]);
        return EXIT_TROUBLE;
    }

    if (pair_dataset_append(&dataset, argv[optind], argv[optind + 1]) != 0) {
        pair_dataset_destroy(&dataset);
        return EXIT_TROUBLE;
    }

    if (siamese_net_load(&net, weights_path) != 0) {
        pair_dataset_destroy(&dataset);
        return EXIT_TROUBLE;
    }
    change_detector_init(&detector, &net, dcfg);

    DetectionResult result;
    if (change_detector_infer(&detector, &dataset.items[0], &result) != 0) {
        pair_dataset_destroy(&dataset);
        return EXIT_TROUBLE;
    }

    if (pgm_image_write(&result.change_map, out_path) != 0) {
        detection_result_destroy(&result);
        pair_dataset_destroy(&dataset);
        return EXIT_TROUBLE;
    }

    int is_alert = result.is_alert;
    detection_result_destroy(&result);
    pair_dataset_destroy(&dataset);
    return is_alert ? EXIT_ALERT : EXIT_NORMAL;
}
