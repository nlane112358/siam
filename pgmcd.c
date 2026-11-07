/*
 * pgmcd.c
 * C11 Siamese Contrastive (InfoNCE / Cosine) Engine for PGM Satellite Images
 *
 * Compile:
 *   cc -O3 -march=native -Wall -Wextra -Wno-unknown-pragmas pgmcd.c -lm -o pgmcd
 *
 * Usage: 
 *  ./pgmcd -t -w flood.w -n 300 pre.pgm post.pgm
 *  ./pgmcd -w flood.w -a 0.35 pre.pgm post.pgm > change_map.pgm || notify_alert   
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h> 
#include <ctype.h> 
#include <math.h>
#include <unistd.h>

/* Exit status codes following diff(1) and cmp(1) conventions */
#define EXIT_NORMAL     0 /* No disaster anomaly detected (or training succeeded) */
#define EXIT_ALERT      1 /* Disaster anomaly detected above scene threshold */
#define EXIT_TROUBLE    2 /* Invalid arguments, I/O error, or dimension mismatch */

/* --- Network Hyperparameters --- */
#define PATCH_S     32      /* 32x32 spatial patch */
#define C1          16      /* Conv1 output channels */
#define H1          16      /* 32 / 2 */
#define C2          32      /* Conv2 output channels */
#define H2          8       /* 16 / 2 */
#define EMBED_D     32      /* Final L2-normalised embedding dimension */
#define BATCH_N     64      /* Contrastive mini-batch size (pairs per step) */
#define TEMP        0.07f   /* InfoNCE temperature */

/* 
 * 1. Netpbm PGM (P5 & P2, 8-bit and 16-bit) I/O)
 */

typedef struct {
    int width; 
    int height;
    int maxval;
    float *pixels; /* Normalised to [0.0, 1.0], row-major [height * width] */
} PGMImage;

static void skip_comments_and_ws(FILE *fp) {
    int ch; 
    while ((ch = fgetc(fp)) != EOF) {
        if (isspace(ch)) continue;
        if (ch == '#') {
            while ((ch = fgetc(fp)) != EOF && ch != '\n' && ch != '\r');
        } else {
            ungetc(ch, fp);
            break;
        } 
    }
}

int pgm_read(const char *path, PGMImage *img) {
    FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "rb");
    if (!fp) { perror("fopen pgm_read"); return -1; }

    char magic[3] = {0};
    if (fread(magic, 1, 2, fp) != 2 || (strcmp(magic, "P5") != 0 && strcmp(magic, "P2") != 0)) {
        fprintf(stderr, "Error: %s is not a valid P5/P2 PGM file\n", path);
        if (fp != stdin) fclose(fp);
        return -1;
    }

    skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &img->width) != 1) return -1;
    skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &img->height) != 1) return -1;
    skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &img->maxval) != 1) return -1;
    fgetc(fp); /* Consume single whitespace character after maxval */

    size_t npixels = (size_t)img->width * img->height;
    img->pixels = malloc(npixels * sizeof(float));
    if (!img->pixels) return -1;

    float inv_max = 1.0f / (float)(img->maxval > 0 ? img->maxval : 255);

    if (strcmp(magic, "P5") == 0) {
        if (img->maxval < 256) {
            uint8_t *buf = malloc(npixels);
            if (fread(buf, 1, npixels, fp) != npixels) { free(buf); return -1; }
            for (size_t i = 0; i < npixels; ++i) img->pixels[i] = (float)buf[i] * inv_max;
            free(buf);
        } else {
            /* 16-bit PGM is big-endian (MSB first) per Netpbm spec */
            uint8_t *buf = malloc(npixels * 2);
            if (fread(buf, 2, npixels, fp) != npixels) { free(buf); return -1; }
            for (size_t i = 0; i < npixels; ++i) {
                uint16_t val = ((uint16_t)buf[2 * i] << 8) | (uint16_t)buf[2 * i + 1];
                img->pixels[i] = (float)val * inv_max;
            }
            free(buf);
        }
    } else {
        for (size_t i = 0; i < npixels; ++i) {
            int val = 0;
            if (fscanf(fp, "%d", &val) != 1) break;
            img->pixels[i] = (float)val * inv_max;
        }
    }

    if (fp != stdin) fclose(fp);
    return 0;
}

int pgm_write(const char *path, const float *data, int width, int height) {
    FILE *fp = (strcmp(path, "-") == 0) ? stdout : fopen(path, "wb");
    if (!fp) { perror("fopen pgm_write"); return -1; }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    size_t npixels = (size_t)width * height;
    uint8_t *buf = malloc(npixels);
    for (size_t i = 0; i < npixels; ++i) {
        float v = data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        buf[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
    fwrite(buf, 1, npixels, fp); 
    free(buf); 
    if (fp != stdout) fclose(fp);
    return 0;
}

void pgm_free(PGMImage *img) {
    free(img->pixels);
    img->pixels = NULL;
}

/* 
 * Siamese Encoder Parameters, Activations & Gradients
 */

typedef struct {
    float w1[C1 * 1 * 3 * 3]; 
    float b1[C1];
    float w2[C2 * C1 * 3 * 3];
    float b2[C2];
    float w3[EMBED_D * C2];
    float b3[EMBED_D];
} SiameseNet;

typedef struct {
    float input[PATCH_S * PATCH_S];
    float z1[C1 * H1 * H1];
    float a1[C1 * H1 * H1];
    float z2[C2 * H2 * H2];
    float a2[C2 * H2 * H2];
    float gap[C2];
    float u[EMBED_D];
    float u_hat[EMBED_D];
    float inv_norm;
} BranchCache;

static float rand_uniform(float scale) {
    return (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
}

void net_init(SiameseNet *net) {
    /* He / Kaiming uniform initialisation */
    float s1 = sqrtf(6.0f / 9.0f);
    for (int i = 0; i < C1 * 9; ++i) net->w1[i] = rand_uniform(s1);
    memset(net->b1, 0, sizeof(net->b1));

    float s2 = sqrtf(6.0f / (C1 * 9.0f));
    for (int i = 0; i < C2 * C1 * 9; ++i) net->w2[i] = rand_uniform(s2);
    memset(net->b2, 0, sizeof(net->b2));

    float s3 = sqrtf(6.0f / (float)C2);
    for (int i = 0; i < EMBED_D * C2; ++i) net->w3[i] = rand_uniform(s3);
    memset(net->b3, 0, sizeof(net->b3));
}

/* Forward pass for a single 32x32 patch through one branch */
void branch_forward(const SiameseNet *net, const float *patch, BranchCache *c) {
    memcpy(c->input, patch, sizeof(c->input));

    /* Conv1: 1 -> C1, 3x3, stride 2, pad 1 + LeakyReLU(0.1) */
    for (int oc = 0; oc < C1; ++oc) {
        for (int oh = 0; oh < H1; ++oh) {
            for (int ow = 0; ow < H1; ++ow) {
                float sum = net->b1[oc];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int ih = oh * 2 + kh - 1;
                        int iw = ow * 2 + kw - 1;
                        if (ih >= 0 && ih < PATCH_S && iw >= 0 && iw < PATCH_S) {
                            sum += c->input[ih * PATCH_S + iw] * net->w1[(oc * 3 + kh) * 3 + kw];
                        }
                    }
                }
                int idx = (oc * H1 + oh) * H1 + ow;
                c->z1[idx] = sum;
                c->a1[idx] = (sum > 0.0f) ? sum : 0.1f * sum;
            }
        }
    }

    /* Conv2: C1 -> C2, 3x3, stride 2, pad 1 + LeakyReLU(0.1) */
    for (int oc = 0; oc < C2; ++oc) {
        for (int oh = 0; oh < H2; ++oh) {
            for (int ow = 0; ow < H2; ++ow) {
                float sum = net->b2[oc];
                for (int ic = 0; ic < C1; ++ic) {
                    const float *w_ptr = &net->w2[((oc * C1 + ic) * 3) * 3];
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            int ih = oh * 2 + kh - 1;
                            int iw = ow * 2 + kw - 1;
                            if (ih >= 0 && ih < H1 && iw >= 0 && iw < H1) {
                                sum += c->a1[(ic * H1 + ih) * H1 + iw] * w_ptr[kh * 3 + kw];
                            } 
                        } 
                    } 
                }
                int idx = (oc * H2 + oh) * H2 + ow;
                c->z2[idx] = sum; 
                c->a2[idx] = (sum > 0.0f) ? sum : 0.1f * sum;
            } 
        }
    }

    /* Global Average Pooling (H2 x H2 -> 1) */
    float inv_spatial = 1.0f / (float)(H2 * H2);
    for (int oc = 0; oc < C2; ++oc) {
        float sum = 0.0f;
        for (int i = 0; i < H2 * H2; ++i) sum += c->a2[oc * H2 * H2 + i];
        c->gap[oc] = sum * inv_spatial;
    }

    /* Linear Projection + L2 Normalisation */
    float norm_sq = 1e-12f;
    for (int d = 0; d < EMBED_D; ++d) {
        float sum = net->b3[d];
        for (int ic = 0; ic < C2; ++ic) {
            sum += net->w3[d * C2 + ic] * c->gap[ic];
        }
        c->u[d] = sum;
        norm_sq += sum * sum; 
    }
    c->inv_norm = 1.0f / sqrtf(norm_sq);
    for (int d = 0; d < EMBED_D; ++d) {
        c->u_hat[d] = c->u[d] * c->inv_norm;
    }
}

/* Backward pass for one branch, accumulating into shared 'grad' struct */
void branch_backward(const SiameseNet *net, SiameseNet *grad,
                    const BranchCache *c, const float *d_u_hat) {
    /* 1. L2-Norm Jacobian backward: d_u_hat -> d_u */
    float dot = 0.0f; 
    for (int d = 0; d < EMBED_D; ++d) dot += c->u_hat[d] * d_u_hat[d];

    float d_u[EMBED_D];
    for (int d = 0; d < EMBED_D; ++d) {
        d_u[d] = c->inv_norm * (d_u_hat[d] - c->u_hat[d] * dot); 
    }

    /* 2. Linear Projection backward */ 
    float d_gap[C2] = {0};
    for (int d = 0; d < EMBED_D; ++d) {
        grad->b3[d] += d_u[d];
        for (int ic = 0; ic < C2; ++ic) {
            grad->w3[d * C2 + ic] += d_u[d] * c->gap[ic];
            d_gap[ic] += net->w3[d * C2 + ic] * d_u[d];
        }
    }

    /* 3. GAP + LeakyReLU2 backward */
    float d_z2[C2 * H2 * H2];
    float inv_spatial = 1.0f / (float)(H2 * H2);
    for (int oc = 0; oc < C2; ++oc) {
        float g = d_gap[oc] * inv_spatial;
        for (int i = 0; i < H2 * H2; ++i) {
            int idx = oc * H2 * H2 + i;
            d_z2[idx] = (c->z2[idx] > 0.0f) ? g : 0.1f * g;
        } 
    } 

    /* 4. Conv2 backward */
    float d_a1[C1 * H1 * H1] = {0};
    for (int oc = 0; oc < C2; ++oc) {
        for (int oh = 0; oh < H2; ++oh) { 
            for (int ow = 0; ow < H2; ++ow) {
                float dz = d_z2[(oc * H2 + oh) * H2 + ow];
                grad->b2[oc] += dz;
                for (int ic = 0; ic < C1; ++ic) {
                    float *gw_ptr = &grad->w2[((oc * C1 + ic) * 3) * 3];
                    const float *w_ptr = &net->w2[((oc * C1 + ic) * 3) * 3];
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            int ih = oh * 2 + kh - 1;
                            int iw = ow * 2 + kw - 1;
                            if (ih >= 0 && ih < H1 && iw >= 0 && iw < H1) {
                                int a1_idx = (ic * H1 + ih) * H1 + iw;
                                gw_ptr[kh * 3 + kw] += dz * c->a1[a1_idx];
                                d_a1[a1_idx] += dz * w_ptr[kh * 3 + kw];
                            }
                        }
                    }
                }
            }
        }
    }

    /* 5. LeakyReLU1 + Conv1 backward */
    for (int oc = 0; oc < C1; ++oc) {
        for (int oh = 0; oh < H1; ++oh) {
            for (int ow = 0; ow < H1; ++ow) {
                int idx = (oc * H1 + oh) * H1 + ow;
                float dz = (c->z1[idx] > 0.0f) ? d_a1[idx] : 0.1f * d_a1[idx];
                grad->b1[oc] += dz;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int ih = oh * 2 + kh - 1; 
                        int iw = ow * 2 + kw - 1;
                        if (ih >= 0 && ih < PATCH_S && iw >= 0 && iw < PATCH_S) {
                            grad->w1[(oc * 3 + kh) * 3 + kw] += dz * c->input[ih * PATCH_S + iw];
                        }
                    }
                }
            }
        }
    }
}

/* 
 * 3. InfoNCE Loss & SGD-Momentum Parameter Update
 */

float infonce_step(SiameseNet *net, SiameseNet *velocity, 
                   BranchCache *cache_t0, BranchCache *cache_t1,
                   const float patches_t0[BATCH_N][PATCH_S * PATCH_S],
                   const float patches_t1[BATCH_N][PATCH_S * PATCH_S],
                   float lr, float weight_decay) {
    /* Forward pass across mini-batch in parallel */
    #pragma omp parallel for 
    for (int i = 0; i < BATCH_N; ++i) {
        branch_forward(net, patches_t0[i], &cache_t0[i]);
        branch_forward(net, patches_t1[i], &cache_t1[i]);
    }

    /* Compute InfoNCE similarity matrix and softmax probabilities */
    float logits[BATCH_N][BATCH_N];
    float probs[BATCH_N][BATCH_N];
    float inv_temp = 1.0f / TEMP;
    float loss = 0.0f;

    for (int i = 0; i < BATCH_N; ++i) {
        float max_l = -1e30f;
        for (int j = 0; j < BATCH_N; ++j) {
            float dot = 0.0f; 
            for (int d = 0; d < EMBED_D; ++d) {
                dot += cache_t0[i].u_hat[d] * cache_t1[j].u_hat[d];
            }
            logits[i][j] = dot * inv_temp;
            if (logits[i][j] > max_l) max_l = logits[i][j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < BATCH_N; ++j) {
            probs[i][j] = expf(logits[i][j] - max_l);
            sum_exp += probs[i][j];
        }
        for (int j = 0; j < BATCH_N; ++j) probs[i][j] /= sum_exp;
        loss += -logf(probs[i][i] + 1e-12f);
    }

    /* Compute gradients w.r.t. normalised embeddings u_hat and v_hat */
    float d_u_hat[BATCH_N][EMBED_D] = {{0}};
    float d_v_hat[BATCH_N][EMBED_D] = {{0}};
    float scale = inv_temp / (float)BATCH_N;

    for (int i = 0; i < BATCH_N; ++i) {
        for (int j = 0; j < BATCH_N; ++j) {
            float g = (probs[i][j] - (i == j ? 1.0f : 0.0f)) * scale;
            for (int d = 0; d < EMBED_D; ++d) {
                d_u_hat[i][d] += g * cache_t1[j].u_hat[d];
                d_v_hat[j][d] += g * cache_t0[i].u_hat[d];
            }
        }
    }

    /* Accumulate parameter gradients across both twin branches */
    SiameseNet grad; 
    memset(&grad, 0, sizeof(SiameseNet));
    for (int i = 0; i < BATCH_N; ++i) {
        branch_backward(net, &grad, &cache_t0[i], d_u_hat[i]);
        branch_backward(net, &grad, &cache_t1[i], d_v_hat[i]);
    }

    /* Update shared weights with momentum (0.9) + Decoupled Weight Decay */
    float *w_ptr = (float *)net;
    float *g_ptr = (float *)&grad;
    float *v_ptr = (float *)velocity;
    size_t num_params = sizeof(SiameseNet) / sizeof(float);

    for (size_t k = 0; k < num_params; ++k) {
        v_ptr[k] = 0.9f * v_ptr[k] + g_ptr[k];
        w_ptr[k] = w_ptr[k] * (1.0f - lr * weight_decay) - lr * v_ptr[k];
    }

    return loss / (float)BATCH_N;
} 

/* 
 * 4. Patch Extraction, Training & Full-Scene Change Map Inference
 */

static void extract_patch(const PGMImage *img, int y0, int x0, float *out, float gain, float bias) {
    for (int dy = 0; dy < PATCH_S; ++dy) {
        for (int dx = 0; dx < PATCH_S; ++dx) {
            float v = img->pixels[(y0 + dy) * img->width + (x0 + dx)] * gain + bias;
            if (v < 0.0f) v = 0.0f; 
            if (v > 1.0f) v = 1.0f;
            out[dy * PATCH_S + dx] = v;
        }
    }
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [-tcq] [-w weights] [-o out.pgm] [-n steps] [-l lr]\n"
        "             [-a tile_thresh] [-f alert_frac] [-s stride] <pre.pgm> <post.pgm>\n"
        "options:\n"
        "  -t            train mode (default: inference mode)\n"
        "  -c            continue/fine-tune from existing weights file\n"
        "  -w <file>     weights file path (default: pgmcd.w)\n"
        "  -o <file>     output change map PGM (default: '-' for stdout)\n"
        "  -n <int>      training steps (default: 200)\n"
        "  -l <float>    learning rate (default: 0.01)\n"
        "  -a <float>    tile cosine-distance anomaly threshold [0,1] (default: 0.35)\n"
        "  -f <float>    scene alert fraction threshold [0,1] (default: 0.05)\n"
        "  -s <int>      inference sliding-window stride in pixels (default: 8)\n"
        "  -q            quiet mode (suppress stderr diagnostics)\n"
        "exit status:\n"
        "   0            normal (or training complete)\n"
        "   1            disaster anomaly alert triggered\n"
        "   2            trouble (I/O or argument error)\n",
        prog);
}

int main(int argc, char **argv) {
    int train_mode = 0;
    int continue_weights = 0;
    int quiet = 0;
    int steps = 200;
    int stride = 8;
    float lr = 0.01f;
    float tile_thresh = 0.35f;
    float alert_frac = 0.05f;
    const char *weights_path = "pgmcd.w";
    const char *out_path = "-"; /* stdout by default */

    int opt;
    while ((opt = getopt(argc, argv, "tcqw:o:n:l:a:f:s:h")) != -1) {
        switch (opt) {
            case 't': train_mode = 1; break;
            case 'c': continue_weights = 1; break;
            case 'q': quiet = 1; break;
            case 'w': weights_path = optarg; break;
            case 'o': out_path = optarg; break;
            case 'n': steps = atoi(optarg); break;
            case 'l': lr = strtof(optarg, NULL); break;
            case 'a': tile_thresh = strtof(optarg, NULL); break;
            case 'f': alert_frac = strtof(optarg, NULL); break;
            case 's': stride = atoi(optarg); break;
            case 'h':
                usage(argv[0]);
                return EXIT_NORMAL;
            default:
                usage(argv[0]);
                return EXIT_TROUBLE;
        }
    }

    /* Expect exactly two positional arguments: <pre.pgm> <post.pgm> */
    if (argc - optind != 2) {
        usage(argv[0]);
        return EXIT_TROUBLE;
    }
    if (stride < 1 || steps < 1) {
        fprintf(stderr, "pgmcd: stride (-s) and steps (-n) must be >= 1\n");
        return EXIT_TROUBLE;
    }

    const char *pre_path  = argv[optind];
    const char *post_path = argv[optind + 1];

    /* Prevent reading both inputs from stdin simultaneously */
    if (strcmp(pre_path, "-") == 0 && strcmp(post_path, "-") == 0) {
        fprintf(stderr, "pgmcd: cannot stream both pre and post images from stdin\n");
        return EXIT_TROUBLE;
    }

    PGMImage pre = {0}, post = {0};
    if (pgm_read(pre_path, &pre) != 0 || pgm_read(post_path, &post) != 0) {
        pgm_free(&pre);
        pgm_free(&post);
        return EXIT_TROUBLE;
    }

    if (pre.width != post.width || pre.height != post.height ||
        pre.width < PATCH_S || pre.height < PATCH_S) {
        fprintf(stderr, "pgmcd: dimension mismatch or image smaller than %dx%d\n",
                PATCH_S, PATCH_S);
        pgm_free(&pre);
        pgm_free(&post);
        return EXIT_TROUBLE;
    }

    SiameseNet net;
    int exit_code = EXIT_NORMAL;

    if (train_mode) {
        if (continue_weights) {
            FILE *wf = fopen(weights_path, "rb");
            if (!wf || fread(&net, sizeof(SiameseNet), 1, wf) != 1) {
                perror("pgmcd: fread weights (-c)");
                if (wf) fclose(wf);
                pgm_free(&pre); pgm_free(&post);
                return EXIT_TROUBLE;
            }
            fclose(wf);
        } else {
            net_init(&net);
        }

        SiameseNet velocity;
        memset(&velocity, 0, sizeof(SiameseNet));

        BranchCache *c_t0 = malloc(BATCH_N * sizeof(BranchCache));
        BranchCache *c_t1 = malloc(BATCH_N * sizeof(BranchCache));
        float (*batch_t0)[PATCH_S * PATCH_S] = malloc(BATCH_N * sizeof(*batch_t0));
        float (*batch_t1)[PATCH_S * PATCH_S] = malloc(BATCH_N * sizeof(*batch_t1));

        if (!c_t0 || !c_t1 || !batch_t0 || !batch_t1) {
            perror("pgmcd: malloc");
            return EXIT_TROUBLE;
        }

        for (int step = 1; step <= steps; ++step) {
            for (int b = 0; b < BATCH_N; ++b) {
                int y0 = rand() % (pre.height - PATCH_S + 1);
                int x0 = rand() % (pre.width  - PATCH_S + 1);
                float gain = 1.0f + rand_uniform(0.15f);
                float bias = rand_uniform(0.08f);
                extract_patch(&pre,  y0, x0, batch_t0[b], 1.0f, 0.0f);
                extract_patch(&post, y0, x0, batch_t1[b], gain, bias);
            }
            float loss = infonce_step(&net, &velocity, c_t0, c_t1,
                                      batch_t0, batch_t1, lr, 1e-4f);
            if (!quiet && (step == 1 || step % 25 == 0 || step == steps)) {
                fprintf(stderr, "pgmcd: step %4d/%d loss=%.5f\n", step, steps, loss);
            }
        }

        FILE *wf = fopen(weights_path, "wb");
        if (!wf || fwrite(&net, sizeof(SiameseNet), 1, wf) != 1) {
            perror("pgmcd: fwrite weights");
            if (wf) fclose(wf);
            exit_code = EXIT_TROUBLE;
        } else {
            fclose(wf);
        }

        free(c_t0); free(c_t1); free(batch_t0); free(batch_t1);
    } else {
        /* Default mode: Inference (streams binary P5 PGM to stdout unless -o set) */
        FILE *wf = fopen(weights_path, "rb");
        if (!wf || fread(&net, sizeof(SiameseNet), 1, wf) != 1) {
            perror("pgmcd: fread weights");
            if (wf) fclose(wf);
            pgm_free(&pre); pgm_free(&post);
            return EXIT_TROUBLE;
        }
        fclose(wf);

        int grid_h = (pre.height - PATCH_S) / stride + 1;
        int grid_w = (pre.width  - PATCH_S) / stride + 1;
        float *change_map = calloc((size_t)grid_h * grid_w, sizeof(float));
        if (!change_map) {
            perror("pgmcd: calloc");
            pgm_free(&pre); pgm_free(&post);
            return EXIT_TROUBLE;
        }

        int anomalous_tiles = 0;

        #pragma omp parallel for collapse(2) reduction(+:anomalous_tiles)
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                float p0[PATCH_S * PATCH_S], p1[PATCH_S * PATCH_S];
                BranchCache c0, c1;
                extract_patch(&pre,  gy * stride, gx * stride, p0, 1.0f, 0.0f);
                extract_patch(&post, gy * stride, gx * stride, p1, 1.0f, 0.0f);
                branch_forward(&net, p0, &c0);
                branch_forward(&net, p1, &c1);

                float cos_sim = 0.0f;
                for (int d = 0; d < EMBED_D; ++d) cos_sim += c0.u_hat[d] * c1.u_hat[d];
                float cos_dist = 0.5f * (1.0f - cos_sim);
                change_map[gy * grid_w + gx] = cos_dist;
                if (cos_dist >= tile_thresh) anomalous_tiles++;
            }
        }

        if (pgm_write(out_path, change_map, grid_w, grid_h) != 0) {
            free(change_map);
            pgm_free(&pre); pgm_free(&post);
            return EXIT_TROUBLE;
        }

        float frac = (float)anomalous_tiles / (float)(grid_h * grid_w);
        int is_alert = (frac >= alert_frac);

        if (!quiet) {
            fprintf(stderr, "pgmcd: fraction=%.4f (%d/%d tiles >= %.2f) status=%s\n",
                    frac, anomalous_tiles, grid_h * grid_w, tile_thresh,
                    is_alert ? "ALERT" : "NORMAL");
        }

        free(change_map);
        exit_code = is_alert ? EXIT_ALERT : EXIT_NORMAL;
    }

    pgm_free(&pre);
    pgm_free(&post);
    return exit_code;
}
