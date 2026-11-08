/*
 * siam.c
 * C11 Siamese Contrastive (InfoNCE / Cosine) Engine for PGM Satellite Images
 *
 * Compile:
 *   cc -O3 -march=native -Wall -Wextra -Wno-unknown-pragmas siam.c -lm -o siam
 *
 * Usage: 
 *  ./siam -t -w flood.w -n 300 pre1.pgm post1.pgm [pre2.pgm post2.pgm ...]
 *  paste <(ls *_pre.pgm) <(ls *_post.pgm) | ./siam -t -w flood.w -n 1000 -
 *  ./siam -w flood.w -a 0.35 pre.pgm post.pgm > change_map.pgm || notify_alert   
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h> 
#include <ctype.h> 
#include <math.h>
#include <time.h>
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

/* =========================================================================
 * 1. Netpbm PGM (P5 & P2, 8-bit and 16-bit) I/O
 * ========================================================================= */

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
    memset(img, 0, sizeof(*img));
    FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "rb");
    if (!fp) { perror("fopen pgm_read"); return -1; }

    char magic[3] = {0};
    if (fread(magic, 1, 2, fp) != 2 || (strcmp(magic, "P5") != 0 && strcmp(magic, "P2") != 0)) {
        fprintf(stderr, "Error: %s is not a valid P5/P2 PGM file\n", path);
        goto fail;
    }

    skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &img->width) != 1 || img->width <= 0) goto fail;
    skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &img->height) != 1 || img->height <= 0) goto fail;
    skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &img->maxval) != 1 || img->maxval <= 0 || img->maxval > 65535) goto fail;

    /* Consume single whitespace separator (handle CRLF safely) */
    int sep = fgetc(fp);
    if (sep == '\r') {
        int next = fgetc(fp);
        if (next != '\n' && next != EOF) ungetc(next, fp);
    }

    if ((size_t)img->width > SIZE_MAX / (size_t)img->height / sizeof(float)) goto fail;
    size_t npixels = (size_t)img->width * (size_t)img->height;
    img->pixels = malloc(npixels * sizeof(float));
    if (!img->pixels) goto fail;

    float inv_max = 1.0f / (float)img->maxval;

    if (strcmp(magic, "P5") == 0) {
        size_t bpp = (img->maxval < 256) ? 1 : 2;
        uint8_t *buf = malloc(npixels * bpp);
        if (!buf || fread(buf, bpp, npixels, fp) != npixels) {
            free(buf);
            goto fail;
        }
        if (bpp == 1) {
            for (size_t i = 0; i < npixels; ++i) img->pixels[i] = (float)buf[i] * inv_max;
        } else {
            /* 16-bit PGM is big-endian (MSB first) per Netpbm spec */
            for (size_t i = 0; i < npixels; ++i) {
                uint16_t val = ((uint16_t)buf[2 * i] << 8) | (uint16_t)buf[2 * i + 1];
                img->pixels[i] = (float)val * inv_max;
            }
        }
        free(buf);
    } else {
        for (size_t i = 0; i < npixels; ++i) {
            int val = 0;
            if (fscanf(fp, "%d", &val) != 1) goto fail;
            img->pixels[i] = (float)val * inv_max;
        }
    }

    if (fp != stdin) fclose(fp);
    return 0;

fail:
    free(img->pixels);
    img->pixels = NULL;
    if (fp && fp != stdin) fclose(fp);
    return -1;
}

int pgm_write(const char *path, const float *data, int width, int height) {
    FILE *fp = (strcmp(path, "-") == 0) ? stdout : fopen(path, "wb");
    if (!fp) { perror("fopen pgm_write"); return -1; }

    size_t npixels = (size_t)width * (size_t)height;
    uint8_t *buf = malloc(npixels);
    if (!buf) {
        perror("malloc pgm_write");
        if (fp != stdout) fclose(fp);
        return -1;
    }
    for (size_t i = 0; i < npixels; ++i) {
        float v = data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        buf[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    size_t written = fwrite(buf, 1, npixels, fp);
    free(buf);
    if (fp != stdout) fclose(fp);
    return (written == npixels) ? 0 : -1;
}

void pgm_free(PGMImage *img) {
    free(img->pixels);
    img->pixels = NULL;
}

/* ========================================================================= 
 * 2. Siamese Encoder Parameters, Activations & Gradients
 * ========================================================================= */

#define NUM_W1        (C1 * 1 * 3 * 3)
#define NUM_W2        (C2 * C1 * 3 * 3)
#define NUM_W3        (EMBED_D * C2)
#define TOTAL_WEIGHTS (NUM_W1 + NUM_W2 + NUM_W3)
#define TOTAL_BIASES  (C1 + C2 + EMBED_D)
#define TOTAL_PARAMS  (TOTAL_WEIGHTS + TOTAL_BIASES)
#define WEIGHT_MAGIC  0x5349414Du /* "SIAM" */

typedef struct {
    uint32_t magic;   /* 0x5349414D ("SIAM") */
    uint16_t patch_s, c1, c2, embed_d;
    uint32_t reserved;
} WeightHeader;

typedef union {
    struct {
        /* Group all weights first so weight_decay only applies to [0 .. TOTAL_WEIGHTS-1] */
        float w1[NUM_W1];
        float w2[NUM_W2];
        float w3[NUM_W3];
        /* Biases excluded from weight decay */
        float b1[C1];
        float b2[C2];
        float b3[EMBED_D];
    };
    float params[TOTAL_PARAMS];
} SiameseNet;

typedef struct {
    float in_pad[(PATCH_S + 2) * (PATCH_S + 2)]; /* 34x34 zero-padded input */
    float z1[C1 * H1 * H1];
    float a1_pad[C1 * (H1 + 2) * (H1 + 2)];     /* 16x18x18 zero-padded a1 */
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
    for (int i = 0; i < NUM_W1; ++i) net->w1[i] = rand_uniform(s1);
    memset(net->b1, 0, sizeof(net->b1));

    float s2 = sqrtf(6.0f / (C1 * 9.0f));
    for (int i = 0; i < NUM_W2; ++i) net->w2[i] = rand_uniform(s2);
    memset(net->b2, 0, sizeof(net->b2));

    float s3 = sqrtf(6.0f / (float)C2);
    for (int i = 0; i < NUM_W3; ++i) net->w3[i] = rand_uniform(s3);
    memset(net->b3, 0, sizeof(net->b3));
}

static int weights_save(const char *path, const SiameseNet *net) {
    FILE *wf = fopen(path, "wb");
    if (!wf) {
        perror("siam: fopen weights");
        return -1;
    }
    WeightHeader hdr = {
        .magic    = WEIGHT_MAGIC,
        .patch_s  = PATCH_S,
        .c1       = C1,
        .c2       = C2,
        .embed_d  = EMBED_D,
        .reserved = 0
    };
    if (fwrite(&hdr, sizeof(hdr), 1, wf) != 1 ||
        fwrite(net, sizeof(*net), 1, wf) != 1) {
        perror("siam: fwrite weights");
        fclose(wf);
        return -1;
    }
    fclose(wf);
    return 0;
}

static int weights_load(const char *path, SiameseNet *net) {
    FILE *wf = fopen(path, "rb");
    if (!wf) {
        perror("siam: fopen weights");
        return -1;
    }
    WeightHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, wf) != 1) {
        fprintf(stderr, "siam: failed to read weights header from %s\n", path);
        fclose(wf);
        return -1;
    }
    if (hdr.magic != WEIGHT_MAGIC ||
        hdr.patch_s != PATCH_S || hdr.c1 != C1 ||
        hdr.c2 != C2 || hdr.embed_d != EMBED_D) {
        fprintf(stderr, "siam: incompatible or corrupt weights header in %s\n", path);
        fclose(wf);
        return -1;
    }
    if (fread(net, sizeof(*net), 1, wf) != 1) {
        fprintf(stderr, "siam: truncated weights payload in %s\n", path);
        fclose(wf);
        return -1;
    }
    fclose(wf);
    return 0;
}

/* Forward pass for a single 32x32 patch through one branch (branchless, SIMD-friendly) */
void branch_forward(const SiameseNet *net, const float *patch, BranchCache *c) {
    const int P0 = PATCH_S + 2; /* 34 */
    const int P1 = H1 + 2;      /* 18 */

    memset(c->in_pad, 0, sizeof(c->in_pad));
    memset(c->a1_pad, 0, sizeof(c->a1_pad));
    for (int h = 0; h < PATCH_S; ++h) {
        memcpy(&c->in_pad[(h + 1) * P0 + 1], &patch[h * PATCH_S], PATCH_S * sizeof(float));
    }

    /* Conv1: 1 -> C1, 3x3, stride 2, pad 1 + LeakyReLU(0.1) */
    for (int oc = 0; oc < C1; ++oc) {
        float *z1_oc = &c->z1[oc * H1 * H1];
        float *a1_oc = &c->a1_pad[oc * P1 * P1];
        const float *w_ptr = &net->w1[oc * 9];
        for (int oh = 0; oh < H1; ++oh) {
            for (int ow = 0; ow < H1; ++ow) z1_oc[oh * H1 + ow] = net->b1[oc];
            for (int kh = 0; kh < 3; ++kh) {
                const float *row = &c->in_pad[(oh * 2 + kh) * P0];
                float w0 = w_ptr[kh * 3 + 0];
                float w1 = w_ptr[kh * 3 + 1];
                float w2 = w_ptr[kh * 3 + 2];
                for (int ow = 0; ow < H1; ++ow) {
                    int iw = ow * 2;
                    z1_oc[oh * H1 + ow] += row[iw] * w0 + row[iw + 1] * w1 + row[iw + 2] * w2;
                }
            }
            for (int ow = 0; ow < H1; ++ow) {
                float s = z1_oc[oh * H1 + ow];
                a1_oc[(oh + 1) * P1 + (ow + 1)] = (s > 0.0f) ? s : 0.1f * s;
            }
        }
    }

    /* Conv2: C1 -> C2, 3x3, stride 2, pad 1 + LeakyReLU(0.1) */
    for (int oc = 0; oc < C2; ++oc) {
        float *z2_oc = &c->z2[oc * H2 * H2];
        for (int i = 0; i < H2 * H2; ++i) z2_oc[i] = net->b2[oc];

        /* Loop order oc -> ic -> oh -> kh -> kw -> ow maximizes spatial locality */
        for (int ic = 0; ic < C1; ++ic) {
            const float *w_ptr = &net->w2[(oc * C1 + ic) * 9];
            const float *a1_ic = &c->a1_pad[ic * P1 * P1];
            for (int oh = 0; oh < H2; ++oh) {
                for (int kh = 0; kh < 3; ++kh) {
                    const float *row = &a1_ic[(oh * 2 + kh) * P1];
                    float w0 = w_ptr[kh * 3 + 0];
                    float w1 = w_ptr[kh * 3 + 1];
                    float w2 = w_ptr[kh * 3 + 2];
                    for (int ow = 0; ow < H2; ++ow) {
                        int iw = ow * 2;
                        z2_oc[oh * H2 + ow] += row[iw] * w0 + row[iw + 1] * w1 + row[iw + 2] * w2;
                    }
                }
            }
        }
        for (int i = 0; i < H2 * H2; ++i) {
            float s = z2_oc[i];
            c->a2[oc * H2 * H2 + i] = (s > 0.0f) ? s : 0.1f * s;
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

/* Backward pass for one branch, accumulating into 'grad' struct (branchless, SIMD-friendly) */
void branch_backward(const SiameseNet *net, SiameseNet *grad,
                     const BranchCache *c, const float *d_u_hat) {
    const int P0 = PATCH_S + 2; /* 34 */
    const int P1 = H1 + 2;      /* 18 */

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

    /* 4. Conv2 backward (branchless over padded a1_pad / d_a1_pad) */
    float d_a1_pad[C1 * (H1 + 2) * (H1 + 2)] = {0};
    for (int oc = 0; oc < C2; ++oc) {
        const float *dz_oc = &d_z2[oc * H2 * H2];
        float b_sum = 0.0f;
        for (int i = 0; i < H2 * H2; ++i) b_sum += dz_oc[i];
        grad->b2[oc] += b_sum;

        for (int ic = 0; ic < C1; ++ic) {
            float *gw_ptr = &grad->w2[(oc * C1 + ic) * 9];
            const float *w_ptr = &net->w2[(oc * C1 + ic) * 9];
            const float *a1_ic = &c->a1_pad[ic * P1 * P1];
            float *da1_ic = &d_a1_pad[ic * P1 * P1];

            for (int oh = 0; oh < H2; ++oh) {
                for (int kh = 0; kh < 3; ++kh) {
                    const float *a1_row = &a1_ic[(oh * 2 + kh) * P1];
                    float *da1_row = &da1_ic[(oh * 2 + kh) * P1];
                    float gw0 = 0.0f, gw1 = 0.0f, gw2 = 0.0f;
                    float w0 = w_ptr[kh * 3 + 0];
                    float w1 = w_ptr[kh * 3 + 1];
                    float w2 = w_ptr[kh * 3 + 2];
                    for (int ow = 0; ow < H2; ++ow) {
                        float dz = dz_oc[oh * H2 + ow];
                        int iw = ow * 2;
                        gw0 += dz * a1_row[iw + 0];
                        gw1 += dz * a1_row[iw + 1];
                        gw2 += dz * a1_row[iw + 2];
                        da1_row[iw + 0] += dz * w0;
                        da1_row[iw + 1] += dz * w1;
                        da1_row[iw + 2] += dz * w2;
                    }
                    gw_ptr[kh * 3 + 0] += gw0;
                    gw_ptr[kh * 3 + 1] += gw1;
                    gw_ptr[kh * 3 + 2] += gw2;
                }
            }
        }
    }

    /* 5. LeakyReLU1 + Conv1 backward (branchless over padded in_pad) */
    for (int oc = 0; oc < C1; ++oc) {
        const float *z1_oc = &c->z1[oc * H1 * H1];
        const float *da1_oc = &d_a1_pad[oc * P1 * P1];
        float *gw_ptr = &grad->w1[oc * 9];
        float b_sum = 0.0f;

        for (int oh = 0; oh < H1; ++oh) {
            float dz_row[H1];
            for (int ow = 0; ow < H1; ++ow) {
                float da = da1_oc[(oh + 1) * P1 + (ow + 1)];
                float dz = (z1_oc[oh * H1 + ow] > 0.0f) ? da : 0.1f * da;
                dz_row[ow] = dz;
                b_sum += dz;
            }
            for (int kh = 0; kh < 3; ++kh) {
                const float *in_row = &c->in_pad[(oh * 2 + kh) * P0];
                float gw0 = 0.0f, gw1 = 0.0f, gw2 = 0.0f;
                for (int ow = 0; ow < H1; ++ow) {
                    float dz = dz_row[ow];
                    int iw = ow * 2;
                    gw0 += dz * in_row[iw + 0];
                    gw1 += dz * in_row[iw + 1];
                    gw2 += dz * in_row[iw + 2];
                }
                gw_ptr[kh * 3 + 0] += gw0;
                gw_ptr[kh * 3 + 1] += gw1;
                gw_ptr[kh * 3 + 2] += gw2;
            }
        }
        grad->b1[oc] += b_sum;
    }
}

/* ========================================================================= 
 * 3. Symmetric InfoNCE Loss & SGD-Momentum Parameter Update
 * ========================================================================= */

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

    /* Compute symmetric InfoNCE similarity matrix and row/col softmax probabilities */
    float logits[BATCH_N][BATCH_N];
    float probs_row[BATCH_N][BATCH_N];
    float probs_col[BATCH_N][BATCH_N];
    float inv_temp = 1.0f / TEMP;
    float loss_row = 0.0f;
    float loss_col = 0.0f;

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
            probs_row[i][j] = expf(logits[i][j] - max_l);
            sum_exp += probs_row[i][j];
        }
        for (int j = 0; j < BATCH_N; ++j) probs_row[i][j] /= sum_exp;
        loss_row += -logf(probs_row[i][i] + 1e-12f);
    }

    for (int j = 0; j < BATCH_N; ++j) {
        float max_l = -1e30f;
        for (int i = 0; i < BATCH_N; ++i) {
            if (logits[i][j] > max_l) max_l = logits[i][j];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < BATCH_N; ++i) {
            probs_col[i][j] = expf(logits[i][j] - max_l);
            sum_exp += probs_col[i][j];
        }
        for (int i = 0; i < BATCH_N; ++i) probs_col[i][j] /= sum_exp;
        loss_col += -logf(probs_col[j][j] + 1e-12f);
    }

    /* Compute symmetric gradients w.r.t. normalised embeddings u_hat and v_hat */
    float d_u_hat[BATCH_N][EMBED_D] = {{0}};
    float d_v_hat[BATCH_N][EMBED_D] = {{0}};
    float scale = 0.5f * inv_temp / (float)BATCH_N;

    for (int i = 0; i < BATCH_N; ++i) {
        for (int j = 0; j < BATCH_N; ++j) {
            float target = (i == j) ? 2.0f : 0.0f;
            float g = (probs_row[i][j] + probs_col[i][j] - target) * scale;
            for (int d = 0; d < EMBED_D; ++d) {
                d_u_hat[i][d] += g * cache_t1[j].u_hat[d];
                d_v_hat[j][d] += g * cache_t0[i].u_hat[d];
            }
        }
    }

    /* Deterministic OpenMP array reduction across both twin branches */
    float grad_arr[TOTAL_PARAMS] = {0};

    #pragma omp parallel for reduction(+:grad_arr[:TOTAL_PARAMS])
    for (int i = 0; i < BATCH_N; ++i) {
        SiameseNet local_g = {0};
        branch_backward(net, &local_g, &cache_t0[i], d_u_hat[i]);
        branch_backward(net, &local_g, &cache_t1[i], d_v_hat[i]);
        for (size_t k = 0; k < TOTAL_PARAMS; ++k) {
            grad_arr[k] += local_g.params[k];
        }
    }

    /* Update shared weights with momentum (0.9) + Decoupled Weight Decay (weights only) */
    for (size_t k = 0; k < TOTAL_PARAMS; ++k) {
        velocity->params[k] = 0.9f * velocity->params[k] + grad_arr[k];
        float decay = (k < TOTAL_WEIGHTS) ? (1.0f - lr * weight_decay) : 1.0f;
        net->params[k] = net->params[k] * decay - lr * velocity->params[k];
    }

    return 0.5f * (loss_row + loss_col) / (float)BATCH_N;
} 

/* ========================================================================= 
 * 4. Patch Extraction, Multi-Pair Loading, Training & Full-Scene Change Map Inference
 * ========================================================================= */

typedef struct {
    PGMImage pre;
    PGMImage post;
} ImagePair;

static void pairs_free(ImagePair *pairs, size_t count) {
    if (!pairs) return;
    for (size_t i = 0; i < count; ++i) {
        pgm_free(&pairs[i].pre);
        pgm_free(&pairs[i].post);
    }
    free(pairs);
}

static int pairs_append(ImagePair **pairs, size_t *count, size_t *cap,
                        const char *pre_path, const char *post_path) {
    if (strcmp(pre_path, "-") == 0 && strcmp(post_path, "-") == 0) {
        fprintf(stderr, "siam: cannot stream both pre and post images from stdin\n");
        return -1;
    }

    PGMImage pre = {0}, post = {0};
    if (pgm_read(pre_path, &pre) != 0 || pgm_read(post_path, &post) != 0) {
        pgm_free(&pre);
        pgm_free(&post);
        return -1;
    }

    if (pre.width != post.width || pre.height != post.height ||
        pre.width < PATCH_S || pre.height < PATCH_S) {
        fprintf(stderr, "siam: dimension mismatch or image smaller than %dx%d (%s, %s)\n",
                PATCH_S, PATCH_S, pre_path, post_path);
        pgm_free(&pre);
        pgm_free(&post);
        return -1;
    }

    if (*count == *cap) {
        size_t new_cap = (*cap == 0) ? 16 : (*cap * 2);
        ImagePair *grown = realloc(*pairs, new_cap * sizeof(ImagePair));
        if (!grown) {
            perror("siam: realloc pairs");
            pgm_free(&pre);
            pgm_free(&post);
            return -1;
        }
        *pairs = grown;
        *cap = new_cap;
    }

    (*pairs)[*count].pre  = pre;
    (*pairs)[*count].post = post;
    (*count)++;
    return 0;
}

static int pairs_load_list(const char *list_path, ImagePair **pairs,
                           size_t *count, size_t *cap) {
    FILE *fp = (strcmp(list_path, "-") == 0) ? stdin : fopen(list_path, "r");
    if (!fp) {
        perror("siam: fopen pair list");
        return -1;
    }

    char pre_path[4096], post_path[4096];
    int rc = 0;
    while (1) {
        skip_comments_and_ws(fp);
        int n1 = fscanf(fp, "%4095s", pre_path);
        if (n1 == EOF) break;
        skip_comments_and_ws(fp);
        int n2 = fscanf(fp, "%4095s", post_path);
        if (n1 != 1 || n2 != 1) {
            fprintf(stderr, "siam: pair list %s has an unpaired image path\n", list_path);
            rc = -1;
            break;
        }
        if (pairs_append(pairs, count, cap, pre_path, post_path) != 0) {
            rc = -1;
            break;
        }
    }

    if (fp != stdin) fclose(fp);
    return rc;
}

/* Extract patch with D4 dihedral symmetry (rot/flip mode 0..7) */
static void extract_patch_aug(const PGMImage *img, int y0, int x0, float *out,
                              float gain, float bias, int d4_mode) {
    for (int dy = 0; dy < PATCH_S; ++dy) {
        for (int dx = 0; dx < PATCH_S; ++dx) {
            int sy = dy, sx = dx;
            if (d4_mode & 1) sx = PATCH_S - 1 - sx; /* Horizontal flip */
            if (d4_mode & 2) sy = PATCH_S - 1 - sy; /* Vertical flip */
            if (d4_mode & 4) { int t = sx; sx = sy; sy = t; } /* Transpose */

            float v = img->pixels[(y0 + sy) * img->width + (x0 + sx)] * gain + bias;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            /* Zero-center [0, 1] -> [-1, +1] while preserving backscatter drops */
            out[dy * PATCH_S + dx] = (v - 0.5f) * 2.0f;
        }
    }
}

static void extract_patch(const PGMImage *img, int y0, int x0, float *out, float gain, float bias) {
    extract_patch_aug(img, y0, x0, out, gain, bias, 0);
}

/*
 * Whole-Scene Shared Convolutions + 2D Summed-Area Table (Integral Image) GAP
 * When stride is a multiple of 4 (the combined Conv1 + Conv2 downsample factor),
 * evaluates Conv1 and Conv2 once across the full scene and computes each 8x8
 * Global Average Pool window in O(1) time (4 SAT lookups per channel).
 * Falls back to per-tile branch_forward when stride is not a multiple of 4.
 */
static int scene_extract_embeddings(const SiameseNet *net, const PGMImage *img,
                                    int height, int width, int stride,
                                    float (*u_hat_grid)[EMBED_D]) {
    int grid_h = (height - PATCH_S) / stride + 1;
    int grid_w = (width  - PATCH_S) / stride + 1;

    if (stride % 4 != 0) {
        #pragma omp parallel for collapse(2)
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                float p[PATCH_S * PATCH_S];
                BranchCache c;
                extract_patch(img, gy * stride, gx * stride, p, 1.0f, 0.0f);
                branch_forward(net, p, &c);
                memcpy(u_hat_grid[(size_t)gy * grid_w + gx], c.u_hat, sizeof(c.u_hat));
            }
        }
        return 0;
    }

    int h1 = height / 2, w1_dim = width / 2;
    int h2 = h1 / 2,     w2_dim = w1_dim / 2;
    int p0_h = height + 2, p0_w = width + 2;
    int p1_h = h1 + 2,     p1_w = w1_dim + 2;
    int sat_h = h2 + 1,    sat_w = w2_dim + 1;

    float  *in_pad = calloc((size_t)p0_h * p0_w, sizeof(float));
    float  *a1_pad = calloc((size_t)C1 * p1_h * p1_w, sizeof(float));
    float  *z2_buf = malloc((size_t)C2 * h2 * w2_dim * sizeof(float));
    double *sat    = calloc((size_t)C2 * sat_h * sat_w, sizeof(double));

    if (!in_pad || !a1_pad || !z2_buf || !sat) {
        perror("siam: scene_extract_embeddings alloc");
        free(in_pad); free(a1_pad); free(z2_buf); free(sat);
        return -1;
    }

    /* 1. Populate zero-padded scene input in [-1, +1] */
    for (int y = 0; y < height; ++y) {
        const float *src = &img->pixels[(size_t)y * img->width];
        float *dst = &in_pad[(size_t)(y + 1) * p0_w + 1];
        for (int x = 0; x < width; ++x) {
            float v = src[x];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            dst[x] = (v - 0.5f) * 2.0f;
        }
    }

    /* 2. Whole-scene Conv1: 1 -> C1, 3x3, stride 2, pad 1 + LeakyReLU(0.1) */
    #pragma omp parallel for
    for (int oc = 0; oc < C1; ++oc) {
        float *a1_oc = &a1_pad[(size_t)oc * p1_h * p1_w];
        const float *w_ptr = &net->w1[oc * 9];
        float b = net->b1[oc];
        for (int oh = 0; oh < h1; ++oh) {
            float *a1_row = &a1_oc[(size_t)(oh + 1) * p1_w + 1];
            for (int ow = 0; ow < w1_dim; ++ow) a1_row[ow] = b;
            for (int kh = 0; kh < 3; ++kh) {
                const float *in_row = &in_pad[(size_t)(oh * 2 + kh) * p0_w];
                float k0 = w_ptr[kh * 3 + 0];
                float k1 = w_ptr[kh * 3 + 1];
                float k2 = w_ptr[kh * 3 + 2];
                for (int ow = 0; ow < w1_dim; ++ow) {
                    int iw = ow * 2;
                    a1_row[ow] += in_row[iw] * k0 + in_row[iw + 1] * k1 + in_row[iw + 2] * k2;
                }
            }
            for (int ow = 0; ow < w1_dim; ++ow) {
                float s = a1_row[ow];
                a1_row[ow] = (s > 0.0f) ? s : 0.1f * s;
            }
        }
    }

    /* 3. Whole-scene Conv2: C1 -> C2, 3x3, stride 2, pad 1 + LeakyReLU(0.1) + 2D Integral Image */
    #pragma omp parallel for
    for (int oc = 0; oc < C2; ++oc) {
        float *z2_oc = &z2_buf[(size_t)oc * h2 * w2_dim];
        double *sat_oc = &sat[(size_t)oc * sat_h * sat_w];
        float b = net->b2[oc];
        for (int i = 0; i < h2 * w2_dim; ++i) z2_oc[i] = b;

        for (int ic = 0; ic < C1; ++ic) {
            const float *w_ptr = &net->w2[(oc * C1 + ic) * 9];
            const float *a1_ic = &a1_pad[(size_t)ic * p1_h * p1_w];
            for (int oh = 0; oh < h2; ++oh) {
                float *z_row = &z2_oc[(size_t)oh * w2_dim];
                for (int kh = 0; kh < 3; ++kh) {
                    const float *row = &a1_ic[(size_t)(oh * 2 + kh) * p1_w];
                    float k0 = w_ptr[kh * 3 + 0];
                    float k1 = w_ptr[kh * 3 + 1];
                    float k2 = w_ptr[kh * 3 + 2];
                    for (int ow = 0; ow < w2_dim; ++ow) {
                        int iw = ow * 2;
                        z_row[ow] += row[iw] * k0 + row[iw + 1] * k1 + row[iw + 2] * k2;
                    }
                }
            }
        }

        for (int oh = 0; oh < h2; ++oh) {
            double row_sum = 0.0;
            const float *z_row = &z2_oc[(size_t)oh * w2_dim];
            const double *prev_sat = &sat_oc[(size_t)oh * sat_w];
            double *cur_sat = &sat_oc[(size_t)(oh + 1) * sat_w];
            for (int ow = 0; ow < w2_dim; ++ow) {
                float s = z_row[ow];
                float a = (s > 0.0f) ? s : 0.1f * s;
                row_sum += (double)a;
                cur_sat[ow + 1] = prev_sat[ow + 1] + row_sum;
            }
        }
    }

    /* 4. O(1) Sliding-Window GAP via Integral Image + Linear Projection + L2 Normalisation */
    const double inv_spatial = 1.0 / (double)(H2 * H2);
    #pragma omp parallel for collapse(2)
    for (int gy = 0; gy < grid_h; ++gy) {
        for (int gx = 0; gx < grid_w; ++gx) {
            int y1 = (gy * stride) >> 2;
            int x1 = (gx * stride) >> 2;
            int y2 = y1 + H2;
            int x2 = x1 + H2;

            float gap[C2];
            for (int oc = 0; oc < C2; ++oc) {
                const double *sat_oc = &sat[(size_t)oc * sat_h * sat_w];
                double sum = sat_oc[(size_t)y2 * sat_w + x2]
                           - sat_oc[(size_t)y1 * sat_w + x2]
                           - sat_oc[(size_t)y2 * sat_w + x1]
                           + sat_oc[(size_t)y1 * sat_w + x1];
                gap[oc] = (float)(sum * inv_spatial);
            }

            float u[EMBED_D];
            float norm_sq = 1e-12f;
            for (int d = 0; d < EMBED_D; ++d) {
                float sum = net->b3[d];
                for (int ic = 0; ic < C2; ++ic) {
                    sum += net->w3[d * C2 + ic] * gap[ic];
                }
                u[d] = sum;
                norm_sq += sum * sum;
            }
            float inv_norm = 1.0f / sqrtf(norm_sq);
            float *u_hat = u_hat_grid[(size_t)gy * grid_w + gx];
            for (int d = 0; d < EMBED_D; ++d) {
                u_hat[d] = u[d] * inv_norm;
            }
        }
    }

    free(in_pad);
    free(a1_pad);
    free(z2_buf);
    free(sat);
    return 0;
}

/* Bilinearly upsample coarse (grid_h x grid_w) anomaly map to (out_h x out_w) */
static void upsample_bilinear(const float *src, int grid_h, int grid_w,
                              float *dst, int out_h, int out_w) {
    float scale_y = (out_h > 1 && grid_h > 1) ? (float)(grid_h - 1) / (float)(out_h - 1) : 0.0f;
    float scale_x = (out_w > 1 && grid_w > 1) ? (float)(grid_w - 1) / (float)(out_w - 1) : 0.0f;

    #pragma omp parallel for
    for (int y = 0; y < out_h; ++y) {
        float fy = (float)y * scale_y;
        int y0 = (int)fy;
        int y1 = (y0 + 1 < grid_h) ? (y0 + 1) : y0;
        float wy = fy - (float)y0;
        const float *row0 = &src[(size_t)y0 * grid_w];
        const float *row1 = &src[(size_t)y1 * grid_w];
        float *dst_row = &dst[(size_t)y * out_w];

        for (int x = 0; x < out_w; ++x) {
            float fx = (float)x * scale_x;
            int x0 = (int)fx;
            int x1 = (x0 + 1 < grid_w) ? (x0 + 1) : x0;
            float wx = fx - (float)x0;

            float top = row0[x0] + wx * (row0[x1] - row0[x0]);
            float bot = row1[x0] + wx * (row1[x1] - row1[x0]);
            dst_row[x] = top + wy * (bot - top);
        }
    }
}

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static int pairs_load_all(int argc, char **argv, int optind, const char *list_path,
                          ImagePair **pairs, size_t *num_pairs, size_t *cap_pairs) {
    int pos_args = argc - optind;
    if (list_path) {
        if (pairs_load_list(list_path, pairs, num_pairs, cap_pairs) != 0) return -1;
    }
    if (pos_args == 1 && strcmp(argv[optind], "-") == 0) {
        if (pairs_load_list("-", pairs, num_pairs, cap_pairs) != 0) return -1;
    } else if (pos_args > 0) {
        if (pos_args % 2 != 0) return -1;
        for (int i = optind; i < argc; i += 2) {
            if (pairs_append(pairs, num_pairs, cap_pairs, argv[i], argv[i + 1]) != 0) return -1;
        }
    }
    return (*num_pairs > 0) ? 0 : -1;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [-tecqu] [-w weights] [-o out.pgm] [-L list.txt] [-n steps] [-l lr]\n"
        "             [-a tile_thresh] [-f alert_frac] [-s stride] [-S seed]\n"
        "             <pre.pgm> <post.pgm> [pre2.pgm post2.pgm ...]\n"
        "options:\n"
        "  -t            train mode (accepts multiple pairs or '-' stdin pair list)\n"
        "  -e            evaluate mode: matched vs. mismatched tile ROC-AUC (no labels)\n"
        "  -c            continue/fine-tune from existing weights file\n"
        "  -u            upsample output change map to full input resolution (bilinear)\n"
        "  -w <file>     weights file path (default: siam.w)\n"
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
    int upsample = 0;
    int quiet = 0;
    int steps = 200;
    int stride = 8;
    float lr = 0.01f;
    float tile_thresh = 0.35f;
    float alert_frac = 0.05f;
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    const char *weights_path = "siam.w";
    const char *list_path = NULL;
    const char *out_path = "-"; /* stdout by default */

    int opt;
    while ((opt = getopt(argc, argv, "tecquw:L:o:n:l:a:f:s:S:h")) != -1) {
        switch (opt) {
            case 't': train_mode = 1; break;
            case 'e': eval_mode = 1; break;
            case 'c': continue_weights = 1; break;
            case 'u': upsample = 1; break;
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
        fprintf(stderr, "siam: cannot specify both -t (train) and -e (evaluate)\n");
        return EXIT_TROUBLE;
    }
    if (stride < 1 || steps < 1) {
        fprintf(stderr, "siam: stride (-s) and steps (-n) must be >= 1\n");
        return EXIT_TROUBLE;
    }

    srand(seed);

    int pos_args = argc - optind;
    SiameseNet net;
    int exit_code = EXIT_NORMAL;

    if (train_mode) {
        ImagePair *pairs = NULL;
        size_t num_pairs = 0, cap_pairs = 0;

        if (pairs_load_all(argc, argv, optind, list_path, &pairs, &num_pairs, &cap_pairs) != 0) {
            usage(argv[0]);
            pairs_free(pairs, num_pairs);
            return EXIT_TROUBLE;
        }

        if (!quiet && num_pairs > 1) {
            fprintf(stderr, "siam: loaded %zu image pairs for training\n", num_pairs);
        }

        if (continue_weights) {
            if (weights_load(weights_path, &net) != 0) {
                pairs_free(pairs, num_pairs);
                return EXIT_TROUBLE;
            }
        } else {
            net_init(&net);
        }

        SiameseNet velocity = {0};

        BranchCache *c_t0 = malloc(BATCH_N * sizeof(BranchCache));
        BranchCache *c_t1 = malloc(BATCH_N * sizeof(BranchCache));
        float (*batch_t0)[PATCH_S * PATCH_S] = malloc(BATCH_N * sizeof(*batch_t0));
        float (*batch_t1)[PATCH_S * PATCH_S] = malloc(BATCH_N * sizeof(*batch_t1));

        if (!c_t0 || !c_t1 || !batch_t0 || !batch_t1) {
            perror("siam: malloc");
            free(c_t0); free(c_t1); free(batch_t0); free(batch_t1);
            pairs_free(pairs, num_pairs);
            return EXIT_TROUBLE;
        }

        const float pi = 3.14159265358979323846f;
        float ema_loss = 0.0f;
        for (int step = 1; step <= steps; ++step) {
            float progress = (steps > 1) ? (float)(step - 1) / (float)(steps - 1) : 0.0f;
            float cur_lr = lr * (0.05f + 0.95f * 0.5f * (1.0f + cosf(pi * progress)));

            int sampled_p[BATCH_N], sampled_y[BATCH_N], sampled_x[BATCH_N];

            for (int b = 0; b < BATCH_N; ++b) {
                size_t p;
                int y0, x0, attempts = 0;
                const PGMImage *pre_img, *post_img;

                /* Rejection-sample to avoid overlapping/duplicate negative tiles in the same batch */
                do {
                    p = (size_t)rand() % num_pairs;
                    pre_img  = &pairs[p].pre;
                    post_img = &pairs[p].post;
                    y0 = (pre_img->height > PATCH_S + 1)
                       ? (1 + rand() % (pre_img->height - PATCH_S - 1)) : 0;
                    x0 = (pre_img->width  > PATCH_S + 1)
                       ? (1 + rand() % (pre_img->width  - PATCH_S - 1)) : 0;

                    int collision = 0;
                    for (int prev = 0; prev < b; ++prev) {
                        if (sampled_p[prev] == (int)p &&
                            abs(sampled_y[prev] - y0) < PATCH_S / 2 &&
                            abs(sampled_x[prev] - x0) < PATCH_S / 2) {
                            collision = 1;
                            break;
                        }
                    }
                    if (!collision) break;
                } while (++attempts < 16);

                sampled_p[b] = (int)p; sampled_y[b] = y0; sampled_x[b] = x0;

                /* +/- 1px co-registration jitter on the post image */
                int jy = (pre_img->height > PATCH_S + 1) ? ((rand() % 3) - 1) : 0;
                int jx = (pre_img->width  > PATCH_S + 1) ? ((rand() % 3) - 1) : 0;
                int d4 = rand() % 8; /* Shared orientation for positive pair */

                float gain0 = 1.0f + rand_uniform(0.06f), bias0 = rand_uniform(0.03f);
                float gain1 = 1.0f + rand_uniform(0.06f), bias1 = rand_uniform(0.03f);

                extract_patch_aug(pre_img,  y0,      x0,      batch_t0[b], gain0, bias0, d4);
                extract_patch_aug(post_img, y0 + jy, x0 + jx, batch_t1[b], gain1, bias1, d4);
            }

            float loss = infonce_step(&net, &velocity, c_t0, c_t1,
                                      batch_t0, batch_t1, cur_lr, 1e-4f);
            ema_loss = (step == 1) ? loss : (0.95f * ema_loss + 0.05f * loss);
            if (!quiet && (step == 1 || step % 25 == 0 || step == steps)) {
                fprintf(stderr, "siam: step %4d/%d lr=%.5f loss=%.5f ema=%.5f\n",
                        step, steps, cur_lr, loss, ema_loss);
            }
        }

        if (weights_save(weights_path, &net) != 0) {
            exit_code = EXIT_TROUBLE;
        }

        free(c_t0); free(c_t1); free(batch_t0); free(batch_t1);
        pairs_free(pairs, num_pairs);
    } else if (eval_mode) {
        /* Evaluate mode (-e): Matched vs. Mismatched Tile ROC-AUC across >= 2 pairs */
        ImagePair *pairs = NULL;
        size_t num_pairs = 0, cap_pairs = 0;

        if (pairs_load_all(argc, argv, optind, list_path, &pairs, &num_pairs, &cap_pairs) != 0 ||
            num_pairs < 2) {
            fprintf(stderr, "siam: evaluate mode (-e) requires at least 2 image pairs\n");
            pairs_free(pairs, num_pairs);
            return EXIT_TROUBLE;
        }

        if (weights_load(weights_path, &net) != 0) {
            pairs_free(pairs, num_pairs);
            return EXIT_TROUBLE;
        }

        size_t shift = num_pairs / 2;
        size_t *pair_offsets = malloc((num_pairs + 1) * sizeof(size_t));
        if (!pair_offsets) {
            perror("siam: malloc offsets");
            pairs_free(pairs, num_pairs);
            return EXIT_TROUBLE;
        }

        pair_offsets[0] = 0;
        for (size_t p = 0; p < num_pairs; ++p) {
            size_t q = (p + shift) % num_pairs;
            int h = (pairs[p].pre.height < pairs[q].post.height) ? pairs[p].pre.height : pairs[q].post.height;
            int w = (pairs[p].pre.width  < pairs[q].post.width)  ? pairs[p].pre.width  : pairs[q].post.width;
            size_t gh = (size_t)((h - PATCH_S) / stride + 1);
            size_t gw = (size_t)((w - PATCH_S) / stride + 1);
            pair_offsets[p + 1] = pair_offsets[p] + gh * gw;
        }

        size_t total_tiles = pair_offsets[num_pairs];
        float *same_dists = malloc(total_tiles * sizeof(float));
        float *diff_dists = malloc(total_tiles * sizeof(float));
        if (!same_dists || !diff_dists) {
            perror("siam: malloc eval dists");
            free(pair_offsets); free(same_dists); free(diff_dists);
            pairs_free(pairs, num_pairs);
            return EXIT_TROUBLE;
        }

        size_t same_alerts = 0, diff_alerts = 0;
        double same_sum = 0.0, diff_sum = 0.0;
        int eval_err = 0;

        #pragma omp parallel for reduction(+:same_alerts,diff_alerts,same_sum,diff_sum) reduction(|:eval_err)
        for (size_t p = 0; p < num_pairs; ++p) {
            size_t q = (p + shift) % num_pairs;
            int h = (pairs[p].pre.height < pairs[q].post.height) ? pairs[p].pre.height : pairs[q].post.height;
            int w = (pairs[p].pre.width  < pairs[q].post.width)  ? pairs[p].pre.width  : pairs[q].post.width;
            int gh = (h - PATCH_S) / stride + 1;
            int gw = (w - PATCH_S) / stride + 1;
            size_t n_tiles = (size_t)gh * (size_t)gw;
            size_t base_idx = pair_offsets[p];

            float (*u_pre)[EMBED_D]  = malloc(n_tiles * sizeof(*u_pre));
            float (*u_same)[EMBED_D] = malloc(n_tiles * sizeof(*u_same));
            float (*u_diff)[EMBED_D] = malloc(n_tiles * sizeof(*u_diff));

            if (!u_pre || !u_same || !u_diff ||
                scene_extract_embeddings(&net, &pairs[p].pre,  h, w, stride, u_pre)  != 0 ||
                scene_extract_embeddings(&net, &pairs[p].post, h, w, stride, u_same) != 0 ||
                scene_extract_embeddings(&net, &pairs[q].post, h, w, stride, u_diff) != 0) {
                free(u_pre); free(u_same); free(u_diff);
                eval_err = 1;
                continue;
            }

            int anom_same = 0, anom_diff = 0;
            for (size_t t = 0; t < n_tiles; ++t) {
                float sim_same = 0.0f, sim_diff = 0.0f;
                for (int d = 0; d < EMBED_D; ++d) {
                    sim_same += u_pre[t][d] * u_same[t][d];
                    sim_diff += u_pre[t][d] * u_diff[t][d];
                }
                float dist_same = 0.5f * (1.0f - sim_same);
                float dist_diff = 0.5f * (1.0f - sim_diff);

                same_dists[base_idx + t] = dist_same;
                diff_dists[base_idx + t] = dist_diff;
                same_sum += (double)dist_same;
                diff_sum += (double)dist_diff;

                if (dist_same >= tile_thresh) anom_same++;
                if (dist_diff >= tile_thresh) anom_diff++;
            }

            if ((float)anom_same / (float)n_tiles >= alert_frac) same_alerts++;
            if ((float)anom_diff / (float)n_tiles >= alert_frac) diff_alerts++;

            free(u_pre);
            free(u_same);
            free(u_diff);
        }

        if (eval_err) {
            free(pair_offsets); free(same_dists); free(diff_dists);
            pairs_free(pairs, num_pairs);
            return EXIT_TROUBLE;
        }

        qsort(same_dists, total_tiles, sizeof(float), cmp_float);
        qsort(diff_dists, total_tiles, sizeof(float), cmp_float);

        /* Exact Mann-Whitney U ROC-AUC via two-pointer scan over sorted arrays */
        double u_sum = 0.0;
        size_t p_lt = 0, p_le = 0;
        for (size_t k = 0; k < total_tiles; ++k) {
            float d = diff_dists[k];
            while (p_lt < total_tiles && same_dists[p_lt] < d) p_lt++;
            if (p_le < p_lt) p_le = p_lt;
            while (p_le < total_tiles && same_dists[p_le] <= d) p_le++;
            u_sum += (double)p_lt + 0.5 * (double)(p_le - p_lt);
        }
        double roc_auc = u_sum / ((double)total_tiles * (double)total_tiles);
        double same_mean = same_sum / (double)total_tiles;
        double diff_mean = diff_sum / (double)total_tiles;
        float same_med = same_dists[total_tiles / 2];
        float diff_med = diff_dists[total_tiles / 2];
        float same_p95 = same_dists[(total_tiles * 95) / 100];
        float diff_p95 = diff_dists[(total_tiles * 95) / 100];

        printf("siam: evaluated %zu pairs (%zu tiles, stride=%d, thresh=%.2f)\n"
               "  matched    (pre_i, post_i): mean=%.4f med=%.4f p95=%.4f alerts=%zu/%zu (%.1f%%)\n"
               "  mismatched (pre_i, post_j): mean=%.4f med=%.4f p95=%.4f alerts=%zu/%zu (%.1f%%)\n"
               "  separation gap:             %+.4f\n"
               "  tile ROC-AUC:               %.4f\n",
               num_pairs, total_tiles, stride, tile_thresh,
               same_mean, same_med, same_p95, same_alerts, num_pairs, 100.0 * (double)same_alerts / (double)num_pairs,
               diff_mean, diff_med, diff_p95, diff_alerts, num_pairs, 100.0 * (double)diff_alerts / (double)num_pairs,
               diff_mean - same_mean, roc_auc);

        free(pair_offsets);
        free(same_dists);
        free(diff_dists);
        pairs_free(pairs, num_pairs);
    } else {
        /* Default mode: Inference (expects exactly <pre.pgm> <post.pgm>) */
        if (list_path || pos_args != 2) {
            usage(argv[0]);
            return EXIT_TROUBLE;
        }

        ImagePair *pair = NULL;
        size_t num_pairs = 0, cap_pairs = 0;
        if (pairs_append(&pair, &num_pairs, &cap_pairs, argv[optind], argv[optind + 1]) != 0) {
            pairs_free(pair, num_pairs);
            return EXIT_TROUBLE;
        }

        const PGMImage pre  = pair[0].pre;
        const PGMImage post = pair[0].post;

        if (weights_load(weights_path, &net) != 0) {
            pairs_free(pair, num_pairs);
            return EXIT_TROUBLE;
        }

        int grid_h = (pre.height - PATCH_S) / stride + 1;
        int grid_w = (pre.width  - PATCH_S) / stride + 1;
        size_t n_tiles = (size_t)grid_h * (size_t)grid_w;

        float *change_map = calloc(n_tiles, sizeof(float));
        float (*u0)[EMBED_D] = malloc(n_tiles * sizeof(*u0));
        float (*u1)[EMBED_D] = malloc(n_tiles * sizeof(*u1));

        if (!change_map || !u0 || !u1 ||
            scene_extract_embeddings(&net, &pre,  pre.height,  pre.width,  stride, u0) != 0 ||
            scene_extract_embeddings(&net, &post, post.height, post.width, stride, u1) != 0) {
            free(change_map); free(u0); free(u1);
            pairs_free(pair, num_pairs);
            return EXIT_TROUBLE;
        }

        int anomalous_tiles = 0;

        #pragma omp parallel for reduction(+:anomalous_tiles)
        for (size_t t = 0; t < n_tiles; ++t) {
            float cos_sim = 0.0f;
            for (int d = 0; d < EMBED_D; ++d) cos_sim += u0[t][d] * u1[t][d];
            float cos_dist = 0.5f * (1.0f - cos_sim);
            change_map[t] = cos_dist;
            if (cos_dist >= tile_thresh) anomalous_tiles++;
        }

        free(u0);
        free(u1);

        int write_rc;
        if (upsample) {
            float *full_map = malloc((size_t)pre.height * (size_t)pre.width * sizeof(float));
            if (!full_map) {
                perror("siam: malloc upsample");
                free(change_map);
                pairs_free(pair, num_pairs);
                return EXIT_TROUBLE;
            }
            upsample_bilinear(change_map, grid_h, grid_w, full_map, pre.height, pre.width);
            write_rc = pgm_write(out_path, full_map, pre.width, pre.height);
            free(full_map);
        } else {
            write_rc = pgm_write(out_path, change_map, grid_w, grid_h);
        }

        if (write_rc != 0) {
            free(change_map);
            pairs_free(pair, num_pairs);
            return EXIT_TROUBLE;
        }

        float frac = (float)anomalous_tiles / (float)n_tiles;
        int is_alert = (frac >= alert_frac);

        if (!quiet) {
            fprintf(stderr, "siam: fraction=%.4f (%d/%zu tiles >= %.2f) status=%s\n",
                    frac, anomalous_tiles, n_tiles, tile_thresh,
                    is_alert ? "ALERT" : "NORMAL");
        }

        free(change_map);
        pairs_free(pair, num_pairs);
        exit_code = is_alert ? EXIT_ALERT : EXIT_NORMAL;
    }

    return exit_code;
}
