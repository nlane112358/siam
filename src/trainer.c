/*
 * trainer.c
 * Contrastive Trainer Object (Symmetric InfoNCE Loss, SGD-Momentum & Cosine LR)
 */

#include "trainer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int trainer_init(ContrastiveTrainer *self, TrainerConfig cfg) {
    self->cfg = cfg;
    siamese_net_zero(&self->velocity);

    self->cache_t0 = malloc(BATCH_N * sizeof(BranchCache));
    self->cache_t1 = malloc(BATCH_N * sizeof(BranchCache));
    self->batch_t0 = malloc(BATCH_N * sizeof(*self->batch_t0));
    self->batch_t1 = malloc(BATCH_N * sizeof(*self->batch_t1));

    if (!self->cache_t0 || !self->cache_t1 || !self->batch_t0 || !self->batch_t1) {
        perror("pgmcd: malloc");
        trainer_destroy(self);
        return -1;
    }
    return 0;
}

void trainer_destroy(ContrastiveTrainer *self) {
    if (!self) return;
    free(self->cache_t0);
    free(self->cache_t1);
    free(self->batch_t0);
    free(self->batch_t1);
    self->cache_t0 = NULL;
    self->cache_t1 = NULL;
    self->batch_t0 = NULL;
    self->batch_t1 = NULL;
}

float trainer_step(ContrastiveTrainer *self, SiameseNet *net, float cur_lr) {
    BranchCache *cache_t0 = self->cache_t0;
    BranchCache *cache_t1 = self->cache_t1;

    /* Forward pass across mini-batch in parallel */
    #pragma omp parallel for
    for (int i = 0; i < BATCH_N; ++i) {
        siamese_net_forward(net, self->batch_t0[i], &cache_t0[i]);
        siamese_net_forward(net, self->batch_t1[i], &cache_t1[i]);
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

    /* Accumulate parameter gradients across both twin branches in parallel */
    SiameseNet grad;
    siamese_net_zero(&grad);
    #pragma omp parallel
    {
        SiameseNet local_grad;
        siamese_net_zero(&local_grad);
        #pragma omp for nowait
        for (int i = 0; i < BATCH_N; ++i) {
            siamese_net_backward(net, &local_grad, &cache_t0[i], d_u_hat[i]);
            siamese_net_backward(net, &local_grad, &cache_t1[i], d_v_hat[i]);
        }
        #pragma omp critical
        {
            float *dst = (float *)&grad;
            const float *src = (const float *)&local_grad;
            size_t n = sizeof(SiameseNet) / sizeof(float);
            for (size_t k = 0; k < n; ++k) dst[k] += src[k];
        }
    }

    /* Update shared weights with momentum (0.9) + Decoupled Weight Decay */
    float *w_ptr = (float *)net;
    float *g_ptr = (float *)&grad;
    float *v_ptr = (float *)&self->velocity;
    size_t num_params = sizeof(SiameseNet) / sizeof(float);

    for (size_t k = 0; k < num_params; ++k) {
        v_ptr[k] = 0.9f * v_ptr[k] + g_ptr[k];
        w_ptr[k] = w_ptr[k] * (1.0f - cur_lr * self->cfg.weight_decay) - cur_lr * v_ptr[k];
    }

    return 0.5f * (loss_row + loss_col) / (float)BATCH_N;
}

void trainer_fit(ContrastiveTrainer *self, SiameseNet *net, const PairDataset *dataset) {
    const float pi = 3.14159265358979323846f;
    float ema_loss = 0.0f;
    int steps = self->cfg.steps;
    float lr = self->cfg.lr;

    for (int step = 1; step <= steps; ++step) {
        float progress = (steps > 1) ? (float)(step - 1) / (float)(steps - 1) : 0.0f;
        float cur_lr = lr * (0.05f + 0.95f * 0.5f * (1.0f + cosf(pi * progress)));

        pair_dataset_sample_batch(dataset, BATCH_N, self->batch_t0, self->batch_t1);
        float loss = trainer_step(self, net, cur_lr);

        ema_loss = (step == 1) ? loss : (0.95f * ema_loss + 0.05f * loss);
        if (!self->cfg.quiet && (step == 1 || step % 25 == 0 || step == steps)) {
            fprintf(stderr, "pgmcd: step %4d/%d lr=%.5f loss=%.5f ema=%.5f\n",
                    step, steps, cur_lr, loss, ema_loss);
        }
    }
}
