/*
 * trainer.h
 * Contrastive Trainer Object (Symmetric InfoNCE Loss, SGD-Momentum & Cosine LR)
 */

#ifndef PGMCD_TRAINER_H
#define PGMCD_TRAINER_H

#include "dataset.h"
#include "network.h"

typedef struct {
    int   steps;
    float lr;
    float weight_decay;
    int   quiet;
} TrainerConfig;

typedef struct {
    TrainerConfig cfg;
    SiameseNet    velocity;
    BranchCache  *cache_t0;
    BranchCache  *cache_t1;
    float       (*batch_t0)[PATCH_S * PATCH_S];
    float       (*batch_t1)[PATCH_S * PATCH_S];
} ContrastiveTrainer;

int   trainer_init(ContrastiveTrainer *self, TrainerConfig cfg);
void  trainer_destroy(ContrastiveTrainer *self);

float trainer_step(ContrastiveTrainer *self, SiameseNet *net, float cur_lr);
void  trainer_fit(ContrastiveTrainer *self, SiameseNet *net, const PairDataset *dataset);

#endif /* PGMCD_TRAINER_H */
