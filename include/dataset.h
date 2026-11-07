/*
 * dataset.h
 * Multi-Pair Satellite Image Dataset Object & Mini-Batch Sampling
 */

#ifndef PGMCD_DATASET_H
#define PGMCD_DATASET_H

#include <stddef.h>
#include "image.h"
#include "network.h"

typedef struct {
    PGMImage pre;
    PGMImage post;
} ImagePair;

typedef struct {
    ImagePair *items;
    size_t count;
    size_t capacity;
} PairDataset;

void pair_dataset_init(PairDataset *self);
void pair_dataset_destroy(PairDataset *self);

int  pair_dataset_append(PairDataset *self, const char *pre_path, const char *post_path);
int  pair_dataset_load_list(PairDataset *self, const char *list_path);
int  pair_dataset_load_cli(PairDataset *self, int argc, char **argv, int optind,
                           const char *list_path);

void pair_dataset_sample_batch(const PairDataset *self, int batch_n,
                               float (*batch_t0)[PATCH_S * PATCH_S],
                               float (*batch_t1)[PATCH_S * PATCH_S]);

#endif /* PGMCD_DATASET_H */
