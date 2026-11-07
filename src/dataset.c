/*
 * dataset.c
 * Multi-Pair Satellite Image Dataset Object & Mini-Batch Sampling
 */

#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void pair_dataset_init(PairDataset *self) {
    self->items = NULL;
    self->count = 0;
    self->capacity = 0;
}

void pair_dataset_destroy(PairDataset *self) {
    if (!self || !self->items) return;
    for (size_t i = 0; i < self->count; ++i) {
        pgm_image_destroy(&self->items[i].pre);
        pgm_image_destroy(&self->items[i].post);
    }
    free(self->items);
    self->items = NULL;
    self->count = 0;
    self->capacity = 0;
}

int pair_dataset_append(PairDataset *self, const char *pre_path, const char *post_path) {
    if (strcmp(pre_path, "-") == 0 && strcmp(post_path, "-") == 0) {
        fprintf(stderr, "pgmcd: cannot stream both pre and post images from stdin\n");
        return -1;
    }

    PGMImage pre = {0}, post = {0};
    if (pgm_image_read(&pre, pre_path) != 0 || pgm_image_read(&post, post_path) != 0) {
        pgm_image_destroy(&pre);
        pgm_image_destroy(&post);
        return -1;
    }

    if (pre.width != post.width || pre.height != post.height ||
        pre.width < PATCH_S || pre.height < PATCH_S) {
        fprintf(stderr, "pgmcd: dimension mismatch or image smaller than %dx%d (%s, %s)\n",
                PATCH_S, PATCH_S, pre_path, post_path);
        pgm_image_destroy(&pre);
        pgm_image_destroy(&post);
        return -1;
    }

    if (self->count == self->capacity) {
        size_t new_cap = (self->capacity == 0) ? 16 : (self->capacity * 2);
        ImagePair *grown = realloc(self->items, new_cap * sizeof(ImagePair));
        if (!grown) {
            perror("pgmcd: realloc pairs");
            pgm_image_destroy(&pre);
            pgm_image_destroy(&post);
            return -1;
        }
        self->items = grown;
        self->capacity = new_cap;
    }

    self->items[self->count].pre  = pre;
    self->items[self->count].post = post;
    self->count++;
    return 0;
}

int pair_dataset_load_list(PairDataset *self, const char *list_path) {
    FILE *fp = (strcmp(list_path, "-") == 0) ? stdin : fopen(list_path, "r");
    if (!fp) {
        perror("pgmcd: fopen pair list");
        return -1;
    }

    char pre_path[4096], post_path[4096];
    int rc = 0;
    while (1) {
        pgm_skip_comments_and_ws(fp);
        int n1 = fscanf(fp, "%4095s", pre_path);
        if (n1 == EOF) break;
        pgm_skip_comments_and_ws(fp);
        int n2 = fscanf(fp, "%4095s", post_path);
        if (n1 != 1 || n2 != 1) {
            fprintf(stderr, "pgmcd: pair list %s has an unpaired image path\n", list_path);
            rc = -1;
            break;
        }
        if (pair_dataset_append(self, pre_path, post_path) != 0) {
            rc = -1;
            break;
        }
    }

    if (fp != stdin) fclose(fp);
    return rc;
}

int pair_dataset_load_cli(PairDataset *self, int argc, char **argv, int optind,
                          const char *list_path) {
    int pos_args = argc - optind;
    if (list_path) {
        if (pair_dataset_load_list(self, list_path) != 0) return -1;
    }
    if (pos_args == 1 && strcmp(argv[optind], "-") == 0) {
        if (pair_dataset_load_list(self, "-") != 0) return -1;
    } else if (pos_args > 0) {
        if (pos_args % 2 != 0) return -1;
        for (int i = optind; i < argc; i += 2) {
            if (pair_dataset_append(self, argv[i], argv[i + 1]) != 0) return -1;
        }
    }
    return (self->count > 0) ? 0 : -1;
}

void pair_dataset_sample_batch(const PairDataset *self, int batch_n,
                               float (*batch_t0)[PATCH_S * PATCH_S],
                               float (*batch_t1)[PATCH_S * PATCH_S]) {
    for (int b = 0; b < batch_n; ++b) {
        size_t p = (size_t)rand() % self->count;
        const PGMImage *pre_img  = &self->items[p].pre;
        const PGMImage *post_img = &self->items[p].post;
        int y0 = rand() % (pre_img->height - PATCH_S + 1);
        int x0 = rand() % (pre_img->width  - PATCH_S + 1);
        float gain0 = 1.0f + rand_uniform(0.06f);
        float bias0 = rand_uniform(0.03f);
        float gain1 = 1.0f + rand_uniform(0.06f);
        float bias1 = rand_uniform(0.03f);
        pgm_image_extract_patch(pre_img,  y0, x0, PATCH_S, gain0, bias0, batch_t0[b]);
        pgm_image_extract_patch(post_img, y0, x0, PATCH_S, gain1, bias1, batch_t1[b]);
    }
}
