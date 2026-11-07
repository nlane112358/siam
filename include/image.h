/*
 * image.h
 * Netpbm PGM (P5 & P2, 8-bit and 16-bit) Image Object & Patch Extraction
 */

#ifndef PGMCD_IMAGE_H
#define PGMCD_IMAGE_H

#include <stdio.h>

typedef struct {
    int width;
    int height;
    int maxval;
    float *pixels; /* Normalised to [0.0, 1.0], row-major [height * width] */
} PGMImage;

void pgm_skip_comments_and_ws(FILE *fp);

int  pgm_image_read(PGMImage *self, const char *path);
int  pgm_image_write(const PGMImage *self, const char *path);
void pgm_image_destroy(PGMImage *self);

/* Extracts and zero-centers a patch_s x patch_s window at (y0, x0) */
void pgm_image_extract_patch(const PGMImage *self, int y0, int x0, int patch_s,
                             float gain, float bias, float *out);

#endif /* PGMCD_IMAGE_H */
