/*
 * image.c
 * Netpbm PGM (P5 & P2, 8-bit and 16-bit) Image Object & Patch Extraction
 */

#include "image.h"

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void pgm_skip_comments_and_ws(FILE *fp) {
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

int pgm_image_read(PGMImage *self, const char *path) {
    FILE *fp = (strcmp(path, "-") == 0) ? stdin : fopen(path, "rb");
    if (!fp) { perror("fopen pgm_read"); return -1; }

    char magic[3] = {0};
    if (fread(magic, 1, 2, fp) != 2 || (strcmp(magic, "P5") != 0 && strcmp(magic, "P2") != 0)) {
        fprintf(stderr, "Error: %s is not a valid P5/P2 PGM file\n", path);
        if (fp != stdin) fclose(fp);
        return -1;
    }

    pgm_skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &self->width) != 1) { if (fp != stdin) fclose(fp); return -1; }
    pgm_skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &self->height) != 1) { if (fp != stdin) fclose(fp); return -1; }
    pgm_skip_comments_and_ws(fp);
    if (fscanf(fp, "%d", &self->maxval) != 1) { if (fp != stdin) fclose(fp); return -1; }
    fgetc(fp); /* Consume single whitespace character after maxval */

    size_t npixels = (size_t)self->width * self->height;
    self->pixels = malloc(npixels * sizeof(float));
    if (!self->pixels) { if (fp != stdin) fclose(fp); return -1; }

    float inv_max = 1.0f / (float)(self->maxval > 0 ? self->maxval : 255);

    if (strcmp(magic, "P5") == 0) {
        if (self->maxval < 256) {
            uint8_t *buf = malloc(npixels);
            if (!buf || fread(buf, 1, npixels, fp) != npixels) {
                free(buf);
                if (fp != stdin) fclose(fp);
                return -1;
            }
            for (size_t i = 0; i < npixels; ++i) self->pixels[i] = (float)buf[i] * inv_max;
            free(buf);
        } else {
            /* 16-bit PGM is big-endian (MSB first) per Netpbm spec */
            uint8_t *buf = malloc(npixels * 2);
            if (!buf || fread(buf, 2, npixels, fp) != npixels) {
                free(buf);
                if (fp != stdin) fclose(fp);
                return -1;
            }
            for (size_t i = 0; i < npixels; ++i) {
                uint16_t val = ((uint16_t)buf[2 * i] << 8) | (uint16_t)buf[2 * i + 1];
                self->pixels[i] = (float)val * inv_max;
            }
            free(buf);
        }
    } else {
        for (size_t i = 0; i < npixels; ++i) {
            int val = 0;
            if (fscanf(fp, "%d", &val) != 1) break;
            self->pixels[i] = (float)val * inv_max;
        }
    }

    if (fp != stdin) fclose(fp);
    return 0;
}

int pgm_image_write(const PGMImage *self, const char *path) {
    FILE *fp = (strcmp(path, "-") == 0) ? stdout : fopen(path, "wb");
    if (!fp) { perror("fopen pgm_write"); return -1; }

    fprintf(fp, "P5\n%d %d\n255\n", self->width, self->height);
    size_t npixels = (size_t)self->width * self->height;
    uint8_t *buf = malloc(npixels);
    if (!buf) {
        if (fp != stdout) fclose(fp);
        return -1;
    }
    for (size_t i = 0; i < npixels; ++i) {
        float v = self->pixels[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        buf[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
    fwrite(buf, 1, npixels, fp);
    free(buf);
    if (fp != stdout) fclose(fp);
    return 0;
}

void pgm_image_destroy(PGMImage *self) {
    if (!self) return;
    free(self->pixels);
    self->pixels = NULL;
}

void pgm_image_extract_patch(const PGMImage *self, int y0, int x0, int patch_s,
                             float gain, float bias, float *out) {
    for (int dy = 0; dy < patch_s; ++dy) {
        for (int dx = 0; dx < patch_s; ++dx) {
            float v = self->pixels[(y0 + dy) * self->width + (x0 + dx)] * gain + bias;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            /* Zero-center [0, 1] -> [-1, +1] while preserving backscatter drops */
            out[dy * patch_s + dx] = (v - 0.5f) * 2.0f;
        }
    }
}
