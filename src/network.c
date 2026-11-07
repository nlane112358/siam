/*
 * network.c
 * Siamese Encoder Parameters, Activations, Forward/Backward & Serialization
 */

#include "network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float rand_uniform(float scale) {
    return (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
}

void siamese_net_init(SiameseNet *self) {
    /* He / Kaiming uniform initialisation */
    float s1 = sqrtf(6.0f / 9.0f);
    for (int i = 0; i < C1 * 9; ++i) self->w1[i] = rand_uniform(s1);
    memset(self->b1, 0, sizeof(self->b1));

    float s2 = sqrtf(6.0f / (C1 * 9.0f));
    for (int i = 0; i < C2 * C1 * 9; ++i) self->w2[i] = rand_uniform(s2);
    memset(self->b2, 0, sizeof(self->b2));

    float s3 = sqrtf(6.0f / (float)C2);
    for (int i = 0; i < EMBED_D * C2; ++i) self->w3[i] = rand_uniform(s3);
    memset(self->b3, 0, sizeof(self->b3));
}

void siamese_net_zero(SiameseNet *self) {
    memset(self, 0, sizeof(*self));
}

int siamese_net_load(SiameseNet *self, const char *path) {
    FILE *wf = fopen(path, "rb");
    if (!wf || fread(self, sizeof(SiameseNet), 1, wf) != 1) {
        perror("pgmcd: fread weights");
        if (wf) fclose(wf);
        return -1;
    }
    fclose(wf);
    return 0;
}

int siamese_net_save(const SiameseNet *self, const char *path) {
    FILE *wf = fopen(path, "wb");
    if (!wf || fwrite(self, sizeof(SiameseNet), 1, wf) != 1) {
        perror("pgmcd: fwrite weights");
        if (wf) fclose(wf);
        return -1;
    }
    fclose(wf);
    return 0;
}

/* Forward pass for a single 32x32 patch through one branch */
void siamese_net_forward(const SiameseNet *self, const float *patch, BranchCache *c) {
    memcpy(c->input, patch, sizeof(c->input));

    /* Conv1: 1 -> C1, 3x3, stride 2, pad 1 + LeakyReLU(0.1) */
    for (int oc = 0; oc < C1; ++oc) {
        for (int oh = 0; oh < H1; ++oh) {
            for (int ow = 0; ow < H1; ++ow) {
                float sum = self->b1[oc];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int ih = oh * 2 + kh - 1;
                        int iw = ow * 2 + kw - 1;
                        if (ih >= 0 && ih < PATCH_S && iw >= 0 && iw < PATCH_S) {
                            sum += c->input[ih * PATCH_S + iw] * self->w1[(oc * 3 + kh) * 3 + kw];
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
                float sum = self->b2[oc];
                for (int ic = 0; ic < C1; ++ic) {
                    const float *w_ptr = &self->w2[((oc * C1 + ic) * 3) * 3];
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
        float sum = self->b3[d];
        for (int ic = 0; ic < C2; ++ic) {
            sum += self->w3[d * C2 + ic] * c->gap[ic];
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
void siamese_net_backward(const SiameseNet *self, SiameseNet *grad,
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
            d_gap[ic] += self->w3[d * C2 + ic] * d_u[d];
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
                    const float *w_ptr = &self->w2[((oc * C1 + ic) * 3) * 3];
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

float branch_cache_cosine_distance(const BranchCache *a, const BranchCache *b) {
    float cos_sim = 0.0f;
    for (int d = 0; d < EMBED_D; ++d) {
        cos_sim += a->u_hat[d] * b->u_hat[d];
    }
    return 0.5f * (1.0f - cos_sim);
}
