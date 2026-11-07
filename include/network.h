/*
 * network.h
 * Siamese Encoder Parameters, Activations, Forward/Backward & Serialization
 */

#ifndef PGMCD_NETWORK_H
#define PGMCD_NETWORK_H

/* --- Network Hyperparameters --- */
#define PATCH_S     32      /* 32x32 spatial patch */
#define C1          16      /* Conv1 output channels */
#define H1          16      /* 32 / 2 */
#define C2          32      /* Conv2 output channels */
#define H2          8       /* 16 / 2 */
#define EMBED_D     32      /* Final L2-normalised embedding dimension */
#define BATCH_N     64      /* Contrastive mini-batch size (pairs per step) */
#define TEMP        0.07f   /* InfoNCE temperature */

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

float rand_uniform(float scale);

void  siamese_net_init(SiameseNet *self);
void  siamese_net_zero(SiameseNet *self);
int   siamese_net_load(SiameseNet *self, const char *path);
int   siamese_net_save(const SiameseNet *self, const char *path);

void  siamese_net_forward(const SiameseNet *self, const float *patch, BranchCache *cache);
void  siamese_net_backward(const SiameseNet *self, SiameseNet *grad,
                           const BranchCache *cache, const float *d_u_hat);

float branch_cache_cosine_distance(const BranchCache *a, const BranchCache *b);

#endif /* PGMCD_NETWORK_H */
