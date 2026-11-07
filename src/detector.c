/*
 * detector.c
 * Change Detector Object (Sliding-Window Inference & Label-Free ROC-AUC Evaluation)
 */

#include "detector.h"

#include <stdio.h>
#include <stdlib.h>

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

void change_detector_init(ChangeDetector *self, const SiameseNet *net, DetectorConfig cfg) {
    self->net = net;
    self->cfg = cfg;
}

int change_detector_infer(const ChangeDetector *self, const ImagePair *pair,
                          DetectionResult *out_result) {
    const PGMImage *pre  = &pair->pre;
    const PGMImage *post = &pair->post;
    int stride = self->cfg.stride;
    float tile_thresh = self->cfg.tile_thresh;

    int grid_h = (pre->height - PATCH_S) / stride + 1;
    int grid_w = (pre->width  - PATCH_S) / stride + 1;
    float *change_map = calloc((size_t)grid_h * grid_w, sizeof(float));
    if (!change_map) {
        perror("pgmcd: calloc");
        return -1;
    }

    int anomalous_tiles = 0;

    #pragma omp parallel for collapse(2) reduction(+:anomalous_tiles)
    for (int gy = 0; gy < grid_h; ++gy) {
        for (int gx = 0; gx < grid_w; ++gx) {
            float p0[PATCH_S * PATCH_S], p1[PATCH_S * PATCH_S];
            BranchCache c0, c1;
            pgm_image_extract_patch(pre,  gy * stride, gx * stride, PATCH_S, 1.0f, 0.0f, p0);
            pgm_image_extract_patch(post, gy * stride, gx * stride, PATCH_S, 1.0f, 0.0f, p1);
            siamese_net_forward(self->net, p0, &c0);
            siamese_net_forward(self->net, p1, &c1);

            float cos_dist = branch_cache_cosine_distance(&c0, &c1);
            change_map[gy * grid_w + gx] = cos_dist;
            if (cos_dist >= tile_thresh) anomalous_tiles++;
        }
    }

    out_result->change_map.width  = grid_w;
    out_result->change_map.height = grid_h;
    out_result->change_map.maxval = 255;
    out_result->change_map.pixels = change_map;
    out_result->anomalous_tiles   = anomalous_tiles;
    out_result->total_tiles       = grid_h * grid_w;
    out_result->fraction          = (float)anomalous_tiles / (float)(grid_h * grid_w);
    out_result->is_alert          = (out_result->fraction >= self->cfg.alert_frac);

    if (!self->cfg.quiet) {
        fprintf(stderr, "pgmcd: fraction=%.4f (%d/%d tiles >= %.2f) status=%s\n",
                out_result->fraction, out_result->anomalous_tiles,
                out_result->total_tiles, tile_thresh,
                out_result->is_alert ? "ALERT" : "NORMAL");
    }

    return 0;
}

void detection_result_destroy(DetectionResult *self) {
    if (!self) return;
    pgm_image_destroy(&self->change_map);
}

int change_detector_evaluate(const ChangeDetector *self, const PairDataset *dataset,
                             EvalMetrics *out_metrics) {
    size_t num_pairs = dataset->count;
    const ImagePair *pairs = dataset->items;
    int stride = self->cfg.stride;
    float tile_thresh = self->cfg.tile_thresh;
    float alert_frac = self->cfg.alert_frac;

    size_t shift = num_pairs / 2;
    size_t *pair_offsets = malloc((num_pairs + 1) * sizeof(size_t));
    if (!pair_offsets) {
        perror("pgmcd: malloc offsets");
        return -1;
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
        perror("pgmcd: malloc eval dists");
        free(pair_offsets);
        free(same_dists);
        free(diff_dists);
        return -1;
    }

    size_t same_alerts = 0, diff_alerts = 0;
    double same_sum = 0.0, diff_sum = 0.0;

    #pragma omp parallel for reduction(+:same_alerts,diff_alerts,same_sum,diff_sum)
    for (size_t p = 0; p < num_pairs; ++p) {
        size_t q = (p + shift) % num_pairs;
        int h = (pairs[p].pre.height < pairs[q].post.height) ? pairs[p].pre.height : pairs[q].post.height;
        int w = (pairs[p].pre.width  < pairs[q].post.width)  ? pairs[p].pre.width  : pairs[q].post.width;
        int gh = (h - PATCH_S) / stride + 1;
        int gw = (w - PATCH_S) / stride + 1;
        size_t base_idx = pair_offsets[p];
        int anom_same = 0, anom_diff = 0;

        for (int gy = 0; gy < gh; ++gy) {
            for (int gx = 0; gx < gw; ++gx) {
                float p_pre[PATCH_S * PATCH_S], p_same[PATCH_S * PATCH_S], p_diff[PATCH_S * PATCH_S];
                BranchCache c_pre, c_same, c_diff;
                pgm_image_extract_patch(&pairs[p].pre,  gy * stride, gx * stride, PATCH_S, 1.0f, 0.0f, p_pre);
                pgm_image_extract_patch(&pairs[p].post, gy * stride, gx * stride, PATCH_S, 1.0f, 0.0f, p_same);
                pgm_image_extract_patch(&pairs[q].post, gy * stride, gx * stride, PATCH_S, 1.0f, 0.0f, p_diff);

                siamese_net_forward(self->net, p_pre,  &c_pre);
                siamese_net_forward(self->net, p_same, &c_same);
                siamese_net_forward(self->net, p_diff, &c_diff);

                float dist_same = branch_cache_cosine_distance(&c_pre, &c_same);
                float dist_diff = branch_cache_cosine_distance(&c_pre, &c_diff);

                size_t idx = base_idx + (size_t)gy * (size_t)gw + (size_t)gx;
                same_dists[idx] = dist_same;
                diff_dists[idx] = dist_diff;
                same_sum += (double)dist_same;
                diff_sum += (double)dist_diff;

                if (dist_same >= tile_thresh) anom_same++;
                if (dist_diff >= tile_thresh) anom_diff++;
            }
        }

        float n_tiles = (float)(gh * gw);
        if ((float)anom_same / n_tiles >= alert_frac) same_alerts++;
        if ((float)anom_diff / n_tiles >= alert_frac) diff_alerts++;
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

    out_metrics->num_pairs   = num_pairs;
    out_metrics->total_tiles = total_tiles;
    out_metrics->same_mean   = same_sum / (double)total_tiles;
    out_metrics->diff_mean   = diff_sum / (double)total_tiles;
    out_metrics->same_med    = same_dists[total_tiles / 2];
    out_metrics->diff_med    = diff_dists[total_tiles / 2];
    out_metrics->same_p95    = same_dists[(total_tiles * 95) / 100];
    out_metrics->diff_p95    = diff_dists[(total_tiles * 95) / 100];
    out_metrics->same_alerts = same_alerts;
    out_metrics->diff_alerts = diff_alerts;
    out_metrics->roc_auc     = u_sum / ((double)total_tiles * (double)total_tiles);

    free(pair_offsets);
    free(same_dists);
    free(diff_dists);
    return 0;
}

void eval_metrics_print(const EvalMetrics *self, const DetectorConfig *cfg) {
    printf("pgmcd: evaluated %zu pairs (%zu tiles, stride=%d, thresh=%.2f)\n"
           "  matched    (pre_i, post_i): mean=%.4f med=%.4f p95=%.4f alerts=%zu/%zu (%.1f%%)\n"
           "  mismatched (pre_i, post_j): mean=%.4f med=%.4f p95=%.4f alerts=%zu/%zu (%.1f%%)\n"
           "  separation gap:             %+.4f\n"
           "  tile ROC-AUC:               %.4f\n",
           self->num_pairs, self->total_tiles, cfg->stride, cfg->tile_thresh,
           self->same_mean, self->same_med, self->same_p95,
           self->same_alerts, self->num_pairs,
           100.0 * (double)self->same_alerts / (double)self->num_pairs,
           self->diff_mean, self->diff_med, self->diff_p95,
           self->diff_alerts, self->num_pairs,
           100.0 * (double)self->diff_alerts / (double)self->num_pairs,
           self->diff_mean - self->same_mean, self->roc_auc);
}
