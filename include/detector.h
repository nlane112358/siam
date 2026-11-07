/*
 * detector.h
 * Change Detector Object (Sliding-Window Inference & Label-Free ROC-AUC Evaluation)
 */

#ifndef PGMCD_DETECTOR_H
#define PGMCD_DETECTOR_H

#include <stddef.h>
#include "dataset.h"
#include "image.h"
#include "network.h"

typedef struct {
    int   stride;
    float tile_thresh;
    float alert_frac;
    int   quiet;
} DetectorConfig;

typedef struct {
    PGMImage change_map;
    int      anomalous_tiles;
    int      total_tiles;
    float    fraction;
    int      is_alert;
} DetectionResult;

typedef struct {
    size_t num_pairs;
    size_t total_tiles;
    double same_mean;
    float  same_med;
    float  same_p95;
    size_t same_alerts;
    double diff_mean;
    float  diff_med;
    float  diff_p95;
    size_t diff_alerts;
    double roc_auc;
} EvalMetrics;

typedef struct {
    const SiameseNet *net;
    DetectorConfig    cfg;
} ChangeDetector;

void change_detector_init(ChangeDetector *self, const SiameseNet *net, DetectorConfig cfg);

int  change_detector_infer(const ChangeDetector *self, const ImagePair *pair,
                           DetectionResult *out_result);
void detection_result_destroy(DetectionResult *self);

int  change_detector_evaluate(const ChangeDetector *self, const PairDataset *dataset,
                              EvalMetrics *out_metrics);
void eval_metrics_print(const EvalMetrics *self, const DetectorConfig *cfg);

#endif /* PGMCD_DETECTOR_H */
