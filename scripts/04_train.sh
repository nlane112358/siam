#!/usr/bin/env bash
# 04_train.sh — Train Siamese contrastive encoder, evaluate ROC-AUC, & calibrate ROI threshold
set -euo pipefail
source "$(dirname "$0")/config.sh"

ACTUAL_TRAIN_FILE="${TRAIN_PAIRS_FILE}"
if [[ ! -s "$ACTUAL_TRAIN_FILE" ]]; then
    ACTUAL_TRAIN_FILE="$PAIRS_FILE"
fi

if [[ ! -s "$ACTUAL_TRAIN_FILE" ]]; then
    echo "Error: training pairs file ($ACTUAL_TRAIN_FILE) is missing or empty. Run ./03_align.sh first." >&2
    exit 2
fi

TRAIN_FLAGS=(-t -w "$WEIGHTS" -n "$STEPS" -l "$LR" -T "$STABILITY_THRESH" -L "$ACTUAL_TRAIN_FILE")
[[ -f "$WEIGHTS" ]] && TRAIN_FLAGS+=(-c)

echo "==> [1/3] Training Siamese contrastive encoder on valid stable patches ($STEPS steps, lr=$LR, stability_thresh=$STABILITY_THRESH)..."
siam "${TRAIN_FLAGS[@]}"

ACTUAL_EVAL_FILE="${EVAL_PAIRS_FILE}"
if [[ ! -s "$ACTUAL_EVAL_FILE" ]]; then
    ACTUAL_EVAL_FILE="$PAIRS_FILE"
fi

NUM_EVAL="$(grep -cve '^\s*$' "$ACTUAL_EVAL_FILE" || true)"
if [[ "$NUM_EVAL" -ge 2 ]]; then
    echo ""
    echo "==> [2/3] Evaluating matched vs. mismatched pairs on holdout tiles ($ACTUAL_EVAL_FILE)..."
    siam -e -w "$WEIGHTS" -s "$STRIDE" -a "$TILE_THRESH" -f "$ALERT_FRAC" -L "$ACTUAL_EVAL_FILE"
fi

ACTUAL_CALIB_FILE="${CALIB_PAIRS_FILE}"
if [[ ! -s "$ACTUAL_CALIB_FILE" ]]; then
    ACTUAL_CALIB_FILE="$PAIRS_FILE"
fi

if [[ -s "$ACTUAL_CALIB_FILE" ]]; then
    echo ""
    echo "==> [3/3] Calibrating target ROI change threshold on baseline history (P_fa=$CALIB_P_FA)..."
    siam -C -w "$WEIGHTS" -s "$STRIDE" -f "$CALIB_P_FA" -o "$CALIB_THRESH_FILE" -L "$ACTUAL_CALIB_FILE"
    if [[ -f "$CALIB_THRESH_FILE" ]]; then
        CAL_VAL="$(cat "$CALIB_THRESH_FILE" | tr -d '[:space:]')"
        echo "==> Stored calibrated threshold: TILE_THRESH=$CAL_VAL in $CALIB_THRESH_FILE"
    fi
fi
