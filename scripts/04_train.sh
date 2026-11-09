#!/usr/bin/env bash
# 04_train.sh — Train Siamese Barlow Twins encoder and evaluate ROC-AUC
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [[ ! -s "$PAIRS_FILE" ]]; then
    echo "Error: $PAIRS_FILE is missing or empty. Run ./03_align.sh first." >&2
    exit 2
fi

TRAIN_FLAGS=(-t -w "$WEIGHTS" -n "$STEPS" -l "$LR" -L "$PAIRS_FILE")
[[ -f "$WEIGHTS" ]] && TRAIN_FLAGS+=(-c)

echo "==> Training Siamese encoder ($STEPS steps, lr=$LR)..."
siam "${TRAIN_FLAGS[@]}"

NUM_PAIRS="$(grep -cve '^\s*$' "$PAIRS_FILE" || true)"
if [[ "$NUM_PAIRS" -ge 2 ]]; then
    echo "==> Evaluating matched vs. mismatched pairs..."
    siam -e -w "$WEIGHTS" -s "$STRIDE" -L "$PAIRS_FILE"
fi
