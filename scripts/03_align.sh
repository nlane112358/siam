#!/usr/bin/env bash
# 03_align.sh — Co-register PRE/POST to shared UTM BBOX, preview PNGs, & evict raw TIFFs
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo "==> Co-registering scenes to BBOX=$BBOX at ${ALIGN_RES}m resolution..."
sarimg \
    -bbox "$BBOX" \
    -res "$ALIGN_RES" \
    -bits "$PGM_BITS" \
    -chip "$CHIP_SIZE" \
    -to pgm,png \
    -pairs \
    -overwrite \
    -o "$DATA_DIR" \
    "$DATA_DIR" > "$PAIRS_FILE"

NUM_PAIRS="$(grep -cve '^\s*$' "$PAIRS_FILE" || true)"
echo "==> Generated $NUM_PAIRS aligned PRE/POST pair(s) in $PAIRS_FILE"

echo "==> Opening aligned AOI PNGs for visual verification..."
open_files "$DATA_DIR"/png/*.png

echo ""
read -r -p "Are the aligned PNGs properly cropped? Delete raw full-swath .tiff files now to save disk space? [y/N] " ans
if [[ "$ans" =~ ^[yY] ]]; then
    rm -f "$DATA_DIR"/*.tiff "$DATA_DIR"/*.tif
    rm -rf "$SCOUT_DIR"
    echo "==> Raw .tiff files and scout previews deleted."
fi
