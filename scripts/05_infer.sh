#!/usr/bin/env bash
# 05_infer.sh — Run full-scene change detection, render maps, and clean up PGMs
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [[ ! -s "$PAIRS_FILE" || ! -f "$WEIGHTS" ]]; then
    echo "Error: Need both $PAIRS_FILE and $WEIGHTS. Run ./03_align.sh and ./04_train.sh first." >&2
    exit 2
fi

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/siam-infer.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT INT TERM

NEW_PNGS=()

if [[ -f "$CALIB_THRESH_FILE" ]]; then
    CALIB_VAL="$(cat "$CALIB_THRESH_FILE" | tr -d '[:space:]')"
    if [[ -n "$CALIB_VAL" ]]; then
        CHANGE_MAP_THRESH="$CALIB_VAL"
        TILE_THRESH="$CALIB_VAL"
        echo "==> Using ROI-calibrated change threshold: TILE_THRESH=$TILE_THRESH (from $CALIB_THRESH_FILE)"
    fi
fi

find_scene_tif() {
    local stem="$1"
    for ext in tiff tif; do
        local path="$DATA_DIR/$stem.$ext"
        if [[ -s "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

full_scene_bbox() {
    python3 - "$1" <<'PY'
import json
import subprocess
import sys

info = json.loads(subprocess.check_output(
    ["gdalinfo", "-json", sys.argv[1]], stderr=subprocess.DEVNULL
))
gcps = info.get("gcps", {}).get("gcpList", [])
if not gcps:
    raise SystemExit("no ground control points")
lons = [point["x"] for point in gcps]
lats = [point["y"] for point in gcps]
print(f"{min(lons):.8f},{min(lats):.8f},{max(lons):.8f},{max(lats):.8f}")
PY
}

while read -r PRE_PGM POST_PGM; do
    [[ -z "${PRE_PGM:-}" ]] && continue
    PRE_STEM="$(basename "${PRE_PGM%.pgm}")"
    POST_STEM="$(basename "${POST_PGM%.pgm}")"
    TAG="${PRE_STEM}__to__${POST_STEM}"

    CHANGE_PNG="$RESULT_DIR/${TAG}_change.png"
    PCA_PNG="$RESULT_DIR/${TAG}_pca.png"
    rm -f "$CHANGE_PNG" "$PCA_PNG" \
          "$RESULT_DIR/${TAG}_overlay.png" \
          "$RESULT_DIR/${TAG}_full_scene_change.png" \
          "$RESULT_DIR/${TAG}_full_scene_pca.png"

    if ! PRE_TIF="$(find_scene_tif "$PRE_STEM")" || ! POST_TIF="$(find_scene_tif "$POST_STEM")"; then
        echo "Skipping $TAG: full-scene inference needs both source TIFFs." >&2
        continue
    fi
    if ! FULL_BBOX="$(full_scene_bbox "$POST_TIF")"; then
        echo "Warning: could not read full-scene bounds for $POST_TIF; skipping $TAG." >&2
        continue
    fi

    echo "==> Creating full-scene change map and PCA for $TAG at ${FULL_PCA_RES}m/pixel..."
    FULL_DIR="$SCRATCH/full_scene"
    mkdir -p "$FULL_DIR"
    if ! sarimg -bbox "$FULL_BBOX" -res "$FULL_PCA_RES" -min-cov 0.01 \
        -bits "$PGM_BITS" -to pgm -overwrite -o "$FULL_DIR" \
        "$PRE_TIF" "$POST_TIF"; then
        echo "Warning: could not align full-scene TIFFs for $TAG; skipping." >&2
        continue
    fi

    FULL_PRE_PGM="$FULL_DIR/$PRE_STEM.pgm"
    FULL_POST_PGM="$FULL_DIR/$POST_STEM.pgm"
    FULL_CHG="$SCRATCH/${TAG}_change.pgm"
    FULL_EMB="$SCRATCH/${TAG}_embeddings.ppm"
    full_rc=0
    siam -q -w "$WEIGHTS" -s "$STRIDE" -o "$FULL_CHG" -v "$FULL_EMB" \
        "$FULL_PRE_PGM" "$FULL_POST_PGM" || full_rc=$?
    if [[ "$full_rc" -eq 2 ]]; then
        echo "Warning: full-scene inference failed for $TAG; skipping." >&2
        continue
    fi

    if sarimg -mask "$FULL_POST_PGM" -upsample "$FULL_POST_PGM" -stride "$STRIDE" \
        -thresh "$CHANGE_MAP_THRESH" -cmap gray -overwrite -o "$CHANGE_PNG" "$FULL_CHG"; then
        NEW_PNGS+=("$CHANGE_PNG")
    else
        echo "Warning: could not render standalone full-scene change map for $TAG." >&2
    fi

    # Keep the native embedding tile grid, and mask no-data from the POST scene.
    if sarimg -mask "$FULL_POST_PGM" -stride "$STRIDE" -overwrite \
        -o "$PCA_PNG" "$FULL_EMB"; then
        NEW_PNGS+=("$PCA_PNG")
    else
        echo "Warning: could not render full-scene PCA for $TAG." >&2
    fi
done < "$PAIRS_FILE"

echo "==> Opening inference results in $RESULT_DIR..."
echo "    PCA colors show relative embedding variation, not detections."
echo "    Change maps show standalone grayscale anomaly intensity on black."
echo "    Change and PCA maps cover the full SAR scenes; no-data areas remain black."
echo "    PCA uses ${FULL_PCA_RES}m input pixels and stride ${STRIDE}; each color represents one tile."
if [[ "${OPEN_RESULTS:-1}" == "1" ]]; then
    open_files "${NEW_PNGS[@]}"
fi

echo ""
read -r -p "Inference complete. Delete aligned PGM files in $DATA_DIR/pgm to free storage? [y/N] " ans
if [[ "$ans" =~ ^[yY] ]]; then
    rm -rf "$DATA_DIR/pgm" "$PAIRS_FILE"
    echo "==> Deleted $DATA_DIR/pgm. Only lightweight PNGs and $WEIGHTS remain on disk."
fi
