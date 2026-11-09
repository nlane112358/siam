#!/usr/bin/env bash
# 05_infer.sh — Run RX-CFAR change detection, render PNG overlays, and clean up PGMs
set -euo pipefail
source "$(dirname "$0")/config.sh"

if [[ ! -s "$PAIRS_FILE" || ! -f "$WEIGHTS" ]]; then
    echo "Error: Need both $PAIRS_FILE and $WEIGHTS. Run ./03_align.sh and ./04_train.sh first." >&2
    exit 2
fi

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/siam-infer.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT INT TERM

NEW_PNGS=()

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

    CHG="$SCRATCH/chg.pgm"
    DRP="$SCRATCH/drp.pgm"
    INC="$SCRATCH/inc.pgm"
    EMB="$SCRATCH/emb.ppm"
    MOT="$SCRATCH/mot.ppm"

    # Exit code 1 = EXIT_ALERT (anomalies detected), 0 = normal, 2 = error
    rc=0
    siam -w "$WEIGHTS" -s "$STRIDE" -p "$FALSE_ALARM" \
        -o "$CHG" -D "$DRP" -I "$INC" -v "$EMB" -m "$MOT" \
        "$PRE_PGM" "$POST_PGM" || rc=$?
    [[ "$rc" -eq 2 ]] && exit 2

    OVERLAY_PNG="$RESULT_DIR/${TAG}_overlay.png"
    CHANGE_PNG="$RESULT_DIR/${TAG}_change.png"
    PCA_PNG="$RESULT_DIR/${TAG}_pca.png"
    FULL_PCA_PNG="$RESULT_DIR/${TAG}_full_scene_pca.png"
    MOTION_PNG="$RESULT_DIR/${TAG}_motion.png"

    sarimg -base "$POST_PGM" -change "$CHG" -drop "$DRP" -inc "$INC" \
           -stride "$STRIDE" -thresh "$OVERLAY_THRESH" -o "$OVERLAY_PNG"
    sarimg -cmap inferno -upsample "$POST_PGM" -stride "$STRIDE" -o "$CHANGE_PNG" "$CHG"
    # PCA colors show relative variation in learned embeddings, not detections.
    sarimg -upsample "$POST_PGM" -stride "$STRIDE" -o "$PCA_PNG" "$EMB"
    sarimg -upsample "$POST_PGM" -stride "$STRIDE" -o "$MOTION_PNG" "$MOT"

    NEW_PNGS+=("$OVERLAY_PNG" "$CHANGE_PNG" "$PCA_PNG")

    if PRE_TIF="$(find_scene_tif "$PRE_STEM")" && POST_TIF="$(find_scene_tif "$POST_STEM")"; then
        if FULL_BBOX="$(full_scene_bbox "$POST_TIF")"; then
            echo "==> Creating full-scene PCA for $TAG at ${FULL_PCA_RES}m/pixel..."
            FULL_DIR="$SCRATCH/full_scene"
            mkdir -p "$FULL_DIR"
            if sarimg -bbox "$FULL_BBOX" -res "$FULL_PCA_RES" -min-cov 0.01 \
                -bits "$PGM_BITS" -to pgm -overwrite -o "$FULL_DIR" \
                "$PRE_TIF" "$POST_TIF"; then
                FULL_PRE_PGM="$FULL_DIR/$PRE_STEM.pgm"
                FULL_POST_PGM="$FULL_DIR/$POST_STEM.pgm"
                FULL_EMB="$SCRATCH/${TAG}_full_scene_embeddings.ppm"
                full_rc=0
                siam -q -w "$WEIGHTS" -s "$STRIDE" -o /dev/null -v "$FULL_EMB" \
                    "$FULL_PRE_PGM" "$FULL_POST_PGM" || full_rc=$?
                if [[ "$full_rc" -eq 2 ]]; then
                    echo "Warning: full-scene PCA inference failed for $TAG; keeping the ROI outputs." >&2
                # Keep the native embedding tile grid; upsampling it softened the full-scene view.
                elif sarimg -o "$FULL_PCA_PNG" "$FULL_EMB"; then
                    NEW_PNGS+=("$FULL_PCA_PNG")
                else
                    echo "Warning: could not render full-scene PCA for $TAG; keeping the ROI outputs." >&2
                fi
            else
                echo "Warning: could not align full-scene TIFFs for $TAG; keeping the ROI outputs." >&2
            fi
        else
            echo "Warning: could not read full-scene bounds for $POST_TIF; keeping the ROI outputs." >&2
        fi
    else
        echo "Skipping full-scene PCA for $TAG: source TIFFs are missing (retain them through inference)." >&2
    fi
done < "$PAIRS_FILE"

echo "==> Opening inference results in $RESULT_DIR..."
echo "    PCA colors show relative embedding variation, not detections."
echo "    Full-scene PCA uses ${FULL_PCA_RES}m input pixels and stride ${STRIDE}; each color represents one tile."
open_files "${NEW_PNGS[@]}"

echo ""
read -r -p "Inference complete. Delete aligned .pgm files ($DATA_DIR/pgm) to free storage? [y/N] " ans
if [[ "$ans" =~ ^[yY] ]]; then
    rm -rf "$DATA_DIR/pgm" "$PAIRS_FILE"
    echo "==> Deleted $DATA_DIR/pgm. Only lightweight PNGs and $WEIGHTS remain on disk."
fi
