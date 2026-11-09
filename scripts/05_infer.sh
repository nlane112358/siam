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
    MOTION_PNG="$RESULT_DIR/${TAG}_motion.png"

    sarimg -base "$POST_PGM" -change "$CHG" -drop "$DRP" -inc "$INC" \
           -stride "$STRIDE" -thresh "$OVERLAY_THRESH" -o "$OVERLAY_PNG"
    sarimg -cmap inferno -upsample "$POST_PGM" -stride "$STRIDE" -o "$CHANGE_PNG" "$CHG"
    sarimg -upsample "$POST_PGM" -stride "$STRIDE" -o "$PCA_PNG" "$EMB"
    sarimg -upsample "$POST_PGM" -stride "$STRIDE" -o "$MOTION_PNG" "$MOT"

    NEW_PNGS+=("$OVERLAY_PNG" "$CHANGE_PNG")
done < "$PAIRS_FILE"

echo "==> Opening inference results in $RESULT_DIR..."
open_files "${NEW_PNGS[@]}"

echo ""
read -r -p "Inference complete. Delete aligned .pgm files ($DATA_DIR/pgm) to free storage? [y/N] " ans
if [[ "$ans" =~ ^[yY] ]]; then
    rm -rf "$DATA_DIR/pgm" "$PAIRS_FILE"
    echo "==> Deleted $DATA_DIR/pgm. Only lightweight PNGs and $WEIGHTS remain on disk."
fi
