#!/usr/bin/env bash
# siam-pipe.sh — Low-footprint SAR co-registration, visualization, training & inference
set -euo pipefail

# --- Configuration ---
BBOX="${BBOX:--3.25,55.90,-3.10,55.98}" # west,south,east,north
RES="${RES:-10}"                         # meters per pixel in shared UTM grid
BITS="${BITS:-8}"                        # 8-bit PGM halves scratch usage after p2-p98 stretch
STRIDE="${STRIDE:-8}"                    # sliding window stride
STEPS="${STEPS:-200}"                    # Barlow Twins training steps
OUT_DIR="${OUT_DIR:-./artifacts}"        # persistent output (PNGs + weights only)
WEIGHTS="${WEIGHTS:-$OUT_DIR/siam.w}"
PURGE_TIFS="${PURGE_TIFS:-0}"            # set to 1 to delete raw TIFFs right after warping
VIEW="${VIEW:-1}"                        # set to 1 to open PNGs for visual inspection

mkdir -p "$OUT_DIR/aligned_png" "$OUT_DIR/results"

# Ephemeral scratch workspace for PGMs & pair lists (auto-deleted on exit)
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/siam.XXXXXX")"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT INT TERM

view_images() {
    [[ "$VIEW" -eq 1 ]] || return 0
    if command -v open >/dev/null 2>&1; then
        open "$@"                        # macOS Preview
    elif command -v feh >/dev/null 2>&1; then
        feh "$@" &                       # Linux X11
    elif command -v xdg-open >/dev/null 2>&1; then
        for f in "$@"; do xdg-open "$f" >/dev/null 2>&1 & done
    fi
}

if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 <pre.tif|dir> [post.tif ...]" >&2
    exit 2
fi

# --- 1. Geographically Align (UTM Warp + Crop) & Emit PNGs + Ephemeral PGMs ---
echo "==> [1/4] Co-registering to shared UTM grid (bbox=$BBOX, res=${RES}m)..." >&2
PAIRS_FILE="$SCRATCH/pairs.txt"

sarimg \
    -bbox "$BBOX" \
    -res "$RES" \
    -bits "$BITS" \
    -to pgm,png \
    -pairs \
    -o "$SCRATCH" \
    "$@" > "$PAIRS_FILE"

# Persist only the lightweight aligned PNGs; leave PGMs in $SCRATCH
cp -f "$SCRATCH"/png/*.png "$OUT_DIR/aligned_png/"

if [[ "$PURGE_TIFS" -eq 1 ]]; then
    echo "==> Evicting raw full-swath GeoTIFFs..." >&2
    for arg in "$@"; do
        [[ -f "$arg" && "$arg" =~ \.tiff?$ ]] && rm -f "$arg"
    done
fi

NUM_PAIRS="$(wc -l < "$PAIRS_FILE" | tr -d ' ')"
if [[ "$NUM_PAIRS" -lt 1 ]]; then
    echo "Error: No valid PRE/POST pairs produced." >&2
    exit 2
fi

# --- 2. Visualise Aligned PRE/POST PNGs ---
echo "==> [2/4] Opening aligned PNGs in $OUT_DIR/aligned_png/..." >&2
view_images "$OUT_DIR"/aligned_png/*.png

if [[ -t 0 && "$VIEW" -eq 1 ]]; then
    read -r -p "Inspect aligned PNGs. Press [Enter] to train & infer (or Ctrl-C to abort)..."
fi

# --- 3. Train (and Evaluate if >= 2 pairs) from Ephemeral mmap'd PGMs ---
echo "==> [3/4] Training Siamese encoder ($STEPS steps on $NUM_PAIRS pair(s))..." >&2
TRAIN_FLAGS=(-t -w "$WEIGHTS" -n "$STEPS" -L "$PAIRS_FILE")
[[ -f "$WEIGHTS" ]] && TRAIN_FLAGS+=(-c) # continue from existing weights if present
siam "${TRAIN_FLAGS[@]}"

# detector.hpp requires >= 2 pairs for matched/mismatched ROC-AUC evaluation
if [[ "$NUM_PAIRS" -ge 2 ]]; then
    echo "==> Evaluating ROC-AUC across $NUM_PAIRS pairs..." >&2
    siam -e -w "$WEIGHTS" -s "$STRIDE" -L "$PAIRS_FILE"
fi

# --- 4. Inference + Compact Directional Overlay Rendering ---
echo "==> [4/4] Running RX-CFAR change & directional inference..." >&2
while read -r PRE_PGM POST_PGM; do
    PRE_STEM="$(basename "${PRE_PGM%.pgm}")"
    POST_STEM="$(basename "${POST_PGM%.pgm}")"
    PAIR_TAG="${PRE_STEM}__to__${POST_STEM}"

    CHG_PGM="$SCRATCH/${PAIR_TAG}_chg.pgm"
    DRP_PGM="$SCRATCH/${PAIR_TAG}_drop.pgm"
    INC_PGM="$SCRATCH/${PAIR_TAG}_inc.pgm"
    EMB_PPM="$SCRATCH/${PAIR_TAG}_emb.ppm"
    MOT_PPM="$SCRATCH/${PAIR_TAG}_mot.ppm"

    # Capture exit code without tripping 'set -e' on EXIT_ALERT (1)
    rc=0
    siam -w "$WEIGHTS" -s "$STRIDE" \
        -o "$CHG_PGM" \
        -D "$DRP_PGM" \
        -I "$INC_PGM" \
        -v "$EMB_PPM" \
        -m "$MOT_PPM" \
        "$PRE_PGM" "$POST_PGM" || rc=$?

    if [[ "$rc" -eq 2 ]]; then
        echo "Error: siam failed on $PAIR_TAG" >&2
        exit 2
    fi

    # Render overlays (sarimg upsamples tile grids in memory to match POST_PGM)
    OVERLAY_PNG="$OUT_DIR/results/${PAIR_TAG}_overlay.png"
    CHANGE_PNG="$OUT_DIR/results/${PAIR_TAG}_change.png"
    EMB_PNG="$OUT_DIR/results/${PAIR_TAG}_pca.png"
    MOT_PNG="$OUT_DIR/results/${PAIR_TAG}_motion.png"

    sarimg -base "$POST_PGM" -change "$CHG_PGM" -drop "$DRP_PGM" -inc "$INC_PGM" -thresh 0.15 -o "$OVERLAY_PNG"
    sarimg -cmap inferno -upsample "$POST_PGM" -stride "$STRIDE" -o "$CHANGE_PNG" "$CHG_PGM"
    sarimg -upsample "$POST_PGM" -stride "$STRIDE" -o "$EMB_PNG" "$EMB_PPM"
    sarimg -upsample "$POST_PGM" -stride "$STRIDE" -o "$MOT_PNG" "$MOT_PPM"

    view_images "$OVERLAY_PNG" "$CHANGE_PNG"
done < "$PAIRS_FILE"

echo "==> Done. Persistent outputs saved to $OUT_DIR (scratch PGMs purged)." >&2
