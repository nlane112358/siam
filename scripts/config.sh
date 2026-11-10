#!/usr/bin/env bash
# config.sh — Shared configuration for the siam workflow

# 1. Geographic Area & ASF Search Parameters
# Optional seed coordinate (lon, lat) to auto-discover scenes and allocate maximum aligned BBOX:
# e.g., SEED_POINT="31.2357,30.0444"
SEED_POINT="${SEED_POINT:-}"

# Target ROI for local calibration & evaluation:
#ROI_BBOX="-22.65,63.80,-22.10,64.05"
#ROI_BBOX="2.5296,50.9653,5.7101,52.2614"
ROI_BBOX="28.3887,21.3610,33.9148,32.0826"
BBOX="$ROI_BBOX"

# Broader regional training footprint (can be larger than ROI for geographical diversity):
TRAIN_BBOX="${TRAIN_BBOX:-$ROI_BBOX}"
#TRAIN_BBOX="-22.65,63.80,-22.10,64.05"

FROM="2023-05-01"
TO="2024-05-01"
DIRECTION="ASCENDING"            # ASCENDING or DESCENDING (keep consistent!)
ORBIT=""                         # Leave empty on first scout; set to lock track
POL="vv"                         # vv, vh, or vv,vh
DOWNLOAD_LIMIT=15                  # Maximum number of scenes returned for download
ASF_WORKERS=6                      # Concurrent ASF scene downloads (tune up/down based on bandwidth)

# 2. Resolution & Alignment
SCOUT_RES=100                    # Coarse meters/pixel to inspect full 250km swath (~2 MB PNG)
ALIGN_RES=10                     # Fine meters/pixel for AOI co-registration
FULL_PCA_RES="${FULL_PCA_RES:-$ALIGN_RES}" # Full-scene PCA resolution; defaults to aligned scene resolution
MIN_COV="0.70"                   # Allow scenes that cover at least 70% of the BBOX
PGM_BITS=8                       # 8-bit halves PGM size after sarimg's p2-p98 stretch
CHIP_SIZE=256                    # Moderate tiles (256-512 px square) as storage units

# 3. Siam Training & Inference
STEPS=10000
LR=0.01
STRIDE=8
STABILITY_THRESH="0.20"          # Temporal backscatter stability threshold for positive patches
CALIB_P_FA="0.005"               # Target false alarm probability P_fa for ROI calibration (0.5%)
TILE_THRESH="0.35"               # Default tile threshold (overridden if calib_thresh.txt exists)
ALERT_FRAC="0.05"
CHANGE_MAP_THRESH="${CHANGE_MAP_THRESH:-$TILE_THRESH}"

# 4. Directories & Files
ROOT_DIR="/Users/nlane/Development/siam"

# Load local Earthdata credentials. Keep this file out of version control.
if [[ -f "$ROOT_DIR/.env" ]]; then
    _siam_env_token_is_set=${EARTHDATA_TOKEN+x}
    _siam_env_token=${EARTHDATA_TOKEN-}
    _siam_env_username_is_set=${EARTHDATA_USERNAME+x}
    _siam_env_username=${EARTHDATA_USERNAME-}
    _siam_env_password_is_set=${EARTHDATA_PASSWORD+x}
    _siam_env_password=${EARTHDATA_PASSWORD-}
    case $- in
        *a*) _config_allexport_was_set=1 ;;
        *) _config_allexport_was_set=0; set -a ;;
    esac
    source "$ROOT_DIR/.env"
    if [[ "$_config_allexport_was_set" -eq 0 ]]; then set +a; fi
    if [[ "$_siam_env_token_is_set" == x ]]; then export EARTHDATA_TOKEN="$_siam_env_token"; fi
    if [[ "$_siam_env_username_is_set" == x ]]; then export EARTHDATA_USERNAME="$_siam_env_username"; fi
    if [[ "$_siam_env_password_is_set" == x ]]; then export EARTHDATA_PASSWORD="$_siam_env_password"; fi
    unset _config_allexport_was_set _siam_env_token_is_set _siam_env_token
    unset _siam_env_username_is_set _siam_env_username _siam_env_password_is_set _siam_env_password
fi

DATA_DIR="$ROOT_DIR/data"
SCOUT_DIR="$DATA_DIR/scout"
RESULT_DIR="$DATA_DIR/results"
PAIRS_FILE="$DATA_DIR/pairs.txt"
TRAIN_PAIRS_FILE="$DATA_DIR/train_pairs.txt"
EVAL_PAIRS_FILE="$DATA_DIR/eval_pairs.txt"
CALIB_PAIRS_FILE="$DATA_DIR/calib_pairs.txt"
CALIB_THRESH_FILE="$DATA_DIR/calib_thresh.txt"
WEIGHTS="$DATA_DIR/siam.w"

# Make binaries built by `make` available to the workflow scripts.
export PATH="$ROOT_DIR/bin:$PATH"

mkdir -p "$DATA_DIR" "$SCOUT_DIR" "$RESULT_DIR"

open_files() {
    if command -v open >/dev/null 2>&1; then open "$@"
    elif command -v xdg-open >/dev/null 2>&1; then for f in "$@"; do xdg-open "$f" & done
    fi
}
