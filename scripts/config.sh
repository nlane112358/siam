#!/usr/bin/env bash
# config.sh — Shared configuration for the siam workflow

# 1. Geographic Area & ASF Search Parameters
#BBOX="-3.25,55.90,-3.10,55.98"   # west,south,east,north (Firth of Forth example)
#BBOX="-1.8979,53.5223,-1.5566,53.6939"
BBOX="-2.7356,53.7390,-0.4120,54.5466"
FROM="2026-01-01"
TO="2026-04-01"
DIRECTION="ASCENDING"            # ASCENDING or DESCENDING (keep consistent!)
ORBIT="132"                         # Leave empty on first scout; set to lock track
POL="vv"                         # vv, vh, or vv,vh
DOWNLOAD_LIMIT=8                  # Maximum number of scenes returned for download

# 2. Resolution & Alignment
SCOUT_RES=100                    # Coarse meters/pixel to inspect full 250km swath (~2 MB PNG)
ALIGN_RES=10                     # Fine meters/pixel for AOI co-registration
FULL_PCA_RES="${FULL_PCA_RES:-$ALIGN_RES}" # Full-scene PCA resolution; defaults to aligned scene resolution
MIN_COV="0.80"                   # Allow scenes that cover at least 80% of the BBOX
PGM_BITS=8                       # 8-bit halves PGM size after sarimg's p2-p98 stretch
CHIP_SIZE=0                      # 0 = full BBOX; or e.g. 1024 for NxN chips

# 3. Siam Training & Inference
STEPS=10000
LR=0.01
STRIDE=8
FALSE_ALARM="1e-6"
OVERLAY_THRESH="0.15"

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
WEIGHTS="$DATA_DIR/siam.w"

# Make binaries built by `make` available to the workflow scripts.
export PATH="$ROOT_DIR/bin:$PATH"

mkdir -p "$DATA_DIR" "$SCOUT_DIR" "$RESULT_DIR"

open_files() {
    if command -v open >/dev/null 2>&1; then open "$@"
    elif command -v xdg-open >/dev/null 2>&1; then for f in "$@"; do xdg-open "$f" & done
    fi
}
