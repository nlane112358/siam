#!/usr/bin/env bash
# 02_scout.sh — Download scenes and generate lightweight full-swath overview PNGs
set -euo pipefail
source "$(dirname "$0")/config.sh"

ASF_ARGS=(-bbox "$BBOX" -from "$FROM" -to "$TO" -pol "$POL" -limit "$DOWNLOAD_LIMIT" -out "$DATA_DIR")
[[ -n "$DIRECTION" ]] && ASF_ARGS+=(-direction "$DIRECTION")
[[ -n "$ORBIT" ]]     && ASF_ARGS+=(-orbit "$ORBIT")

echo "==> [1/3] Catalog of up to $DOWNLOAD_LIMIT matching Sentinel-1 scene(s) on NASA ASF:"
asf -ls "${ASF_ARGS[@]}"

if [[ -z "$ORBIT" ]]; then
    echo ""
    echo "NOTE: Set ORBIT in config.sh to one of the track numbers above so all scenes share the same look angle."
    read -r -p "Continue downloading all listed tracks anyway? [y/N] " ans
    [[ "$ans" =~ ^[yY] ]] || exit 0
fi

echo "==> [2/3] Downloading at most $DOWNLOAD_LIMIT measurement GeoTIFF(s) to $DATA_DIR..."
asf "${ASF_ARGS[@]}"

echo "==> [3/3] Rendering 100m full-swath north-up scout PNGs..."
shopt -s nullglob
for tif in "$DATA_DIR"/*.tiff "$DATA_DIR"/*.tif; do
    stem="$(basename "${tif%.*}")"
    scout_png="$SCOUT_DIR/${stem}.png"
    [[ -f "$scout_png" ]] && continue

    # Read full scene lon/lat extent from GCPs
    FULL_BBOX="$(python3 -c '
import json, subprocess, sys
info = json.loads(subprocess.check_output(["gdalinfo", "-json", sys.argv[1]]))
gcps = info["gcps"]["gcpList"]
lons, lats = [g["x"] for g in gcps], [g["y"] for g in gcps]
print(f"{min(lons):.4f},{min(lats):.4f},{max(lons):.4f},{max(lats):.4f}")
' "$tif")"

    echo "  -> Warping full scene $stem ($FULL_BBOX at${SCOUT_RES}m)..."
    sarimg -bbox "$FULL_BBOX" -res "$SCOUT_RES" -min-cov 0.01 -to png -o "$scout_png" "$tif" >/dev/null
done

# Open full-swath PNGs and refresh the optical + SAR map overlay
open_files "$SCOUT_DIR"/*.png
"$(dirname "$0")/01_locate.sh"
