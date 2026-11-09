#!/usr/bin/env bash
set -euo pipefail

# Define regional UK bounding boxes
REGIONS=(
  "England_South:-5.5,49.9,1.8,53.5"
  "England_North_Scotland_South:-6.5,53.5,0.5,56.5"
  "Scotland_North:-8.6,56.5,-1.0,58.8"
  "Northern_Ireland:-8.2,54.0,-5.4,55.5"
)

FROM_DATE="2025-05-01"
TO_DATE="2025-10-01"
ORBIT="35" # Pick a dominant Sentinel-1 descending track covering the UK

for region in "${REGIONS[@]}"; do
    NAME="${region%%:*}"
    BBOX="${region#*:}"

    echo "=== Processing Region: $NAME ($BBOX) ==="

    mkdir -p "data_$NAME/tif" "data_$NAME/pgm" "data_$NAME/png"

    # 1. Fetch from ASF -> pipe raw TIFF paths to sarimg
    ./asf -bbox="$BBOX" -from="$FROM_DATE" -to="$TO_DATE" \
          -orbit="$ORBIT" -direction="DESCENDING" -pol="vv" \
          -out="data_$NAME/tif" \
      | ./sarimg -bbox="$BBOX" -res=10 -out="data_$NAME" \
                 -to=pgm,png -chip=1024 -min-cov=0.85

    # 2. Generate training/evaluation pairs list for siam
    ./sarimg -pairs -out="data_$NAME" data_$NAME/pgm/*.pgm > "data_$NAME/pairs.txt"

    echo "Region $NAME ready. Training siam..."

    # 3. Train siam on this region's historical sequence
    ./siam -t -w "siam_$NAME.w" -n 300 -L "data_$NAME/pairs.txt"

    echo "Region $NAME complete."
done
