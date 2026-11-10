#!/usr/bin/env bash
# 03_align.sh — Co-register regional scenes, slice into tiles with valid masks, partition train/eval/calib
set -euo pipefail
source "$(dirname "$0")/config.sh"

echo "==> Co-registering scenes to TRAIN_BBOX=$TRAIN_BBOX at ${ALIGN_RES}m (chip=${CHIP_SIZE}px, min-cov=$MIN_COV)..."
sarimg \
    -bbox "$TRAIN_BBOX" \
    -res "$ALIGN_RES" \
    -min-cov "$MIN_COV" \
    -bits "$PGM_BITS" \
    -chip "$CHIP_SIZE" \
    -emit-mask \
    -to pgm,png \
    -pairs \
    -overwrite \
    -o "$DATA_DIR" \
    "$DATA_DIR" > "$PAIRS_FILE"

NUM_PAIRS="$(grep -cve '^\s*$' "$PAIRS_FILE" || true)"
echo "==> Generated $NUM_PAIRS aligned tile pair(s) in $PAIRS_FILE"

# Partition tiles into spatial train set (80%), spatial eval set (20%), and baseline calib set
python3 - "$PAIRS_FILE" "$TRAIN_PAIRS_FILE" "$EVAL_PAIRS_FILE" "$CALIB_PAIRS_FILE" <<'PY'
import sys, re, random

pairs_file, train_out, eval_out, calib_out = sys.argv[1:5]
with open(pairs_file) as f:
    lines = [line.strip() for line in f if line.strip()]

tile_groups = {}
for line in lines:
    parts = line.split()
    if len(parts) < 2:
        continue
    m = re.search(r'(_r\d+_c\d+)', parts[0])
    tile_id = m.group(1) if m else "full"
    tile_groups.setdefault(tile_id, []).append(line)

all_tiles = sorted(tile_groups.keys())
train_lines, eval_lines, calib_lines = [], [], []

if len(all_tiles) >= 4:
    random.seed(42)
    shuffled = list(all_tiles)
    random.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * 0.20))
    eval_tiles = set(shuffled[:n_eval])
    train_tiles = set(shuffled[n_eval:])
else:
    train_tiles = set(all_tiles)
    eval_tiles = set(all_tiles)

for t in train_tiles:
    train_lines.extend(tile_groups[t])
for t in eval_tiles:
    eval_lines.extend(tile_groups[t])

# Use early / baseline pairs for local calibration
for t, group in tile_groups.items():
    if group:
        calib_lines.append(group[0])

with open(train_out, 'w') as f:
    for l in train_lines:
        f.write(l + '\n')
with open(eval_out, 'w') as f:
    for l in eval_lines:
        f.write(l + '\n')
with open(calib_out, 'w') as f:
    for l in calib_lines:
        f.write(l + '\n')

print(f"==> Partitioned {len(all_tiles)} tile locations:")
print(f"    Train: {len(train_lines)} pairs across {len(train_tiles)} tiles -> {train_out}")
print(f"    Eval:  {len(eval_lines)} pairs across {len(eval_tiles)} tiles -> {eval_out}")
print(f"    Calib: {len(calib_lines)} baseline pairs -> {calib_out}")
PY

echo ""
echo "==> Opening aligned PNG previews for verification..."
open_files "$DATA_DIR"/png/*.png

echo ""
read -r -p "Are the aligned PNGs properly cropped? Delete raw full-swath .tiff files now? Full-scene change and PCA maps in 05_infer.sh need them. [y/N] " ans
if [[ "$ans" =~ ^[yY] ]]; then
    rm -f "$DATA_DIR"/*.tiff "$DATA_DIR"/*.tif
    rm -rf "$SCOUT_DIR"
    echo "==> Raw .tiff files and scout previews deleted."
fi
