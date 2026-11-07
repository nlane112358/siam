# pgmcd

Zero-dependency C11 Siamese Contrastive (Symmetric InfoNCE / Cosine Distance) engine for self-supervised satellite & SAR image change detection on Netpbm PGM (`P5`/`P2`, 8-bit & 16-bit) imagery.

## Build

```sh
cc -O3 -march=native -Wall -Wextra -Wno-unknown-pragmas pgmcd.c -lm -o pgmcd
```

## Usage

### 1. Train (`-t`)
Train across multiple `<pre.pgm> <post.pgm>` pairs via positional arguments, `-L <list.txt>`, or streamed from `stdin` (`-`):

```sh
paste <(find pgm_floods -name '*_pre.pgm' | sort | head -n 300) \
      <(find pgm_floods -name '*_post.pgm' | sort | head -n 300) \
  | ./pgmcd -t -w flood.w -n 10000 -
```

### 2. Evaluate (`-e`)
Compute label-free Matched vs. Mismatched Tile ROC-AUC across held-out image pairs:

```sh
paste <(find pgm_floods -name '*_pre.pgm' | sort | tail -n 103) \
      <(find pgm_floods -name '*_post.pgm' | sort | tail -n 103) \
  | ./pgmcd -e -w flood.w -
```

```text
pgmcd: evaluated 103 pairs (383263 tiles, stride=8, thresh=0.35)
  matched    (pre_i, post_i): mean=0.1039 med=0.0896 p95=0.2348 alerts=0/103 (0.0%)
  mismatched (pre_i, post_j): mean=0.3452 med=0.3433 p95=0.5269 alerts=103/103 (100.0%)
  separation gap:             +0.2413
  tile ROC-AUC:               0.9692
```

### 3. Inference & Alerting
Follows POSIX `diff(1)` / `cmp(1)` exit conventions (`0` = normal, `1` = anomaly alert triggered, `2` = error):

```sh
./pgmcd -w flood.w -a 0.35 -f 0.05 pre.pgm post.pgm > change_map.pgm || notify_alert
```

---

## Sen1Floods11 ($512\times512$) Ground-Truth Flood Comparisons

Trained for 10,000 steps (`flood.w`) without labels on 300 Sentinel-1 SAR pairs (`0.9692` Matched-vs-Mismatched Tile ROC-AUC across 383,263 held-out tiles). Each 4-panel figure shows:
1. **Panel 1:** Pre-Flood Sentinel-1 SAR ($512\times512$)
2. **Panel 2:** Post-Flood Sentinel-1 SAR ($512\times512$)
3. **Panel 3:** Hand-Labeled Ground-Truth Flood Mask (**dark blue** = permanent ocean/river water, **bright cyan** = newly flooded land)
4. **Panel 4:** `pgmcd` $61\times61$ Cosine-Distance Change Map (**red/orange** = detected flood change)

### `Sri-Lanka_916628` (`0.9881` Flood Change ROC-AUC)
![Sri-Lanka_916628](docs/Sri-Lanka_916628_panel4.png)

### `USA_170264`
![USA_170264](docs/USA_170264_panel4.png)

### `USA_1068362`
![USA_1068362](docs/USA_1068362_panel4.png)

### `USA_955053`
![USA_955053](docs/USA_955053_panel4.png)

### `USA_217598`
![USA_217598](docs/USA_217598_panel4.png)

### `USA_595451`
![USA_595451](docs/USA_595451_panel4.png)

### `Sri-Lanka_92824`
![Sri-Lanka_92824](docs/Sri-Lanka_92824_panel4.png)

### `Sri-Lanka_713926`
![Sri-Lanka_713926](docs/Sri-Lanka_713926_panel4.png)

---

## xBD-S12 ($128\times128$) Held-Out Sentinel-1 SAR Alert Examples

Trained for 10,000 steps (`s1_b2.w`, `0.8833` Tile ROC-AUC). Each 3-panel figure shows **Pre-Disaster Sentinel-1 SAR** (left, false-color `VV`/`VH`/`VV−VH`), **Post-Disaster Sentinel-1 SAR** (center), and the **$13\times13$ `pgmcd` Cosine-Distance Change Map** (right, with anomalous tiles $\ge 0.35$ highlighted in red/orange):

### `hurricane-matthew_00000003`
![hurricane-matthew_00000003](docs/hurricane-matthew_00000003_panel.png)

### `hurricane-matthew_00000009`
![hurricane-matthew_00000009](docs/hurricane-matthew_00000009_panel.png)

### `hurricane-matthew_00000026`
![hurricane-matthew_00000026](docs/hurricane-matthew_00000026_panel.png)

### `hurricane-matthew_00000033`
![hurricane-matthew_00000033](docs/hurricane-matthew_00000033_panel.png)

### `hurricane-matthew_00000036`
![hurricane-matthew_00000036](docs/hurricane-matthew_00000036_panel.png)

### `hurricane-matthew_00000040`
![hurricane-matthew_00000040](docs/hurricane-matthew_00000040_panel.png)

### `hurricane-matthew_00000094`
![hurricane-matthew_00000094](docs/hurricane-matthew_00000094_panel.png)

### `hurricane-matthew_00000099`
![hurricane-matthew_00000099](docs/hurricane-matthew_00000099_panel.png)

### `hurricane-matthew_00000304`
![hurricane-matthew_00000304](docs/hurricane-matthew_00000304_panel.png)

### `hurricane-matthew_00000308`
![hurricane-matthew_00000308](docs/hurricane-matthew_00000308_panel.png)

### `hurricane-matthew_00000350`
![hurricane-matthew_00000350](docs/hurricane-matthew_00000350_panel.png)

### `hurricane-matthew_00000356`
![hurricane-matthew_00000356](docs/hurricane-matthew_00000356_panel.png)

### `hurricane-matthew_00000362`
![hurricane-matthew_00000362](docs/hurricane-matthew_00000362_panel.png)

### `hurricane-matthew_00000364`
![hurricane-matthew_00000364](docs/hurricane-matthew_00000364_panel.png)

### `hurricane-matthew_00000365`
![hurricane-matthew_00000365](docs/hurricane-matthew_00000365_panel.png)

### `hurricane-matthew_00000370`
![hurricane-matthew_00000370](docs/hurricane-matthew_00000370_panel.png)
