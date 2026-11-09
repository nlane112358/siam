# siam

Zero-dependency C++17 Siamese Contrastive (Symmetric InfoNCE / Cosine Distance) engine for self-supervised satellite & SAR image change detection on Netpbm PGM (`P5`/`P2`, 8-bit & 16-bit) imagery.

## Build

```sh
make          # single-threaded
make OMP=1    # multi-threaded (OpenMP)
```

Or compile directly:

```sh
g++ -std=c++17 -O3 -march=native -Wall -Wextra -Wno-unknown-pragmas main.cpp -lm -o siam
```

## Usage

The command-line reference is also available with `siam -h`:

```text
NAME
    siam - train and run a Siamese image-change detector

SYNOPSIS
    siam [options] PRE POST
    siam -t [options] [PRE POST]...
    siam -e [options] [PRE POST]...
    siam -h

DESCRIPTION
    PRE and POST are P2 or P5 PGM images (8-bit or 16-bit) with matching
    dimensions of at least 32x32 pixels. Training and evaluation accept
    image pairs as operands, from a pair list (-L FILE), or from standard
    input (-). Each pair-list entry is PRE POST.

OPTIONS
    -h              display help and exit
    -t              train
    -e              evaluate matched and mismatched pairs (at least 2 pairs)
    -c              continue training from saved weights
    -q              suppress training progress and inference status
    -u              upsample the inference map to input dimensions
    -w FILE         weights file [siam.w]
    -L FILE         read image pairs from FILE; '-' means standard input
    -o FILE         write the inference map to FILE; '-' means standard output
    -v FILE         write POST tile embeddings as a PCA false-color P6 PPM; '-' means stdout
    -D FILE         write anomalous backscatter-drop tiles to FILE
    -I FILE         write anomalous backscatter-increase tiles to FILE
    -n STEPS        training steps [200]
    -l RATE         learning rate [0.01]
    -S SEED         random seed [time and process ID]
    -a VALUE        tile distance threshold [0.35]
    -f VALUE        alert fraction threshold [0.05]
    -s PIXELS       sliding-window stride [8]

EXIT STATUS
    0   normal result or successful training/evaluation
    1   anomaly alert
    2   invalid arguments or I/O/runtime error

EXAMPLES
    siam -w siam.w -o change.pgm before.pgm after.pgm
    siam -w siam.w -o change.pgm -v embedding.ppm before.pgm after.pgm
    siam -w siam.w -o change.pgm -D drop.pgm -I increase.pgm before.pgm after.pgm
    siam -t -w siam.w -n 200 -L train_pairs.txt
    siam -e -w siam.w -L heldout_pairs.txt
```

### 1. Train (`-t`)
Train across multiple `<pre.pgm> <post.pgm>` pairs via positional arguments, `-L <list.txt>`, or streamed from `stdin` (`-`):

```sh
paste <(find pgm_floods -name '*_pre.pgm' | sort | head -n 300) \
      <(find pgm_floods -name '*_post.pgm' | sort | head -n 300) \
  | ./siam -t -w flood.w -n 100000 -
```

### 2. Evaluate (`-e`)
Compute label-free Matched vs. Mismatched Tile ROC-AUC across held-out image pairs:

```sh
paste <(find pgm_floods -name '*_pre.pgm' | sort | tail -n 103) \
      <(find pgm_floods -name '*_post.pgm' | sort | tail -n 103) \
  | ./siam -e -w flood.w -
```

```text
siam: evaluated 103 pairs (383263 tiles, stride=8, thresh=0.35)
  matched    (pre_i, post_i): mean=0.1268 med=0.1156 p95=0.2554 alerts=1/103 (1.0%)
  mismatched (pre_i, post_j): mean=0.4045 med=0.4048 p95=0.5860 alerts=103/103 (100.0%)
  separation gap:             +0.2777
  tile ROC-AUC:               0.9811
```

### 3. Inference & Alerting
Follows POSIX `diff(1)` / `cmp(1)` exit conventions (`0` = normal, `1` = anomaly alert triggered, `2` = error):

```sh
./siam -w flood.w -a 0.35 -f 0.05 pre.pgm post.pgm > change_map.pgm || notify_alert
```

To split anomalous tiles by backscatter direction, request both directional maps:

```sh
./siam -w flood.w -s 16 -a 0.35 \
  -o change_map.pgm -D backscatter_drop.pgm -I backscatter_increase.pgm \
  pre.pgm post.pgm
```

Each 32×32 window is assigned to the drop or increase map when its cosine distance
meets `-a` and its mean `POST−PRE` pixel difference is negative or positive,
respectively. The map retains the cosine-distance value for those tiles and zeros
the rest. `-u` upsamples all three outputs. This is a directional gate: on
calibrated dB inputs, the sign corresponds to ΔdB; on linear backscatter inputs,
the sign of each pixel difference corresponds to the sign of its dB change. Use
the same calibration and intensity scaling for both images. Independently
stretched or clipped inputs can distort or erase the direction.

The labels describe backscatter decrease and increase, not confirmed flood or
vegetation classes. A decrease can be consistent with open-water inundation;
increases can include double-bounce flooding, agriculture, or construction.

To visualize the learned embedding across the POST scene, request a PCA
false-color image:

```sh
./siam -w flood.w -o change_map.pgm -v embeddings.ppm pre.pgm post.pgm
```

`embeddings.ppm` is a binary P6 image with one pixel per inference tile. PCA is
fit to the POST scene's normalized 32D tile embeddings; the first three
principal component scores are independently stretched to the red, green, and
blue channels. The colors show relative embedding variation within that scene,
so coherent regions may emerge, but colors do not have fixed meanings such as
water or urban and are not comparable across scenes. The `-u` option only
upsamples the PGM inference and directional maps.
