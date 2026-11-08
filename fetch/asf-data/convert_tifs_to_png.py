#!/usr/bin/env python3
"""Convert single-band UInt16 TIFFs in ``tif/`` to full-resolution PNGs.

The image stretch matches the original workflow: ignore zero pixels when
finding the 2nd and 98th percentiles, clip to that range, and scale to 8-bit.

Requires NumPy and the GDAL Python bindings (``osgeo.gdal``).
Run from this directory with:

    python3 convert_tifs_to_png.py

Existing PNGs are skipped by default. Pass ``--overwrite`` to regenerate them.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from osgeo import gdal


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "tif"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "png"
BLOCK_ROWS = 256
UINT16_VALUE_COUNT = 1 << 16


def percentile_from_histogram(
    histogram: np.ndarray, count: int, percentile: float, offset: int
) -> float:
    """Return NumPy's default linear percentile from an integer histogram."""
    position = (count - 1) * percentile / 100.0
    lower_rank = math.floor(position)
    upper_rank = math.ceil(position)
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    lower_bin = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper_bin = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    lower_value = lower_bin + offset
    upper_value = upper_bin + offset
    return lower_value + (upper_value - lower_value) * (position - lower_rank)


def get_percentile_range(band: gdal.Band, width: int, height: int) -> tuple[float, float]:
    """Compute exact nonzero percentiles using a UInt16 histogram."""
    type_name = gdal.GetDataTypeName(band.DataType)
    if type_name != "UInt16":
        raise ValueError(f"expected a UInt16 TIFF, found {type_name}")

    histogram = np.zeros(UINT16_VALUE_COUNT, dtype=np.uint64)
    for y in range(0, height, BLOCK_ROWS):
        rows = min(BLOCK_ROWS, height - y)
        pixels = band.ReadAsArray(0, y, width, rows)
        if pixels is None:
            raise RuntimeError(f"failed reading rows {y}–{y + rows}")
        nonzero = pixels[pixels > 0]
        if nonzero.size:
            counts = np.bincount(
                nonzero.astype(np.int64, copy=False),
                minlength=UINT16_VALUE_COUNT,
            )
            histogram += counts.astype(np.uint64, copy=False)

    count = int(histogram.sum(dtype=np.uint64))
    if count == 0:
        raise ValueError("the TIFF contains no nonzero pixels")

    p2 = percentile_from_histogram(histogram, count, 2, offset=0)
    p98 = percentile_from_histogram(histogram, count, 98, offset=0)
    if p98 <= p2:
        raise ValueError(f"invalid percentile range: p2={p2}, p98={p98}")
    return p2, p98


def convert_tiff(source: Path, destination: Path, overwrite: bool) -> bool:
    """Convert one TIFF; return False if its PNG already exists and is kept."""
    if destination.exists() and not overwrite:
        print(f"Skip existing: {destination.name}")
        return False

    dataset = gdal.Open(str(source), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError("GDAL could not open the TIFF")
    if dataset.RasterCount < 1:
        raise ValueError("the TIFF has no raster bands")

    width, height = dataset.RasterXSize, dataset.RasterYSize
    source_band = dataset.GetRasterBand(1)
    p2, p98 = get_percentile_range(source_band, width, height)

    memory_dataset = gdal.GetDriverByName("MEM").Create(
        "", width, height, 1, gdal.GDT_Byte
    )
    if memory_dataset is None:
        raise RuntimeError("GDAL could not create the in-memory output raster")

    output_band = memory_dataset.GetRasterBand(1)
    output_band.SetColorInterpretation(gdal.GCI_GrayIndex)
    for y in range(0, height, BLOCK_ROWS):
        rows = min(BLOCK_ROWS, height - y)
        pixels = source_band.ReadAsArray(0, y, width, rows)
        if pixels is None:
            raise RuntimeError(f"failed reading rows {y}–{y + rows}")
        clipped = np.clip(pixels, p2, p98)
        scaled = ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
        output_band.WriteArray(scaled, 0, y)

    output_band.FlushCache()
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=destination.parent
    )
    os.close(file_descriptor)
    os.unlink(temporary_name)
    try:
        copied = gdal.GetDriverByName("PNG").CreateCopy(
            temporary_name,
            memory_dataset,
            strict=1,
            options=["ZLEVEL=6"],
        )
        if copied is None:
            raise RuntimeError("GDAL could not write the PNG")
        copied = None
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
        memory_dataset = None
        dataset = None

    print(
        f"Saved {destination.name} ({width}x{height}; "
        f"p2={p2:.3f}, p98={p98:.3f})"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"directory of TIFFs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for PNGs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace PNGs that already exist",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    tiffs = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tiffs:
        parser.error(f"no .tif or .tiff files found in {args.input_dir}")

    gdal.UseExceptions()
    converted = skipped = failed = 0
    for source in tiffs:
        destination = args.output_dir / f"{source.stem}.png"
        try:
            if convert_tiff(source, destination, overwrite=args.overwrite):
                converted += 1
            else:
                skipped += 1
        except Exception as error:
            failed += 1
            print(f"Error converting {source.name}: {error}", file=sys.stderr)

    print(f"Done: {converted} converted, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
