from __future__ import annotations

import argparse
from pathlib import Path
import xarray as xr

from transform_pipeline.stitching import VALID_OVERLAP_POLICIES, stitch_canonical_stores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch canonical CLS SGM Zarr tiles by their X/Y coordinates."
    )
    parser.add_argument("stores", type=Path, nargs="+", help="Two or more canonical_sgm.zarr directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output mosaic Zarr directory.")
    parser.add_argument("--overlap", choices=VALID_OVERLAP_POLICIES, default="mean")
    parser.add_argument("--include-spectral", action="store_true")
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-roundtrip", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store, manifest, preview = stitch_canonical_stores(
        args.stores,
        args.output,
        overlap_policy=args.overlap,
        include_spectral=args.include_spectral,
        coordinate_tolerance=args.coordinate_tolerance,
        overwrite=args.overwrite,
    )
    if args.verify_roundtrip:
        dataset = xr.open_zarr(str(store), consolidated=True, chunks=None)
        try:
            required = {"mosaic_total_counts_registered", "mosaic_validity_mask", "source_tile_index", "overlap_count"}
            missing = required.difference(dataset.data_vars)
            if missing:
                raise ValueError(f"Round-trip output missing variables: {sorted(missing)}")
            if args.include_spectral and "mosaic_counts_registered" not in dataset:
                raise ValueError("Round-trip output is missing mosaic_counts_registered.")
        finally:
            dataset.close()
    print(f"Mosaic: {store}")
    print(f"Manifest: {manifest}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
