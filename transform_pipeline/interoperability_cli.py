from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from transform_pipeline.interoperability import (
    build_canonical_dataset,
    write_canonical_zarr,
    write_dataset_summary,
)
from transform_pipeline.io.readers.cls_sgm import default_playtest_scan_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and export a canonical labeled dataset from validated pipeline outputs."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Defaults to Playtest_model/Testmodel_NCo_7_SW/prototype_output",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="Defaults to <output_dir>/canonical_sgm.zarr",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-roundtrip",
        action="store_true",
        help="Reopen the written store and compare dimensions and variable names.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or default_playtest_scan_dir() / "prototype_output"
    zarr_path = args.zarr_path or output_dir / "canonical_sgm.zarr"

    dataset = build_canonical_dataset(output_dir)
    summary_path = write_dataset_summary(dataset, output_dir / "canonical_dataset_summary.json")
    write_canonical_zarr(dataset, zarr_path, overwrite=args.overwrite)

    if args.verify_roundtrip:
        reopened = xr.open_zarr(str(zarr_path), consolidated=True)
        try:
            if dict(reopened.sizes) != dict(dataset.sizes):
                raise ValueError("Round-trip dimension mismatch.")
            if set(reopened.data_vars) != set(dataset.data_vars):
                raise ValueError("Round-trip data-variable mismatch.")
        finally:
            reopened.close()

    print(f"Canonical dataset: {zarr_path}")
    print(f"Dataset summary: {summary_path}")
    print(f"Dimensions: {dict(dataset.sizes)}")
    print(f"Variables: {', '.join(dataset.data_vars)}")


if __name__ == "__main__":
    main()
