from __future__ import annotations

import argparse
from pathlib import Path

from transform_pipeline.quadrant_pipeline import run_quadrant_pipeline
from transform_pipeline.stitching import VALID_OVERLAP_POLICIES
from transform_pipeline.transforms.detectors import VALID_COMBINATION_MODES


def default_four_quadrant_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Playtest_model"
        / "Testmodel_4_quadrant"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process all quadrant acquisitions under one root, export each "
            "canonical dataset, and create a coordinate-based mosaic."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=None,
        help="Defaults to Playtest_model/Testmodel_4_quadrant",
    )
    parser.add_argument(
        "--combine",
        choices=VALID_COMBINATION_MODES,
        default="sum",
    )
    parser.add_argument(
        "--overlap",
        choices=VALID_OVERLAP_POLICIES,
        default="mean",
    )
    parser.add_argument("--sdd-pattern", default="sdd*_0.bin")
    parser.add_argument("--no-save-cubes", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--include-spectral-stitch", action="store_true")
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify-roundtrip", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root or default_four_quadrant_root()
    result = run_quadrant_pipeline(
        root=root,
        combination_mode=args.combine,
        overlap_policy=args.overlap,
        sdd_pattern=args.sdd_pattern,
        save_cubes=not args.no_save_cubes,
        save_comparison=not args.no_comparison,
        include_spectral_stitch=args.include_spectral_stitch,
        coordinate_tolerance=args.coordinate_tolerance,
        overwrite=args.overwrite,
        verify_roundtrip=not args.no_verify_roundtrip,
    )
    print(f"Processed quadrants: {len(result['quadrants'])}")
    print(f"Mosaic: {result['mosaic_store']}")
    print(f"Preview: {result['preview']}")
    print(
        "Pipeline manifest: "
        + str(
            Path(result["root"])
            / "stitched_output"
            / "quadrant_pipeline_manifest.json"
        )
    )


if __name__ == "__main__":
    main()
