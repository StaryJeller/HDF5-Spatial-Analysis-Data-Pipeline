from __future__ import annotations

import argparse
from pathlib import Path

from transform_pipeline.io.readers.cls_sgm import default_playtest_scan_dir
from transform_pipeline.pipeline import (
    process_multi_sdd_scan,
    process_sdd_scan,
    write_multi_detector_outputs,
    write_pipeline_outputs,
)
from transform_pipeline.transforms.detectors import VALID_COMBINATION_MODES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct and subpixel-register CLS SGM SDD raster data."
    )
    parser.add_argument("scan_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--h5", default=None)
    parser.add_argument("--sdd", default="sdd1_0.bin")
    parser.add_argument("--all-detectors", action="store_true")
    parser.add_argument("--sdd-pattern", default="sdd*_0.bin")
    parser.add_argument(
        "--combine",
        choices=VALID_COMBINATION_MODES,
        default="sum",
        help="Detector fusion mode used with --all-detectors.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-cubes", action="store_true")
    parser.add_argument("--comparison", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scan_dir = args.scan_dir or default_playtest_scan_dir()
    output_dir = args.output_dir or scan_dir / "prototype_output"

    if args.all_detectors:
        reconstruction, registration = process_multi_sdd_scan(
            scan_dir=scan_dir,
            h5_filename=args.h5,
            sdd_pattern=args.sdd_pattern,
            combination_mode=args.combine,
        )
        outputs = write_multi_detector_outputs(
            reconstruction,
            registration,
            output_dir,
            save_cubes=args.save_cubes,
            save_comparison=args.comparison,
        )
        print(f"Multi-detector pipeline complete. Output: {outputs.output_dir}")
        print(f"Detectors: {', '.join(registration.detector_ids)}")
        print(f"Combination mode: {registration.combination_mode}")
        if outputs.comparison_grid_png is not None:
            print(f"Comparison grid: {outputs.comparison_grid_png}")
        return

    reconstruction, registration = process_sdd_scan(
        scan_dir=scan_dir,
        sdd_filename=args.sdd,
        h5_filename=args.h5,
    )
    outputs = write_pipeline_outputs(
        reconstruction,
        registration,
        output_dir,
        save_cubes=args.save_cubes,
        save_comparison=args.comparison,
    )
    print(f"Pipeline complete. Output: {outputs.output_dir}")
    if outputs.comparison_png is not None:
        print(f"Comparison: {outputs.comparison_png}")


if __name__ == "__main__":
    main()
