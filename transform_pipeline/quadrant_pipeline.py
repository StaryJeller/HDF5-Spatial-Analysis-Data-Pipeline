from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from transform_pipeline.interoperability import (
    build_canonical_dataset,
    write_canonical_zarr,
    write_dataset_summary,
)
from transform_pipeline.pipeline import (
    process_multi_sdd_scan,
    write_multi_detector_outputs,
)
from transform_pipeline.stitching import stitch_canonical_stores
from transform_pipeline.stitching.preview import write_physical_mosaic_preview


@dataclass(frozen=True)
class QuadrantResult:
    quadrant_id: str
    scan_directory: Path
    output_directory: Path
    canonical_store: Path
    detector_ids: tuple[str, ...]
    combination_mode: str


def discover_quadrant_directories(
    root: Path,
    sdd_pattern: str = "sdd*_0.bin",
) -> tuple[Path, ...]:
    """Find immediate child directories containing one HDF5 and SDD data."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    matches = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if directory.name in {"stitched_output", "prototype_output"}:
            continue
        hdf5_files = sorted(directory.glob("*.h5"))
        sdd_files = sorted(directory.glob(sdd_pattern))
        if len(hdf5_files) == 1 and sdd_files:
            matches.append(directory)

    if len(matches) < 2:
        raise ValueError(
            f"Expected at least two processable quadrant directories under {root}; "
            f"found {len(matches)}."
        )
    return tuple(matches)


def process_quadrant(
    scan_directory: Path,
    combination_mode: str = "sum",
    sdd_pattern: str = "sdd*_0.bin",
    save_cubes: bool = True,
    save_comparison: bool = True,
    overwrite_canonical: bool = True,
    verify_roundtrip: bool = True,
) -> QuadrantResult:
    scan_directory = Path(scan_directory).resolve()
    output_directory = scan_directory / "prototype_output"

    reconstruction, registration = process_multi_sdd_scan(
        scan_dir=scan_directory,
        sdd_pattern=sdd_pattern,
        combination_mode=combination_mode,
    )
    write_multi_detector_outputs(
        reconstruction,
        registration,
        output_directory,
        save_cubes=save_cubes,
        save_comparison=save_comparison,
    )

    dataset = build_canonical_dataset(output_directory)
    canonical_store = output_directory / "canonical_sgm.zarr"
    write_dataset_summary(
        dataset,
        output_directory / "canonical_dataset_summary.json",
    )
    write_canonical_zarr(
        dataset,
        canonical_store,
        overwrite=overwrite_canonical,
    )

    if verify_roundtrip:
        import xarray as xr

        reopened = xr.open_zarr(
            str(canonical_store),
            consolidated=True,
            chunks=None,
        )
        try:
            if dict(reopened.sizes) != dict(dataset.sizes):
                raise ValueError(
                    f"Canonical round-trip dimension mismatch for {scan_directory.name}."
                )
            if set(reopened.data_vars) != set(dataset.data_vars):
                raise ValueError(
                    f"Canonical round-trip variable mismatch for {scan_directory.name}."
                )
        finally:
            reopened.close()
            dataset.close()

    return QuadrantResult(
        quadrant_id=scan_directory.name,
        scan_directory=scan_directory,
        output_directory=output_directory,
        canonical_store=canonical_store,
        detector_ids=tuple(registration.detector_ids),
        combination_mode=registration.combination_mode,
    )


def run_quadrant_pipeline(
    root: Path,
    combination_mode: str = "sum",
    overlap_policy: str = "mean",
    sdd_pattern: str = "sdd*_0.bin",
    save_cubes: bool = True,
    save_comparison: bool = True,
    include_spectral_stitch: bool = False,
    coordinate_tolerance: float = 1e-6,
    overwrite: bool = False,
    verify_roundtrip: bool = True,
) -> dict:
    """Process every quadrant, create canonical stores, then stitch them."""
    root = Path(root).resolve()
    quadrants = discover_quadrant_directories(root, sdd_pattern=sdd_pattern)
    stitched_directory = root / "stitched_output"
    stitched_directory.mkdir(parents=True, exist_ok=True)
    run_manifest = stitched_directory / "quadrant_pipeline_manifest.json"

    if run_manifest.exists() and not overwrite:
        raise FileExistsError(
            f"Pipeline manifest already exists: {run_manifest}. Use --overwrite to rerun."
        )

    results = []
    for quadrant in quadrants:
        results.append(
            process_quadrant(
                quadrant,
                combination_mode=combination_mode,
                sdd_pattern=sdd_pattern,
                save_cubes=save_cubes,
                save_comparison=save_comparison,
                overwrite_canonical=True,
                verify_roundtrip=verify_roundtrip,
            )
        )

    mosaic_name = (
        "mosaic_spectral.zarr"
        if include_spectral_stitch
        else "mosaic.zarr"
    )
    mosaic_store = stitched_directory / mosaic_name
    mosaic_store, stitch_manifest, preview = stitch_canonical_stores(
        [result.canonical_store for result in results],
        mosaic_store,
        overlap_policy=overlap_policy,
        include_spectral=include_spectral_stitch,
        coordinate_tolerance=coordinate_tolerance,
        overwrite=overwrite,
    )

    payload = {
        "root": str(root),
        "combination_mode": combination_mode,
        "overlap_policy": overlap_policy,
        "sdd_pattern": sdd_pattern,
        "save_cubes": save_cubes,
        "save_comparison": save_comparison,
        "spectral_stitch": include_spectral_stitch,
        "coordinate_tolerance": coordinate_tolerance,
        "quadrants": [
            {
                **asdict(result),
                "scan_directory": str(result.scan_directory),
                "output_directory": str(result.output_directory),
                "canonical_store": str(result.canonical_store),
                "detector_ids": list(result.detector_ids),
            }
            for result in results
        ],
        "mosaic_store": str(mosaic_store),
        "stitch_manifest": str(stitch_manifest),
        "preview": str(preview),
    }
    run_manifest.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    preview_size = write_physical_mosaic_preview(
        mosaic_store=mosaic_store,
        preview_path=preview,
        run_manifest_path=run_manifest,
    )
    payload["preview_scaling"] = {
        "mode": "physical_coordinate_aspect",
        "preview_size_width_height": [int(preview_size[0]), int(preview_size[1])],
    }
    return payload
