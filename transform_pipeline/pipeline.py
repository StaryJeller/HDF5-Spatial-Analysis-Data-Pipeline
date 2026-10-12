from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from transform_pipeline.io.readers.cls_sgm import (
    discover_sdd_files,
    find_hdf5_file,
    reconstruct_detectors,
    reconstruct_scan,
    resolve_scan_dir,
)
from transform_pipeline.io.writers.detectors import (
    write_detector_comparison_grid,
    write_detector_manifest,
    write_detector_outputs,
    write_detector_shift_tables,
)
from transform_pipeline.io.writers.tiff import (
    make_uint8_preview,
    save_preview,
    write_float_tiff,
    write_shift_csv,
    write_side_by_side_comparison,
    write_validity_mask,
)
from transform_pipeline.transforms.detectors import register_detectors
from transform_pipeline.transforms.registration import apply_subpixel_registration, measure_subpixel_shifts

@dataclass(frozen=True)
class PipelineOutputs:
    output_dir: Path
    float_tiff: Path
    preview_png: Path
    validity_mask: Path
    shift_csv: Path
    comparison_png: Path | None

@dataclass(frozen=True)
class MultiDetectorPipelineOutputs:
    output_dir: Path
    detectors_directory: Path
    combined_directory: Path | None
    manifest_json: Path
    shift_csv_files: tuple[Path, ...]
    comparison_grid_png: Path | None


def process_sdd_scan(scan_dir=None, sdd_filename="sdd1_0.bin", h5_filename=None):
    scan_dir = resolve_scan_dir(scan_dir)
    h5_path = find_hdf5_file(scan_dir, h5_filename)
    binary_path = scan_dir / sdd_filename
    if not binary_path.is_file():
        raise FileNotFoundError(binary_path)
    reconstruction = reconstruct_scan(h5_path, binary_path)
    shifts = measure_subpixel_shifts(reconstruction.total_counts)
    registration = apply_subpixel_registration(reconstruction.cube, shifts)
    return reconstruction, registration


def process_multi_sdd_scan(scan_dir=None, h5_filename=None, sdd_pattern="sdd*_0.bin", combination_mode="sum"):
    scan_dir = resolve_scan_dir(scan_dir)
    h5_path = find_hdf5_file(scan_dir, h5_filename)
    reconstruction = reconstruct_detectors(h5_path, discover_sdd_files(scan_dir, sdd_pattern))
    return reconstruction, register_detectors(reconstruction, combination_mode)


def write_pipeline_outputs(reconstruction, registration, output_dir, save_cubes=False, save_comparison=False):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if save_cubes:
        np.save(output_dir / "sdd_cube_reconstructed.npy", reconstruction.cube)
        np.save(output_dir / "sdd_cube_subpixel_registered.npy", registration.cube)
    np.save(output_dir / "total_counts_subpixel_registered.npy", registration.total_counts)
    np.save(output_dir / "x_axis.npy", reconstruction.geometry.x_axis)
    np.save(output_dir / "y_axis.npy", reconstruction.geometry.y_axis)
    float_tiff = output_dir / "total_counts_subpixel_registered_float32.tif"
    preview_png = output_dir / "total_counts_subpixel_registered_preview.png"
    validity_mask = output_dir / "subpixel_validity_mask.tif"
    shift_csv = output_dir / "subpixel_shifts.csv"
    write_float_tiff(float_tiff, registration.total_counts, "Subpixel-registered SDD total counts")
    write_validity_mask(validity_mask, registration.validity_mask)
    save_preview(preview_png, make_uint8_preview(registration.total_counts, registration.validity_mask))
    write_shift_csv(shift_csv, registration.shifts)
    comparison_png = None
    if save_comparison:
        comparison_png = output_dir / "original_vs_subpixel_corrected.png"
        write_side_by_side_comparison(comparison_png, reconstruction.total_counts, registration.total_counts, registration.validity_mask)
    return PipelineOutputs(output_dir, float_tiff, preview_png, validity_mask, shift_csv, comparison_png)


def write_multi_detector_outputs(reconstruction, registration, output_dir, save_cubes=False, save_comparison=False):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "x_axis.npy", reconstruction.geometry.x_axis)
    np.save(output_dir / "y_axis.npy", reconstruction.geometry.y_axis)
    write_detector_outputs(output_dir, reconstruction, registration, save_cubes)
    shift_files = write_detector_shift_tables(output_dir, registration.shifts_by_detector)
    manifest = output_dir / "detector_manifest.json"
    write_detector_manifest(
        manifest, reconstruction.detector_ids, reconstruction.source_files,
        registration.combination_mode, tuple(reconstruction.cubes.shape[1:]),
        reconstruction.cubes.shape[-1],
    )
    comparison_grid = None
    if save_comparison:
        comparison_grid = output_dir / "detector_original_vs_corrected_grid.png"
        write_detector_comparison_grid(
            comparison_grid, reconstruction.total_counts, registration.total_counts,
            registration.validity_masks, registration.detector_ids,
        )
    combined_dir = output_dir / "combined" if registration.combined_cube is not None else None
    return MultiDetectorPipelineOutputs(
        output_dir, output_dir / "detectors", combined_dir, manifest,
        shift_files, comparison_grid,
    )
