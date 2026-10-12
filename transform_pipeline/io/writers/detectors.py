from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from transform_pipeline.io.writers.tiff import (
    calculate_shared_preview_limits,
    make_uint8_preview_with_limits,
    write_float_tiff,
    write_shift_csv,
    write_validity_mask,
)


def write_detector_manifest(
    path: Path,
    detector_ids: tuple[str, ...],
    source_files: tuple[Path, ...],
    combination_mode: str,
    cube_shape: tuple[int, ...],
    channels: int,
) -> None:
    payload = {
        "detector_ids": list(detector_ids),
        "source_files": [source.name for source in source_files],
        "combination_mode": combination_mode,
        "per_detector_cube_shape": list(cube_shape),
        "detector_count": len(detector_ids),
        "channels": channels,
        "input_dtype": "<u4",
        "registered_dtype": "float32",
        "registration_policy": (
            "independent row-shift profile measured and applied per detector; "
            "fusion occurs only after detector-specific registration"
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_detector_shift_tables(
    output_dir: Path,
    shifts_by_detector: dict,
) -> tuple[Path, ...]:
    shift_dir = Path(output_dir) / "detector_shifts"
    shift_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for detector_id, shifts in shifts_by_detector.items():
        path = shift_dir / f"{detector_id}_subpixel_shifts.csv"
        write_shift_csv(path, shifts)
        paths.append(path)
    return tuple(paths)


def write_detector_outputs(output_dir, reconstruction, registration, save_cubes):
    output_dir = Path(output_dir)
    detector_dir = output_dir / "detectors"
    detector_dir.mkdir(parents=True, exist_ok=True)
    for index, detector_id in enumerate(registration.detector_ids):
        np.save(
            detector_dir / f"{detector_id}_total_counts_registered.npy",
            registration.total_counts[index],
        )
        write_float_tiff(
            detector_dir / f"{detector_id}_total_counts_registered_float32.tif",
            registration.total_counts[index],
            f"Independently registered total counts for {detector_id}",
        )
        write_validity_mask(
            detector_dir / f"{detector_id}_validity_mask.tif",
            registration.validity_masks[index],
        )
        if save_cubes:
            np.save(
                detector_dir / f"{detector_id}_cube_reconstructed.npy",
                reconstruction.cubes[index],
            )
            np.save(
                detector_dir / f"{detector_id}_cube_registered.npy",
                registration.cubes[index],
            )

    if registration.combined_cube is not None:
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        np.save(combined_dir / "combined_total_counts_registered.npy", registration.combined_total_counts)
        write_float_tiff(
            combined_dir / "combined_total_counts_registered_float32.tif",
            registration.combined_total_counts,
            f"Fusion after independent detector registration ({registration.combination_mode})",
        )
        write_validity_mask(
            combined_dir / "combined_validity_mask.tif",
            registration.combined_validity_mask,
        )
        if save_cubes:
            np.save(combined_dir / "combined_cube_registered.npy", registration.combined_cube)


def write_detector_comparison_grid(path, original_totals, corrected_totals, corrected_masks, detector_ids):
    original_totals = np.asarray(original_totals)
    corrected_totals = np.asarray(corrected_totals)
    corrected_masks = np.asarray(corrected_masks, dtype=bool)
    original_masks = np.isfinite(original_totals)
    low, high = calculate_shared_preview_limits(
        [original_totals, corrected_totals],
        [original_masks, corrected_masks],
    )
    detector_count, rows, columns = original_totals.shape
    label_height, gap, divider = 24, 2, 2
    panel_height = rows + label_height
    canvas = Image.new(
        "L",
        (columns * 2 + divider, detector_count * panel_height + (detector_count - 1) * gap),
        color=255,
    )
    draw = ImageDraw.Draw(canvas)
    for index, detector_id in enumerate(detector_ids):
        top = index * (panel_height + gap)
        original = make_uint8_preview_with_limits(
            original_totals[index], original_masks[index], low, high
        )
        corrected = make_uint8_preview_with_limits(
            corrected_totals[index], corrected_masks[index], low, high
        )
        canvas.paste(Image.fromarray(original, mode="L"), (0, top + label_height))
        canvas.paste(Image.fromarray(corrected, mode="L"), (columns + divider, top + label_height))
        draw.text((6, top + 6), f"{detector_id}: ORIGINAL", fill=0)
        draw.text((columns + divider + 6, top + 6), f"{detector_id}: OWN CORRECTION", fill=0)
        draw.rectangle([(columns, top), (columns + divider - 1, top + panel_height - 1)], fill=128)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
