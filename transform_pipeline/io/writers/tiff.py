from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
from PIL import Image, ImageDraw


def write_float_tiff(path: Path, image: np.ndarray, description: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.asarray(image, dtype=np.float32),
        photometric="minisblack",
        metadata={"axes": "YX", "description": description},
    )


def write_validity_mask(path: Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.asarray(mask, dtype=np.uint8) * 255,
        photometric="minisblack",
        metadata={"axes": "YX", "description": "255=valid; 0=invalid"},
    )


def make_uint8_preview(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(image)
    mask = np.asarray(mask, dtype=bool)

    if image.shape != mask.shape:
        raise ValueError("image and mask must have the same shape")

    values = image[mask & np.isfinite(image)]
    if values.size == 0:
        raise ValueError("No valid finite values are available for preview scaling")

    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    return make_uint8_preview_with_limits(image, mask, low, high)


def calculate_shared_preview_limits(
    images: Sequence[np.ndarray],
    validity_masks: Sequence[np.ndarray] | None = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> tuple[float, float]:
    """Calculate one display range for multiple quantitative images."""
    if not images:
        raise ValueError("At least one image is required")

    if validity_masks is None:
        validity_masks = [np.isfinite(image) for image in images]

    if len(images) != len(validity_masks):
        raise ValueError("images and validity_masks must have the same length")

    valid_value_sets: list[np.ndarray] = []
    for image, validity_mask in zip(images, validity_masks):
        image = np.asarray(image)
        validity_mask = np.asarray(validity_mask, dtype=bool)
        if image.shape != validity_mask.shape:
            raise ValueError("Each image and validity mask must have the same shape")
        selected = image[validity_mask & np.isfinite(image)]
        if selected.size:
            valid_value_sets.append(selected.astype(np.float64, copy=False))

    if not valid_value_sets:
        raise ValueError("No valid finite values are available for preview scaling")

    combined = np.concatenate(valid_value_sets)
    low = float(np.percentile(combined, low_percentile))
    high = float(np.percentile(combined, high_percentile))
    if not high > low:
        raise ValueError(f"Invalid shared preview range: low={low}, high={high}")
    return low, high


def make_uint8_preview_with_limits(
    image: np.ndarray,
    validity_mask: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Create a display-only uint8 image using explicit shared limits."""
    image = np.asarray(image, dtype=np.float64)
    validity_mask = np.asarray(validity_mask, dtype=bool)
    if image.shape != validity_mask.shape:
        raise ValueError("image and validity_mask must have the same shape")
    if not high > low:
        raise ValueError(f"Invalid preview range: low={low}, high={high}")

    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    preview = np.rint(scaled * 255.0).astype(np.uint8)
    preview[~validity_mask | ~np.isfinite(image)] = 0
    return preview


def save_preview(path: Path, preview: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(preview, dtype=np.uint8), mode="L").save(path)


def write_side_by_side_comparison(
    path: Path,
    original_image: np.ndarray,
    corrected_image: np.ndarray,
    corrected_validity_mask: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> tuple[float, float]:
    """Write a labeled original-versus-corrected PNG with shared scaling."""
    path = Path(path)
    original_image = np.asarray(original_image)
    corrected_image = np.asarray(corrected_image)
    corrected_validity_mask = np.asarray(corrected_validity_mask, dtype=bool)

    if original_image.shape != corrected_image.shape:
        raise ValueError("Original and corrected images must have the same shape")
    if corrected_image.shape != corrected_validity_mask.shape:
        raise ValueError("Corrected image and validity mask must have the same shape")

    original_validity_mask = np.isfinite(original_image)
    low, high = calculate_shared_preview_limits(
        [original_image, corrected_image],
        [original_validity_mask, corrected_validity_mask],
        low_percentile,
        high_percentile,
    )

    original_preview = make_uint8_preview_with_limits(
        original_image, original_validity_mask, low, high
    )
    corrected_preview = make_uint8_preview_with_limits(
        corrected_image, corrected_validity_mask, low, high
    )

    rows, columns = original_preview.shape
    label_height = 28
    divider_width = 2
    canvas = Image.new(
        "L",
        (columns * 2 + divider_width, rows + label_height),
        color=255,
    )
    canvas.paste(Image.fromarray(original_preview, mode="L"), (0, label_height))
    canvas.paste(
        Image.fromarray(corrected_preview, mode="L"),
        (columns + divider_width, label_height),
    )

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "LEFT: ORIGINAL", fill=0)
    draw.text(
        (columns + divider_width + 8, 8),
        "RIGHT: SUBPIXEL CORRECTED",
        fill=0,
    )
    draw.rectangle(
        [(columns, 0), (columns + divider_width - 1, rows + label_height - 1)],
        fill=128,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return low, high


def write_shift_csv(path: Path, shifts) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row",
        "raw_fractional_shift",
        "smoothed_fractional_shift",
        "peak_correlation",
        "zero_shift_correlation",
        "correlation_gain",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for shift in shifts:
            writer.writerow(
                {
                    "row": shift.row,
                    "raw_fractional_shift": shift.raw_shift,
                    "smoothed_fractional_shift": shift.smoothed_shift,
                    "peak_correlation": shift.peak_correlation,
                    "zero_shift_correlation": shift.zero_correlation,
                    "correlation_gain": shift.correlation_gain,
                }
            )
