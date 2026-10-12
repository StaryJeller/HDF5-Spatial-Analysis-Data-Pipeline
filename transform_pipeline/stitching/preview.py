from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

from transform_pipeline.io.writers.tiff import make_uint8_preview


def physical_preview_size(
    image_shape: tuple[int, int],
    x_spacing: float,
    y_spacing: float,
) -> tuple[int, int]:
    """Calculate PNG dimensions that preserve physical X/Y proportions."""
    rows, columns = image_shape
    if rows <= 0 or columns <= 0:
        raise ValueError("image_shape must contain positive dimensions")
    if not np.isfinite(x_spacing) or not np.isfinite(y_spacing):
        raise ValueError("X and Y spacing must be finite")
    if x_spacing <= 0 or y_spacing <= 0:
        raise ValueError("X and Y spacing must be positive")

    scale = 1.0 / min(x_spacing, y_spacing)
    width = max(1, int(round(columns * x_spacing * scale)))
    height = max(1, int(round(rows * y_spacing * scale)))
    return width, height


def write_physical_mosaic_preview(
    mosaic_store: Path,
    preview_path: Path,
    run_manifest_path: Path | None = None,
) -> tuple[int, int]:
    """Rewrite the mosaic preview using coordinate-derived physical aspect."""
    mosaic_store = Path(mosaic_store)
    preview_path = Path(preview_path)
    dataset = xr.open_zarr(str(mosaic_store), consolidated=True, chunks=None)
    try:
        image = np.asarray(dataset["mosaic_total_counts_registered"].values)
        mask = np.asarray(dataset["mosaic_validity_mask"].values, dtype=bool)
        x = np.asarray(dataset.x.values, dtype=np.float64)
        y = np.asarray(dataset.y.values, dtype=np.float64)
        if x.size < 2 or y.size < 2:
            raise ValueError("Mosaic X and Y coordinates require at least two values")
        x_spacing = float(np.median(np.diff(x)))
        y_spacing = float(np.median(np.diff(y)))
        preview = make_uint8_preview(image, mask)
        output_size = physical_preview_size(preview.shape, x_spacing, y_spacing)
        display = Image.fromarray(preview, mode="L").resize(
            output_size,
            resample=Image.Resampling.NEAREST,
        )
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        display.save(preview_path)
    finally:
        dataset.close()

    if run_manifest_path is not None:
        run_manifest_path = Path(run_manifest_path)
        payload = (
            json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest_path.is_file()
            else {}
        )
        payload["preview_scaling"] = {
            "mode": "physical_coordinate_aspect",
            "resampling": "nearest",
            "analytical_shape_yx": [int(image.shape[0]), int(image.shape[1])],
            "preview_size_width_height": [int(output_size[0]), int(output_size[1])],
            "x_spacing": x_spacing,
            "y_spacing": y_spacing,
            "pixel_aspect_ratio_y_over_x": y_spacing / x_spacing,
        }
        run_manifest_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return output_size
