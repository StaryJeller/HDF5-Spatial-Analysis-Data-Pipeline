from __future__ import annotations

import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

from transform_pipeline.io.writers.tiff import make_uint8_preview
from .coordinate import StitchPlan, build_stitch_plan

VALID_OVERLAP_POLICIES = ("first", "mean")


def _create_memmap(directory: Path, name: str, shape: tuple[int, ...], dtype, fill_value=0):
    path = directory / f"{name}.dat"
    array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[...] = fill_value
    return array


def _write_preview(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    preview = make_uint8_preview(image, mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(path)


def _close_memmap(array: np.memmap | None) -> None:
    """Flush and close a NumPy memory map so Windows can delete its file."""
    if array is None:
        return
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def stitch_canonical_stores(
    store_paths,
    output_store: Path,
    overlap_policy: str = "mean",
    include_spectral: bool = False,
    coordinate_tolerance: float = 1e-6,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    if overlap_policy not in VALID_OVERLAP_POLICIES:
        raise ValueError(f"Unsupported overlap policy {overlap_policy!r}.")
    output_store = Path(output_store).resolve()
    if output_store.exists():
        if not overwrite:
            raise FileExistsError(f"Output store exists: {output_store}")
        shutil.rmtree(output_store)
    output_store.parent.mkdir(parents=True, exist_ok=True)

    plan = build_stitch_plan(store_paths, coordinate_tolerance)
    y_size, x_size = plan.y.size, plan.x.size
    temp_parent = output_store.parent
    with tempfile.TemporaryDirectory(prefix="stitch_", dir=temp_parent) as temp_name:
        temp_dir = Path(temp_name)
        total_sum = _create_memmap(temp_dir, "total_sum", (y_size, x_size), np.float64)
        valid_count = _create_memmap(temp_dir, "valid_count", (y_size, x_size), np.uint16)
        source_tile_index = _create_memmap(temp_dir, "source_tile_index", (y_size, x_size), np.int32, -1)
        spectral_sum = None
        if include_spectral:
            spectral_sum = _create_memmap(
                temp_dir,
                "spectral_sum",
                (y_size, x_size, plan.channel.size),
                np.float32,
            )

        has_spectral = []
        for tile_index, placement in enumerate(plan.placements):
            dataset = xr.open_zarr(str(placement.store_path), consolidated=True, chunks=None)
            try:
                total = np.asarray(dataset["combined_total_counts_registered"].values, dtype=np.float32)
                mask = np.asarray(dataset["combined_validity_mask"].values, dtype=bool) & np.isfinite(total)
                ys = slice(placement.y_start, placement.y_stop)
                xs = slice(placement.x_start, placement.x_stop)
                destination_count = valid_count[ys, xs]
                destination_source = source_tile_index[ys, xs]
                first_valid = mask & (destination_count == 0)
                destination_source[first_valid] = tile_index

                if overlap_policy == "first":
                    writable = first_valid
                    destination_total = total_sum[ys, xs]
                    destination_total[writable] = total[writable]
                else:
                    destination_total = total_sum[ys, xs]
                    destination_total[mask] += total[mask]
                destination_count[mask] += 1

                spectral_present = "combined_counts_registered" in dataset.data_vars
                has_spectral.append(spectral_present)
                if include_spectral:
                    if not spectral_present:
                        raise ValueError(f"{placement.store_path} lacks combined_counts_registered.")
                    cube = np.asarray(dataset["combined_counts_registered"].values, dtype=np.float32)
                    if cube.shape != (total.shape[0], total.shape[1], plan.channel.size):
                        raise ValueError(f"Spectral shape mismatch for {placement.store_path}.")
                    destination_cube = spectral_sum[ys, xs, :]
                    if overlap_policy == "first":
                        destination_cube[first_valid, :] = cube[first_valid, :]
                    else:
                        destination_cube[mask, :] += cube[mask, :]
            finally:
                dataset.close()
                # Release all views into the disk-backed destination arrays before
                # Windows is asked to remove the temporary files.
                for local_name in (
                    "destination_count",
                    "destination_source",
                    "destination_total",
                    "destination_cube",
                    "total",
                    "mask",
                    "cube",
                    "first_valid",
                    "writable",
                ):
                    if local_name in locals():
                        del locals()[local_name]

        mosaic_mask = np.asarray(valid_count > 0)
        mosaic_total = np.asarray(total_sum, dtype=np.float32)
        if overlap_policy == "mean":
            mosaic_total[mosaic_mask] /= valid_count[mosaic_mask]
        mosaic_total[~mosaic_mask] = np.nan

        variables = {
            "mosaic_total_counts_registered": (
                ("y", "x"), mosaic_total,
                {"long_name": "coordinate-stitched fused total counts", "overlap_policy": overlap_policy},
            ),
            "mosaic_validity_mask": (("y", "x"), mosaic_mask),
            "source_tile_index": (
                ("y", "x"), np.asarray(source_tile_index),
                {"long_name": "zero-based first valid source tile index", "fill_value": -1},
            ),
            "overlap_count": (
                ("y", "x"), np.asarray(valid_count),
                {"long_name": "number of valid source tiles covering each pixel"},
            ),
        }
        if include_spectral:
            mosaic_cube = np.asarray(spectral_sum, dtype=np.float32)
            if overlap_policy == "mean":
                for channel_start in range(0, plan.channel.size, 32):
                    channel_stop = min(channel_start + 32, plan.channel.size)
                    block = mosaic_cube[:, :, channel_start:channel_stop]
                    block[mosaic_mask, :] /= valid_count[mosaic_mask][:, None]
            mosaic_cube[~mosaic_mask, :] = np.nan
            variables["mosaic_counts_registered"] = (
                ("y", "x", "channel"), mosaic_cube,
                {"long_name": "coordinate-stitched fused spectral counts", "overlap_policy": overlap_policy},
            )

        tile_ids = np.asarray([item.tile_id for item in plan.placements], dtype=str)
        dataset = xr.Dataset(
            data_vars=variables,
            coords={
                "y": ("y", plan.y, {"long_name": "local scan Y coordinate"}),
                "x": ("x", plan.x, {"long_name": "local scan X coordinate"}),
                "channel": ("channel", plan.channel, {"long_name": "uncalibrated detector channel index"}),
                "tile": ("tile", tile_ids),
            },
            attrs={
                "title": "Coordinate-stitched CLS SGM mosaic",
                "stitch_method": "coordinate placement only; no image registration or resampling",
                "overlap_policy": overlap_policy,
                "coordinate_reference": "local CLS scan coordinates; no CRS assigned",
                "coordinate_tolerance": float(coordinate_tolerance),
                "x_spacing": plan.x_spacing,
                "y_spacing": plan.y_spacing,
                "spectral_included": bool(include_spectral),
                "canonical_schema_version": "0.2.0",
            },
        )
        encoding = {
            "mosaic_total_counts_registered": {"chunks": (min(128, y_size), min(128, x_size))},
            "mosaic_validity_mask": {"chunks": (min(128, y_size), min(128, x_size))},
            "source_tile_index": {"chunks": (min(128, y_size), min(128, x_size))},
            "overlap_count": {"chunks": (min(128, y_size), min(128, x_size))},
        }
        if include_spectral:
            encoding["mosaic_counts_registered"] = {
                "chunks": (min(64, y_size), min(64, x_size), min(64, plan.channel.size))
            }
        dataset.to_zarr(
            str(output_store),
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding=encoding,
        )

        manifest_path = output_store.parent / f"{output_store.stem}_stitch_manifest.json"
        manifest = {
            "output_store": str(output_store),
            "overlap_policy": overlap_policy,
            "spectral_included": include_spectral,
            "coordinate_tolerance": coordinate_tolerance,
            "x_spacing": plan.x_spacing,
            "y_spacing": plan.y_spacing,
            "output_shape_yx": [int(y_size), int(x_size)],
            "channel_count": int(plan.channel.size),
            "tiles": [
                {
                    "tile_index": index,
                    "tile_id": item.tile_id,
                    "store_path": str(item.store_path),
                    "output_window": {
                        "y_start": item.y_start, "y_stop": item.y_stop,
                        "x_start": item.x_start, "x_stop": item.x_stop,
                    },
                    "coordinate_bounds": {
                        "x_min": item.x_min, "x_max": item.x_max,
                        "y_min": item.y_min, "y_max": item.y_max,
                    },
                }
                for index, item in enumerate(plan.placements)
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        preview_path = output_store.parent / f"{output_store.stem}_preview.png"
        _write_preview(preview_path, mosaic_total, mosaic_mask)
        dataset.close()

        # Xarray variables and NumPy arrays can retain views of the memmaps even
        # after the Zarr write completes. Explicitly release those views, collect
        # them, then close the underlying mappings before TemporaryDirectory
        # performs Windows cleanup.
        del dataset
        del variables
        del mosaic_total
        del mosaic_mask
        if include_spectral:
            del mosaic_cube
        gc.collect()

        _close_memmap(spectral_sum)
        _close_memmap(source_tile_index)
        _close_memmap(valid_count)
        _close_memmap(total_sum)
        del spectral_sum, source_tile_index, valid_count, total_sum
        gc.collect()

        return output_store, manifest_path, preview_path
