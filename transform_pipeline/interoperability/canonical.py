from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
import xarray as xr


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_detector_manifest(output_dir: Path) -> dict:
    path = _require_file(Path(output_dir) / "detector_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_validity_mask(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(_require_file(path)), dtype=np.uint8) > 0


def _load_detector_totals(
    output_dir: Path,
    detector_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    detector_dir = output_dir / "detectors"
    totals = []
    masks = []
    for detector_id in detector_ids:
        totals.append(
            np.load(
                _require_file(
                    detector_dir / f"{detector_id}_total_counts_registered.npy"
                )
            )
        )
        masks.append(
            _load_validity_mask(
                detector_dir / f"{detector_id}_validity_mask.tif"
            )
        )
    return np.stack(totals, axis=0), np.stack(masks, axis=0)


def _load_optional_detector_cubes(
    output_dir: Path,
    detector_ids: tuple[str, ...],
) -> np.ndarray | None:
    detector_dir = output_dir / "detectors"
    paths = [detector_dir / f"{detector_id}_cube_registered.npy" for detector_id in detector_ids]
    present = [path.is_file() for path in paths]
    if not any(present):
        return None
    if not all(present):
        missing = [path.name for path, exists in zip(paths, present) if not exists]
        raise ValueError(
            "Registered detector cubes are incomplete. Missing: " + ", ".join(missing)
        )
    return np.stack([np.load(path, mmap_mode="r") for path in paths], axis=0)


def build_canonical_dataset(output_dir: Path) -> xr.Dataset:
    """Build one labeled dataset from validated multi-detector pipeline outputs."""
    output_dir = Path(output_dir).resolve()
    manifest = load_detector_manifest(output_dir)
    detector_ids = tuple(manifest["detector_ids"])
    x = np.asarray(np.load(_require_file(output_dir / "x_axis.npy")), dtype=np.float64)
    y = np.asarray(np.load(_require_file(output_dir / "y_axis.npy")), dtype=np.float64)
    detector_totals, detector_masks = _load_detector_totals(output_dir, detector_ids)

    expected_shape = (len(detector_ids), y.size, x.size)
    if detector_totals.shape != expected_shape:
        raise ValueError(
            f"Detector total-count shape {detector_totals.shape} does not match "
            f"coordinates {expected_shape}."
        )
    if detector_masks.shape != expected_shape:
        raise ValueError(
            f"Detector mask shape {detector_masks.shape} does not match "
            f"coordinates {expected_shape}."
        )

    data_vars = {
        "detector_total_counts_registered": (
            ("detector", "y", "x"),
            detector_totals.astype(np.float32, copy=False),
            {
                "long_name": "independently registered detector total counts",
                "status": "quantitative",
            },
        ),
        "detector_validity_mask": (
            ("detector", "y", "x"),
            detector_masks,
            {
                "long_name": "validity after detector-specific subpixel registration",
            },
        ),
    }

    combined_dir = output_dir / "combined"
    combined_total_path = combined_dir / "combined_total_counts_registered.npy"
    combined_mask_path = combined_dir / "combined_validity_mask.tif"
    if combined_total_path.is_file() or combined_mask_path.is_file():
        if not combined_total_path.is_file() or not combined_mask_path.is_file():
            raise ValueError("Combined total-count image and mask must both be present.")
        combined_total = np.asarray(np.load(combined_total_path), dtype=np.float32)
        combined_mask = _load_validity_mask(combined_mask_path)
        if combined_total.shape != (y.size, x.size):
            raise ValueError("Combined total-count shape does not match X/Y coordinates.")
        if combined_mask.shape != (y.size, x.size):
            raise ValueError("Combined validity-mask shape does not match X/Y coordinates.")
        data_vars["combined_total_counts_registered"] = (
            ("y", "x"),
            combined_total,
            {
                "long_name": "fused total counts after detector-specific registration",
                "combination_mode": manifest.get("combination_mode", "unknown"),
                "status": "quantitative",
            },
        )
        data_vars["combined_validity_mask"] = (
            ("y", "x"),
            combined_mask,
            {
                "long_name": "validity intersection for fused detector product",
            },
        )

    detector_cubes = _load_optional_detector_cubes(output_dir, detector_ids)
    channel_count = int(manifest.get("channels", 256))
    if detector_cubes is not None:
        expected_cube_shape = (len(detector_ids), y.size, x.size, channel_count)
        if detector_cubes.shape != expected_cube_shape:
            raise ValueError(
                f"Detector cube shape {detector_cubes.shape} does not match "
                f"expected {expected_cube_shape}."
            )
        data_vars["detector_counts_registered"] = (
            ("detector", "y", "x", "channel"),
            detector_cubes,
            {
                "long_name": "independently registered detector spectral counts",
                "status": "quantitative",
            },
        )

    combined_cube_path = combined_dir / "combined_cube_registered.npy"
    if combined_cube_path.is_file():
        combined_cube = np.load(combined_cube_path, mmap_mode="r")
        if combined_cube.shape != (y.size, x.size, channel_count):
            raise ValueError("Combined cube shape does not match X/Y/channel coordinates.")
        data_vars["combined_counts_registered"] = (
            ("y", "x", "channel"),
            combined_cube,
            {
                "long_name": "fused spectral counts after detector-specific registration",
                "combination_mode": manifest.get("combination_mode", "unknown"),
                "status": "quantitative",
            },
        )

    attributes = {
        "title": "CLS SGM registered multi-detector dataset",
        "source_output_directory": str(output_dir),
        "source_files": json.dumps(manifest.get("source_files", [])),
        "detector_registration_policy": manifest.get("registration_policy", "unknown"),
        "detector_combination_mode": manifest.get("combination_mode", "none"),
        "input_dtype": manifest.get("input_dtype", "unknown"),
        "registered_dtype": manifest.get("registered_dtype", "unknown"),
        "coordinate_reference": "local CLS scan coordinates; no CRS assigned",
        "canonical_schema_version": "0.1.0",
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "detector": np.asarray(detector_ids, dtype=str),
            "y": ("y", y, {"long_name": "local scan Y coordinate"}),
            "x": ("x", x, {"long_name": "local scan X coordinate"}),
            "channel": (
                "channel",
                np.arange(channel_count, dtype=np.int32),
                {"long_name": "uncalibrated detector channel index"},
            ),
        },
        attrs=attributes,
    )


def summarize_dataset(dataset: xr.Dataset) -> dict:
    return {
        "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
        "coordinates": list(dataset.coords),
        "data_variables": {
            name: {
                "dimensions": list(variable.dims),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
            }
            for name, variable in dataset.data_vars.items()
        },
        "attributes": dict(dataset.attrs),
    }
