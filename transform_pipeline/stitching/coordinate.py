from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class TilePlacement:
    tile_id: str
    store_path: Path
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_snap_residual: float = 0.0
    y_snap_residual: float = 0.0
    x_snap_fraction: float = 0.0
    y_snap_fraction: float = 0.0


@dataclass(frozen=True)
class StitchPlan:
    x: np.ndarray
    y: np.ndarray
    channel: np.ndarray
    x_spacing: float
    y_spacing: float
    placements: tuple[TilePlacement, ...]


def _regular_spacing(
    values: np.ndarray,
    name: str,
    rtol: float,
    atol: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            f"{name} must be a one-dimensional coordinate with at least two values."
        )
    differences = np.diff(values)
    if not np.all(differences > 0):
        raise ValueError(f"{name} coordinates must be strictly increasing.")
    spacing = float(np.median(differences))
    if not np.allclose(differences, spacing, rtol=rtol, atol=atol):
        raise ValueError(f"{name} coordinates are not regularly spaced.")
    return spacing


def _snap_indices(
    values: np.ndarray,
    origin: float,
    spacing: float,
    coordinate_tolerance: float,
    axis_name: str,
    maximum_grid_offset_fraction: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Place a regular coordinate axis on the nearest common-grid cells.

    No image values are interpolated. The complete axis is shifted by the
    nearest integer number of grid cells. The maximum coordinate residual and
    its fraction of one pixel are returned for audit and testing.
    """
    if not 0.0 <= maximum_grid_offset_fraction <= 0.5:
        raise ValueError(
            "maximum_grid_offset_fraction must be between 0 and 0.5."
        )

    values = np.asarray(values, dtype=np.float64)
    fractional = (values - origin) / spacing
    indices = np.rint(fractional).astype(np.int64)
    snapped = origin + indices * spacing
    residuals = values - snapped
    maximum_residual = float(np.max(np.abs(residuals)))
    maximum_fraction = maximum_residual / abs(spacing)
    allowed = (
        maximum_grid_offset_fraction * abs(spacing)
        + coordinate_tolerance
    )

    if maximum_residual > allowed:
        raise ValueError(
            f"{axis_name} coordinates exceed nearest-grid placement limits; "
            f"maximum residual={maximum_residual:.12g}, "
            f"fraction of cell={maximum_fraction:.6f}, "
            f"allowed={allowed:.12g}."
        )

    expected = np.arange(indices[0], indices[0] + indices.size)
    if not np.array_equal(indices, expected):
        raise ValueError(
            f"{axis_name} coordinates do not map to contiguous grid cells."
        )

    return indices, maximum_residual, maximum_fraction


def build_stitch_plan(
    store_paths: Sequence[Path],
    coordinate_tolerance: float = 1e-6,
    maximum_grid_offset_fraction: float = 0.5,
) -> StitchPlan:
    """Build a union-grid plan using nearest-cell coordinate placement."""
    if len(store_paths) < 2:
        raise ValueError(
            "Coordinate stitching requires at least two canonical Zarr stores."
        )

    metadata = []
    reference_dx = reference_dy = None
    reference_channel = None

    for index, raw_path in enumerate(store_paths):
        path = Path(raw_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)

        dataset = xr.open_zarr(
            str(path),
            consolidated=True,
            chunks=None,
        )
        try:
            required = {
                "combined_total_counts_registered",
                "combined_validity_mask",
            }
            missing = required.difference(dataset.data_vars)
            if missing:
                raise ValueError(
                    f"{path} is missing variables: {sorted(missing)}"
                )

            x = np.asarray(dataset.x.values, dtype=np.float64)
            y = np.asarray(dataset.y.values, dtype=np.float64)
            channel = np.asarray(dataset.channel.values, dtype=np.int32)
            dx = _regular_spacing(
                x,
                f"x for tile {index}",
                1e-6,
                1e-12,
            )
            dy = _regular_spacing(
                y,
                f"y for tile {index}",
                1e-6,
                1e-12,
            )

            if reference_dx is None:
                reference_dx = dx
                reference_dy = dy
                reference_channel = channel
            else:
                if not np.isclose(
                    dx,
                    reference_dx,
                    rtol=1e-6,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"X spacing mismatch for {path}: "
                        f"{dx} versus {reference_dx}."
                    )
                if not np.isclose(
                    dy,
                    reference_dy,
                    rtol=1e-6,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Y spacing mismatch for {path}: "
                        f"{dy} versus {reference_dy}."
                    )
                if not np.array_equal(channel, reference_channel):
                    raise ValueError(
                        f"Channel coordinate mismatch for {path}."
                    )

            tile_id = (
                path.parent.name
                if path.name == "canonical_sgm.zarr"
                else path.stem
            )
            metadata.append((tile_id, path, x, y))
        finally:
            dataset.close()

    x_origin = min(float(item[2][0]) for item in metadata)
    y_origin = min(float(item[3][0]) for item in metadata)
    placements = []
    maximum_x_index = 0
    maximum_y_index = 0

    for tile_id, path, x, y in metadata:
        x_indices, x_residual, x_fraction = _snap_indices(
            x,
            x_origin,
            reference_dx,
            coordinate_tolerance,
            "X",
            maximum_grid_offset_fraction,
        )
        y_indices, y_residual, y_fraction = _snap_indices(
            y,
            y_origin,
            reference_dy,
            coordinate_tolerance,
            "Y",
            maximum_grid_offset_fraction,
        )

        placement = TilePlacement(
            tile_id=tile_id,
            store_path=path,
            y_start=int(y_indices[0]),
            y_stop=int(y_indices[-1] + 1),
            x_start=int(x_indices[0]),
            x_stop=int(x_indices[-1] + 1),
            x_min=float(x[0]),
            x_max=float(x[-1]),
            y_min=float(y[0]),
            y_max=float(y[-1]),
            x_snap_residual=x_residual,
            y_snap_residual=y_residual,
            x_snap_fraction=x_fraction,
            y_snap_fraction=y_fraction,
        )
        placements.append(placement)
        maximum_x_index = max(
            maximum_x_index,
            placement.x_stop - 1,
        )
        maximum_y_index = max(
            maximum_y_index,
            placement.y_stop - 1,
        )

    union_x = (
        x_origin
        + np.arange(maximum_x_index + 1, dtype=np.float64)
        * reference_dx
    )
    union_y = (
        y_origin
        + np.arange(maximum_y_index + 1, dtype=np.float64)
        * reference_dy
    )

    return StitchPlan(
        x=union_x,
        y=union_y,
        channel=np.asarray(reference_channel, dtype=np.int32),
        x_spacing=float(reference_dx),
        y_spacing=float(reference_dy),
        placements=tuple(placements),
    )
