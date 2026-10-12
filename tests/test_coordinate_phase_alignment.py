from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from transform_pipeline.stitching.coordinate import build_stitch_plan


def make_tile(path: Path, x, y) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.ones((y.size, x.size), dtype=np.float32)
    dataset = xr.Dataset(
        {
            "combined_total_counts_registered": (
                ("y", "x"),
                values,
            ),
            "combined_validity_mask": (
                ("y", "x"),
                np.ones_like(values, dtype=bool),
            ),
        },
        coords={
            "x": x,
            "y": y,
            "channel": np.arange(2, dtype=np.int32),
        },
    )
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        zarr_format=2,
    )


def test_accepts_observed_sub_half_cell_y_phase(tmp_path: Path) -> None:
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    spacing = 0.0183333333333333
    offset = 0.00833333333328
    make_tile(
        first,
        [0.0, 1.0, 2.0],
        [0.0, spacing, 2 * spacing],
    )
    make_tile(
        second,
        [2.0, 3.0, 4.0],
        [offset, offset + spacing, offset + 2 * spacing],
    )

    plan = build_stitch_plan([first, second])

    assert plan.placements[1].y_snap_residual == pytest.approx(offset)
    assert plan.placements[1].y_snap_fraction < 0.5


def test_rejects_offset_when_limit_is_tighter(tmp_path: Path) -> None:
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    make_tile(first, [0, 1], [0, 1])
    make_tile(second, [1, 2], [0.45, 1.45])

    with pytest.raises(
        ValueError,
        match="nearest-grid placement limits",
    ):
        build_stitch_plan(
            [first, second],
            maximum_grid_offset_fraction=0.25,
        )
