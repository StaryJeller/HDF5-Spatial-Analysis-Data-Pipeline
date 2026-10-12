from __future__ import annotations

import json
import shutil
from pathlib import Path

import xarray as xr

from .canonical import summarize_dataset


def default_zarr_encoding(dataset: xr.Dataset) -> dict:
    """Return conservative chunks for raster and spectral access patterns."""
    encoding = {}
    for name, variable in dataset.data_vars.items():
        if variable.dims == ("detector", "y", "x", "channel"):
            chunks = (1, min(64, variable.sizes["y"]), min(64, variable.sizes["x"]), min(64, variable.sizes["channel"]))
        elif variable.dims == ("y", "x", "channel"):
            chunks = (min(64, variable.sizes["y"]), min(64, variable.sizes["x"]), min(64, variable.sizes["channel"]))
        elif variable.dims == ("detector", "y", "x"):
            chunks = (1, min(128, variable.sizes["y"]), min(128, variable.sizes["x"]))
        elif variable.dims == ("y", "x"):
            chunks = (min(128, variable.sizes["y"]), min(128, variable.sizes["x"]))
        else:
            continue
        encoding[name] = {"chunks": chunks}
    return encoding


def write_canonical_zarr(
    dataset: xr.Dataset,
    store_path: Path,
    overwrite: bool = False,
) -> Path:
    store_path = Path(store_path)
    if store_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Zarr store already exists: {store_path}. Use --overwrite to replace it."
            )
        if store_path.is_dir():
            shutil.rmtree(store_path)
        else:
            store_path.unlink()

    store_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(
        store=str(store_path),
        mode="w",
        consolidated=True,
        zarr_format=2,
        encoding=default_zarr_encoding(dataset),
    )
    return store_path


def write_dataset_summary(dataset: xr.Dataset, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarize_dataset(dataset), indent=2) + "\n", encoding="utf-8")
    return path
