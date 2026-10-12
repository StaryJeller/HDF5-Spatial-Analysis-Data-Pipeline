import json
from pathlib import Path

import numpy as np
import tifffile
import xarray as xr

from transform_pipeline.interoperability import (
    build_canonical_dataset,
    write_canonical_zarr,
)


def make_output_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "detectors").mkdir()
    (root / "combined").mkdir()
    np.save(root / "x_axis.npy", np.array([0.0, 0.5, 1.0]))
    np.save(root / "y_axis.npy", np.array([10.0, 11.0]))
    manifest = {
        "detector_ids": ["sdd1_0", "sdd2_0"],
        "source_files": ["sdd1_0.bin", "sdd2_0.bin"],
        "combination_mode": "sum",
        "channels": 4,
        "input_dtype": "<u4",
        "registered_dtype": "float32",
        "registration_policy": "independent detector registration",
    }
    (root / "detector_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    for index, detector_id in enumerate(manifest["detector_ids"], start=1):
        values = np.full((2, 3), float(index), dtype=np.float32)
        np.save(root / "detectors" / f"{detector_id}_total_counts_registered.npy", values)
        tifffile.imwrite(
            root / "detectors" / f"{detector_id}_validity_mask.tif",
            np.full((2, 3), 255, dtype=np.uint8),
        )

    np.save(root / "combined" / "combined_total_counts_registered.npy", np.full((2, 3), 3.0, dtype=np.float32))
    tifffile.imwrite(root / "combined" / "combined_validity_mask.tif", np.full((2, 3), 255, dtype=np.uint8))
    return root


def test_build_canonical_dataset(tmp_path: Path) -> None:
    output_dir = make_output_fixture(tmp_path / "output")
    dataset = build_canonical_dataset(output_dir)
    assert dict(dataset.sizes) == {"detector": 2, "y": 2, "x": 3, "channel": 4}
    assert dataset["detector_total_counts_registered"].dims == ("detector", "y", "x")
    assert dataset["combined_total_counts_registered"].dims == ("y", "x")
    assert dataset.detector.values.tolist() == ["sdd1_0", "sdd2_0"]
    assert dataset.attrs["coordinate_reference"] == "local CLS scan coordinates; no CRS assigned"


def test_zarr_roundtrip(tmp_path: Path) -> None:
    output_dir = make_output_fixture(tmp_path / "output")
    dataset = build_canonical_dataset(output_dir)
    store = write_canonical_zarr(dataset, tmp_path / "canonical.zarr")
    reopened = xr.open_zarr(store, consolidated=True)
    try:
        assert dict(reopened.sizes) == dict(dataset.sizes)
        assert set(reopened.data_vars) == set(dataset.data_vars)
        np.testing.assert_allclose(
            reopened["combined_total_counts_registered"].values,
            dataset["combined_total_counts_registered"].values,
        )
    finally:
        reopened.close()
