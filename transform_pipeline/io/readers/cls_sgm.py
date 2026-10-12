from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from transform_pipeline.core.models import (
    MultiDetectorReconstructionResult,
    ReconstructionResult,
)
from transform_pipeline.transforms.raster import (
    deserpentine,
    infer_serpentine_geometry,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_playtest_scan_dir() -> Path:
    return repository_root() / "Playtest_model" / "Testmodel_NCo_7_SW"


def resolve_scan_dir(scan_dir=None) -> Path:
    path = default_playtest_scan_dir() if scan_dir is None else Path(scan_dir)
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Scan directory not found: {path}")
    return path


def find_hdf5_file(scan_dir, filename: str | None = None) -> Path:
    scan_dir = resolve_scan_dir(scan_dir)
    if filename:
        path = scan_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    matches = sorted(scan_dir.glob("*.h5"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one .h5 file in {scan_dir}; found {len(matches)}. "
            "Use --h5 to select one explicitly."
        )
    return matches[0]


def detector_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"sdd(\d+)", path.stem, flags=re.IGNORECASE)
    detector_number = int(match.group(1)) if match else 10**9
    return detector_number, path.name.lower()


def discover_sdd_files(
    scan_dir,
    pattern: str = "sdd*_0.bin",
) -> tuple[Path, ...]:
    scan_dir = resolve_scan_dir(scan_dir)
    matches = tuple(sorted(scan_dir.glob(pattern), key=detector_sort_key))
    if not matches:
        raise FileNotFoundError(
            f"No SDD files matching {pattern!r} were found in {scan_dir}."
        )
    return matches


def load_coordinates(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        x = np.asarray(h5["/hexapod_waves/x"][:], dtype=np.float64)
        y = np.asarray(h5["/hexapod_waves/y"][:], dtype=np.float64)
    return x, y


def read_sdd_cube(
    path: Path,
    geometry,
    channels: int = 256,
    dtype: str = "<u4",
) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.dtype(dtype))
    expected = geometry.rows * geometry.columns * channels
    if raw.size != expected:
        raise ValueError(
            f"{path.name}: expected {expected} values for "
            f"{geometry.rows} x {geometry.columns} x {channels}, found {raw.size}."
        )
    chronological = raw.reshape(geometry.rows, geometry.columns, channels)
    return deserpentine(chronological)


def reconstruct_scan(
    h5_path: Path,
    bin_path: Path,
    channels: int = 256,
    dtype: str = "<u4",
) -> ReconstructionResult:
    x, y = load_coordinates(h5_path)
    geometry = infer_serpentine_geometry(x, y)
    cube = read_sdd_cube(bin_path, geometry, channels, dtype)
    return ReconstructionResult(
        cube=cube,
        total_counts=cube.sum(axis=2, dtype=np.uint64),
        geometry=geometry,
    )


def reconstruct_detectors(
    h5_path: Path,
    detector_files: Sequence[Path],
    channels: int = 256,
    dtype: str = "<u4",
) -> MultiDetectorReconstructionResult:
    """Reconstruct multiple co-registered detector files on one raster geometry."""
    detector_files = tuple(Path(path) for path in detector_files)
    if not detector_files:
        raise ValueError("At least one detector file is required.")

    x, y = load_coordinates(h5_path)
    geometry = infer_serpentine_geometry(x, y)
    cubes = []
    detector_ids = []

    for path in detector_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        cube = read_sdd_cube(path, geometry, channels=channels, dtype=dtype)
        cubes.append(cube)
        detector_ids.append(path.stem)

    stacked = np.stack(cubes, axis=0)
    totals = stacked.sum(axis=3, dtype=np.uint64)
    return MultiDetectorReconstructionResult(
        detector_ids=tuple(detector_ids),
        source_files=detector_files,
        cubes=stacked,
        total_counts=totals,
        geometry=geometry,
    )
