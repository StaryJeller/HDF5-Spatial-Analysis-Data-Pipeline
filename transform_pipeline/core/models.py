from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class RasterGeometry:
    rows: int
    columns: int
    x_axis: np.ndarray
    y_axis: np.ndarray

@dataclass(frozen=True)
class ShiftMeasurement:
    row: int
    raw_shift: float
    smoothed_shift: float
    peak_correlation: float
    zero_correlation: float
    correlation_gain: float

@dataclass
class ReconstructionResult:
    cube: np.ndarray
    total_counts: np.ndarray
    geometry: RasterGeometry

@dataclass
class RegistrationResult:
    cube: np.ndarray
    total_counts: np.ndarray
    validity_mask: np.ndarray
    shifts: list[ShiftMeasurement]

@dataclass
class MultiDetectorReconstructionResult:
    detector_ids: tuple[str, ...]
    source_files: tuple[Path, ...]
    cubes: np.ndarray
    total_counts: np.ndarray
    geometry: RasterGeometry

@dataclass
class MultiDetectorRegistrationResult:
    detector_ids: tuple[str, ...]
    cubes: np.ndarray
    total_counts: np.ndarray
    validity_masks: np.ndarray
    shifts_by_detector: dict[str, list[ShiftMeasurement]]
    combined_cube: np.ndarray | None
    combined_total_counts: np.ndarray | None
    combined_validity_mask: np.ndarray | None
    combination_mode: str
