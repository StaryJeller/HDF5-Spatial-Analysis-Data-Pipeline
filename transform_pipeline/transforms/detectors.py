from __future__ import annotations
import numpy as np
from transform_pipeline.core.models import (
    MultiDetectorReconstructionResult,
    MultiDetectorRegistrationResult,
)
from transform_pipeline.transforms.registration import (
    apply_subpixel_registration,
    measure_subpixel_shifts,
)

VALID_COMBINATION_MODES = ("none", "sum", "mean")


def combine_registered_detectors(
    cubes: np.ndarray,
    validity_masks: np.ndarray,
    mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Fuse detectors only after each detector has been independently registered."""
    if mode not in VALID_COMBINATION_MODES:
        raise ValueError(
            f"Unsupported detector combination mode {mode!r}; "
            f"choose from {VALID_COMBINATION_MODES}."
        )
    if mode == "none":
        return None, None, None

    cubes = np.asarray(cubes, dtype=np.float32)
    masks = np.asarray(validity_masks, dtype=bool)
    if cubes.ndim != 4:
        raise ValueError("cubes must have shape (detector, y, x, channel).")
    if masks.shape != cubes.shape[:3]:
        raise ValueError("validity_masks must match (detector, y, x).")

    # Fusion is restricted to pixels valid for every independently corrected detector.
    combined_mask = np.all(masks, axis=0)
    if mode == "sum":
        combined_cube = np.nansum(cubes, axis=0, dtype=np.float64).astype(np.float32)
    else:
        combined_cube = np.nanmean(cubes, axis=0, dtype=np.float64).astype(np.float32)
    combined_cube[~combined_mask] = np.nan

    combined_total = np.nansum(combined_cube, axis=2, dtype=np.float64).astype(np.float32)
    combined_total[~combined_mask] = np.nan
    return combined_cube, combined_total, combined_mask


def register_detectors(
    reconstruction: MultiDetectorReconstructionResult,
    combination_mode: str = "sum",
) -> MultiDetectorRegistrationResult:
    """Measure and apply an independent subpixel profile for each detector."""
    registered_cubes: list[np.ndarray] = []
    registered_totals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    shifts_by_detector = {}

    for detector_id, detector_cube, detector_total in zip(
        reconstruction.detector_ids,
        reconstruction.cubes,
        reconstruction.total_counts,
    ):
        detector_shifts = measure_subpixel_shifts(detector_total)
        registered = apply_subpixel_registration(detector_cube, detector_shifts)
        shifts_by_detector[detector_id] = detector_shifts
        registered_cubes.append(registered.cube)
        registered_totals.append(registered.total_counts)
        masks.append(registered.validity_mask)

    cube_stack = np.stack(registered_cubes, axis=0)
    total_stack = np.stack(registered_totals, axis=0)
    mask_stack = np.stack(masks, axis=0)
    combined_cube, combined_total, combined_mask = combine_registered_detectors(
        cube_stack,
        mask_stack,
        combination_mode,
    )

    return MultiDetectorRegistrationResult(
        detector_ids=reconstruction.detector_ids,
        cubes=cube_stack,
        total_counts=total_stack,
        validity_masks=mask_stack,
        shifts_by_detector=shifts_by_detector,
        combined_cube=combined_cube,
        combined_total_counts=combined_total,
        combined_validity_mask=combined_mask,
        combination_mode=combination_mode,
    )
