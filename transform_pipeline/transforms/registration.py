from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from transform_pipeline.core.models import RegistrationResult, ShiftMeasurement


DEFAULT_MINIMUM_SHIFT = -2.0
DEFAULT_MAXIMUM_SHIFT = 2.0
DEFAULT_SHIFT_STEP = 0.025
DEFAULT_MAXIMUM_ABSOLUTE_APPLIED_SHIFT = 2.0


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return np.nan if denominator <= 1e-12 else float(np.sum(a * b) / denominator)


def shift_scalar_row(row: np.ndarray, shift: float) -> np.ndarray:
    x = np.arange(row.size, dtype=np.float64)
    return np.interp(x - shift, x, row, left=np.nan, right=np.nan)


def score_shift(
    moving: np.ndarray,
    reference: np.ndarray,
    shift: float,
    edge_margin: int,
) -> float:
    shifted = shift_scalar_row(moving, shift)
    section = slice(edge_margin, moving.size - edge_margin)
    a = shifted[section]
    b = reference[section]
    valid = np.isfinite(a) & np.isfinite(b)
    return np.nan if valid.sum() < 3 else pearson_correlation(a[valid], b[valid])


def quadratic_peak(
    shifts: np.ndarray,
    scores: np.ndarray,
    index: int,
    step: float,
) -> tuple[float, float]:
    if index == 0 or index == scores.size - 1:
        return float(shifts[index]), float(scores[index])

    left, center, right = scores[index - 1:index + 2]
    denominator = left - 2.0 * center + right
    if not np.all(np.isfinite([left, center, right])) or abs(denominator) <= 1e-12:
        return float(shifts[index]), float(scores[index])

    offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    return (
        float(shifts[index] + offset * step),
        float(center - 0.25 * (left - right) * offset),
    )


def gaussian_weighted_smooth(
    values: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)

    full_numerator = np.convolve(values * weights, kernel, mode="full")
    full_denominator = np.convolve(weights, kernel, mode="full")
    start = (kernel.size - 1) // 2
    stop = start + values.size
    numerator = full_numerator[start:stop]
    denominator = full_denominator[start:stop]

    output = np.full(values.shape, np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    output[valid] = numerator[valid] / denominator[valid]
    return output


def measure_subpixel_shifts(
    total_counts: np.ndarray,
    minimum_shift: float = DEFAULT_MINIMUM_SHIFT,
    maximum_shift: float = DEFAULT_MAXIMUM_SHIFT,
    shift_step: float = DEFAULT_SHIFT_STEP,
    edge_margin: int = 16,
    smoothing_sigma_odd_rows: float = 5.0,
    minimum_weight: float = 0.0005,
) -> list[ShiftMeasurement]:
    rows, columns = total_counts.shape
    if minimum_shift >= maximum_shift:
        raise ValueError("minimum_shift must be less than maximum_shift")
    if shift_step <= 0:
        raise ValueError("shift_step must be positive")
    if 2 * edge_margin >= columns:
        raise ValueError("Edge margin removes the entire row")

    candidates = np.arange(
        minimum_shift,
        maximum_shift + 0.5 * shift_step,
        shift_step,
    )
    zero_index = int(np.argmin(np.abs(candidates)))
    odd_rows = np.arange(1, rows - 1, 2)
    raw_shifts: list[float] = []
    peak_scores: list[float] = []
    zero_scores: list[float] = []
    gains: list[float] = []

    for row_index in odd_rows:
        moving = total_counts[row_index].astype(np.float64)
        reference = 0.5 * (
            total_counts[row_index - 1].astype(np.float64)
            + total_counts[row_index + 1].astype(np.float64)
        )
        scores = np.array(
            [score_shift(moving, reference, float(s), edge_margin) for s in candidates]
        )
        best_index = int(np.nanargmax(scores))
        refined_shift, peak_score = quadratic_peak(
            candidates,
            scores,
            best_index,
            shift_step,
        )
        zero_score = float(scores[zero_index])
        raw_shifts.append(refined_shift)
        peak_scores.append(peak_score)
        zero_scores.append(zero_score)
        gains.append(peak_score - zero_score)

    raw_array = np.asarray(raw_shifts)
    gain_array = np.asarray(gains)
    smoothed = gaussian_weighted_smooth(
        raw_array,
        np.maximum(gain_array, minimum_weight),
        smoothing_sigma_odd_rows,
    )

    return [
        ShiftMeasurement(
            row=int(row),
            raw_shift=float(raw),
            smoothed_shift=float(smooth),
            peak_correlation=float(peak),
            zero_correlation=float(zero),
            correlation_gain=float(gain),
        )
        for row, raw, smooth, peak, zero, gain in zip(
            odd_rows,
            raw_array,
            smoothed,
            peak_scores,
            zero_scores,
            gain_array,
        )
    ]


def shift_spectral_row(
    row: np.ndarray,
    shift: float,
    maximum_absolute_shift: float = DEFAULT_MAXIMUM_ABSOLUTE_APPLIED_SHIFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift one spectral row along X without wraparound."""
    if maximum_absolute_shift <= 0:
        raise ValueError("maximum_absolute_shift must be positive")
    if not np.isfinite(shift):
        raise ValueError(f"Invalid nonfinite shift: {shift}")
    if abs(shift) > maximum_absolute_shift:
        raise ValueError(
            f"Shift {shift:+.6f} exceeds the configured application limit "
            f"of +/-{maximum_absolute_shift:.6f} pixels."
        )

    columns, channels = row.shape
    destination = np.arange(columns, dtype=np.float64)
    source_positions = destination - shift
    valid = (source_positions >= 0.0) & (source_positions <= columns - 1)
    output = np.full((columns, channels), np.nan, dtype=np.float32)

    positions = source_positions[valid]
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, columns - 1)
    right_weight = (positions - left).astype(np.float32)
    left_weight = 1.0 - right_weight
    source = np.asarray(row, dtype=np.float32)
    output[valid] = (
        source[left] * left_weight[:, None]
        + source[right] * right_weight[:, None]
    )
    return output, valid


def apply_subpixel_registration(
    cube: np.ndarray,
    shifts: Iterable[ShiftMeasurement],
    maximum_absolute_shift: float = DEFAULT_MAXIMUM_ABSOLUTE_APPLIED_SHIFT,
) -> RegistrationResult:
    """Apply measured odd-row shifts while leaving even anchor rows unchanged."""
    rows, columns, channels = cube.shape
    shift_list = list(shifts)
    by_row = {measurement.row: measurement for measurement in shift_list}
    expected = set(range(1, rows - 1, 2))
    if set(by_row) != expected:
        missing = sorted(expected.difference(by_row))
        extra = sorted(set(by_row).difference(expected))
        raise ValueError(
            "Shift measurements do not cover all interior odd rows. "
            f"Missing={missing}, extra={extra}"
        )

    corrected = np.full((rows, columns, channels), np.nan, dtype=np.float32)
    validity_mask = np.zeros((rows, columns), dtype=bool)

    for row_index in range(rows):
        if row_index % 2 == 0:
            corrected[row_index] = np.asarray(cube[row_index], dtype=np.float32)
            validity_mask[row_index] = True
        else:
            corrected[row_index], validity_mask[row_index] = shift_spectral_row(
                cube[row_index],
                by_row[row_index].smoothed_shift,
                maximum_absolute_shift=maximum_absolute_shift,
            )

    total_counts = np.nansum(
        corrected,
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)
    total_counts[~validity_mask] = np.nan

    return RegistrationResult(
        cube=corrected,
        total_counts=total_counts,
        validity_mask=validity_mask,
        shifts=shift_list,
    )
