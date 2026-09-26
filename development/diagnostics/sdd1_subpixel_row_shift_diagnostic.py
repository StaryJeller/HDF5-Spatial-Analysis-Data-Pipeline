from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

SCAN_DIR = Path(
    r"FILE_PATH"
)
OUTPUT_DIR = SCAN_DIR / "prototype_output"

CUBE_FILE = OUTPUT_DIR / "sdd1_spectral_cube.npy"
CSV_FILE = OUTPUT_DIR / "sdd1_subpixel_row_shift_diagnostic.csv"
SHIFT_PLOT_FILE = OUTPUT_DIR / "sdd1_subpixel_row_shift_profile.png"
HEATMAP_FILE = OUTPUT_DIR / "sdd1_subpixel_shift_score_heatmap.png"
SUMMARY_FILE = OUTPUT_DIR / "sdd1_subpixel_row_shift_summary.txt"

# Even rows remain the fixed reference. Only odd rows are measured.
MIN_SHIFT = -2.0
MAX_SHIFT = 2.0
SHIFT_STEP = 0.025
EDGE_MARGIN_PIXELS = 16

# Weighted Gaussian smoothing along the odd-row sequence.
# This is diagnostic only. It does not modify the spectral cube.
SMOOTHING_SIGMA_ODD_ROWS = 5.0
MIN_GAIN_FOR_SMOOTHING_WEIGHT = 0.0005

LOW_INFORMATION_EPSILON = 1e-12


# ============================================================
# Helpers
# ============================================================

def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator <= LOW_INFORMATION_EPSILON:
        return np.nan

    return float(np.sum(a * b) / denominator)


def interpolate_shifted_row(row: np.ndarray, shift: float) -> np.ndarray:
    """
    Return row sampled after a fractional horizontal shift.

    Positive shift moves image content to the right.
    Negative shift moves image content to the left.
    Samples outside the row are returned as NaN.
    """
    row = np.asarray(row, dtype=np.float64)
    x = np.arange(row.size, dtype=np.float64)
    source_positions = x - shift

    return np.interp(
        source_positions,
        x,
        row,
        left=np.nan,
        right=np.nan,
    )


def score_fractional_shift(
    moving_row: np.ndarray,
    reference_row: np.ndarray,
    shift: float,
) -> float:
    shifted = interpolate_shifted_row(moving_row, shift)

    start = EDGE_MARGIN_PIXELS
    stop = moving_row.size - EDGE_MARGIN_PIXELS

    moving_section = shifted[start:stop]
    reference_section = reference_row[start:stop]

    valid = np.isfinite(moving_section) & np.isfinite(reference_section)
    if np.count_nonzero(valid) < 3:
        return np.nan

    return pearson_correlation(
        moving_section[valid],
        reference_section[valid],
    )


def quadratic_peak_refinement(
    shifts: np.ndarray,
    scores: np.ndarray,
    peak_index: int,
) -> tuple[float, float]:
    """Refine the grid maximum with a local three-point parabola."""
    if peak_index == 0 or peak_index == len(scores) - 1:
        return float(shifts[peak_index]), float(scores[peak_index])

    y_left = scores[peak_index - 1]
    y_center = scores[peak_index]
    y_right = scores[peak_index + 1]

    if not np.all(np.isfinite([y_left, y_center, y_right])):
        return float(shifts[peak_index]), float(scores[peak_index])

    denominator = y_left - 2.0 * y_center + y_right
    if abs(denominator) <= LOW_INFORMATION_EPSILON:
        return float(shifts[peak_index]), float(scores[peak_index])

    offset_in_steps = 0.5 * (y_left - y_right) / denominator
    offset_in_steps = float(np.clip(offset_in_steps, -1.0, 1.0))

    refined_shift = float(shifts[peak_index] + offset_in_steps * SHIFT_STEP)
    refined_score = float(
        y_center - 0.25 * (y_left - y_right) * offset_in_steps
    )

    return refined_shift, refined_score


def gaussian_weighted_smooth(
    values: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("Smoothing sigma must be positive.")

    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)

    weighted_values = np.convolve(values * weights, kernel, mode="same")
    smoothed_weights = np.convolve(weights, kernel, mode="same")

    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = smoothed_weights > LOW_INFORMATION_EPSILON
    result[valid] = weighted_values[valid] / smoothed_weights[valid]
    return result


# ============================================================
# Load the original reconstructed cube
# ============================================================

if not CUBE_FILE.is_file():
    raise FileNotFoundError(
        f"Spectral cube not found: {CUBE_FILE}\n"
        "Run reconstruct_sdd1.py with SAVE_SPECTRAL_CUBE = True."
    )

cube = np.load(CUBE_FILE, mmap_mode="r")
if cube.ndim != 3 or cube.shape[2] != 256:
    raise ValueError(
        f"Expected cube shape (rows, columns, 256), found {cube.shape}."
    )

rows, columns, channels = cube.shape
if rows < 3:
    raise ValueError("At least three raster rows are required.")
if 2 * EDGE_MARGIN_PIXELS >= columns:
    raise ValueError("EDGE_MARGIN_PIXELS removes the entire row.")

summed_image = cube.sum(axis=2, dtype=np.uint64).astype(np.float64)

candidate_shifts = np.arange(
    MIN_SHIFT,
    MAX_SHIFT + 0.5 * SHIFT_STEP,
    SHIFT_STEP,
    dtype=np.float64,
)

zero_index = int(np.argmin(np.abs(candidate_shifts)))
if abs(candidate_shifts[zero_index]) > 1e-12:
    raise RuntimeError("Candidate shift grid does not contain zero.")

# Only interior odd rows have an even row both above and below.
odd_rows = np.arange(1, rows - 1, 2, dtype=np.int64)
score_matrix = np.full(
    (odd_rows.size, candidate_shifts.size),
    np.nan,
    dtype=np.float64,
)
records = []


# ============================================================
# Measure subpixel shifts for odd rows
# ============================================================

for output_index, row_index in enumerate(odd_rows):
    moving_row = summed_image[row_index]
    reference_row = 0.5 * (
        summed_image[row_index - 1] + summed_image[row_index + 1]
    )

    scores = np.array(
        [
            score_fractional_shift(moving_row, reference_row, float(shift))
            for shift in candidate_shifts
        ],
        dtype=np.float64,
    )
    score_matrix[output_index] = scores

    finite = np.isfinite(scores)
    if not np.any(finite):
        grid_shift = np.nan
        refined_shift = np.nan
        peak_score = np.nan
        zero_score = np.nan
        gain = np.nan
        confidence = 0.0
    else:
        valid_indices = np.flatnonzero(finite)
        peak_index = int(valid_indices[np.argmax(scores[finite])])
        grid_shift = float(candidate_shifts[peak_index])
        refined_shift, peak_score = quadratic_peak_refinement(
            candidate_shifts,
            scores,
            peak_index,
        )
        zero_score = float(scores[zero_index])
        gain = peak_score - zero_score

        # Confidence is an audit metric, not a probability. It combines
        # nonnegative correlation gain with local peak curvature.
        if 0 < peak_index < len(scores) - 1:
            neighbors = 0.5 * (scores[peak_index - 1] + scores[peak_index + 1])
            peak_prominence = max(0.0, peak_score - neighbors)
        else:
            peak_prominence = 0.0

        confidence = max(0.0, gain) * (1.0 + peak_prominence)

    records.append(
        {
            "row": int(row_index),
            "grid_shift": grid_shift,
            "raw_fractional_shift": refined_shift,
            "peak_correlation": peak_score,
            "zero_shift_correlation": zero_score,
            "correlation_gain": gain,
            "local_shift_confidence": confidence,
        }
    )

raw_shifts = np.array(
    [record["raw_fractional_shift"] for record in records],
    dtype=np.float64,
)
gains = np.array(
    [record["correlation_gain"] for record in records],
    dtype=np.float64,
)

finite_shifts = np.isfinite(raw_shifts)
weights = np.zeros(raw_shifts.shape, dtype=np.float64)
weights[finite_shifts] = np.maximum(
    gains[finite_shifts],
    MIN_GAIN_FOR_SMOOTHING_WEIGHT,
)

values_for_smoothing = np.where(finite_shifts, raw_shifts, 0.0)
smoothed_shifts = gaussian_weighted_smooth(
    values_for_smoothing,
    weights,
    SMOOTHING_SIGMA_ODD_ROWS,
)

for record, smoothed_shift in zip(records, smoothed_shifts):
    record["smoothed_fractional_shift"] = float(smoothed_shift)


# ============================================================
# Save detailed CSV
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with CSV_FILE.open("w", encoding="utf-8", newline="") as out:
    fieldnames = [
        "row",
        "grid_shift",
        "raw_fractional_shift",
        "smoothed_fractional_shift",
        "peak_correlation",
        "zero_shift_correlation",
        "correlation_gain",
        "local_shift_confidence",
    ]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()

    for record in records:
        output_record = {}
        for field in fieldnames:
            value = record[field]
            if isinstance(value, float):
                output_record[field] = (
                    "nan" if not np.isfinite(value) else f"{value:.10f}"
                )
            else:
                output_record[field] = value
        writer.writerow(output_record)


# ============================================================
# Save shift-profile plot
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

axes[0].plot(
    odd_rows,
    raw_shifts,
    marker=".",
    linestyle="none",
    alpha=0.65,
    label="Raw fractional shift",
)
axes[0].plot(
    odd_rows,
    smoothed_shifts,
    linewidth=2.0,
    label="Weighted Gaussian smooth",
)
axes[0].axhline(0.0, color="black", linewidth=1)
axes[0].set_ylabel("Shift (pixels)")
axes[0].set_title("SDD1 anchored subpixel row-shift diagnostic")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(odd_rows, gains, marker=".", linestyle="none")
axes[1].axhline(0.0, color="black", linewidth=1)
axes[1].set_xlabel("Raster row (odd rows only)")
axes[1].set_ylabel("Correlation gain")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(SHIFT_PLOT_FILE, dpi=160)
plt.close(fig)


# ============================================================
# Save score heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))
image = ax.imshow(
    score_matrix,
    aspect="auto",
    origin="lower",
    extent=(MIN_SHIFT, MAX_SHIFT, odd_rows[0], odd_rows[-1]),
    interpolation="nearest",
)
ax.plot(smoothed_shifts, odd_rows, color="white", linewidth=1.5)
ax.axvline(0.0, color="black", linewidth=1)
ax.set_xlabel("Candidate shift (pixels)")
ax.set_ylabel("Raster row (odd rows only)")
ax.set_title("Correlation score by row and fractional shift")
fig.colorbar(image, ax=ax, label="Pearson correlation")
fig.tight_layout()
fig.savefig(HEATMAP_FILE, dpi=160)
plt.close(fig)


# ============================================================
# Save summary
# ============================================================

valid_raw = raw_shifts[np.isfinite(raw_shifts)]
valid_smooth = smoothed_shifts[np.isfinite(smoothed_shifts)]
valid_gains = gains[np.isfinite(gains)]

summary = {
    "source_cube": str(CUBE_FILE),
    "cube_shape": list(cube.shape),
    "anchor_rows": "even",
    "measured_rows": "odd interior rows",
    "odd_rows_measured": int(odd_rows.size),
    "candidate_shift_min": MIN_SHIFT,
    "candidate_shift_max": MAX_SHIFT,
    "candidate_shift_step": SHIFT_STEP,
    "edge_margin_pixels": EDGE_MARGIN_PIXELS,
    "smoothing_sigma_odd_rows": SMOOTHING_SIGMA_ODD_ROWS,
    "minimum_smoothing_weight": MIN_GAIN_FOR_SMOOTHING_WEIGHT,
    "raw_shift_min": float(np.min(valid_raw)),
    "raw_shift_max": float(np.max(valid_raw)),
    "smoothed_shift_min": float(np.min(valid_smooth)),
    "smoothed_shift_max": float(np.max(valid_smooth)),
    "median_correlation_gain": float(np.median(valid_gains)),
    "diagnostic_only": True,
    "csv": str(CSV_FILE),
    "shift_profile_plot": str(SHIFT_PLOT_FILE),
    "score_heatmap": str(HEATMAP_FILE),
}

with SUMMARY_FILE.open("w", encoding="utf-8") as out:
    out.write("SDD1 SUBPIXEL ROW-SHIFT DIAGNOSTIC SUMMARY\n")
    out.write("==========================================\n\n")
    for key, value in summary.items():
        out.write(f"{key}: {json.dumps(value)}\n")
    out.write("\n")
    out.write("Shift convention: positive moves odd-row content right.\n")
    out.write("No image or spectral-cube correction was applied.\n")

print("Subpixel row-shift diagnostic complete.")
print()
print(f"Cube shape:               {cube.shape}")
print(f"Odd rows measured:        {odd_rows.size}")
print(f"Candidate shift range:    {MIN_SHIFT:+.3f} to {MAX_SHIFT:+.3f}")
print(f"Candidate shift step:     {SHIFT_STEP:.3f}")
print(f"Raw shift range:          {np.min(valid_raw):+.4f} to {np.max(valid_raw):+.4f}")
print(
    f"Smoothed shift range:     "
    f"{np.min(valid_smooth):+.4f} to {np.max(valid_smooth):+.4f}"
)
print()
print(f"CSV:                      {CSV_FILE}")
print(f"Shift profile:            {SHIFT_PLOT_FILE}")
print(f"Score heatmap:            {HEATMAP_FILE}")
print(f"Summary:                  {SUMMARY_FILE}")
