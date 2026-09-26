from pathlib import Path

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
CSV_FILE = OUTPUT_DIR / "sdd1_row_shift_diagnostic.csv"
SUMMARY_FILE = OUTPUT_DIR / "sdd1_row_shift_summary.txt"
PLOT_FILE = OUTPUT_DIR / "sdd1_row_shift_diagnostic.png"

MAX_SHIFT_PIXELS = 6
EDGE_MARGIN_PIXELS = 12
MIN_STANDARD_DEVIATION = 1e-12

# Use adjacent rows to construct the reference for each interior row.
# With 301 raster rows, rows 1 through 299 can be evaluated.
FIRST_ROW_TO_TEST = 1
LAST_ROW_TO_TEST = None


# ============================================================
# Helpers
# ============================================================

def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for a constant input."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    a_norm = np.sqrt(np.sum(a_centered * a_centered))
    b_norm = np.sqrt(np.sum(b_centered * b_centered))

    if a_norm <= MIN_STANDARD_DEVIATION:
        return np.nan

    if b_norm <= MIN_STANDARD_DEVIATION:
        return np.nan

    return float(
        np.sum(a_centered * b_centered) / (a_norm * b_norm)
    )


def overlapping_vectors(
    moving_row: np.ndarray,
    reference_row: np.ndarray,
    shift: int,
    edge_margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return comparable samples after shifting moving_row horizontally.

    Shift convention:
        positive shift = move moving_row to the right
        negative shift = move moving_row to the left

    No circular wrapping is used. Only overlapping samples are compared.
    """
    columns = moving_row.size

    if edge_margin < 0:
        raise ValueError("EDGE_MARGIN_PIXELS cannot be negative.")

    if 2 * edge_margin >= columns:
        raise ValueError(
            "EDGE_MARGIN_PIXELS removes the entire row. "
            f"Margin={edge_margin}, columns={columns}"
        )

    if shift >= 0:
        moving_start = edge_margin
        moving_stop = columns - edge_margin - shift
        reference_start = edge_margin + shift
        reference_stop = columns - edge_margin
    else:
        offset = -shift
        moving_start = edge_margin + offset
        moving_stop = columns - edge_margin
        reference_start = edge_margin
        reference_stop = columns - edge_margin - offset

    moving = moving_row[moving_start:moving_stop]
    reference = reference_row[reference_start:reference_stop]

    if moving.size != reference.size:
        raise RuntimeError(
            "Internal overlap error: moving and reference lengths differ."
        )

    if moving.size < 3:
        raise ValueError(
            "Too few overlapping pixels remain for correlation."
        )

    return moving, reference


def score_shift(
    moving_row: np.ndarray,
    reference_row: np.ndarray,
    shift: int,
) -> float:
    """Score one candidate integer shift using Pearson correlation."""
    moving, reference = overlapping_vectors(
        moving_row=moving_row,
        reference_row=reference_row,
        shift=shift,
        edge_margin=EDGE_MARGIN_PIXELS,
    )

    return pearson_correlation(moving, reference)


def summarize_group(name: str, records: list[dict]) -> list[str]:
    """Return text-summary lines for a group of row records."""
    valid = [record for record in records if np.isfinite(record["best_score"])]

    lines = [f"{name} rows evaluated: {len(records)}"]
    lines.append(f"{name} rows with valid scores: {len(valid)}")

    if not valid:
        lines.append(f"{name} median best shift: unavailable")
        return lines

    shifts = np.array([record["best_shift"] for record in valid], dtype=np.int64)
    scores = np.array([record["best_score"] for record in valid], dtype=np.float64)
    improvements = np.array(
        [record["score_improvement"] for record in valid],
        dtype=np.float64,
    )

    unique_shifts, counts = np.unique(shifts, return_counts=True)
    distribution = ", ".join(
        f"{int(shift):+d}:{int(count)}"
        for shift, count in zip(unique_shifts, counts)
    )

    lines.extend(
        [
            f"{name} median best shift: {float(np.median(shifts)):+.3f}",
            f"{name} mean best shift: {float(np.mean(shifts)):+.3f}",
            f"{name} median best score: {float(np.median(scores)):.6f}",
            (
                f"{name} median score improvement over zero shift: "
                f"{float(np.median(improvements)):.6f}"
            ),
            f"{name} shift distribution (shift:count): {distribution}",
        ]
    )

    return lines


# ============================================================
# Load the reconstructed spectral cube and summed image
# ============================================================

if not CUBE_FILE.is_file():
    raise FileNotFoundError(
        "The reconstructed spectral cube was not found.\n"
        f"Expected: {CUBE_FILE}\n"
        "Run reconstruct_sdd1.py with SAVE_SPECTRAL_CUBE = True."
    )

cube = np.load(CUBE_FILE, mmap_mode="r")

if cube.ndim != 3:
    raise ValueError(
        f"Expected a three-dimensional cube, found shape {cube.shape}."
    )

rows, columns, channels = cube.shape

if channels != 256:
    raise ValueError(
        f"Expected 256 spectral channels, found {channels}."
    )

if rows < 3:
    raise ValueError("At least three raster rows are required.")

summed_image = cube.sum(axis=2, dtype=np.uint64).astype(np.float64)


# ============================================================
# Determine the tested row range
# ============================================================

first_row = max(1, FIRST_ROW_TO_TEST)

if LAST_ROW_TO_TEST is None:
    last_row = rows - 2
else:
    last_row = min(rows - 2, LAST_ROW_TO_TEST)

if first_row > last_row:
    raise ValueError(
        f"Invalid tested row range: {first_row} through {last_row}."
    )

candidate_shifts = np.arange(
    -MAX_SHIFT_PIXELS,
    MAX_SHIFT_PIXELS + 1,
    dtype=np.int64,
)


# ============================================================
# Estimate the best integer shift for each interior row
# ============================================================

records = []
score_matrix = []

for row_index in range(first_row, last_row + 1):
    moving_row = summed_image[row_index]

    # The mean of the rows immediately above and below provides a local
    # reference while avoiding use of the row being measured.
    reference_row = 0.5 * (
        summed_image[row_index - 1] + summed_image[row_index + 1]
    )

    scores = np.array(
        [
            score_shift(
                moving_row=moving_row,
                reference_row=reference_row,
                shift=int(shift),
            )
            for shift in candidate_shifts
        ],
        dtype=np.float64,
    )

    score_matrix.append(scores)

    finite_mask = np.isfinite(scores)

    if not np.any(finite_mask):
        best_shift = 0
        best_score = np.nan
        zero_shift_score = np.nan
        score_improvement = np.nan
    else:
        finite_indices = np.flatnonzero(finite_mask)
        best_relative_index = int(np.argmax(scores[finite_mask]))
        best_index = int(finite_indices[best_relative_index])

        best_shift = int(candidate_shifts[best_index])
        best_score = float(scores[best_index])

        zero_index = int(np.flatnonzero(candidate_shifts == 0)[0])
        zero_shift_score = float(scores[zero_index])
        score_improvement = best_score - zero_shift_score

    records.append(
        {
            "row": row_index,
            "parity": "even" if row_index % 2 == 0 else "odd",
            "best_shift": best_shift,
            "best_score": best_score,
            "zero_shift_score": zero_shift_score,
            "score_improvement": score_improvement,
        }
    )

score_matrix = np.asarray(score_matrix, dtype=np.float64)


# ============================================================
# Write detailed CSV
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with CSV_FILE.open("w", encoding="utf-8") as out:
    score_headers = ",".join(
        f"score_shift_{int(shift):+d}"
        for shift in candidate_shifts
    )

    out.write(
        "row,parity,best_shift,best_score,zero_shift_score,"
        f"score_improvement,{score_headers}\n"
    )

    for record, scores in zip(records, score_matrix):
        score_text = ",".join(
            "nan" if not np.isfinite(score) else f"{score:.10f}"
            for score in scores
        )

        best_score_text = (
            "nan"
            if not np.isfinite(record["best_score"])
            else f"{record['best_score']:.10f}"
        )

        zero_score_text = (
            "nan"
            if not np.isfinite(record["zero_shift_score"])
            else f"{record['zero_shift_score']:.10f}"
        )

        improvement_text = (
            "nan"
            if not np.isfinite(record["score_improvement"])
            else f"{record['score_improvement']:.10f}"
        )

        out.write(
            f"{record['row']},"
            f"{record['parity']},"
            f"{record['best_shift']},"
            f"{best_score_text},"
            f"{zero_score_text},"
            f"{improvement_text},"
            f"{score_text}\n"
        )


# ============================================================
# Write summary
# ============================================================

even_records = [record for record in records if record["parity"] == "even"]
odd_records = [record for record in records if record["parity"] == "odd"]

summary_lines = [
    "SDD1 ROW-SHIFT DIAGNOSTIC SUMMARY",
    "=================================",
    "",
    f"Source cube: {CUBE_FILE}",
    f"Cube shape: {cube.shape}",
    f"Summed image shape: {summed_image.shape}",
    f"Rows tested: {first_row} through {last_row}",
    f"Candidate shifts: {-MAX_SHIFT_PIXELS} through {MAX_SHIFT_PIXELS}",
    f"Edge margin excluded from scoring: {EDGE_MARGIN_PIXELS} pixels",
    "Shift convention: positive moves the tested row to the right",
    "Reference: mean of the immediately adjacent rows",
    "Scoring method: Pearson correlation on overlapping pixels",
    "",
]

summary_lines.extend(summarize_group("All", records))
summary_lines.append("")
summary_lines.extend(summarize_group("Even", even_records))
summary_lines.append("")
summary_lines.extend(summarize_group("Odd", odd_records))
summary_lines.extend(
    [
        "",
        "Interpretation caution:",
        "This diagnostic estimates integer row offsets only.",
        "A high best score with negligible improvement over zero shift does not",
        "justify applying the nonzero shift. Inspect score improvement, parity",
        "consistency, the diagnostic plot, and the reconstructed image together.",
    ]
)

SUMMARY_FILE.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)


# ============================================================
# Create a diagnostic plot
# ============================================================

row_numbers = np.array([record["row"] for record in records], dtype=np.int64)
best_shifts = np.array(
    [record["best_shift"] for record in records],
    dtype=np.int64,
)
best_scores = np.array(
    [record["best_score"] for record in records],
    dtype=np.float64,
)
improvements = np.array(
    [record["score_improvement"] for record in records],
    dtype=np.float64,
)

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10, 10),
    sharex=True,
)

axes[0].plot(row_numbers, best_shifts, marker=".", linestyle="none")
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_ylabel("Best shift (pixels)")
axes[0].set_title("SDD1 row-shift diagnostic")
axes[0].grid(True, alpha=0.3)

axes[1].plot(row_numbers, best_scores, marker=".", linestyle="none")
axes[1].set_ylabel("Best correlation")
axes[1].grid(True, alpha=0.3)

axes[2].plot(row_numbers, improvements, marker=".", linestyle="none")
axes[2].axhline(0, color="black", linewidth=1)
axes[2].set_xlabel("Raster row")
axes[2].set_ylabel("Score improvement")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(PLOT_FILE, dpi=150)
plt.close(fig)


# ============================================================
# Console summary
# ============================================================

print("Row-shift diagnostic complete.")
print()
print(f"Cube shape:          {cube.shape}")
print(f"Rows tested:         {first_row} through {last_row}")
print(f"Candidate shifts:    {-MAX_SHIFT_PIXELS} through {MAX_SHIFT_PIXELS}")
print(f"CSV:                 {CSV_FILE}")
print(f"Summary:             {SUMMARY_FILE}")
print(f"Plot:                {PLOT_FILE}")
print()

for line in summarize_group("Even", even_records):
    print(line)

print()

for line in summarize_group("Odd", odd_records):
    print(line)
