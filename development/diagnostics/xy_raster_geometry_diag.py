from pathlib import Path

import h5py
import numpy as np


# ============================================================
# Configuration
# ============================================================

SCAN_DIR = Path(
    r"FILE_PATH"
)

H5_FILE = (
    SCAN_DIR
    / ".h5 file from directory"
)

OUTPUT = SCAN_DIR / "xy_raster_geometry_diagnostic.txt"

# Floating-point tolerance used when deciding whether Y changed.
#
# We do NOT yet know the exact precision/noise characteristics
# of the coordinate stream, so the script reports results for
# several tolerances instead of silently choosing one.
Y_TOLERANCES = [
    1e-12,
    1e-9,
    1e-6,
    1e-4,
]

# Tolerance used for the detailed line segmentation.
WORKING_Y_TOLERANCE = 1e-6

# Number of scan lines to print in detail.
PREVIEW_LINES = 20


# ============================================================
# Helpers
# ============================================================

def write_section(out, title):
    out.write("\n")
    out.write("=" * 72 + "\n")
    out.write(title + "\n")
    out.write("=" * 72 + "\n")


def describe_array(array):
    return {
        "shape": array.shape,
        "dtype": array.dtype,
        "min": np.min(array),
        "max": np.max(array),
        "mean": np.mean(array),
    }


def find_line_boundaries(y_values, tolerance):
    """
    Define a candidate line boundary wherever consecutive Y
    coordinates differ by more than the specified tolerance.

    Returns an array of start indices plus the final array length.
    """
    dy = np.diff(y_values)

    change_indices = np.flatnonzero(
        np.abs(dy) > tolerance
    ) + 1

    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            change_indices,
            np.array([len(y_values)], dtype=np.int64),
        )
    )

    return boundaries


def direction_from_x(x_line):
    """
    Describe overall X direction based only on the line endpoints.
    """
    if len(x_line) < 2:
        return "single-point"

    delta = x_line[-1] - x_line[0]

    if delta > 0:
        return "increasing"

    if delta < 0:
        return "decreasing"

    return "same-endpoints"


# ============================================================
# Load coordinates
# ============================================================

with h5py.File(H5_FILE, "r") as h5:

    x = np.asarray(
        h5["/hexapod_waves/x"][:],
        dtype=np.float64
    )

    y = np.asarray(
        h5["/hexapod_waves/y"][:],
        dtype=np.float64
    )


if x.shape != y.shape:
    raise ValueError(
        "X and Y coordinate arrays do not have matching shapes.\n"
        f"X shape: {x.shape}\n"
        f"Y shape: {y.shape}"
    )


if x.ndim != 1:
    raise ValueError(
        "Expected one-dimensional coordinate arrays.\n"
        f"Observed shape: {x.shape}"
    )


# ============================================================
# Basic differences
# ============================================================

dx = np.diff(x)
dy = np.diff(y)

x_info = describe_array(x)
y_info = describe_array(y)


# ============================================================
# Working line segmentation
# ============================================================

boundaries = find_line_boundaries(
    y,
    WORKING_Y_TOLERANCE
)

line_starts = boundaries[:-1]
line_stops = boundaries[1:]

line_lengths = line_stops - line_starts

number_of_candidate_lines = len(line_lengths)


# ============================================================
# Per-line summaries
# ============================================================

line_summaries = []

for line_number, (start, stop) in enumerate(
    zip(line_starts, line_stops)
):

    x_line = x[start:stop]
    y_line = y[start:stop]

    if len(x_line) > 1:
        line_dx = np.diff(x_line)

        median_dx = np.median(line_dx)
        min_dx = np.min(line_dx)
        max_dx = np.max(line_dx)

        positive_dx = np.count_nonzero(line_dx > 0)
        negative_dx = np.count_nonzero(line_dx < 0)
        zero_dx = np.count_nonzero(line_dx == 0)

    else:
        median_dx = np.nan
        min_dx = np.nan
        max_dx = np.nan

        positive_dx = 0
        negative_dx = 0
        zero_dx = 0

    summary = {
        "line": line_number,
        "start": start,
        "stop": stop,
        "length": len(x_line),

        "x_start": x_line[0],
        "x_end": x_line[-1],
        "x_min": np.min(x_line),
        "x_max": np.max(x_line),

        "y_start": y_line[0],
        "y_end": y_line[-1],
        "y_min": np.min(y_line),
        "y_max": np.max(y_line),

        "direction": direction_from_x(x_line),

        "median_dx": median_dx,
        "min_dx": min_dx,
        "max_dx": max_dx,

        "positive_dx": positive_dx,
        "negative_dx": negative_dx,
        "zero_dx": zero_dx,
    }

    line_summaries.append(summary)


# ============================================================
# Detect unusual line lengths
# ============================================================

unique_line_lengths, line_length_counts = np.unique(
    line_lengths,
    return_counts=True
)

median_line_length = np.median(line_lengths)


# ============================================================
# Write diagnostic
# ============================================================

with OUTPUT.open("w", encoding="utf-8") as out:

    write_section(
        out,
        "X/Y RASTER-GEOMETRY DIAGNOSTIC"
    )

    out.write(f"HDF5 file: {H5_FILE}\n")
    out.write(
        f"Number of coordinate samples: {len(x)}\n"
    )


    # --------------------------------------------------------
    # Coordinate ranges
    # --------------------------------------------------------

    write_section(
        out,
        "COORDINATE SUMMARY"
    )

    out.write(
        f"X shape: {x_info['shape']}\n"
    )

    out.write(
        f"X dtype: {x_info['dtype']}\n"
    )

    out.write(
        f"X minimum: {x_info['min']:.12g}\n"
    )

    out.write(
        f"X maximum: {x_info['max']:.12g}\n"
    )

    out.write(
        f"X mean: {x_info['mean']:.12g}\n\n"
    )

    out.write(
        f"Y shape: {y_info['shape']}\n"
    )

    out.write(
        f"Y dtype: {y_info['dtype']}\n"
    )

    out.write(
        f"Y minimum: {y_info['min']:.12g}\n"
    )

    out.write(
        f"Y maximum: {y_info['max']:.12g}\n"
    )

    out.write(
        f"Y mean: {y_info['mean']:.12g}\n"
    )


    # --------------------------------------------------------
    # Global coordinate differences
    # --------------------------------------------------------

    write_section(
        out,
        "CONSECUTIVE COORDINATE DIFFERENCES"
    )

    out.write(
        f"dx minimum: {np.min(dx):.12g}\n"
    )

    out.write(
        f"dx maximum: {np.max(dx):.12g}\n"
    )

    out.write(
        f"dx median: {np.median(dx):.12g}\n"
    )

    out.write(
        f"dx == 0 count: {np.count_nonzero(dx == 0)}\n"
    )

    out.write(
        f"dx > 0 count: {np.count_nonzero(dx > 0)}\n"
    )

    out.write(
        f"dx < 0 count: {np.count_nonzero(dx < 0)}\n\n"
    )

    out.write(
        f"dy minimum: {np.min(dy):.12g}\n"
    )

    out.write(
        f"dy maximum: {np.max(dy):.12g}\n"
    )

    out.write(
        f"dy median: {np.median(dy):.12g}\n"
    )

    out.write(
        f"dy == 0 count: {np.count_nonzero(dy == 0)}\n"
    )

    out.write(
        f"dy > 0 count: {np.count_nonzero(dy > 0)}\n"
    )

    out.write(
        f"dy < 0 count: {np.count_nonzero(dy < 0)}\n"
    )


    # --------------------------------------------------------
    # Sensitivity to Y tolerance
    # --------------------------------------------------------

    write_section(
        out,
        "CANDIDATE LINE COUNT BY Y TOLERANCE"
    )

    out.write(
        "A candidate line boundary is defined here as a change "
        "in Y larger than the stated tolerance.\n\n"
    )

    for tolerance in Y_TOLERANCES:

        test_boundaries = find_line_boundaries(
            y,
            tolerance
        )

        test_lengths = np.diff(
            test_boundaries
        )

        out.write(
            f"Tolerance {tolerance:.1e}: "
            f"{len(test_lengths)} candidate segments\n"
        )

        out.write(
            f"    minimum segment length: "
            f"{np.min(test_lengths)}\n"
        )

        out.write(
            f"    maximum segment length: "
            f"{np.max(test_lengths)}\n"
        )

        out.write(
            f"    median segment length: "
            f"{np.median(test_lengths):.3f}\n"
        )


    # --------------------------------------------------------
    # Working segmentation
    # --------------------------------------------------------

    write_section(
        out,
        "WORKING LINE SEGMENTATION"
    )

    out.write(
        f"Working Y tolerance: "
        f"{WORKING_Y_TOLERANCE:.1e}\n"
    )

    out.write(
        f"Candidate scan lines: "
        f"{number_of_candidate_lines}\n"
    )

    out.write(
        f"Minimum line length: "
        f"{np.min(line_lengths)}\n"
    )

    out.write(
        f"Maximum line length: "
        f"{np.max(line_lengths)}\n"
    )

    out.write(
        f"Median line length: "
        f"{median_line_length:.3f}\n\n"
    )

    out.write(
        "Unique line lengths and frequencies:\n"
    )

    for length, count in zip(
        unique_line_lengths,
        line_length_counts
    ):

        out.write(
            f"    {length}: {count}\n"
        )


    # --------------------------------------------------------
    # First candidate lines
    # --------------------------------------------------------

    write_section(
        out,
        f"FIRST {PREVIEW_LINES} CANDIDATE LINES"
    )

    out.write(
        "line,start,stop,length,"
        "x_start,x_end,x_min,x_max,"
        "y_start,y_end,y_min,y_max,"
        "direction,median_dx,min_dx,max_dx,"
        "positive_dx,negative_dx,zero_dx\n"
    )

    for summary in line_summaries[:PREVIEW_LINES]:

        out.write(
            f"{summary['line']},"
            f"{summary['start']},"
            f"{summary['stop']},"
            f"{summary['length']},"
            f"{summary['x_start']:.10g},"
            f"{summary['x_end']:.10g},"
            f"{summary['x_min']:.10g},"
            f"{summary['x_max']:.10g},"
            f"{summary['y_start']:.10g},"
            f"{summary['y_end']:.10g},"
            f"{summary['y_min']:.10g},"
            f"{summary['y_max']:.10g},"
            f"{summary['direction']},"
            f"{summary['median_dx']:.10g},"
            f"{summary['min_dx']:.10g},"
            f"{summary['max_dx']:.10g},"
            f"{summary['positive_dx']},"
            f"{summary['negative_dx']},"
            f"{summary['zero_dx']}\n"
        )


    # --------------------------------------------------------
    # Last candidate lines
    # --------------------------------------------------------

    write_section(
        out,
        f"LAST {PREVIEW_LINES} CANDIDATE LINES"
    )

    out.write(
        "line,start,stop,length,"
        "x_start,x_end,"
        "y_start,y_end,"
        "direction,median_dx\n"
    )

    for summary in line_summaries[-PREVIEW_LINES:]:

        out.write(
            f"{summary['line']},"
            f"{summary['start']},"
            f"{summary['stop']},"
            f"{summary['length']},"
            f"{summary['x_start']:.10g},"
            f"{summary['x_end']:.10g},"
            f"{summary['y_start']:.10g},"
            f"{summary['y_end']:.10g},"
            f"{summary['direction']},"
            f"{summary['median_dx']:.10g}\n"
        )


    # --------------------------------------------------------
    # Largest X jumps
    # --------------------------------------------------------

    write_section(
        out,
        "20 LARGEST ABSOLUTE X JUMPS"
    )

    largest_dx_indices = np.argsort(
        np.abs(dx)
    )[-20:][::-1]

    out.write(
        "index,x_before,x_after,dx,"
        "y_before,y_after,dy\n"
    )

    for i in largest_dx_indices:
        out.write(
            f"{i},"
            f"{x[i]:.10g},"
            f"{x[i + 1]:.10g},"
            f"{dx[i]:.10g},"
            f"{y[i]:.10g},"
            f"{y[i + 1]:.10g},"
            f"{dy[i]:.10g}\n"
        )


    # --------------------------------------------------------
    # Largest Y jumps
    # --------------------------------------------------------

    write_section(
        out,
        "20 LARGEST ABSOLUTE Y JUMPS"
    )

    largest_dy_indices = np.argsort(
        np.abs(dy)
    )[-20:][::-1]

    out.write(
        "index,x_before,x_after,dx,"
        "y_before,y_after,dy\n"
    )

    for i in largest_dy_indices:
        out.write(
            f"{i},"
            f"{x[i]:.10g},"
            f"{x[i + 1]:.10g},"
            f"{dx[i]:.10g},"
            f"{y[i]:.10g},"
            f"{y[i + 1]:.10g},"
            f"{dy[i]:.10g}\n"
        )


    # --------------------------------------------------------
    # First and last observations
    # --------------------------------------------------------

    write_section(
        out,
        "FIRST 30 COORDINATES"
    )

    out.write("index,x,y\n")

    for i in range(min(30, len(x))):

        out.write(
            f"{i},"
            f"{x[i]:.10g},"
            f"{y[i]:.10g}\n"
        )


    write_section(
        out,
        "LAST 30 COORDINATES"
    )

    out.write("index,x,y\n")

    start_index = max(
        0,
        len(x) - 30
    )

    for i in range(start_index, len(x)):

        out.write(
            f"{i},"
            f"{x[i]:.10g},"
            f"{y[i]:.10g}\n"
        )


# ============================================================
# Console output
# ============================================================

print("X/Y geometry diagnostic complete.")
print()

print(f"Coordinate samples: {len(x)}")
print(
    f"Candidate lines at tolerance "
    f"{WORKING_Y_TOLERANCE:.1e}: "
    f"{number_of_candidate_lines}"
)

print(
    f"Line-length range: "
    f"{np.min(line_lengths)} "
    f"to "
    f"{np.max(line_lengths)}"
)

print(
    f"Median line length: "
    f"{median_line_length:.3f}"
)

print()
print("Diagnostic written to:")
print(OUTPUT)