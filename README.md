# HDF5-to-GeoTIFF Data Pipeline

A reproducible Python pipeline for reconstructing spatial raster products from Canadian Light Source SGM fly-scan acquisitions that store stage coordinates in HDF5 and spectral detector counts in separate SDD binary files.

> **Current implementation status**
>
> The validated prototype reads one HDF5 coordinate file and one SDD detector binary, reconstructs the serpentine raster, measures row-dependent bidirectional displacement, applies an anchored subpixel correction to the full spectral cube, and exports quantitative TIFF/NumPy products.
>
> Despite the repository name, **validated GeoTIFF export is not implemented yet**. The current TIFF product is a locally referenced analytical raster. The X and Y scan axes are exported separately as NumPy arrays.

---

## Table of contents

1. [Purpose](#purpose)
2. [What the pipeline currently does](#what-the-pipeline-currently-does)
3. [What the pipeline does not yet do](#what-the-pipeline-does-not-yet-do)
4. [Scientific and data assumptions](#scientific-and-data-assumptions)
5. [Repository layout](#repository-layout)
6. [Input data contract](#input-data-contract)
7. [Installation](#installation)
8. [Quick start with the playtest model](#quick-start-with-the-playtest-model)
9. [Run another scan](#run-another-scan)
10. [Command-line reference](#command-line-reference)
11. [Output products](#output-products)
12. [How reconstruction works](#how-reconstruction-works)
13. [How subpixel registration works](#how-subpixel-registration-works)
14. [How to verify a run](#how-to-verify-a-run)
15. [Use from Python](#use-from-python)
16. [Troubleshooting](#troubleshooting)
17. [Tests](#tests)
18. [Development diagnostics](#development-diagnostics)
19. [Data management and Git](#data-management-and-git)
20. [Known limitations](#known-limitations)
21. [Roadmap](#roadmap)
22. [Reproducibility record](#reproducibility-record)
23. [Citation and acknowledgement](#citation-and-acknowledgement)
24. [License](#license)

---

## Purpose

CLS SGM acquisitions can distribute information required for a single spatial image across several files:

- an HDF5 file containing chronological X and Y sample coordinates;
- one or more SDD binary detector files containing a spectrum for every acquired position;
- an MCC flyer CSV containing auxiliary channels and acquisition information.

The objective of this package is to convert those acquisition-level files into explicit, inspectable spatial products while preserving the detector-channel dimension and documenting every geometric correction.

The current processing path is:

```text
HDF5 coordinate arrays + one SDD binary
                    |
                    v
validate observation counts and raster geometry
                    |
                    v
reshape chronological spectra into raster rows
                    |
                    v
reverse alternating serpentine rows
                    |
                    v
measure odd-row displacement against even-row anchors
                    |
                    v
smooth the row-dependent fractional-shift profile
                    |
                    v
apply subpixel X registration to all 256 channels
                    |
                    v
export total-count raster, spectral cubes, axes, mask, and shift table
```

The package is designed so that acquisition reading, raster reconstruction, registration, and output writing remain separate and reusable.

---

## What the pipeline currently does

The validated prototype performs the following operations:

- Locates one HDF5 file in a scan directory, or accepts an explicitly selected HDF5 filename.
- Reads coordinate arrays from:

  ```text
  /hexapod_waves/x
  /hexapod_waves/y
  ```

- Detects raster-row boundaries from changes in Y.
- Verifies that every inferred row has the same number of observations.
- Verifies that Y is constant within each raster row.
- Verifies that Y increases monotonically between rows.
- Verifies alternating X direction consistent with serpentine acquisition.
- Reads one SDD binary as little-endian unsigned 32-bit integers.
- Assumes 256 detector channels per spatial observation.
- Verifies that the SDD value count matches:

  ```text
  rows × columns × 256
  ```

- Reshapes chronological detector values to:

  ```text
  (Y rows, X columns, detector channels)
  ```

- Reverses alternating rows so every output row uses increasing X order.
- Constructs a total-count image by summing the 256 detector channels.
- Measures fractional horizontal displacement for interior odd rows.
- Keeps even rows as fixed spatial anchors.
- Smooths the measured odd-row shift profile along Y.
- Applies the smoothed shift to all 256 detector channels using linear X interpolation.
- Uses no circular wrapping.
- Marks pixels outside the valid source extent as invalid.
- Exports quantitative and display products.

---

## What the pipeline does not yet do

The following are not part of the validated production path:

- combining `sdd1_0.bin` through `sdd4_0.bin`;
- detector dead-time correction;
- detector energy-channel calibration;
- spectral ROI selection;
- I0 normalization;
- MCC channel identification or normalization;
- background subtraction;
- scan stitching or quadrant mosaicking;
- conversion of local scan coordinates to a defined coordinate reference system;
- validated GeoTIFF export;
- NXstxm export;
- PyMca-compatible NeXus output;
- XANES stack analysis;
- PCA or clustering;
- automated quality-control acceptance criteria;
- support for arbitrary SDD channel counts or alternate binary dtypes without explicit configuration.

A placeholder GeoTIFF writer intentionally raises `NotImplementedError` rather than producing an apparently georeferenced file without a scientifically validated spatial-reference model.

---

## Scientific and data assumptions

The current prototype depends on the assumptions below. A run should not be considered scientifically valid if these assumptions do not hold.

### Coordinate assumptions

- X and Y are one-dimensional chronological arrays.
- X and Y have identical lengths.
- Y is constant within each raster line.
- Y increases from one raster line to the next.
- Raster rows have equal observation counts.
- X alternates between strictly increasing and strictly decreasing rows.
- After reversing odd rows, every raster line shares the same X grid.
- Coordinates are local scan coordinates, not yet a projected or geographic coordinate reference system.

### Detector assumptions

- The selected binary belongs to the selected HDF5 acquisition.
- Detector values are little-endian unsigned 32-bit integers:

  ```python
  dtype = "<u4"
  ```

- Every spatial observation contains exactly 256 detector-channel values.
- The binary contains no undocumented header, footer, padding, flyback records, or extra observations.
- Detector spectra appear in the same chronological order as the X/Y coordinates.

### Registration assumptions

- Even rows provide a suitable fixed reference.
- Odd-row displacement is predominantly horizontal along X.
- An odd row can be compared with the mean of the even rows immediately above and below.
- The row-displacement field changes smoothly along Y.
- Linear interpolation is acceptable for the current prototype.
- Edge pixels that cannot be interpolated from acquired samples must be marked invalid rather than wrapped or invented.

---

## Repository layout

```text
HDF5-to-GeoTiff-Data-Pipeline/
├── transform_pipeline/
│   ├── __init__.py
│   ├── cli.py
│   ├── example.py
│   ├── pipeline.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── readers/
│   │   │   ├── __init__.py
│   │   │   └── cls_sgm.py
│   │   └── writers/
│   │       ├── __init__.py
│   │       ├── geotiff.py
│   │       └── tiff.py
│   └── transforms/
│       ├── __init__.py
│       ├── normalize.py
│       ├── raster.py
│       └── registration.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_paths.py
├── development/
│   └── diagnostics/
│       ├── README.md
│       ├── sdd1_0_diagnostic.txt
│       ├── sdd1_row_shift_diagnostic.py
│       ├── sdd1_subpixel_row_shift_diagnostic.py
│       └── xy_raster_geometry_diag.py
├── Playtest_model/
│   └── Testmodel_NCo_7_SW/
├── paper/
├── guides/
├── pyproject.toml
├── requirements.txt
├── makefile
├── LICENSE
└── README.md
```

### Module responsibilities

#### `transform_pipeline/core/models.py`

Contains shared dataclasses used to pass structured results between processing stages:

- `RasterGeometry`
- `ShiftMeasurement`
- `ReconstructionResult`
- `RegistrationResult`

#### `transform_pipeline/io/readers/cls_sgm.py`

Handles CLS SGM input discovery and decoding:

- repository-root resolution;
- default playtest-directory resolution;
- HDF5 discovery;
- coordinate reading;
- SDD binary reading;
- initial cube reconstruction.

#### `transform_pipeline/transforms/raster.py`

Handles spatial organization:

- serpentine-row reversal;
- raster-geometry inference;
- coordinate-grid validation.

#### `transform_pipeline/transforms/registration.py`

Handles row alignment:

- correlation scoring;
- subpixel peak refinement;
- weighted smoothing;
- spectral-row interpolation;
- application of the odd-row shift profile.

#### `transform_pipeline/io/writers/tiff.py`

Handles output serialization:

- quantitative float TIFF;
- validity-mask TIFF;
- 8-bit preview generation;
- PNG preview;
- shift CSV.

#### `transform_pipeline/pipeline.py`

Coordinates the production workflow through two high-level functions:

```python
process_sdd_scan(...)
write_pipeline_outputs(...)
```

#### `transform_pipeline/cli.py`

Provides the command-line interface.

#### `development/diagnostics/`

Contains exploratory scripts retained for provenance and debugging. These files are not imported by the production package.

---

## Input data contract

## Minimum scan-directory contents

A processable scan directory must contain:

```text
scan_directory/
├── one_file.h5
└── sdd1_0.bin
```

If more than one `.h5` file is present, use `--h5` to select the intended file.

The detector filename may be changed with `--sdd`.

### Typical CLS scan bundle

The repository playtest model uses the following bundle:

```text
Playtest_model/
└── Testmodel_NCo_7_SW/
    ├── NCo_7_SW_2026-05-09_150721_0.00eV.h5
    ├── mcc_flyer_0.csv
    ├── sdd1_0.bin
    ├── sdd2_0.bin
    ├── sdd3_0.bin
    └── sdd4_0.bin
```

The current default run processes only:

```text
NCo_7_SW_2026-05-09_150721_0.00eV.h5
sdd1_0.bin
```

The MCC CSV and SDD2-SDD4 files are retained for future development but are not used in the present production path.

### HDF5 requirements

The HDF5 file must expose readable datasets at:

```text
/hexapod_waves/x
/hexapod_waves/y
```

Both datasets must:

- be one-dimensional;
- have the same length;
- describe the chronological acquisition sequence.

### SDD requirements

The selected binary must match:

```text
number of coordinate observations × 256 channels × 4 bytes
```

For the validated playtest:

```text
raster rows:       301
samples per row:   302
spatial samples:   90,902
channels:          256
cube shape:        (301, 302, 256)
```

These dimensions are inferred and validated. They are not hard-coded into the production reader.

---

## Installation

## Supported Python

The project metadata requires:

```text
Python >= 3.11
```

The current Windows playtest was validated with:

```text
Python 3.14
```

Use one Python interpreter consistently for environment creation, installation, testing, and execution.

### Windows PowerShell installation

From the repository root:

```powershell
py -3.14 -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the package and test dependencies in editable mode:

```powershell
python -m pip install -e ".[test]"
```

Verify the interpreter:

```powershell
python --version
python -m pip --version
```

Verify that Python imports the repository package:

```powershell
python -c "import transform_pipeline; print(transform_pipeline.__file__)"
```

The printed path should end with:

```text
HDF5-to-GeoTiff-Data-Pipeline\transform_pipeline\__init__.py
```

### Installation without activating the environment

You can call the environment interpreter explicitly:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Then run commands with:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m transform_pipeline.cli
```

### Dependency files

- `pyproject.toml` defines package metadata, runtime dependencies, optional test dependencies, package discovery, and tool configuration.
- `requirements.txt` provides a simple dependency list.
- `environment.yml` is inherited repository infrastructure and may not match the Python 3.14 local environment exactly.

For the validated Windows workflow, editable pip installation from `pyproject.toml` is the authoritative setup route.

---

## Quick start with the playtest model

The command-line interface resolves the playtest directory relative to the repository root:

```text
Playtest_model/Testmodel_NCo_7_SW
```

No machine-specific absolute path is required.

From the repository root, run:

```powershell
python -m transform_pipeline.cli
```

The default run:

- uses the repository playtest directory;
- selects `sdd1_0.bin`;
- locates the single HDF5 file automatically;
- creates or updates `prototype_output` inside the scan directory;
- does not save the two full spectral cubes unless requested.

To save both the reconstructed and registered spectral cubes:

```powershell
python -m transform_pipeline.cli --save-cubes
```

Expected terminal completion message:

```text
Pipeline complete. Output: ...\Playtest_model\Testmodel_NCo_7_SW\prototype_output
```

---

## Run another scan

### Use another scan directory

```powershell
python -m transform_pipeline.cli `
    "D:\path\to\scan_directory"
```

### Select another SDD file

```powershell
python -m transform_pipeline.cli `
    "D:\path\to\scan_directory" `
    --sdd "sdd2_0.bin"
```

This changes only the detector file. The selected binary must still satisfy the current `<u4`, 256-channel assumptions.

### Select an HDF5 file explicitly

Use this when the scan directory contains more than one `.h5` file:

```powershell
python -m transform_pipeline.cli `
    "D:\path\to\scan_directory" `
    --h5 "selected_scan.h5"
```

### Select both HDF5 and SDD files

```powershell
python -m transform_pipeline.cli `
    "D:\path\to\scan_directory" `
    --h5 "selected_scan.h5" `
    --sdd "sdd1_0.bin"
```

### Choose a separate output directory

```powershell
python -m transform_pipeline.cli `
    "D:\path\to\scan_directory" `
    --output-dir "D:\path\to\output_directory"
```

### Save full cubes

```powershell
python -m transform_pipeline.cli `
    "D:\path\to\scan_directory" `
    --save-cubes
```

Full cubes can be large. Omit `--save-cubes` when only the registered total-count image, axes, mask, shift table, and preview are needed.

---

## Command-line reference

General syntax:

```text
python -m transform_pipeline.cli [scan_dir] [options]
```

### Positional argument

#### `scan_dir`

Optional path to a CLS SGM scan directory.

If omitted, the CLI uses:

```text
<repository root>/Playtest_model/Testmodel_NCo_7_SW
```

### Options

#### `--h5 FILENAME`

Select a specific HDF5 filename within `scan_dir`.

Use this when more than one `.h5` file exists.

#### `--sdd FILENAME`

Select an SDD binary filename within `scan_dir`.

Default:

```text
sdd1_0.bin
```

#### `--output-dir PATH`

Write outputs to a custom directory.

Default:

```text
scan_dir/prototype_output
```

#### `--save-cubes`

Save both full `(Y, X, 256)` spectral cubes.

Without this option, the pipeline still performs registration in memory and writes the total-count output, axes, mask, and shift table.

### Display CLI help

```powershell
python -m transform_pipeline.cli --help
```

---

## Output products

Default output directory:

```text
scan_directory/
└── prototype_output/
```

Standard outputs:

```text
prototype_output/
├── total_counts_subpixel_registered.npy
├── total_counts_subpixel_registered_float32.tif
├── total_counts_subpixel_registered_preview.png
├── subpixel_validity_mask.tif
├── subpixel_shifts.csv
├── x_axis.npy
└── y_axis.npy
```

Additional outputs created with `--save-cubes`:

```text
prototype_output/
├── sdd_cube_reconstructed.npy
└── sdd_cube_subpixel_registered.npy
```

### `total_counts_subpixel_registered.npy`

Purpose:

- primary NumPy total-count raster after registration.

Shape:

```text
(Y rows, X columns)
```

Data characteristics:

- floating-point output;
- summed across all 256 detector channels;
- invalid shifted-edge pixels represented as `NaN`.

Recommended use:

- quantitative Python analysis;
- downstream masking;
- comparison with the validity mask;
- conversion to other analytical formats.

### `total_counts_subpixel_registered_float32.tif`

Purpose:

- quantitative floating-point TIFF representation of the registered total-count raster.

Data characteristics:

- `float32`;
- no display rescaling applied to the quantitative values;
- invalid edge pixels remain `NaN`.

Recommended use:

- ImageJ/Fiji or other scientific-image software that supports floating TIFF;
- quantitative raster inspection;
- downstream format conversion.

Do not interpret an all-white initial display as a constant image. Some general-purpose viewers do not automatically scale floating-point TIFF values.

### `total_counts_subpixel_registered_preview.png`

Purpose:

- quick visual verification.

Data characteristics:

- 8-bit grayscale;
- percentile-stretched;
- invalid pixels displayed as black;
- display-only.

Do not use this PNG for quantitative analysis.

### `subpixel_validity_mask.tif`

Purpose:

- identifies pixels supported by acquired detector samples after interpolation.

Values:

```text
255 = valid
0   = invalid
```

Invalid pixels occur at the horizontal edge exposed when a row is shifted. The pipeline does not wrap data from the opposite edge.

### `subpixel_shifts.csv`

Purpose:

- records the estimated and applied row-registration model.

Columns:

```text
row
raw_fractional_shift
smoothed_fractional_shift
peak_correlation
zero_shift_correlation
correlation_gain
```

Interpretation:

- `row`: zero-based raster-row index;
- `raw_fractional_shift`: individual optimal odd-row estimate;
- `smoothed_fractional_shift`: shift used by the registration stage;
- `peak_correlation`: correlation after the best measured shift;
- `zero_shift_correlation`: correlation without horizontal correction;
- `correlation_gain`: improvement relative to zero shift.

Positive shift means odd-row content moves right. Negative shift means odd-row content moves left.

### `x_axis.npy`

Purpose:

- reconstructed increasing-X coordinate axis.

Expected shape:

```text
(X columns,)
```

### `y_axis.npy`

Purpose:

- reconstructed increasing-Y coordinate axis.

Expected shape:

```text
(Y rows,)
```

### `sdd_cube_reconstructed.npy`

Created only with `--save-cubes`.

Purpose:

- spectral cube after chronological reshape and serpentine correction, before subpixel registration.

Shape:

```text
(Y rows, X columns, 256 channels)
```

### `sdd_cube_subpixel_registered.npy`

Created only with `--save-cubes`.

Purpose:

- full spectral cube after applying the smoothed odd-row shift profile.

Shape:

```text
(Y rows, X columns, 256 channels)
```

Data characteristics:

- `float32` because fractional interpolation creates noninteger values;
- even rows remain fixed;
- odd rows are linearly interpolated along X;
- invalid edges are `NaN` across all channels.

---

## How reconstruction works

## 1. Read chronological coordinates

The reader loads:

```python
x = h5["/hexapod_waves/x"][:]
y = h5["/hexapod_waves/y"][:]
```

The arrays describe acquisition order, not yet a conventional spatial image.

## 2. Detect raster-row boundaries

A row boundary is inferred when consecutive Y values differ beyond the configured tolerance.

The resulting row lengths must be uniform.

## 3. Validate the serpentine pattern

For the validated scan, acquisition follows:

```text
row 0: X increasing
row 1: X decreasing
row 2: X increasing
row 3: X decreasing
...
```

Y increases between rows.

## 4. Read detector spectra

The selected SDD binary is read as:

```python
np.dtype("<u4")
```

The flat value count must equal:

```text
rows × columns × 256
```

## 5. Reshape chronological spectra

The flat binary becomes:

```text
(rows, columns, 256)
```

At this stage, odd rows still follow reverse acquisition direction.

## 6. De-serpentine

Odd rows are reversed along X:

```text
raw chronological odd row:    right -> left
spatial output odd row:        left  -> right
```

Every output row then shares the same increasing-X orientation.

## 7. Create the total-count image

The preview registration signal is formed by summing detector channels:

```python
total_counts = cube.sum(axis=2)
```

This does not remove the channel dimension from the saved cube. It creates a two-dimensional signal used for alignment and total-count output.

---

## How subpixel registration works

The need for registration arose because de-serpentining corrected row direction but left a row-dependent horizontal displacement. Integer shifting was insufficient because the optimal displacement varied smoothly from positive at one end of the image, through approximately zero near the center, to negative at the other end.

### Anchor model

- Even rows are fixed.
- Interior odd rows are the moving rows.
- An odd row is compared with the average of the even rows directly above and below.

For odd row `r`:

```text
reference(r) = 0.5 × [row(r - 1) + row(r + 1)]
```

### Candidate-shift search

The current defaults search:

```text
minimum shift:  -2.0 pixels
maximum shift:  +2.0 pixels
step:            0.025 pixel
```

For each candidate shift:

1. the odd-row total-count signal is linearly interpolated;
2. edge margins are excluded from scoring;
3. Pearson correlation with the reference is calculated.

### Peak refinement

The best grid location is refined with a local three-point quadratic estimate to produce a fractional shift finer than the initial search grid.

### Smoothing

Individual row estimates can be affected by texture and count variation. The raw odd-row profile is therefore smoothed along Y using correlation-gain-weighted Gaussian smoothing.

The current defaults use:

```text
Gaussian sigma:          5 odd-row samples
minimum smoothing weight: 0.0005
```

### Spectral application

The smoothed shift is applied to every one of the 256 detector channels for that odd row.

For destination X position `x` and shift `s`:

```text
source position = x - s
```

Linear interpolation is performed between neighboring source columns.

### Boundary treatment

If a shifted destination pixel requires a source position outside the acquired row:

- the output is set to `NaN`;
- the validity mask is set to invalid;
- no circular wrapping is used;
- no edge value is fabricated.

---

## How to verify a run

Do not rely on the presence of files alone. Use the checks below.

### 1. Confirm terminal completion

The CLI should end with a message containing:

```text
Pipeline complete.
```

### 2. Inspect the preview

Open:

```text
total_counts_subpixel_registered_preview.png
```

Check for:

- recognizable sample structure;
- correct left-to-right orientation;
- absence of alternating full-row reversal;
- reduced row zig-zagging;
- no circularly wrapped edge content.

### 3. Inspect the quantitative TIFF

Open:

```text
total_counts_subpixel_registered_float32.tif
```

Use a scientific viewer and apply a display range if necessary.

### 4. Inspect the mask

Open:

```text
subpixel_validity_mask.tif
```

Most pixels should be valid. Invalid pixels should occur only at the horizontal edge exposed by the corresponding fractional shift.

### 5. Inspect the shift table

Open:

```text
subpixel_shifts.csv
```

Check that:

- only interior odd rows are listed;
- shifts are finite;
- shift magnitude remains within the allowed registration range;
- the smoothed profile varies gradually rather than jumping randomly;
- correlation gain is recorded.

### 6. Confirm array dimensions

```powershell
python -c "import numpy as np; a=np.load(r'Playtest_model\Testmodel_NCo_7_SW\prototype_output\total_counts_subpixel_registered.npy'); print(a.shape, a.dtype)"
```

For the validated playtest, the expected shape is:

```text
(301, 302)
```

If full cubes were saved:

```powershell
python -c "import numpy as np; a=np.load(r'Playtest_model\Testmodel_NCo_7_SW\prototype_output\sdd_cube_subpixel_registered.npy', mmap_mode='r'); print(a.shape, a.dtype)"
```

Expected playtest shape:

```text
(301, 302, 256)
```

### 7. Run automated tests

```powershell
python -m pytest -q
```

All tests should pass before accepting a code change.

---

## Use from Python

## Complete processing example

```python
from pathlib import Path

from transform_pipeline.pipeline import (
    process_sdd_scan,
    write_pipeline_outputs,
)


scan_dir = Path(
    r"D:\path\to\scan_directory"
)

reconstruction, registration = process_sdd_scan(
    scan_dir=scan_dir,
    sdd_filename="sdd1_0.bin",
    h5_filename=None,
)

outputs = write_pipeline_outputs(
    reconstruction,
    registration,
    output_dir=scan_dir / "prototype_output",
    save_cubes=True,
)

print(outputs)
```

## Use the repository playtest path

```python
from transform_pipeline.io.readers.cls_sgm import (
    default_playtest_scan_dir,
)
from transform_pipeline.pipeline import (
    process_sdd_scan,
    write_pipeline_outputs,
)


scan_dir = default_playtest_scan_dir()

reconstruction, registration = process_sdd_scan()

outputs = write_pipeline_outputs(
    reconstruction,
    registration,
    output_dir=scan_dir / "prototype_output",
    save_cubes=False,
)
```

## Access reconstruction products in memory

```python
geometry = reconstruction.geometry
cube_before_registration = reconstruction.cube
total_before_registration = reconstruction.total_counts

print(geometry.rows)
print(geometry.columns)
print(geometry.x_axis)
print(geometry.y_axis)
print(cube_before_registration.shape)
```

## Access registration products in memory

```python
cube_after_registration = registration.cube
total_after_registration = registration.total_counts
validity_mask = registration.validity_mask
shift_measurements = registration.shifts

print(cube_after_registration.shape)
print(total_after_registration.shape)
print(validity_mask.shape)
```

## Inspect individual shifts

```python
for measurement in registration.shifts[:5]:
    print(
        measurement.row,
        measurement.raw_shift,
        measurement.smoothed_shift,
        measurement.correlation_gain,
    )
```

---

## Troubleshooting

## `No module named transform_pipeline`

Cause:

- package not installed in the active environment;
- command executed from an unrelated environment;
- repository root not on the import path.

Fix:

```powershell
python -m pip install -e ".[test]"
```

Verify:

```powershell
python -c "import transform_pipeline; print(transform_pipeline.__file__)"
```

## Wrong Python interpreter

List installed interpreters:

```powershell
py -0p
```

Check the current interpreter:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

Use Python 3.14 explicitly when needed:

```powershell
py -3.14 -m transform_pipeline.cli
```

## Default playtest directory not found

Expected path:

```text
<repository root>/Playtest_model/Testmodel_NCo_7_SW
```

Check it:

```powershell
Test-Path "Playtest_model\Testmodel_NCo_7_SW"
```

Expected:

```text
True
```

Alternatively, provide a scan directory explicitly.

## No HDF5 file found

Confirm the scan directory contains a `.h5` file:

```powershell
Get-ChildItem "D:\path\to\scan_directory" -Filter "*.h5"
```

## More than one HDF5 file found

Select one explicitly:

```powershell
python -m transform_pipeline.cli `
    "D:\path\to\scan_directory" `
    --h5 "selected_scan.h5"
```

## SDD file not found

Confirm the detector filename:

```powershell
Get-ChildItem "D:\path\to\scan_directory" -Filter "sdd*.bin"
```

Then select the correct file with `--sdd`.

## SDD value-count mismatch

This means the selected binary cannot be reshaped under the inferred raster geometry and current 256-channel assumption.

Check:

- HDF5 and SDD originated from the same scan;
- the selected binary is complete;
- channel count is 256;
- dtype is `<u4`;
- no header or extra records are present;
- coordinate count matches detector-spectrum count.

Do not bypass this error by forcing a reshape.

## Unexpected X direction

The coordinate sequence does not match the expected alternating serpentine pattern.

Possible causes include:

- another scan mode;
- duplicated coordinates;
- partial acquisition;
- flyback records;
- a coordinate dataset that does not correspond to the SDD sequence.

Use the scripts under `development/diagnostics` before changing production assumptions.

## TIFF appears entirely white

The quantitative TIFF is floating point and may not be automatically contrast-stretched.

Open:

```text
total_counts_subpixel_registered_preview.png
```

or manually adjust the display range in scientific-image software.

Do not replace the analytical TIFF with an 8-bit image for convenience.

## Black pixels at one horizontal edge

These pixels are expected when subpixel shifting exposes positions not supported by acquired samples.

Use:

```text
subpixel_validity_mask.tif
```

to identify valid pixels.

## Pipeline runs but row drift remains

Review:

```text
subpixel_shifts.csv
```

Then inspect whether:

- shifts reach the configured maximum;
- correlation gains are weak;
- the signal lacks spatial texture;
- total counts are a poor registration signal for that acquisition;
- displacement is not exclusively horizontal;
- even rows are not a stable reference.

Do not add manual integer correction regions to production code without a documented diagnostic basis.

## `pytest` finds no tests

Run from the repository root:

```powershell
python -m pytest -q
```

Confirm that `pyproject.toml` contains:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

## Generated files appear in Git status

Check `.gitignore` and confirm that raw data and generated products remain ignored.

Examples:

```powershell
git check-ignore -v -- "Playtest_model\Testmodel_NCo_7_SW\sdd1_0.bin"
git check-ignore -v -- "Playtest_model\Testmodel_NCo_7_SW\prototype_output"
```

---

## Tests

Run the test suite:

```powershell
python -m pytest -q
```

The current tests cover:

### Raster behavior

- alternating-row reversal;
- rectangular geometry inference;
- expected X and Y axes for a synthetic serpentine raster.

### Repository path behavior

- repository-root resolution;
- default playtest path:

  ```text
  Playtest_model/Testmodel_NCo_7_SW
  ```

### Recommended future tests

The current test suite is intentionally small. Future additions should cover:

- temporary HDF5 and binary end-to-end reconstruction;
- incorrect SDD count rejection;
- irregular row-length rejection;
- nonmonotonic Y rejection;
- known synthetic fractional-shift recovery;
- preservation of even rows;
- invalid-edge mask behavior;
- equal shift application across all detector channels;
- output-file creation;
- CLI argument handling.

---

## Development diagnostics

The production pipeline is intentionally narrower than the diagnostic history used to derive it.

Retained diagnostic files include:

```text
development/diagnostics/
├── sdd1_0_diagnostic.txt
├── xy_raster_geometry_diag.py
├── sdd1_row_shift_diagnostic.py
└── sdd1_subpixel_row_shift_diagnostic.py
```

### `sdd1_0_diagnostic.txt`

Records the binary-format investigation that supported the `<u4`, 256-channel interpretation.

### `xy_raster_geometry_diag.py`

Examines:

- coordinate ranges;
- consecutive X/Y differences;
- candidate row boundaries;
- row lengths;
- scan direction;
- endpoint transitions.

### `sdd1_row_shift_diagnostic.py`

Tests integer shifts and helped establish that:

- the artifact was direction-dependent;
- whole-pixel symmetric correction was inappropriate;
- even and odd measurements described the same relative mismatch from opposite viewpoints.

### `sdd1_subpixel_row_shift_diagnostic.py`

Measures the continuous odd-row displacement profile and generates:

- row-shift CSV;
- shift-profile plot;
- score heatmap;
- diagnostic summary.

These scripts may contain acquisition-specific configuration placeholders. Review their configuration blocks before running them.

---

## Data management and Git

Raw acquisitions and generated analytical products can be large. They should not be committed to ordinary Git history.

The repository is configured to ignore appropriate patterns such as:

```text
*.h5
*.bin
*.npy
*.npz
*.tif
*.tiff
prototype_output/
*.egg-info/
__pycache__/
.pytest_cache/
.idea/
```

Recommended Git content:

- source code;
- tests;
- configuration;
- small documentation;
- schemas;
- small, nonrestricted metadata;
- diagnostic methodology;
- release notes.

Recommended external-data content:

- raw HDF5 acquisitions;
- SDD binaries;
- full spectral cubes;
- generated TIFF products;
- large previews and intermediate outputs.

Before committing:

```powershell
git status --short
```

Before pushing:

```powershell
python -m pytest -q
git --no-pager log --oneline -3
```

---

## Known limitations

### One detector at a time

The CLI selects one SDD binary. It does not combine detector quadrants or perform cross-detector calibration.

### Fixed detector interpretation

The production reader assumes:

```text
little-endian uint32
256 channels
```

### Registration uses total counts

The registration signal is the sum across all detector channels. This may not be optimal for acquisitions where a spectral ROI provides stronger spatial contrast.

### Even-row anchoring

The current model holds even rows fixed and moves odd rows. This is a relative-registration choice, not proof that even rows represent absolute ground truth.

### Horizontal registration only

The method corrects X displacement. It does not estimate Y displacement, rotation, shear, nonlinear warping within a row, or specimen motion independent of scan direction.

### Linear interpolation

Linear interpolation changes integer counts into floating-point values and introduces channel-wise interpolation. The method preserves channel alignment but does not preserve the original integer-valued detector samples at shifted positions.

### Edge data loss

Subpixel shifting creates invalid pixels at one horizontal edge. The pipeline marks these pixels rather than filling them.

### No calibrated GeoTIFF

X and Y axes are stored separately. The output TIFF has no validated CRS, affine transform, or documented mapping to a geographic reference frame.

### No automatic scientific acceptance threshold

The software records correlation information but does not automatically declare a reconstruction scientifically acceptable. Visual and quantitative review remain required.

### Playtest-specific validation

The prototype has been validated against the included local playtest acquisition. Other scan modes may require additional format diagnostics.

---

## Roadmap

Potential next steps, in dependency order:

1. Expand automated end-to-end tests with temporary HDF5/BIN fixtures.
2. Add structured run metadata and a machine-readable processing manifest.
3. Parameterize channel count and binary dtype explicitly.
4. Add SDD2-SDD4 processing and detector-combination rules.
5. Parse and identify MCC flyer channels.
6. Add I0 normalization.
7. Add spectral energy/channel calibration.
8. Add spectral ROI extraction and energy-resolved raster export.
9. Add registration-quality plots to the production output.
10. Establish a scientifically justified coordinate model.
11. Implement validated GeoTIFF export.
12. Add NXstxm or other interoperable scientific export where appropriate.
13. Add CLI configuration files for reproducible batch processing.
14. Add release versioning and citation metadata.

---

## Reproducibility record

A reproducible processing record should capture at least:

- repository commit hash;
- Python version;
- package versions;
- HDF5 filename;
- SDD filename;
- input file sizes and checksums;
- coordinate dataset paths;
- inferred raster dimensions;
- detector dtype;
- channel count;
- candidate shift range;
- shift step;
- excluded edge margin;
- smoothing sigma;
- applied row shifts;
- output filenames;
- validity-mask statistics.

Record the current commit:

```powershell
git rev-parse HEAD
```

Record the environment:

```powershell
python --version
python -m pip freeze > environment-lock.txt
```

The current production pipeline does not yet write a complete processing manifest automatically. Until that feature is implemented, retain the shift CSV and record the commit and environment alongside analytical outputs.

---

## Citation and acknowledgement

A formal software citation has not yet been defined in this repository.

Until a release and citation record are created, users should record:

- repository name;
- repository URL;
- commit hash;
- access date;
- software authors or maintainers;
- CLS SGM beamline acknowledgement appropriate to the associated experiment;
- related dataset or publication identifiers where available.

Do not cite the repository as a validated GeoTIFF converter until GeoTIFF export and coordinate-reference behavior are implemented and documented.

---

## License

See [`LICENSE`](LICENSE) for the repository license.
