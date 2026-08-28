# Sea-Ice Draft Reconstruction from 2D Imaging Sonar

[English](README.md) | [中文](README_CN.md)

> Last updated: 2026-08-28

**Version 1.0.0**

A research reference implementation for Blueprint Subsea Oculus imaging sonar,
covering raw `.oculus`-to-NetCDF preprocessing and reconstruction of
along-track sea-ice draft from upward-looking two-dimensional sonar, AUV depth,
and vehicle attitude data.

## Scope

This repository provides the general algorithmic framework, processing
workflow, and implementation required to understand and reproduce the method.
It does not contain field sonar data, navigation records, manual annotations,
or formal research products. The repository is a self-contained public-release
copy and does not read from or modify any external research workspace.

The current input adapter is designed primarily for Oculus NetCDF files
exported by `oculus-python`. Other two-dimensional imaging sonars may reuse the
core algorithm after their echo matrix, timing, and sampling geometry have been
adapted to [`docs/INPUT_FORMAT.md`](docs/INPUT_FORMAT.md). Raw `.oculus` parsing
is delegated to the external
[`oculus-python`](https://gitlab.gbar.dtu.dk/fletho/oculus-python) project; its
source code is not included here.

The method extracts a representative ice-bottom boundary from each sonar frame
under a center-plane imaging assumption and reconstructs an along-track series
of sea-ice draft. It is not a complete three-dimensional reconstruction of the
ice underside. It also does not estimate total sea-ice thickness without
independent freeboard, snow-depth, density, and hydrostatic constraints.

## Pipeline

```text
Oculus .oculus log
        ↓  oculus-python / bps_oculus_io
Oculus NetCDF polar sonar frames
        ↓
center-plane backward mapping
        ↓
range-adaptive background and artifact suppression
        ↓
candidate ice-band segmentation
        ↓
intensity-weighted representative boundary
        ↓
roll and pitch correction
        ↓
AUV depth − upward vertical range
        ↓
along-track sea-ice draft and QC tables
```

See [`docs/METHOD.md`](docs/METHOD.md) for the detailed method description.

## Repository structure

```text
imaging-sonar-sea-ice-draft/
├── ice_sonar_pipeline/       # Core processing modules
├── config/                   # Generic example configuration
├── docs/                     # Method, input format, and release checks
├── examples/                 # Synthetic test-data generation only
├── tests/                    # Geometry and attitude-correction tests
├── convert_oculus_to_netcdf.py # Oculus raw-log conversion entry point
├── run_pipeline.py           # Command-line entry point
├── run_from_config.py        # JSON-configuration entry point
├── requirements.txt
└── pyproject.toml
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell, including the optional Oculus raw-log converter:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[oculus]"
```

If the input NetCDF file has already been prepared and raw-log conversion is
not required:

```bash
python -m pip install -e .
```

`oculus-python 0.0.2a1` is currently a pre-release, and its upstream project
labels v2 `.oculus` support as experimental. The conversion extra constrains
`numpy<2` and `opencv-python<4.12` to keep the current dependency set compatible
without modifying any installed third-party source file.

## Input data

Sea-ice draft reconstruction requires two synchronized input files:

1. an imaging-sonar NetCDF file containing per-ping echo matrices, time, and
   sonar sampling geometry;
2. an AUV motion CSV file containing synchronized time, depth, and
   quality-controlled roll and pitch fields.

See [`docs/INPUT_FORMAT.md`](docs/INPUT_FORMAT.md) for the expected variables
and columns. The current reader supports the NetCDF structure exported by
`oculus-python`. When another system uses different variable names or
dimensions, implement a separate data adapter instead of changing the core
geometric conventions directly.

## Oculus raw-data preprocessing

After installing the conversion extra, convert one `.oculus` log to NetCDF4:

```bash
python convert_oculus_to_netcdf.py \
  --input path/to/input.oculus \
  --output-dir path/to/netcdf_output
```

The equivalent installed command is:

```bash
oculus-to-netcdf --input path/to/input.oculus --output-dir path/to/netcdf_output
```

The wrapper invokes `bps_oculus_io` from `oculus-python` and then verifies that
the exported dataset contains a non-empty `backscatter` variable. It refuses
to overwrite a matching NetCDF file by default and removes a temporary hard
link or copy after conversion. On Windows, use a short ASCII-only output path
if the third-party HDF5/NetCDF stack cannot handle a non-ASCII path.

## Command-line use

```bash
python run_pipeline.py \
  --nc-file path/to/sonar.nc \
  --motion-file path/to/auv_motion.csv \
  --output-root outputs/example \
  --timezone UTC \
  --ping-step 5 \
  --grid-range-mode metadata \
  --binary-tuning-workers 1
```

Display all available options:

```bash
python run_pipeline.py --help
```

Relative output paths are resolved from the current working directory. The
default single-process parameter search is recommended for reproducible and
auditable runs.

## JSON configuration

Copy and edit [`config/example_run.json`](config/example_run.json), then run:

```bash
python run_from_config.py --config config/example_run.json
```

Relative paths in the JSON file are resolved from the directory containing the
configuration file.

## Synthetic smoke test

No field observations are distributed with this repository. A small artificial
dataset can be generated to verify installation, input reading, and pipeline
integration:

```bash
python examples/generate_synthetic_inputs.py --output-dir examples/generated
python run_from_config.py --config config/example_run.json
```

The synthetic dataset is intended only for software smoke testing. It must not
be used as evidence of algorithm accuracy or field performance in polar
environments.

## Optional manual masks

The repository does not include observation-derived manual masks. Users may
provide their own remove-only polygon mask through `--manual-mask-json`. Any
manual editing should be reported explicitly in the method and results.

## Checks required for a new dataset

Before processing new observations, verify at least the following:

- sonar range, azimuth, and beam ordering;
- AUV and sonar coordinate directions and roll/pitch sign conventions;
- clock alignment between sonar and navigation data;
- the reference used for AUV depth and any sensor mounting offsets;
- whether the default thresholds and morphological constraints are appropriate
  for the operating range and signal-to-noise ratio;
- whether automated QC flags agree with representative manual review.

## Data and privacy safeguards

The supplied `.gitignore` excludes `.oculus` logs, NetCDF, MAT, HDF5, CSV,
NumPy arrays, images, animations, result directories, manual masks, local
configurations, and archives by default. Always inspect `git status` and the
staged diff and perform a content scan before publishing; ignore rules alone
are not a complete safeguard.

## Citation, third-party acknowledgment, and license

The associated method paper has not yet been published, so Version 1.0.0 does
not provide a paper citation. Formal citation information will be added after
publication.

Copyright in the original code in this repository is held personally by Liyibo
and released under the [`MIT License`](LICENSE). Raw-log conversion depends on
`oculus-python 0.0.2a1`, licensed under Apache-2.0. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and thanks.
