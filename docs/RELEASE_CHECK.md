# Public-release check

> Latest local check: 2026-09-16

Release: **1.0.0**

## Included

- center-plane polar-to-Cartesian backward mapping;
- range-adaptive and structured-artifact suppression;
- candidate ice-band and representative-boundary extraction;
- attitude correction, sea-ice draft calculation, and quality-control output;
- Oculus `.oculus`-to-NetCDF wrapper based on the external `oculus-python`
  exporter;
- Oculus NetCDF and generic AUV-motion interfaces;
- synthetic input generator, example configuration, documentation, and tests.

## Intentionally excluded

- all field sonar, AUV, CTD, navigation, and hydrographic data;
- all formal figures, tables, animations, and reconstructed survey results;
- all observation-derived polygon masks and other manual annotations;
- experiment-specific compatibility entry points and historical notebooks;
- local absolute paths, user names, expedition identifiers, and internal
  dataset/version labels;
- caches, temporary files, local configurations, and archives.

## Current local verification (2026-09-16)

- Python syntax and CLI help passed in the existing Python 3.10 environment.
- Fifteen unit tests passed, covering conversion wrappers, coordinate geometry,
  beam reversal, aperture cropping, parameter validation/propagation, preserved
  center support defaults, and optional or time-bounded postprocessing.
- A 12-ping synthetic run completed with the fixed initial configuration.
- A 3-ping run with changed thresholds, grid, beam order and roll sign completed;
  disabling postprocessing preserved raw draft values in the processed column.
- A 3-ping run with two tuning workers completed using the supplied overrides.
- No field observations or real raw-log conversion were used for these checks.
- This update was checked in an existing environment; the clean installation
  results below belong to the earlier release and were not repeated.

## Historical release verification (2026-08-28)

- editable installation with the `oculus` extra passed in a newly created
  Python 3.10.20 virtual environment;
- the clean environment resolved `numpy 1.26.4`, `opencv-python 4.11.0.86`,
  and `oculus-python 0.0.2a1` without modifying installed package files;
- all Python files passed `compileall`, package metadata reported Version
  1.0.0, and both installed command-line entry points displayed help;
- seven standard-library unit tests passed, including NetCDF validation and a
  mocked end-to-end check of converter invocation, output validation, source
  preservation, and staged-file cleanup;
- a full 12-ping synthetic run completed through mapping, denoising,
  segmentation, representative-line extraction, attitude correction, draft
  calculation, and QC export;
- the clean-environment synthetic run produced 12 draft rows, with no no-ice
  or low-confidence candidates for the generated test case;
- `bps_oculus_io --help` passed in the clean environment. A field `.oculus`
  conversion was not rerun as part of the public smoke test because raw survey
  data are intentionally excluded.

The synthetic test verifies software integration only. It does not establish
accuracy or field performance.

## Before publishing

1. confirm the personal copyright name shown in `LICENSE`;
2. add formal citation metadata after the associated paper is published;
3. review `git status` and the staged diff;
4. run a final secret and large-file scan;
5. create the remote repository and push only after owner approval.
