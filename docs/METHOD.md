# Method overview

This repository illustrates a general center-plane processing workflow for
upward-looking two-dimensional imaging sonar. It is a runnable reference
implementation with adjustable initial parameters, not a complete survey-specific
reproduction package. Users supply and validate their own observations.

## Processing sequence

1. **Geometry and mapping.** Convert calibrated range/bearing samples to a regular
   lateral/upward grid by backward bilinear mapping. Initial spacing is 0.05 m.
   This is a numerical grid interval, not a measurement-accuracy specification.
   Beam order, range scale, aperture, resolution and interpolation are configurable.
2. **Range-dependent background.** Initial range bins are 0.25 m, with lower/upper
   percentiles 35/74 and a spread coefficient of 0.55. Missing bin estimates are
   interpolated. `smooth_bins=5` corresponds to Gaussian sigma 5/3 bins under
   the current implementation. Positive residuals are normalized by Q99.
3. **Artifact suppression.** Geometric and intensity criteria suppress narrow
   central artifacts and lower lobe-like responses. A preliminary guidance curve
   protects plausible interface support during iterative refinement. It is not
   the final boundary used for draft. Parameters are grouped in the JSON file.
4. **Candidate detection.** A normalized projection combines echo energy and
   strong-pixel support with initial weights 0.65/0.35. Its dominant peak anchors
   an initial window extending 1.20 m downward and 0.60 m upward. Local per-ping
   intensity thresholds use initial percentiles 86/66, followed by morphology,
   connected-component screening and echo-supported mask bridges. The initial
   component area/span settings are 0.030 m² and 0.65 m. Optional tuning searches
   user-defined ranges; the default applies fixed initial settings.
5. **Representative boundary.** At each lateral column, candidate pixels define
   an intensity-weighted mean elevation. Short curve gaps are interpolated only
   within configurable horizontal/vertical limits (initially 0.60/0.35 m).
   Mask bridging and final curve interpolation are separate operations.
6. **Attitude and draft.** Follow the right-handed conventions in
   [COORDINATES.md](COORDINATES.md), apply roll/pitch correction and subtract the
   upward range from AUV depth. Initial center statistics use a 0.50 m half-window
   and a weighted median. The existing support rule is retained: fewer than five
   center points triggers a nearest-point fallback of up to nine points, which
   may extend outside the window. `center_method` identifies this in the output.
7. **Optional postprocessing.** Users may adjust or disable temporal QC,
   interpolation and smoothing according to their own data quality. Raw and
   processed draft columns are separate; effective settings are saved per run.

## Interpretation boundary

Two-dimensional sonar resolves range and in-plane bearing but not the individual
arrival angle within its out-of-plane aperture. Returns are approximated as
lying on the center plane. This estimates representative near-track sea-ice
draft; it does not recover a complete 3D underside or total ice thickness.
Complex or vertically dispersed returns require further methodological study
and independent validation. QC scores are review aids, not calibrated uncertainty.

All initial values should be assessed for the user's sensor, operating range,
ice morphology and signal quality. See [PARAMETERS.md](PARAMETERS.md).
