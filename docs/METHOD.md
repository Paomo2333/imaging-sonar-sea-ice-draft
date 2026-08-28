# Method overview

This repository implements a center-plane, per-ping reconstruction workflow
for upward-looking two-dimensional imaging sonar.

Raw Blueprint Subsea Oculus logs may first be exported to NetCDF4 through the
external `oculus-python` command-line tool. That file-format conversion is an
input preprocessing step rather than part of the ice-bottom extraction
algorithm described below.

## Processing sequence

1. **Polar-to-Cartesian mapping**

   Each sonar frame is represented by echo intensity as a function of slant
   range and azimuth. Target pixels on a regular lateral–vertical grid are
   mapped back to fractional range and beam indices and evaluated by bilinear
   interpolation. The default grid spacing is 0.05 m.

2. **Range-adaptive background suppression**

   Valid pixels are grouped in 0.15 m slant-range bins. Local 25th and 75th
   percentiles characterize the range-dependent background and intensity
   spread. The resulting threshold curve is smoothed before being mapped back
   to the image grid.

3. **Structured-artifact suppression**

   The implementation identifies and attenuates a narrow central-beam artifact
   and compact lobe-shaped responses below the candidate ice band. These masks
   are based on image geometry and intensity contrast; they do not assign a
   unique physical origin to the artifacts.

4. **Candidate ice-band detection**

   A vertical projection combines normalized echo energy and strong-pixel
   coverage with weights 0.65 and 0.35. Its principal peak limits the local
   search window. Hysteresis-style thresholding, connected-component filters,
   and light morphology produce the candidate ice-bottom region.

5. **Representative boundary extraction**

   At each lateral grid position, the representative vertical coordinate is
   calculated from the intensity-weighted candidate pixels. Only short gaps
   satisfying horizontal-span and vertical-continuity constraints are bridged;
   longer unsupported intervals remain missing.

6. **Attitude correction and draft calculation**

   The extracted boundary is corrected using synchronized AUV roll and pitch.
   A robust statistic within the central lateral window provides the upward
   vertical range. Sea-ice draft is then estimated as AUV depth minus this
   vertical range. Quality-control fields retain geometric support, temporal
   residuals, and review-priority indicators.

## Method boundary

The sonar resolves range and azimuth within its fan-shaped imaging plane but
does not resolve arrival angle within the non-imaging aperture. Echoes are
therefore projected onto the center plane. The method reconstructs an
along-track series of representative sea-ice draft; it is not a complete
three-dimensional reconstruction of the ice underside and does not estimate
total sea-ice thickness.

Parameter values are research defaults rather than universal constants. They
should be evaluated for the sonar model, operating range, environment, and
signal-to-noise characteristics of each new dataset.
