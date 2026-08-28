# Input format

## Oculus raw log preprocessing

The supplied preprocessing wrapper accepts one Blueprint Subsea Oculus
`.oculus` log and calls the external `bps_oculus_io` exporter from
`oculus-python 0.0.2a1`:

```bash
oculus-to-netcdf --input path/to/input.oculus --output-dir path/to/output
```

The output is validated before it is passed to the reconstruction pipeline.
Raw-log version 1 is supported by the upstream package; its version 2 support
is currently marked experimental. The wrapper contains no Oculus binary parser
and does not modify the installed third-party package.

## Imaging-sonar NetCDF

The command-line pipeline expects one NetCDF file containing:

| Variable | Required | Expected form | Meaning |
| --- | --- | --- | --- |
| `backscatter` | yes | `beam × sample × ping` or `sample × beam × ping` | Echo intensity |
| `time` | yes | `ping` | Unix time in seconds |
| `sample_size` | recommended | scalar or `ping` | Range-bin size in metres |
| `nsamples` | recommended | scalar or `ping` | Valid range-bin count before padding |
| `azimuth_range` | optional | scalar or `ping` | Full azimuth aperture in radians |

If `sample_size`, `nsamples`, or `azimuth_range` is absent, the reader uses a
fallback value. Check these metadata carefully before interpreting distances.
This variable layout matches the NetCDF output currently produced by
`oculus-python`; metadata and dimension order should still be inspected for
each software and sonar version.

## Synchronized AUV motion CSV

The motion file must contain `time_unix_s` and should contain:

| Column | Unit | Use |
| --- | --- | --- |
| `time_unix_s` | s | Synchronization with sonar pings |
| `roll_filtered_deg` | degree | Roll correction |
| `pitch_filtered_deg` | degree | Vertical-range projection |
| `NAV_Heading` | degree | Retained navigation context |
| `NAV_DEPTH` | m | AUV depth used in draft calculation |
| `NAV_ALTITUDE` | m | Optional contextual field |

Rows are sorted by time, duplicate timestamps are removed, and navigation
values are linearly interpolated to sonar ping times. The user is responsible
for confirming clock alignment, sign conventions, sensor offsets, and whether
the supplied depth is referenced to the sea surface.

## Data policy

No field observations, navigation records, manual annotations, or derived
survey products are included. The generator in `examples/` creates artificial
inputs only for software testing and does not represent real ice morphology.
