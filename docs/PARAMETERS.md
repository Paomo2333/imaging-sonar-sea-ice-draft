# Adjustable initial parameters / 可调初始参数

`config/initial_parameters.json` contains all exposed algorithm settings.
The file is optional: omitted fields use the same built-in initial values.
Supply overrides with `--algorithm-config`, or the `algorithm_config` path in
the run JSON. Unknown keys, invalid types and inconsistent limits fail early.

| Group | Controls | Units / notes |
| --- | --- | --- |
| `grid` | Mapping, beam order, range scale, hole fill | metres, degrees; `range_scale` is dimensionless |
| `background` | Range bins, percentiles, normalization | `low_q/high_q` are 0–100; smoothing is in range bins |
| `gray` | Weak-pixel floor and artifact attenuation | normalized intensity / dimensionless factors |
| `denoising` | Artifact geometry, projection, adaptive protection bands | `_m` metres; `_m2` m²; `_deg` degrees; pixel counts are integers |
| `guidance` | Refinement passes, residual-background trigger | count and score |
| `binary` | Fixed candidate window, quantiles, morphology and support | metres, m², 0–100 percentiles |
| `binary_refinement` | Mask bridging, edge extension, curve gap limits | explicit separate limits for masks and curves |
| `selection` | `fixed` or optional `tune`; search lists | tuning changes only the listed candidate fields |
| `attitude` | Input roll/pitch signs and in-plane tilt | calibrated signs ±1, tilt in degrees |
| `center` | Window and support counts | metres / point counts; existing fallback rule retained |
| `postprocess` | Optional temporal QC, interpolation and smoothing | metres, point counts, seconds |

Initial candidate settings are `high_q=86`, `low_q=66`, `window_down_m=1.20`,
`window_up_m=0.60`, `min_area_m2=0.030`, and `min_y_span_m=0.65`.
These are editable starting values, not site-independent physical constants.
Geometry and sampling calibration must be checked separately.

## Partial override example

```json
{
  "grid": {"beam_order": "port_to_starboard"},
  "binary": {"high_q": 84.0, "low_q": 62.0, "window_down_m": 1.4},
  "postprocess": {"enabled": false}
}
```

The beam-order change is appropriate only if verified against the input data.
Do not copy it as a universal recommendation. Range limits, file paths, time
selection and plot switches remain in the run configuration.

## Temporal options

`enabled=false` disables temporal QC and smoothing; raw values are retained in
the processed columns for output compatibility. `interpolate=false` leaves
rejected values missing. `max_gap_s=null` preserves the original unlimited,
index-based interpolation, including endpoint filling. A finite `max_gap_s`
limits interpolation to internally bracketed samples whose bounding timestamps
are within that duration, uses their timestamps, and prevents smoothing across
larger time breaks. Unfilled values remain missing. Window lengths are counts of
processed pings, so consider sampling interval before changing them.

## Traceability and implementation scope

Every run writes `effective_config.json` with fully resolved algorithm values,
run options and final selected parameters. Fresh selection prevents stale cached
settings from overriding edits; `--force-retune` remains a compatibility flag.
Spatial geometry is stored in metres and discretized to pixels as required.
The JSON exposes processing parameters, not every plotting constant or internal
score coefficient. The CLI initializes one parameter set per process; use
separate processes for concurrent independent configurations.

## 中文说明

所有列出的配置均可调整，省略项使用初始值。请先核实设备坐标和标定，再调整与数据质量有关的处理参数。`fixed` 为默认模式；使用 `tune` 时，请同时检查搜索列表，固定参数中对应的字段会被搜索值覆盖。初始值不代表普适最优参数。建议保留自己的配置文件及每次运行生成的 `effective_config.json`，并区分 `ice_draft_m` 原始估计与后处理列。
