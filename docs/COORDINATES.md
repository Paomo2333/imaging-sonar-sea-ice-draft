# Coordinate and calibration checks / 坐标与标定检查

## Canonical right-handed frames

| Frame | +x | +y | +z |
| --- | --- | --- | --- |
| Body | Forward | Starboard | Down |
| Sonar, nominal upward mounting | Forward | Port | Up |

For the nominal mounting the sonar-to-body rotation is `diag(1, -1, -1)`.
The center-plane approximation sets `x_s = 0`; unresolved out-of-plane arrival
angles are not recovered. Positive sonar bearing points toward port:
`y_s = r sin(bearing)`, `z_s = r cos(bearing)`.

## Determine these from your own equipment

1. **Beam ordering.** `grid.beam_order: "starboard_to_port"` means source column
   indices run from negative bearing/starboard to positive bearing/port.
   Choose `"port_to_starboard"` to reverse the source columns before mapping.
   This describes physical input order, not how a screen displays left/right.
   The current adapter assumes uniformly spaced beams over `azimuth_range`;
   use a dedicated adapter for explicit nonuniform beam bearings.
2. **Angles.** `azimuth_range` is a full aperture in radians. Navigation roll
   and pitch are in degrees. The body attitude convention follows a right-handed
   forward/starboard/down frame and Z-Y-X Euler rotation. Initial roll/pitch
   signs are +1; set -1 only when the input convention requires conversion.
   The program no longer chooses roll sign by making the ice surface flatter.
3. **Installation.** Verify the nominal 180-degree rotation about the forward
   axis. The optional tilt parameter is a scalar in-plane approximation, not
   a general three-axis mounting/lever-arm calibration. Adapt the transform
   for other installations. Do not merely rename plot axes.
4. **Range.** `sample_size` must represent metres per sample bin. Confirm the
   acquisition/export sound speed. `grid.range_scale` initially equals 1 and
   multiplies the range-bin size once. Use a target/reference sound-speed
   ratio only if the source range definition supports that correction and it
   has not already been applied upstream. No site-specific sound speed is
   assumed. `grid.azimuth_half_angle_deg` crops the output fan without changing
   the physical source beam bearings.
5. **Time and depth.** Confirm UTC timestamps and clock offsets. Supply AUV
   depth positive downward relative to the sea surface, referenced to the
   transducer or corrected for known sensor offsets. Navigation is interpolated
   to ping time; the current reader holds endpoint values outside its support,
   so restrict processing to valid navigation coverage.
6. **Physical check.** Review an identifiable off-axis echo or a calibrated
   target, verify the left/right orientation and tilt response, and inspect
   representative raw and corrected images before batch interpretation.

The scalar geometry for the nominal mounting is:

```text
phi = roll_sign * input_roll
theta = tilt_from_vertical + pitch_sign * input_pitch
y_c = y_s cos(phi) - z_s sin(phi)
z_c = y_s sin(phi) + z_s cos(phi)
upward_range = z_c cos(theta)
draft = AUV_depth - upward_range
```

All trigonometric calculations internally convert degrees to radians. The
coordinate conversion, range scale and initial thresholds must be checked for
each deployment; plausible-looking output is not evidence of correct calibration.

## 中文提示

请根据声呐说明书、原始数据字段、实际安装记录和可判方向的目标，逐项确认声束排列、角度定义、姿态符号、距离标定、时间同步和深度参考。坐标变换应同时作用于数据、公式及图轴说明。设置 `beam_order` 可反转输入列；设置姿态符号可转换约定，但不能替代未知安装关系的标定。已有人工多边形掩膜若使用了另一横轴方向，也必须转换到本仓库的坐标后再使用。
