#!/usr/bin/env python
"""
Backward mapping utilities for Oculus imaging-sonar NetCDF frames.

Coordinate convention used in this analysis:

- x_s: vehicle-forward direction (unresolved aperture direction).
- y_s: positive toward port.
- z_s: positive upward along the nominal center beam.

This is a right-handed sonar frame. The body frame is forward/starboard/down.
Verify the actual installation, beam ordering and attitude conventions before
using this approximation; the software cannot infer physical calibration.

The NetCDF image is a polar/fan frame, not a Cartesian image. For each ping,
``backscatter(sample, beam)`` is mapped onto a regular y_s-z_s grid by
backward mapping:

1. Build target Cartesian grid points (y_s, z_s).
2. Convert every target point to source polar coordinates (range, azimuth).
3. Convert source coordinates to fractional sample/beam indices.
4. Bilinearly interpolate the source ``backscatter`` matrix.

The 20 degree vertical aperture is not a resolved image dimension in the
NetCDF file. The first implementation maps the center plane, phi = 0, and
reports the aperture envelope as x_s uncertainty.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


InterpolationMode = Literal["linear", "nearest"]


@dataclass(frozen=True)
class SonarGridConfig:
    """Physical and numerical settings for a single-ping center-plane map."""

    range_min_m: float = 0.0
    range_max_m: float = 15.0
    grid_resolution_m: float = 0.05
    azimuth_half_angle_deg: float = 65.0
    vertical_aperture_deg: float = 20.0
    interpolation: InterpolationMode = "linear"
    beam_order: str = "starboard_to_port"
    outside_value: float = np.nan

    @property
    def vertical_half_angle_deg(self) -> float:
        return 0.5 * self.vertical_aperture_deg

    @property
    def azimuth_range_rad(self) -> float:
        return np.deg2rad(2.0 * self.azimuth_half_angle_deg)

    @property
    def azimuth_min_rad(self) -> float:
        return -np.deg2rad(self.azimuth_half_angle_deg)

    @property
    def azimuth_max_rad(self) -> float:
        return np.deg2rad(self.azimuth_half_angle_deg)


def open_oculus_dataset(nc_path: str | Path) -> xr.Dataset:
    """
    Open an Oculus NetCDF file without decoding time values eagerly.

    On Windows, netCDF4/HDF5 can fail to open long non-ASCII paths and may
    report that an existing file does not exist. When that happens, retry
    through an ASCII-only hardlink/copy path on the same drive.
    """
    path = Path(nc_path)
    try:
        return xr.open_dataset(path, decode_times=False)
    except Exception:
        if not _path_has_non_ascii(path):
            raise
        fallback_path = _make_ascii_read_path(path)
        return xr.open_dataset(fallback_path, decode_times=False)


def read_ping_frame(
    ds: xr.Dataset,
    ping_index: int,
    image_var: str = "backscatter",
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Read one ping as ``image[sample, beam]`` plus geometry metadata.

    The current Oculus export stores ``backscatter`` as ``beam, sample, ping``.
    This function also handles the more convenient ``sample, beam, ping`` order.
    """
    if image_var not in ds:
        raise KeyError(f"Image variable not found: {image_var}")

    arr = ds[image_var]
    if "ping" in arr.dims:
        n_ping = arr.sizes["ping"]
        if not 0 <= ping_index < n_ping:
            raise IndexError(f"ping_index={ping_index} outside [0, {n_ping - 1}]")
        frame = arr.isel(ping=ping_index)
    else:
        frame = arr
        while frame.ndim > 2:
            frame = frame.isel({frame.dims[0]: 0})

    sample_dim = "sample" if "sample" in frame.dims else frame.dims[-1]
    beam_dim = "beam" if "beam" in frame.dims else frame.dims[0]
    frame = frame.transpose(sample_dim, beam_dim)
    image = np.asarray(frame.values, dtype=float)

    sample_size_m = _read_scalar(ds, "sample_size", ping_index, default=1.0)
    declared_n_sample = int(round(_read_scalar(ds, "nsamples", ping_index, default=image.shape[0])))
    valid_n_sample = int(np.clip(declared_n_sample, 1, image.shape[0]))
    padded_n_sample, n_beam = image.shape
    if valid_n_sample < padded_n_sample:
        image = image[:valid_n_sample, :]

    n_sample = int(image.shape[0])
    if "azimuth_range" in ds:
        azimuth_range_rad = _read_scalar(ds, "azimuth_range", ping_index, default=np.deg2rad(130.0))
    else:
        azimuth_range_rad = np.deg2rad(130.0)

    metadata = {
        "ping_index": int(ping_index),
        "n_sample": int(n_sample),
        "declared_n_sample": int(declared_n_sample),
        "padded_n_sample": int(padded_n_sample),
        "n_beam": int(n_beam),
        "sample_size_m": float(sample_size_m),
        "azimuth_range_rad": float(azimuth_range_rad),
        "azimuth_range_deg": float(np.rad2deg(azimuth_range_rad)),
        "raw_range_max_m": float(n_sample * sample_size_m),
    }
    return image, metadata


def make_yz_grid(config: SonarGridConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular y_s-z_s target grid for center-plane backward mapping.

    The grid spans the full declared range box. Points outside the fan are
    later masked, so the output remains rectangular while invalid pixels are
    filled with ``outside_value``.
    """
    res = float(config.grid_resolution_m)
    if res <= 0:
        raise ValueError("grid_resolution_m must be positive.")

    y_axis = np.arange(-config.range_max_m, config.range_max_m + 0.5 * res, res)
    z_axis = np.arange(0.0, config.range_max_m + 0.5 * res, res)
    y_grid, z_grid = np.meshgrid(y_axis, z_axis)
    return y_grid, z_grid, y_axis, z_axis


def source_axes_from_metadata(
    metadata: dict[str, float],
    config: SonarGridConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return source range and azimuth centers.

    Range centers use ``(i + 0.5) * sample_size`` so interpolation is consistent
    with sample-bin centers. Beam centers are spread across the available
    azimuth range from negative y_s (starboard) to positive y_s (port).
    Source columns are reversed during mapping if their declared order differs.
    """
    n_sample = int(metadata["n_sample"])
    n_beam = int(metadata["n_beam"])
    sample_size = float(metadata["sample_size_m"])

    azimuth_range_rad = float(metadata["azimuth_range_rad"])
    # A smaller output fan crops source bearings; it must not compress them.

    range_centers = (np.arange(n_sample, dtype=float) + 0.5) * sample_size
    if n_beam <= 1:
        theta_centers = np.array([0.0], dtype=float)
    else:
        theta_centers = np.linspace(-0.5 * azimuth_range_rad, 0.5 * azimuth_range_rad, n_beam)
    return range_centers, theta_centers


def backward_map_ping_center_plane(
    image_sample_beam: np.ndarray,
    metadata: dict[str, float],
    config: SonarGridConfig = SonarGridConfig(),
) -> dict[str, np.ndarray | dict[str, float]]:
    """
    Backward-map one ping onto a regular y_s-z_s grid.

    Returns a dictionary with the mapped image, target axes, source coordinates,
    and validity mask. The image shape is ``(z, y)``.
    """
    if image_sample_beam.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape={image_sample_beam.shape}")

    if config.beam_order == "port_to_starboard":
        image_sample_beam = image_sample_beam[:, ::-1]
    elif config.beam_order != "starboard_to_port":
        raise ValueError("Unknown beam_order; verify the source column orientation")
    y_grid, z_grid, y_axis, z_axis = make_yz_grid(config)
    range_m = np.hypot(y_grid, z_grid)
    theta_rad = np.arctan2(y_grid, z_grid)

    raw_range_max = float(metadata["raw_range_max_m"])
    usable_range_max = min(float(config.range_max_m), raw_range_max)
    azimuth_half = min(np.deg2rad(config.azimuth_half_angle_deg), 0.5 * float(metadata["azimuth_range_rad"]))

    valid = (
        (range_m >= float(config.range_min_m))
        & (range_m <= usable_range_max)
        & (np.abs(theta_rad) <= azimuth_half)
        & (z_grid >= 0.0)
    )

    range_centers, theta_centers = source_axes_from_metadata(metadata, config)
    sample_index = _axis_to_fractional_index(range_m, range_centers)
    beam_index = _axis_to_fractional_index(theta_rad, theta_centers)

    if config.interpolation == "nearest":
        mapped = _nearest_interpolate(image_sample_beam, sample_index, beam_index, valid, config.outside_value)
    elif config.interpolation == "linear":
        mapped = _bilinear_interpolate(image_sample_beam, sample_index, beam_index, valid, config.outside_value)
    else:
        raise ValueError(f"Unsupported interpolation mode: {config.interpolation}")

    x_uncertainty_m = vertical_aperture_x_uncertainty(range_m, config)
    return {
        "mapped_backscatter": mapped,
        "valid_mask": valid,
        "y_grid_m": y_grid,
        "z_grid_m": z_grid,
        "y_axis_m": y_axis,
        "z_axis_m": z_axis,
        "source_range_m": range_m,
        "source_theta_rad": theta_rad,
        "x_uncertainty_m": x_uncertainty_m,
        "metadata": {
            **metadata,
            "usable_range_max_m": usable_range_max,
            "grid_resolution_m": float(config.grid_resolution_m),
            "vertical_aperture_deg": float(config.vertical_aperture_deg),
            "coordinate_frame": "right-handed: x_s forward, y_s port, z_s upward",
            "mapping_plane": "center plane phi=0, output image axes are z_s rows and y_s columns",
        },
    }


def vertical_aperture_x_uncertainty(range_m: np.ndarray | float, config: SonarGridConfig) -> np.ndarray:
    """Return half-width of the unresolved vertical aperture in x_s direction."""
    return np.asarray(range_m, dtype=float) * np.sin(np.deg2rad(config.vertical_half_angle_deg))


def fill_internal_holes(
    image: np.ndarray,
    valid_mask: np.ndarray,
    y_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    max_fill_distance_m: float = 0.15,
) -> dict[str, np.ndarray | dict[str, float]]:
    """
    Fill small internal holes in a backward-mapped Cartesian image.

    Only pixels inside ``valid_mask`` are eligible for filling. Pixels outside
    the insonified fan remain NaN in all returned images. Three comparable
    outputs are returned:

    - nearest: nearest-neighbor fill for holes within the distance limit.
    - linear: row/column linear fill on the regular mapped grid.
    - cubic: local row/column cubic polynomial fill on the regular mapped grid.
    """
    raw = np.asarray(image, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    y_axis = np.asarray(y_axis_m, dtype=float)
    z_axis = np.asarray(z_axis_m, dtype=float)

    if raw.shape != valid.shape:
        raise ValueError(f"image and valid_mask shape mismatch: {raw.shape} vs {valid.shape}")
    if raw.shape != (z_axis.size, y_axis.size):
        raise ValueError(
            "image shape must match (len(z_axis_m), len(y_axis_m)); "
            f"got {raw.shape}, axes=({z_axis.size}, {y_axis.size})"
        )
    if max_fill_distance_m < 0:
        raise ValueError("max_fill_distance_m must be non-negative.")

    observed = valid & np.isfinite(raw)
    holes = valid & ~np.isfinite(raw)
    if not np.any(observed):
        raise ValueError("No finite pixels inside the valid sonar fan; cannot interpolate holes.")

    dy = float(np.nanmean(np.diff(y_axis))) if y_axis.size > 1 else 1.0
    dz = float(np.nanmean(np.diff(z_axis))) if z_axis.size > 1 else 1.0
    nearest_distance_m, nearest_indices = distance_transform_edt(
        ~observed,
        sampling=(abs(dz), abs(dy)),
        return_indices=True,
    )
    small_holes = holes & (nearest_distance_m <= max_fill_distance_m)

    nearest_values = raw[nearest_indices[0], nearest_indices[1]]
    linear_values = np.full(raw.shape, np.nan, dtype=float)
    cubic_values = np.full(raw.shape, np.nan, dtype=float)
    if np.any(small_holes):
        row_linear = np.full(raw.shape, np.nan, dtype=float)
        col_linear = np.full(raw.shape, np.nan, dtype=float)
        col_index = np.arange(raw.shape[1], dtype=float)
        row_index = np.arange(raw.shape[0], dtype=float)
        local_radius_m = max(max_fill_distance_m * 2.0, max(abs(dy), abs(dz)) * 4.0)

        def local_cubic_value(
            axis_values: np.ndarray,
            samples: np.ndarray,
            finite_indices: np.ndarray,
            target_index: int,
        ) -> float:
            if finite_indices.size < 4:
                return np.nan
            target_coord = axis_values[target_index]
            distance = np.abs(axis_values[finite_indices] - target_coord)
            within = distance <= local_radius_m
            if np.count_nonzero(within) < 4:
                return np.nan
            candidate_indices = finite_indices[within]
            candidate_distance = distance[within]
            order = np.argsort(candidate_distance)[:8]
            fit_indices = candidate_indices[order]
            x = axis_values[fit_indices] - target_coord
            if np.unique(x).size < 4:
                return np.nan
            y = samples[fit_indices]
            try:
                coeff = np.polyfit(x, y, deg=3)
            except np.linalg.LinAlgError:
                return np.nan
            return float(np.polyval(coeff, 0.0))

        for row in np.unique(np.where(small_holes)[0]):
            finite_cols = np.where(observed[row, :])[0]
            if finite_cols.size >= 2:
                row_linear[row, :] = np.interp(
                    col_index,
                    finite_cols.astype(float),
                    raw[row, finite_cols],
                    left=np.nan,
                    right=np.nan,
                )

        for col in np.unique(np.where(small_holes)[1]):
            finite_rows = np.where(observed[:, col])[0]
            if finite_rows.size >= 2:
                col_linear[:, col] = np.interp(
                    row_index,
                    finite_rows.astype(float),
                    raw[finite_rows, col],
                    left=np.nan,
                    right=np.nan,
                )

        both_linear = np.isfinite(row_linear) & np.isfinite(col_linear)
        row_only = np.isfinite(row_linear) & ~np.isfinite(col_linear)
        col_only = ~np.isfinite(row_linear) & np.isfinite(col_linear)
        linear_values[both_linear] = 0.5 * (row_linear[both_linear] + col_linear[both_linear])
        linear_values[row_only] = row_linear[row_only]
        linear_values[col_only] = col_linear[col_only]

        for row, col in zip(*np.where(small_holes)):
            finite_cols = np.where(observed[row, :])[0]
            finite_rows = np.where(observed[:, col])[0]
            row_value = local_cubic_value(y_axis, raw[row, :], finite_cols, col)
            col_value = local_cubic_value(z_axis, raw[:, col], finite_rows, row)
            if np.isfinite(row_value) and np.isfinite(col_value):
                cubic_values[row, col] = 0.5 * (row_value + col_value)
            elif np.isfinite(row_value):
                cubic_values[row, col] = row_value
            elif np.isfinite(col_value):
                cubic_values[row, col] = col_value

        observed_min = float(np.nanmin(raw[observed]))
        observed_max = float(np.nanmax(raw[observed]))
        cubic_values = np.clip(cubic_values, observed_min, observed_max)

    nearest = raw.copy()
    nearest_fill_mask = small_holes & np.isfinite(nearest_values)
    nearest[nearest_fill_mask] = nearest_values[nearest_fill_mask]
    nearest[~valid] = np.nan

    linear = raw.copy()
    linear_fill_mask = small_holes & np.isfinite(linear_values)
    linear[linear_fill_mask] = linear_values[linear_fill_mask]
    linear[~valid] = np.nan

    cubic = raw.copy()
    cubic_fill_mask = small_holes & np.isfinite(cubic_values)
    cubic[cubic_fill_mask] = cubic_values[cubic_fill_mask]
    cubic[~valid] = np.nan

    valid_pixels = int(np.count_nonzero(valid))
    metrics = {
        "valid_pixels": valid_pixels,
        "raw_internal_holes": int(np.count_nonzero(holes)),
        "small_internal_holes": int(np.count_nonzero(small_holes)),
        "raw_internal_nan_fraction": (
            int(np.count_nonzero(holes)) / valid_pixels if valid_pixels else np.nan
        ),
        "nearest_fill_fraction": (
            int(np.count_nonzero(nearest_fill_mask)) / valid_pixels if valid_pixels else np.nan
        ),
        "linear_fill_fraction": (
            int(np.count_nonzero(linear_fill_mask)) / valid_pixels if valid_pixels else np.nan
        ),
        "cubic_fill_fraction": (
            int(np.count_nonzero(cubic_fill_mask)) / valid_pixels if valid_pixels else np.nan
        ),
        "max_fill_distance_m": float(max_fill_distance_m),
    }

    return {
        "raw": raw,
        "nearest": nearest,
        "linear": linear,
        "cubic": cubic,
        "nearest_fill_mask": nearest_fill_mask,
        "linear_fill_mask": linear_fill_mask,
        "cubic_fill_mask": cubic_fill_mask,
        "nearest_distance_m": nearest_distance_m,
        "small_hole_mask": small_holes,
        "metrics": metrics,
    }


def sonar_center_plane_points(
    bottom_range_m: np.ndarray,
    theta_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert polar bottom picks to center-plane sonar-local points.

    This is useful after extracting an ice-bottom range for each beam:
    x_s remains zero because the vertical aperture is unresolved.
    """
    r = np.asarray(bottom_range_m, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    x_s = np.zeros_like(r)
    y_s = r * np.sin(theta)
    z_s = r * np.cos(theta)
    return x_s, y_s, z_s


def rotation_matrix_zyx(
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """
    Build a yaw-heading, pitch, roll rotation matrix using Rz * Ry * Rx.

    This helper is provided for the later survey-coordinate stage. The sign
    convention must be checked against the AUV NAV log before using it for
    final ice-draft products.
    """
    h, p, r = np.deg2rad([heading_deg, pitch_deg, roll_deg])
    ch, sh = np.cos(h), np.sin(h)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    rz = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _read_scalar(ds: xr.Dataset, name: str, ping_index: int, default: float) -> float:
    if name not in ds:
        return float(default)
    da = ds[name]
    try:
        if "ping" in da.dims:
            return float(da.isel(ping=ping_index).values)
        return float(da.values)
    except Exception:
        return float(default)


def _path_has_non_ascii(path: Path) -> bool:
    return any(ord(ch) > 127 for ch in str(path))


def _make_ascii_read_path(nc_path: Path) -> Path:
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    ascii_dir = Path(tempfile.gettempdir()) / "imaging_sonar_ice_draft_input"
    ascii_dir.mkdir(parents=True, exist_ok=True)

    ascii_path = ascii_dir / nc_path.name
    if ascii_path.exists() and ascii_path.stat().st_size == nc_path.stat().st_size:
        return ascii_path

    if ascii_path.exists():
        ascii_path.unlink()

    try:
        os.link(nc_path, ascii_path)
    except Exception:
        shutil.copy2(nc_path, ascii_path)
    return ascii_path


def _axis_to_fractional_index(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    if axis.size == 1:
        return np.zeros_like(values, dtype=float)
    return np.interp(values, axis, np.arange(axis.size, dtype=float), left=np.nan, right=np.nan)


def _nearest_interpolate(
    image: np.ndarray,
    sample_index: np.ndarray,
    beam_index: np.ndarray,
    valid: np.ndarray,
    outside_value: float,
) -> np.ndarray:
    out = np.full(sample_index.shape, outside_value, dtype=float)
    si = np.rint(sample_index).astype(int)
    bi = np.rint(beam_index).astype(int)
    in_bounds = valid & (si >= 0) & (si < image.shape[0]) & (bi >= 0) & (bi < image.shape[1])
    out[in_bounds] = image[si[in_bounds], bi[in_bounds]]
    return out


def _bilinear_interpolate(
    image: np.ndarray,
    sample_index: np.ndarray,
    beam_index: np.ndarray,
    valid: np.ndarray,
    outside_value: float,
) -> np.ndarray:
    out = np.full(sample_index.shape, outside_value, dtype=float)
    finite = valid & np.isfinite(sample_index) & np.isfinite(beam_index)
    if not np.any(finite):
        return out

    max_s = image.shape[0] - 1
    max_b = image.shape[1] - 1
    in_bounds = finite & (sample_index >= 0.0) & (sample_index <= max_s) & (beam_index >= 0.0) & (beam_index <= max_b)
    if not np.any(in_bounds):
        return out

    s0 = np.floor(sample_index[in_bounds]).astype(int)
    b0 = np.floor(beam_index[in_bounds]).astype(int)
    s1 = np.minimum(s0 + 1, max_s)
    b1 = np.minimum(b0 + 1, max_b)

    ds = sample_index[in_bounds] - s0
    db = beam_index[in_bounds] - b0

    v00 = image[s0, b0]
    v10 = image[s1, b0]
    v01 = image[s0, b1]
    v11 = image[s1, b1]

    interp = (
        (1.0 - ds) * (1.0 - db) * v00
        + ds * (1.0 - db) * v10
        + (1.0 - ds) * db * v01
        + ds * db * v11
    )
    out[in_bounds] = interp
    return out
