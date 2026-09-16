from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    gaussian_filter,
    gaussian_filter1d,
    label,
    map_coordinates,
)


LOCAL_TZ = "UTC"

OCULUS_CMAP = LinearSegmentedColormap.from_list(
    "oculus_amber",
    ["#000000", "#140b00", "#5c3400", "#c98200", "#ffd447", "#fffdf2"],
    N=256,
)

ICE_MIN_Z_M = 6.0
ICE_BAND_HALF_WIDTH_M = 0.55
CENTER_HALF_WIDTH_M = 0.34
CENTER_HALF_ANGLE_DEG = 2.4
CENTER_SIDE_INNER_M = 0.52
CENTER_SIDE_OUTER_M = 1.55
CENTER_SIDE_PERCENTILE = 70.0
CENTER_EXCESS = 0.10
CENTER_SIDE_PRESERVE_LEVEL = 0.55
LOBE_Z_MARGIN_M = 1.25
LOBE_MAX_WIDTH_M = 7.0
COMPONENT_THRESHOLD = 0.24
PROJECTION_PIXEL_THRESHOLD = 0.35
BELOW_ICE_MARGIN_MIN_M = 1.4
BELOW_ICE_MARGIN_MAX_M = 4.2
BELOW_ICE_MARGIN_BASE_M = 1.5
BELOW_ICE_MARGIN_GAP_GAIN_M = 1.2
BELOW_ICE_MARGIN_ROUGHNESS_GAIN = 1.8
ABOVE_ICE_MARGIN_MIN_M = 1.8
ABOVE_ICE_MARGIN_MAX_M = 5.8
ABOVE_ICE_MARGIN_BASE_M = 2.0
ABOVE_ICE_MARGIN_GAP_GAIN_M = 1.2
ABOVE_ICE_MARGIN_ROUGHNESS_GAIN = 1.4
CURVE_CENTER_PROTECT_M = 0.75
CURVE_CENTER_MIN_LEVEL = 0.035

# Additional exposed geometric and projection settings (configured per CLI run).
CENTER_REFERENCE_Z_M = 1.0
CENTER_BELOW_MARGIN_M = 0.15
LOBE_TOP_MARGIN_M = 0.45
LOBE_MAX_AREA_M2 = 1.2
LOBE_MIN_PIXELS = 20
LOBE_CLOSING_ROWS = 3
LOBE_CLOSING_COLS = 5
PROJECTION_ENERGY_WEIGHT = 0.65
PROJECTION_SUPPORT_WEIGHT = 0.35
PROJECTION_SIGMA_M = 0.15

GRAY_CONFIGS = [
    {"name": "conservative", "weak_floor": 0.05, "post_floor": 0.04, "center_min_level": 0.25, "center_attenuation": 0.35, "lobe_attenuation": 0.45},
    {"name": "balanced", "weak_floor": 0.08, "post_floor": 0.06, "center_min_level": 0.25, "center_attenuation": 0.22, "lobe_attenuation": 0.30},
    {"name": "strong", "weak_floor": 0.11, "post_floor": 0.08, "center_min_level": 0.28, "center_attenuation": 0.08, "lobe_attenuation": 0.14},
    {"name": "center_focused", "weak_floor": 0.08, "post_floor": 0.06, "center_min_level": 0.22, "center_attenuation": 0.04, "lobe_attenuation": 0.24},
    {"name": "high_threshold", "weak_floor": 0.13, "post_floor": 0.10, "center_min_level": 0.32, "center_attenuation": 0.02, "lobe_attenuation": 0.05},
]
DEFAULT_GRAY_CONFIG = next(row for row in GRAY_CONFIGS if row["name"] == "high_threshold")


@dataclass
class FrameResult:
    ping: int
    ping_time_unix_s: float
    ping_time_local: str
    range_max_m: float
    sample_size_m: float
    nsamples: int
    nbeams: int
    azimuth_range_deg: float
    y_axis: np.ndarray
    z_axis: np.ndarray
    linear_image: np.ndarray
    first_denoised: np.ndarray
    denoised: np.ndarray
    suppression_mask: np.ndarray
    score_map: np.ndarray
    projection_score: np.ndarray
    projection_smooth: np.ndarray
    selected_ice_z_m: float
    gray_config_name: str
    gray_score: float
    ice_retention: float
    center_residual: float
    lobe_residual: float
    weak_annular_remaining: float
    below_ice_margin_m: float
    below_ice_mask_fraction: float
    above_ice_margin_m: float
    above_ice_mask_fraction: float
    line_y: np.ndarray
    line_z: np.ndarray
    line_status: np.ndarray
    observed_fraction: float
    center_z_s_m: float
    center_status: str
    frame_status: str
    confidence: float
    annular_score: float
    annular_second_pass: bool
    selected_component_count: int
    selected_median_z_m: float
    selected_y_span_m: float
    selected_mean_score: float
    candidate_count: int


def open_oculus_dataset(nc_path: str | Path) -> xr.Dataset:
    return xr.open_dataset(Path(nc_path), decode_times=False, engine="netcdf4")


def ping_time_unix(ds: xr.Dataset, ping: int) -> float:
    return float(np.asarray(ds["time"].isel(ping=ping).values))


def ping_time_local_string(ping_time_s: float) -> str:
    ts = pd.to_datetime(ping_time_s, unit="s", utc=True).tz_convert(LOCAL_TZ)
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def ping_range_metadata(ds: xr.Dataset, ping: int) -> dict:
    nsamples = int(np.asarray(ds["nsamples"].isel(ping=ping).values))
    nbeams = int(np.asarray(ds["nbeams"].isel(ping=ping).values))
    sample_size = float(np.asarray(ds["sample_size"].isel(ping=ping).values))
    azimuth_range_raw = float(np.asarray(ds["azimuth_range"].isel(ping=ping).values))
    azimuth_range = float(np.rad2deg(azimuth_range_raw)) if azimuth_range_raw <= 2.0 * np.pi else azimuth_range_raw
    return {
        "nsamples": nsamples,
        "nbeams": nbeams,
        "sample_size_m": sample_size,
        "azimuth_range_deg": azimuth_range,
        "azimuth_range_raw": azimuth_range_raw,
        "range_max_m": nsamples * sample_size,
    }


def read_ping_image(ds: xr.Dataset, ping: int, image_var: str = "backscatter") -> np.ndarray:
    meta = ping_range_metadata(ds, ping)
    image = np.asarray(ds[image_var].isel(ping=ping).values, dtype=float)
    return image[: meta["nsamples"], : meta["nbeams"]]


def variable_range_grid(range_max_m: float, azimuth_range_deg: float, resolution_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_angle = np.deg2rad(azimuth_range_deg / 2.0)
    y_extent = range_max_m * np.sin(half_angle)
    y_axis = np.arange(-y_extent, y_extent + 0.5 * resolution_m, resolution_m)
    z_axis = np.arange(0.0, range_max_m + 0.5 * resolution_m, resolution_m)
    y_grid, z_grid = np.meshgrid(y_axis, z_axis)
    return y_axis, z_axis, y_grid, z_grid


def backward_map_ping(
    ds: xr.Dataset,
    ping: int,
    *,
    image_var: str = "backscatter",
    resolution_m: float = 0.08,
) -> dict:
    meta = ping_range_metadata(ds, ping)
    image = read_ping_image(ds, ping, image_var=image_var)
    y_axis, z_axis, y_grid, z_grid = variable_range_grid(meta["range_max_m"], meta["azimuth_range_deg"], resolution_m)

    sample_size = meta["sample_size_m"]
    nbeams = meta["nbeams"]
    half_angle_rad = np.deg2rad(meta["azimuth_range_deg"] / 2.0)
    r_grid = np.hypot(y_grid, z_grid)
    theta_grid = np.arctan2(y_grid, z_grid)

    sample_coord = r_grid / sample_size
    beam_coord = (theta_grid + half_angle_rad) / (2.0 * half_angle_rad) * (nbeams - 1)
    valid = (
        np.isfinite(sample_coord)
        & np.isfinite(beam_coord)
        & (sample_coord >= 0)
        & (sample_coord <= meta["nsamples"] - 1)
        & (beam_coord >= 0)
        & (beam_coord <= nbeams - 1)
        & (np.abs(theta_grid) <= half_angle_rad)
    )

    mapped = map_coordinates(
        image,
        [sample_coord, beam_coord],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    mapped[~valid] = np.nan
    return {
        **meta,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "y_grid": y_grid,
        "z_grid": z_grid,
        "r_grid": r_grid,
        "valid_mask": valid,
        "linear_image": mapped,
    }


def normalize_positive(values: np.ndarray, valid_mask: np.ndarray | None = None, q: float = 99.2) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if valid_mask is None:
        valid_mask = np.isfinite(out)
    finite = np.isfinite(out) & valid_mask
    if finite.sum() == 0:
        return np.zeros_like(out, dtype=float)
    scale = np.nanpercentile(out[finite], q)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(out[finite])
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(out, dtype=float)
    out = np.clip(out / scale, 0.0, 1.0)
    out[~finite] = 0.0
    return out


def normalized_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmax = np.nanmax(values) if np.any(np.isfinite(values)) else np.nan
    if not np.isfinite(vmax) or vmax <= 0:
        return np.zeros_like(values, dtype=float)
    return values / vmax


def center_artifact_core_mask(y_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    angular_half_width = np.tan(np.deg2rad(CENTER_HALF_ANGLE_DEG)) * np.maximum(z_grid, CENTER_REFERENCE_Z_M)
    half_width = np.maximum(CENTER_HALF_WIDTH_M, angular_half_width)
    return np.abs(y_grid) <= half_width


def compute_z_projection(image: np.ndarray, valid_mask: np.ndarray, z_axis: np.ndarray, dz_m: float) -> tuple[np.ndarray, np.ndarray]:
    finite = valid_mask & np.isfinite(image)
    projection_sum = np.nansum(np.where(finite, image, 0.0), axis=1)
    projection_count = np.sum(finite & (image >= PROJECTION_PIXEL_THRESHOLD), axis=1).astype(float)
    weight_sum = PROJECTION_ENERGY_WEIGHT + PROJECTION_SUPPORT_WEIGHT
    projection_score = (PROJECTION_ENERGY_WEIGHT * normalized_profile(projection_sum) + PROJECTION_SUPPORT_WEIGHT * normalized_profile(projection_count)) / weight_sum
    smooth_sigma = max(1.0, PROJECTION_SIGMA_M / max(abs(dz_m), 1e-6)) if PROJECTION_SIGMA_M > 0 else 0.0
    projection_smooth = gaussian_filter1d(np.nan_to_num(projection_score, nan=0.0), sigma=smooth_sigma, mode="nearest") if smooth_sigma > 0 else np.nan_to_num(projection_score, nan=0.0)
    return projection_score, projection_smooth


def estimate_ice_z_from_projection(projection_smooth: np.ndarray, z_axis: np.ndarray) -> float:
    valid_rows = np.isfinite(projection_smooth) & (z_axis >= ICE_MIN_Z_M)
    if np.any(valid_rows):
        rows = np.flatnonzero(valid_rows)
        return float(z_axis[rows[int(np.nanargmax(projection_smooth[rows]))]])
    if np.any(np.isfinite(projection_smooth)):
        return float(z_axis[int(np.nanargmax(projection_smooth))])
    return float("nan")


def safe_energy(image: np.ndarray, mask: np.ndarray) -> float:
    values = image[mask & np.isfinite(image)]
    return float(np.nansum(values)) if values.size else 0.0


def build_noise_context(adaptive_image: np.ndarray, valid_mask: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> dict:
    y_grid, z_grid = np.meshgrid(y_axis, z_axis)
    dy = float(np.nanmean(np.diff(y_axis))) if len(y_axis) > 1 else 1.0
    dz = float(np.nanmean(np.diff(z_axis))) if len(z_axis) > 1 else 1.0
    projection_score, projection_smooth = compute_z_projection(adaptive_image, valid_mask, z_axis, dz)
    ice_z = estimate_ice_z_from_projection(projection_smooth, z_axis)
    center_core = center_artifact_core_mask(y_grid, z_grid)
    masks = {
        "center_core": center_core,
        "center_nonice": center_core & (z_grid < ice_z - ICE_BAND_HALF_WIDTH_M),
        "lower_lobe_region": (z_grid < ice_z - LOBE_Z_MARGIN_M) & (np.abs(y_grid) <= 0.35 * max(float(np.nanmax(np.abs(y_axis))), 1.0)),
        "ice_band": valid_mask & (np.abs(z_grid - ice_z) <= ICE_BAND_HALF_WIDTH_M),
        "weak_annular": valid_mask & np.isfinite(adaptive_image) & (adaptive_image < 0.18),
    }
    baseline = {
        "ice_energy": safe_energy(adaptive_image, masks["ice_band"]),
        "center_energy": safe_energy(adaptive_image, masks["center_nonice"]),
        "lower_energy": safe_energy(adaptive_image, masks["lower_lobe_region"]),
        "weak_pixels": int(np.count_nonzero(masks["weak_annular"])),
        "finite_pixels": int(np.count_nonzero(np.isfinite(adaptive_image) & valid_mask)),
    }
    return {
        "y_grid": y_grid,
        "z_grid": z_grid,
        "dy": dy,
        "dz": dz,
        "initial_ice_z": ice_z,
        "projection_score": projection_score,
        "projection_smooth": projection_smooth,
        "masks": masks,
        "baseline": baseline,
    }


def run_unified_noise_suppression(
    adaptive_image: np.ndarray,
    valid_mask: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    config: dict | None = None,
) -> dict:
    if config is None:
        config = DEFAULT_GRAY_CONFIG

    context = build_noise_context(adaptive_image, valid_mask, y_axis, z_axis)
    cleaned = np.asarray(adaptive_image, dtype=float).copy()
    cleaned[~valid_mask] = 0.0
    y_grid = context["y_grid"]
    z_grid = context["z_grid"]
    ice_z = context["initial_ice_z"]

    weak_mask = valid_mask & np.isfinite(cleaned) & (cleaned < config["weak_floor"])
    cleaned[weak_mask] = 0.0

    side_cols = (np.abs(y_axis) >= CENTER_SIDE_INNER_M) & (np.abs(y_axis) <= CENTER_SIDE_OUTER_M)
    side_support = np.zeros(cleaned.shape[0], dtype=float)
    if np.count_nonzero(side_cols) >= 2:
        side_values = cleaned[:, side_cols]
        for row in range(cleaned.shape[0]):
            row_values = side_values[row, :]
            finite_row = np.isfinite(row_values) & (row_values > 0)
            if np.count_nonzero(finite_row) >= 3:
                side_support[row] = np.nanpercentile(row_values[finite_row], CENTER_SIDE_PERCENTILE)

    center_threshold = np.maximum(side_support + CENTER_EXCESS, config["center_min_level"])
    center_artifact_mask = (
        context["masks"]["center_core"]
        & np.isfinite(cleaned)
        & (cleaned > center_threshold[:, None])
        & (side_support[:, None] < CENTER_SIDE_PRESERVE_LEVEL)
        & (z_grid < ice_z - CENTER_BELOW_MARGIN_M)
    )
    cleaned[center_artifact_mask] = cleaned[center_artifact_mask] * config["center_attenuation"]

    binary = valid_mask & np.isfinite(cleaned) & (cleaned >= COMPONENT_THRESHOLD)
    binary = binary_closing(binary, structure=np.ones((LOBE_CLOSING_ROWS, LOBE_CLOSING_COLS), dtype=bool))
    component_labels, component_count = label(binary)
    lobe_mask = np.zeros_like(binary, dtype=bool)
    for idx in range(1, component_count + 1):
        rows, cols = np.where(component_labels == idx)
        if rows.size < LOBE_MIN_PIXELS:
            continue
        z_values = z_axis[rows]
        y_values = y_axis[cols]
        z_max = float(np.nanmax(z_values))
        z_centroid = float(np.nanmedian(z_values))
        y_width = float(np.nanmax(y_values) - np.nanmin(y_values)) if y_values.size else 0.0
        area_m2 = float(rows.size * abs(context["dy"]) * abs(context["dz"]))
        below_ice = (z_max < ice_z - LOBE_TOP_MARGIN_M) or (z_centroid < ice_z - LOBE_Z_MARGIN_M)
        compact_or_low = (y_width <= LOBE_MAX_WIDTH_M) or (area_m2 < LOBE_MAX_AREA_M2)
        center_column_like = (abs(float(np.nanmedian(y_values))) <= CENTER_HALF_WIDTH_M * 1.8) and (y_width <= 1.2)
        if below_ice and (compact_or_low or center_column_like):
            lobe_mask[rows, cols] = True
    cleaned[lobe_mask] = cleaned[lobe_mask] * config["lobe_attenuation"]

    post_floor_mask = valid_mask & np.isfinite(cleaned) & (cleaned < config["post_floor"])
    cleaned[post_floor_mask] = 0.0
    cleaned[~valid_mask] = 0.0
    cleaned = np.clip(cleaned, 0.0, 1.0)

    suppress_mask = np.full(cleaned.shape, np.nan, dtype=float)
    suppress_mask[weak_mask | post_floor_mask] = 1.0
    suppress_mask[center_artifact_mask] = 2.0
    suppress_mask[lobe_mask] = 3.0

    baseline = context["baseline"]
    masks = context["masks"]
    ice_retention = safe_energy(cleaned, masks["ice_band"]) / baseline["ice_energy"] if baseline["ice_energy"] else np.nan
    center_residual = safe_energy(cleaned, masks["center_nonice"]) / baseline["center_energy"] if baseline["center_energy"] else 0.0
    lobe_residual = safe_energy(cleaned, masks["lower_lobe_region"]) / baseline["lower_energy"] if baseline["lower_energy"] else 0.0
    weak_remaining = int(np.count_nonzero((cleaned > 0) & masks["weak_annular"])) / baseline["weak_pixels"] if baseline["weak_pixels"] else 0.0
    finite_ratio = int(np.count_nonzero((cleaned > 0) & valid_mask)) / baseline["finite_pixels"] if baseline["finite_pixels"] else np.nan
    gray_score = (
        0.55 * (ice_retention if np.isfinite(ice_retention) else 0.0)
        - 0.28 * center_residual
        - 0.22 * lobe_residual
        - 0.10 * weak_remaining
        - 0.04 * abs((finite_ratio if np.isfinite(finite_ratio) else 0.55) - 0.55)
    )

    return {
        "cleaned": cleaned,
        "suppress_mask": suppress_mask,
        "context": context,
        "config": dict(config),
        "metrics": {
            "gray_score": float(gray_score),
            "ice_retention": float(ice_retention) if np.isfinite(ice_retention) else np.nan,
            "center_residual": float(center_residual),
            "lobe_residual": float(lobe_residual),
            "weak_annular_remaining": float(weak_remaining),
            "finite_ratio": float(finite_ratio) if np.isfinite(finite_ratio) else np.nan,
            "selected_ice_z_m": float(ice_z),
            "center_mask_pixels": int(np.count_nonzero(center_artifact_mask)),
            "lobe_mask_pixels": int(np.count_nonzero(lobe_mask)),
            "weak_mask_pixels": int(np.count_nonzero(weak_mask | post_floor_mask)),
        },
    }


def adaptive_range_threshold(
    image: np.ndarray,
    r_grid: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bin_m: float = 0.25,
    low_q: float = 35.0,
    high_q: float = 74.0,
    iqr_gain: float = 0.55,
    smooth_bins: int = 5,
    min_bin_pixels: int = 25,
    normalization_q: float = 99.0,
) -> tuple[np.ndarray, dict]:
    finite = np.isfinite(image) & valid_mask
    if finite.sum() == 0:
        return np.zeros_like(image, dtype=float), {"range_bin_count": 0}

    rmax = float(np.nanmax(r_grid[finite]))
    edges = np.arange(0.0, rmax + bin_m, bin_m)
    if len(edges) < 2:
        return np.zeros_like(image, dtype=float), {"range_bin_count": 0}

    centers = 0.5 * (edges[:-1] + edges[1:])
    thresholds = np.full(len(centers), np.nan, dtype=float)
    bin_index = np.digitize(r_grid, edges) - 1

    for i in range(len(centers)):
        m = finite & (bin_index == i)
        if m.sum() < min_bin_pixels:
            continue
        values = image[m]
        lo = np.nanpercentile(values, low_q)
        hi = np.nanpercentile(values, high_q)
        thresholds[i] = hi + iqr_gain * max(0.0, hi - lo)

    good = np.isfinite(thresholds)
    if good.sum() == 0:
        threshold_map = np.full_like(image, np.nanpercentile(image[finite], high_q), dtype=float)
    else:
        thresholds = np.interp(np.arange(len(thresholds)), np.flatnonzero(good), thresholds[good])
        if smooth_bins > 1:
            thresholds = gaussian_filter1d(thresholds, sigma=max(1.0, smooth_bins / 3.0), mode="nearest")
        threshold_map = np.interp(r_grid.ravel(), centers, thresholds, left=thresholds[0], right=thresholds[-1]).reshape(image.shape)

    residual = np.maximum(image - threshold_map, 0.0)
    residual[~finite] = 0.0
    normalized = normalize_positive(residual, finite, q=normalization_q)
    return normalized, {
        "range_bin_count": int(len(centers)),
        "threshold_median": float(np.nanmedian(thresholds)) if good.sum() else float("nan"),
    }


def build_score_map(denoised: np.ndarray, valid_mask: np.ndarray, resolution_m: float) -> np.ndarray:
    clean = np.nan_to_num(denoised, nan=0.0)
    smooth = gaussian_filter(clean, sigma=1.0)
    grad_z = np.abs(np.gradient(smooth, resolution_m, axis=0))
    grad_score = normalize_positive(grad_z, valid_mask, q=99.0)
    score = 0.70 * clean + 0.30 * grad_score
    score[~valid_mask] = 0.0
    return np.clip(score, 0.0, 1.0)


def connected_components_from_score(
    score: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    valid_mask: np.ndarray,
    range_max_m: float,
    *,
    q: float = 91.0,
    min_score: float = 0.10,
) -> tuple[np.ndarray, list[dict]]:
    finite = valid_mask & np.isfinite(score)
    if finite.sum() == 0:
        return np.zeros_like(score, dtype=int), []

    threshold = max(float(np.nanpercentile(score[finite], q)), min_score)
    binary = (score >= threshold) & finite
    binary = binary_closing(binary, structure=np.ones((3, 5), dtype=bool))
    binary = binary_opening(binary, structure=np.ones((2, 3), dtype=bool))
    labels, count = label(binary)

    yy, zz = np.meshgrid(y_axis, z_axis)
    components: list[dict] = []
    pixel_area = float(abs(y_axis[1] - y_axis[0]) * abs(z_axis[1] - z_axis[0])) if len(y_axis) > 1 and len(z_axis) > 1 else 1.0
    for idx in range(1, count + 1):
        m = labels == idx
        area_px = int(m.sum())
        if area_px < 20:
            continue
        y_values = yy[m]
        z_values = zz[m]
        values = score[m]
        y_span = float(np.nanmax(y_values) - np.nanmin(y_values))
        z_span = float(np.nanmax(z_values) - np.nanmin(z_values))
        median_z = float(np.nanmedian(z_values))
        mean_score = float(np.nanmean(values))
        p90_score = float(np.nanpercentile(values, 90))
        band_likeness = float(y_span / max(z_span, 0.5))
        component_score = mean_score * min(1.0, y_span / 5.0) * min(1.0, area_px / 160.0)
        if y_span < 1.2:
            continue
        if median_z < max(1.5, 0.20 * range_max_m):
            continue
        components.append(
            {
                "label": idx,
                "area_px": area_px,
                "area_m2": area_px * pixel_area,
                "y_span_m": y_span,
                "z_span_m": z_span,
                "median_z_m": median_z,
                "band_likeness": band_likeness,
                "mean_score": mean_score,
                "p90_score": p90_score,
                "component_score": component_score,
                "y_min_m": float(np.nanmin(y_values)),
                "y_max_m": float(np.nanmax(y_values)),
                "z_min_m": float(np.nanmin(z_values)),
                "z_max_m": float(np.nanmax(z_values)),
            }
        )
    components.sort(key=lambda row: row["component_score"], reverse=True)
    return labels, components


def select_lower_candidate_group(components: list[dict], *, z_merge_m: float = 4.0) -> tuple[list[int], dict | None, str]:
    if not components:
        return [], None, "no_candidate"
    best_score = max(row["component_score"] for row in components)
    best = components[0]
    eligible = [
        row
        for row in components
        if row["component_score"] >= 0.45 * best_score and row["y_span_m"] >= 1.5
    ]
    if not eligible:
        return [], None, "no_candidate"

    band_like = [row for row in eligible if row.get("band_likeness", 0.0) >= 0.85]
    if band_like:
        eligible = band_like

    lower_than_best = [
        row
        for row in eligible
        if row["median_z_m"] < best["median_z_m"] - 1.0
        and row["component_score"] >= 0.65 * best_score
        and row["y_span_m"] >= 0.55 * best["y_span_m"]
    ]
    if lower_than_best:
        # When a plausible upper false line and lower ice-bottom band coexist,
        # choose the lower-z band; otherwise keep the strongest ice-like band.
        selected = min(lower_than_best, key=lambda row: row["median_z_m"])
        status = "double_candidate_lower_selected"
    else:
        selected = max(eligible, key=lambda row: row["component_score"])
        status = "candidate_selected" if selected["label"] == components[0]["label"] else "ice_band_selected"

    labels = [
        row["label"]
        for row in eligible
        if row["median_z_m"] <= selected["median_z_m"] + z_merge_m
        and row["component_score"] >= 0.35 * best_score
        and row.get("band_likeness", 0.0) >= 0.55
    ]
    return labels, selected, status


def extract_line_from_labels(
    labels: np.ndarray,
    selected_labels: list[int],
    score: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    *,
    smooth_sigma_cols: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not selected_labels:
        return np.array([]), np.array([]), np.array([], dtype=object), 0.0, float("nan")

    selected_mask = np.isin(labels, selected_labels)
    observed_z = np.full(len(y_axis), np.nan, dtype=float)
    for col in range(len(y_axis)):
        column_mask = selected_mask[:, col]
        if not np.any(column_mask):
            continue
        weights = score[:, col].copy()
        weights[~column_mask] = 0.0
        if np.nanmax(weights) <= 0:
            continue
        observed_z[col] = z_axis[int(np.nanargmax(weights))]

    observed_cols = np.flatnonzero(np.isfinite(observed_z))
    if len(observed_cols) < 3:
        return np.array([]), np.array([]), np.array([], dtype=object), 0.0, float("nan")

    first, last = int(observed_cols[0]), int(observed_cols[-1])
    line_y = y_axis[first : last + 1]
    interp_cols = np.arange(first, last + 1)
    line_z = np.interp(interp_cols, observed_cols, observed_z[observed_cols])
    status = np.where(np.isfinite(observed_z[first : last + 1]), "observed", "gap_interpolated").astype(object)
    if smooth_sigma_cols > 0 and len(line_z) > 5:
        line_z = gaussian_filter1d(line_z, sigma=smooth_sigma_cols, mode="nearest")

    observed_fraction = float(np.mean(status == "observed"))
    center_z = float(np.interp(0.0, line_y, line_z)) if line_y.min() <= 0.0 <= line_y.max() else float("nan")
    return line_y, line_z, status, observed_fraction, center_z


def huber_penalty(value: float | np.ndarray, delta: float) -> float | np.ndarray:
    value = np.asarray(value, dtype=float)
    abs_value = np.abs(value)
    penalty = np.where(abs_value <= delta, 0.5 * abs_value**2 / max(delta, 1e-9), abs_value - 0.5 * delta)
    return penalty


def find_column_ridge_candidates(
    score: np.ndarray,
    selected_mask: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    *,
    max_candidates_per_col: int = 5,
    min_candidate_score: float = 0.08,
    include_segment_centroids: bool = True,
) -> list[list[dict]]:
    candidates_by_col: list[list[dict]] = []
    for col in range(len(y_axis)):
        column_mask = selected_mask[:, col] & np.isfinite(score[:, col]) & (score[:, col] >= min_candidate_score)
        rows = np.flatnonzero(column_mask)
        if rows.size == 0:
            candidates_by_col.append([])
            continue

        values = score[:, col]
        local_rows = []
        for row in rows:
            left = values[row - 1] if row > 0 and column_mask[row - 1] else -np.inf
            right = values[row + 1] if row + 1 < len(z_axis) and column_mask[row + 1] else -np.inf
            if values[row] >= left and values[row] >= right:
                local_rows.append(row)
        if not local_rows:
            local_rows = [int(rows[int(np.nanargmax(values[rows]))])]

        if include_segment_centroids:
            breaks = np.flatnonzero(np.diff(rows) > 1)
            segment_starts = np.r_[0, breaks + 1]
            segment_ends = np.r_[breaks + 1, len(rows)]
            for start, end in zip(segment_starts, segment_ends):
                segment_rows = rows[start:end]
                if segment_rows.size < 3:
                    continue
                segment_scores = np.clip(values[segment_rows], 0.0, None)
                weights = segment_scores**2
                if np.nansum(weights) <= 0:
                    continue
                z_center = float(np.nansum(z_axis[segment_rows] * weights) / np.nansum(weights))
                centroid_row = int(segment_rows[int(np.nanargmin(np.abs(z_axis[segment_rows] - z_center)))])
                local_rows.append(centroid_row)

        local_rows = sorted(set(local_rows), key=lambda row: float(values[row]), reverse=True)[:max_candidates_per_col]
        candidates_by_col.append(
            [
                {
                    "col": col,
                    "row": int(row),
                    "y": float(y_axis[col]),
                    "z": float(z_axis[row]),
                    "score": float(values[row]),
                }
                for row in local_rows
            ]
        )
    return candidates_by_col


def extract_line_dynamic_programming(
    score: np.ndarray,
    valid_mask: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    range_max_m: float,
    *,
    ridge_image: np.ndarray | None = None,
    z_hint_m: float | None = None,
    search_half_width_m: float = 3.0,
    reference_line_y: np.ndarray | None = None,
    reference_line_z: np.ndarray | None = None,
    reference_half_width_m: float = 1.4,
    max_candidates_per_col: int = 5,
    min_candidate_score: float = 0.08,
    data_weight: float = 1.0,
    jump_weight: float = 3.2,
    curvature_weight: float = 1.1,
    gap_weight: float = 0.20,
    lower_candidate_weight: float = 0.08,
) -> dict:
    if ridge_image is None:
        ridge_score = np.asarray(score, dtype=float)
    else:
        ridge_score = 0.75 * np.nan_to_num(ridge_image, nan=0.0) + 0.25 * np.nan_to_num(score, nan=0.0)
        ridge_score[~valid_mask] = 0.0
        ridge_score = np.clip(ridge_score, 0.0, 1.0)

    labels, components = connected_components_from_score(ridge_score, y_axis, z_axis, valid_mask, range_max_m)
    selected_labels, selected_component, lower_status = select_lower_candidate_group(components)
    z_hint = float(z_hint_m) if z_hint_m is not None and np.isfinite(z_hint_m) else float("nan")
    if np.isfinite(z_hint):
        _, z_grid = np.meshgrid(y_axis, z_axis)
        selected_mask = valid_mask & np.isfinite(ridge_score) & (ridge_score >= min_candidate_score)
        selected_mask &= np.abs(z_grid - z_hint) <= search_half_width_m
        lower_status = "projection_guided_search_band"
        selected_component = None
        selected_labels = []
    elif selected_labels:
        selected_mask = np.isin(labels, selected_labels)
    else:
        return {
            "line_y": np.array([]),
            "line_z": np.array([]),
            "line_status": np.array([], dtype=object),
            "center_z_s_m": float("nan"),
            "observed_fraction": 0.0,
            "candidate_points": pd.DataFrame(),
            "metrics": {"status": "no_candidate"},
        }

    if reference_line_y is not None and reference_line_z is not None and len(reference_line_y) >= 3:
        reference_line_y = np.asarray(reference_line_y, dtype=float)
        reference_line_z = np.asarray(reference_line_z, dtype=float)
        reference_curve = np.interp(y_axis, reference_line_y, reference_line_z, left=np.nan, right=np.nan)
        _, z_grid = np.meshgrid(y_axis, z_axis)
        reference_grid = np.tile(reference_curve[None, :], (len(z_axis), 1))
        reference_support = np.isfinite(reference_grid) & (np.abs(z_grid - reference_grid) <= reference_half_width_m)
        selected_mask &= reference_support
        lower_status = f"{lower_status}+reference_corridor"

    candidates_by_col = find_column_ridge_candidates(
        ridge_score,
        selected_mask,
        y_axis,
        z_axis,
        max_candidates_per_col=max_candidates_per_col,
        min_candidate_score=min_candidate_score,
    )
    active_cols = [col for col, candidates in enumerate(candidates_by_col) if candidates]
    if len(active_cols) < 3:
        return {
            "line_y": np.array([]),
            "line_z": np.array([]),
            "line_status": np.array([], dtype=object),
            "center_z_s_m": float("nan"),
            "observed_fraction": 0.0,
            "candidate_points": pd.DataFrame(),
            "metrics": {"status": "too_few_candidates", "active_columns": len(active_cols)},
        }

    dy_nominal = float(np.nanmedian(np.diff(y_axis))) if len(y_axis) > 1 else 0.08
    dy_nominal = abs(dy_nominal) if np.isfinite(dy_nominal) and dy_nominal != 0 else 0.08

    def data_cost(candidate: dict) -> float:
        score_term = -np.log(np.clip(candidate["score"], 1e-5, 1.0))
        lower_term = lower_candidate_weight * candidate["z"] / max(range_max_m, 1e-6)
        return data_weight * score_term + lower_term

    first_col = active_cols[0]
    first_candidates = candidates_by_col[first_col]
    if len(active_cols) == 1:
        best = first_candidates[int(np.argmin([data_cost(c) for c in first_candidates]))]
        return {
            "line_y": np.array([best["y"]]),
            "line_z": np.array([best["z"]]),
            "line_status": np.array(["observed_dp"], dtype=object),
            "center_z_s_m": float("nan"),
            "observed_fraction": 1.0,
            "candidate_points": pd.DataFrame(first_candidates),
            "metrics": {"status": "single_column"},
        }

    second_col = active_cols[1]
    second_candidates = candidates_by_col[second_col]
    states: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}
    for i, cand0 in enumerate(first_candidates):
        for j, cand1 in enumerate(second_candidates):
            dy = max(abs(cand1["y"] - cand0["y"]), dy_nominal)
            dz = cand1["z"] - cand0["z"]
            jump_delta = max(0.35, 2.5 * dy)
            cost = data_cost(cand0) + data_cost(cand1) + jump_weight * float(huber_penalty(dz, jump_delta))
            cost += gap_weight * max(0.0, dy - dy_nominal)
            states[(i, j)] = (float(cost), [(first_col, i), (second_col, j)])

    prev_prev_col = first_col
    prev_col = second_col
    for col in active_cols[2:]:
        current_candidates = candidates_by_col[col]
        new_states: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}
        for (prevprev_idx, prev_idx), (state_cost, path) in states.items():
            cand_prevprev = candidates_by_col[prev_prev_col][prevprev_idx]
            cand_prev = candidates_by_col[prev_col][prev_idx]
            dy1 = max(abs(cand_prev["y"] - cand_prevprev["y"]), dy_nominal)
            slope1 = (cand_prev["z"] - cand_prevprev["z"]) / dy1
            for cur_idx, cand_cur in enumerate(current_candidates):
                dy2 = max(abs(cand_cur["y"] - cand_prev["y"]), dy_nominal)
                dz = cand_cur["z"] - cand_prev["z"]
                slope2 = dz / dy2
                jump_delta = max(0.35, 2.5 * dy2)
                curvature_delta = 2.2
                transition = jump_weight * float(huber_penalty(dz, jump_delta))
                transition += curvature_weight * float(huber_penalty(slope2 - slope1, curvature_delta))
                transition += gap_weight * max(0.0, dy2 - dy_nominal)
                cost = state_cost + data_cost(cand_cur) + transition
                key = (prev_idx, cur_idx)
                if key not in new_states or cost < new_states[key][0]:
                    new_states[key] = (float(cost), path + [(col, cur_idx)])
        states = new_states
        prev_prev_col, prev_col = prev_col, col

    best_cost, best_path = min(states.values(), key=lambda item: item[0])
    observed_cols = np.array([col for col, _ in best_path], dtype=int)
    observed_z = np.array([candidates_by_col[col][idx]["z"] for col, idx in best_path], dtype=float)
    observed_score = np.array([candidates_by_col[col][idx]["score"] for col, idx in best_path], dtype=float)

    first = int(observed_cols[0])
    last = int(observed_cols[-1])
    interp_cols = np.arange(first, last + 1)
    line_y = y_axis[interp_cols]
    line_z = np.interp(interp_cols, observed_cols, observed_z)
    status = np.where(np.isin(interp_cols, observed_cols), "observed_dp", "gap_interpolated").astype(object)

    center_z = float(np.interp(0.0, line_y, line_z)) if line_y.min() <= 0.0 <= line_y.max() else float("nan")
    jumps = np.abs(np.diff(observed_z))
    curvature = np.abs(np.diff(observed_z, n=2)) if len(observed_z) >= 3 else np.array([], dtype=float)
    selected_pairs = {(col, candidates_by_col[col][idx]["row"]) for col, idx in best_path}
    candidate_points = [
        {
            **candidate,
            "selected": (candidate["col"], candidate["row"]) in selected_pairs,
        }
        for column_candidates in candidates_by_col
        for candidate in column_candidates
    ]

    return {
        "line_y": line_y,
        "line_z": line_z,
        "line_status": status,
        "center_z_s_m": center_z,
        "observed_fraction": float(np.mean(status == "observed_dp")),
        "candidate_points": pd.DataFrame(candidate_points),
        "metrics": {
            "status": "ok",
            "path_cost": float(best_cost),
            "candidate_columns": int(len(active_cols)),
            "candidate_points": int(sum(len(candidates_by_col[col]) for col in active_cols)),
            "selected_component_count": int(len(selected_labels)),
            "selection_status": lower_status,
            "selected_median_z_m": float(selected_component["median_z_m"]) if selected_component else float("nan"),
            "mean_selected_score": float(np.nanmean(observed_score)) if observed_score.size else 0.0,
            "max_jump_m": float(np.nanmax(jumps)) if jumps.size else 0.0,
            "p95_jump_m": float(np.nanpercentile(jumps, 95)) if jumps.size else 0.0,
            "mean_curvature_m": float(np.nanmean(curvature)) if curvature.size else 0.0,
        },
    }


def annular_residual_metrics(
    denoised: np.ndarray,
    r_grid: np.ndarray,
    y_axis: np.ndarray,
    z_grid: np.ndarray,
    valid_mask: np.ndarray,
    line_y: np.ndarray,
    line_z: np.ndarray,
    *,
    bin_m: float = 0.35,
    protect_m: float = 1.0,
    threshold: float = 0.08,
) -> dict:
    finite = valid_mask & np.isfinite(denoised)
    if finite.sum() == 0:
        return {"annular_score": 0.0, "max_coverage": 0.0, "mean_coverage": 0.0}

    protect = np.zeros_like(denoised, dtype=bool)
    if len(line_y) >= 3:
        line_interp = np.interp(y_axis, line_y, line_z, left=np.nan, right=np.nan)
        line_grid = np.tile(line_interp[None, :], (denoised.shape[0], 1))
        protect = np.isfinite(line_grid) & (np.abs(z_grid - line_grid) <= protect_m)

    work_mask = finite & ~protect
    if work_mask.sum() < 50:
        return {"annular_score": 0.0, "max_coverage": 0.0, "mean_coverage": 0.0}

    rmax = float(np.nanmax(r_grid[work_mask]))
    edges = np.arange(0.0, rmax + bin_m, bin_m)
    bin_index = np.digitize(r_grid, edges) - 1
    coverages = []
    means = []
    for i in range(len(edges) - 1):
        m = work_mask & (bin_index == i)
        if m.sum() < 30:
            continue
        vals = denoised[m]
        coverages.append(float(np.mean(vals > threshold)))
        means.append(float(np.nanmean(vals)))
    if not coverages:
        return {"annular_score": 0.0, "max_coverage": 0.0, "mean_coverage": 0.0}
    max_cov = float(np.nanmax(coverages))
    mean_cov = float(np.nanmean(coverages))
    mean_level = float(np.nanmean(means))
    annular_score = float(np.clip(0.65 * max_cov + 1.8 * mean_level + 0.35 * mean_cov, 0.0, 1.0))
    return {"annular_score": annular_score, "max_coverage": max_cov, "mean_coverage": mean_cov, "mean_level": mean_level}


def secondary_annular_suppression(
    denoised: np.ndarray,
    r_grid: np.ndarray,
    y_axis: np.ndarray,
    z_grid: np.ndarray,
    valid_mask: np.ndarray,
    line_y: np.ndarray,
    line_z: np.ndarray,
    *,
    bin_m: float = 0.35,
    protect_m: float = 1.0,
) -> np.ndarray:
    out = np.asarray(denoised, dtype=float).copy()
    finite = valid_mask & np.isfinite(out)
    protect = np.zeros_like(out, dtype=bool)
    if len(line_y) >= 3:
        line_interp = np.interp(y_axis, line_y, line_z, left=np.nan, right=np.nan)
        line_grid = np.tile(line_interp[None, :], (out.shape[0], 1))
        protect = np.isfinite(line_grid) & (np.abs(z_grid - line_grid) <= protect_m)
    work_mask = finite & ~protect
    if work_mask.sum() < 50:
        return out

    rmax = float(np.nanmax(r_grid[work_mask]))
    edges = np.arange(0.0, rmax + bin_m, bin_m)
    bin_index = np.digitize(r_grid, edges) - 1
    for i in range(len(edges) - 1):
        m = work_mask & (bin_index == i)
        if m.sum() < 30:
            continue
        vals = out[m]
        coverage = float(np.mean(vals > 0.08))
        if coverage < 0.22:
            continue
        floor = float(np.nanpercentile(vals, 62.0))
        out[m] = np.maximum(vals - 0.85 * floor, 0.0)
    out[~finite] = 0.0
    return normalize_positive(out, finite, q=99.0)


def classify_frame_status(
    *,
    range_max_m: float,
    candidate_count: int,
    confidence: float,
    center_status: str,
    lower_selection_status: str,
    annular_second_pass: bool,
) -> str:
    if candidate_count == 0:
        if range_max_m < 20.0:
            return "out_of_range_possible"
        return "weak_or_ambiguous"
    if confidence < 0.22:
        if range_max_m < 20.0:
            return "out_of_range_possible"
        return "weak_or_ambiguous"
    labels = []
    if lower_selection_status == "double_candidate_lower_selected":
        labels.append("double_candidate_lower_selected")
    if center_status == "gap_interpolated":
        labels.append("central_gap_interpolated")
    if annular_second_pass:
        labels.append("secondary_annular_suppression")
    if not labels:
        labels.append("valid_ice_observed")
    return "+".join(labels)


def sample_score_along_line(
    score: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    line_y: np.ndarray,
    line_z: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor sampling of a 2-D score map along an extracted curve."""
    if len(line_y) == 0:
        return np.array([], dtype=float)
    y_idx = np.searchsorted(y_axis, line_y)
    z_idx = np.searchsorted(z_axis, line_z)
    y_idx = np.clip(y_idx, 0, len(y_axis) - 1)
    z_idx = np.clip(z_idx, 0, len(z_axis) - 1)
    return score[z_idx, y_idx]


def safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def active_echo_metrics(result: FrameResult, *, active_floor: float = 0.12, strong_floor: float = 0.35) -> dict:
    valid = np.isfinite(result.denoised) & (result.linear_image > 0)
    active = valid & (result.denoised >= active_floor)
    strong = valid & (result.denoised >= strong_floor)

    valid_count = int(np.count_nonzero(valid))
    active_count = int(np.count_nonzero(active))
    strong_count = int(np.count_nonzero(strong))

    active_cols = np.flatnonzero(np.any(active, axis=0))
    if active_cols.size:
        active_y_span_m = float(result.y_axis[int(active_cols[-1])] - result.y_axis[int(active_cols[0])])
    else:
        active_y_span_m = 0.0

    return {
        "active_echo_px": active_count,
        "strong_echo_px": strong_count,
        "active_echo_fraction": float(active_count / valid_count) if valid_count else 0.0,
        "strong_echo_fraction": float(strong_count / valid_count) if valid_count else 0.0,
        "active_echo_y_span_m": active_y_span_m,
    }


def select_final_line(result: FrameResult, dp: dict) -> dict:
    metrics = dict(dp.get("metrics", {}))
    echo = active_echo_metrics(result)

    orig_center = safe_float(result.center_z_s_m)
    dp_center = safe_float(dp.get("center_z_s_m"))
    orig_has_center = np.isfinite(orig_center)
    dp_has_center = np.isfinite(dp_center)

    candidate_columns = int(safe_float(metrics.get("candidate_columns", 0), 0.0))
    candidate_points = int(safe_float(metrics.get("candidate_points", 0), 0.0))
    mean_selected_score = safe_float(metrics.get("mean_selected_score", 0.0), 0.0)
    p95_jump_m = safe_float(metrics.get("p95_jump_m", float("inf")), float("inf"))
    max_jump_m = safe_float(metrics.get("max_jump_m", float("inf")), float("inf"))
    dp_observed_fraction = safe_float(dp.get("observed_fraction", 0.0), 0.0)
    center_delta_m = abs(dp_center - orig_center) if orig_has_center and dp_has_center else float("nan")

    min_support_columns = max(55, int(0.12 * len(result.y_axis)))
    weak_frame = result.frame_status in {"out_of_range_possible", "weak_or_ambiguous"}
    weak_dp_support = (
        candidate_columns < min_support_columns
        or candidate_points < 120
        or mean_selected_score < 0.55
        or echo["active_echo_y_span_m"] < max(2.5, 0.12 * result.range_max_m)
    )

    if weak_frame and weak_dp_support:
        return {
            **echo,
            "line_decision": "no_data",
            "line_decision_reason": "weak_frame_and_insufficient_echo_support",
            "use_for_reconstruction": False,
            "final_line_source": "none",
            "final_center_z_s_m": float("nan"),
            "center_delta_m": center_delta_m,
        }

    if (not orig_has_center) and weak_dp_support:
        return {
            **echo,
            "line_decision": "no_data",
            "line_decision_reason": "no_original_center_and_insufficient_dp_support",
            "use_for_reconstruction": False,
            "final_line_source": "none",
            "final_center_z_s_m": float("nan"),
            "center_delta_m": center_delta_m,
        }

    orig_good = (
        orig_has_center
        and result.confidence >= 0.32
        and result.observed_fraction >= 0.55
        and result.selected_y_span_m >= max(3.5, 0.18 * result.range_max_m)
    )
    dp_good = (
        metrics.get("status") == "ok"
        and dp_has_center
        and candidate_columns >= min_support_columns
        and dp_observed_fraction >= 0.55
        and mean_selected_score >= 0.58
        and p95_jump_m <= 0.45
        and max_jump_m <= 2.0
    )

    close_to_original = orig_has_center and dp_has_center and center_delta_m <= 0.40
    moderate_difference = orig_has_center and dp_has_center and center_delta_m <= 0.70
    has_gap_or_complexity = "central_gap" in result.frame_status or "double_candidate" in result.frame_status

    if dp_good and close_to_original:
        return {
            **echo,
            "line_decision": "dp_enhanced",
            "line_decision_reason": "dp_good_and_consistent_with_original",
            "use_for_reconstruction": True,
            "final_line_source": "dp",
            "final_center_z_s_m": dp_center,
            "center_delta_m": center_delta_m,
        }

    if dp_good and has_gap_or_complexity and moderate_difference:
        return {
            **echo,
            "line_decision": "dp_enhanced",
            "line_decision_reason": "dp_good_for_gap_or_complex_ice",
            "use_for_reconstruction": True,
            "final_line_source": "dp",
            "final_center_z_s_m": dp_center,
            "center_delta_m": center_delta_m,
        }

    side_gap_connected = (
        orig_good
        and "central_gap" in result.frame_status
        and result.center_status == "gap_interpolated"
        and (not moderate_difference)
    )
    if side_gap_connected:
        return {
            **echo,
            "line_decision": "side_gap_connected",
            "line_decision_reason": "center_gap_connected_from_two_side_ice",
            "use_for_reconstruction": True,
            "final_line_source": "original_gap_connected",
            "final_center_z_s_m": orig_center,
            "center_delta_m": center_delta_m,
        }

    if orig_good and dp_good and not moderate_difference:
        return {
            **echo,
            "line_decision": "review_conflict",
            "line_decision_reason": "original_and_dp_disagree",
            "use_for_reconstruction": False,
            "final_line_source": "review",
            "final_center_z_s_m": orig_center,
            "center_delta_m": center_delta_m,
        }

    if orig_good:
        return {
            **echo,
            "line_decision": "original_baseline",
            "line_decision_reason": "original_line_more_reliable",
            "use_for_reconstruction": True,
            "final_line_source": "original",
            "final_center_z_s_m": orig_center,
            "center_delta_m": center_delta_m,
        }

    if dp_good:
        return {
            **echo,
            "line_decision": "dp_review",
            "line_decision_reason": "dp_only_candidate_needs_review",
            "use_for_reconstruction": False,
            "final_line_source": "review",
            "final_center_z_s_m": dp_center,
            "center_delta_m": center_delta_m,
        }

    return {
        **echo,
        "line_decision": "no_data",
        "line_decision_reason": "no_reliable_line_candidate",
        "use_for_reconstruction": False,
        "final_line_source": "none",
        "final_center_z_s_m": float("nan"),
        "center_delta_m": center_delta_m,
    }


def adaptive_below_ice_margin(
    line_y: np.ndarray,
    line_z: np.ndarray,
    line_status: np.ndarray,
) -> tuple[float, dict]:
    if len(line_y) < 3 or len(line_z) < 3:
        return float("nan"), {"gap_fraction": 1.0, "roughness_m": float("nan"), "vertical_span_m": float("nan")}

    line_z = np.asarray(line_z, dtype=float)
    observed = np.asarray(line_status) == "observed"
    gap_fraction = 1.0 - float(np.mean(observed)) if len(line_status) else 1.0
    vertical_span = float(np.nanmax(line_z) - np.nanmin(line_z)) if np.any(np.isfinite(line_z)) else 0.0

    if len(line_z) >= 7:
        smooth_sigma = max(1.2, min(4.0, len(line_z) / 45.0))
        smooth_z = gaussian_filter1d(line_z, sigma=smooth_sigma, mode="nearest")
        residual_roughness = float(np.nanpercentile(np.abs(line_z - smooth_z), 75))
        local_step = float(np.nanpercentile(np.abs(np.diff(line_z)), 75))
        roughness = residual_roughness + 0.35 * local_step
    else:
        roughness = 0.0

    margin = (
        BELOW_ICE_MARGIN_BASE_M
        + BELOW_ICE_MARGIN_ROUGHNESS_GAIN * roughness
        + BELOW_ICE_MARGIN_GAP_GAIN_M * gap_fraction
        + 0.04 * vertical_span
    )
    margin = float(np.clip(margin, BELOW_ICE_MARGIN_MIN_M, BELOW_ICE_MARGIN_MAX_M))
    return margin, {
        "gap_fraction": float(gap_fraction),
        "roughness_m": float(roughness),
        "vertical_span_m": float(vertical_span),
    }


def build_below_ice_non_target_mask(
    valid_mask: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    line_y: np.ndarray,
    line_z: np.ndarray,
    line_status: np.ndarray,
) -> tuple[np.ndarray, dict]:
    mask = np.zeros_like(valid_mask, dtype=bool)
    margin, meta = adaptive_below_ice_margin(line_y, line_z, line_status)
    if len(line_y) < 3 or not np.isfinite(margin):
        return mask, {**meta, "margin_m": float("nan"), "mask_fraction": 0.0}

    extension_m = max(0.8, 0.5 * margin)
    y_support = (y_axis >= float(np.nanmin(line_y)) - extension_m) & (y_axis <= float(np.nanmax(line_y)) + extension_m)
    if not np.any(y_support):
        return mask, {**meta, "margin_m": margin, "mask_fraction": 0.0}

    line_curve = np.interp(y_axis, line_y, line_z, left=np.nan, right=np.nan)
    near_left = (y_axis < line_y[0]) & (y_axis >= line_y[0] - extension_m)
    near_right = (y_axis > line_y[-1]) & (y_axis <= line_y[-1] + extension_m)
    line_curve[near_left] = line_z[0]
    line_curve[near_right] = line_z[-1]

    _, z_grid = np.meshgrid(y_axis, z_axis)
    line_grid = np.tile(line_curve[None, :], (len(z_axis), 1))
    mask = valid_mask & y_support[None, :] & np.isfinite(line_grid) & (z_grid < line_grid - margin)
    valid_count = int(np.count_nonzero(valid_mask))
    mask_fraction = float(np.count_nonzero(mask) / valid_count) if valid_count else 0.0
    return mask, {**meta, "margin_m": margin, "mask_fraction": mask_fraction}


def adaptive_above_ice_margin(
    line_y: np.ndarray,
    line_z: np.ndarray,
    line_status: np.ndarray,
) -> tuple[float, dict]:
    if len(line_y) < 3 or len(line_z) < 3:
        return float("nan"), {"gap_fraction": 1.0, "roughness_m": float("nan"), "vertical_span_m": float("nan")}

    line_z = np.asarray(line_z, dtype=float)
    observed = np.asarray(line_status) == "observed"
    gap_fraction = 1.0 - float(np.mean(observed)) if len(line_status) else 1.0
    vertical_span = float(np.nanmax(line_z) - np.nanmin(line_z)) if np.any(np.isfinite(line_z)) else 0.0

    if len(line_z) >= 7:
        smooth_sigma = max(1.2, min(4.0, len(line_z) / 45.0))
        smooth_z = gaussian_filter1d(line_z, sigma=smooth_sigma, mode="nearest")
        residual_roughness = float(np.nanpercentile(np.abs(line_z - smooth_z), 75))
        local_step = float(np.nanpercentile(np.abs(np.diff(line_z)), 75))
        roughness = residual_roughness + 0.35 * local_step
    else:
        roughness = 0.0

    margin = (
        ABOVE_ICE_MARGIN_BASE_M
        + ABOVE_ICE_MARGIN_ROUGHNESS_GAIN * roughness
        + ABOVE_ICE_MARGIN_GAP_GAIN_M * gap_fraction
        + 0.05 * vertical_span
    )
    margin = float(np.clip(margin, ABOVE_ICE_MARGIN_MIN_M, ABOVE_ICE_MARGIN_MAX_M))
    return margin, {
        "gap_fraction": float(gap_fraction),
        "roughness_m": float(roughness),
        "vertical_span_m": float(vertical_span),
    }


def build_above_ice_multipath_mask(
    valid_mask: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    line_y: np.ndarray,
    line_z: np.ndarray,
    line_status: np.ndarray,
) -> tuple[np.ndarray, dict]:
    mask = np.zeros_like(valid_mask, dtype=bool)
    margin, meta = adaptive_above_ice_margin(line_y, line_z, line_status)
    if len(line_y) < 3 or not np.isfinite(margin):
        return mask, {**meta, "margin_m": float("nan"), "mask_fraction": 0.0}

    extension_m = max(0.8, 0.45 * margin)
    y_support = (y_axis >= float(np.nanmin(line_y)) - extension_m) & (y_axis <= float(np.nanmax(line_y)) + extension_m)
    if not np.any(y_support):
        return mask, {**meta, "margin_m": margin, "mask_fraction": 0.0}

    line_curve = np.interp(y_axis, line_y, line_z, left=np.nan, right=np.nan)
    near_left = (y_axis < line_y[0]) & (y_axis >= line_y[0] - extension_m)
    near_right = (y_axis > line_y[-1]) & (y_axis <= line_y[-1] + extension_m)
    line_curve[near_left] = line_z[0]
    line_curve[near_right] = line_z[-1]

    _, z_grid = np.meshgrid(y_axis, z_axis)
    line_grid = np.tile(line_curve[None, :], (len(z_axis), 1))
    mask = valid_mask & y_support[None, :] & np.isfinite(line_grid) & (z_grid > line_grid + margin)
    valid_count = int(np.count_nonzero(valid_mask))
    mask_fraction = float(np.count_nonzero(mask) / valid_count) if valid_count else 0.0
    return mask, {**meta, "margin_m": margin, "mask_fraction": mask_fraction}


def build_curve_guided_center_artifact_mask(
    image: np.ndarray,
    valid_mask: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    line_y: np.ndarray,
    line_z: np.ndarray,
    *,
    protect_m: float = CURVE_CENTER_PROTECT_M,
    min_level: float = CURVE_CENTER_MIN_LEVEL,
) -> np.ndarray:
    mask = np.zeros_like(valid_mask, dtype=bool)
    if len(line_y) < 3 or len(line_z) < 3:
        return mask

    line_curve = np.interp(y_axis, line_y, line_z, left=np.nan, right=np.nan)
    extension_m = max(0.8, protect_m)
    near_left = (y_axis < line_y[0]) & (y_axis >= line_y[0] - extension_m)
    near_right = (y_axis > line_y[-1]) & (y_axis <= line_y[-1] + extension_m)
    line_curve[near_left] = line_z[0]
    line_curve[near_right] = line_z[-1]

    y_grid, z_grid = np.meshgrid(y_axis, z_axis)
    line_grid = np.tile(line_curve[None, :], (len(z_axis), 1))
    protect = np.isfinite(line_grid) & (np.abs(z_grid - line_grid) <= protect_m)
    center_core = center_artifact_core_mask(y_grid, z_grid)
    return valid_mask & center_core & np.isfinite(image) & (image >= min_level) & ~protect


def process_ping(
    ds: xr.Dataset,
    ping: int,
    *,
    image_var: str = "backscatter",
    resolution_m: float = 0.08,
    annular_trigger_score: float = 0.24,
) -> FrameResult:
    mapped = backward_map_ping(ds, ping, image_var=image_var, resolution_m=resolution_m)
    valid = mapped["valid_mask"]
    y_axis = mapped["y_axis"]
    z_axis = mapped["z_axis"]
    y_grid = mapped["y_grid"]
    z_grid = mapped["z_grid"]
    r_grid = mapped["r_grid"]
    linear = mapped["linear_image"]

    first, _ = adaptive_range_threshold(linear, r_grid, valid)
    gray = run_unified_noise_suppression(first, valid, y_axis, z_axis)
    artifact_cleaned = gray["cleaned"]

    def extract_candidate(image: np.ndarray):
        candidate_score = build_score_map(image, valid, resolution_m)
        candidate_labels, candidate_components = connected_components_from_score(
            candidate_score,
            y_axis,
            z_axis,
            valid,
            mapped["range_max_m"],
        )
        candidate_selected_labels, candidate_component, candidate_lower_status = select_lower_candidate_group(candidate_components)
        candidate_line_y, candidate_line_z, candidate_line_status, candidate_obs_frac, candidate_center_z = extract_line_from_labels(
            candidate_labels,
            candidate_selected_labels,
            candidate_score,
            y_axis,
            z_axis,
        )
        return (
            candidate_score,
            candidate_labels,
            candidate_components,
            candidate_selected_labels,
            candidate_component,
            candidate_lower_status,
            candidate_line_y,
            candidate_line_z,
            candidate_line_status,
            candidate_obs_frac,
            candidate_center_z,
        )

    (
        score,
        labels,
        components,
        selected_labels,
        selected_component,
        lower_status,
        line_y,
        line_z,
        line_status,
        obs_frac,
        center_z,
    ) = extract_candidate(artifact_cleaned)

    below_ice_mask, below_ice_meta = build_below_ice_non_target_mask(valid, y_axis, z_axis, line_y, line_z, line_status)
    above_ice_mask, above_ice_meta = build_above_ice_multipath_mask(valid, y_axis, z_axis, line_y, line_z, line_status)
    curve_guided_cleaned = artifact_cleaned.copy()
    if np.any(below_ice_mask):
        curve_guided_cleaned[below_ice_mask] = 0.0
    if np.any(above_ice_mask):
        curve_guided_cleaned[above_ice_mask] = 0.0

    combined_suppression_mask = np.array(gray["suppress_mask"], copy=True)
    below_visible = below_ice_mask & (~np.isfinite(combined_suppression_mask) | (combined_suppression_mask == 1.0))
    below_lobe_like = below_visible & (artifact_cleaned >= COMPONENT_THRESHOLD)
    combined_suppression_mask[below_visible] = 1.0
    combined_suppression_mask[below_lobe_like] = 3.0
    above_visible = above_ice_mask & (~np.isfinite(combined_suppression_mask) | (combined_suppression_mask == 1.0))
    combined_suppression_mask[above_visible] = 1.0

    curve_center_mask = build_curve_guided_center_artifact_mask(curve_guided_cleaned, valid, y_axis, z_axis, line_y, line_z)
    if np.any(curve_center_mask):
        curve_guided_cleaned[curve_center_mask] = 0.0
        combined_suppression_mask[curve_center_mask] = 2.0

    (
        score,
        labels,
        components,
        selected_labels,
        selected_component,
        lower_status,
        line_y,
        line_z,
        line_status,
        obs_frac,
        center_z,
    ) = extract_candidate(curve_guided_cleaned)

    refined_above_mask, refined_above_meta = build_above_ice_multipath_mask(valid, y_axis, z_axis, line_y, line_z, line_status)
    refined_above_mask = refined_above_mask & ~above_ice_mask
    if np.any(refined_above_mask):
        curve_guided_cleaned[refined_above_mask] = 0.0
        refined_above_visible = refined_above_mask & (
            ~np.isfinite(combined_suppression_mask) | (combined_suppression_mask == 1.0)
        )
        combined_suppression_mask[refined_above_visible] = 1.0
        above_ice_mask = above_ice_mask | refined_above_mask
        above_ice_meta = refined_above_meta
        (
            score,
            labels,
            components,
            selected_labels,
            selected_component,
            lower_status,
            line_y,
            line_z,
            line_status,
            obs_frac,
            center_z,
        ) = extract_candidate(curve_guided_cleaned)

    refined_center_mask = build_curve_guided_center_artifact_mask(curve_guided_cleaned, valid, y_axis, z_axis, line_y, line_z)
    refined_center_mask = refined_center_mask & ~curve_center_mask
    if np.any(refined_center_mask):
        curve_guided_cleaned[refined_center_mask] = 0.0
        combined_suppression_mask[refined_center_mask] = 2.0
        (
            score,
            labels,
            components,
            selected_labels,
            selected_component,
            lower_status,
            line_y,
            line_z,
            line_status,
            obs_frac,
            center_z,
        ) = extract_candidate(curve_guided_cleaned)

    final_above_mask, final_above_meta = build_above_ice_multipath_mask(valid, y_axis, z_axis, line_y, line_z, line_status)
    final_above_mask = final_above_mask & ~above_ice_mask
    if np.any(final_above_mask):
        curve_guided_cleaned[final_above_mask] = 0.0
        final_above_visible = final_above_mask & (
            ~np.isfinite(combined_suppression_mask) | (combined_suppression_mask == 1.0)
        )
        combined_suppression_mask[final_above_visible] = 1.0
        above_ice_mask = above_ice_mask | final_above_mask
        above_ice_meta = final_above_meta
        (
            score,
            labels,
            components,
            selected_labels,
            selected_component,
            lower_status,
            line_y,
            line_z,
            line_status,
            obs_frac,
            center_z,
        ) = extract_candidate(curve_guided_cleaned)

    valid_count = int(np.count_nonzero(valid))
    if valid_count:
        above_ice_meta = {
            **above_ice_meta,
            "mask_fraction": float(np.count_nonzero(above_ice_mask) / valid_count),
        }

    annular = annular_residual_metrics(curve_guided_cleaned, r_grid, y_axis, z_grid, valid, line_y, line_z)
    do_second = bool(annular["annular_score"] >= annular_trigger_score and len(line_y) >= 3)
    denoised = secondary_annular_suppression(curve_guided_cleaned, r_grid, y_axis, z_grid, valid, line_y, line_z) if do_second else curve_guided_cleaned

    if do_second:
        (
            score,
            labels,
            components,
            selected_labels,
            selected_component,
            lower_status,
            line_y,
            line_z,
            line_status,
            obs_frac,
            center_z,
        ) = extract_candidate(denoised)

    dz = float(np.nanmean(np.diff(z_axis))) if len(z_axis) > 1 else resolution_m
    projection_score, projection_smooth = compute_z_projection(denoised, valid, z_axis, dz)
    selected_ice_z = estimate_ice_z_from_projection(projection_smooth, z_axis)
    gray_metrics = gray["metrics"]

    if len(line_y) == 0:
        center_status = "invalid"
        confidence = 0.0
        selected_component_count = 0
        selected_median_z = float("nan")
        selected_y_span = 0.0
        selected_mean_score = 0.0
    else:
        center_idx = int(np.argmin(np.abs(line_y)))
        center_status = str(line_status[center_idx])
        selected_component_count = len(selected_labels)
        selected_median_z = float(selected_component["median_z_m"]) if selected_component else float("nan")
        selected_y_span = float(line_y.max() - line_y.min())
        line_scores = sample_score_along_line(score, y_axis, z_axis, line_y, line_z)
        selected_mean_score = float(np.nanmean(line_scores)) if len(line_scores) else 0.0
        span_score = np.clip(selected_y_span / max(4.0, 0.45 * mapped["range_max_m"]), 0.0, 1.0)
        confidence = float(np.clip(0.45 * span_score + 0.35 * obs_frac + 0.20 * min(1.0, selected_mean_score * 2.5), 0.0, 1.0))

    frame_status = classify_frame_status(
        range_max_m=mapped["range_max_m"],
        candidate_count=len(components),
        confidence=confidence,
        center_status=center_status,
        lower_selection_status=lower_status,
        annular_second_pass=do_second,
    )

    ping_time_s = ping_time_unix(ds, ping)
    return FrameResult(
        ping=ping,
        ping_time_unix_s=ping_time_s,
        ping_time_local=ping_time_local_string(ping_time_s),
        range_max_m=float(mapped["range_max_m"]),
        sample_size_m=float(mapped["sample_size_m"]),
        nsamples=int(mapped["nsamples"]),
        nbeams=int(mapped["nbeams"]),
        azimuth_range_deg=float(mapped["azimuth_range_deg"]),
        y_axis=y_axis,
        z_axis=z_axis,
        linear_image=normalize_positive(linear, valid, q=99.2),
        first_denoised=first,
        denoised=denoised,
        suppression_mask=combined_suppression_mask,
        score_map=score,
        projection_score=projection_score,
        projection_smooth=projection_smooth,
        selected_ice_z_m=float(selected_ice_z),
        gray_config_name=str(gray["config"]["name"]),
        gray_score=float(gray_metrics["gray_score"]),
        ice_retention=float(gray_metrics["ice_retention"]),
        center_residual=float(gray_metrics["center_residual"]),
        lobe_residual=float(gray_metrics["lobe_residual"]),
        weak_annular_remaining=float(gray_metrics["weak_annular_remaining"]),
        below_ice_margin_m=float(below_ice_meta.get("margin_m", float("nan"))),
        below_ice_mask_fraction=float(below_ice_meta.get("mask_fraction", 0.0)),
        above_ice_margin_m=float(above_ice_meta.get("margin_m", float("nan"))),
        above_ice_mask_fraction=float(above_ice_meta.get("mask_fraction", 0.0)),
        line_y=line_y,
        line_z=line_z,
        line_status=line_status,
        observed_fraction=obs_frac,
        center_z_s_m=float(center_z),
        center_status=center_status,
        frame_status=frame_status,
        confidence=confidence,
        annular_score=float(annular["annular_score"]),
        annular_second_pass=do_second,
        selected_component_count=selected_component_count,
        selected_median_z_m=selected_median_z,
        selected_y_span_m=selected_y_span,
        selected_mean_score=selected_mean_score,
        candidate_count=len(components),
    )


def load_auv_motion(project_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(project_dir) / "01_preprocessing" / "AUV_Data_moving" / "auv_motion_attitude_ctd_window_filtered.csv"
    motion = pd.read_csv(path)
    time_col = "time_unix_s" if "time_unix_s" in motion.columns else None
    if time_col is None:
        motion["time_dt"] = pd.to_datetime(motion["time_iso"], errors="coerce")
        motion["time_unix_s"] = motion["time_dt"].astype("int64") / 1e9
    for col in ["NAV_DEPTH", "NAV_ALTITUDE", "pitch_filtered_deg", "roll_filtered_deg", "NAV_Heading"]:
        if col in motion.columns:
            motion[col] = pd.to_numeric(motion[col], errors="coerce")
    motion = motion.dropna(subset=["time_unix_s"]).sort_values("time_unix_s").reset_index(drop=True)
    return motion, motion["time_unix_s"].to_numpy(dtype=float)


def interpolate_motion_value(motion: pd.DataFrame, motion_time: np.ndarray, column: str, ping_time_s: float) -> float:
    if column not in motion.columns:
        return float("nan")
    values = motion[column].to_numpy(dtype=float)
    good = np.isfinite(values) & np.isfinite(motion_time)
    if good.sum() == 0:
        return float("nan")
    return float(np.interp(ping_time_s, motion_time[good], values[good]))


def roll_correct_yz(y_m: np.ndarray, z_m: np.ndarray, roll_deg: float, sign: int = 1) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(sign * roll_deg)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    y_corr = y * np.cos(angle) - z * np.sin(angle)
    z_corr = y * np.sin(angle) + z * np.cos(angle)
    return y_corr, z_corr


def frame_to_summary_row(
    result: FrameResult,
    project_dir: str | Path,
    *,
    roll_sign: int = 1,
    line_selection: dict | None = None,
) -> dict:
    motion, motion_time = load_auv_motion(project_dir)
    ping_t = result.ping_time_unix_s
    nav_depth = interpolate_motion_value(motion, motion_time, "NAV_DEPTH", ping_t)
    nav_altitude = interpolate_motion_value(motion, motion_time, "NAV_ALTITUDE", ping_t)
    pitch = interpolate_motion_value(motion, motion_time, "pitch_filtered_deg", ping_t)
    roll = interpolate_motion_value(motion, motion_time, "roll_filtered_deg", ping_t)
    heading = interpolate_motion_value(motion, motion_time, "NAV_Heading", ping_t)

    if line_selection is None:
        line_selection = {
            "line_decision": "original_baseline",
            "line_decision_reason": "legacy_original_line",
            "use_for_reconstruction": len(result.line_y) >= 3 and np.isfinite(result.center_z_s_m),
            "final_line_source": "original",
            "final_center_z_s_m": result.center_z_s_m,
            "center_delta_m": float("nan"),
        }

    use_for_reconstruction = bool(line_selection.get("use_for_reconstruction", False))
    final_center_z = safe_float(line_selection.get("final_center_z_s_m", result.center_z_s_m))

    if use_for_reconstruction and np.isfinite(final_center_z):
        center_y = 0.0
        _, z_corr = roll_correct_yz(np.array([center_y]), np.array([final_center_z]), roll, sign=roll_sign)
        theta_deg = pitch
        vertical_range = float(z_corr[0] * np.cos(np.deg2rad(theta_deg)))
        ice_draft = float(nav_depth - vertical_range)
    else:
        vertical_range = float("nan")
        ice_draft = float("nan")

    return {
        "ping": result.ping,
        "ping_time_unix_s": result.ping_time_unix_s,
        "ping_time_local": result.ping_time_local,
        "range_max_m": result.range_max_m,
        "sample_size_m": result.sample_size_m,
        "nsamples": result.nsamples,
        "nav_depth_m": nav_depth,
        "nav_altitude_m": nav_altitude,
        "pitch_filtered_deg": pitch,
        "roll_filtered_deg": roll,
        "nav_heading_deg": heading,
        "center_z_s_m": result.center_z_s_m,
        "final_center_z_s_m": final_center_z,
        "vertical_range_m": vertical_range,
        "ice_draft_m": ice_draft,
        "line_decision": line_selection.get("line_decision", "unclassified"),
        "line_decision_reason": line_selection.get("line_decision_reason", ""),
        "use_for_reconstruction": int(use_for_reconstruction),
        "final_line_source": line_selection.get("final_line_source", ""),
        "center_delta_m": line_selection.get("center_delta_m", float("nan")),
        "active_echo_px": line_selection.get("active_echo_px", float("nan")),
        "strong_echo_px": line_selection.get("strong_echo_px", float("nan")),
        "active_echo_fraction": line_selection.get("active_echo_fraction", float("nan")),
        "strong_echo_fraction": line_selection.get("strong_echo_fraction", float("nan")),
        "active_echo_y_span_m": line_selection.get("active_echo_y_span_m", float("nan")),
        "center_status": result.center_status,
        "frame_status": result.frame_status,
        "confidence": result.confidence,
        "gray_config_name": result.gray_config_name,
        "gray_score": result.gray_score,
        "ice_retention": result.ice_retention,
        "center_residual": result.center_residual,
        "lobe_residual": result.lobe_residual,
        "weak_annular_remaining": result.weak_annular_remaining,
        "below_ice_margin_m": result.below_ice_margin_m,
        "below_ice_mask_fraction": result.below_ice_mask_fraction,
        "above_ice_margin_m": result.above_ice_margin_m,
        "above_ice_mask_fraction": result.above_ice_mask_fraction,
        "selected_ice_z_m": result.selected_ice_z_m,
        "observed_fraction": result.observed_fraction,
        "annular_score": result.annular_score,
        "annular_second_pass": int(result.annular_second_pass),
        "candidate_count": result.candidate_count,
        "selected_component_count": result.selected_component_count,
        "selected_median_z_m": result.selected_median_z_m,
        "selected_y_span_m": result.selected_y_span_m,
        "selected_mean_score": result.selected_mean_score,
    }


def save_frame_diagnostic(result: FrameResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extent = [result.y_axis[0], result.y_axis[-1], result.z_axis[0], result.z_axis[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=170, constrained_layout=True)
    fig.suptitle(
        f"Range-adaptive sonar noise suppression | ping {result.ping} | "
        f"ice z={result.selected_ice_z_m:.2f} m | score={result.gray_score:.3f}",
        fontsize=13,
    )

    ax_input, ax_output, ax_mask, ax_projection = axes.flat
    sonar_cmap = OCULUS_CMAP.copy()
    sonar_cmap.set_bad("#000000")
    im0 = ax_input.imshow(
        result.linear_image,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=sonar_cmap,
        vmin=0,
        vmax=1,
    )
    ax_input.set_title("Linear-interpolated input")

    im1 = ax_output.imshow(
        result.denoised,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=sonar_cmap,
        vmin=0,
        vmax=1,
    )
    ax_output.set_title(f"Noise-suppressed output | {result.gray_config_name}")
    fig.colorbar(im1, ax=ax_output, fraction=0.046, pad=0.02, label="Normalized intensity")

    mask_cmap = ListedColormap(["#481567", "#238A8D", "#FDE725"])
    mask_cmap.set_bad("white")
    mask_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], mask_cmap.N)
    im2 = ax_mask.imshow(
        result.suppression_mask,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=mask_cmap,
        norm=mask_norm,
    )
    ax_mask.set_title("Suppression masks: 1 range/weak, 2 center, 3 lobe")
    cbar = fig.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.02, ticks=[1, 2, 3])
    cbar.set_label("Suppression class")

    ax_projection.plot(result.projection_score, result.z_axis, color="0.70", linewidth=1.2, label="projection score")
    ax_projection.plot(result.projection_smooth, result.z_axis, color="#D88900", linewidth=1.8, label="smoothed")
    if np.isfinite(result.selected_ice_z_m):
        ax_projection.axhline(result.selected_ice_z_m, color="#4C78D0", linestyle="--", linewidth=1.1, label="selected candidate")
        ax_projection.text(
            0.98,
            result.selected_ice_z_m,
            f" {result.selected_ice_z_m:.2f} m",
            ha="right",
            va="center",
            fontsize=8,
            color="0.25",
        )
    ax_projection.set_title("z_s projection / peak detection")
    ax_projection.set_xlabel("Normalized projection score")
    ax_projection.set_ylabel("z_s center-beam forward (m)")
    ax_projection.set_xlim(-0.02, 1.05)
    ax_projection.set_ylim(result.z_axis[0], result.z_axis[-1])
    ax_projection.grid(True, color="0.85", linestyle=":", linewidth=0.7, alpha=0.85)
    ax_projection.legend(loc="lower right", fontsize=8)

    for ax in [ax_input, ax_output, ax_mask]:
        ax.set_xlabel("y_s port / cross-track (m)")
        ax.set_ylabel("z_s center-beam forward (m)")
        ax.grid(True, color="0.25", linestyle=":", linewidth=0.6, alpha=0.45)

    ax_output.text(
        0.02,
        0.03,
        "\n".join(
            [
                f"center residual: {result.center_residual:.3f}",
                f"lobe residual: {result.lobe_residual:.3f}",
                f"weak remaining: {result.weak_annular_remaining:.3f}",
                f"lower margin: {result.below_ice_margin_m:.2f} m",
                f"upper margin: {result.above_ice_margin_m:.2f} m",
                f"guided clear: {result.below_ice_mask_fraction + result.above_ice_mask_fraction:.3f}",
                f"ice retention: {result.ice_retention:.3f}",
            ]
        ),
        transform=ax_output.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.75"},
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_dp_ridge_diagnostic(result: FrameResult, dp_result: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extent = [result.y_axis[0], result.y_axis[-1], result.z_axis[0], result.z_axis[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=170, constrained_layout=True)
    fig.suptitle(f"Dynamic-programming ice-ridge check | ping {result.ping}", fontsize=13)

    ax_img, ax_score, ax_curve, ax_metrics = axes.flat
    sonar_cmap = OCULUS_CMAP.copy()
    sonar_cmap.set_bad("#000000")

    ax_img.imshow(result.denoised, origin="lower", extent=extent, aspect="auto", cmap=sonar_cmap, vmin=0, vmax=1)
    ax_img.plot(result.line_y, result.line_z, color="#00D7FF", linewidth=1.8, label="current line")
    if len(dp_result.get("line_y", [])):
        ax_img.plot(dp_result["line_y"], dp_result["line_z"], color="#FF4B6E", linewidth=1.7, label="DP ridge")
    ax_img.set_title("Denoised image + extracted lines")
    ax_img.legend(loc="lower right", fontsize=8)

    ax_score.imshow(result.score_map, origin="lower", extent=extent, aspect="auto", cmap=sonar_cmap, vmin=0, vmax=1)
    candidate_points = dp_result.get("candidate_points", pd.DataFrame())
    if isinstance(candidate_points, pd.DataFrame) and not candidate_points.empty:
        plot_points = candidate_points
        if len(plot_points) > 2500:
            plot_points = plot_points.sample(2500, random_state=0)
        ax_score.scatter(plot_points["y"], plot_points["z"], s=4, c="white", alpha=0.18, linewidths=0, label="candidate peaks")
        selected_points = candidate_points[candidate_points["selected"]]
        if not selected_points.empty:
            ax_score.scatter(selected_points["y"], selected_points["z"], s=7, c="#FF4B6E", alpha=0.85, linewidths=0, label="selected DP nodes")
    if len(dp_result.get("line_y", [])):
        ax_score.plot(dp_result["line_y"], dp_result["line_z"], color="#FF4B6E", linewidth=1.4)
    ax_score.set_title("Score map + DP candidates")
    ax_score.legend(loc="lower right", fontsize=8)

    ax_curve.plot(result.line_y, result.line_z, color="#00A7D7", linewidth=1.6, label="current line")
    if len(dp_result.get("line_y", [])):
        ax_curve.plot(dp_result["line_y"], dp_result["line_z"], color="#FF4B6E", linewidth=1.6, label="DP ridge")
        observed = np.asarray(dp_result["line_status"]) == "observed_dp"
        if np.any(observed):
            ax_curve.scatter(np.asarray(dp_result["line_y"])[observed], np.asarray(dp_result["line_z"])[observed], s=8, color="#FF4B6E", alpha=0.7)
    ax_curve.set_title("Curve comparison")
    ax_curve.set_xlabel("y_s port / cross-track (m)")
    ax_curve.set_ylabel("z_s center-beam forward (m)")
    ax_curve.grid(True, color="0.85", linestyle=":", linewidth=0.7)
    ax_curve.legend(loc="best", fontsize=8)

    metrics = dp_result.get("metrics", {})
    selection = dp_result.get("selection", {})
    summary_lines = [
        f"decision: {selection.get('line_decision', 'unclassified')}",
        f"include: {selection.get('use_for_reconstruction', False)}",
        f"reason: {selection.get('line_decision_reason', 'n/a')}",
        f"status: {metrics.get('status', 'unknown')}",
        f"candidate columns: {metrics.get('candidate_columns', 0)}",
        f"candidate points: {metrics.get('candidate_points', 0)}",
        f"mean selected score: {metrics.get('mean_selected_score', float('nan')):.3f}",
        f"max jump: {metrics.get('max_jump_m', float('nan')):.2f} m",
        f"p95 jump: {metrics.get('p95_jump_m', float('nan')):.2f} m",
        f"mean curvature: {metrics.get('mean_curvature_m', float('nan')):.3f} m",
        f"center z current: {result.center_z_s_m:.2f} m",
        f"center z DP: {dp_result.get('center_z_s_m', float('nan')):.2f} m",
        f"center delta: {selection.get('center_delta_m', float('nan')):.2f} m",
    ]
    ax_metrics.axis("off")
    ax_metrics.text(0.02, 0.95, "\n".join(summary_lines), ha="left", va="top", fontsize=10, family="monospace")
    ax_metrics.set_title("DP path QC")

    for ax in [ax_img, ax_score]:
        ax.set_xlabel("y_s port / cross-track (m)")
        ax.set_ylabel("z_s center-beam forward (m)")
        ax.grid(True, color="0.25", linestyle=":", linewidth=0.6, alpha=0.45)

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_line_segments(ax, result: FrameResult) -> None:
    if len(result.line_y) < 2:
        return
    observed = result.line_status == "observed"
    interpolated = result.line_status != "observed"
    ax.plot(result.line_y, result.line_z, color="#00D7FF", linewidth=2.0, label="connected ice line")
    if observed.any():
        ax.scatter(result.line_y[observed], result.line_z[observed], s=6, color="#00D7FF", alpha=0.88)
    if interpolated.any():
        ax.scatter(result.line_y[interpolated], result.line_z[interpolated], s=7, color="#FF4B6E", alpha=0.90)
    if np.isfinite(result.center_z_s_m):
        ax.scatter([0.0], [result.center_z_s_m], s=44, facecolor="white", edgecolor="#00D7FF", linewidth=1.3, zorder=6)


def save_draft_mvp_figure(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    if summary_df.empty:
        return
    df = summary_df.copy()
    df["time_dt"] = pd.to_datetime(df["ping_time_local"])
    if "ice_draft_raw_m" in df.columns:
        raw_draft = pd.to_numeric(df["ice_draft_raw_m"], errors="coerce")
    else:
        raw_draft = pd.to_numeric(df["ice_draft_m"], errors="coerce")
    if "draft_qc_flag" in df.columns:
        qc_flag = pd.to_numeric(df["draft_qc_flag"], errors="coerce").fillna(0).astype(int) == 1
    else:
        qc_flag = pd.to_numeric(df.get("draft_post_qc_excluded", 0), errors="coerce").fillna(0).astype(int) == 1
    manual_qc = pd.to_numeric(df.get("qc_excluded", 0), errors="coerce").fillna(0).astype(int) == 1
    raw_valid = raw_draft.where(~qc_flag & ~manual_qc)
    if "ice_draft_smoothed_m" in df.columns:
        draft_smoothed = pd.to_numeric(df["ice_draft_smoothed_m"], errors="coerce")
    elif "ice_draft_ma3_m" in df.columns:
        draft_smoothed = pd.to_numeric(df["ice_draft_ma3_m"], errors="coerce")
    else:
        draft_smoothed = raw_valid.rolling(window=3, center=True, min_periods=1).median().where(raw_valid.notna())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=170, sharex=True, constrained_layout=True)
    ax1.plot(df["time_dt"], raw_valid, marker="o", color="0.70", linewidth=1.0, markersize=3.2, label="Initial draft")
    ax1.plot(df["time_dt"], draft_smoothed, marker="o", color="#1f77b4", linewidth=1.8, markersize=3.4, label="3-point smoothed draft")
    low = df["confidence"] < 0.35
    if low.any():
        ax1.scatter(df.loc[low, "time_dt"], raw_valid.loc[low], s=56, facecolor="none", edgecolor="#C44E52", linewidth=1.2, label="low confidence")
    ax1.axhline(0.0, color="0.25", linestyle="--", linewidth=0.9)
    ax1.invert_yaxis()
    ax1.set_ylabel("Ice draft (m)")
    ax1.grid(True, alpha=0.32, linestyle=":")
    ax1.legend(loc="best", fontsize=8)
    ax2.plot(df["time_dt"], df["nav_depth_m"], color="#C44E52", linewidth=1.6, label="AUV depth")
    ax2.invert_yaxis()
    ax2.set_ylabel("AUV depth (m)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.32, linestyle=":")
    ax2.legend(loc="best", fontsize=8)
    fig.suptitle("Dynamic ice-line draft check", fontsize=13)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
