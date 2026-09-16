"""Validated, dataset-independent initial settings for the public CLI.

Each CLI process uses one parameter set. Initial values are starting points,
not instrument calibration or universal thresholds.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path


DEFAULT_PARAMETERS = {
    "grid": {
        "grid_resolution_m": 0.05, "range_min_m": 0.0,
        "azimuth_half_angle_deg": 65.0, "vertical_aperture_deg": 20.0,
        "interpolation": "linear", "beam_order": "starboard_to_port",
        "range_scale": 1.0, "hole_fill_max_distance_m": 0.15,
    },
    "background": {
        "bin_m": 0.25, "low_q": 35.0, "high_q": 74.0,
        "iqr_gain": 0.55, "smooth_bins": 5,
        "min_bin_pixels": 25, "normalization_q": 99.0,
    },
    "gray": {
        "weak_floor": 0.13, "post_floor": 0.10, "center_min_level": 0.32,
        "center_attenuation": 0.02, "lobe_attenuation": 0.05,
    },
    "denoising": {
        "ice_min_z_m": 6.0, "ice_band_half_width_m": 0.55,
        "center_half_width_m": 0.34, "center_half_angle_deg": 2.4,
        "center_reference_z_m": 1.0,
        "center_side_inner_m": 0.52, "center_side_outer_m": 1.55,
        "center_side_percentile": 70.0, "center_excess": 0.10,
        "center_side_preserve_level": 0.55, "center_below_margin_m": 0.15,
        "lobe_z_margin_m": 1.25, "lobe_top_margin_m": 0.45,
        "lobe_max_width_m": 7.0, "lobe_max_area_m2": 1.2,
        "lobe_min_pixels": 20, "lobe_closing_rows": 3, "lobe_closing_cols": 5,
        "component_threshold": 0.24, "projection_pixel_threshold": 0.35,
        "projection_energy_weight": 0.65, "projection_support_weight": 0.35,
        "projection_sigma_m": 0.15,
        "below_ice_margin_min_m": 1.4, "below_ice_margin_max_m": 4.2,
        "below_ice_margin_base_m": 1.5, "below_ice_margin_gap_gain_m": 1.2,
        "below_ice_margin_roughness_gain": 1.8,
        "above_ice_margin_min_m": 1.8, "above_ice_margin_max_m": 5.8,
        "above_ice_margin_base_m": 2.0, "above_ice_margin_gap_gain_m": 1.2,
        "above_ice_margin_roughness_gain": 1.4,
        "curve_center_protect_m": 0.75, "curve_center_min_level": 0.035,
    },
    "guidance": {"iterations": 2, "annular_trigger_score": 0.24},
    "binary": {
        "high_q": 86.0, "low_q": 66.0,
        "window_down_m": 1.20, "window_up_m": 0.60,
        "closing_y_m": 0.25, "closing_z_m": 0.08,
        "opening_y_m": 0.05, "opening_z_m": 0.05,
        "min_area_m2": 0.030, "min_y_span_m": 0.65, "hole_area_m2": 0.030,
    },
    "binary_refinement": {
        "edge_extension_down_m": 1.70, "edge_extension_max_y_m": 4.50,
        "edge_extension_step_z_m": 0.35, "edge_extension_half_width_m": 0.12,
        "edge_extension_max_misses": 8, "edge_min_intensity": 0.35,
        "edge_threshold_scale": 0.90,
        "gap_connect_max_y_m": 1.80, "gap_connect_max_z_m": 0.35,
        "mask_bridge_half_width_m": 0.10, "mask_bridge_support_scale": 0.40,
        "mask_bridge_support_min": 0.12, "mask_bridge_support_fraction": 0.20,
        "line_connect_max_y_m": 0.60, "line_connect_max_z_m": 0.35,
        "min_window_pixels": 10,
        "center_reject_abs_y_m": 0.35, "center_reject_width_m": 0.50,
        "center_reject_height_m": 1.20, "center_reject_below_m": 0.20,
        "lower_reject_below_m": 1.15, "lower_reject_abs_y_m": 7.50,
        "lower_reject_width_m": 4.50,
    },
    "selection": {
        "mode": "fixed",
        "high_q": [82.0, 86.0, 88.0], "low_q": [58.0, 62.0, 66.0],
        "window_down_m": [1.20, 1.35], "window_up_m": [0.60, 0.70],
        "closing_y_m": [0.25, 0.45, 0.65], "closing_z_m": [0.08, 0.12, 0.16],
    },
    "attitude": {"roll_sign": 1, "pitch_sign": 1, "tilt_from_vertical_deg": 0.0},
    "center": {"half_width_m": 0.50, "min_points": 5, "fallback_points": 9},
    "postprocess": {
        "enabled": True, "residual_threshold_m": 0.15,
        "baseline_window_points": 3, "smooth_window_points": 3,
        "interpolate": True, "max_gap_s": None,
    },
}


def _merge(default, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError(f"{prefix or 'parameters'} must be an object")
    unknown = set(override) - set(default)
    if unknown:
        raise ValueError(f"Unknown parameters in {prefix or 'root'}: {sorted(unknown)}")
    result = deepcopy(default)
    for key, value in override.items():
        path = f"{prefix}.{key}".strip(".")
        if isinstance(default[key], dict):
            result[key] = _merge(default[key], value, path)
        else:
            result[key] = value
    return result


def validate_parameters(p):
    def walk(default, values, prefix=""):
        for key, base in default.items():
            value, path = values[key], f"{prefix}.{key}".strip(".")
            if isinstance(base, dict):
                walk(base, value, path)
            elif isinstance(base, bool):
                if type(value) is not bool:
                    raise ValueError(f"{path} must be a boolean")
            elif isinstance(base, (float, int)):
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"{path} must be a finite number")
                if isinstance(base, int) and not isinstance(value, int):
                    raise ValueError(f"{path} must be an integer")
                if path not in {"attitude.roll_sign", "attitude.pitch_sign", "attitude.tilt_from_vertical_deg"} and value < 0:
                    raise ValueError(f"{path} must be nonnegative")
            elif isinstance(base, list):
                if not isinstance(value, list) or not value or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in value):
                    raise ValueError(f"{path} must be a nonempty list of finite nonnegative numbers")
            elif isinstance(base, str) and not isinstance(value, str):
                raise ValueError(f"{path} must be a string")
    walk(DEFAULT_PARAMETERS, p)
    for group, key in [("grid", "grid_resolution_m"), ("grid", "range_scale"), ("background", "bin_m"), ("background", "min_bin_pixels"), ("denoising", "lobe_min_pixels"), ("denoising", "lobe_closing_rows"), ("denoising", "lobe_closing_cols"), ("binary_refinement", "min_window_pixels"), ("binary_refinement", "edge_extension_max_misses"), ("center", "min_points"), ("center", "fallback_points"), ("postprocess", "baseline_window_points"), ("postprocess", "smooth_window_points")]:
        if p[group][key] <= 0:
            raise ValueError(f"{group}.{key} must be positive")
    for group in ("background", "binary"):
        if not 0 <= p[group]["low_q"] < p[group]["high_q"] <= 100:
            raise ValueError(f"{group} requires 0 <= low_q < high_q <= 100")
    if not 0 < p["background"]["normalization_q"] <= 100:
        raise ValueError("background.normalization_q must be in (0, 100]")
    if not 0 < p["grid"]["azimuth_half_angle_deg"] < 90 or not 0 < p["grid"]["vertical_aperture_deg"] < 180:
        raise ValueError("Grid apertures must describe an upward-facing fan")
    if p["grid"]["beam_order"] not in {"starboard_to_port", "port_to_starboard"}:
        raise ValueError("grid.beam_order must be starboard_to_port or port_to_starboard")
    if p["grid"]["interpolation"] not in {"linear", "nearest"}:
        raise ValueError("grid.interpolation must be linear or nearest")
    if p["selection"]["mode"] not in {"fixed", "tune"}:
        raise ValueError("selection.mode must be fixed or tune")
    for key in ("roll_sign", "pitch_sign"):
        if p["attitude"][key] not in {-1, 1}:
            raise ValueError(f"attitude.{key} must be -1 or +1; determine it from calibration")
    if abs(p["attitude"]["tilt_from_vertical_deg"]) >= 90:
        raise ValueError("The scalar tilt correction requires an upward-looking installation")
    d = p["denoising"]
    if d["center_side_inner_m"] >= d["center_side_outer_m"]:
        raise ValueError("The side-reference inner bound must be smaller than its outer bound")
    if not 0 <= d["center_side_percentile"] <= 100:
        raise ValueError("denoising.center_side_percentile must be in [0, 100]")
    if d["projection_energy_weight"] + d["projection_support_weight"] <= 0:
        raise ValueError("Projection weights cannot both be zero")
    for side in ("below", "above"):
        if d[f"{side}_ice_margin_min_m"] > d[f"{side}_ice_margin_max_m"]:
            raise ValueError(f"Invalid {side}-ice guidance margin interval")
    for value in p["gray"].values():
        if not 0 <= value <= 1:
            raise ValueError("Gray intensity and attenuation settings must be in [0, 1]")
    gap = p["postprocess"]["max_gap_s"]
    if gap is not None and (isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0):
        raise ValueError("postprocess.max_gap_s must be null or a positive number")
    if any(x > 100 for key in ("high_q", "low_q") for x in p["selection"][key]):
        raise ValueError("Search percentiles must be in [0, 100]")
    if min(p["selection"]["high_q"]) <= max(p["selection"]["low_q"]):
        raise ValueError("All searched high percentiles must exceed all searched low percentiles")
    count = math.prod(len(v) for k, v in p["selection"].items() if k != "mode")
    if count > 10000:
        raise ValueError("Search grid exceeds 10000 candidates; narrow selection ranges")


def load_parameters(path=None, *, overrides=None):
    if path is not None and overrides is not None:
        raise ValueError("Use a file or overrides, not both")
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path is not None else (overrides or {})
    p = _merge(DEFAULT_PARAMETERS, raw)
    validate_parameters(p)
    return p


def parameters_for(ns):
    return ns.get("ALGORITHM_PARAMETERS", DEFAULT_PARAMETERS)


def configure_mapping(ns, p):
    """Apply settings after loading the runtime imports, before mapping any ping."""
    ns["ALGORITHM_PARAMETERS"] = p
    g = p["grid"]
    ns["GRID_CONFIG"] = replace(ns["GRID_CONFIG"], **{k: v for k, v in g.items() if k not in {"range_scale", "hole_fill_max_distance_m"}})
    read_frame = ns["read_ping_frame"]
    def read_calibrated_frame(*args, **kwargs):
        image, metadata = read_frame(*args, **kwargs)
        metadata = dict(metadata)
        metadata["sample_size_m"] *= g["range_scale"]
        metadata["raw_range_max_m"] *= g["range_scale"]
        metadata["range_scale"] = g["range_scale"]
        return image, metadata
    ns["read_ping_frame"] = read_calibrated_frame


def configure_binary_candidates(ns):
    p = parameters_for(ns)
    base = dict(p["binary"], name="initial")
    if p["selection"]["mode"] == "fixed":
        ns["BINARY_CANDIDATE_CONFIGS"] = [base]
        return
    keys = [k for k in p["selection"] if k != "mode"]
    ns["BINARY_CANDIDATE_CONFIGS"] = [dict(base, **dict(zip(keys, values)), name=f"candidate_{i:04d}") for i, values in enumerate(product(*(p["selection"][k] for k in keys)))]
