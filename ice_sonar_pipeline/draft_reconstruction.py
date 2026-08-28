"""AUV attitude correction and centerline ice-draft reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

DRAFT_QC_RESIDUAL_THRESHOLD_M = 0.15
DRAFT_QC_GOOD_RESIDUAL_M = 0.05


def load_auv_motion(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a synchronized AUV motion table supplied by the user."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).sort_values("time_unix_s").drop_duplicates("time_unix_s").reset_index(drop=True)
    for col in ["time_unix_s", "pitch_filtered_deg", "roll_filtered_deg", "NAV_Heading", "NAV_DEPTH", "NAV_ALTITUDE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, df["time_unix_s"].to_numpy(dtype=float)


def ping_time_unix(ds, ping_idx: int) -> float:
    return float(ds["time"].isel(ping=int(ping_idx)).values)


def ping_time_local_string(ping_time_s: float, timezone: str = "UTC") -> str:
    return str(pd.to_datetime(ping_time_s, unit="s", utc=True).tz_convert(timezone).tz_localize(None))


def interpolate_auv_value(auv_motion: pd.DataFrame, auv_time: np.ndarray, column: str, ping_time_s: float) -> float:
    if column not in auv_motion.columns:
        return np.nan
    values = auv_motion[column].to_numpy(dtype=float)
    valid = np.isfinite(auv_time) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.nan
    return float(np.interp(ping_time_s, auv_time[valid], values[valid]))


def roll_correct_yz(y_m: np.ndarray, z_m: np.ndarray, roll_deg: float, sign: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(sign * roll_deg)
    y_corr = y_m * np.cos(theta) - z_m * np.sin(theta)
    z_corr = y_m * np.sin(theta) + z_m * np.cos(theta)
    return y_corr, z_corr


def robust_curve_metrics(y_m: np.ndarray, z_m: np.ndarray) -> dict:
    finite = np.isfinite(y_m) & np.isfinite(z_m)
    y_m = np.asarray(y_m, dtype=float)[finite]
    z_m = np.asarray(z_m, dtype=float)[finite]
    if y_m.size < 5:
        return {"slope": np.nan, "roughness_m": np.nan, "span_m": 0.0, "median_z_m": np.nan}
    order = np.argsort(y_m)
    y_sorted = y_m[order]
    z_sorted = z_m[order]
    slope = float(np.polyfit(y_sorted, z_sorted, 1)[0])
    smooth = gaussian_filter1d(z_sorted, sigma=max(1.0, 0.30 / 0.05))
    roughness = float(np.nanmedian(np.abs(z_sorted - smooth)))
    return {
        "slope": slope,
        "roughness_m": roughness,
        "span_m": float(np.nanmax(y_sorted) - np.nanmin(y_sorted)),
        "median_z_m": float(np.nanmedian(z_sorted)),
    }


def build_roll_and_curve_tables(ns: dict, ds, curve_records: list[dict]) -> tuple[list[dict], list[dict], int]:
    auv_motion, auv_time = load_auv_motion(ns["MOTION_FILE"])
    timezone = str(ns.get("TIMEZONE", "UTC"))
    roll_output_dir = ns["OUTPUT_ROOT"] / "04_attitude_corrected_curves"
    roll_output_dir.mkdir(parents=True, exist_ok=True)

    roll_sign_rows: list[dict] = []
    roll_candidate_cache: dict[int, list[dict]] = {}
    for sign in [-1, 1]:
        per_ping: list[dict] = []
        for record in curve_records:
            curve_z = record["curve_z"]
            active = np.isfinite(curve_z)
            y = record["y_axis"][active]
            z = curve_z[active]
            ping_t = ping_time_unix(ds, record["ping"])
            roll_deg = interpolate_auv_value(auv_motion, auv_time, "roll_filtered_deg", ping_t)
            y_corr, z_corr = roll_correct_yz(y, z, roll_deg, sign=sign)
            before = robust_curve_metrics(y, z)
            after = robust_curve_metrics(y_corr, z_corr)
            per_ping.append({
                "ping": int(record["ping"]),
                "ping_time_unix_s": ping_t,
                "ping_time_local": ping_time_local_string(ping_t, timezone=timezone),
                "roll_sign": sign,
                "roll_filtered_deg": roll_deg,
                "pitch_filtered_deg": interpolate_auv_value(auv_motion, auv_time, "pitch_filtered_deg", ping_t),
                "nav_heading_deg": interpolate_auv_value(auv_motion, auv_time, "NAV_Heading", ping_t),
                "before_slope": before["slope"],
                "before_roughness_m": before["roughness_m"],
                "before_span_m": before["span_m"],
                "before_median_z_m": before["median_z_m"],
                "corrected_slope": after["slope"],
                "corrected_roughness_m": after["roughness_m"],
                "corrected_span_m": after["span_m"],
                "corrected_median_z_m": after["median_z_m"],
            })
        roll_candidate_cache[sign] = per_ping
        slopes = np.asarray([abs(row["corrected_slope"]) for row in per_ping], dtype=float)
        roughness = np.asarray([row["corrected_roughness_m"] for row in per_ping], dtype=float)
        median_z = np.asarray([row["corrected_median_z_m"] for row in per_ping], dtype=float)
        sign_score = float(np.nanmedian(slopes) + 0.40 * np.nanmedian(roughness) + 0.03 * np.nanstd(median_z))
        roll_sign_rows.append({
            "roll_sign": sign,
            "sign_score": sign_score,
            "median_abs_corrected_slope": float(np.nanmedian(slopes)),
            "median_corrected_roughness_m": float(np.nanmedian(roughness)),
            "std_corrected_median_z_m": float(np.nanstd(median_z)),
        })

    roll_sign_rows = sorted(roll_sign_rows, key=lambda row: row["sign_score"])
    selected_roll_sign = int(roll_sign_rows[0]["roll_sign"])
    roll_summary_rows = roll_candidate_cache[selected_roll_sign]
    ns["write_csv"](
        roll_output_dir / "roll_sign_score_summary.csv",
        roll_sign_rows,
        ["roll_sign", "sign_score", "median_abs_corrected_slope", "median_corrected_roughness_m", "std_corrected_median_z_m"],
    )
    ns["write_csv"](
        roll_output_dir / "roll_corrected_ping_summary.csv",
        roll_summary_rows,
        [
            "ping", "ping_time_unix_s", "ping_time_local", "roll_sign", "roll_filtered_deg",
            "pitch_filtered_deg", "nav_heading_deg", "before_slope", "before_roughness_m",
            "before_span_m", "before_median_z_m", "corrected_slope", "corrected_roughness_m",
            "corrected_span_m", "corrected_median_z_m",
        ],
    )

    curve_point_rows: list[dict] = []
    for record in curve_records:
        ping_idx = int(record["ping"])
        active = np.isfinite(record["curve_z"])
        y = record["y_axis"][active]
        z = record["curve_z"][active]
        w = record["curve_weight"][active]
        point_sources = np.where(record["curve_interpolated_mask"][active], "interpolated_short_gap", "measured")
        ping_t = ping_time_unix(ds, ping_idx)
        roll_deg = interpolate_auv_value(auv_motion, auv_time, "roll_filtered_deg", ping_t)
        pitch_deg = interpolate_auv_value(auv_motion, auv_time, "pitch_filtered_deg", ping_t)
        heading_deg = interpolate_auv_value(auv_motion, auv_time, "NAV_Heading", ping_t)
        y_corr, z_corr = roll_correct_yz(y, z, roll_deg, selected_roll_sign)
        if np.isfinite(pitch_deg):
            z_att = z_corr * np.cos(np.deg2rad(pitch_deg))
        else:
            z_att = np.full_like(z_corr, np.nan, dtype=float)
        for y0, z0, wc, yc, zc, za, source in zip(y, z, w, y_corr, z_corr, z_att, point_sources):
            curve_point_rows.append({
                "ping": ping_idx,
                "ping_time_unix_s": ping_t,
                "ping_time_local": ping_time_local_string(ping_t, timezone=timezone),
                "roll_sign": selected_roll_sign,
                "roll_filtered_deg": roll_deg,
                "pitch_filtered_deg": pitch_deg,
                "nav_heading_deg": heading_deg,
                "y_s_m": float(y0),
                "z_s_m": float(z0),
                "curve_weight": float(wc),
                "curve_point_source": str(source),
                "y_roll_corrected_m": float(yc),
                "z_roll_corrected_m": float(zc),
                "y_attitude_corrected_m": float(yc),
                "z_attitude_corrected_m": float(za),
            })

    ns["write_csv"](
        roll_output_dir / "roll_corrected_ice_curve_points.csv",
        curve_point_rows,
        [
            "ping", "ping_time_unix_s", "ping_time_local", "roll_sign", "roll_filtered_deg",
            "pitch_filtered_deg", "nav_heading_deg", "y_s_m", "z_s_m", "curve_weight",
            "curve_point_source", "y_roll_corrected_m", "z_roll_corrected_m",
            "y_attitude_corrected_m", "z_attitude_corrected_m",
        ],
    )
    print(f"Selected roll sign: {selected_roll_sign}")
    return roll_summary_rows, curve_point_rows, selected_roll_sign


def render_attitude_curve_figures(ns: dict, curve_point_rows: list[dict]) -> list[dict]:
    """Render publication-style original and attitude-corrected ice-bottom curves."""
    output_dir = ns["OUTPUT_ROOT"] / "04_attitude_corrected_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not curve_point_rows:
        return []

    by_ping: dict[int, list[dict]] = {}
    for row in curve_point_rows:
        by_ping.setdefault(int(row["ping"]), []).append(row)

    fig_rows: list[dict] = []
    for ping_idx in sorted(by_ping):
        rows = by_ping[ping_idx]
        y_raw = np.asarray([row["y_s_m"] for row in rows], dtype=float)
        z_raw = np.asarray([row["z_s_m"] for row in rows], dtype=float)
        y_corr = np.asarray([row.get("y_attitude_corrected_m", row.get("y_roll_corrected_m", np.nan)) for row in rows], dtype=float)
        z_corr = np.asarray([row.get("z_attitude_corrected_m", row.get("z_roll_corrected_m", np.nan)) for row in rows], dtype=float)
        roll_deg = float(rows[0].get("roll_filtered_deg", np.nan))
        pitch_deg = float(rows[0].get("pitch_filtered_deg", np.nan))
        ping_time = rows[0].get("ping_time_local", "")

        raw_valid = np.isfinite(y_raw) & np.isfinite(z_raw)
        corr_valid = np.isfinite(y_corr) & np.isfinite(z_corr)
        if np.count_nonzero(raw_valid) < 2 or np.count_nonzero(corr_valid) < 2:
            continue

        raw_order = np.argsort(y_raw[raw_valid])
        corr_order = np.argsort(y_corr[corr_valid])
        y_raw_plot = y_raw[raw_valid][raw_order]
        z_raw_plot = z_raw[raw_valid][raw_order]
        y_corr_plot = y_corr[corr_valid][corr_order]
        z_corr_plot = z_corr[corr_valid][corr_order]

        all_y = np.concatenate([y_raw_plot, y_corr_plot])
        all_z = np.concatenate([z_raw_plot, z_corr_plot])
        x_abs = max(15.0, float(np.nanmax(np.abs(all_y))) + 0.5)
        z_low, z_high = np.nanpercentile(all_z, [1.0, 99.0])
        z_center = 0.5 * (float(z_low) + float(z_high))
        z_span = max(3.60, float(z_high - z_low) + 1.40)
        z_min = max(0.0, z_center - 0.62 * z_span)
        z_max = z_min + z_span

        fig_width = 18.0 / 2.54
        fig_height = 5.8 / 2.54
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, bottom=0.22, top=0.82, wspace=0.25)
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_corr = fig.add_subplot(gs[0, 1])

        ax_raw.plot(y_raw_plot, z_raw_plot, color="#00d5ff", linewidth=1.15, label="Original Line")
        ax_raw.set_title("Original Ice-bottom Line", fontname="Arial", fontsize=8.8, fontweight="bold")
        ax_raw.legend(loc="lower right", fontsize=7.0, frameon=True)

        ax_corr.plot(y_raw_plot, z_raw_plot, color="#a8a8a8", linewidth=0.95, alpha=0.95, label="Original Line")
        ax_corr.plot(y_corr_plot, z_corr_plot, color="#1f77b4", linewidth=1.15, label="Attitude-corrected Line")
        ax_corr.set_title("Original vs Attitude-corrected Line", fontname="Arial", fontsize=8.8, fontweight="bold")
        ax_corr.legend(loc="lower right", fontsize=7.0, frameon=True)

        for ax in (ax_raw, ax_corr):
            ax.set_xlim(-x_abs, x_abs)
            ax.set_ylim(z_min, z_max)
            ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)
            ax.tick_params(labelsize=6.6, width=0.7, length=2.6)
            ax.set_xlabel("Sonar-frame Y (m)", fontname="Arial", fontsize=7.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
        ax_raw.set_ylabel("Z (m)", fontname="Arial", fontsize=7.5)

        output_path = output_dir / f"attitude_curve_ping_{ping_idx:04d}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        fig_rows.append({
            "ping": int(ping_idx),
            "ping_time_local": str(ping_time),
            "roll_filtered_deg": roll_deg,
            "pitch_filtered_deg": pitch_deg,
            "raw_point_count": int(np.count_nonzero(raw_valid)),
            "corrected_point_count": int(np.count_nonzero(corr_valid)),
            "figure_path": str(output_path),
        })

    ns["write_csv"](
        output_dir / "attitude_curve_figure_summary.csv",
        fig_rows,
        ["ping", "ping_time_local", "roll_filtered_deg", "pitch_filtered_deg", "raw_point_count", "corrected_point_count", "figure_path"],
    )
    return fig_rows


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(valid) == 0:
        return np.nan
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = float(quantile) * cumulative[-1]
    return float(values[np.searchsorted(cumulative, cutoff, side="left")])


def robust_centerline_stat(point_rows: list[dict], center_window_m: float, min_count: int, fallback_count: int) -> dict | None:
    if not point_rows:
        return None
    y_corr = np.asarray([row["y_roll_corrected_m"] for row in point_rows], dtype=float)
    z_corr = np.asarray([row["z_roll_corrected_m"] for row in point_rows], dtype=float)
    y_s = np.asarray([row["y_s_m"] for row in point_rows], dtype=float)
    z_s = np.asarray([row["z_s_m"] for row in point_rows], dtype=float)
    weights = np.asarray([row["curve_weight"] for row in point_rows], dtype=float)
    valid = np.isfinite(y_corr) & np.isfinite(z_corr) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(valid) == 0:
        return None
    center_mask = valid & (np.abs(y_corr) <= center_window_m)
    method = "roll_corrected_center_window"
    if np.count_nonzero(center_mask) < min_count:
        valid_indices = np.flatnonzero(valid)
        nearest_order = valid_indices[np.argsort(np.abs(y_corr[valid_indices]))]
        take_n = min(max(fallback_count, min_count), nearest_order.size)
        center_mask = np.zeros_like(valid, dtype=bool)
        center_mask[nearest_order[:take_n]] = True
        method = "nearest_roll_corrected_center_points"
    if np.count_nonzero(center_mask) == 0:
        return None
    selected_y_corr = y_corr[center_mask]
    selected_z_corr = z_corr[center_mask]
    selected_y_s = y_s[center_mask]
    selected_z_s = z_s[center_mask]
    selected_weights = weights[center_mask]
    z50 = weighted_quantile(selected_z_corr, selected_weights, 0.50)
    z25 = weighted_quantile(selected_z_corr, selected_weights, 0.25)
    z75 = weighted_quantile(selected_z_corr, selected_weights, 0.75)
    return {
        "center_method": method,
        "center_point_count": int(selected_z_corr.size),
        "center_y_roll_corrected_m": weighted_quantile(selected_y_corr, selected_weights, 0.50),
        "center_z_roll_corrected_m": z50,
        "center_z_roll_corrected_q25_m": z25,
        "center_z_roll_corrected_q75_m": z75,
        "center_z_roll_corrected_iqr_m": float(z75 - z25) if np.isfinite(z75) and np.isfinite(z25) else np.nan,
        "center_y_s_m": weighted_quantile(selected_y_s, selected_weights, 0.50),
        "center_z_s_m": weighted_quantile(selected_z_s, selected_weights, 0.50),
        "center_weight_sum": float(np.nansum(selected_weights)),
    }


def build_draft_table(ns: dict, ds, curve_point_rows: list[dict], selected_roll_sign: int) -> pd.DataFrame:
    auv_motion, auv_time = load_auv_motion(ns["MOTION_FILE"])
    timezone = str(ns.get("TIMEZONE", "UTC"))
    draft_output_dir = ns["OUTPUT_ROOT"] / "05_draft_timeseries"
    draft_output_dir.mkdir(parents=True, exist_ok=True)
    center_window_m = 0.50
    min_count = 5
    fallback_count = 9
    sonar_center_beam_tilt_from_vertical_deg = 0.0
    pitch_sign_for_vertical = 1.0

    by_ping: dict[int, list[dict]] = {}
    for row in curve_point_rows:
        by_ping.setdefault(int(row["ping"]), []).append(row)

    rows: list[dict] = []
    for ping_idx in ns["BATCH_PINGS"]:
        center = robust_centerline_stat(by_ping.get(int(ping_idx), []), center_window_m, min_count, fallback_count)
        if center is None:
            continue
        ping_t = ping_time_unix(ds, int(ping_idx))
        nav_depth = interpolate_auv_value(auv_motion, auv_time, "NAV_DEPTH", ping_t)
        nav_altitude = interpolate_auv_value(auv_motion, auv_time, "NAV_ALTITUDE", ping_t)
        pitch_deg = interpolate_auv_value(auv_motion, auv_time, "pitch_filtered_deg", ping_t)
        roll_deg = interpolate_auv_value(auv_motion, auv_time, "roll_filtered_deg", ping_t)
        heading_deg = interpolate_auv_value(auv_motion, auv_time, "NAV_Heading", ping_t)
        theta_deg = sonar_center_beam_tilt_from_vertical_deg + pitch_sign_for_vertical * pitch_deg
        theta_plus_deg = sonar_center_beam_tilt_from_vertical_deg + pitch_deg
        theta_minus_deg = sonar_center_beam_tilt_from_vertical_deg - pitch_deg
        vertical_range = center["center_z_roll_corrected_m"] * np.cos(np.deg2rad(theta_deg))
        vertical_range_plus = center["center_z_roll_corrected_m"] * np.cos(np.deg2rad(theta_plus_deg))
        vertical_range_minus = center["center_z_roll_corrected_m"] * np.cos(np.deg2rad(theta_minus_deg))
        rows.append({
            "ping": int(ping_idx),
            "ping_time_unix_s": ping_t,
            "ping_time_local": ping_time_local_string(ping_t, timezone=timezone),
            "nav_depth_m": nav_depth,
            "nav_altitude_m": nav_altitude,
            "pitch_filtered_deg": pitch_deg,
            "roll_filtered_deg": roll_deg,
            "nav_heading_deg": heading_deg,
            "roll_sign": selected_roll_sign,
            "center_window_y_m": center_window_m,
            "sonar_center_beam_tilt_from_vertical_deg": sonar_center_beam_tilt_from_vertical_deg,
            "pitch_sign_for_vertical": pitch_sign_for_vertical,
            "theta_vertical_deg": theta_deg,
            "vertical_range_m": vertical_range,
            "ice_draft_m": nav_depth - vertical_range,
            "ice_draft_pitch_plus_m": nav_depth - vertical_range_plus,
            "ice_draft_pitch_minus_m": nav_depth - vertical_range_minus,
            "pitch_sign_sensitivity_m": abs((nav_depth - vertical_range_plus) - (nav_depth - vertical_range_minus)),
            **center,
        })
    df = pd.DataFrame(rows).sort_values("ping_time_unix_s").reset_index(drop=True)
    if df.empty:
        return df
    df["time_dt"] = pd.to_datetime(df["ping_time_local"])
    raw_draft = pd.to_numeric(df["ice_draft_m"], errors="coerce")
    physical_invalid = raw_draft < 0.0
    baseline_input = raw_draft.mask(physical_invalid)
    baseline_interp = baseline_input.interpolate(limit_direction="both")
    df["ice_draft_baseline_m"] = baseline_interp.rolling(window=3, center=True, min_periods=1).median()
    df["draft_temporal_residual_m"] = raw_draft - df["ice_draft_baseline_m"]
    residual_invalid = df["draft_temporal_residual_m"].abs() > DRAFT_QC_RESIDUAL_THRESHOLD_M
    draft_qc_flag = physical_invalid | residual_invalid
    df["draft_qc_physical_flag"] = physical_invalid.astype(int)
    df["draft_qc_residual_flag"] = residual_invalid.astype(int)
    df["draft_qc_flag"] = draft_qc_flag.astype(int)
    df["draft_qc_reason"] = np.where(
        physical_invalid,
        "negative_draft",
        np.where(residual_invalid, "temporal_residual", ""),
    )
    qc_input = raw_draft.mask(draft_qc_flag).interpolate(limit_direction="both")
    df["ice_draft_qc_input_m"] = qc_input
    df["ice_draft_smoothed_m"] = qc_input.rolling(window=3, center=True, min_periods=1).median()
    fieldnames = [
        "ping", "ping_time_unix_s", "ping_time_local", "nav_depth_m", "nav_altitude_m",
        "pitch_filtered_deg", "roll_filtered_deg", "nav_heading_deg", "roll_sign",
        "center_window_y_m", "sonar_center_beam_tilt_from_vertical_deg", "pitch_sign_for_vertical",
        "theta_vertical_deg", "vertical_range_m", "ice_draft_m", "ice_draft_baseline_m",
        "ice_draft_qc_input_m", "ice_draft_smoothed_m", "draft_temporal_residual_m",
        "draft_qc_flag", "draft_qc_physical_flag", "draft_qc_residual_flag",
        "draft_qc_reason", "ice_draft_pitch_plus_m",
        "ice_draft_pitch_minus_m", "pitch_sign_sensitivity_m", "center_method", "center_point_count",
        "center_y_roll_corrected_m", "center_z_roll_corrected_m", "center_z_roll_corrected_q25_m",
        "center_z_roll_corrected_q75_m", "center_z_roll_corrected_iqr_m", "center_y_s_m",
        "center_z_s_m", "center_weight_sum",
    ]
    ns["write_csv"](draft_output_dir / "ice_draft_centerline_summary.csv", df.drop(columns=["time_dt"]).to_dict("records"), fieldnames)
    save_draft_figure(draft_output_dir / "ice_draft_time_series_with_auv_depth.png", df, ns["BATCH_PINGS"])
    return df


def save_draft_figure(output_path: Path, df: pd.DataFrame, batch_pings: list[int]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), dpi=150, sharex=True, constrained_layout=True)
    ax_draft, ax_depth = axes
    ax_draft.plot(df["time_dt"], df["ice_draft_m"], color="0.68", marker="o", markersize=3.8, linewidth=1.0, label="raw centerline draft")
    ax_draft.plot(df["time_dt"], df["ice_draft_smoothed_m"], color="#1f77b4", marker="o", markersize=4.4, linewidth=1.8, label="3-point median draft")
    ax_draft.fill_between(
        df["time_dt"],
        df["ice_draft_smoothed_m"] - 0.5 * df["center_z_roll_corrected_iqr_m"].fillna(0.0),
        df["ice_draft_smoothed_m"] + 0.5 * df["center_z_roll_corrected_iqr_m"].fillna(0.0),
        color="#1f77b4", alpha=0.14, linewidth=0, label="centerline IQR / 2",
    )
    qc_points = df[df["draft_qc_flag"] == 1]
    if not qc_points.empty:
        ax_draft.scatter(
            qc_points["time_dt"],
            qc_points["ice_draft_m"],
            s=44,
            facecolors="none",
            edgecolors="#c44e52",
            linewidths=1.2,
            label=f"QC residual > {DRAFT_QC_RESIDUAL_THRESHOLD_M:.2f} m or draft < 0",
        )
    ax_draft.axhline(0.0, color="0.25", linewidth=0.9, linestyle="--")
    ax_draft.set_ylabel("Ice draft (m)", fontname="Arial")
    ax_draft.invert_yaxis()
    ax_draft.set_title("Centerline sea-ice draft along time", fontname="Arial")
    ax_draft.grid(True, alpha=0.28, linestyle=":")
    ax_draft.legend(loc="best", fontsize=8, frameon=True)

    ax_depth.plot(df["time_dt"], df["nav_depth_m"], color="#c44e52", marker="s", markersize=3.8, linewidth=1.5, label="AUV NAV_DEPTH")
    ax_depth.set_ylabel("AUV depth (m)", fontname="Arial")
    ax_depth.invert_yaxis()
    ax_depth.set_xlabel("Time", fontname="Arial")
    ax_depth.grid(True, alpha=0.28, linestyle=":")
    ax_depth.legend(loc="best", fontsize=8, frameon=True)
    ax_depth.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.suptitle(
        f"Sea-ice draft MVP | pings {batch_pings[0]}-{batch_pings[-1]} | center window +/-0.50 m",
        y=1.02,
        fontname="Arial",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
