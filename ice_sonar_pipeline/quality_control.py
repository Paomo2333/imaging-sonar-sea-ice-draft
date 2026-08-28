"""QC tables and no-ice/low-confidence candidate summaries."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

DRAFT_QC_RESIDUAL_THRESHOLD_M = 0.15
DRAFT_QC_GOOD_RESIDUAL_M = 0.05


def finite_clip(values, lower=0.0, upper=1.0):
    return np.clip(np.asarray(values, dtype=float), lower, upper)


def badness_high(values, good_level, bad_level):
    values = np.asarray(values, dtype=float)
    if bad_level == good_level:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - good_level) / (bad_level - good_level), 0.0, 1.0)


def badness_low(values, good_level, bad_level):
    values = np.asarray(values, dtype=float)
    if good_level == bad_level:
        return np.zeros_like(values, dtype=float)
    return np.clip((good_level - values) / (good_level - bad_level), 0.0, 1.0)


def build_qc_table(ns: dict, gray_rows: list[dict], manual_rows: list[dict], binary_rows: list[dict], roll_rows: list[dict], draft_df: pd.DataFrame) -> pd.DataFrame:
    qc_dir = ns["OUTPUT_ROOT"] / "06_qc_dashboard"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_df = pd.DataFrame(binary_rows).copy()
    gray_df = pd.DataFrame(gray_rows)
    manual_df = pd.DataFrame(manual_rows)
    roll_df = pd.DataFrame(roll_rows)
    if not gray_df.empty:
        qc_df = qc_df.merge(
            gray_df[["ping", "score", "ice_retention", "center_residual", "lobe_residual", "peak_quality", "selected_ice_z_m"]],
            on="ping",
            how="left",
            suffixes=("", "_gray"),
        )
    if not manual_df.empty:
        manual_cols = [col for col in ["ping", "manual_edited", "manual_remove_polygon_count", "manual_removed_area_m2", "manual_removed_energy_fraction"] if col in manual_df.columns]
        qc_df = qc_df.drop(columns=[col for col in manual_cols if col != "ping" and col in qc_df.columns], errors="ignore")
        qc_df = qc_df.merge(manual_df[manual_cols], on="ping", how="left")
    if not roll_df.empty:
        qc_df = qc_df.merge(
            roll_df[["ping", "before_slope", "corrected_slope", "corrected_roughness_m", "roll_filtered_deg", "pitch_filtered_deg"]],
            on="ping",
            how="left",
        )
    if not draft_df.empty:
        draft_cols = [
            "ping", "ping_time_local", "nav_depth_m", "ice_draft_m", "ice_draft_smoothed_m",
            "draft_temporal_residual_m", "draft_qc_flag", "draft_qc_physical_flag",
            "draft_qc_residual_flag", "draft_qc_reason", "center_z_roll_corrected_iqr_m",
            "center_point_count", "pitch_sign_sensitivity_m",
        ]
        qc_df = qc_df.merge(draft_df[[col for col in draft_cols if col in draft_df.columns]], on="ping", how="left")
    for col in ["manual_edited", "manual_remove_polygon_count", "manual_removed_area_m2", "manual_removed_energy_fraction"]:
        if col in qc_df.columns:
            qc_df[col] = qc_df[col].fillna(0)

    score_bad = badness_low(qc_df["binary_score"], good_level=0.65, bad_level=0.25)
    capture_bad = badness_low(qc_df["ice_energy_capture"], good_level=0.70, bad_level=0.45)
    left_bad = badness_low(qc_df["left_support_score"], good_level=0.70, bad_level=0.35)
    center_bad = badness_high(qc_df["binary_center_residual"], good_level=0.01, bad_level=0.08)
    lobe_bad = badness_high(qc_df["binary_lobe_residual"], good_level=0.02, bad_level=0.12)
    span_bad = badness_low(qc_df["y_span_m"], good_level=7.0, bad_level=2.0)
    rough_bad = badness_high(qc_df["curve_roughness_m"], good_level=0.08, bad_level=0.35)
    align_bad = badness_high(qc_df["z_alignment_m"], good_level=0.20, bad_level=0.85)
    draft_bad = badness_high(
        np.abs(qc_df.get("draft_temporal_residual_m", pd.Series(0.0, index=qc_df.index)).fillna(0.0)),
        good_level=DRAFT_QC_GOOD_RESIDUAL_M,
        bad_level=DRAFT_QC_RESIDUAL_THRESHOLD_M,
    )
    if "draft_qc_physical_flag" in qc_df.columns:
        draft_bad = np.maximum(draft_bad, qc_df["draft_qc_physical_flag"].fillna(0).to_numpy(dtype=float))
    iqr_bad = badness_high(qc_df.get("center_z_roll_corrected_iqr_m", pd.Series(0.0, index=qc_df.index)).fillna(0.0), good_level=0.12, bad_level=0.60)
    qc_df["qc_review_score"] = (
        0.18 * score_bad + 0.16 * capture_bad + 0.12 * left_bad + 0.13 * center_bad
        + 0.13 * lobe_bad + 0.09 * span_bad + 0.08 * rough_bad + 0.06 * align_bad
        + 0.03 * draft_bad + 0.02 * iqr_bad
    )
    qc_df["qc_span_support"] = 1.0 - span_bad
    qc_df["qc_left_support"] = 1.0 - left_bad
    qc_df["qc_alignment_support"] = 1.0 - align_bad
    qc_df["qc_roughness_support"] = 1.0 - rough_bad
    qc_df["qc_artifact_support"] = 1.0 - np.maximum(center_bad, lobe_bad)
    qc_df["qc_review_level"] = pd.cut(qc_df["qc_review_score"], bins=[-0.01, 0.25, 0.45, 1.01], labels=["pass", "check", "review"]).astype(str)
    qc_df["qc_review_percentile"] = qc_df["qc_review_score"].rank(method="average", pct=True)
    qc_df["qc_relative_priority"] = pd.cut(qc_df["qc_review_percentile"], bins=[-0.01, 0.70, 0.90, 1.01], labels=["low", "medium", "high"]).astype(str)

    fieldnames = [
        "ping", "ping_time_local", "binary_score", "ice_energy_capture", "left_support_score",
        "left_support_needed", "left_signal_fraction", "left_ice_energy_capture", "left_y_span_m",
        "left_gray_y_span_m", "left_edge_gap_m", "left_max_gap_m", "left_continuity_score",
        "left_rescue_applied", "left_rescue_pixel_count", "binary_center_residual", "binary_lobe_residual",
        "y_span_m", "continuity_score", "curve_roughness_m", "z_alignment_m", "manual_edited",
        "manual_remove_polygon_count", "manual_removed_area_m2", "manual_removed_energy_fraction",
        "before_slope", "corrected_slope", "corrected_roughness_m", "roll_filtered_deg", "pitch_filtered_deg",
        "nav_depth_m", "ice_draft_m", "ice_draft_smoothed_m", "draft_temporal_residual_m", "draft_qc_flag",
        "draft_qc_physical_flag", "draft_qc_residual_flag", "draft_qc_reason",
        "center_z_roll_corrected_iqr_m", "center_point_count", "pitch_sign_sensitivity_m",
        "qc_span_support", "qc_left_support", "qc_alignment_support", "qc_roughness_support",
        "qc_artifact_support", "qc_review_score", "qc_review_percentile", "qc_review_level", "qc_relative_priority",
    ]
    ns["write_csv"](qc_dir / "qc_ping_score_summary.csv", qc_df.replace({np.nan: None}).to_dict("records"), fieldnames)
    return qc_df


def classify_no_ice_candidates(ns: dict, gray_rows: list[dict], binary_rows: list[dict], qc_df: pd.DataFrame, draft_df: pd.DataFrame) -> pd.DataFrame:
    review_dir = ns["OUTPUT_ROOT"] / "07_no_ice_bottom_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = review_dir / "candidate_process_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    gray_df = pd.DataFrame(gray_rows)
    binary_df = pd.DataFrame(binary_rows)
    merged = binary_df.copy()
    if not gray_df.empty:
        cols = [c for c in ["ping", "score", "peak_quality", "selected_ice_z_m", "manual_review_flag"] if c in gray_df.columns]
        merged = merged.merge(gray_df[cols], on="ping", how="left")
    if not qc_df.empty:
        cols = [c for c in ["ping", "qc_review_score", "qc_review_level", "qc_relative_priority"] if c in qc_df.columns]
        merged = merged.merge(qc_df[cols], on="ping", how="left")
    if not draft_df.empty:
        cols = [
            c
            for c in [
                "ping", "ice_draft_m", "ice_draft_smoothed_m", "draft_temporal_residual_m",
                "draft_qc_flag", "draft_qc_physical_flag", "draft_qc_residual_flag",
                "draft_qc_reason", "center_point_count", "center_z_roll_corrected_iqr_m",
            ]
            if c in draft_df.columns
        ]
        merged = merged.merge(draft_df[cols], on="ping", how="left")

    for col in [
        "score", "peak_quality", "binary_score", "ice_energy_capture", "y_span_m",
        "center_point_count", "center_z_roll_corrected_iqr_m", "draft_temporal_residual_m",
        "draft_qc_physical_flag", "draft_qc_residual_flag",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    reasons: list[str] = []
    no_ice_flags: list[bool] = []
    low_conf_flags: list[bool] = []
    for _, row in merged.iterrows():
        row_reasons: list[str] = []
        if row.get("score", np.nan) < 0.20:
            row_reasons.append("weak_gray_ice_peak")
        if row.get("peak_quality", np.nan) < 0.15:
            row_reasons.append("low_projection_peak_quality")
        if row.get("ice_energy_capture", np.nan) < 0.40:
            row_reasons.append("low_ice_energy_capture")
        if row.get("binary_score", np.nan) < 0.25:
            row_reasons.append("low_binary_score")
        if row.get("y_span_m", np.nan) < 2.0:
            row_reasons.append("short_lateral_span")
        if row.get("center_point_count", np.inf) < 5:
            row_reasons.append("too_few_center_points")
        if row.get("center_z_roll_corrected_iqr_m", 0.0) > 0.80:
            row_reasons.append("unstable_centerline_iqr")
        if row.get("draft_qc_physical_flag", 0) == 1:
            row_reasons.append("negative_ice_draft")
        elif row.get("draft_qc_residual_flag", 0) == 1:
            row_reasons.append("draft_residual_gt_0.15m")
        strong_no_ice = (
            ("low_ice_energy_capture" in row_reasons and "short_lateral_span" in row_reasons)
            or ("weak_gray_ice_peak" in row_reasons and "low_binary_score" in row_reasons)
            or (row.get("ice_energy_capture", np.nan) < 0.30)
        )
        no_ice_flags.append(bool(strong_no_ice))
        low_conf_flags.append(bool(row_reasons))
        reasons.append(";".join(row_reasons))
    merged["no_ice_bottom_candidate"] = no_ice_flags
    merged["low_confidence_candidate"] = low_conf_flags
    merged["candidate_reason"] = reasons
    merged.to_csv(review_dir / "no_ice_bottom_candidate_summary.csv", index=False, encoding="utf-8-sig")
    no_ice = merged[merged["no_ice_bottom_candidate"]].copy()
    (review_dir / "no_ice_bottom_ping_list.txt").write_text("\n".join(str(int(p)) for p in no_ice["ping"].dropna()), encoding="utf-8")
    for ping in no_ice["ping"].dropna().astype(int).tolist():
        source = ns["BINARY_OUTPUT_DIR"] / f"binary_ping_{ping:04d}.png"
        if source.exists():
            shutil.copy2(source, figures_dir / f"no_ice_candidate_ping_{ping:04d}.png")
    print(f"No-ice candidates: {int(merged['no_ice_bottom_candidate'].sum())}; low-confidence candidates: {int(merged['low_confidence_candidate'].sum())}")
    return merged
