"""Optional, user-configurable time-series QC and display smoothing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fill(values, times, options):
    if not options["interpolate"]:
        return values.copy()
    limit = options["max_gap_s"]
    if limit is None:
        # Preserve the original index-based behavior when no limit is requested.
        return values.interpolate(limit_direction="both")
    out = values.copy()
    valid = np.flatnonzero(np.isfinite(values.to_numpy(dtype=float)))
    for left, right in zip(valid[:-1], valid[1:]):
        if right > left + 1 and 0 < times[right] - times[left] <= limit:
            out.iloc[left + 1:right] = np.interp(times[left + 1:right], times[[left, right]], values.iloc[[left, right]])
    return out


def _median(values, times, window, gap):
    if gap is None:
        out = values.rolling(window=window, center=True, min_periods=1).median()
    else:
        out = values.copy()
        cuts = np.r_[0, np.flatnonzero(np.diff(times) > gap) + 1, len(times)]
        for left, right in zip(cuts[:-1], cuts[1:]):
            out.iloc[left:right] = values.iloc[left:right].rolling(window=window, center=True, min_periods=1).median()
    # Smoothing must not silently restore samples that were deliberately unfilled.
    return out.where(values.notna())


def add_postprocessing_columns(df, options):
    """Retain ice_draft_m unchanged; append QC and optional processed columns."""
    df = df.copy()
    raw = pd.to_numeric(df["ice_draft_m"], errors="coerce")
    times = df["ping_time_unix_s"].to_numpy(dtype=float)
    if options["enabled"]:
        physical = raw < 0
        baseline_input = _fill(raw.mask(physical), times, options)
        baseline = _median(baseline_input, times, options["baseline_window_points"], options["max_gap_s"])
        residual = raw - baseline
        residual_flag = residual.abs() > options["residual_threshold_m"]
        flagged = physical | residual_flag
        qc_input = _fill(raw.mask(flagged), times, options)
        smoothed = _median(qc_input, times, options["smooth_window_points"], options["max_gap_s"])
    else:
        physical = residual_flag = flagged = pd.Series(False, index=df.index)
        baseline = qc_input = smoothed = raw.copy()
        residual = pd.Series(np.nan, index=df.index)
    df["ice_draft_baseline_m"] = baseline
    df["draft_temporal_residual_m"] = residual
    df["draft_qc_physical_flag"] = physical.astype(int)
    df["draft_qc_residual_flag"] = residual_flag.astype(int)
    df["draft_qc_flag"] = flagged.astype(int)
    df["draft_qc_reason"] = np.where(physical, "negative_draft", np.where(residual_flag, "temporal_residual", ""))
    df["ice_draft_qc_input_m"] = qc_input
    df["ice_draft_smoothed_m"] = smoothed
    return df
