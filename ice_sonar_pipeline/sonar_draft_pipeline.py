"""Imaging-sonar ice-draft batch preprocessing pipeline.

This driver preserves the validated notebook algorithm and the original
per-frame process figure style, but avoids keeping every mapped sonar image in
memory. It is for visual review runs: one output folder per NetCDF file,
process figures for each sampled frame, and automatic low-confidence/no-ice
candidate summaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .algorithm_cells import get_cell_source
from .artifact_suppression import install_gray_denoising
from .draft_reconstruction import (
    build_draft_table,
    build_roll_and_curve_tables,
    interpolate_auv_value,
    load_auv_motion,
    ping_time_local_string,
    ping_time_unix,
    render_attitude_curve_figures,
    robust_curve_metrics,
    roll_correct_yz,
)
from .ice_candidate import install_binary_band_refinement, install_publication_binary_renderer
from .ice_line_extraction import build_curve_record, selected_binary_row
from .quality_control import build_qc_table, classify_no_ice_candidates


PACKAGE_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = PACKAGE_DIR
DRAFT_QC_RESIDUAL_THRESHOLD_M = 0.15
DRAFT_QC_GOOD_RESIDUAL_M = 0.05
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))
BINARY_MIN_WINDOW_DOWN_M = 1.20
BINARY_MIN_WINDOW_UP_M = 0.60
BINARY_EDGE_EXTENSION_DOWN_M = 1.70
BINARY_EDGE_EXTENSION_MAX_Y_M = 4.50
BINARY_EDGE_EXTENSION_STEP_Z_M = 0.35
BINARY_EDGE_EXTENSION_HALF_WIDTH_M = 0.12
BINARY_EDGE_EXTENSION_MAX_MISSES = 8
BINARY_GAP_CONNECT_MAX_Y_M = 1.80
BINARY_GAP_CONNECT_MAX_Z_M = 0.35
BINARY_MASK_BRIDGE_HALF_WIDTH_M = 0.10
BINARY_MASK_BRIDGE_SUPPORT_SCALE = 0.40
BINARY_MASK_BRIDGE_SUPPORT_MIN = 0.12
BINARY_MASK_BRIDGE_SUPPORT_FRACTION = 0.20
BINARY_LINE_CONNECT_MAX_Y_M = 0.60

_BINARY_TUNING_WORKER_NS: dict | None = None
_BINARY_TUNING_WORKER_DS = None
_BINARY_TUNING_WORKER_SELECTED_CONFIG: dict | None = None
_BINARY_TUNING_WORKER_MANUAL_STORE: dict | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct along-track sea-ice draft from upward-looking 2D imaging-sonar NetCDF data."
    )
    parser.add_argument("--nc-file", required=True, help="NetCDF file to process.")
    parser.add_argument("--motion-file", required=True, help="CSV file containing synchronized AUV motion and depth data.")
    parser.add_argument("--output-root", required=True, help="Output directory; relative paths are resolved from the current directory.")
    parser.add_argument("--timezone", default="UTC", help="IANA timezone used for optional local-time selection and labels.")
    parser.add_argument("--start-local", default=None, help="Optional local start time interpreted using --timezone.")
    parser.add_argument("--end-local", default=None, help="Optional inclusive local end time interpreted using --timezone.")
    parser.add_argument("--start-ping", type=int, default=None, help="Optional explicit start frame index.")
    parser.add_argument("--end-ping", type=int, default=None, help="Optional inclusive final frame index.")
    parser.add_argument("--ping-step", type=int, default=5, help="Frame sampling step.")
    parser.add_argument(
        "--exclude-ping-ranges",
        default="",
        help="Comma-separated inclusive frame ranges to skip, for example 165-980,1200,1300-1310.",
    )
    parser.add_argument(
        "--grid-range-mode",
        choices=["fixed", "metadata"],
        default="fixed",
        help="Use the fixed notebook grid or adapt each frame grid to its NetCDF range metadata.",
    )
    parser.add_argument(
        "--grid-range-max",
        type=float,
        default=None,
        help="Optional cap for mapped grid range in metres. With fixed mode this sets the fixed range.",
    )
    parser.add_argument(
        "--grid-range-round-m",
        type=float,
        default=5.0,
        help="Round metadata-derived frame range up to this metre step before optional capping.",
    )
    parser.add_argument("--binary-tuning-ping-step", type=int, default=50, help="Representative binary tuning interval in ping units.")
    parser.add_argument(
        "--manual-mask-json",
        default=None,
        help="Optional user-supplied remove-only polygon mask JSON. No masks are bundled with this repository.",
    )
    parser.add_argument("--no-process-figures", action="store_true", help="Skip per-frame binary process figures.")
    parser.add_argument("--no-gray-figures", action="store_true", help="Skip per-frame gray-stage noise-suppression figures.")
    parser.add_argument("--gray-only", action="store_true", help="Only render gray-stage denoising figures and skip binary/draft processing.")
    parser.add_argument("--max-count", type=int, default=None, help="Optional test limit for sampled frames.")
    parser.add_argument(
        "--force-retune",
        action="store_true",
        help="Ignore existing selected config cache files and rerun gray/binary tuning.",
    )
    parser.add_argument(
        "--binary-tuning-workers",
        type=int,
        default=1,
        help="Parallel workers for binary candidate tuning. Use 1 for serial/reproducible default.",
    )
    return parser.parse_args()


def execute_notebook_cell(ns: dict, cell_index: int) -> None:
    code = get_cell_source(cell_index)
    if not code:
        return
    exec(compile(code, f"base_oculus_batch_processing:cell_{cell_index}", "exec"), ns)




def ping_time_local_series(ds, timezone: str = "UTC") -> pd.DatetimeIndex:
    values = ds["time"].values.astype(float)
    return pd.to_datetime(values, unit="s", utc=True).tz_convert(timezone).tz_localize(None)


def find_start_ping(ds, target_local: str | None, timezone: str = "UTC") -> int:
    if not target_local:
        return 0
    local = ping_time_local_series(ds, timezone=timezone)
    target = pd.Timestamp(target_local)
    index = int(np.searchsorted(local.values.astype("datetime64[ns]"), np.datetime64(target), side="left"))
    if index >= len(local):
        raise ValueError(f"Start time {target_local} is after the final sonar frame.")
    return index


def find_end_ping_exclusive(ds, target_local: str | None, timezone: str = "UTC") -> int:
    if not target_local:
        return int(ds.sizes["ping"])
    local = ping_time_local_series(ds, timezone=timezone)
    target = pd.Timestamp(target_local)
    index = int(np.searchsorted(local.values.astype("datetime64[ns]"), np.datetime64(target), side="right"))
    if index <= 0:
        raise ValueError(f"End time {target_local} is before the first sonar frame.")
    return min(index, int(ds.sizes["ping"]))


def resolve_end_ping_exclusive(
    ds,
    start_ping: int,
    end_ping: int | None,
    end_local: str | None,
    timezone: str = "UTC",
) -> int:
    if end_ping is not None and end_local is not None:
        raise ValueError("Use only one of --end-ping or --end-local.")
    if end_ping is not None:
        end_exclusive = min(int(end_ping) + 1, int(ds.sizes["ping"]))
    else:
        end_exclusive = find_end_ping_exclusive(ds, end_local, timezone=timezone)
    if end_exclusive <= start_ping:
        raise ValueError("The selected end frame is not after the start frame.")
    return end_exclusive


def parse_ping_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
        else:
            start = end = int(item)
        if start < 0 or end < 0:
            raise ValueError("--exclude-ping-ranges only accepts non-negative frame numbers.")
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def ping_is_excluded(ping_idx: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= int(ping_idx) <= end for start, end in ranges)


def write_excluded_ping_log(
    output_root: Path,
    sampled_pings_before_exclusion: list[int],
    excluded_ranges: list[tuple[int, int]],
) -> list[int]:
    excluded = [int(p) for p in sampled_pings_before_exclusion if ping_is_excluded(p, excluded_ranges)]
    if excluded:
        rows = [{"ping": ping, "reason": "user_excluded_ping_range"} for ping in excluded]
        pd.DataFrame(rows).to_csv(output_root / "excluded_sampled_pings.csv", index=False, encoding="utf-8-sig")
    return excluded


def install_metadata_range_mapping(
    ns: dict,
    mode: str,
    range_max_cap_m: float | None,
    range_round_m: float,
) -> None:
    base_config = ns["GRID_CONFIG"]
    if mode == "fixed":
        if range_max_cap_m is not None:
            ns["GRID_CONFIG"] = replace(base_config, range_max_m=float(range_max_cap_m))
        ns["GRID_RANGE_MODE"] = "fixed"
        ns["GRID_RANGE_MAX_CAP_M"] = range_max_cap_m
        ns["GRID_RANGE_ROUND_M"] = range_round_m
        return

    if range_round_m < 0:
        raise ValueError("--grid-range-round-m must be non-negative.")

    def frame_grid_config(metadata: dict[str, float]):
        raw_range = float(metadata["raw_range_max_m"])
        if range_round_m > 0:
            range_max = float(np.ceil(raw_range / range_round_m) * range_round_m)
        else:
            range_max = raw_range
        if range_max_cap_m is not None:
            range_max = min(range_max, float(range_max_cap_m))
        range_max = max(float(base_config.range_min_m) + float(base_config.grid_resolution_m), range_max)
        return replace(base_config, range_max_m=range_max)

    def map_ping_metadata_range(dataset, ping_idx, image_var, grid_config):
        image, metadata = ns["read_ping_frame"](dataset, ping_idx, image_var=image_var)
        ping_grid_config = frame_grid_config(metadata)
        mapped = ns["backward_map_ping_center_plane"](image, metadata, ping_grid_config)
        filled = ns["fill_internal_holes"](
            image=mapped["mapped_backscatter"],
            valid_mask=mapped["valid_mask"],
            y_axis_m=mapped["y_axis_m"],
            z_axis_m=mapped["z_axis_m"],
            max_fill_distance_m=0.15,
        )
        metadata = dict(metadata)
        metadata["grid_range_max_m"] = float(ping_grid_config.range_max_m)
        metadata["grid_range_mode"] = mode
        metadata["grid_range_max_cap_m"] = np.nan if range_max_cap_m is None else float(range_max_cap_m)
        return {
            "metadata": metadata,
            "linear": filled["linear"],
            "valid_mask": mapped["valid_mask"],
            "y_axis": mapped["y_axis_m"],
            "z_axis": mapped["z_axis_m"],
        }

    ns["map_ping"] = map_ping_metadata_range
    ns["GRID_RANGE_MODE"] = "metadata"
    ns["GRID_RANGE_MAX_CAP_M"] = range_max_cap_m
    ns["GRID_RANGE_ROUND_M"] = range_round_m


def numeric_array(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray([row.get(field, np.nan) for row in rows], dtype=float)


def load_selected_config_cache(config_path: Path, candidate_configs: list[dict], label: str) -> dict | None:
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read cached {label} config from {config_path}: {exc}")
        return None
    cached_config = data.get("config") or data.get("selected_config")
    if not isinstance(cached_config, dict):
        print(f"Cached {label} config has no usable config object: {config_path}")
        return None
    cached_name = cached_config.get("name")
    for config in candidate_configs:
        if config.get("name") == cached_name:
            print(f"Reusing cached {label} config: {cached_name}")
            return dict(config)
    print(f"Cached {label} config name is not in the current candidate list: {cached_name}")
    return None


def write_gray_score_tables(ns: dict, score_rows: list[dict]) -> tuple[dict, list[dict]]:
    score_fieldnames = [
        "ping", "candidate", "score", "ice_retention", "center_residual", "lobe_residual",
        "weak_annular_remaining", "finite_ratio", "peak_quality", "peak_score",
        "peak_prominence", "peak_contrast", "peak_z_shift_m", "initial_ice_z_m",
        "selected_ice_z_m", "weak_floor", "post_floor", "center_min_level",
        "center_attenuation", "lobe_attenuation",
    ]
    ns["write_csv"](ns["OUTPUT_DIR"] / "parameter_score_by_ping.csv", score_rows, score_fieldnames)

    global_rows: list[dict] = []
    for config in ns["CANDIDATE_CONFIGS"]:
        rows = [row for row in score_rows if row["candidate"] == config["name"]]
        scores = numeric_array(rows, "score")
        ice = numeric_array(rows, "ice_retention")
        center = numeric_array(rows, "center_residual")
        lobe = numeric_array(rows, "lobe_residual")
        weak = numeric_array(rows, "weak_annular_remaining")
        peak_quality = numeric_array(rows, "peak_quality")
        z_shift = numeric_array(rows, "peak_z_shift_m")
        fail_mask = (ice < 0.85) | (scores < 0.45) | (peak_quality < 0.15) | (z_shift > 1.25)
        fail_rate = float(np.mean(fail_mask)) if fail_mask.size else 1.0
        robust_score = float(
            np.nanmedian(scores)
            - 0.15 * np.nanstd(scores)
            - 0.25 * fail_rate
            - 0.20 * max(0.0, 0.85 - np.nanmin(ice))
        )
        global_rows.append({
            "candidate": config["name"],
            "robust_global_score": robust_score,
            "median_score": float(np.nanmedian(scores)),
            "std_score": float(np.nanstd(scores)),
            "fail_rate": fail_rate,
            "min_ice_retention": float(np.nanmin(ice)),
            "median_ice_retention": float(np.nanmedian(ice)),
            "median_center_residual": float(np.nanmedian(center)),
            "median_lobe_residual": float(np.nanmedian(lobe)),
            "median_weak_annular_remaining": float(np.nanmedian(weak)),
            "median_peak_quality": float(np.nanmedian(peak_quality)),
            "median_peak_z_shift_m": float(np.nanmedian(z_shift)),
            "weak_floor": config["weak_floor"],
            "post_floor": config["post_floor"],
            "center_min_level": config["center_min_level"],
            "center_attenuation": config["center_attenuation"],
            "lobe_attenuation": config["lobe_attenuation"],
        })

    global_rows = sorted(global_rows, key=lambda row: row["robust_global_score"], reverse=True)
    selected_row = global_rows[0]
    selected_config = next(config for config in ns["CANDIDATE_CONFIGS"] if config["name"] == selected_row["candidate"])
    ns["write_csv"](
        ns["OUTPUT_DIR"] / "global_parameter_score_summary.csv",
        global_rows,
        [
            "candidate", "robust_global_score", "median_score", "std_score", "fail_rate",
            "min_ice_retention", "median_ice_retention", "median_center_residual",
            "median_lobe_residual", "median_weak_annular_remaining", "median_peak_quality",
            "median_peak_z_shift_m", "weak_floor", "post_floor", "center_min_level",
            "center_attenuation", "lobe_attenuation",
        ],
    )
    (ns["OUTPUT_DIR"] / "selected_global_config.json").write_text(
        json.dumps({"selected": selected_row, "config": selected_config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selected_config, global_rows


def score_gray_configs(ns: dict, ds, batch_pings: list[int]) -> tuple[dict, list[dict]]:
    score_rows: list[dict] = []
    print("Gray first pass: scoring denoising configs without caching images...")
    for pos, ping_idx in enumerate(batch_pings, start=1):
        frame = ns["map_ping"](ds, ping_idx, ns["IMAGE_VAR"], ns["GRID_CONFIG"])
        prepared = ns["prepare_noise_suppression"](
            frame["linear"], frame["valid_mask"], frame["y_axis"], frame["z_axis"]
        )
        best_name = None
        best_score = -np.inf
        for config in ns["CANDIDATE_CONFIGS"]:
            suppression = ns["run_noise_suppression"](
                prepared, frame["valid_mask"], frame["y_axis"], frame["z_axis"], config
            )
            metrics = ns["score_result"](
                suppression["cleaned"], prepared["adaptive_image"], prepared["context"], frame["z_axis"]
            )
            if metrics["score"] > best_score:
                best_score = float(metrics["score"])
                best_name = config["name"]
            score_rows.append({
                "ping": int(ping_idx),
                "candidate": config["name"],
                "score": metrics["score"],
                "ice_retention": metrics["ice_retention"],
                "center_residual": metrics["center_residual"],
                "lobe_residual": metrics["lobe_residual"],
                "weak_annular_remaining": metrics["weak_annular_remaining"],
                "finite_ratio": metrics["finite_ratio"],
                "peak_quality": metrics["peak_quality"],
                "peak_score": metrics["peak_score"],
                "peak_prominence": metrics["peak_prominence"],
                "peak_contrast": metrics["peak_contrast"],
                "peak_z_shift_m": metrics["peak_z_shift_m"],
                "initial_ice_z_m": prepared["context"]["initial_ice_z"],
                "selected_ice_z_m": metrics["selected_ice_z_m"],
                "weak_floor": config["weak_floor"],
                "post_floor": config["post_floor"],
                "center_min_level": config["center_min_level"],
                "center_attenuation": config["center_attenuation"],
                "lobe_attenuation": config["lobe_attenuation"],
            })
        if pos == 1 or pos == len(batch_pings) or pos % 40 == 0:
            print(f"[gray-score {pos:04d}/{len(batch_pings):04d}] ping {ping_idx}: local={best_name}, score={best_score:.3f}")
        del frame, prepared
        if pos % 40 == 0:
            gc.collect()
    selected_config, global_rows = write_gray_score_tables(ns, score_rows)
    print(f"Selected gray config: {selected_config['name']}")
    return selected_config, global_rows


def build_gray_record(ns: dict, ds, ping_idx: int, selected_config: dict, manual_store: dict | None) -> tuple[dict, dict, dict]:
    frame = ns["map_ping"](ds, ping_idx, ns["IMAGE_VAR"], ns["GRID_CONFIG"])
    prepared = ns["prepare_noise_suppression"](
        frame["linear"], frame["valid_mask"], frame["y_axis"], frame["z_axis"]
    )
    suppression = ns["run_noise_suppression"](
        prepared, frame["valid_mask"], frame["y_axis"], frame["z_axis"], selected_config
    )
    metrics = ns["score_result"](
        suppression["cleaned"], prepared["adaptive_image"], prepared["context"], frame["z_axis"]
    )
    match = {
        "config": selected_config,
        "cleaned": suppression["cleaned"],
        "masks": suppression["masks"],
        "metrics": metrics,
    }
    record = {
        "ping": int(ping_idx),
        "frame": frame,
        "context": prepared["context"],
        "candidate_results": [match],
        "gray_cleaned_auto": np.asarray(suppression["cleaned"], dtype=float).copy(),
        "gray_cleaned": np.asarray(suppression["cleaned"], dtype=float).copy(),
    }
    gray_row = {
        "ping": int(ping_idx),
        "selected_candidate": selected_config["name"],
        "output_file": "",
        "score": metrics["score"],
        "ice_retention": metrics["ice_retention"],
        "center_residual": metrics["center_residual"],
        "lobe_residual": metrics["lobe_residual"],
        "weak_annular_remaining": metrics["weak_annular_remaining"],
        "peak_quality": metrics["peak_quality"],
        "initial_ice_z_m": prepared["context"]["initial_ice_z"],
        "selected_ice_z_m": metrics["selected_ice_z_m"],
        "manual_review_flag": int(
            (metrics["ice_retention"] < 0.85)
            or (metrics["score"] < 0.45)
            or (metrics["peak_quality"] < 0.15)
            or (metrics["peak_z_shift_m"] > 1.25)
        ),
    }
    if manual_store is None:
        manual_meta = {
            "ping": int(ping_idx),
            "manual_edited": 0,
            "manual_remove_polygon_count": 0,
            "manual_removed_pixel_count": 0,
            "manual_removed_area_m2": 0.0,
            "manual_removed_energy": 0.0,
            "manual_removed_energy_fraction": 0.0,
            "manual_mask_json": str(ns["MANUAL_MASK_JSON"]),
        }
        record["manual_mask_meta"] = manual_meta
        record["manual_remove_mask"] = np.zeros_like(record["gray_cleaned"], dtype=bool)
    else:
        manual_meta = ns["apply_manual_masks_to_record"](record, manual_store)
    return record, gray_row, manual_meta


def write_binary_score_tables(ns: dict, binary_score_rows: list[dict]) -> tuple[dict, list[dict]]:
    binary_score_fieldnames = [
        "ping", "candidate", "binary_score", "ice_energy_capture", "broader_ice_capture",
        "left_support_score", "left_support_needed", "left_signal_fraction",
        "left_ice_energy_capture", "left_y_span_m", "left_gray_y_span_m", "left_edge_gap_m",
        "left_max_gap_m", "left_continuity_score", "left_rescue_applied", "left_rescue_pixel_count",
        "left_support_before", "left_support_after", "binary_center_residual", "binary_lobe_residual",
        "y_span_m", "max_gap_m", "continuity_score", "curve_roughness_m", "z_alignment_m",
        "area_fraction", "mask_pixels", "threshold_high", "threshold_low", "left_rescue_threshold_high",
        "left_rescue_threshold_low", "ice_z_m", "high_q", "low_q", "window_down_m", "window_up_m",
        "min_area_m2", "min_y_span_m", "closing_y_m", "closing_z_m", "manual_edited",
        "manual_remove_polygon_count", "manual_removed_area_m2", "manual_removed_energy_fraction",
    ]
    ns["write_csv"](ns["BINARY_OUTPUT_DIR"] / "binary_parameter_score_by_ping.csv", binary_score_rows, binary_score_fieldnames)

    global_rows: list[dict] = []
    for config in ns["BINARY_CANDIDATE_CONFIGS"]:
        rows = [row for row in binary_score_rows if row["candidate"] == config["name"]]
        scores = numeric_array(rows, "binary_score")
        capture = numeric_array(rows, "ice_energy_capture")
        left_support = numeric_array(rows, "left_support_score")
        left_capture = numeric_array(rows, "left_ice_energy_capture")
        left_needed = numeric_array(rows, "left_support_needed")
        center = numeric_array(rows, "binary_center_residual")
        lobe = numeric_array(rows, "binary_lobe_residual")
        span = numeric_array(rows, "y_span_m")
        align = numeric_array(rows, "z_alignment_m")
        continuity = numeric_array(rows, "continuity_score")
        fail_mask = (
            (capture < 0.45)
            | ((left_needed > 0) & (left_support < 0.40))
            | (span < 2.0)
            | (align > 0.85)
            | (scores < 0.25)
        )
        fail_rate = float(np.mean(fail_mask)) if fail_mask.size else 1.0
        robust_binary_score = float(
            np.nanmedian(scores)
            + 0.08 * np.nanmedian(capture)
            + 0.04 * np.nanmedian(left_support)
            - 0.12 * np.nanstd(scores)
            - 0.25 * fail_rate
            - 0.20 * max(0.0, 0.45 - np.nanmin(capture))
            - 0.10 * max(0.0, 0.50 - np.nanmedian(left_support))
        )
        global_rows.append({
            "candidate": config["name"],
            "robust_binary_score": robust_binary_score,
            "median_binary_score": float(np.nanmedian(scores)),
            "std_binary_score": float(np.nanstd(scores)),
            "fail_rate": fail_rate,
            "min_ice_energy_capture": float(np.nanmin(capture)),
            "median_ice_energy_capture": float(np.nanmedian(capture)),
            "median_left_support_score": float(np.nanmedian(left_support)),
            "median_left_ice_energy_capture": float(np.nanmedian(left_capture)),
            "left_support_needed_rate": float(np.nanmean(left_needed)),
            "median_center_residual": float(np.nanmedian(center)),
            "median_lobe_residual": float(np.nanmedian(lobe)),
            "median_y_span_m": float(np.nanmedian(span)),
            "median_continuity_score": float(np.nanmedian(continuity)),
            "median_z_alignment_m": float(np.nanmedian(align)),
            "high_q": config["high_q"],
            "low_q": config["low_q"],
            "window_down_m": config["window_down_m"],
            "window_up_m": config["window_up_m"],
            "min_area_m2": config["min_area_m2"],
            "min_y_span_m": config["min_y_span_m"],
            "closing_y_m": config["closing_y_m"],
            "closing_z_m": config["closing_z_m"],
        })

    global_rows = sorted(global_rows, key=lambda row: row["robust_binary_score"], reverse=True)
    selected_row = global_rows[0]
    selected_config = next(config for config in ns["BINARY_CANDIDATE_CONFIGS"] if config["name"] == selected_row["candidate"])
    ns["write_csv"](
        ns["BINARY_OUTPUT_DIR"] / "binary_global_parameter_score_summary.csv",
        global_rows,
        [
            "candidate", "robust_binary_score", "median_binary_score", "std_binary_score", "fail_rate",
            "min_ice_energy_capture", "median_ice_energy_capture", "median_left_support_score",
            "median_left_ice_energy_capture", "left_support_needed_rate", "median_center_residual",
            "median_lobe_residual", "median_y_span_m", "median_continuity_score", "median_z_alignment_m",
            "high_q", "low_q", "window_down_m", "window_up_m", "min_area_m2", "min_y_span_m",
            "closing_y_m", "closing_z_m",
        ],
    )
    (ns["BINARY_OUTPUT_DIR"] / "binary_selected_global_config.json").write_text(
        json.dumps({"selected": selected_row, "config": selected_config, "tuning_pings": ns["BINARY_TUNING_PINGS"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selected_config, global_rows


def score_binary_candidates_for_record(ns: dict, ping_idx: int, record: dict, manual_meta: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    gray_match = record["candidate_results"][0]
    cleaned = np.asarray(record.get("gray_cleaned", gray_match["cleaned"]), dtype=float)
    frame = record["frame"]
    ice_z_m = float(gray_match["metrics"]["selected_ice_z_m"])
    best = {
        "ping": int(ping_idx),
        "name": None,
        "score": -np.inf,
        "capture": np.nan,
        "left": np.nan,
        "span": np.nan,
    }
    for config in ns["BINARY_CANDIDATE_CONFIGS"]:
        mask, meta = ns["binary_segment_ice"](cleaned, frame["valid_mask"], frame["y_axis"], frame["z_axis"], ice_z_m, config)
        metrics = ns["binary_score_mask"](mask, cleaned, record["context"], frame["y_axis"], frame["z_axis"], ice_z_m)
        metrics["ice_z_m"] = ice_z_m
        if metrics["binary_score"] > best["score"]:
            best = {
                "ping": int(ping_idx),
                "name": config["name"],
                "score": float(metrics["binary_score"]),
                "capture": float(metrics["ice_energy_capture"]),
                "left": float(metrics["left_support_score"]),
                "span": float(metrics["y_span_m"]),
            }
        rows.append({
            "ping": int(ping_idx),
            "candidate": config["name"],
            "binary_score": metrics["binary_score"],
            "ice_energy_capture": metrics["ice_energy_capture"],
            "broader_ice_capture": metrics["broader_ice_capture"],
            "left_support_score": metrics["left_support_score"],
            "left_support_needed": metrics["left_support_needed"],
            "left_signal_fraction": metrics["left_signal_fraction"],
            "left_ice_energy_capture": metrics["left_ice_energy_capture"],
            "left_y_span_m": metrics["left_y_span_m"],
            "left_gray_y_span_m": metrics["left_gray_y_span_m"],
            "left_edge_gap_m": metrics["left_edge_gap_m"],
            "left_max_gap_m": metrics["left_max_gap_m"],
            "left_continuity_score": metrics["left_continuity_score"],
            "left_rescue_applied": int(meta.get("left_rescue_applied", 0)),
            "left_rescue_pixel_count": int(meta.get("left_rescue_pixel_count", 0)),
            "left_support_before": meta.get("left_support_before", np.nan),
            "left_support_after": meta.get("left_support_after", np.nan),
            "binary_center_residual": metrics["binary_center_residual"],
            "binary_lobe_residual": metrics["binary_lobe_residual"],
            "y_span_m": metrics["y_span_m"],
            "max_gap_m": metrics["max_gap_m"],
            "continuity_score": metrics["continuity_score"],
            "curve_roughness_m": metrics["curve_roughness_m"],
            "z_alignment_m": metrics["z_alignment_m"],
            "area_fraction": metrics["area_fraction"],
            "mask_pixels": metrics["mask_pixels"],
            "threshold_high": meta["threshold_high"],
            "threshold_low": meta["threshold_low"],
            "left_rescue_threshold_high": meta.get("left_rescue_threshold_high", np.nan),
            "left_rescue_threshold_low": meta.get("left_rescue_threshold_low", np.nan),
            "ice_z_m": ice_z_m,
            "high_q": config["high_q"],
            "low_q": config["low_q"],
            "window_down_m": config["window_down_m"],
            "window_up_m": config["window_up_m"],
            "min_area_m2": config["min_area_m2"],
            "min_y_span_m": config["min_y_span_m"],
            "closing_y_m": config["closing_y_m"],
            "closing_z_m": config["closing_z_m"],
            "manual_edited": int(manual_meta.get("manual_edited", 0)),
            "manual_remove_polygon_count": int(manual_meta.get("manual_remove_polygon_count", 0)),
            "manual_removed_area_m2": float(manual_meta.get("manual_removed_area_m2", 0.0)),
            "manual_removed_energy_fraction": float(manual_meta.get("manual_removed_energy_fraction", 0.0)),
        })
    return rows, best


def init_binary_tuning_worker(payload: dict) -> None:
    global _BINARY_TUNING_WORKER_NS
    global _BINARY_TUNING_WORKER_DS
    global _BINARY_TUNING_WORKER_SELECTED_CONFIG
    global _BINARY_TUNING_WORKER_MANUAL_STORE

    os.chdir(NOTEBOOK_DIR)
    ns: dict = {"__name__": "__binary_tuning_worker__"}
    execute_notebook_cell(ns, 2)
    ns["NC_FILE"] = Path(payload["nc_file"])
    ns["OUTPUT_ROOT"] = Path(payload["output_root"])
    ns["OUTPUT_DIR"] = Path(payload["output_dir"])
    ns["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    ns["BATCH_PINGS"] = [int(ping) for ping in payload["batch_pings"]]
    ns["BINARY_TUNING_PING_STEP"] = int(payload["binary_tuning_ping_step"])
    ns["SAVE_BINARY_SELECTED_FIGURES"] = False
    execute_notebook_cell(ns, 4)
    execute_notebook_cell(ns, 6)
    ns["MANUAL_MASK_JSON"] = Path(payload["manual_mask_json"])
    install_gray_denoising(ns)
    execute_notebook_cell(ns, 14)
    install_metadata_range_mapping(
        ns,
        mode=str(payload["grid_range_mode"]),
        range_max_cap_m=payload["grid_range_max_cap_m"],
        range_round_m=float(payload["grid_range_round_m"]),
    )
    execute_notebook_cell(ns, 21)
    install_binary_band_refinement(ns)
    install_publication_binary_renderer(ns)
    _BINARY_TUNING_WORKER_NS = ns
    _BINARY_TUNING_WORKER_DS = ns["ds"]
    _BINARY_TUNING_WORKER_SELECTED_CONFIG = dict(payload["selected_config"])
    _BINARY_TUNING_WORKER_MANUAL_STORE = ns["load_manual_mask_store"](ns["MANUAL_MASK_JSON"])


def score_binary_tuning_ping_worker(ping_idx: int) -> dict:
    if _BINARY_TUNING_WORKER_NS is None or _BINARY_TUNING_WORKER_DS is None:
        raise RuntimeError("Binary tuning worker was not initialized.")
    record, _, manual_meta = build_gray_record(
        _BINARY_TUNING_WORKER_NS,
        _BINARY_TUNING_WORKER_DS,
        int(ping_idx),
        _BINARY_TUNING_WORKER_SELECTED_CONFIG,
        _BINARY_TUNING_WORKER_MANUAL_STORE,
    )
    rows, best = score_binary_candidates_for_record(_BINARY_TUNING_WORKER_NS, int(ping_idx), record, manual_meta)
    del record
    gc.collect()
    return {"ping": int(ping_idx), "rows": rows, "best": best}


def tune_binary_config(
    ns: dict,
    ds,
    selected_config: dict,
    manual_store: dict | None,
    worker_count: int = 1,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    tuning_pings = list(ns["BINARY_TUNING_PINGS"])
    print("Binary tuning pass: scoring candidate masks on representative pings...")
    print(f"- tuning ping count: {len(tuning_pings)} / {len(ns['BATCH_PINGS'])}")
    print(f"- candidate config count: {len(ns['BINARY_CANDIDATE_CONFIGS'])}")
    worker_count = max(1, min(int(worker_count), len(tuning_pings) if tuning_pings else 1))
    if worker_count > 1 and len(tuning_pings) > 1:
        print(f"- binary tuning workers: {worker_count}")
        payload = {
            "nc_file": str(ns["NC_FILE"]),
            "output_root": str(ns["OUTPUT_ROOT"]),
            "output_dir": str(ns["OUTPUT_DIR"]),
            "manual_mask_json": str(ns["MANUAL_MASK_JSON"]),
            "batch_pings": [int(ping) for ping in ns["BATCH_PINGS"]],
            "binary_tuning_ping_step": int(ns["BINARY_TUNING_PING_STEP"]),
            "grid_range_mode": ns["GRID_RANGE_MODE"],
            "grid_range_max_cap_m": ns["GRID_RANGE_MAX_CAP_M"],
            "grid_range_round_m": ns["GRID_RANGE_ROUND_M"],
            "selected_config": selected_config,
        }
        completed = 0
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_binary_tuning_worker,
            initargs=(payload,),
        ) as executor:
            futures = {executor.submit(score_binary_tuning_ping_worker, int(ping_idx)): int(ping_idx) for ping_idx in tuning_pings}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                rows.extend(result["rows"])
                best = result["best"]
                print(
                    f"[binary-tune {completed:03d}/{len(tuning_pings):03d}] ping {best['ping']}: "
                    f"local={best['name']}, score={best['score']:.3f}, capture={best['capture']:.3f}, "
                    f"left={best['left']:.3f}, span={best['span']:.2f} m"
                )
    else:
        for pos, ping_idx in enumerate(tuning_pings, start=1):
            record, _, manual_meta = build_gray_record(ns, ds, int(ping_idx), selected_config, manual_store)
            ping_rows, best = score_binary_candidates_for_record(ns, int(ping_idx), record, manual_meta)
            rows.extend(ping_rows)
            print(
                f"[binary-tune {pos:03d}/{len(tuning_pings):03d}] ping {ping_idx}: "
                f"local={best['name']}, score={best['score']:.3f}, capture={best['capture']:.3f}, "
                f"left={best['left']:.3f}, span={best['span']:.2f} m"
            )
            del record
            gc.collect()
    rows = sorted(rows, key=lambda row: (int(row["ping"]), str(row["candidate"])))
    selected_config_binary, global_rows = write_binary_score_tables(ns, rows)
    print(f"Selected binary config: {selected_config_binary['name']}")
    return selected_config_binary, global_rows








def process_selected_pings(
    ns: dict,
    ds,
    selected_config: dict,
    selected_binary_config: dict,
    manual_store: dict | None,
    save_process_figures: bool,
    save_gray_figures: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    gray_rows: list[dict] = []
    manual_rows: list[dict] = []
    binary_rows: list[dict] = []
    curve_records: list[dict] = []
    print("Final pass: applying selected configs and saving per-frame process figures...")
    for pos, ping_idx in enumerate(ns["BATCH_PINGS"], start=1):
        record, gray_row, manual_meta = build_gray_record(ns, ds, int(ping_idx), selected_config, manual_store)
        gray_match = record["candidate_results"][0]
        frame = record["frame"]
        if save_gray_figures:
            gray_output_path = ns["OUTPUT_DIR"] / f"ping_{int(ping_idx):04d}.png"
            ns["render_final_figure"](
                int(ping_idx),
                frame["linear"],
                gray_match["cleaned"],
                gray_match["masks"],
                gray_match["metrics"],
                selected_config,
                frame["y_axis"],
                frame["z_axis"],
                gray_output_path,
            )
            gray_row["output_file"] = gray_output_path.name
        gray_rows.append(gray_row)
        manual_rows.append(manual_meta)
        cleaned = np.asarray(record.get("gray_cleaned", gray_match["cleaned"]), dtype=float)
        ice_z_m = float(gray_match["metrics"]["selected_ice_z_m"])
        mask, meta = ns["binary_segment_ice"](
            cleaned, frame["valid_mask"], frame["y_axis"], frame["z_axis"], ice_z_m, selected_binary_config
        )
        metrics = ns["binary_score_mask"](mask, cleaned, record["context"], frame["y_axis"], frame["z_axis"], ice_z_m)
        metrics["ice_z_m"] = ice_z_m
        match = {"config": selected_binary_config, "mask": mask, "meta": meta, "metrics": metrics}
        output_name = ""
        if save_process_figures:
            output_path = ns["BINARY_OUTPUT_DIR"] / f"binary_ping_{int(ping_idx):04d}.png"
            ns["binary_render_figure"](
                int(ping_idx), cleaned, mask, meta, metrics, selected_binary_config,
                frame["y_axis"], frame["z_axis"], output_path
            )
            output_name = output_path.name
        binary_rows.append(selected_binary_row(record, selected_binary_config, match, output_name))
        curve_records.append(build_curve_record(record, frame, metrics))
        if pos == 1 or pos == len(ns["BATCH_PINGS"]) or pos % 40 == 0:
            print(
                f"[final {pos:04d}/{len(ns['BATCH_PINGS']):04d}] ping {ping_idx}: "
                f"score={metrics['binary_score']:.3f}, capture={metrics['ice_energy_capture']:.3f}, "
                f"span={metrics['y_span_m']:.2f} m"
            )
        del record, mask, metrics, match, cleaned
        if pos % 20 == 0:
            gc.collect()

    ns["write_csv"](
        ns["OUTPUT_DIR"] / "selected_ping_summary.csv",
        gray_rows,
        [
            "ping", "selected_candidate", "output_file", "score", "ice_retention",
            "center_residual", "lobe_residual", "weak_annular_remaining", "peak_quality",
            "initial_ice_z_m", "selected_ice_z_m", "manual_review_flag",
        ],
    )
    ns["write_csv"](
        ns["MANUAL_MASK_SUMMARY_CSV"],
        manual_rows,
        [
            "ping", "manual_edited", "manual_remove_polygon_count", "manual_removed_pixel_count",
            "manual_removed_area_m2", "manual_removed_energy", "manual_removed_energy_fraction",
            "manual_mask_json",
        ],
    )
    ns["write_csv"](
        ns["BINARY_OUTPUT_DIR"] / "binary_selected_ping_summary.csv",
        binary_rows,
        [
            "ping", "selected_binary_candidate", "output_file", "binary_score",
            "ice_energy_capture", "broader_ice_capture", "left_support_score",
            "left_support_needed", "left_signal_fraction", "left_ice_energy_capture",
            "left_y_span_m", "left_gray_y_span_m", "left_edge_gap_m", "left_max_gap_m",
            "left_continuity_score", "left_rescue_applied", "left_rescue_pixel_count",
            "binary_center_residual", "binary_lobe_residual", "y_span_m", "max_gap_m",
            "continuity_score", "curve_roughness_m", "z_alignment_m", "area_fraction",
            "mask_pixels", "ice_z_m", "manual_edited", "manual_remove_polygon_count",
            "manual_removed_area_m2", "manual_removed_energy_fraction", "manual_review_flag",
        ],
    )
    return gray_rows, manual_rows, binary_rows, curve_records


def process_gray_only_pings(ns: dict, ds, selected_config: dict, save_gray_figures: bool) -> list[dict]:
    gray_rows: list[dict] = []
    print("Gray-only pass: applying denoising config and saving noise-suppression figures...")
    for pos, ping_idx in enumerate(ns["BATCH_PINGS"], start=1):
        record, gray_row, _ = build_gray_record(ns, ds, int(ping_idx), selected_config, manual_store=None)
        gray_match = record["candidate_results"][0]
        frame = record["frame"]
        if save_gray_figures:
            gray_output_path = ns["OUTPUT_DIR"] / f"ping_{int(ping_idx):04d}.png"
            ns["render_final_figure"](
                int(ping_idx),
                frame["linear"],
                gray_match["cleaned"],
                gray_match["masks"],
                gray_match["metrics"],
                selected_config,
                frame["y_axis"],
                frame["z_axis"],
                gray_output_path,
            )
            gray_row["output_file"] = gray_output_path.name
        gray_rows.append(gray_row)
        if pos == 1 or pos == len(ns["BATCH_PINGS"]) or pos % 40 == 0:
            print(
                f"[gray-only {pos:04d}/{len(ns['BATCH_PINGS']):04d}] ping {ping_idx}: "
                f"score={gray_match['metrics']['score']:.3f}, "
                f"ice z={gray_match['metrics']['selected_ice_z_m']:.2f} m"
            )
        del record
        if pos % 20 == 0:
            gc.collect()

    ns["write_csv"](
        ns["OUTPUT_DIR"] / "selected_ping_summary.csv",
        gray_rows,
        [
            "ping", "selected_candidate", "output_file", "score", "ice_retention",
            "center_residual", "lobe_residual", "weak_annular_remaining", "peak_quality",
            "initial_ice_z_m", "selected_ice_z_m", "manual_review_flag",
        ],
    )
    (ns["OUTPUT_DIR"] / "selected_global_config.json").write_text(
        json.dumps({"selected_config": selected_config, "mode": "gray_only"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gray_rows




































def main() -> None:
    args = parse_args()
    nc_file = Path(args.nc_file).expanduser().resolve()
    motion_file = Path(args.motion_file).expanduser().resolve()
    if not nc_file.is_file():
        raise FileNotFoundError(nc_file)
    if not motion_file.is_file():
        raise FileNotFoundError(motion_file)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(NOTEBOOK_DIR)
    ns: dict = {
        "__name__": "__imaging_sonar_ice_draft_pipeline__",
        "NC_FILE": nc_file,
        "MOTION_FILE": motion_file,
        "TIMEZONE": str(args.timezone),
        "OUTPUT_ROOT": output_root,
        "OUTPUT_DIR": output_root / "01_gray_preprocessing",
    }
    try:
        execute_notebook_cell(ns, 2)
        ns["NC_FILE"] = nc_file
        ns["MOTION_FILE"] = motion_file
        ns["TIMEZONE"] = str(args.timezone)
        ns["OUTPUT_ROOT"] = output_root
        ns["OUTPUT_DIR"] = output_root / "01_gray_preprocessing"
        ns["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
        ns["BINARY_TUNING_PING_STEP"] = int(args.binary_tuning_ping_step)
        ns["SAVE_BINARY_SELECTED_FIGURES"] = not args.no_process_figures
        execute_notebook_cell(ns, 4)
        ds = ns["ds"]
        start_ping = (
            int(args.start_ping)
            if args.start_ping is not None
            else find_start_ping(ds, args.start_local, timezone=args.timezone)
        )
        end_ping_exclusive = resolve_end_ping_exclusive(
            ds,
            start_ping,
            args.end_ping,
            args.end_local,
            timezone=args.timezone,
        )
        sampled_pings_before_exclusion = list(range(start_ping, end_ping_exclusive, int(args.ping_step)))
        excluded_ping_ranges = parse_ping_ranges(args.exclude_ping_ranges)
        excluded_sampled_pings = write_excluded_ping_log(
            output_root,
            sampled_pings_before_exclusion,
            excluded_ping_ranges,
        )
        batch_pings = [
            ping_idx
            for ping_idx in sampled_pings_before_exclusion
            if not ping_is_excluded(ping_idx, excluded_ping_ranges)
        ]
        if args.max_count is not None:
            batch_pings = batch_pings[: int(args.max_count)]
        if not batch_pings:
            raise ValueError("No sampled pings selected.")
        ns["BATCH_PINGS"] = batch_pings
        print("Imaging-sonar sea-ice draft reconstruction configuration:")
        print(f"- NetCDF: {ns['NC_FILE']}")
        print(f"- start ping: {start_ping}")
        print(f"- end ping exclusive: {end_ping_exclusive}")
        print(f"- first/last sampled ping: {batch_pings[0]} / {batch_pings[-1]}")
        print(f"- ping step: {args.ping_step}")
        if excluded_ping_ranges:
            print(f"- excluded ping ranges: {excluded_ping_ranges}")
            print(f"- excluded sampled ping count: {len(excluded_sampled_pings)}")
        print(f"- sampled ping count: {len(batch_pings)}")
        print(f"- binary tuning ping step: {args.binary_tuning_ping_step}")
        print(f"- binary tuning workers: {args.binary_tuning_workers}")
        print(f"- force retune: {args.force_retune}")
        print(f"- output root: {output_root}")
        print(f"- process figures: {not args.no_process_figures}")
        print(f"- gray noise-suppression figures: {not args.no_gray_figures}")
        execute_notebook_cell(ns, 6)
        install_gray_denoising(ns)
        print("- gray denoising: range-adaptive structured-noise suppression")
        execute_notebook_cell(ns, 14)
        install_metadata_range_mapping(
            ns,
            mode=args.grid_range_mode,
            range_max_cap_m=args.grid_range_max,
            range_round_m=float(args.grid_range_round_m),
        )
        print(f"- grid range mode: {ns['GRID_RANGE_MODE']}")
        print(f"- grid range max cap: {ns['GRID_RANGE_MAX_CAP_M']}")
        print(f"- grid range round: {ns['GRID_RANGE_ROUND_M']}")
        if args.gray_only:
            selected_gray_config = dict(ns["CANDIDATE_CONFIGS"][0])
            gray_rows = process_gray_only_pings(
                ns,
                ds,
                selected_gray_config,
                save_gray_figures=not args.no_gray_figures,
            )
            summary = {
                "nc_file": str(ns["NC_FILE"]),
                "output_root": str(output_root),
                "mode": "gray_only",
                "start_ping": int(start_ping),
                "end_ping_exclusive": int(end_ping_exclusive),
                "first_sampled_ping": int(batch_pings[0]),
                "last_sampled_ping": int(batch_pings[-1]),
                "ping_step": int(args.ping_step),
                "sampled_ping_count_before_exclusion": len(sampled_pings_before_exclusion),
                "excluded_ping_ranges": [[int(start), int(end)] for start, end in excluded_ping_ranges],
                "excluded_sampled_ping_count": len(excluded_sampled_pings),
                "excluded_sampled_pings_preview": excluded_sampled_pings[:20],
                "sampled_ping_count": len(batch_pings),
                "grid_range_mode": ns["GRID_RANGE_MODE"],
                "grid_range_max_cap_m": ns["GRID_RANGE_MAX_CAP_M"],
                "grid_range_round_m": ns["GRID_RANGE_ROUND_M"],
                "selected_gray_config": selected_gray_config["name"],
                "gray_figure_dir": str(ns["OUTPUT_DIR"]),
                "gray_figure_count": int(len(list(ns["OUTPUT_DIR"].glob("ping_*.png")))),
                "gray_row_count": int(len(gray_rows)),
            }
            (output_root / "run_review_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("Gray-only review run completed.")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        if args.manual_mask_json is not None:
            manual_mask_path = Path(args.manual_mask_json).expanduser().resolve()
            if not manual_mask_path.is_file():
                raise FileNotFoundError(manual_mask_path)
            ns["MANUAL_MASK_JSON"] = manual_mask_path
        manual_store = ns["load_manual_mask_store"](ns["MANUAL_MASK_JSON"])
        execute_notebook_cell(ns, 21)
        install_binary_band_refinement(ns)
        install_publication_binary_renderer(ns)

        selected_gray_config = None
        if not args.force_retune:
            selected_gray_config = load_selected_config_cache(
                ns["OUTPUT_DIR"] / "selected_global_config.json",
                ns["CANDIDATE_CONFIGS"],
                "gray",
            )
        if selected_gray_config is None:
            selected_gray_config, gray_global_rows = score_gray_configs(ns, ds, batch_pings)
        else:
            gray_global_rows = []

        selected_binary_config = None
        if not args.force_retune:
            selected_binary_config = load_selected_config_cache(
                ns["BINARY_OUTPUT_DIR"] / "binary_selected_global_config.json",
                ns["BINARY_CANDIDATE_CONFIGS"],
                "binary",
            )
        if selected_binary_config is None:
            selected_binary_config, binary_global_rows = tune_binary_config(
                ns,
                ds,
                selected_gray_config,
                manual_store,
                worker_count=int(args.binary_tuning_workers),
            )
        else:
            binary_global_rows = []
        gray_rows, manual_rows, binary_rows, curve_records = process_selected_pings(
            ns,
            ds,
            selected_gray_config,
            selected_binary_config,
            manual_store,
            save_process_figures=not args.no_process_figures,
            save_gray_figures=not args.no_gray_figures,
        )
        roll_rows, curve_point_rows, selected_roll_sign = build_roll_and_curve_tables(ns, ds, curve_records)
        attitude_figure_rows = render_attitude_curve_figures(ns, curve_point_rows)
        draft_df = build_draft_table(ns, ds, curve_point_rows, selected_roll_sign)
        qc_df = build_qc_table(ns, gray_rows, manual_rows, binary_rows, roll_rows, draft_df)
        candidates_df = classify_no_ice_candidates(ns, gray_rows, binary_rows, qc_df, draft_df)

        summary = {
            "nc_file": str(ns["NC_FILE"]),
            "output_root": str(output_root),
            "start_ping": int(start_ping),
            "end_ping_exclusive": int(end_ping_exclusive),
            "first_sampled_ping": int(batch_pings[0]),
            "last_sampled_ping": int(batch_pings[-1]),
            "ping_step": int(args.ping_step),
            "sampled_ping_count_before_exclusion": len(sampled_pings_before_exclusion),
            "excluded_ping_ranges": [[int(start), int(end)] for start, end in excluded_ping_ranges],
            "excluded_sampled_ping_count": len(excluded_sampled_pings),
            "excluded_sampled_pings_preview": excluded_sampled_pings[:20],
            "sampled_ping_count": len(batch_pings),
            "grid_range_mode": ns["GRID_RANGE_MODE"],
            "grid_range_max_cap_m": ns["GRID_RANGE_MAX_CAP_M"],
            "grid_range_round_m": ns["GRID_RANGE_ROUND_M"],
            "binary_tuning_ping_step": int(args.binary_tuning_ping_step),
            "binary_tuning_workers": int(args.binary_tuning_workers),
            "force_retune": bool(args.force_retune),
            "binary_tuning_ping_count": len(ns["BINARY_TUNING_PINGS"]),
            "selected_gray_config": selected_gray_config["name"],
            "selected_binary_config": selected_binary_config["name"],
            "selected_roll_sign": int(selected_roll_sign),
            "attitude_figure_dir": str(ns["OUTPUT_ROOT"] / "04_attitude_corrected_curves"),
            "attitude_figure_count": int(len(attitude_figure_rows)),
            "process_figure_dir": str(ns["BINARY_OUTPUT_DIR"]),
            "process_figure_count": int(len(list(ns["BINARY_OUTPUT_DIR"].glob("binary_ping_*.png")))),
            "gray_figure_dir": str(ns["OUTPUT_DIR"]),
            "gray_figure_count": int(len(list(ns["OUTPUT_DIR"].glob("ping_*.png")))),
            "no_ice_bottom_candidate_count": int(candidates_df["no_ice_bottom_candidate"].sum()),
            "low_confidence_candidate_count": int(candidates_df["low_confidence_candidate"].sum()),
            "draft_row_count": int(len(draft_df)),
        }
        (output_root / "run_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Sea-ice draft reconstruction completed.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if "ds" in ns:
            ns["ds"].close()
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
