"""Ice-bottom line extraction records and binary-line summaries."""

from __future__ import annotations

import numpy as np


def selected_binary_row(record: dict, selected_binary_config: dict, match: dict, output_file: str) -> dict:
    metrics = match["metrics"]
    meta = match["meta"]
    manual_meta = record.get("manual_mask_meta", {})
    return {
        "ping": int(record["ping"]),
        "selected_binary_candidate": selected_binary_config["name"],
        "output_file": output_file,
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
        "binary_center_residual": metrics["binary_center_residual"],
        "binary_lobe_residual": metrics["binary_lobe_residual"],
        "y_span_m": metrics["y_span_m"],
        "max_gap_m": metrics["max_gap_m"],
        "continuity_score": metrics["continuity_score"],
        "curve_roughness_m": metrics["curve_roughness_m"],
        "z_alignment_m": metrics["z_alignment_m"],
        "area_fraction": metrics["area_fraction"],
        "mask_pixels": metrics["mask_pixels"],
        "ice_z_m": metrics["ice_z_m"],
        "manual_edited": int(manual_meta.get("manual_edited", 0)),
        "manual_remove_polygon_count": int(manual_meta.get("manual_remove_polygon_count", 0)),
        "manual_removed_area_m2": float(manual_meta.get("manual_removed_area_m2", 0.0)),
        "manual_removed_energy_fraction": float(manual_meta.get("manual_removed_energy_fraction", 0.0)),
        "manual_review_flag": int(
            (metrics["ice_energy_capture"] < 0.45)
            or (metrics["left_support_needed"] and metrics["left_support_score"] < 0.40)
            or (metrics["y_span_m"] < 2.0)
            or (metrics["z_alignment_m"] > 0.85)
            or (metrics["binary_score"] < 0.25)
        ),
    }


def build_curve_record(record: dict, frame: dict, metrics: dict) -> dict:
    """Build a compact extracted ice-bottom line record for downstream attitude correction."""
    return {
        "ping": int(record["ping"]),
        "y_axis": np.asarray(frame["y_axis"], dtype=float).copy(),
        "curve_z": np.asarray(metrics.get("curve_z_reconstructed", metrics["curve_z"]), dtype=float).copy(),
        "curve_weight": np.asarray(metrics.get("curve_weight_reconstructed", metrics["curve_weight"]), dtype=float).copy(),
        "curve_interpolated_mask": np.asarray(
            metrics.get("curve_interpolated_mask", np.zeros_like(metrics["curve_z"], dtype=bool)),
            dtype=bool,
        ).copy(),
    }
