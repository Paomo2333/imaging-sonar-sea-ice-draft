"""Gray-stage sonar artifact suppression hooks for the draft pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.signal import find_peaks

from .parameters import parameters_for

DENOISING_MODULE_DIR = Path(__file__).resolve().parent
ANNULAR_TRIGGER_SCORE = 0.24


def install_gray_denoising(ns: dict) -> None:
    """Install the range-adaptive gray-image denoising workflow."""
    if str(DENOISING_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(DENOISING_MODULE_DIR))
    from . import denoise_line_utils as s2

    parameters = parameters_for(ns)
    for key, value in parameters["denoising"].items():
        constant = key.upper()
        if not hasattr(s2, constant):
            raise ValueError(f"Denoising setting has no implementation: {key}")
        setattr(s2, constant, value)
    ns["CANDIDATE_CONFIGS"] = [dict(parameters["gray"], name="initial_gray")]
    ns["GRAY_DENOISING_ENABLED"] = True

    def _axis_resolution(y_axis: np.ndarray, z_axis: np.ndarray) -> float:
        values = []
        for axis in (np.asarray(y_axis, dtype=float), np.asarray(z_axis, dtype=float)):
            if axis.size > 1:
                diffs = np.abs(np.diff(axis))
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    values.append(float(np.nanmedian(diffs)))
        return min(values) if values else 0.08

    def _compatible_cleaned(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        out = np.asarray(image, dtype=float).copy()
        out[(~valid_mask) | (~np.isfinite(out)) | (out <= 0.0)] = np.nan
        return out

    def _candidate_line(image: np.ndarray, valid_mask: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray):
        resolution_m = _axis_resolution(y_axis, z_axis)
        y_grid, z_grid = np.meshgrid(y_axis, z_axis)
        range_max_m = float(np.nanmax(np.hypot(y_grid[valid_mask], z_grid[valid_mask]))) if np.any(valid_mask) else 0.0
        score_map = s2.build_score_map(image, valid_mask, resolution_m)
        labels, components = s2.connected_components_from_score(score_map, y_axis, z_axis, valid_mask, range_max_m)
        selected_labels, selected_component, _ = s2.select_lower_candidate_group(components)
        line_y, line_z, line_status, observed_fraction, center_z = s2.extract_line_from_labels(
            labels,
            selected_labels,
            score_map,
            y_axis,
            z_axis,
        )
        return score_map, selected_labels, selected_component, line_y, line_z, line_status, observed_fraction, center_z

    def _apply_curve_guided_masks(cleaned: np.ndarray, suppress_mask: np.ndarray, valid_mask: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray):
        score_map, selected_labels, selected_component, line_y, line_z, line_status, observed_fraction, center_z = _candidate_line(
            cleaned,
            valid_mask,
            y_axis,
            z_axis,
        )
        out = np.asarray(cleaned, dtype=float).copy()
        combined = np.asarray(suppress_mask, dtype=float).copy()

        for _ in range(parameters["guidance"]["iterations"]):
            below_mask, _ = s2.build_below_ice_non_target_mask(valid_mask, y_axis, z_axis, line_y, line_z, line_status)
            above_mask, _ = s2.build_above_ice_multipath_mask(valid_mask, y_axis, z_axis, line_y, line_z, line_status)
            if np.any(below_mask):
                below_visible = below_mask & (~np.isfinite(combined) | (combined == 1.0))
                below_lobe_like = below_visible & (out >= s2.COMPONENT_THRESHOLD)
                out[below_mask] = 0.0
                combined[below_visible] = 1.0
                combined[below_lobe_like] = 3.0
            if np.any(above_mask):
                above_visible = above_mask & (~np.isfinite(combined) | (combined == 1.0))
                out[above_mask] = 0.0
                combined[above_visible] = 1.0

            center_mask = s2.build_curve_guided_center_artifact_mask(out, valid_mask, y_axis, z_axis, line_y, line_z)
            if np.any(center_mask):
                out[center_mask] = 0.0
                combined[center_mask] = 2.0

            score_map, selected_labels, selected_component, line_y, line_z, line_status, observed_fraction, center_z = _candidate_line(
                out,
                valid_mask,
                y_axis,
                z_axis,
            )

        y_grid, z_grid = np.meshgrid(y_axis, z_axis)
        annular = s2.annular_residual_metrics(out, np.hypot(y_grid, z_grid), y_axis, z_grid, valid_mask, line_y, line_z)
        if annular.get("annular_score", 0.0) >= parameters["guidance"]["annular_trigger_score"] and len(line_y) >= 3:
            out = s2.secondary_annular_suppression(out, np.hypot(y_grid, z_grid), y_axis, z_grid, valid_mask, line_y, line_z)
            score_map, selected_labels, selected_component, line_y, line_z, line_status, observed_fraction, center_z = _candidate_line(
                out,
                valid_mask,
                y_axis,
                z_axis,
            )

        return {
            "cleaned": out,
            "suppress_mask": combined,
            "score_map": score_map,
            "line_y": line_y,
            "line_z": line_z,
            "line_status": line_status,
            "center_z": center_z,
            "observed_fraction": observed_fraction,
            "selected_component": selected_component,
            "selected_label_count": len(selected_labels),
            "annular_score": float(annular.get("annular_score", 0.0)),
        }

    def prepare_noise_suppression(linear_image, valid_mask, y_axis, z_axis):
        y_grid, z_grid = np.meshgrid(y_axis, z_axis)
        r_grid = np.hypot(y_grid, z_grid)
        adaptive_image, adaptive_meta = s2.adaptive_range_threshold(linear_image, r_grid, valid_mask, **parameters["background"])
        context = s2.build_noise_context(adaptive_image, valid_mask, y_axis, z_axis)
        context["initial_projection_score"] = context["projection_score"]
        context["initial_projection_smooth"] = context["projection_smooth"]
        context["r_grid"] = r_grid
        context["valid_mask"] = np.asarray(valid_mask, dtype=bool)
        return {
            "adaptive_image": adaptive_image,
            "adaptive_meta": adaptive_meta,
            "context": context,
        }

    def run_noise_suppression(prepared, valid_mask, y_axis, z_axis, config):
        first = s2.run_unified_noise_suppression(
            prepared["adaptive_image"],
            valid_mask,
            y_axis,
            z_axis,
            config,
        )
        guided = _apply_curve_guided_masks(
            first["cleaned"],
            first["suppress_mask"],
            valid_mask,
            y_axis,
            z_axis,
        )
        prepared["context"].update(first["context"])
        prepared["context"]["guided_line_y"] = guided["line_y"]
        prepared["context"]["guided_line_z"] = guided["line_z"]
        prepared["context"]["guided_annular_score"] = guided["annular_score"]
        return {
            "cleaned": _compatible_cleaned(guided["cleaned"], valid_mask),
            "masks": {"suppress_mask": guided["suppress_mask"]},
            "guided_denoising": guided,
        }

    def _projection_peaks(projection_smooth: np.ndarray, z_axis: np.ndarray, selected_z: float, dz_m: float):
        max_score = float(np.nanmax(projection_smooth)) if np.any(np.isfinite(projection_smooth)) else 0.0
        prominence = max(1e-6, 0.06 * max_score)
        distance = max(1, int(round(0.45 / max(abs(dz_m), 1e-6))))
        rows, props = find_peaks(np.nan_to_num(projection_smooth, nan=0.0), prominence=prominence, distance=distance)
        peaks = []
        for i, row in enumerate(rows):
            peaks.append(
                {
                    "row": int(row),
                    "z_m": float(z_axis[row]),
                    "score": float(projection_smooth[row]),
                    "prominence": float(props["prominences"][i]) if "prominences" in props else 0.0,
                }
            )
        if not peaks and np.any(np.isfinite(projection_smooth)):
            row = int(np.nanargmax(projection_smooth))
            peaks.append({"row": row, "z_m": float(z_axis[row]), "score": float(projection_smooth[row]), "prominence": 0.0})
        peaks = sorted(peaks, key=lambda item: item["prominence"], reverse=True)
        if np.isfinite(selected_z):
            selected_row = int(np.nanargmin(np.abs(z_axis - selected_z)))
            if all(abs(item["z_m"] - selected_z) > max(abs(dz_m), 1e-6) for item in peaks):
                peaks.insert(
                    0,
                    {
                        "row": selected_row,
                        "z_m": float(z_axis[selected_row]),
                        "score": float(projection_smooth[selected_row]),
                        "prominence": 0.0,
                    },
                )
        return peaks

    def score_result(cleaned, adaptive_image, context, z_axis):
        valid_mask = np.asarray(context.get("valid_mask", np.isfinite(adaptive_image)), dtype=bool)
        cleaned_zero = np.nan_to_num(cleaned, nan=0.0)
        dz_m = float(context.get("dz", np.nanmean(np.diff(z_axis)) if len(z_axis) > 1 else 0.08))
        projection_score, projection_smooth = s2.compute_z_projection(cleaned_zero, valid_mask, z_axis, dz_m)
        selected_ice_z = s2.estimate_ice_z_from_projection(projection_smooth, z_axis)
        peaks = _projection_peaks(projection_smooth, z_axis, selected_ice_z, dz_m)
        selected_peak = peaks[0] if peaks else {"score": 0.0, "prominence": 0.0, "z_m": selected_ice_z}

        baseline = context["baseline"]
        masks = context["masks"]
        ice_retention = s2.safe_energy(cleaned_zero, masks["ice_band"]) / baseline["ice_energy"] if baseline["ice_energy"] else np.nan
        center_residual = s2.safe_energy(cleaned_zero, masks["center_nonice"]) / baseline["center_energy"] if baseline["center_energy"] else 0.0
        lobe_residual = s2.safe_energy(cleaned_zero, masks["lower_lobe_region"]) / baseline["lower_energy"] if baseline["lower_energy"] else 0.0
        weak_remaining = (
            int(np.count_nonzero((cleaned_zero > 0) & masks["weak_annular"])) / baseline["weak_pixels"]
            if baseline["weak_pixels"]
            else 0.0
        )
        finite_ratio = (
            int(np.count_nonzero((cleaned_zero > 0) & np.isfinite(adaptive_image))) / baseline["finite_pixels"]
            if baseline["finite_pixels"]
            else np.nan
        )
        peak_score = float(selected_peak.get("score", 0.0))
        peak_prominence = float(selected_peak.get("prominence", 0.0))
        peak_quality = float(np.clip(0.70 * peak_score + 0.30 * min(1.0, 4.0 * peak_prominence), 0.0, 1.0))
        finite_projection = projection_smooth[np.isfinite(projection_smooth)]
        peak_contrast = float(peak_score - np.nanmedian(finite_projection)) if finite_projection.size else 0.0
        peak_z_shift = abs(float(selected_ice_z) - float(context.get("initial_ice_z", selected_ice_z))) if np.isfinite(selected_ice_z) else np.nan
        score = (
            0.58 * (ice_retention if np.isfinite(ice_retention) else 0.0)
            + 0.22 * peak_quality
            - 0.24 * center_residual
            - 0.20 * lobe_residual
            - 0.10 * weak_remaining
            - 0.04 * abs((finite_ratio if np.isfinite(finite_ratio) else 0.55) - 0.55)
        )
        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "ice_retention": float(ice_retention) if np.isfinite(ice_retention) else np.nan,
            "center_residual": float(center_residual),
            "lobe_residual": float(lobe_residual),
            "weak_annular_remaining": float(weak_remaining),
            "finite_ratio": float(finite_ratio) if np.isfinite(finite_ratio) else np.nan,
            "peak_quality": peak_quality,
            "peak_score": peak_score,
            "peak_prominence": peak_prominence,
            "peak_contrast": peak_contrast,
            "peak_z_shift_m": float(peak_z_shift) if np.isfinite(peak_z_shift) else np.nan,
            "selected_ice_z_m": float(selected_ice_z),
            "projection_score": projection_score,
            "projection_smooth": projection_smooth,
            "peaks": peaks,
        }

    def render_final_figure(ping_idx, linear_input, cleaned, masks, metrics, config, y_axis, z_axis, output_path):
        extent = [float(y_axis[0]), float(y_axis[-1]), float(z_axis[0]), float(z_axis[-1])]
        mvp_cmap = ns["CMAP"].copy()
        mvp_cmap.set_bad("#000000")
        linear_display, _, _ = ns["percentile_normalize"](linear_input, ns["CLIP_PERCENTILES"])

        mask_cmap = ListedColormap(["#482475", "#21918c", "#fde725"])
        mask_cmap.set_bad("#ffffff")
        mask_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], mask_cmap.N)
        mask_display = np.asarray(masks["suppress_mask"], dtype=float).copy()
        mask_display[(~np.isfinite(mask_display)) | (mask_display <= 0.0)] = np.nan

        fig_width = 18.0 / 2.54
        fig_height = 11.8 / 2.54
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        gs = fig.add_gridspec(
            2,
            2,
            left=0.075,
            right=0.965,
            bottom=0.12,
            top=0.92,
            wspace=0.30,
            hspace=0.25,
        )
        ax_input = fig.add_subplot(gs[0, 0])
        ax_mask = fig.add_subplot(gs[0, 1])
        ax_output = fig.add_subplot(gs[1, 0])
        ax_projection = fig.add_subplot(gs[1, 1])

        image_axes = [ax_input, ax_mask, ax_output]
        main_axes = [ax_input, ax_mask, ax_output, ax_projection]
        axis_xlabel = "sonar-frame Y (m)"
        axis_ylabel = "sonar-frame Z (m)"
        box_aspect = abs((extent[3] - extent[2]) / max(abs(extent[1] - extent[0]), 1e-6))

        im_input = ax_input.imshow(
            linear_display,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=mvp_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ax_input.set_title("Raw Sonar Image", fontname="Arial", fontsize=8.8, fontweight="bold")

        im_mask = ax_mask.imshow(
            mask_display,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=mask_cmap,
            norm=mask_norm,
        )
        ax_mask.set_title("Noise-Suppression Mask", fontname="Arial", fontsize=8.8, fontweight="bold")
        legend_handles = [
            Patch(facecolor="#482475", edgecolor="none", label="Background Echo"),
            Patch(facecolor="#21918c", edgecolor="none", label="Central-Beam Artifact"),
            Patch(facecolor="#fde725", edgecolor="none", label="Lobe-Shaped Artifact"),
        ]
        ax_mask.legend(handles=legend_handles, loc="lower right", fontsize=5.7, frameon=True)

        im_output = ax_output.imshow(
            cleaned,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=mvp_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ax_output.set_title("Denoised Sonar Image", fontname="Arial", fontsize=8.8, fontweight="bold")

        ax_projection.plot(metrics["projection_score"], z_axis, color="#9aa0a6", linewidth=0.75, label="Raw score")
        ax_projection.plot(metrics["projection_smooth"], z_axis, color="#c98200", linewidth=1.1, label="Smoothed score")
        for peak in metrics["peaks"][:6]:
            ax_projection.scatter(peak["score"], peak["z_m"], s=10, color="#c44e52", zorder=3)
            label_y = peak["z_m"] - 0.22
            ax_projection.text(peak["score"], label_y, f" {peak['z_m']:.2f} m", fontsize=6.6, va="top")
        ax_projection.axhline(
            metrics["selected_ice_z_m"],
            color="#4c72b0",
            linewidth=0.85,
            linestyle="--",
            label="Selected pick",
        )
        ax_projection.set_title("Composite Ice-Bottom Score and Pick", fontname="Arial", fontsize=8.8, fontweight="bold")
        ax_projection.set_xlabel("Normalized ice-bottom score", fontname="Arial", fontsize=7.5)
        ax_projection.set_ylabel("")
        ax_projection.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)
        ax_projection.legend(loc="lower right", fontsize=5.7, frameon=True)

        for ax in image_axes:
            ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)

        ax_input.set_ylabel(axis_ylabel, fontname="Arial", fontsize=7.5)
        ax_output.set_ylabel(axis_ylabel, fontname="Arial", fontsize=7.5)
        ax_output.set_xlabel(axis_xlabel, fontname="Arial", fontsize=7.5)
        ax_mask.set_xlabel(axis_xlabel, fontname="Arial", fontsize=7.5)
        ax_mask.set_ylabel("")
        ax_output.set_title("Denoised Sonar Image", fontname="Arial", fontsize=8.8, fontweight="bold")
        ax_mask.set_title("Noise-Suppression Mask", fontname="Arial", fontsize=8.8, fontweight="bold")
        ax_projection.set_xlabel("Normalized ice-bottom score", fontname="Arial", fontsize=7.5)
        ax_projection.set_ylabel("")
        for ax in main_axes:
            ax.set_box_aspect(box_aspect)
            ax.tick_params(labelsize=6.6, width=0.7, length=2.6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        fig.canvas.draw()
        for ax, image in ((ax_input, im_input), (ax_output, im_output)):
            pos = ax.get_position()
            cax = fig.add_axes([pos.x1 + 0.006, pos.y0, 0.010, pos.height])
            cbar = fig.colorbar(image, cax=cax, ticks=[0.0, 0.5, 1.0])
            cbar.ax.tick_params(labelsize=6.2, width=0.6, length=2.3)
            cbar.outline.set_linewidth(0.6)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    ns["prepare_noise_suppression"] = prepare_noise_suppression
    ns["run_noise_suppression"] = run_noise_suppression
    ns["score_result"] = score_result
    ns["render_final_figure"] = render_final_figure
