"""Binary ice-bottom candidate extraction and binary-stage figure hooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from .parameters import parameters_for

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


def install_binary_band_refinement(ns: dict) -> None:
    """Refine binary ice-band extraction after gray-stage denoising."""

    refinement = parameters_for(ns)["binary_refinement"]

    def _effective_config(config: dict) -> dict:
        return dict(config)

    def _left_rescue_meta(final_mask: np.ndarray, cleaned: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray, ice_z_m: float) -> dict:
        meta = ns["binary_empty_left_rescue_meta"](final_mask.shape)
        before = ns["binary_left_support_metrics"](final_mask, cleaned, y_axis, z_axis, ice_z_m)
        meta["left_support_before"] = before["left_support_score"]
        meta["left_support_after"] = before["left_support_score"]
        meta["left_edge_gap_before"] = before["left_edge_gap_m"]
        meta["left_edge_gap_after"] = before["left_edge_gap_m"]
        return meta

    def _extend_one_side(
        final_mask: np.ndarray,
        cleaned: np.ndarray,
        valid_mask: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        ice_z_m: float,
        threshold_low: float,
        side: str,
        window_up_m: float,
    ) -> np.ndarray:
        curve_z, _ = ns["binary_extract_curve"](final_mask, cleaned, y_axis, z_axis)
        active_cols = np.flatnonzero(np.isfinite(curve_z))
        extension = np.zeros_like(final_mask, dtype=bool)
        if active_cols.size == 0:
            return extension

        dz = float(abs(np.nanmean(np.diff(z_axis))))
        half_width_px = ns["meters_to_pixels"](refinement["edge_extension_half_width_m"], dz, minimum=1)
        min_intensity = max(refinement["edge_min_intensity"], min(float(threshold_low), refinement["edge_threshold_scale"] * float(threshold_low)))
        if side == "left":
            start_col = int(active_cols.min())
            col_iter = range(start_col - 1, -1, -1)
        else:
            start_col = int(active_cols.max())
            col_iter = range(start_col + 1, len(y_axis))
        start_y = float(y_axis[start_col])
        previous_z = float(curve_z[start_col])
        misses = 0

        for col in col_iter:
            if abs(float(y_axis[col]) - start_y) > refinement["edge_extension_max_y_m"]:
                break
            col_valid = (
                valid_mask[:, col]
                & np.isfinite(cleaned[:, col])
                & (z_axis >= ice_z_m - refinement["edge_extension_down_m"])
                & (z_axis <= ice_z_m + window_up_m)
                & (np.abs(z_axis - previous_z) <= refinement["edge_extension_step_z_m"])
            )
            rows = np.flatnonzero(col_valid)
            if rows.size == 0:
                misses += 1
            else:
                values = cleaned[rows, col]
                peak_local = int(np.nanargmax(values))
                peak_row = int(rows[peak_local])
                peak_value = float(values[peak_local])
                if peak_value >= min_intensity:
                    row0 = max(0, peak_row - half_width_px)
                    row1 = min(final_mask.shape[0], peak_row + half_width_px + 1)
                    extension[row0:row1, col] = True
                    previous_z = float(z_axis[peak_row])
                    misses = 0
                else:
                    misses += 1
            if misses >= refinement["edge_extension_max_misses"]:
                break
        return extension

    def _bridge_short_gaps_in_mask(
        final_mask: np.ndarray,
        cleaned: np.ndarray,
        valid_mask: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        threshold_low: float,
    ) -> np.ndarray:
        """Add narrow binary-mask bridges only where a gap has weak echo support."""
        curve_z, _ = ns["binary_extract_curve"](final_mask, cleaned, y_axis, z_axis)
        active_cols = np.flatnonzero(np.isfinite(curve_z))
        bridge_mask = np.zeros_like(final_mask, dtype=bool)
        if active_cols.size <= 1:
            return bridge_mask

        dz = float(abs(np.nanmean(np.diff(z_axis))))
        half_width_px = ns["meters_to_pixels"](refinement["mask_bridge_half_width_m"], dz, minimum=1)
        support_floor = max(
            refinement["mask_bridge_support_min"],
            refinement["mask_bridge_support_scale"] * float(threshold_low),
        )

        for left_col, right_col in zip(active_cols[:-1], active_cols[1:]):
            if right_col <= left_col + 1:
                continue
            gap_y_m = float(abs(y_axis[right_col] - y_axis[left_col]))
            gap_z_m = float(abs(curve_z[right_col] - curve_z[left_col]))
            if gap_y_m > refinement["gap_connect_max_y_m"] or gap_z_m > refinement["gap_connect_max_z_m"]:
                continue

            fill_cols = np.arange(left_col + 1, right_col, dtype=int)
            if fill_cols.size == 0:
                continue
            fill_z = np.interp(
                y_axis[fill_cols],
                [y_axis[left_col], y_axis[right_col]],
                [curve_z[left_col], curve_z[right_col]],
            )

            supported_cols = 0
            candidate_rows: list[tuple[int, int, int]] = []
            for col, z_center in zip(fill_cols, fill_z):
                row_center = int(np.nanargmin(np.abs(z_axis - z_center)))
                row0 = max(0, row_center - half_width_px)
                row1 = min(final_mask.shape[0], row_center + half_width_px + 1)
                col_support = (
                    valid_mask[row0:row1, col]
                    & np.isfinite(cleaned[row0:row1, col])
                    & (cleaned[row0:row1, col] >= support_floor)
                )
                if np.any(col_support):
                    supported_cols += 1
                candidate_rows.append((row0, row1, int(col)))

            support_fraction = supported_cols / max(fill_cols.size, 1)
            if support_fraction < refinement["mask_bridge_support_fraction"]:
                continue

            for row0, row1, col in candidate_rows:
                bridge_mask[row0:row1, col] = True

        bridge_mask &= valid_mask & np.isfinite(cleaned) & (~final_mask)
        return bridge_mask

    def refined_binary_interpolate_short_gaps(curve_z, y_axis, max_gap_m=refinement["line_connect_max_y_m"]):
        """Connect measured curve gaps only when horizontal and vertical jumps are both small."""
        curve_z = np.asarray(curve_z, dtype=float)
        y_axis = np.asarray(y_axis, dtype=float)
        out = np.array(curve_z, copy=True)
        interpolated = np.zeros(curve_z.shape, dtype=bool)
        active_cols = np.flatnonzero(np.isfinite(curve_z))
        if active_cols.size <= 1:
            return out, interpolated

        for left_col, right_col in zip(active_cols[:-1], active_cols[1:]):
            if right_col <= left_col + 1:
                continue
            gap_y_m = float(abs(y_axis[right_col] - y_axis[left_col]))
            gap_z_m = float(abs(curve_z[right_col] - curve_z[left_col]))
            if gap_y_m > max_gap_m or gap_z_m > refinement["line_connect_max_z_m"]:
                continue
            fill_cols = np.arange(left_col + 1, right_col, dtype=int)
            out[fill_cols] = np.interp(
                y_axis[fill_cols],
                [y_axis[left_col], y_axis[right_col]],
                [curve_z[left_col], curve_z[right_col]],
            )
            interpolated[fill_cols] = True
        return out, interpolated

    def refined_binary_segment_ice(cleaned, valid_mask, y_axis, z_axis, ice_z_m, config):
        effective = _effective_config(config)
        dy = float(np.nanmean(np.diff(y_axis)))
        dz = float(np.nanmean(np.diff(z_axis)))
        y_grid, z_grid = np.meshgrid(y_axis, z_axis)
        cleaned = np.asarray(cleaned, dtype=float).copy()
        cleaned[~valid_mask] = np.nan

        window_mask = (
            valid_mask
            & np.isfinite(cleaned)
            & (z_grid >= ice_z_m - effective["window_down_m"])
            & (z_grid <= ice_z_m + effective["window_up_m"])
        )
        values = cleaned[window_mask]
        if values.size < refinement["min_window_pixels"]:
            empty = np.zeros_like(valid_mask, dtype=bool)
            meta = {
                "raw_mask": empty,
                "seed_mask": empty,
                "weak_mask": empty,
                "removed_mask": empty,
                "edge_extension_mask": empty,
                "threshold_high": np.nan,
                "threshold_low": np.nan,
                "effective_window_down_m": effective["window_down_m"],
                "effective_window_up_m": effective["window_up_m"],
            }
            meta.update(ns["binary_empty_left_rescue_meta"](valid_mask.shape))
            return empty, meta

        threshold_high = float(np.nanpercentile(values, effective["high_q"]))
        threshold_low = float(np.nanpercentile(values, effective["low_q"]))
        threshold_low = min(threshold_low, threshold_high)

        seed_mask = window_mask & (cleaned >= threshold_high)
        weak_mask = window_mask & (cleaned >= threshold_low)
        raw_mask = ns["binary_component_hysteresis"](seed_mask, weak_mask)

        close_y = ns["meters_to_pixels"](effective["closing_y_m"], dy, minimum=1)
        close_z = ns["meters_to_pixels"](effective["closing_z_m"], dz, minimum=1)
        if close_y > 0 and close_z > 0:
            raw_mask = ns["binary_closing"](raw_mask, footprint=np.ones((close_z, close_y), dtype=bool))

        open_y = ns["meters_to_pixels"](effective["opening_y_m"], dy, minimum=1)
        open_z = ns["meters_to_pixels"](effective["opening_z_m"], dz, minimum=1)
        if open_y > 0 and open_z > 0:
            raw_mask = ns["binary_opening"](raw_mask, footprint=np.ones((open_z, open_y), dtype=bool))

        min_pixels = max(1, int(round(effective["min_area_m2"] / max(abs(dy * dz), 1e-9))))
        raw_mask = ns["remove_small_objects"](raw_mask, min_size=min_pixels, connectivity=2)
        hole_pixels = max(1, int(round(effective["hole_area_m2"] / max(abs(dy * dz), 1e-9))))
        raw_mask = ns["remove_small_holes"](raw_mask, area_threshold=hole_pixels, connectivity=2)
        raw_mask &= valid_mask

        component_labels = ns["label"](raw_mask, connectivity=2)
        final_mask = np.zeros_like(raw_mask, dtype=bool)
        removed_mask = np.zeros_like(raw_mask, dtype=bool)
        for region in ns["regionprops"](component_labels, intensity_image=np.nan_to_num(cleaned, nan=0.0)):
            rows = region.coords[:, 0]
            cols = region.coords[:, 1]
            y_min = float(y_axis[int(cols.min())])
            y_max = float(y_axis[int(cols.max())])
            z_min = float(z_axis[int(rows.min())])
            z_max = float(z_axis[int(rows.max())])
            y_span = y_max - y_min
            z_span = z_max - z_min
            y_centroid = float(y_axis[int(round(region.centroid[1]))])
            z_centroid = float(z_axis[int(round(region.centroid[0]))])
            area_m2 = float(region.area * abs(dy) * abs(dz))

            near_ice = (z_max >= ice_z_m - effective["window_down_m"]) and (z_min <= ice_z_m + effective["window_up_m"])
            enough_support = (y_span >= effective["min_y_span_m"]) or (area_m2 >= effective["min_area_m2"])
            center_line_like = (abs(y_centroid) <= refinement["center_reject_abs_y_m"]) and (y_span <= refinement["center_reject_width_m"]) and (z_span >= refinement["center_reject_height_m"]) and (z_centroid < ice_z_m - refinement["center_reject_below_m"])
            lower_lobe_like = (z_centroid < ice_z_m - refinement["lower_reject_below_m"]) and (abs(y_centroid) <= refinement["lower_reject_abs_y_m"]) and (y_span <= refinement["lower_reject_width_m"])

            if near_ice and enough_support and not center_line_like and not lower_lobe_like:
                final_mask[rows, cols] = True
            else:
                removed_mask[rows, cols] = True

        final_mask &= window_mask
        raw_mask_before_extension = np.asarray(raw_mask, dtype=bool).copy()
        edge_extension_mask = (
            _extend_one_side(final_mask, cleaned, valid_mask, y_axis, z_axis, ice_z_m, threshold_low, "left", effective["window_up_m"])
            | _extend_one_side(final_mask, cleaned, valid_mask, y_axis, z_axis, ice_z_m, threshold_low, "right", effective["window_up_m"])
        )
        edge_extension_mask &= valid_mask & np.isfinite(cleaned) & (~final_mask)
        if np.any(edge_extension_mask):
            final_mask |= edge_extension_mask

        final_mask &= (
            valid_mask
            & np.isfinite(cleaned)
            & (z_grid >= ice_z_m - refinement["edge_extension_down_m"])
            & (z_grid <= ice_z_m + effective["window_up_m"])
        )
        bridge_fill_mask = _bridge_short_gaps_in_mask(
            final_mask,
            cleaned,
            valid_mask,
            y_axis,
            z_axis,
            threshold_low,
        )
        if np.any(bridge_fill_mask):
            final_mask |= bridge_fill_mask
            final_mask &= (
                valid_mask
                & np.isfinite(cleaned)
                & (z_grid >= ice_z_m - refinement["edge_extension_down_m"])
                & (z_grid <= ice_z_m + effective["window_up_m"])
            )
        meta = {
            "raw_mask": raw_mask_before_extension,
            "seed_mask": seed_mask,
            "weak_mask": weak_mask,
            "removed_mask": removed_mask,
            "edge_extension_mask": edge_extension_mask,
            "bridge_fill_mask": bridge_fill_mask,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "effective_window_down_m": effective["window_down_m"],
            "effective_window_up_m": effective["window_up_m"],
            "edge_extension_down_m": refinement["edge_extension_down_m"],
            "bridge_fill_pixels": int(np.count_nonzero(bridge_fill_mask)),
            "bridge_fill_half_width_m": refinement["mask_bridge_half_width_m"],
            "line_connect_max_y_m": refinement["line_connect_max_y_m"],
        }
        meta.update(_left_rescue_meta(final_mask, cleaned, y_axis, z_axis, ice_z_m))
        return final_mask, meta

    ns["binary_interpolate_short_gaps"] = refined_binary_interpolate_short_gaps
    ns["binary_segment_ice"] = refined_binary_segment_ice


def install_publication_binary_renderer(ns: dict) -> None:
    """Install a compact publication-style binary-mask renderer."""

    def binary_render_figure(ping_idx, cleaned, binary_mask, binary_meta, metrics, config, y_axis, z_axis, output_path):
        extent = [float(y_axis[0]), float(y_axis[-1]), float(z_axis[0]), float(z_axis[-1])]
        mvp_cmap = ns["CMAP"].copy()
        mvp_cmap.set_bad("#000000")

        fig_width = 18.0 / 2.54
        fig_height = 5.8 / 2.54
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        gs = fig.add_gridspec(
            1,
            2,
            left=0.08,
            right=0.925,
            bottom=0.22,
            top=0.82,
            wspace=0.32,
        )
        ax_selected = fig.add_subplot(gs[0, 0])
        ax_line = fig.add_subplot(gs[0, 1])
        image_axes = [ax_selected, ax_line]

        axis_xlabel = "Sonar-frame Y (m)"
        axis_ylabel = "Sonar-frame Z (m)"
        box_aspect = abs((extent[3] - extent[2]) / max(abs(extent[1] - extent[0]), 1e-6))

        selected_mask = np.asarray(binary_mask, dtype=bool)

        selected_display = np.full(selected_mask.shape, np.nan, dtype=float)
        selected_display[selected_mask] = 1.0

        im_selected = ax_selected.imshow(cleaned, extent=extent, origin="lower", aspect="equal", cmap=mvp_cmap, vmin=0.0, vmax=1.0)
        ax_selected.imshow(selected_display, extent=extent, origin="lower", aspect="equal", cmap=ListedColormap(["#21c45a"]), alpha=0.68, vmin=0, vmax=1)
        ax_selected.set_title("Selected Ice-bottom Candidate", fontname="Arial", fontsize=8.8, fontweight="bold")
        candidate_handles = [
            Patch(facecolor="#21c45a", edgecolor="none", alpha=0.68, label="Ice-bottom Candidate"),
        ]
        ax_selected.legend(handles=candidate_handles, loc="lower right", fontsize=7.0, frameon=True)

        final_curve = np.asarray(metrics.get("curve_z_reconstructed", metrics["curve_z"]), dtype=float)
        interpolated_mask = np.asarray(metrics.get("curve_interpolated_mask", np.zeros_like(final_curve, dtype=bool)), dtype=bool)

        im_line = ax_line.imshow(cleaned, extent=extent, origin="lower", aspect="equal", cmap=mvp_cmap, vmin=0.0, vmax=1.0)
        ns["plot_indexed_segments"](
            ax_line,
            y_axis,
            final_curve,
            final_curve,
            y_axis,
            color="#00d5ff",
            linewidth=1.05,
            alpha=0.95,
            label="Ice-bottom Line",
        )
        if np.any(interpolated_mask):
            ax_line.scatter(y_axis[interpolated_mask], final_curve[interpolated_mask], s=4.0, color="#ff6b6b", alpha=0.90, linewidths=0)
        ax_line.set_title("Extracted Ice-bottom Line", fontname="Arial", fontsize=8.8, fontweight="bold")
        ax_line.legend(loc="lower right", fontsize=7.0, frameon=True)

        for ax in image_axes:
            ax.set_box_aspect(box_aspect)
            ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)
            ax.tick_params(labelsize=6.6, width=0.7, length=2.6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        ax_selected.set_ylabel(axis_ylabel, fontname="Arial", fontsize=7.5)
        ax_selected.set_xlabel(axis_xlabel, fontname="Arial", fontsize=7.5)
        ax_line.set_xlabel(axis_xlabel, fontname="Arial", fontsize=7.5)

        fig.canvas.draw()
        pos_selected = ax_selected.get_position()
        cax_selected = fig.add_axes([pos_selected.x1 + 0.005, pos_selected.y0, 0.008, pos_selected.height])
        cbar_selected = fig.colorbar(im_selected, cax=cax_selected, ticks=[0.0, 0.5, 1.0])
        cbar_selected.ax.tick_params(labelsize=6.2, width=0.6, length=2.3)
        cbar_selected.ax.set_yticklabels(["0", "0.5", "1"])
        cbar_selected.outline.set_linewidth(0.6)

        pos_line = ax_line.get_position()
        cax_line = fig.add_axes([pos_line.x1 + 0.006, pos_line.y0, 0.010, pos_line.height])
        cbar_line = fig.colorbar(im_line, cax=cax_line, ticks=[0.0, 0.5, 1.0])
        cbar_line.ax.tick_params(labelsize=6.2, width=0.6, length=2.3)
        cbar_line.ax.set_yticklabels(["0", "0.5", "1"])
        cbar_line.outline.set_linewidth(0.6)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    ns["binary_render_figure"] = binary_render_figure
