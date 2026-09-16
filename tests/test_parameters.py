from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from ice_sonar_pipeline.parameters import load_parameters, configure_binary_candidates, configure_mapping
from ice_sonar_pipeline.oculus_backward_mapping import SonarGridConfig, backward_map_ping_center_plane
from ice_sonar_pipeline.draft_reconstruction import roll_correct_yz, robust_centerline_stat
from ice_sonar_pipeline.postprocessing import add_postprocessing_columns
from ice_sonar_pipeline.artifact_suppression import install_gray_denoising
from ice_sonar_pipeline.ice_candidate import install_binary_band_refinement


class ParameterBehaviorTests(unittest.TestCase):
    def test_initial_file_matches_cli_and_fixed_candidate(self):
        path = Path(__file__).resolve().parents[1] / 'config' / 'initial_parameters.json'
        p = load_parameters(path)
        self.assertEqual(p, load_parameters())
        ns = {'ALGORITHM_PARAMETERS': p}
        configure_binary_candidates(ns)
        self.assertEqual(len(ns['BINARY_CANDIDATE_CONFIGS']), 1)
        candidate = ns['BINARY_CANDIDATE_CONFIGS'][0]
        self.assertEqual((candidate['high_q'], candidate['low_q']), (86, 66))
        self.assertEqual((candidate['window_down_m'], candidate['window_up_m']), (1.2, .6))

    def test_invalid_or_misspelled_settings_fail_early(self):
        for patch in [{'binary': {'low_q': 99}}, {'background': {'bin_m': 0}},
                      {'grid': {'beam_order': 'unknown'}}, {'center': {'min_points': 2.5}},
                      {'postprocess': {'max_gap_s': -1}}, {'gray': {'weak_floor': 2}},
                      {'attitude': {'roll_sign': 0}}, {'binary': {'high_Q': 86}}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                load_parameters(overrides=patch)

    def test_beam_reversal_preserves_port_positive_geometry(self):
        im = np.zeros((100, 21))
        im[49:52, 15] = 1
        meta = dict(n_sample=100, n_beam=21, sample_size_m=.1,
                    raw_range_max_m=10, azimuth_range_rad=np.deg2rad(120))
        a = backward_map_ping_center_plane(im, meta, SonarGridConfig(range_max_m=10, grid_resolution_m=.1, azimuth_half_angle_deg=60))
        b = backward_map_ping_center_plane(im[:, ::-1], meta, SonarGridConfig(range_max_m=10, grid_resolution_m=.1, azimuth_half_angle_deg=60, beam_order='port_to_starboard'))
        np.testing.assert_allclose(a['mapped_backscatter'], b['mapped_backscatter'], equal_nan=True)
        peak = np.unravel_index(np.nanargmax(a['mapped_backscatter']), a['mapped_backscatter'].shape)
        self.assertGreater(a['y_axis_m'][peak[1]], 0)
        cropped = backward_map_ping_center_plane(im, meta, SonarGridConfig(range_max_m=10, grid_resolution_m=.1, azimuth_half_angle_deg=40))
        keep = cropped['valid_mask']
        np.testing.assert_allclose(cropped['mapped_backscatter'][keep], a['mapped_backscatter'][keep], equal_nan=True)

    def test_range_scale_reaches_input_metadata_once(self):
        ns = {'GRID_CONFIG': SonarGridConfig(), 'read_ping_frame': lambda: (np.zeros((2, 2)), {'sample_size_m': .1, 'raw_range_max_m': 1.0})}
        configure_mapping(ns, load_parameters(overrides={'grid': {'range_scale': 1.2, 'grid_resolution_m': .08}}))
        _, meta = ns['read_ping_frame']()
        self.assertAlmostEqual(meta['sample_size_m'], .12)
        self.assertAlmostEqual(meta['raw_range_max_m'], 1.2)
        self.assertEqual(ns['GRID_CONFIG'].grid_resolution_m, .08)

    def test_installed_background_and_gap_functions_use_overrides(self):
        p = load_parameters(overrides={'background': {'bin_m': .5}, 'binary_refinement': {'line_connect_max_y_m': .8, 'line_connect_max_z_m': .05}})
        ns = {'ALGORITHM_PARAMETERS': p}
        install_gray_denoising(ns)
        y = np.linspace(-1, 1, 41)
        z = np.linspace(0, 10, 201)
        yy, zz = np.meshgrid(y, z)
        image = .01 + .02 * np.sin(zz) ** 2 + np.exp(-((zz-8)**2)/.02)
        prepared = ns['prepare_noise_suppression'](image, np.ones_like(image, bool), y, z)
        expected = len(np.arange(0, np.hypot(yy, zz).max() + .5, .5)) - 1
        self.assertEqual(prepared['adaptive_meta']['range_bin_count'], expected)
        install_binary_band_refinement(ns)
        _, filled = ns['binary_interpolate_short_gaps'](np.array([9, np.nan, 9.1]), np.array([0, .35, .7]))
        self.assertFalse(filled[1])
        p['binary_refinement']['line_connect_max_z_m'] = .2
        _, filled = ns['binary_interpolate_short_gaps'](np.array([9, np.nan, 9.1]), np.array([0, .35, .7]))
        self.assertTrue(filled[1])
        install_gray_denoising({'ALGORITHM_PARAMETERS': load_parameters()})

    def test_postprocessing_can_be_disabled_or_time_bounded(self):
        frame = pd.DataFrame({'ice_draft_m': [1., -1., -1., -1., -1., 2.], 'ping_time_unix_s': np.arange(6.)})
        options = load_parameters()['postprocess']
        processed = add_postprocessing_columns(frame, options)
        np.testing.assert_allclose(processed['ice_draft_qc_input_m'], np.linspace(1, 2, 6))
        bounded = add_postprocessing_columns(frame, dict(options, max_gap_s=2.))
        self.assertTrue(bounded['ice_draft_smoothed_m'].iloc[1:5].isna().all())
        disabled = add_postprocessing_columns(frame, dict(options, enabled=False))
        np.testing.assert_array_equal(disabled['ice_draft_smoothed_m'], frame['ice_draft_m'])
        np.testing.assert_array_equal(processed['ice_draft_m'], frame['ice_draft_m'])
        no_fill = add_postprocessing_columns(frame, dict(options, interpolate=False))
        self.assertTrue(no_fill['ice_draft_smoothed_m'].iloc[1:5].isna().all())

    def test_right_handed_scalar_vertical_matches_rotation(self):
        p = load_parameters()
        y, z = np.array([-2., .3, 2.]), np.array([8., 9., 10.])
        roll, pitch = np.deg2rad([7., -4.])
        rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        points = np.vstack([np.zeros_like(y), y, z])
        matrix_h = -(ry @ rx @ np.diag([1, -1, -1]) @ points)[2]
        _, zc = roll_correct_yz(y, z, 7, p['attitude']['roll_sign'])
        np.testing.assert_allclose(zc * np.cos(pitch), matrix_h, atol=1e-12)

    def test_center_fallback_defaults_are_retained(self):
        points = [dict(y_roll_corrected_m=y, z_roll_corrected_m=9., y_s_m=y, z_s_m=9., curve_weight=1.) for y in [1., 1.1, 1.2, 1.3, 1.4, 1.5]]
        c = load_parameters()['center']
        result = robust_centerline_stat(points, c['half_width_m'], c['min_points'], c['fallback_points'])
        self.assertEqual(result['center_method'], 'nearest_roll_corrected_center_points')
        self.assertEqual(result['center_point_count'], 6)


if __name__ == '__main__':
    unittest.main()
