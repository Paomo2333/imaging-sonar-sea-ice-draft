from __future__ import annotations

import unittest

import numpy as np

from ice_sonar_pipeline.algorithm_cells import CELL_SOURCES
from ice_sonar_pipeline.draft_reconstruction import roll_correct_yz
from ice_sonar_pipeline.oculus_backward_mapping import (
    SonarGridConfig,
    backward_map_ping_center_plane,
)


class CorePipelineTests(unittest.TestCase):
    def test_public_algorithm_blocks_are_limited_to_runtime_dependencies(self) -> None:
        retained = {index for index, source in enumerate(CELL_SOURCES) if source}
        self.assertEqual(retained, {2, 4, 6, 14, 21})

    def test_center_plane_backward_mapping_preserves_center_echo_range(self) -> None:
        image = np.zeros((100, 21), dtype=float)
        image[49:52, 10] = 1.0
        metadata = {
            "n_sample": 100,
            "n_beam": 21,
            "sample_size_m": 0.1,
            "azimuth_range_rad": np.deg2rad(120.0),
            "raw_range_max_m": 10.0,
        }
        result = backward_map_ping_center_plane(
            image,
            metadata,
            SonarGridConfig(
                range_max_m=10.0,
                grid_resolution_m=0.1,
                azimuth_half_angle_deg=60.0,
            ),
        )
        mapped = np.asarray(result["mapped_backscatter"])
        y_axis = np.asarray(result["y_axis_m"])
        z_axis = np.asarray(result["z_axis_m"])
        center_col = int(np.argmin(np.abs(y_axis)))
        center_peak_z = float(z_axis[np.nanargmax(mapped[:, center_col])])
        self.assertLessEqual(abs(center_peak_z - 5.0), 0.2)

    def test_roll_correction_is_reversible(self) -> None:
        y = np.array([-2.0, 0.0, 2.0])
        z = np.array([8.0, 9.0, 10.0])
        y_rot, z_rot = roll_correct_yz(y, z, roll_deg=7.5, sign=1)
        y_back, z_back = roll_correct_yz(y_rot, z_rot, roll_deg=7.5, sign=-1)
        np.testing.assert_allclose(y_back, y, atol=1e-12)
        np.testing.assert_allclose(z_back, z, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
