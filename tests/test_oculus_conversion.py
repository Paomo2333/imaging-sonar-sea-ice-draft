from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import xarray as xr

from ice_sonar_pipeline.oculus_to_netcdf import (
    convert_oculus_to_netcdf,
    find_bps_oculus_io,
    validate_oculus_netcdf,
)


class OculusConversionTests(unittest.TestCase):
    def test_validate_oculus_netcdf_accepts_non_empty_backscatter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "synthetic.nc"
            dataset = xr.Dataset(
                {
                    "backscatter": (
                        ("sample", "beam", "ping"),
                        np.ones((8, 5, 3), dtype=np.float32),
                    ),
                    "time": (("ping",), np.arange(3, dtype=float)),
                }
            )
            dataset.to_netcdf(path)

            summary = validate_oculus_netcdf(path)

            self.assertEqual(summary["dimensions"], {"sample": 8, "beam": 5, "ping": 3})
            self.assertIn("backscatter", summary["variables"])

    def test_validate_oculus_netcdf_rejects_missing_backscatter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "invalid.nc"
            xr.Dataset({"time": (("ping",), np.arange(3, dtype=float))}).to_netcdf(path)

            with self.assertRaisesRegex(ValueError, "backscatter"):
                validate_oculus_netcdf(path)

    def test_explicit_missing_cli_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            find_bps_oculus_io(Path("definitely_missing_bps_oculus_io.exe"))

    def test_conversion_wrapper_validates_output_and_removes_staged_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            source = source_dir / "example.oculus"
            source.write_bytes(b"synthetic placeholder")
            fake_cli = root / "bps_oculus_io.exe"
            fake_cli.write_bytes(b"")

            def fake_run(command, **kwargs):
                dataset = xr.Dataset(
                    {
                        "backscatter": (
                            ("sample", "beam", "ping"),
                            np.ones((4, 3, 2), dtype=np.float32),
                        ),
                        "time": (("ping",), np.arange(2, dtype=float)),
                    }
                )
                dataset.to_netcdf(output_dir / "example.nc")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch(
                "ice_sonar_pipeline.oculus_to_netcdf.subprocess.run",
                side_effect=fake_run,
            ):
                result = convert_oculus_to_netcdf(
                    source,
                    output_dir,
                    cli=fake_cli,
                )

            self.assertEqual(result.output, (output_dir / "example.nc").resolve())
            self.assertTrue(source.exists())
            self.assertFalse((output_dir / source.name).exists())


if __name__ == "__main__":
    unittest.main()
