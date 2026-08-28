"""Generate a small, non-observational dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def build_sonar_dataset(n_ping: int = 12, n_sample: int = 300, n_beam: int = 96) -> xr.Dataset:
    rng = np.random.default_rng(42)
    sample_size_m = 0.05
    time_s = 1_700_000_000.0 + np.arange(n_ping, dtype=float)
    beam_angle = np.linspace(-65.0, 65.0, n_beam)
    backscatter = rng.gamma(shape=1.3, scale=0.012, size=(n_beam, n_sample, n_ping))

    for ping in range(n_ping):
        ice_range_m = 10.4 + 0.18 * np.sin(np.deg2rad(beam_angle * 1.4)) + 0.03 * ping
        ice_sample = ice_range_m / sample_size_m
        sample_axis = np.arange(n_sample, dtype=float)
        for beam in range(n_beam):
            band = np.exp(-0.5 * ((sample_axis - ice_sample[beam]) / 2.2) ** 2)
            backscatter[beam, :, ping] += 0.85 * band

        center = n_beam // 2
        backscatter[center - 1 : center + 2, 25:190, ping] += 0.20
        for beam in range(center + 8, center + 20):
            lobe_sample = 95 + 2 * (beam - center)
            backscatter[beam, max(0, lobe_sample - 2) : lobe_sample + 3, ping] += 0.18

    return xr.Dataset(
        data_vars={
            "backscatter": (("beam", "sample", "ping"), backscatter.astype(np.float32)),
            "sample_size": (("ping",), np.full(n_ping, sample_size_m, dtype=np.float32)),
            "nsamples": (("ping",), np.full(n_ping, n_sample, dtype=np.int32)),
            "azimuth_range": (("ping",), np.full(n_ping, np.deg2rad(130.0), dtype=np.float32)),
            "time": (("ping",), time_s),
        },
        coords={
            "beam": np.arange(n_beam),
            "sample": np.arange(n_sample),
            "ping": np.arange(n_ping),
        },
        attrs={
            "title": "Synthetic upward-looking imaging-sonar example",
            "note": "Generated data only; no field observations are included.",
        },
    )


def build_motion_table(n_ping: int = 12) -> pd.DataFrame:
    time_s = 1_700_000_000.0 + np.arange(n_ping, dtype=float)
    phase = np.linspace(0.0, 2.0 * np.pi, n_ping)
    return pd.DataFrame(
        {
            "time_unix_s": time_s,
            "pitch_filtered_deg": 1.2 * np.sin(phase),
            "roll_filtered_deg": 2.0 * np.cos(phase),
            "NAV_Heading": 15.0 + np.linspace(0.0, 3.0, n_ping),
            "NAV_DEPTH": 12.0 + 0.03 * np.sin(phase),
            "NAV_ALTITUDE": np.full(n_ping, 4.0),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="generated", help="Directory for generated test inputs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sonar_path = output_dir / "synthetic_sonar.nc"
    motion_path = output_dir / "synthetic_motion.csv"
    build_sonar_dataset().to_netcdf(sonar_path)
    build_motion_table().to_csv(motion_path, index=False)
    print(f"Synthetic sonar: {sonar_path}")
    print(f"Synthetic motion: {motion_path}")


if __name__ == "__main__":
    main()
