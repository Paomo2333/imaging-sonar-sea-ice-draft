"""Run the command-line pipeline from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


VALUE_OPTIONS = {
    "nc_file": "--nc-file",
    "algorithm_config": "--algorithm-config",
    "motion_file": "--motion-file",
    "output_root": "--output-root",
    "timezone": "--timezone",
    "start_local": "--start-local",
    "end_local": "--end-local",
    "start_ping": "--start-ping",
    "end_ping": "--end-ping",
    "ping_step": "--ping-step",
    "exclude_ping_ranges": "--exclude-ping-ranges",
    "grid_range_mode": "--grid-range-mode",
    "grid_range_max": "--grid-range-max",
    "grid_range_round_m": "--grid-range-round-m",
    "binary_tuning_ping_step": "--binary-tuning-ping-step",
    "max_count": "--max-count",
    "binary_tuning_workers": "--binary-tuning-workers",
    "manual_mask_json": "--manual-mask-json",
}

FLAG_OPTIONS = {
    "no_process_figures": "--no-process-figures",
    "no_gray_figures": "--no-gray-figures",
    "gray_only": "--gray-only",
    "force_retune": "--force-retune",
}

PATH_KEYS = {"nc_file", "motion_file", "output_root", "manual_mask_json", "algorithm_config"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON run configuration.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unknown = set(config) - set(VALUE_OPTIONS) - set(FLAG_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")

    command = [sys.executable, "-m", "ice_sonar_pipeline.sonar_draft_pipeline"]
    for key, option in VALUE_OPTIONS.items():
        value = config.get(key)
        if value is None:
            continue
        if key in PATH_KEYS:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (config_path.parent / path).resolve()
            value = path
        command.extend([option, str(value)])
    for key, option in FLAG_OPTIONS.items():
        if bool(config.get(key, False)):
            command.append(option)

    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
