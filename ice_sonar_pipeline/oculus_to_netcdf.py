"""Convert Blueprint Subsea Oculus logs to pipeline-ready NetCDF files.

The binary parser and NetCDF exporter are provided by the external
``oculus-python`` package.  This module supplies a small, auditable wrapper
around its ``bps_oculus_io`` command and validates the resulting dataset; it
does not contain or modify third-party parser code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np
import xarray as xr


_CLI_NAMES = (
    "bps_oculus_io",
    "bps_oculus_io.exe",
    "bps-oculus-io",
    "bps-oculus-io.exe",
)
_OUTPUT_SUFFIXES = (".nc", ".nc4", ".netcdf", ".netcdf4")


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one successful Oculus-to-NetCDF conversion."""

    source: Path
    output: Path
    cli: Path
    dimensions: dict[str, int]
    variables: tuple[str, ...]


def find_bps_oculus_io(explicit: str | Path | None = None) -> Path:
    """Locate the ``bps_oculus_io`` executable installed by oculus-python."""

    if explicit is not None:
        requested = Path(explicit).expanduser()
        found = shutil.which(str(requested))
        candidate = Path(found) if found else requested
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"bps_oculus_io executable not found: {requested}")

    candidates: list[Path] = []
    for name in _CLI_NAMES:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    executable_dir = Path(sys.executable).resolve().parent
    for directory in (
        executable_dir,
        executable_dir / "Scripts",
        executable_dir.parent / "Scripts",
    ):
        candidates.extend(directory / name for name in _CLI_NAMES)

    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "bps_oculus_io was not found. Install the conversion extra with "
        "`python -m pip install -e \".[oculus]\"`."
    )


def _candidate_outputs(output_dir: Path, stem: str) -> list[Path]:
    candidates: list[Path] = []
    for suffix in _OUTPUT_SUFFIXES:
        candidates.extend(output_dir.glob(f"{stem}*{suffix}"))
    return sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def _stage_input(source: Path, output_dir: Path) -> tuple[Path, bool]:
    """Stage the raw log beside the output, preferring a zero-copy hard link."""

    staged = output_dir / source.name
    try:
        if staged.resolve() == source.resolve():
            return source, False
    except OSError:
        pass

    if staged.exists():
        if staged.stat().st_size != source.stat().st_size:
            raise FileExistsError(
                f"Staging path exists with a different size: {staged}"
            )
        return staged, False

    try:
        os.link(source, staged)
    except OSError:
        shutil.copy2(source, staged)
    return staged, True


def validate_oculus_netcdf(path: str | Path) -> dict[str, object]:
    """Validate that a converted NetCDF contains non-empty sonar data."""

    nc_path = Path(path).expanduser().resolve()
    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF output not found: {nc_path}")

    with xr.open_dataset(nc_path, decode_times=False) as dataset:
        dimensions = {name: int(size) for name, size in dataset.sizes.items()}
        empty = {name: size for name, size in dimensions.items() if size <= 0}
        if empty:
            raise ValueError(f"NetCDF contains empty dimensions: {empty}")
        if "backscatter" not in dataset.data_vars:
            raise ValueError("NetCDF does not contain the required `backscatter` variable")

        backscatter = dataset["backscatter"]
        if backscatter.ndim < 2 or any(size <= 0 for size in backscatter.shape):
            raise ValueError(
                "`backscatter` must be a non-empty two- or three-dimensional array"
            )
        if not np.issubdtype(backscatter.dtype, np.number):
            raise TypeError("`backscatter` must contain numeric echo intensities")

        variables = tuple(sorted(dataset.data_vars))

    return {
        "dimensions": dimensions,
        "variables": variables,
    }


def convert_oculus_to_netcdf(
    source: str | Path,
    output_dir: str | Path,
    *,
    cli: str | Path | None = None,
    overwrite: bool = False,
    keep_work_copy: bool = False,
) -> ConversionResult:
    """Convert one ``.oculus`` log and validate the exported NetCDF file."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Oculus log not found: {source_path}")
    if source_path.suffix.lower() != ".oculus":
        raise ValueError(f"Expected a .oculus file, received: {source_path.name}")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    existing = _candidate_outputs(destination, source_path.stem)
    if existing and not overwrite:
        listed = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "A matching NetCDF output already exists. Use --overwrite only after "
            f"checking it:\n{listed}"
        )

    executable = find_bps_oculus_io(cli)
    staged_path, staged_here = _stage_input(source_path, destination)
    command = [str(executable), str(staged_path), "--output", "netcdf4"]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        completed = subprocess.run(
            command,
            cwd=destination,
            env=environment,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            if len(details) > 6000:
                details = details[-6000:]
            raise RuntimeError(
                f"bps_oculus_io failed with exit code {completed.returncode}.\n{details}"
            )

        outputs = _candidate_outputs(destination, source_path.stem)
        if not outputs:
            raise FileNotFoundError(
                f"The converter finished but no NetCDF output was found in {destination}"
            )

        validation_errors: list[str] = []
        for candidate in outputs:
            try:
                summary = validate_oculus_netcdf(candidate)
            except (OSError, TypeError, ValueError) as exc:
                validation_errors.append(f"{candidate.name}: {exc}")
                continue
            return ConversionResult(
                source=source_path,
                output=candidate,
                cli=executable,
                dimensions=summary["dimensions"],
                variables=summary["variables"],
            )

        raise ValueError(
            "No valid NetCDF output was produced:\n" + "\n".join(validation_errors)
        )
    finally:
        if staged_here and not keep_work_copy and staged_path.exists():
            staged_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Blueprint Subsea Oculus .oculus log to NetCDF4 using "
            "the external oculus-python command-line exporter."
        )
    )
    parser.add_argument("--input", required=True, help="Path to one .oculus log")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the exported NetCDF file",
    )
    parser.add_argument(
        "--cli",
        default=None,
        help="Optional explicit path to bps_oculus_io",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow the external converter to replace a matching output",
    )
    parser.add_argument(
        "--keep-work-copy",
        action="store_true",
        help="Keep the staged .oculus hard link or copy in the output directory",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = convert_oculus_to_netcdf(
        args.input,
        args.output_dir,
        cli=args.cli,
        overwrite=args.overwrite,
        keep_work_copy=args.keep_work_copy,
    )
    print(f"NetCDF: {result.output}")
    print(f"Dimensions: {result.dimensions}")
    print(f"Variables: {', '.join(result.variables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
