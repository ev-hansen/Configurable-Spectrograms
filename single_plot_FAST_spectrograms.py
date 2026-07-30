#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI to render a single FAST ESA spectrogram figure (one CDF file, or one orbit).

Companion to ``batch_multi_plot_FAST_spectrograms.py``: where that script
(via ``configurable_spectrograms.fast.batch_directory``) processes every
orbit in a directory in parallel, this script renders exactly one figure
and exits. All rendering logic lives in
``configurable_spectrograms.fast.plotting``; this script only parses
arguments and calls it -- the same function the GUI's Single Plot page
calls.
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
]

__date__: str = "2026-07-30"
__status__: str = "Development"
__version__: str = "0.0.1"
__license__: str = "GPL-3.0"

import argparse
import sys

from configurable_spectrograms.cdf_utils import load_filtered_orbits
from configurable_spectrograms.fast.orbit_discovery import discover_orbit_files, extract_orbit_and_instrument
from configurable_spectrograms.fast.plotting import FAST_plot_instrument_grid, FAST_plot_pitch_angle_grid


def render_single_pitch_angle_grid(
    cdf_file_path: str,
    output_path: str,
    y_scale: str = "linear",
    z_scale: str = "linear",
    colormap: str = "viridis",
    cusp_marker_style: str = "line",
) -> bool:
    """Render one CDF file's pitch-angle grid and save it to *output_path*.

    The orbit number (used to look up the cusp boundary) is parsed
    automatically from the filename.

    Parameters
    ----------
    cdf_file_path : str
        Path to the instrument CDF file.
    output_path : str
        Destination PNG path.
    y_scale, z_scale : {'linear', 'log'}, default 'linear'
        Axis scaling.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    cusp_marker_style : {'line', 'bracket'}, default 'line'
        Cusp-boundary marker style.

    Returns
    -------
    bool
        ``True`` if a figure was produced and saved, ``False`` otherwise.
    """
    filtered_orbits_df = load_filtered_orbits()
    parsed = extract_orbit_and_instrument(cdf_file_path)
    orbit_number = parsed[0] if parsed is not None else None
    fig, _canvas = FAST_plot_pitch_angle_grid(
        cdf_file_path,
        filtered_orbits_df=filtered_orbits_df,
        orbit_number=orbit_number,
        scale_function_y=y_scale,
        scale_function_z=z_scale,
        show=False,
        colormap=colormap,
        cusp_marker_style=cusp_marker_style,
    )
    if fig is None:
        return False
    fig.savefig(output_path, dpi=200)
    return True


def render_single_instrument_grid(
    data_folder: str,
    orbit_number: int,
    output_path: str,
    y_scale: str = "linear",
    z_scale: str = "linear",
    colormap: str = "viridis",
    cusp_marker_style: str = "line",
) -> bool:
    """Render one orbit's multi-instrument grid resolved from a data folder.

    Parameters
    ----------
    data_folder : str
        Root folder to search for the orbit's instrument CDF files.
    orbit_number : int
        Orbit number to resolve within *data_folder*.
    output_path : str
        Destination PNG path.
    y_scale, z_scale : {'linear', 'log'}, default 'linear'
        Axis scaling.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    cusp_marker_style : {'line', 'bracket'}, default 'line'
        Cusp-boundary marker style.

    Returns
    -------
    bool
        ``True`` if a figure was produced and saved, ``False`` otherwise.
    """
    filtered_orbits_df = load_filtered_orbits()
    instrument_files = discover_orbit_files(data_folder).get(orbit_number, {})
    if not instrument_files:
        return False
    fig, _canvas = FAST_plot_instrument_grid(
        instrument_files,
        filtered_orbits_df=filtered_orbits_df,
        orbit_number=orbit_number,
        scale_function_y=y_scale,
        scale_function_z=z_scale,
        show=False,
        colormap=colormap,
        cusp_marker_style=cusp_marker_style,
    )
    if fig is None:
        return False
    fig.savefig(output_path, dpi=200)
    return True


def main() -> int:
    """Parse CLI arguments and render a single FAST ESA spectrogram figure."""
    parser = argparse.ArgumentParser(description="Render a single FAST ESA spectrogram figure.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--cdf-file", help="Single CDF file to render as a pitch-angle grid.")
    mode_group.add_argument("--data-folder", help="Data folder to search for one orbit's instrument grid.")
    parser.add_argument("--orbit", type=int, help="Orbit number (required with --data-folder).")
    parser.add_argument("--output", required=True, help="Destination PNG file path.")
    parser.add_argument("--y-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--z-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--cusp-style", choices=("line", "bracket"), default="line")
    args = parser.parse_args()

    if args.data_folder is not None and args.orbit is None:
        parser.error("--orbit is required when using --data-folder")

    if args.cdf_file is not None:
        produced = render_single_pitch_angle_grid(
            args.cdf_file, args.output, args.y_scale, args.z_scale, args.colormap, args.cusp_style
        )
    else:
        produced = render_single_instrument_grid(
            args.data_folder, args.orbit, args.output, args.y_scale, args.z_scale, args.colormap, args.cusp_style
        )

    if not produced:
        print("[WARNING] No data available to plot for the given input.")
        return 1
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Aborted by user.")
        sys.exit(130)
