#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI to render a single generic spectrogram figure from one CDF file.

Companion to ``batch_multi_plot_spectrogram.py``: where that script (via
``configurable_spectrograms.generic_batch``) renders many items in
parallel, this script renders exactly one item and exits -- no
``ProcessPoolExecutor``, no progress JSON. All rendering logic lives in
``configurable_spectrograms``; this script only parses arguments, loads one
CDF file, and calls the library.
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
from pathlib import Path

from configurable_spectrograms.cdf_utils import load_fast_cdf_dataset
from configurable_spectrograms.plotting import generic_plot_spectrogram_set


def render_single_spectrogram(
    cdf_file_path: str,
    output_path: str,
    y_scale: str = "linear",
    z_scale: str = "linear",
    colormap: str = "viridis",
    cusp_marker_style: str = "line",
    vertical_lines: list[float] | None = None,
) -> bool:
    """Render a single generic spectrogram from one CDF file and save it.

    Parameters
    ----------
    cdf_file_path : str
        Path to the CDF file (must contain ``time_unix``, ``data``,
        ``energy``, and ``pitch_angle`` variables).
    output_path : str
        Destination PNG path.
    y_scale : {'linear', 'log'}, default 'linear'
        Y-axis scaling.
    z_scale : {'linear', 'log'}, default 'linear'
        Color scale for intensity.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    cusp_marker_style : {'line', 'bracket'}, default 'line'
        Cusp-boundary marker style; see
        :mod:`configurable_spectrograms.cusp_marking`.
    vertical_lines : list of float or None, optional
        UNIX timestamps to mark as a cusp boundary.

    Returns
    -------
    bool
        ``True`` if a figure was produced and saved, ``False`` otherwise.
    """
    dataset = load_fast_cdf_dataset(cdf_file_path)
    datasets = [
        {
            "x": dataset["times"],
            "y": dataset["energy"],
            "data": dataset["data"],
            "label": Path(cdf_file_path).stem,
        }
    ]
    fig, _canvas = generic_plot_spectrogram_set(
        datasets,
        vertical_lines=vertical_lines,
        y_scale=y_scale,
        z_scale=z_scale,
        colormap=colormap,
        cusp_marker_style=cusp_marker_style,
        show=False,
    )
    if fig is None:
        return False
    fig.savefig(output_path, dpi=150)
    return True


def main() -> int:
    """Parse CLI arguments and render a single generic spectrogram figure."""
    parser = argparse.ArgumentParser(description="Render a single generic spectrogram figure from one CDF file.")
    parser.add_argument("--cdf-file", required=True, help="Path to the CDF file to plot.")
    parser.add_argument("--output", required=True, help="Destination PNG file path.")
    parser.add_argument("--y-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--z-scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--cusp-style", choices=("line", "bracket"), default="line")
    args = parser.parse_args()

    produced = render_single_spectrogram(
        args.cdf_file,
        args.output,
        y_scale=args.y_scale,
        z_scale=args.z_scale,
        colormap=args.colormap,
        cusp_marker_style=args.cusp_style,
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
