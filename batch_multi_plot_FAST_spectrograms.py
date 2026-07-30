# -*- coding: utf-8 -*-
"""
Plots a folder of FAST ESA data as spectrograms.

Assumed folder layout is::
    {FAST_CDF_DATA_FOLDER_PATH}/year/month

Filenames in the month folders assumed to be in the following formats::
    {??}_{??}_{??}_{instrument}_{timestamp}_{orbit}_v02.cdf      (known "instruments" are ees, eeb, ies, or ieb)
    {??}_{??}_orb_{orbit}_{??}.cdf

Examples::
    FAST_data/2000/01/fa_esa_l2_eeb_20000101001737_13312_v02.cdf
    FAST_data/2000/01/fa_k0_orb_13312_v01.cdf

All FAST-specific plotting/batch logic lives in
``configurable_spectrograms.fast``; this module re-exports the public
functions/constants for backward compatibility and provides the CLI entry
point that runs every y/z scale combination in sequence.
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
__version__: str = "0.0.3"
__license__: str = "GPL-3.0"

import sys

from configurable_spectrograms.fast.batch_directory import FAST_plot_spectrograms_directory
from configurable_spectrograms.fast.constants import (
    CDF_VARIABLES,
    DEFAULT_COLORMAP_LINEAR_Y_LINEAR_Z,
    DEFAULT_COLORMAP_LINEAR_Y_LOG_Z,
    DEFAULT_COLORMAP_LOG_Y_LINEAR_Z,
    DEFAULT_COLORMAP_LOG_Y_LOG_Z,
    DEFAULT_INSTRUMENT_ORDER,
    FAST_CDF_DATA_FOLDER_PATH,
    FAST_FILTERED_ORBITS_CSV_PATH,
    FAST_OUTPUT_BASE,
    FAST_PLOTTING_PROGRESS_JSON,
)
from configurable_spectrograms.fast.extrema import compute_global_extrema
from configurable_spectrograms.fast.orbit_discovery import extract_orbit_and_instrument
from configurable_spectrograms.fast.plotting import FAST_plot_instrument_grid, FAST_plot_pitch_angle_grid
from configurable_spectrograms.fast.process_orbit import FAST_process_single_orbit
from configurable_spectrograms.logging_utils import get_logfile_path, log_exception, set_logfile_path
from configurable_spectrograms.percentile_utils import round_extrema

__all__ = [
    "CDF_VARIABLES",
    "DEFAULT_COLORMAP_LINEAR_Y_LINEAR_Z",
    "DEFAULT_COLORMAP_LINEAR_Y_LOG_Z",
    "DEFAULT_COLORMAP_LOG_Y_LINEAR_Z",
    "DEFAULT_COLORMAP_LOG_Y_LOG_Z",
    "DEFAULT_INSTRUMENT_ORDER",
    "FAST_CDF_DATA_FOLDER_PATH",
    "FAST_FILTERED_ORBITS_CSV_PATH",
    "FAST_OUTPUT_BASE",
    "FAST_PLOTTING_PROGRESS_JSON",
    "FAST_plot_instrument_grid",
    "FAST_plot_pitch_angle_grid",
    "FAST_plot_spectrograms_directory",
    "FAST_process_single_orbit",
    "compute_global_extrema",
    "extract_orbit_and_instrument",
    "round_extrema",
]


def main() -> None:
    """Run the FAST batch plotter for all y/z scale combinations sequentially.

    Invokes ``FAST_plot_spectrograms_directory`` for each combination of
    linear/log y and z, using colormaps tailored for each. An interrupt
    during any run stops the sequence without starting subsequent
    combinations.
    """
    set_logfile_path(get_logfile_path("./batch_multi_plot_FAST_log", "./batch_multi_plot_FAST_logfile_datetime.txt"))
    for y_scale, z_scale, colormap in [
        ("linear", "linear", DEFAULT_COLORMAP_LINEAR_Y_LINEAR_Z),
        ("linear", "log", DEFAULT_COLORMAP_LINEAR_Y_LOG_Z),
        ("log", "linear", DEFAULT_COLORMAP_LOG_Y_LINEAR_Z),
        ("log", "log", DEFAULT_COLORMAP_LOG_Y_LOG_Z),
    ]:
        FAST_plot_spectrograms_directory(
            FAST_CDF_DATA_FOLDER_PATH,
            verbose=False,
            y_scale=y_scale,
            z_scale=z_scale,
            use_tqdm=True,
            colormap=colormap,
            max_processing_percentile=99,
            override_plots=False,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_exception("[INTERRUPT] Batch plotting aborted by user.", level="message")
        print("\n[INTERRUPT] Aborted by user.")
        sys.exit(130)
