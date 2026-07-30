#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides batch spectrogram plotting utilities.
Should work with CDFs like those from FAST (see batch_multi_plot_FAST_spectrograms.py)
but should also be flexible with other data.

Assumed folder layout is::
    {CDF_DATA_DIRECTORY}/year/month

Filenames in the month folders assumed to be in the following formats::
    {??}_{??}_{??}_{instrument}_{timestamp}_{orbit}_v02.cdf      (known "instruments" are ees, eeb, ies, or ieb)
    {??}_{??}_orb_{orbit}_{??}.cdf

Examples::
    FAST_data/2000/01/fa_esa_l2_eeb_20000101001737_13312_v02.cdf
    FAST_data/2000/01/fa_k0_orb_13312_v01.cdf

All plotting/batch logic lives in the ``configurable_spectrograms`` package;
this module re-exports the public functions/constants for backward
compatibility with existing imports (``from batch_multi_plot_spectrogram
import make_spectrogram``, etc).
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
]

__date__: str = "2025-08-13"
__status__: str = "Development"
__version__: str = "0.0.2"
__license__: str = "GPL-3.0"

from configurable_spectrograms.cdf_utils import (
    get_cdf_file_type,
    get_cdf_var_shapes,
    get_timestamps_for_orbit,
    get_variable_shape,
    load_filtered_orbits,
)
from configurable_spectrograms.constants import (
    CDF_DATA_DIRECTORY,
    CDF_VARIABLE_NAMES,
    COLLAPSE_FUNCTION,
    COLORMAP_LINEAR_Y_LINEAR_Z,
    COLORMAP_LINEAR_Y_LOG_Z,
    COLORMAP_LOG_Y_LINEAR_Z,
    COLORMAP_LOG_Y_LOG_Z,
    DEFAULT_ZOOM_WINDOW_MINUTES,
    FILTERED_ORBITS_CSV_PATH,
    OUTPUT_BASE_DIRECTORY,
    PLOTTING_PROGRESS_JSON_PATH,
)
from configurable_spectrograms.generic_batch import generic_batch_plot
from configurable_spectrograms.logging_utils import (
    configure_log_batch,
    log_error,
    log_message,
)
from configurable_spectrograms.plotting import (
    close_all_axes_and_clear,
    generic_plot_multirow_optional_zoom,
    generic_plot_spectrogram_set,
    make_spectrogram,
)

__all__ = [
    "CDF_DATA_DIRECTORY",
    "CDF_VARIABLE_NAMES",
    "COLLAPSE_FUNCTION",
    "COLORMAP_LINEAR_Y_LINEAR_Z",
    "COLORMAP_LINEAR_Y_LOG_Z",
    "COLORMAP_LOG_Y_LINEAR_Z",
    "COLORMAP_LOG_Y_LOG_Z",
    "DEFAULT_ZOOM_WINDOW_MINUTES",
    "FILTERED_ORBITS_CSV_PATH",
    "OUTPUT_BASE_DIRECTORY",
    "PLOTTING_PROGRESS_JSON_PATH",
    "close_all_axes_and_clear",
    "configure_log_batch",
    "generic_batch_plot",
    "generic_plot_multirow_optional_zoom",
    "generic_plot_spectrogram_set",
    "get_cdf_file_type",
    "get_cdf_var_shapes",
    "get_timestamps_for_orbit",
    "get_variable_shape",
    "load_filtered_orbits",
    "log_error",
    "log_message",
    "make_spectrogram",
]
