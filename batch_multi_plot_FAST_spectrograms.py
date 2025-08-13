#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plots a folder of FAST ESA data as spectrograms

Assumed folder layout is {FAST_CDF_DATA_FOLDER_PATH}/year/month
Filenames in the month folders assumed to be in the following formats:
    {??}_{??}_{??}_{instrument}_{timestamp}_{orbit}_v02.cdf      (known "instruments" are ees, eeb, ies, or ieb)
    {??}_{??}_orb_{orbit}_{??}.cdf
Examples:
    FAST_data/2000/01/fa_esa_l2_eeb_20000101001737_13312_v02.cdf
    FAST_data/2000/01/fa_k0_orb_13312_v01.cdf
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
__version__: str = "0.0.1"
__license__: str = "GPL-3.0"

import signal
import os
import sys
import gc
import concurrent.futures
import json
from collections import defaultdict, deque
import math
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
import traceback
import numpy as np
import cdflib
import time as _time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Import only required helpers from generic spectrogram module (avoid wildcard import for linting clarity)
from batch_multi_plot_spectrogram import (
    load_filtered_orbits,
    get_cdf_file_type,
    get_timestamps_for_orbit,
    generic_plot_multirow_optional_zoom,
    close_all_axes_and_clear,
    log_error,
    log_message,
    DEFAULT_ZOOM_WINDOW_MINUTES,
)

# FAST-specific paths (renamed for FAST batch)
FAST_CDF_DATA_FOLDER_PATH = "./FAST_data/"
FAST_FILTERED_ORBITS_CSV_PATH = "./FAST_Cusp_Indices.csv"
FAST_PLOTTING_PROGRESS_JSON = "./batch_multi_plot_FAST_progress.json"
FAST_LOGFILE_PATH = "./batch_multi_plot_FAST_log.log"
FAST_OUTPUT_BASE = "./FAST_plots/"
FAST_LOGFILE_DATETIME_PATH = "./batch_multi_plot_FAST_logfile_datetime.txt"
if os.path.exists(FAST_LOGFILE_DATETIME_PATH):
    with open(FAST_LOGFILE_DATETIME_PATH, "r") as f:
        FAST_LOGFILE_DATETIME_STRING = f.read().strip()
    if not FAST_LOGFILE_DATETIME_STRING:
        FAST_LOGFILE_DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(FAST_LOGFILE_DATETIME_PATH, "w") as f:
            f.write(FAST_LOGFILE_DATETIME_STRING)
else:
    FAST_LOGFILE_DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(FAST_LOGFILE_DATETIME_PATH, "w") as f:
        f.write(FAST_LOGFILE_DATETIME_STRING)
FAST_COLLAPSE_FUNCTION = np.nansum
CDF_VARIABLES = ["time_unix", "data", "energy", "pitch_angle"]

# Colormaps for different axis scaling combinations (colorblind-friendly and visually distinct)
DEFAULT_COLORMAP_LINEAR_Y_LINEAR_Z = "viridis"
DEFAULT_COLORMAP_LINEAR_Y_LOG_Z = "cividis"
DEFAULT_COLORMAP_LOG_Y_LINEAR_Z = "plasma"
DEFAULT_COLORMAP_LOG_Y_LOG_Z = "inferno"


# Buffered logging configuration (batching to reduce I/O)
INFO_LOG_BATCH_SIZE_DEFAULT = 10  # fallback default
_INFO_LOG_BATCH_SIZE = INFO_LOG_BATCH_SIZE_DEFAULT
_INFO_LOG_BUFFER: List[Tuple[str, str]] = []  # (level, message)


def configure_info_logger_batch(batch_size: int) -> None:
    """Configure the batch size for buffered info logging.

    Parameters
    ----------
    batch_size : int
        Number of log entries to accumulate before an automatic flush.
        Values < 1 disable buffering (flush every call).
    """
    global _INFO_LOG_BATCH_SIZE
    if batch_size < 1:
        _INFO_LOG_BATCH_SIZE = 1
    else:
        _INFO_LOG_BATCH_SIZE = batch_size


def flush_info_logger_buffer(force: bool = True) -> None:
    """Flush any buffered info/error log messages immediately.

    Parameters
    ----------
    force : bool, default True
        Present for future extensibility; currently ignored (always flushes).
    """
    global _INFO_LOG_BUFFER
    if not _INFO_LOG_BUFFER:
        return
    # Emit each buffered message using underlying logger functions directly
    for level, msg in _INFO_LOG_BUFFER:
        try:
            if level == "error":
                try:
                    log_error(msg)
                except Exception as buffered_error_emit_exception:
                    print(msg, file=sys.stderr)
            else:
                try:
                    log_message(msg)
                except Exception as buffered_message_emit_exception:
                    print(msg)
        except Exception as buffered_flush_loop_exception:
            # Suppress any failure in flush loop, continue with remaining entries
            pass
    _INFO_LOG_BUFFER = []


# Section: Logging & Helper Functions
def info_logger(
    prefix: str,
    exception: Optional[BaseException] = None,
    level: str = "error",
    include_trace: bool = False,
    force_flush: bool = False,
) -> None:
    """Unified logger for messages and exceptions.

    This helper formats a message with an optional exception, includes the
    exception class name when provided, and delegates to generic logging
    helpers from the base module (``log_message``/``log_error``). When
    ``include_trace`` is True and an exception is given, a traceback is also
    emitted.

    Parameters
    ----------
    prefix : str
        Human-readable message prefix for the log line.
    exception : BaseException or None
        Optional exception instance. If ``None``, only ``prefix`` is logged.
    level : {'error', 'message'}, default 'error'
        Logging level. ``'error'`` routes to ``log_error``; otherwise to
        ``log_message``.
    include_trace : bool, default False
        When ``True`` and ``exception`` is not ``None``, include a formatted
        traceback after the primary log message.

    Parameters
    ----------
    force_flush : bool, default False
        When True, forces an immediate flush of the buffered log messages
        (including the current one) regardless of batch size.

    Returns
    -------
    None

    Notes
    -----
    If the underlying logging helpers fail (e.g., misconfigured), this
    function falls back to printing to stdout/stderr to avoid silent loss. When
    an exception is provided, messages are formatted as
    ``"{prefix} [<ExceptionClass>]: {exception}"``.
    """
    if exception is None:
        message = str(prefix)
    else:
        try:
            name = getattr(exception, "__class__", type(exception)).__name__
        except Exception as exception_name_introspection_exception:
            name = "Exception"
        message = f"{prefix} [{name}]: {exception}"

    # Prepare trace (as separate buffered messages) if requested
    trace_lines: List[str] = []
    if include_trace and exception is not None:
        try:
            trace = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
            trace_lines.append("[TRACE]\n" + trace)
        except Exception as trace_format_exception:
            pass

    # Buffer message
    try:
        _INFO_LOG_BUFFER.append((level, message))
        for tr in trace_lines:
            _INFO_LOG_BUFFER.append(("message", tr))
        # Decide flush
        if (
            force_flush
            or _INFO_LOG_BATCH_SIZE <= 1
            or len(_INFO_LOG_BUFFER) >= _INFO_LOG_BATCH_SIZE
        ):
            flush_info_logger_buffer(force=True)
    except Exception as info_logger_buffer_append_exception:
        # On any buffering failure, attempt direct emission
        try:
            if level == "error":
                log_error(message)
            else:
                log_message(message)
        except Exception as info_logger_direct_emit_exception:
            try:
                print(message)
            except Exception as info_logger_print_fallback_exception:
                pass


def _terminate_all_child_processes() -> None:
    """Attempt to terminate all child processes of the current process.

    Uses ``psutil``, if available, to iterate over child processes recursively
    and call ``terminate()`` on each. Errors are suppressed, as this is
    typically invoked during shutdown.

    Returns
    -------
    None

    Notes
    -----
    This is best-effort and does not guarantee exit. Callers may follow with
    stronger measures (e.g., ``kill``) after a brief grace period.
    """
    try:
        import psutil
    except Exception as psutil_import_exception:
        return
    try:
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.terminate()
            except Exception as child_terminate_exception:
                pass
    except Exception as psutil_process_iter_exception:
        pass


def FAST_plot_pitch_angle_grid(
    cdf_file_path: str,
    filtered_orbits_df=None,
    orbit_number: Optional[int] = None,
    zoom_duration_minutes: float = 6.25,
    scale_function_y: str = "linear",
    scale_function_z: str = "linear",
    pitch_angle_categories: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    show: bool = True,
    colormap: str = "viridis",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Tuple[Any, Any]:
    """Plot a grid of ESA spectrograms collapsed by pitch-angle categories.

    Each row corresponds to a pitch-angle category (e.g., downgoing, upgoing,
    perpendicular, all). If orbit boundary timestamps are available for this
    instrument/orbit, a zoom column is added. Data are loaded from the CDF,
    oriented so that energy is the y-axis, and collapsed over pitch-angle via
    ``FAST_COLLAPSE_FUNCTION`` (``np.nansum`` by default).

    Parameters
    ----------
    cdf_file_path : str
        Path to the instrument CDF file.
    filtered_orbits_df : pandas.DataFrame or None
        DataFrame used to compute vertical lines for the ``orbit_number``.
        If ``None``, vertical lines are omitted.
    orbit_number : int or None
        Orbit number used to compute and label vertical lines.
    zoom_duration_minutes : float, default 6.25
        Window length (minutes) for the optional zoom column.
    scale_function_y : {'linear', 'log'}, default 'linear'
        Y-axis scaling for the spectrogram.
    scale_function_z : {'linear', 'log'}, default 'linear'
        Color scale for the spectrogram intensity.
    pitch_angle_categories : dict or None
        Mapping of label -> list of (min_deg, max_deg) ranges; if ``None``,
        defaults to standard four groups.
    show : bool, default True
        If ``True``, display the plot; otherwise render off-screen.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    y_min, y_max : float or None, optional
        Optional explicit energy (y-axis) limits override. When ``None`` the
        default lower bound (0) and observed upper bound (<=4000) subset is
        used. These overrides are typically provided by precomputed global
        extrema.
    z_min, z_max : float or None, optional
        Optional explicit color (intensity) scale limits. When ``None`` the
        1st / 99th percentiles per row are used (robust to outliers). When
        provided (e.g., global extrema), they are applied uniformly across
        rows.

    Returns
    -------
    tuple[Figure or None, FigureCanvasBase or None]
        Figure and Canvas for the grid, or ``(None, None)`` if no datasets.

    Notes
    -----
    - Energy bins are filtered to ``[0, 4000]`` (or explicit ``y_min`` / ``y_max``).
    - ``vmin``/``vmax`` (row color bounds) are derived from 1st/99th percentiles
        unless explicit ``z_min`` / ``z_max`` provided.
    - Each dataset row includes ``y_label='Energy (eV)'`` and ``z_label='Counts'``;
        modify after return if alternative units are desired.
    - When no pitch-angle category yields data, the function logs a message and
        returns ``(None, None)``.
    """
    # TODO: record orbits when error contains "is not a CDF file or a non-supported CDF!" in json log file
    if pitch_angle_categories is None:
        pitch_angle_categories = {
            "downgoing\n(0, 30), (330, 360)": [(0, 30), (330, 360)],
            "upgoing\n(150, 210)": [(150, 210)],
            "perpendicular\n(40, 140), (210, 330)": [(40, 140), (210, 330)],
            "all\n(0, 360)": [(0, 360)],
        }
    instrument_type = get_cdf_file_type(cdf_file_path)
    cdf_file = cdflib.CDF(cdf_file_path)
    times = np.asarray(cdf_file.varget(CDF_VARIABLES[0]))
    data = np.asarray(cdf_file.varget(CDF_VARIABLES[1]))
    energy_full = np.asarray(cdf_file.varget(CDF_VARIABLES[2]))
    pitchangle_full = np.asarray(cdf_file.varget(CDF_VARIABLES[3]))
    energy = energy_full[0, 0, :] if energy_full.ndim == 3 else energy_full
    pitchangle = (
        pitchangle_full[0, :, 0] if pitchangle_full.ndim == 3 else pitchangle_full
    )
    if data.shape[1] == len(energy) and data.shape[2] == len(pitchangle):
        data = np.transpose(data, (0, 2, 1))
    vertical_lines = None
    if filtered_orbits_df is not None and orbit_number is not None:
        vertical_lines = get_timestamps_for_orbit(
            filtered_orbits_df, orbit_number, instrument_type, times
        )
        if (vertical_lines is None) or (len(vertical_lines) == 0):
            info_logger(
                f"No vertical lines found for orbit {orbit_number} in {cdf_file_path}. Skipping.",
                level="message",
            )
    pa_keys = [
        "all\n(0, 360)",
        "downgoing\n(0, 30), (330, 360)",
        "upgoing\n(150, 210)",
        "perpendicular\n(40, 140), (210, 330)",
    ]
    datasets = []
    for key in pa_keys:
        mask = np.zeros_like(pitchangle, dtype=bool)
        for rng in pitch_angle_categories[key]:
            mask |= (pitchangle >= rng[0]) & (pitchangle <= rng[1])
        pa_data = data[:, mask, :]
        matrix_full = FAST_COLLAPSE_FUNCTION(pa_data, axis=1)
        nan_col_mask = ~np.all(np.isnan(matrix_full), axis=0)
        # Apply energy (y-axis) limits; default 0-4000 if not overridden
        y_lower = 0 if y_min is None else y_min
        y_upper = 4000 if y_max is None else y_max
        valid_energy_mask = (energy >= y_lower) & (energy <= y_upper)
        combined_mask = nan_col_mask & valid_energy_mask
        matrix_full = matrix_full[:, combined_mask]
        matrix_full_plot = matrix_full.T
        if matrix_full_plot.size == 0:
            continue
        # Color (z-axis) min/max percentiles unless overridden
        if z_min is None:
            vmin = np.nanpercentile(matrix_full_plot, 1)
        else:
            vmin = z_min
        if z_max is None:
            vmax = np.nanpercentile(matrix_full_plot, 99)
        else:
            vmax = z_max
        # Include per-row y/z overrides so downstream generic grid honors them
        datasets.append(
            {
                "x": times,
                "y": energy,
                "data": pa_data,
                "label": key.title(),
                "y_label": "Energy (eV)",
                "z_label": "Counts",
                "vmin": vmin,  # color range (row-specific)
                "vmax": vmax,
                "y_min": y_lower,
                "y_max": y_upper,
                # z_min/z_max are not repeated unless provided explicitly to avoid
                # forcing identical bounds when percentile scaling applied
                **({"z_min": z_min} if z_min is not None else {}),
                **({"z_max": z_max} if z_max is not None else {}),
            }
        )
    if not datasets:
        info_logger(
            f"[WARNING] No pitch angle datasets to plot for {cdf_file_path}.",
            level="message",
        )
        return None, None
    title = f"Orbit {orbit_number} - Pitch Angle {instrument_type} ESA Spectrograms"
    return generic_plot_multirow_optional_zoom(
        datasets,
        vertical_lines=vertical_lines,
        zoom_duration_minutes=zoom_duration_minutes,
        y_scale=scale_function_y,
        z_scale=scale_function_z,
        colormap=colormap,
        show=show,
        title=title,
        row_label_pad=50,
        row_label_rotation=90,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )


def FAST_plot_instrument_grid(
    cdf_file_paths: Dict[str, str],
    filtered_orbits_df=None,
    orbit_number: Optional[int] = None,
    zoom_duration_minutes: float = 6.25,
    scale_function_y: str = "linear",
    scale_function_z: str = "linear",
    instrument_order: Tuple[str, ...] = ("ees", "eeb", "ies", "ieb"),
    show: bool = True,
    colormap: str = "viridis",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    global_extrema: Optional[Dict[str, Union[int, float]]] = None,
) -> Tuple[Any, Any]:
    """Plot a multi-instrument ESA spectrogram grid for a single orbit.

    Loads each instrument CDF, orients and filters the data, collapses across
    pitch-angle, and constructs datasets for
    ``generic_plot_multirow_optional_zoom``. When vertical lines are available
    for the orbit, a zoom column is included.

    Parameters
    ----------
    cdf_file_paths : dict of {str: str}
        Mapping of instrument key (``'ees'``, ``'eeb'``, ``'ies'``, ``'ieb'``)
        to CDF file path. Missing instruments are skipped.
    filtered_orbits_df : pandas.DataFrame or None
        DataFrame for vertical line computation; if ``None``, lines are omitted.
    orbit_number : int or None
        Orbit identifier used in titles/lines.
    zoom_duration_minutes : float, default 6.25
        Zoom window length (minutes).
    scale_function_y : {'linear', 'log'}, default 'linear'
        Y-axis scaling.
    scale_function_z : {'linear', 'log'}, default 'linear'
        Color scale for intensity.
    instrument_order : tuple of str, default ('ees', 'eeb', 'ies', 'ieb')
        Display order of instrument rows.
    show : bool, default True
        Whether to show the figure interactively.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    y_min, y_max, z_min, z_max : float or None, optional
        Global fallback overrides for axis/color limits. Per-instrument
        overrides from ``global_extrema`` take precedence; finally row-level
        percentile scaling is used when neither is provided.
    global_extrema : dict or None
        Mapping containing precomputed extrema keys (``{instrument}_{y_scale}_{z_scale}_{axis}_{min|max}``) used
        to supply per-instrument (row-specific) limits. This enables distinct
        y/z ranges for ``ees``, ``eeb``, ``ies``, and ``ieb`` within the same
        figure, improving contrast when dynamic ranges differ.
    y_min, y_max : float or None, optional
        Direct energy bounds override applied when ``global_extrema`` is not
        provided. Ignored per-instrument when ``global_extrema`` supplies
        instrument-specific keys.
    z_min, z_max : float or None, optional
        Direct intensity scale overrides (see above re: ``global_extrema``).
    global_extrema : dict or None, optional
        Mapping containing precomputed extrema keys of the form
        ``{instrument}_{y_scale}_{z_scale}_{axis}_{min|max}``. When present
        these take precedence over ``y_min`` / ``y_max`` / ``z_min`` /
        ``z_max``.

    Returns
    -------
    tuple[Figure or None, FigureCanvasBase or None]
        Figure and Canvas, or ``(None, None)`` if no datasets.

        Notes
        -----
        - Files that fail to load are logged and skipped; remaining instruments may
            still render.
        - Energy bins are restricted to ``[0, 4000]`` (or overridden via ``global_extrema``
            or explicit ``y_min`` / ``y_max``).
        - ``vmin``/``vmax`` per row use 1st/99th percentiles for robust scaling unless
            ``global_extrema`` provides per-instrument ``z_min`` / ``z_max``.
        - Each dataset row sets ``y_label='Energy (eV)'`` and ``z_label='Counts'`` by
            default for clarity of physical units.
    """
    datasets = []
    vertical_lines = None
    first_times = None
    for inst in instrument_order:
        cdf_path = cdf_file_paths.get(inst)
        if not cdf_path:
            continue
        try:
            cdf_file = cdflib.CDF(cdf_path)
            times = np.asarray(cdf_file.varget(CDF_VARIABLES[0]))
            data = np.asarray(cdf_file.varget(CDF_VARIABLES[1]))
            energy_full = np.asarray(cdf_file.varget(CDF_VARIABLES[2]))
            energy = energy_full[0, 0, :] if energy_full.ndim == 3 else energy_full
            pitchangle_full = np.asarray(cdf_file.varget(CDF_VARIABLES[3]))
            pitchangle = (
                pitchangle_full[0, :, 0]
                if pitchangle_full.ndim == 3
                else pitchangle_full
            )
            if data.shape[1] == len(energy) and data.shape[2] == len(pitchangle):
                data = np.transpose(data, (0, 2, 1))
            if first_times is None:
                first_times = times
            if (
                vertical_lines is None
                and filtered_orbits_df is not None
                and orbit_number is not None
            ):
                instrument_type = get_cdf_file_type(cdf_path)
                vertical_lines = get_timestamps_for_orbit(
                    filtered_orbits_df, orbit_number, instrument_type, times
                )
                if (vertical_lines is None) or (len(vertical_lines) == 0):
                    info_logger(
                        f"No vertical lines found for orbit {orbit_number} in {cdf_path}. Skipping.",
                        level="message",
                    )
            matrix_full = FAST_COLLAPSE_FUNCTION(data, axis=1)
            nan_col_mask = ~np.all(np.isnan(matrix_full), axis=0)
            # Determine instrument-specific y bounds (energy)
            if isinstance(global_extrema, dict):
                key_prefix = f"{inst}_{scale_function_y}_{scale_function_z}"
                y_lower = global_extrema.get(
                    f"{key_prefix}_y_min", 0 if y_min is None else y_min
                )
                y_upper = global_extrema.get(
                    f"{key_prefix}_y_max", 4000 if y_max is None else y_max
                )
            else:
                y_lower = 0 if y_min is None else y_min
                y_upper = 4000 if y_max is None else y_max
            valid_energy_mask = (energy >= y_lower) & (energy <= y_upper)
            combined_mask = nan_col_mask & valid_energy_mask
            matrix_full = matrix_full[:, combined_mask]
            matrix_full_plot = matrix_full.T
            if matrix_full_plot.size == 0:
                continue
            # Determine instrument-specific z bounds (intensity)
            if isinstance(global_extrema, dict):
                key_prefix = f"{inst}_{scale_function_y}_{scale_function_z}"
                vmin = global_extrema.get(f"{key_prefix}_z_min")
                vmax = global_extrema.get(f"{key_prefix}_z_max")
                if vmin is None:
                    vmin = np.nanpercentile(matrix_full_plot, 1)
                if vmax is None:
                    vmax = np.nanpercentile(matrix_full_plot, 99)
            else:
                if z_min is None:
                    vmin = np.nanpercentile(matrix_full_plot, 1)
                else:
                    vmin = z_min
                if z_max is None:
                    vmax = np.nanpercentile(matrix_full_plot, 99)
                else:
                    vmax = z_max
            # Provide per-row overrides so generic multi-row plot can honor
            # distinct instrument ranges for both y (energy) and z (intensity).
            datasets.append(
                {
                    "x": times,
                    "y": energy,
                    "data": data,
                    "label": inst.upper(),
                    "y_label": "Energy (eV)",
                    "z_label": "Counts",
                    "vmin": vmin,
                    "vmax": vmax,
                    "y_min": y_lower,
                    "y_max": y_upper,
                    # Only include z_min/z_max if explicitly fixed by global extrema
                    **({"z_min": z_min} if z_min is not None else {}),
                    **({"z_max": z_max} if z_max is not None else {}),
                }
            )
        except Exception as file_load_failure:
            info_logger(
                f"Failed to load CDF for {inst} at {cdf_path}. Skipping.",
                file_load_failure,
                level="error",
            )
            continue
    if not datasets:
        return None, None
    title = f"Orbit {orbit_number} -  ESA Spectrograms"
    return generic_plot_multirow_optional_zoom(
        datasets,
        vertical_lines=vertical_lines,
        zoom_duration_minutes=zoom_duration_minutes,
        y_scale=scale_function_y,
        z_scale=scale_function_z,
        colormap=colormap,
        show=show,
        title=title,
        row_label_pad=50,
        row_label_rotation=90,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )

    # (Residual code from previous implementation removed in refactor.)


def FAST_process_single_orbit(
    orbit_number: int,
    instrument_file_paths: Dict[str, str],
    filtered_orbits_dataframe,
    zoom_duration_minutes: float,
    y_axis_scale: str,
    z_axis_scale: str,
    instrument_order: Tuple[str, ...],
    colormap: str,
    output_base_directory: str,
    orbit_timeout_seconds: Union[int, float] = 60,
    instrument_timeout_seconds: Union[int, float] = 30,
    global_extrema: Optional[Dict[str, Union[int, float]]] = None,
) -> Dict[str, Any]:
    """Process all plots for a single orbit with timeouts and deferred saving.

    For each available instrument, renders a pitch-angle grid, then renders a
    combined instrument grid. Figures are accumulated in memory and saved only
    if no timeout thresholds are exceeded.

    Parameters
    ----------
    orbit_number : int
        The orbit identifier.
    instrument_file_paths : dict of {str: str}
        Mapping of instrument key to CDF file path.
    filtered_orbits_dataframe : pandas.DataFrame
        DataFrame used to compute orbit boundary timestamps.
    zoom_duration_minutes : float
        Zoom window length for zoomed plots.
    y_axis_scale : {'linear', 'log'}
        Y-axis scaling.
    z_axis_scale : {'linear', 'log'}
        Color scale for intensity.
    instrument_order : tuple of str
        Order used in the instrument grid.
    colormap : str
        Matplotlib colormap.
    output_base_directory : str
        Root folder for saving figures; year/month are inferred from the CDF
        path when possible, else ``'unknown'``.
    orbit_timeout_seconds : int or float, default 60
        Maximum wall-clock seconds permitted for the entire orbit processing (summed).
    instrument_timeout_seconds : int or float, default 30
        Per-instrument rendering timeout; exceeded instruments are skipped and noted.
    global_extrema : dict or None
        Precomputed extrema mapping used to supply uniform axis limits to all
        instrument plots for deterministic scaling; produced by
        ``compute_global_extrema``.
    orbit_timeout_seconds : int or float, default 60
        Max total time for this orbit; if exceeded, status becomes ``'timeout'``
        and no figures are saved.
    instrument_timeout_seconds : int or float, default 30
        Max time per instrument (and for the instrument grid). Exceeding this
        aborts the orbit without saving.
    global_extrema : dict or None, optional
        Precomputed extrema dictionary (see ``compute_global_extrema``) used
        to supply consistent per-instrument axis limits across all orbits.

    Returns
    -------
    dict
        Result dictionary with keys:
        ``orbit`` (int), ``status`` (``'ok'``, ``'error'``, or ``'timeout'``),
        ``errors`` (list of str). On timeout, includes ``timeout_type`` and
        ``timeout_instrument`` when applicable.

    Notes
    -----
    - Timing diagnostics are logged per instrument and for the grid.
    - Exceptions during plotting/saving are logged; processing continues when
      safe. Figures are closed in all cases to free memory.
    """
    result = {"orbit": orbit_number, "status": "ok", "errors": []}
    orbit_start_time = _time.time()
    pending_figures = []  # defer saving until timeouts cleared
    timeout_triggered = False
    timeout_type = None
    timeout_instrument = None

    try:
        # Derive year/month for output path
        year = "unknown"
        month = "unknown"
        first_path = next(
            (
                instrument_file_paths[k]
                for k in ("ees", "eeb", "ies", "ieb")
                if k in instrument_file_paths
            ),
            None,
        )
        if first_path:
            try:
                parts = Path(first_path).parts
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 4:
                        year = part
                        if (
                            i + 1 < len(parts)
                            and parts[i + 1].isdigit()
                            and len(parts[i + 1]) == 2
                        ):
                            month = parts[i + 1]
                        break
            except Exception as year_month_parse_exception:
                info_logger(
                    "[WARN] Could not parse year/month",
                    year_month_parse_exception,
                    level="message",
                )
        output_dir = os.path.join(
            output_base_directory, str(year), str(month), str(orbit_number)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Per-instrument processing
        for inst_type in ("ees", "eeb", "ies", "ieb"):
            if timeout_triggered:
                break
            cdf_path = instrument_file_paths.get(inst_type)
            if not cdf_path:
                continue
            inst_start = _time.time()
            try:
                inst_detected = get_cdf_file_type(cdf_path)
                if inst_detected is None or inst_detected == "orb":
                    continue
                cdf_obj = cdflib.CDF(cdf_path)
                time_unix_array = np.asarray(cdf_obj.varget("time_unix"))
                vertical_lines = get_timestamps_for_orbit(
                    filtered_orbits_dataframe,
                    orbit_number,
                    inst_detected,
                    time_unix_array,
                )
                # Lookup global extrema overrides
                y_min_override = None
                y_max_override = None
                z_min_override = None
                z_max_override = None
                if isinstance(global_extrema, dict):
                    key_base = f"{inst_detected}_{y_axis_scale}_{z_axis_scale}"
                    y_min_override = global_extrema.get(f"{key_base}_y_min")
                    y_max_override = global_extrema.get(f"{key_base}_y_max")
                    z_min_override = global_extrema.get(f"{key_base}_z_min")
                    z_max_override = global_extrema.get(f"{key_base}_z_max")
                fig_pa, canvas_pa = FAST_plot_pitch_angle_grid(
                    cdf_path,
                    filtered_orbits_df=filtered_orbits_dataframe,
                    orbit_number=orbit_number,
                    zoom_duration_minutes=zoom_duration_minutes,
                    scale_function_y=y_axis_scale,
                    scale_function_z=z_axis_scale,
                    show=False,
                    colormap=colormap,
                    y_min=y_min_override,
                    y_max=y_max_override,
                    z_min=z_min_override,
                    z_max=z_max_override,
                )
                if fig_pa is not None:
                    cusp = vertical_lines is not None and len(vertical_lines) > 0
                    filename = f"{orbit_number}{'_cusp' if cusp else ''}_pitch-angle_ESA_{inst_detected}_y-{y_axis_scale}_z-{z_axis_scale}.png"
                    pending_figures.append(
                        {
                            "figure": fig_pa,
                            "canvas": canvas_pa,
                            "path": os.path.join(output_dir, filename),
                            "desc": f"pitch-angle {inst_detected}",
                        }
                    )
            except Exception as instrument_exception:
                err = f"[FAIL] Plotting Orbit {orbit_number} pitch angle grid for {inst_type}"
                info_logger(err, instrument_exception, level="error")
                result["status"] = "error"
                result.setdefault("errors", []).append(err)
                continue
            finally:
                inst_elapsed = _time.time() - inst_start
                info_logger(
                    f"[TIMING] Orbit {orbit_number} instrument {inst_type} elapsed {inst_elapsed:.3f}s",
                    level="message",
                )
                if inst_elapsed > instrument_timeout_seconds and not timeout_triggered:
                    timeout_triggered = True
                    timeout_type = "instrument"
                    timeout_instrument = inst_type
                    info_logger(
                        f"[TIMEOUT] Instrument {inst_type} in orbit {orbit_number} exceeded {instrument_timeout_seconds:.0f}s ({inst_elapsed:.2f}s). Aborting orbit without saving.",
                        level="message",
                    )
                    break

        # Instrument grid (if still OK)
        grid_elapsed = None
        if not timeout_triggered:
            grid_start = _time.time()
            try:
                # Provide global_extrema dict so per-instrument limits are applied inside grid helper
                fig_grid, canvas_grid = FAST_plot_instrument_grid(
                    instrument_file_paths,
                    filtered_orbits_df=filtered_orbits_dataframe,
                    orbit_number=orbit_number,
                    zoom_duration_minutes=zoom_duration_minutes,
                    scale_function_y=y_axis_scale,
                    scale_function_z=z_axis_scale,
                    instrument_order=instrument_order,
                    show=False,
                    colormap=colormap,
                    global_extrema=global_extrema,
                )
                if fig_grid is not None:
                    pending_figures.append(
                        {
                            "figure": fig_grid,
                            "canvas": canvas_grid,
                            "path": os.path.join(
                                output_dir,
                                f"{orbit_number}_instrument-grid_ESA_y-{y_axis_scale}_z-{z_axis_scale}.png",
                            ),
                            "desc": "instrument-grid",
                        }
                    )
            except Exception as instrument_grid_exception:
                err = f"[FAIL] Plotting Orbit {orbit_number} instrument grid"
                info_logger(err, instrument_grid_exception, level="error")
                result["status"] = "error"
                result.setdefault("errors", []).append(err)
            finally:
                grid_elapsed = _time.time() - grid_start
                info_logger(
                    f"[TIMING] Orbit {orbit_number} instrument-grid elapsed {grid_elapsed:.3f}s",
                    level="message",
                )
                if (
                    grid_elapsed is not None
                    and grid_elapsed > instrument_timeout_seconds
                    and not timeout_triggered
                ):
                    timeout_triggered = True
                    timeout_type = "instrument"
                    timeout_instrument = "instrument_grid"
                    info_logger(
                        f"[TIMEOUT] Instrument grid in orbit {orbit_number} exceeded {instrument_timeout_seconds:.0f}s ({grid_elapsed:.2f}s). Aborting orbit without saving.",
                        level="message",
                    )

        # Orbit total timeout check
        orbit_elapsed = _time.time() - orbit_start_time
        if orbit_elapsed > orbit_timeout_seconds and not timeout_triggered:
            timeout_triggered = True
            timeout_type = "orbit"
            info_logger(
                f"[TIMEOUT] Orbit {orbit_number} exceeded {orbit_timeout_seconds:.0f}s total ({orbit_elapsed:.2f}s). Aborting without saving.",
                level="message",
            )

        # If timeout -> discard pending figures
        if timeout_triggered:
            for fig_item in pending_figures:
                try:
                    close_all_axes_and_clear(fig_item["figure"])
                except Exception as pitch_angle_energy_transpose_exception:
                    pass
            pending_figures.clear()
            result["status"] = "timeout"
            result["timeout_type"] = timeout_type
            if timeout_instrument:
                result["timeout_instrument"] = timeout_instrument
            return result

        # Save figures now
        for fig_item in pending_figures:
            fig = fig_item["figure"]
            fpath = fig_item["path"]
            try:
                info_logger(
                    f"[DEBUG] Saving {fig_item['desc']} plot: y_axis_scale={y_axis_scale}, z_axis_scale={z_axis_scale}, filename={fpath}",
                    level="message",
                )
                fig.savefig(fpath, dpi=200)
                info_logger(f"[SAVED] {fpath}", level="message")
            except Exception as save_exception:
                info_logger(
                    f"[FAIL] Saving figure {fpath}", save_exception, level="error"
                )
                result["status"] = "error"
                result.setdefault("errors", []).append(str(save_exception))
            finally:
                try:
                    close_all_axes_and_clear(fig)
                except Exception as pitch_angle_percentile_exception:
                    pass
        pending_figures.clear()
    except Exception as orbit_processing_exception:
        err = f"[FAIL] Orbit {orbit_number} processing"
        info_logger(err, orbit_processing_exception, level="error")
        result["status"] = "error"
        result.setdefault("errors", []).append(err)
    finally:
        gc.collect()
    return result


def FAST_plot_spectrograms_directory(
    directory_path: str = FAST_CDF_DATA_FOLDER_PATH,
    output_base: str = FAST_OUTPUT_BASE,
    y_scale: str = "linear",
    z_scale: str = "log",
    zoom_duration_minutes: float = DEFAULT_ZOOM_WINDOW_MINUTES,
    instrument_order: Tuple[str, ...] = ("ees", "eeb", "ies", "ieb"),
    verbose: bool = True,
    progress_json_path: Optional[str] = FAST_PLOTTING_PROGRESS_JSON,
    ignore_progress_json: bool = False,
    use_tqdm: Optional[bool] = None,
    colormap: str = "viridis",
    max_workers: int = 4,
    orbit_timeout_seconds: Union[int, float] = 60,
    instrument_timeout_seconds: Union[int, float] = 30,
    retry_timeouts: bool = True,
    flush_batch_size: int = 10,
    log_flush_batch_size: Optional[int] = None,
    max_processing_percentile: float = 95.0,
) -> List[Dict[str, Any]]:
    """Batch process ESA spectrogram plots for all orbits in a directory.

    Discovers instrument CDF files (excluding filenames containing
    ``"_orb_"``), groups them by orbit number, and processes each orbit in
    parallel worker processes for matplotlib safety. Progress is persisted to a
    JSON file to support resume and to record error/timeout orbits per y/z
    scale combo. Before scheduling orbit tasks a global extrema pass runs
    (``compute_global_extrema``) for the requested (y_scale, z_scale). For
    log scales, if a completed linear counterpart (same other-axis scale)
    exists, its maxima are reused and log-transformed with a floor (see
    ``log_floor_cutoff`` / ``log_floor_value``) instead of rescanning files.

    Parameters
    ----------
    directory_path : str, default FAST_CDF_DATA_FOLDER_PATH
        Root folder containing CDF files.
    output_base : str, default FAST_OUTPUT_BASE
        Base output directory for plots organized as
        ``output_base/year/month/orbit``.
    y_scale : {'linear', 'log'}, default 'linear'
        Y-axis scaling.
    z_scale : {'linear', 'log'}, default 'log'
        Color scale for intensity.
    zoom_duration_minutes : float, default DEFAULT_ZOOM_WINDOW_MINUTES
        Zoom window length for zoom columns.
    instrument_order : tuple of str, default ('ees', 'eeb', 'ies', 'ieb')
        Display order for instrument grid.
    verbose : bool, default True
        If ``True``, prints additional batch messages.
    progress_json_path : str or None, default FAST_PLOTTING_PROGRESS_JSON
        Path to persist progress across runs.
    ignore_progress_json : bool, default False
        If ``True``, don't read previous progress before starting.
    use_tqdm : bool or None, default None
        If ``True``, show a tqdm progress bar; if ``None``, defaults to ``False``.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    max_workers : int, default 4
        Max number of worker processes.
    orbit_timeout_seconds : int or float, default 60
        Total per-orbit timeout.
    instrument_timeout_seconds : int or float, default 30
        Per-instrument/grid timeout.
    retry_timeouts : bool, default True
        If ``True``, retry timed-out orbits once after the initial pass.
    flush_batch_size : int, default 10
        Batch size applied to both extrema JSON and progress JSON writes to
        reduce I/O. Values < 1 are coerced to 1. Final partial batch always flushes.
    log_flush_batch_size : int or None, default None
        Batch size for buffered logging. If None, defaults to ``flush_batch_size``.
    max_processing_percentile : float, default 95.0
        Upper percentile (0 < p <= 100) forwarded to the internal
        ``compute_global_extrema`` helper as ``max_percentile``. Currently this
        percentile controls the pooled positive intensity (Z) maxima selection.
        Energy (Y) maxima continue to use a fixed 99% cumulative positive count
        coverage rule (hard-coded) – i.e., the smallest energy whose cumulative
        positive finite sample count reaches 99% of total positive samples. A
        future refactor may unify these so that ``max_processing_percentile``
        also governs the Y coverage threshold.

    Returns
    -------
    list of dict
        Result dictionaries from ``FAST_process_single_orbit`` (and any
        retries), in no particular order.

    Raises
    ------
    KeyboardInterrupt
        Raised immediately on SIGINT/SIGTERM to stop scheduling and terminate
        workers.

    Notes
    -----
        - Signal handling installs handlers that request shutdown, terminate child
            processes, and raise ``KeyboardInterrupt`` promptly.
        - Progress JSON tracks last completed orbit per scale combo under key
            ``f"progress_{y_scale}_{z_scale}_last_orbit"`` and records error/timeout
            orbits under dedicated keys (including per-instrument).
        - If ``ignore_progress_json`` is ``True``, progress is not read; writes are
            still attempted.
        - If ``retry_timeouts`` is ``True``, timed-out orbits are retried once with
            a smaller pool.
        - ``max_processing_percentile`` only impacts intensity maxima today; Y
            maxima coverage threshold is fixed at 99%.
    """
    # Placeholders for restoring prior signal handlers at function exit (runtime vars, not part of docstring)
    previous_sigint: Any = None
    previous_sigterm: Any = None

    # Section: Early shutdown support
    # A lightweight flag set when SIGINT/SIGTERM received; workers are separate processes so we mainly
    # use this to abort scheduling / wait loops and then terminate children explicitly.
    shutdown_requested = {"flag": False}

    # Section: Global extrema computation (cached)
    def compute_global_extrema(
        directory_path: str,
        y_scale: str,
        z_scale: str,
        instrument_order: Iterable[str],
        extrema_json_path: str = "./FAST_calculated_extrema.json",
        compute_mins: bool = False,
        max_percentile: float = max_processing_percentile,
        log_floor_cutoff: float = 0.1,
        log_floor_value: float = -1.0,
        flush_batch_size: int = 10,
    ) -> Dict[str, Union[int, float, Dict[str, Union[int, float]]]]:
        """Compute (or incrementally update) cached axis extrema per instrument.

            This routine performs a resumable pass over all instrument CDF files
            (excluding those with ``"_orb_"`` in the filename). Processing is
            incremental: after each file the current maxima and progress index
            are flushed to ``extrema_json_path`` so interrupted executions can
            resume without losing prior work.

            Extrema Logic
            -------------
            - Y (energy) minima are fixed to 0 (not computed) unless ``compute_mins`` (linear scale).
            - Linear Y maxima: smallest energy whose cumulative positive finite count reaches
                99% of total positive finite samples (fixed threshold; independent of
                ``max_percentile`` for now).
            - Linear Z maxima: ``max_percentile``th percentile of pooled positive finite intensity
                samples (configurable via outer ``max_processing_percentile``).
            - If either axis is requested in log scale and the corresponding linear-scale extrema
                (same other-axis scale) are already complete (``complete: True`` in their progress key),
                the linear maxima are re-used and transformed via base-10 log without re-scanning files.
            - Log transform applies a floor: any linear value ``<= log_floor_cutoff`` or non-finite
                is replaced by ``log_floor_value`` (default -1). Minima in log scale are set to this
                floor value.
            - If a linear precursor is not available for a requested log axis, a full scan occurs and
                the resulting maxima are then log-transformed with the same floor rule.
            - Maxima are monotonically non-decreasing across incremental updates
                and re-ceiled each iteration (``math.ceil``); energy maxima are
                additionally capped at 4000.
            - Empty / non-positive datasets for a file are skipped silently.

            Resume Strategy
            ---------------
            A progress entry with key ``{instrument}_{y_scale}_{z_scale}_extrema_progress``
            stores ``processed_index``, ``total`` and a ``complete`` flag. Files
            are processed in deterministic (sorted) order; any index less than or
            equal to ``processed_index`` is skipped on subsequent runs.

            Stored JSON Keys
            ----------------
            ``{instrument}_{y_scale}_{z_scale}_y_min`` (always 0)
            ``{instrument}_{y_scale}_{z_scale}_y_max``
            ``{instrument}_{y_scale}_{z_scale}_z_min`` (always 0)
            ``{instrument}_{y_scale}_{z_scale}_z_max``

            Parameters
            ----------
            directory_path : str
                Root directory containing instrument CDF files.
            y_scale : {'linear', 'log'}
                Y scaling label (used for key names only at present).
            z_scale : {'linear', 'log'}
                Z scaling label (used for key names only; percentile selection
                already ignores zeros/NaNs).
            instrument_order : iterable of str
                Instruments to process (e.g., ("ees", "eeb", "ies", "ieb")).
            extrema_json_path : str, default './FAST_calculated_extrema.json'
                Path to the JSON cache file (created if absent).
            max_percentile : float, default value forwarded from the enclosing
                ``FAST_plot_spectrograms_directory`` via ``max_processing_percentile``
                (current default 95.0 there). At present this percentile is applied
                ONLY to the pooled positive intensity (Z) distribution to derive
                ``z_max``. Energy (Y) maxima instead use a fixed 99% cumulative
                positive sample coverage heuristic (independent of this value).
                Future revisions may allow this percentile to also govern the
                energy coverage threshold for consistency.
            log_floor_cutoff : float, default 0.1
                Threshold below which linear-domain values are treated as at/under floor when
                converting to log space (avoids log10 of very small or non-positive values).
            log_floor_value : float, default -1.0
                Value substituted for any log-transformed minima and for maxima that fall
                at or below the cutoff or are non-finite.
            flush_batch_size : int, default 10
                Number of orbits (that produced updates) to accumulate before flushing the
                extrema JSON to disk. Final partial batch is always flushed. Values < 1
                are coerced to 1.

            Returns
            -------
            dict
                Updated mapping containing extrema values and progress entries.

            Notes
            -----
        Previously computed maxima are reused; only instruments missing at least one
        required key are revisited. JSON writes are batched: after every
        ``flush_batch_size`` orbits that produced updates (or at the very end) a flush
        occurs. This reduces I/O frequency at the cost of potentially losing up to
        ``flush_batch_size - 1`` orbits worth of new extrema work if interrupted.
        """
        # Load (or initialise) persistent extrema cache from disk.
        if os.path.exists(extrema_json_path):
            try:
                with open(extrema_json_path, "r") as file_in:
                    extrema_state: Dict[
                        str, Union[int, float, Dict[str, Union[int, float]]]
                    ] = json.load(file_in)
            except Exception as extrema_json_read_exception:
                # Fall back to a fresh state if the file is corrupt / unreadable.
                info_logger(
                    f"[EXTREMA] Failed to read existing extrema JSON '{extrema_json_path}' (starting fresh)",
                    extrema_json_read_exception,
                    level="message",
                )
                extrema_state = {}
        else:
            extrema_state = {}

        # Helper: safe base-10 log transform honoring floor cutoff/value.
        def _safe_log_transform(linear_value: Optional[Union[int, float]]) -> float:
            """Convert a linear-domain positive value to log10 with floor handling.

            Parameters
            ----------
            linear_value : float or int or None
                The original (linear scale) extrema value. If None, non-finite, or
                <= ``log_floor_cutoff`` it is mapped to ``log_floor_value``.

            Returns
            -------
            float
                The transformed log10 value or the configured floor value when the
                input is absent / invalid / below threshold.
            """
            try:
                if linear_value is None:
                    return float(log_floor_value)
                numeric_value = float(linear_value)
                if not np.isfinite(numeric_value) or numeric_value <= log_floor_cutoff:
                    return float(log_floor_value)
                return float(np.log10(numeric_value))
            except Exception as safe_log_transform_exception:
                info_logger(
                    "[EXTREMA] _safe_log_transform failed; substituting floor value",
                    safe_log_transform_exception,
                    level="message",
                )
                return float(log_floor_value)

        # Section: Discover CDF files and group by orbit/instrument
        # orbit_map[orbit_number][instrument_name] -> list[cdf_paths]; skips
        # paths containing '_orb_' and filenames lacking a parsable orbit.
        orbit_map: Dict[int, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for path_obj in Path(directory_path).rglob("*.[cC][dD][fF]"):
            candidate_path = str(path_obj)
            if "_orb_" in candidate_path.lower():  # skip aggregated/orbit products
                continue
            instrument_name = get_cdf_file_type(candidate_path)
            if not (
                isinstance(instrument_name, str) and instrument_name in instrument_order
            ):
                continue  # not one of the target instruments
            file_name = os.path.basename(candidate_path)
            file_parts = file_name.split("_")
            if len(file_parts) < 5:
                continue  # unexpected naming, cannot parse orbit
            try:
                orbit_number = int(file_parts[-2])
            except Exception as instrument_processing_exception:
                continue
            orbit_map[orbit_number][instrument_name].append(candidate_path)

        sorted_orbit_numbers = sorted(orbit_map.keys())

        # Section: Accumulators (resumable, maxima never decrease)
        # energy_positive_counts_by_instrument -> per-energy positive finite counts
        # positive_sample_arrays_by_instrument -> positive intensity arrays for z
        # file_progress_index_by_instrument    -> last processed file index
        # total_files_per_instrument           -> file counts per instrument
        energy_positive_counts_by_instrument: Dict[str, Dict[float, int]] = {
            inst: defaultdict(int) for inst in instrument_order
        }
        positive_sample_arrays_by_instrument: Dict[str, List[np.ndarray]] = {
            inst: [] for inst in instrument_order
        }
        file_progress_index_by_instrument: Dict[str, int] = {
            inst: -1 for inst in instrument_order
        }
        total_files_per_instrument: Dict[str, int] = {
            inst: sum(
                len(orbit_map[orbit_number].get(inst, []))
                for orbit_number in sorted_orbit_numbers
            )
            for inst in instrument_order
        }

        # Aggregate total files across all instruments for a single tqdm bar.
        total_discovered_files = sum(total_files_per_instrument.values())
        processed_files_all = 0  # (presently informational only)
        extrema_progress_bar = tqdm(
            total=total_discovered_files,
            desc=f"Extrema {y_scale}/{z_scale}",
            unit="file",
            leave=False,
            disable=(total_discovered_files == 0),
        )

        try:
            orbits_since_last_flush = (
                0  # count of orbits with updates since last disk flush
            )
            for orbit_number in sorted_orbit_numbers:
                orbit_had_any_updates = (
                    False  # track whether this orbit produced any new data
                )
                for instrument_name in instrument_order:
                    key_prefix = f"{instrument_name}_{y_scale}_{z_scale}"
                    progress_key = f"{key_prefix}_extrema_progress"

                    # Fetch prior progress entry for this instrument+scale combo (if any)
                    raw_progress_entry = (
                        extrema_state.get(progress_key)
                        if isinstance(extrema_state, dict)
                        else None
                    )
                    progress_entry = (
                        raw_progress_entry
                        if isinstance(raw_progress_entry, dict)
                        else {}
                    )
                    if isinstance(progress_entry, dict) and progress_entry.get(
                        "complete"
                    ):
                        continue  # Already finished in a previous run

                    # Attempt reuse if requesting log scaling for either axis and a completed
                    # linear counterpart exists (same other-axis scale). This must happen
                    # before we derive file lists to avoid unnecessary scans.
                    reuse_performed = False
                    need_log_y = y_scale == "log"
                    need_log_z = z_scale == "log"
                    if need_log_y or need_log_z:
                        linear_y_progress_key = (
                            f"{instrument_name}_linear_{z_scale}_extrema_progress"
                            if need_log_y
                            else None
                        )
                        linear_z_progress_key = (
                            f"{instrument_name}_{y_scale}_linear_extrema_progress"
                            if need_log_z
                            else None
                        )
                        linear_y_complete = False
                        if linear_y_progress_key:
                            linear_y_progress_entry = extrema_state.get(
                                linear_y_progress_key
                            )
                            if isinstance(linear_y_progress_entry, dict):
                                linear_y_complete = bool(
                                    linear_y_progress_entry.get("complete")
                                )
                        linear_z_complete = False
                        if linear_z_progress_key:
                            linear_z_progress_entry = extrema_state.get(
                                linear_z_progress_key
                            )
                            if isinstance(linear_z_progress_entry, dict):
                                linear_z_complete = bool(
                                    linear_z_progress_entry.get("complete")
                                )
                        if (not need_log_y or linear_y_complete) and (
                            not need_log_z or linear_z_complete
                        ):
                            # Obtain linear maxima sources
                            if need_log_y:
                                linear_y_prefix = f"{instrument_name}_linear_{z_scale}"
                                raw_linear_y_max = extrema_state.get(
                                    f"{linear_y_prefix}_y_max"
                                )
                                linear_y_max_val = (
                                    raw_linear_y_max
                                    if isinstance(raw_linear_y_max, (int, float))
                                    else None
                                )
                                transformed_y_max = _safe_log_transform(
                                    linear_y_max_val
                                )
                                y_min_val = log_floor_value
                            else:
                                transformed_y_max = extrema_state.get(
                                    f"{key_prefix}_y_max"
                                )
                                y_min_val = 0
                            if need_log_z:
                                linear_z_prefix = f"{instrument_name}_{y_scale}_linear"
                                raw_linear_z_max = extrema_state.get(
                                    f"{linear_z_prefix}_z_max"
                                )
                                linear_z_max_val = (
                                    raw_linear_z_max
                                    if isinstance(raw_linear_z_max, (int, float))
                                    else None
                                )
                                transformed_z_max = _safe_log_transform(
                                    linear_z_max_val
                                )
                                z_min_val = log_floor_value
                            else:
                                transformed_z_max = extrema_state.get(
                                    f"{key_prefix}_z_max"
                                )
                                z_min_val = 0
                            # Persist transformed (reused) extrema directly without scanning.
                            extrema_state[f"{key_prefix}_y_min"] = y_min_val
                            extrema_state[f"{key_prefix}_y_max"] = (
                                transformed_y_max
                                if transformed_y_max is not None
                                else log_floor_value
                            )
                            extrema_state[f"{key_prefix}_z_min"] = z_min_val
                            extrema_state[f"{key_prefix}_z_max"] = (
                                transformed_z_max
                                if transformed_z_max is not None
                                else log_floor_value
                            )
                            total_for_inst = total_files_per_instrument[instrument_name]
                            extrema_state[progress_key] = {
                                "processed_index": max(total_for_inst - 1, -1),
                                "total": total_for_inst,
                                "complete": True,
                            }
                            reuse_performed = True
                            info_logger(
                                f"[EXTREMA] Reused linear extrema for log scaling instrument={instrument_name} y_scale={y_scale} z_scale={z_scale}",
                                level="message",
                            )
                    if reuse_performed:
                        # Skip scanning
                        orbit_had_any_updates = True
                        continue

                    # Files for the current orbit/instrument
                    orbit_instrument_files = orbit_map[orbit_number].get(
                        instrument_name, []
                    )
                    if not orbit_instrument_files:
                        continue

                    # Section: Per-file ingestion & collapse
                    for cdf_path in sorted(orbit_instrument_files):
                        try:
                            cdf_obj = cdflib.CDF(cdf_path)
                            data_raw = np.asarray(cdf_obj.varget("data"))
                            energy_raw = np.asarray(cdf_obj.varget("energy"))
                            pitch_angle_raw = np.asarray(cdf_obj.varget("pitch_angle"))

                            # Normalize dimensionality: handle (t, energy, pa) vs collapsed shapes.
                            energy_axis = (
                                energy_raw[0, 0, :]
                                if energy_raw.ndim == 3
                                else energy_raw
                            )
                            pitch_angle_axis = (
                                pitch_angle_raw[0, :, 0]
                                if pitch_angle_raw.ndim == 3
                                else pitch_angle_raw
                            )
                            # If dimensions are reversed (energy, pa) vs expected (pa, energy) fix ordering.
                            if data_raw.shape[1] == len(energy_axis) and data_raw.shape[
                                2
                            ] == len(pitch_angle_axis):
                                data_raw = np.transpose(data_raw, (0, 2, 1))

                            # Collapse pitch-angle dimension (axis=1 after potential transpose) to pool intensities.
                            collapsed_intensity_matrix = FAST_COLLAPSE_FUNCTION(
                                data_raw, axis=1
                            )

                            # Mask columns that are entirely NaN and restrict acceptable energy bounds.
                            non_all_nan_column_mask = ~np.all(
                                np.isnan(collapsed_intensity_matrix), axis=0
                            )
                            valid_energy_range_mask = (energy_axis >= 0) & (
                                energy_axis <= 4000
                            )
                            combined_valid_column_mask = (
                                non_all_nan_column_mask & valid_energy_range_mask
                            )
                            collapsed_intensity_matrix = collapsed_intensity_matrix[
                                :, combined_valid_column_mask
                            ]
                            if collapsed_intensity_matrix.size == 0:
                                continue  # Nothing useful in this file

                            # Energy values that correspond to retained (non-empty) columns.
                            retained_energy_values = energy_axis[
                                combined_valid_column_mask
                            ]

                            # Identify strictly positive finite samples.
                            with np.errstate(invalid="ignore"):
                                positive_sample_mask = np.isfinite(
                                    collapsed_intensity_matrix
                                ) & (collapsed_intensity_matrix > 0)
                            if positive_sample_mask.any():
                                # Count positive samples for each retained energy column.
                                positive_counts_per_energy_column = (
                                    positive_sample_mask.sum(axis=0)
                                )
                                for energy_value, positive_count in zip(
                                    retained_energy_values,
                                    positive_counts_per_energy_column,
                                ):
                                    if positive_count > 0:
                                        energy_positive_counts_by_instrument[
                                            instrument_name
                                        ][float(energy_value)] += int(positive_count)
                                # Flatten (extract) positive intensity values for percentile computation.
                                flattened_positive_values = collapsed_intensity_matrix[
                                    positive_sample_mask
                                ]
                                if flattened_positive_values.size:
                                    positive_sample_arrays_by_instrument[
                                        instrument_name
                                    ].append(flattened_positive_values)

                            # Update progress for this instrument (file index)
                            file_progress_index_by_instrument[instrument_name] += 1
                            orbit_had_any_updates = True
                            processed_files_all += 1  # informational
                            try:
                                extrema_progress_bar.update(1)
                            except Exception as extrema_progress_postfix_exception:
                                pass  # tqdm may be disabled in some environments
                        except Exception as ingest_file_exception:
                            # Capture ingest/parse issues but continue with remaining files.
                            info_logger(
                                f"[EXTREMA] Ingest failure inst={instrument_name} orbit={orbit_number} file={cdf_path}",
                                ingest_file_exception,
                                level="message",
                            )

                    # Section: Post-orbit extrema update (monotonic merge)
                    try:
                        energy_positive_counts_map = (
                            energy_positive_counts_by_instrument[instrument_name]
                        )
                        positive_value_blocks = positive_sample_arrays_by_instrument[
                            instrument_name
                        ]

                        # Candidate Y (energy) maximum via configurable cumulative coverage rule.
                        candidate_energy_max = 0.0
                        if energy_positive_counts_map:
                            sorted_energy_values = sorted(
                                energy_positive_counts_map.keys()
                            )
                            positive_counts_array = np.array(
                                [
                                    energy_positive_counts_map[e_val]
                                    for e_val in sorted_energy_values
                                ]
                            )
                            cumulative_positive_counts_array = np.cumsum(
                                positive_counts_array
                            )
                            coverage_target_count = (
                                0.99 * cumulative_positive_counts_array[-1]
                            )
                            coverage_index = np.searchsorted(
                                cumulative_positive_counts_array,
                                coverage_target_count,
                                side="right",
                            )
                            if coverage_index >= len(sorted_energy_values):
                                coverage_index = len(sorted_energy_values) - 1
                            candidate_energy_max = float(
                                sorted_energy_values[coverage_index]
                            )

                        # Candidate Z (intensity) maximum via configurable percentile of pooled positive values.
                        candidate_intensity_max = 0.0
                        if positive_value_blocks:
                            aggregated_positive_values = np.concatenate(
                                positive_value_blocks
                            )
                            finite_positive_values = aggregated_positive_values[
                                np.isfinite(aggregated_positive_values)
                                & (aggregated_positive_values > 0)
                            ]
                            if finite_positive_values.size:
                                candidate_intensity_max = float(
                                    np.nanpercentile(
                                        finite_positive_values, max_percentile
                                    )
                                )

                        # Retrieve any previously stored maxima to enforce monotonic growth.
                        previous_energy_max_raw = (
                            extrema_state.get(f"{key_prefix}_y_max")
                            if isinstance(extrema_state, dict)
                            else None
                        )
                        previous_energy_max = (
                            previous_energy_max_raw
                            if isinstance(previous_energy_max_raw, (int, float))
                            else None
                        )
                        previous_intensity_max_raw = (
                            extrema_state.get(f"{key_prefix}_z_max")
                            if isinstance(extrema_state, dict)
                            else None
                        )
                        previous_intensity_max = (
                            previous_intensity_max_raw
                            if isinstance(previous_intensity_max_raw, (int, float))
                            else None
                        )

                        # Monotonic merge: prefer the larger of previous and new candidate.
                        merged_energy_max = (
                            max(float(previous_energy_max), candidate_energy_max)
                            if previous_energy_max is not None
                            else candidate_energy_max
                        )
                        merged_intensity_max = (
                            max(float(previous_intensity_max), candidate_intensity_max)
                            if previous_intensity_max is not None
                            else candidate_intensity_max
                        )

                        # Apply final normalisation: ceil, energy cap at 4000.
                        try:
                            merged_energy_max = int(
                                min(4000, math.ceil(merged_energy_max))
                            )
                        except Exception as extrema_bar_update_exception:
                            pass
                        try:
                            merged_intensity_max = float(
                                math.ceil(merged_intensity_max)
                            )
                        except Exception as extrema_reuse_logging_exception:
                            pass

                        # Optional minima (currently fixed 0 unless compute_mins True)
                        if compute_mins and positive_value_blocks:
                            try:
                                aggregated_positive_values = np.concatenate(
                                    positive_value_blocks
                                )
                                finite_positive_values = aggregated_positive_values[
                                    np.isfinite(aggregated_positive_values)
                                    & (aggregated_positive_values > 0)
                                ]
                                if finite_positive_values.size:
                                    intensity_min_store = float(
                                        np.nanpercentile(finite_positive_values, 1)
                                    )
                                else:
                                    intensity_min_store = 0.0
                            except Exception as extrema_flush_exception:
                                intensity_min_store = 0.0
                            energy_min_store = 0
                        else:
                            energy_min_store = 0
                            intensity_min_store = 0

                        # Persist updated extrema & progress bookkeeping.
                        extrema_state[f"{key_prefix}_y_min"] = energy_min_store
                        extrema_state[f"{key_prefix}_y_max"] = merged_energy_max
                        extrema_state[f"{key_prefix}_z_min"] = intensity_min_store
                        extrema_state[f"{key_prefix}_z_max"] = merged_intensity_max
                        extrema_state[progress_key] = {
                            "processed_index": file_progress_index_by_instrument[
                                instrument_name
                            ],
                            "total": total_files_per_instrument[instrument_name],
                            "complete": file_progress_index_by_instrument[
                                instrument_name
                            ]
                            + 1
                            >= total_files_per_instrument[instrument_name],
                        }

                        # Progress bar suffix: latest instrument + snapshot of maxima.
                        try:
                            extrema_progress_bar.set_postfix(
                                inst=instrument_name,
                                y_max=merged_energy_max,
                                z_max=f"{merged_intensity_max:.2e}",
                                refresh=False,
                            )
                        except Exception as maxima_progress_postfix_exception:
                            pass
                    except Exception as maxima_update_exception:
                        info_logger(
                            f"[EXTREMA] Update failure inst={instrument_name} orbit={orbit_number}",
                            maxima_update_exception,
                            level="message",
                        )
                # Section: Batched flush decision
                if orbit_had_any_updates:
                    orbits_since_last_flush += 1
                    if orbits_since_last_flush >= flush_batch_size:
                        try:
                            with open(extrema_json_path, "w") as file_out:
                                json.dump(extrema_state, file_out, indent=2)
                            orbits_since_last_flush = 0
                        except Exception as orbit_flush_exception:
                            info_logger(
                                f"[EXTREMA] Batched flush failure after orbit {orbit_number}",
                                orbit_flush_exception,
                                level="message",
                            )
            # Section: Final flush if pending updates remain
            if orbits_since_last_flush > 0:
                try:
                    with open(extrema_json_path, "w") as file_out:
                        json.dump(extrema_state, file_out, indent=2)
                except Exception as final_flush_exception:
                    info_logger(
                        "[EXTREMA] Final batched flush failure",
                        final_flush_exception,
                        level="message",
                    )
        finally:
            try:
                extrema_progress_bar.close()
            except Exception as maxima_update_exception_outer:
                pass
        return extrema_state

    def _signal_handler(signum, frame):  # frame unused
        # On first signal, request shutdown and immediately terminate children
        if not shutdown_requested["flag"]:
            info_logger(
                f"[INTERRUPT] Signal {signum} received. Requesting shutdown...",
                level="message",
            )
            shutdown_requested["flag"] = True
            try:
                _terminate_all_child_processes()
            finally:
                # Interrupt the main loop immediately
                raise KeyboardInterrupt
        else:
            # Second Ctrl+C -> hard exit
            info_logger(
                "[INTERRUPT] Second interrupt – forcing immediate exit.",
                level="message",
            )
            try:
                _terminate_all_child_processes()
            finally:
                raise SystemExit(130)

    # Register (idempotent inside single main process invocation)
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception as signal_registration_exception:
        info_logger(
            "[WARN] Could not register signal handlers",
            signal_registration_exception,
            level="message",
        )
    # Load the DataFrame of filtered orbits (for vertical line marking, etc.)
    filtered_orbits_dataframe = load_filtered_orbits()

    # Configure buffered logging batch size (reuse flush_batch_size if not explicitly set)
    try:
        configure_info_logger_batch(log_flush_batch_size or flush_batch_size)
    except Exception as signal_registration_exception_outer:
        pass

    # Compute or load global extrema for this y/z combo prior to scheduling futures
    # Note: 'max_processing_percentile' controls intensity percentile; energy coverage fixed at 99%.
    # Derive (or refresh) global extrema; intensity percentile governed by
    # 'max_processing_percentile' (energy coverage remains fixed at 99%).
    global_extrema = compute_global_extrema(
        directory_path,
        y_scale,
        z_scale,
        instrument_order,
        compute_mins=False,
        max_percentile=max_processing_percentile,
        log_floor_cutoff=0.1,
        log_floor_value=-1.0,
        flush_batch_size=flush_batch_size,
    )

    # Gather all CDF files, skipping any with '_orb_' in the filename (these are not data files)
    cdf_file_paths = [
        str(path)
        for path in Path(directory_path).rglob("*.[cC][dD][fF]")
        if "_orb_" not in str(path).lower()
    ]

    # Helper function: extract orbit number and instrument type from a CDF file path
    def extract_orbit_and_instrument(cdf_path: str):
        """Parse filename to ``(orbit_number, instrument_type, cdf_path)`` or ``None``.

        Parameters
        ----------
        cdf_path : str
            Path to a CDF file.

        Returns
        -------
        tuple of (int, str, str) or None
            Parsed ``(orbit_number, instrument_type, cdf_path)`` or ``None``.

        Notes
        -----
        Returns ``None`` when the filename doesn't match the expected pattern,
        the orbit number cannot be parsed, or the path corresponds to a non-data
        CDF (e.g., when ``instrument_type`` is ``None`` or ``'orb'``).
        """
        filename = os.path.basename(cdf_path)
        parts = filename.split("_")
        if len(parts) < 5:
            return None
        try:
            orbit_number = int(parts[-2])
        except Exception as invalid_orbit_number_exception:
            info_logger(
                f"[ERROR] Invalid orbit number in filename: {filename}",
                invalid_orbit_number_exception,
                level="message",
            )
            return None
        instrument_type = get_cdf_file_type(cdf_path)
        if instrument_type is None or instrument_type == "orb":
            return None
        return (orbit_number, instrument_type, cdf_path)

    # Build a mapping: orbit_number -> {instrument_type: cdf_path, ...}
    orbit_to_instruments = defaultdict(dict)
    for idx, cdf_path in enumerate(cdf_file_paths):
        result = extract_orbit_and_instrument(cdf_path)
        if result is not None:
            orbit_number, instrument_type, cdf_path = result
            orbit_to_instruments[orbit_number][instrument_type] = cdf_path

    # Sort orbits by orbit number (ascending)
    sorted_orbits = sorted(orbit_to_instruments.items(), key=lambda x: x[0])
    total_orbits = len(sorted_orbits)

    # Progress and error tracking (per y/z scale combo)
    progress_key = f"progress_{y_scale}_{z_scale}_last_orbit"
    error_key = f"{y_scale}_{z_scale}_error_plotting"
    progress_data = {}
    last_completed_orbit = None
    error_orbits = set()
    if (progress_json_path is not None) and (not ignore_progress_json):
        try:
            with open(progress_json_path, "r") as f:
                progress_data = json.load(f)
            last_completed_orbit = progress_data.get(progress_key, None)
            error_orbits = set(progress_data.get(error_key, []))
        except Exception as progress_json_initial_read_exception:
            info_logger(
                f"[ERROR] Failed to load progress JSON from {progress_json_path}. Starting fresh.",
                progress_json_initial_read_exception,
                level="error",
            )
            progress_data = {}
            last_completed_orbit = None
            error_orbits = set()

    # Determine where to start (resume from last completed orbit, if any)
    start_idx = 0
    if last_completed_orbit is not None:
        for i, (orbit, _) in enumerate(sorted_orbits):
            if orbit > last_completed_orbit:
                start_idx = i
                break
        else:
            start_idx = total_orbits  # All done
        info_logger(
            f"[RESUME] Skipping {start_idx} orbits (up to and including orbit {last_completed_orbit}) based on progress file. {len(error_orbits)} error orbits will also be skipped.",
            level="message",
        )
    else:
        info_logger(
            f"[RESUME] No previous progress found. Starting from the first orbit. {len(error_orbits)} error orbits will be skipped if present.",
            level="message",
        )

    # Batch processing state
    # Use LOGFILE_DATETIME to improve ETA if possible
    batch_start_time = _time.time()
    try:
        # Correct variable: FAST_LOGFILE_DATETIME_STRING (not path) holds the timestamp string
        if "FAST_LOGFILE_DATETIME_STRING" in globals():
            dt = datetime.strptime(FAST_LOGFILE_DATETIME_STRING, "%Y-%m-%d_%H-%M-%S")
            batch_start_time = dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception as logfile_datetime_parse_exception:
        # Non-fatal; continue with current time
        info_logger(
            "[WARN] LOGFILE_DATETIME_STRING parsing failed, using current time for batch start.",
            logfile_datetime_parse_exception,
            level="message",
        )
    orbit_times = deque(maxlen=20)  # Track only last 20 orbit times for ETA
    div_zero_orbits = set()  # Track orbits with divide by zero warnings

    # Progress bar selection
    use_tqdm_bar = bool(use_tqdm) if use_tqdm is not None else False

    # Prepare the main orbit argument list
    orbit_args_list = []
    for orbit_number, instrument_files in sorted_orbits[start_idx:]:
        if orbit_number in error_orbits:
            continue
        orbit_args = (
            orbit_number,
            instrument_files,
            filtered_orbits_dataframe,
            zoom_duration_minutes,
            y_scale,
            z_scale,
            instrument_order,
            colormap,
            output_base,
            orbit_timeout_seconds,
            instrument_timeout_seconds,
            global_extrema,
        )
        orbit_args_list.append(orbit_args)

    results = []

    # Load progress data again in case of concurrent updates
    # Progress batching state
    if flush_batch_size < 1:
        flush_batch_size = 1
    _batched_progress_dirty = {"count": 0}

    def save_progress_json(progress_data: Dict[str, Any], force: bool = False) -> None:
        """Write ``progress_data`` to disk, logging on failure.

        Parameters
        ----------
        progress_data : dict
            Progress dictionary to persist.

        Returns
        -------
        None
        """
        if progress_json_path is None:
            return
        if not force:
            _batched_progress_dirty["count"] += 1
            if _batched_progress_dirty["count"] < flush_batch_size:
                return
        # Flush now
        _batched_progress_dirty["count"] = 0
        try:
            with open(progress_json_path, "w") as f:
                json.dump(progress_data, f, indent=2)
        except Exception as progress_json_write_exception:
            info_logger(
                "[FAIL] Could not write progress JSON",
                progress_json_write_exception,
                level="error",
            )

    # Multiprocessing batch execution with correct tqdm update
    executor = None
    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        future_to_orbit = {}
        for args in orbit_args_list:
            if shutdown_requested["flag"]:
                break
            future = executor.submit(FAST_process_single_orbit, *args)
            future_to_orbit[future] = args[0]
        futures = set(future_to_orbit.keys())

        def _handle_completed_future(fut, orbit_number):
            """Consume a completed future and update progress JSON.

            Parameters
            ----------
            fut : concurrent.futures.Future
                The completed future.
            orbit_number : int
                Orbit number corresponding to this future.

            Returns
            -------
            None
            """
            try:
                result = fut.result()
                results.append(result)
                status_value = result.get("status")
                if verbose and use_tqdm_bar:
                    tqdm.write(
                        f"[BATCH] Completed orbit {orbit_number}: {status_value}"
                    )
                if progress_json_path is not None:
                    try:
                        with open(progress_json_path, "r") as f:
                            pdisk = json.load(f)
                    except Exception as progress_json_read_exception:
                        info_logger(
                            f"[WARN] Could not read progress JSON (using empty). Exception: {progress_json_read_exception}",
                            level="message",
                        )
                        pdisk = {}
                    # Ensure base keys
                    pdisk[progress_key] = orbit_number
                    if error_key not in pdisk:
                        pdisk[error_key] = []
                    # Timeout keys
                    orbit_timeout_key = f"orbit_{y_scale}_{z_scale}_timed_out"
                    if orbit_timeout_key not in pdisk:
                        pdisk[orbit_timeout_key] = []
                    # If instrument timeout, store under instrument specific list
                    if status_value == "error":
                        # Record generic error orbit list
                        pdisk[error_key] = sorted(
                            list(set(pdisk[error_key]) | {orbit_number})
                        )
                        # Capture detailed error reasons (if provided)
                        error_list = result.get("errors") or []
                        for err_msg in error_list:
                            # Derive a short reason token
                            reason_token = "generic"
                            lowered = err_msg.lower()
                            if "divide" in lowered and "zero" in lowered:
                                reason_token = "divide-by-zero"
                            elif "invalid" in lowered and "cdf" in lowered:
                                reason_token = "invalid-cdf"
                            elif "timeout" in lowered:
                                reason_token = "timeout"
                            elif "plotting" in lowered:
                                reason_token = "plotting"
                            # Determine instrument if present in message (ees/eeb/ies/ieb)
                            instrument_match = None
                            for cand in ("ees", "eeb", "ies", "ieb"):
                                if cand in lowered:
                                    instrument_match = cand
                                    break
                            inst_token = instrument_match or "unknown"
                            detailed_key = (
                                f"{inst_token}_{y_scale}_{z_scale}_error-{reason_token}"
                            )
                            if detailed_key not in pdisk:
                                pdisk[detailed_key] = []
                            if orbit_number not in pdisk[detailed_key]:
                                pdisk[detailed_key].append(orbit_number)
                            # Also maintain a reason-only aggregate (no instrument)
                            agg_reason_key = f"{y_scale}_{z_scale}_error-{reason_token}"
                            if agg_reason_key not in pdisk:
                                pdisk[agg_reason_key] = []
                            if orbit_number not in pdisk[agg_reason_key]:
                                pdisk[agg_reason_key].append(orbit_number)
                    elif status_value == "timeout":
                        timeout_type = result.get("timeout_type")
                        timeout_instrument = result.get("timeout_instrument")
                        if timeout_type == "orbit":
                            pdisk[orbit_timeout_key] = sorted(
                                list(set(pdisk[orbit_timeout_key]) | {orbit_number})
                            )
                            info_logger(
                                f"[TIMEOUT-LOG] Recorded orbit timeout for orbit {orbit_number} under key {orbit_timeout_key}",
                                level="message",
                            )
                        elif timeout_type == "instrument":
                            if timeout_instrument is None:
                                timeout_instrument = "unknown_instrument"
                            instrument_timeout_key = (
                                f"{timeout_instrument}_{y_scale}_{z_scale}_timed_out"
                            )
                            if instrument_timeout_key not in pdisk:
                                pdisk[instrument_timeout_key] = []
                            pdisk[instrument_timeout_key] = sorted(
                                list(
                                    set(pdisk[instrument_timeout_key]) | {orbit_number}
                                )
                            )
                            info_logger(
                                f"[TIMEOUT-LOG] Recorded instrument timeout for orbit {orbit_number}, instrument {timeout_instrument} under key {instrument_timeout_key}",
                                level="message",
                            )
                    save_progress_json(pdisk)
            except Exception as orbit_future_exception:
                info_logger(
                    f"[BATCH] Orbit {orbit_number} generated an exception",
                    orbit_future_exception,
                    level="error",
                )
                results.append(
                    {
                        "orbit": orbit_number,
                        "status": "error",
                        "errors": [str(orbit_future_exception)],
                    }
                )
                if progress_json_path is not None:
                    try:
                        with open(progress_json_path, "r") as f:
                            pdisk = json.load(f)
                    except Exception as progress_json_read_exception2:
                        info_logger(
                            "[WARN] Could not read progress JSON after exception (using empty).",
                            progress_json_read_exception2,
                            level="message",
                        )
                        pdisk = {}
                    pdisk[progress_key] = orbit_number
                    if error_key not in pdisk:
                        pdisk[error_key] = []
                    pdisk[error_key] = sorted(
                        list(set(pdisk[error_key]) | {orbit_number})
                    )
                    # Record exception reason
                    exception_str = str(orbit_future_exception)
                    lowered = exception_str.lower()
                    reason_token = "generic"
                    if "divide" in lowered and "zero" in lowered:
                        reason_token = "divide-by-zero"
                    elif "invalid" in lowered and "cdf" in lowered:
                        reason_token = "invalid-cdf"
                    elif "timeout" in lowered:
                        reason_token = "timeout"
                    elif "plot" in lowered:
                        reason_token = "plotting"
                    detailed_key = f"unknown_{y_scale}_{z_scale}_error-{reason_token}"
                    if detailed_key not in pdisk:
                        pdisk[detailed_key] = []
                    if orbit_number not in pdisk[detailed_key]:
                        pdisk[detailed_key].append(orbit_number)
                    agg_reason_key = f"{y_scale}_{z_scale}_error-{reason_token}"
                    if agg_reason_key not in pdisk:
                        pdisk[agg_reason_key] = []
                    if orbit_number not in pdisk[agg_reason_key]:
                        pdisk[agg_reason_key].append(orbit_number)
                    save_progress_json(pdisk)

        if shutdown_requested["flag"]:
            info_logger(
                "[INTERRUPT] Shutdown requested before any futures executed.",
                level="message",
            )
            raise KeyboardInterrupt

        # Non-blocking wait loop with polling so Ctrl+C is handled immediately
        total_count = len(futures)
        processed_count = 0
        progress_bar = None
        if use_tqdm_bar:
            if start_idx > 0:
                info_logger(
                    f"[RESUME] Resuming progress bar at orbit {start_idx+1} of {total_orbits} for y_scale={y_scale}, z_scale={z_scale}.",
                    level="message",
                )
            progress_bar = tqdm(
                total=total_count,
                initial=0,
                desc=f"Orbits - {y_scale} / {z_scale}",
                unit="orbit",
                leave=False,
            )
        try:
            while futures:
                if shutdown_requested["flag"]:
                    break
                done, pending = concurrent.futures.wait(
                    futures, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    futures.discard(fut)
                    orbit_number = future_to_orbit.get(fut)
                    _handle_completed_future(fut, orbit_number)
                    processed_count += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if shutdown_requested["flag"]:
            info_logger(
                "[INTERRUPT] Cancelling remaining futures and terminating workers...",
                level="message",
            )
            for fut in list(futures):
                try:
                    fut.cancel()
                except Exception as future_cancel_exception:
                    pass
            # Forcefully terminate worker processes
            try:
                executor.shutdown(wait=False, cancel_futures=True)
                if hasattr(executor, "_processes"):
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            proc.terminate()
                        except Exception as process_terminate_exception_inner:
                            pass
                    # brief grace period
                    _time.sleep(0.05)
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            if proc.is_alive():
                                proc.kill()
                        except Exception as process_kill_exception_inner:
                            pass
            except Exception as executor_shutdown_exception_inner:
                pass
            raise KeyboardInterrupt
    except KeyboardInterrupt as keyboard_interrupt_exception:
        info_logger(
            f"[INTERRUPT] KeyboardInterrupt caught. Terminating worker processes... Exception: {keyboard_interrupt_exception}",
            level="message",
        )
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
                if hasattr(executor, "_processes"):
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            proc.terminate()
                        except Exception as process_terminate_exception:
                            # Suppress termination errors during shutdown
                            pass
                    # Force kill if still alive after short grace
                    _time.sleep(0.05)
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            if proc.is_alive():
                                proc.kill()
                        except Exception as retry_cleanup_exception:
                            pass
            except Exception as executor_shutdown_exception:
                # Suppress shutdown errors during intentional interrupt
                pass
        # Propagate interrupt so caller (main) stops and doesn't start next scale combo
        raise
    finally:
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as final_executor_shutdown_exception:
                # Suppress final shutdown errors; we're exiting anyway
                pass
    # Force flush any remaining batched progress updates
    try:
        if progress_json_path is not None and os.path.exists(progress_json_path):
            with open(progress_json_path, "r") as f:
                final_progress_data = json.load(f)
        else:
            final_progress_data = (
                progress_data if isinstance(progress_data, dict) else {}
            )
        save_progress_json(final_progress_data, force=True)
    except Exception as final_progress_flush_exception:
        pass
    # Flush any remaining buffered logs
    try:
        flush_info_logger_buffer(force=True)
    except Exception as final_log_flush_exception:
        pass
    # Retry logic for timeouts (run once, sequentially or with a small pool)
    if retry_timeouts and not shutdown_requested["flag"]:
        timeout_orbits = [r["orbit"] for r in results if r.get("status") == "timeout"]
        if timeout_orbits:
            info_logger(
                f"[RETRY] Retrying {len(timeout_orbits)} timed-out orbits once (thresholds orbit={orbit_timeout_seconds}s, instrument={instrument_timeout_seconds}s).",
                level="message",
            )
            # Rebuild args only for timeout orbits
            retry_args = [
                (
                    o,
                    orbit_to_instruments[o],
                    filtered_orbits_dataframe,
                    zoom_duration_minutes,
                    y_scale,
                    z_scale,
                    instrument_order,
                    colormap,
                    output_base,
                    orbit_timeout_seconds,
                    instrument_timeout_seconds,
                )
                for o in timeout_orbits
                if o in orbit_to_instruments
            ]
            retry_results = []
            # Use a smaller pool (min of current max_workers and 2) to reduce overhead
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(max_workers, 2)
                ) as retry_executor:
                    retry_future_map = {
                        retry_executor.submit(FAST_process_single_orbit, *ra): ra[0]
                        for ra in retry_args
                    }
                    for rfut in concurrent.futures.as_completed(retry_future_map):
                        r_orbit = retry_future_map[rfut]
                        try:
                            r_result = rfut.result()
                            retry_results.append(r_result)
                            info_logger(
                                f"[RETRY] Completed orbit {r_orbit}: {r_result.get('status')}",
                                level="message",
                            )
                            # Update JSON: remove from timeout lists if success
                            if (
                                progress_json_path is not None
                                and r_result.get("status") == "ok"
                            ):
                                try:
                                    with open(progress_json_path, "r") as f:
                                        pdisk_retry = json.load(f)
                                except Exception as retry_json_read_exception:
                                    info_logger(
                                        "[WARN] Could not read progress JSON for retry cleanup",
                                        retry_json_read_exception,
                                        level="message",
                                    )
                                    pdisk_retry = {}
                                # Remove orbit from any timeout key lists for this scale combo
                                keys_to_check = [
                                    k
                                    for k in pdisk_retry.keys()
                                    if k.endswith(f"_{y_scale}_{z_scale}_timed_out")
                                ]
                                modified = False
                                for tk in keys_to_check:
                                    if (
                                        isinstance(pdisk_retry.get(tk), list)
                                        and r_orbit in pdisk_retry[tk]
                                    ):
                                        pdisk_retry[tk] = [
                                            x for x in pdisk_retry[tk] if x != r_orbit
                                        ]
                                        modified = True
                                if modified:
                                    try:
                                        with open(progress_json_path, "w") as f:
                                            json.dump(pdisk_retry, f, indent=2)
                                        info_logger(
                                            f"[RETRY] Cleaned orbit {r_orbit} from timeout lists after successful retry.",
                                            level="message",
                                        )
                                    except Exception as retry_json_write_exception:
                                        info_logger(
                                            "[WARN] Could not write cleaned progress JSON",
                                            retry_json_write_exception,
                                            level="message",
                                        )
                        except Exception as retry_orbit_exception:
                            info_logger(
                                f"[RETRY] Orbit {r_orbit} retry failed with exception",
                                retry_orbit_exception,
                                level="error",
                            )
                            retry_results.append(
                                {
                                    "orbit": r_orbit,
                                    "status": "error",
                                    "errors": [str(retry_orbit_exception)],
                                }
                            )
            except Exception as retry_pool_exception:
                info_logger(
                    "[RETRY] Failed to execute retry pool",
                    retry_pool_exception,
                    level="message",
                )
            # Merge retry results: replace original entries for those orbits
            results_map = {r["orbit"]: r for r in results}
            for rr in retry_results:
                results_map[rr["orbit"]] = rr
            results = list(results_map.values())
    # Restore original signal handlers (best-effort) to avoid lingering changes
    try:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
    except Exception as fast_handler_restore_exception:
        info_logger(
            "[WARN] Could not restore original FAST signal handlers",
            fast_handler_restore_exception,
            level="message",
        )
    return results


def main() -> None:
    """Run the FAST batch plotter for all y/z scale combinations sequentially.

    Invokes ``FAST_plot_spectrograms_directory`` four times with combinations
    of linear/log y and z, using colormaps tailored for each.

    Returns
    -------
    None

    Notes
    -----
    An interrupt during any run stops the sequence immediately without
    starting subsequent combinations.
    """
    # Use the batch-runner's own handlers; avoid top-level sys.exit to ensure clean shutdown
    FAST_plot_spectrograms_directory(
        FAST_CDF_DATA_FOLDER_PATH,
        verbose=False,
        y_scale="linear",
        z_scale="linear",
        use_tqdm=True,
        colormap=DEFAULT_COLORMAP_LINEAR_Y_LINEAR_Z,
    )
    # If interrupted, exit without starting next scale combo
    FAST_plot_spectrograms_directory(
        FAST_CDF_DATA_FOLDER_PATH,
        verbose=False,
        y_scale="linear",
        z_scale="log",
        use_tqdm=True,
        colormap=DEFAULT_COLORMAP_LINEAR_Y_LOG_Z,
        max_processing_percentile=94.0,
    )
    FAST_plot_spectrograms_directory(
        FAST_CDF_DATA_FOLDER_PATH,
        verbose=False,
        y_scale="log",
        z_scale="linear",
        use_tqdm=True,
        colormap=DEFAULT_COLORMAP_LOG_Y_LINEAR_Z,
        max_processing_percentile=94.0,
    )
    FAST_plot_spectrograms_directory(
        FAST_CDF_DATA_FOLDER_PATH,
        verbose=False,
        y_scale="log",
        z_scale="log",
        use_tqdm=True,
        colormap=DEFAULT_COLORMAP_LOG_Y_LOG_Z,
        max_processing_percentile=94.0,
    )


# Section: Script entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        info_logger("[INTERRUPT] Batch plotting aborted by user.", level="message")
        print("\n[INTERRUPT] Aborted by user.")
        sys.exit(130)
