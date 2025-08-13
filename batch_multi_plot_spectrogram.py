#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides batch spectrogram plotting utilities.
Should work with CDFs like those from FAST (see batch_multi_plot_FAST_spectrograms.py) but should also be flexible with other data

Assumed folder layout is {CDF_DATA_DIRECTORY}/year/month
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

# Main imports for CDF data and plotting
import pandas as pd
import cdflib
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for batch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib import _pylab_helpers
from datetime import datetime, timezone
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
import json
import time as _time
import concurrent.futures

# garbage collection, parallel processing, and profiling
import gc
import concurrent.futures
import signal
import sys


# Section: Constants and Configuration
# Directory containing CDF data files
CDF_DATA_DIRECTORY = "./FAST_data/"
# List of variable names expected in CDF files
CDF_VARIABLE_NAMES = ["time_unix", "data", "energy", "pitch_angle"]
# Function to collapse 3D data arrays to 2D (e.g., sum over axis)
COLLAPSE_FUNCTION = np.nansum

# Colormaps for different axis scaling combinations (colorblind-friendly and visually distinct)
COLORMAP_LINEAR_Y_LINEAR_Z = "viridis"
COLORMAP_LINEAR_Y_LOG_Z = "cividis"
COLORMAP_LOG_Y_LINEAR_Z = "plasma"
COLORMAP_LOG_Y_LOG_Z = "inferno"

# Plot configuration
PLOT_FIGURE_WIDTH_INCHES = 6.25
PLOT_FIGURE_HEIGHT_INCHES = 2.0
TICK_LABEL_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 18
DEFAULT_ZOOM_WINDOW_MINUTES = 6  # Default zoom window duration in minutes
FILTERED_ORBITS_CSV_PATH = "./FAST_Cusp_Indices.csv"  # Path to filtered cusp orbits CSV
PLOTTING_PROGRESS_JSON_PATH = "./batch_multi_plot_progress.json"  # Path to JSON for tracking plotting progress across sessions
OUTPUT_BASE_DIRECTORY = "./plots/"  # Parent directory to save plots

# Logfile configuration for batch restarts
LOGFILE_DATETIME_PATH = "./batch_multi_plot_logfile_datetime.txt"
if os.path.exists(LOGFILE_DATETIME_PATH):
    with open(LOGFILE_DATETIME_PATH, "r") as f:
        LOGFILE_DATETIME_STRING = f.read().strip()
    if not LOGFILE_DATETIME_STRING:
        LOGFILE_DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(LOGFILE_DATETIME_PATH, "w") as f:
            f.write(LOGFILE_DATETIME_STRING)
else:
    LOGFILE_DATETIME_STRING = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(LOGFILE_DATETIME_PATH, "w") as f:
        f.write(LOGFILE_DATETIME_STRING)
LOGFILE_PATH = f"./batch_multi_plot_log_{LOGFILE_DATETIME_STRING}.log"

# Pitch angle category definitions for spectrogram plots
PITCH_ANGLE_CATEGORY_RANGES = {
    "downgoing": [(0, 30), (330, 360)],
    "upgoing": [(150, 210)],
    "perpendicular": [(40, 140), (210, 330)],
    "all": [(0, 360)],
}

# Global caches for batch optimization
filtered_orbits_cache = {}
orbit_column_cache = {}
cdf_type_cache = {}


# Section: Functions
# Section: Utility Functions
def load_filtered_orbits(csv_path=FILTERED_ORBITS_CSV_PATH):
    """Load the filtered orbits CSV with a simple cache.

    Parameters
    ----------
    csv_path : str, default FILTERED_ORBITS_CSV_PATH
        Path to the filtered orbits TSV/CSV file.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame of filtered orbits, or ``None`` if loading fails.

    Notes
    -----
    A module-level dictionary caches previously loaded DataFrames keyed by absolute
    path string to avoid repeated disk I/O in batch routines.
    """
    global filtered_orbits_cache
    if csv_path in filtered_orbits_cache:
        return filtered_orbits_cache[csv_path]
    try:
        dataframe = pd.read_csv(csv_path, sep="\t")
        filtered_orbits_cache[csv_path] = dataframe
        return dataframe
    except Exception as exc:
        log_error(f"Error loading CSV {csv_path}: {exc}")
        return None


# Section: SIGINT Handling
def _terminate_all_child_processes():
    """Attempt to terminate all child processes of this process.

    Returns
    -------
    None

    Notes
    -----
    Uses :mod:`psutil` (imported lazily) to enumerate child processes recursively
    and invoke ``terminate()`` on each. Exceptions during termination are suppressed
    because this function is used during best-effort shutdown handling.
    """
    import psutil

    current_process = psutil.Process()
    for child in current_process.children(recursive=True):
        try:
            child.terminate()
        except Exception as child_termination_exception:
            # Suppress individual child termination issues during shutdown.
            # Variable name intentionally descriptive for lint clarity.
            _ = child_termination_exception  # explicit no-op reference


def _sigint_handler(signum, frame):
    """SIGINT handler to terminate children and exit promptly.

    Parameters
    ----------
    signum : int
        Signal number.
    frame : FrameType or None
        Current execution frame (unused).

    Returns
    -------
    None
    """
    log_message("[INFO] SIGINT received. Terminating all child processes and exiting.")
    _terminate_all_child_processes()
    sys.exit(1)


# Section: Logging
_LOG_BUFFER = []  # list[tuple[str, str]]: buffered (level, message) entries
_LOG_BATCH_SIZE = 10  # default batch size for buffered logging; configurable


def configure_log_batch(batch_size: int):
    """Configure buffered logging batch size.

    Parameters
    ----------
    batch_size : int
        Desired number of log records to accumulate before an automatic flush.
        Values less than 1 are coerced to 1.
    """
    global _LOG_BATCH_SIZE
    _LOG_BATCH_SIZE = max(1, int(batch_size))


def _flush_log_buffer(force: bool = False):
    """Flush buffered log messages to disk.

    Parameters
    ----------
    force : bool, default False
        If True, flush even if the current buffer length is below the configured
        batch size threshold.
    """
    if not _LOG_BUFFER:
        return
    if (len(_LOG_BUFFER) >= _LOG_BATCH_SIZE) or force:
        try:
            with open(LOGFILE_PATH, "a") as logfile_out:
                for level, msg in _LOG_BUFFER:
                    if level == "error":
                        logfile_out.write(f"[ERROR] {msg}\n")
                    else:
                        logfile_out.write(msg + "\n")
        except Exception as log_flush_exception:
            # Last-resort console output
            tqdm.write(f"[ERROR] Failed flushing log buffer: {log_flush_exception}")
        finally:
            _LOG_BUFFER.clear()


def log_message(message: str, force_flush: bool = False):
    """Queue an informational log message.

    Messages are appended to an in-memory buffer; a flush occurs automatically
    once the configured batch size is reached or ``force_flush`` is True.
    """
    _LOG_BUFFER.append(("info", message))
    _flush_log_buffer(force=force_flush)


def log_error(message: str, force_flush: bool = False):
    """Queue an error log message and echo to console immediately."""
    tqdm.write("[ERROR] " + message)
    _LOG_BUFFER.append(("error", message))
    _flush_log_buffer(force=force_flush)


def get_timestamps_for_orbit(
    filtered_orbits_dataframe, orbit_number, instrument_type, time_unix_array
):
    """Compute orbit boundary UNIX timestamps from filtered indices.

    Parameters
    ----------
    filtered_orbits_dataframe : pandas.DataFrame
        DataFrame containing filtered orbits and min/max indices per instrument.
    orbit_number : int
        Orbit number to look up.
    instrument_type : str
        Instrument type identifier (e.g., ``'ees'``, ``'ies'``).
    time_unix_array : numpy.ndarray
        1D array of UNIX timestamps for the instrument.

    Returns
    -------
    list of float
        Boundary UNIX timestamps for the orbit (one value for a degenerate span
        or two values for start/end). Returns an empty list on invalid indices.
    """
    global orbit_column_cache
    dataframe = filtered_orbits_dataframe
    cache = orbit_column_cache
    if dataframe is None or instrument_type is None or time_unix_array is None:
        return []
    key = (id(dataframe), instrument_type)
    if key not in cache:
        orbit_column = next(col for col in dataframe.columns if "orbit" in col.lower())
        min_index_column = next(
            col
            for col in dataframe.columns
            if instrument_type in col.lower() and "min index" in col.lower()
        )
        max_index_column = next(
            col
            for col in dataframe.columns
            if instrument_type in col.lower() and "max index" in col.lower()
        )
        cache[key] = (orbit_column, min_index_column, max_index_column)
    else:
        orbit_column, min_index_column, max_index_column = cache[key]
    row = dataframe[dataframe[orbit_column] == orbit_number]
    if row.empty:
        return []
    min_index = row.iloc[0][cache[key][1]]
    max_index = row.iloc[0][cache[key][2]]
    try:
        min_index = int(min_index)
        max_index = int(max_index)
    except Exception as orbit_index_cast_exception:
        log_message("[WARN] Non-integer indices found in orbit row, using 0.")
        _ = orbit_index_cast_exception  # explicit no-op reference
        return []
    min_index = max(0, min(min_index, len(time_unix_array) - 1))
    max_index = max(0, min(max_index, len(time_unix_array) - 1))
    if min_index == max_index:
        return [float(time_unix_array[min_index])]
    return [float(time_unix_array[min_index]), float(time_unix_array[max_index])]


def get_cdf_file_type(cdf_file_path: str):
    """Infer instrument type from a CDF file path.

    Parameters
    ----------
    cdf_file_path : str
        Path to the CDF file.

    Returns
    -------
    str or None
        Instrument type string (e.g., ``'ees'``), ``'orb'`` for orbit files, or ``None`` if not recognized.
    """
    path_lower = cdf_file_path.lower()
    instrument_tags = ["ees", "eeb", "ies", "ieb"]
    if "_orb_" in path_lower:
        return "orb"
    for tag in instrument_tags:
        if f"_{tag}_" in path_lower:
            return tag
    log_error(f"Unknown CDF file type for path: {cdf_file_path}")
    return None


def get_variable_shape(cdf_path, variable_name):
    """Return the shape of a variable in a CDF file.

    Parameters
    ----------
    cdf_path : str
        Path to the CDF file.
    variable_name : str
        Variable name to inspect.

    Returns
    -------
    tuple or None
        Variable shape tuple, or ``None`` if variable absent / not array or an error occurs.
    """
    global cdf_type_cache
    instrument_type = cdf_type_cache.get(cdf_path)
    if instrument_type is None:
        instrument_type = get_cdf_file_type(cdf_path)
        cdf_type_cache[cdf_path] = instrument_type
    if instrument_type is None or instrument_type == "orb":
        return None
    try:
        with cdflib.CDF(cdf_path) as cdf:
            variable_data = cdf.varget(variable_name)
            return (
                variable_data.shape if isinstance(variable_data, np.ndarray) else None
            )
    except Exception as exc:
        log_error(f"Error reading {cdf_path} for variable {variable_name}: {exc}")
        return None


def get_cdf_var_shapes(
    cdf_folder_path=CDF_DATA_DIRECTORY, variable_names=CDF_VARIABLE_NAMES
):
    """Collect shapes of variables across CDF files in a folder.

    Parameters
    ----------
    cdf_folder_path : str, default CDF_DATA_DIRECTORY
        Directory containing CDF files.
    variable_names : list of str, default CDF_VARIABLE_NAMES
        Variable names to inspect.

    Returns
    -------
    dict
        Mapping from variable name (str) to list of shape tuples (or None) per file.
    """
    cdf_file_paths = [str(p) for p in Path(cdf_folder_path).rglob("*.[cC][dD][fF]")]
    shapes_by_variable = {}
    for variable_name in variable_names:
        shapes_by_variable[variable_name] = []
        for cdf_path in tqdm(
            cdf_file_paths,
            desc=f"Processing CDF files ({variable_name})",
            unit="file",
            total=len(cdf_file_paths),
        ):
            shapes_by_variable[variable_name].append(
                get_variable_shape(cdf_path, variable_name)
            )
    return shapes_by_variable


def close_all_axes_and_clear(fig):
    """Close axes/subplots and clear a figure to free memory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance to clear and dispose.

    Returns
    -------
    None

    Notes
    -----
    Ensures axes are deleted, the canvas is closed/detached, and removes the figure
    from the global Gcf registry when possible to mitigate memory growth during
    large batch operations.
    """
    for axis in list(fig.axes):
        try:
            fig.delaxes(axis)
        except Exception as axis_close_error:
            log_error(f"Error closing axis: {axis_close_error}")
    fig.clf()
    if hasattr(fig, "canvas") and fig.canvas is not None:
        try:
            fig.canvas.close()
        except Exception as canvas_close_error:
            log_message(f"[WARN] Error closing canvas: {canvas_close_error}")
        try:
            fig.canvas.figure = None
        except Exception as canvas_figure_clear_error:
            log_message(
                f"[WARN] Error clearing canvas figure: {canvas_figure_clear_error}"
            )
        fig.canvas = None
    try:
        if hasattr(fig, "number") and fig.number is not None:
            _pylab_helpers.Gcf.destroy(fig.number)
    except Exception as gcf_registry_error:
        log_error(f"Error removing figure from Gcf registry: {gcf_registry_error}")


# Section: Spectrogram Plotting
def make_spectrogram(
    x_axis_values,
    y_axis_values,
    data_array_3d,
    x_axis_min=None,
    x_axis_max=None,
    x_axis_is_unix=True,
    x_axis_label=None,
    center_timestamp=None,
    window_duration_seconds=None,
    y_axis_scale_function=None,
    y_axis_label=None,
    y_axis_min=0,
    y_axis_max=4000,
    z_axis_scale_function=None,
    z_axis_min=None,
    z_axis_max=None,
    z_axis_label=None,
    collapse_axis=1,
    colormap="viridis",
    axis_object=None,
    instrument_label=None,
    vertical_lines_unix=None,  # list of unix timestamps to mark
):
    """Plot a spectrogram by collapsing a 3D data array along an axis.

    Parameters
    ----------
    x_axis_values : array-like
        1D array for x (horizontal) axis (e.g., time sequence).
    y_axis_values : array-like
        1D array for y (vertical) axis (e.g., energy bins).
    data_array_3d : numpy.ndarray
        3D data array, e.g. ``(time, angle/pitch, energy)``.
    x_axis_min, x_axis_max : float, optional
        Explicit x-axis clipping bounds before plotting.
    x_axis_is_unix : bool, default True
        If ``True``, x-axis treated as UNIX seconds and converted to dates.
    x_axis_label : str, optional
        Custom x-axis label (default depends on ``x_axis_is_unix``).
    center_timestamp : float, optional
        Center of requested zoom window (UNIX seconds).
    window_duration_seconds : float, optional
        Duration of zoom window; both must be provided for zoom to apply.
    y_axis_scale_function : {'linear', 'log'}, optional
        Y-axis scaling; ``None`` behaves as ``'linear'``.
    y_axis_label : str, optional
        Y-axis label text.
    y_axis_min, y_axis_max : float, default 0, 4000
        Y-axis clipping range applied before filtering / plotting.
    z_axis_scale_function : {'linear', 'log'}, optional
        Color scale mode; ``None`` behaves as ``'linear'``.
    z_axis_min, z_axis_max : float, optional
        Optional color scale bounds (percentiles chosen if omitted).
    z_axis_label : str, optional
        Colorbar label text.
    collapse_axis : int, default 1
        Axis index along which to collapse the 3D data array.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    axis_object : matplotlib.axes.Axes, optional
        Existing axes to draw into; if ``None`` a new figure/axes created.
    instrument_label : str, optional
        Title string applied to the axes.
    vertical_lines_unix : list of float, optional
        UNIX timestamps to annotate with vertical lines.

    Returns
    -------
    axis_object : matplotlib.axes.Axes or None
        The axis object used for plotting (``None`` if no data plotted).
    x_axis_plot : numpy.ndarray or None
        X values actually used (possibly filtered / converted), or ``None`` if skipped.
    """
    # Log the function call and key parameters for debugging
    log_message(
        f"[DEBUG] make_spectrogram: y_axis_scale_function={y_axis_scale_function}, z_axis_scale_function={z_axis_scale_function}, z_axis_min={z_axis_min}, z_axis_max={z_axis_max}, colormap={colormap}"
    )

    # Convert input arrays to numpy arrays for consistency
    x_axis = np.asarray(x_axis_values)
    y_axis = np.asarray(y_axis_values)
    data_array = np.asarray(data_array_3d)

    # Collapse the 3D data array along the specified axis (e.g., sum over pitch angle)
    collapsed_matrix = COLLAPSE_FUNCTION(data_array, axis=collapse_axis)

    # Mask out columns that are all NaN and restrict to valid energy range
    nan_column_mask = ~np.all(np.isnan(collapsed_matrix), axis=0)
    valid_energy_mask = (y_axis >= y_axis_min) & (y_axis <= y_axis_max)
    combined_mask = nan_column_mask & valid_energy_mask
    collapsed_matrix = collapsed_matrix[:, combined_mask]
    y_axis = y_axis[combined_mask]
    if collapsed_matrix.size == 0 or y_axis.size == 0:
        log_message("[WARNING] All energy bins were filtered out. No data to plot.")
        return None, None

    # Ensure y-axis is increasing (for plotting)
    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        collapsed_matrix = collapsed_matrix[:, ::-1]

    # If a zoom window is specified, restrict to that window
    if center_timestamp is not None and window_duration_seconds is not None:
        half_window = window_duration_seconds / 2
        left_bound = center_timestamp - half_window
        right_bound = center_timestamp + half_window
        zoom_mask = (x_axis >= left_bound) & (x_axis <= right_bound)
        x_axis = x_axis[zoom_mask]
        collapsed_matrix = collapsed_matrix[zoom_mask, :]

    # Restrict to specified x-axis min/max if provided
    if x_axis_min is not None or x_axis_max is not None:
        x_mask = np.ones_like(x_axis, dtype=bool)
        if x_axis_min is not None:
            x_mask &= x_axis >= x_axis_min
        if x_axis_max is not None:
            x_mask &= x_axis <= x_axis_max
        x_axis = x_axis[x_mask]
        collapsed_matrix = collapsed_matrix[x_mask, :]

    # Convert x-axis to matplotlib date format if using unix timestamps
    if x_axis_is_unix:
        x_axis_datetime = np.array(
            [datetime.fromtimestamp(x, tz=timezone.utc) for x in x_axis]
        )
        x_axis_plot = date2num(x_axis_datetime)
        x_label = x_axis_label if x_axis_label is not None else "Time (UTC)"
    else:
        x_axis_plot = x_axis
        x_label = x_axis_label if x_axis_label is not None else "X"

    # Create a new figure and axis if not provided
    if axis_object is None:
        fig = Figure(figsize=(PLOT_FIGURE_WIDTH_INCHES, PLOT_FIGURE_HEIGHT_INCHES))
        canvas = FigureCanvas(fig)
        axis_object = fig.add_subplot(1, 1, 1)
    else:
        fig = axis_object.figure

    # Transpose matrix for plotting (so y-axis is vertical)
    matrix_plot = collapsed_matrix.T

    # Set x-axis limits to zoom window if specified, otherwise to full range
    if center_timestamp is not None and window_duration_seconds is not None:
        if x_axis_is_unix:
            left_num = float(
                date2num(
                    datetime.fromtimestamp(
                        center_timestamp - window_duration_seconds / 2, tz=timezone.utc
                    )
                )
            )
            right_num = float(
                date2num(
                    datetime.fromtimestamp(
                        center_timestamp + window_duration_seconds / 2, tz=timezone.utc
                    )
                )
            )
            axis_object.set_xlim(left_num, right_num)
        else:
            axis_object.set_xlim(
                center_timestamp - window_duration_seconds / 2,
                center_timestamp + window_duration_seconds / 2,
            )
    else:
        axis_object.set_xlim(x_axis_plot[0], x_axis_plot[-1])

    # If no data remains after filtering, skip plotting
    if matrix_plot.size == 0:
        log_message("[WARNING] No data to plot after filtering. Skipping plot.")
        return None, None

    # Set colorbar min/max if not provided
    if z_axis_min is None:
        z_axis_min = np.nanpercentile(matrix_plot, 1)
    if z_axis_max is None:
        z_axis_max = np.nanpercentile(matrix_plot, 99)

    # Find the smallest positive value for safe log scaling
    finite_positive = matrix_plot[np.isfinite(matrix_plot) & (matrix_plot > 0)]
    safe_vmin = np.nanmin(finite_positive) if finite_positive.size > 0 else 1e-10

    # Plot with log colorbar if requested, masking non-positive values
    if z_axis_scale_function == "log":
        if np.any(matrix_plot <= 0) or not (
            np.isfinite(z_axis_min)
            and np.isfinite(z_axis_max)
            and z_axis_min > 0
            and z_axis_max > 0
            and z_axis_max > z_axis_min
        ):
            log_message(
                "[WARNING] Non-positive values found in matrix for log colorbar. Masking to z_axis_min and enforcing log scale."
            )
        z_axis_min = float(max(z_axis_min, safe_vmin, 1e-10))
        z_axis_max = float(z_axis_max)
        # Mask all non-positive and non-finite values for log scale
        matrix_plot = np.where(
            ~np.isfinite(matrix_plot) | (matrix_plot <= 0), z_axis_min, matrix_plot
        )
        norm = mcolors.LogNorm(vmin=z_axis_min, vmax=z_axis_max)
        im = axis_object.imshow(
            matrix_plot,
            aspect="auto",
            origin="lower",
            extent=(x_axis_plot[0], x_axis_plot[-1], y_axis[0], y_axis[-1]),
            cmap=colormap,
            norm=norm,
        )
        # Compute tick marks for every integer power of 10 in range
        min_exponent = int(np.floor(np.log10(z_axis_min)))
        max_exponent = int(np.ceil(np.log10(z_axis_max)))
        ticks = [
            10**i
            for i in range(min_exponent, max_exponent + 1)
            if z_axis_min <= 10**i <= z_axis_max
        ]
        log_message(f"[DEBUG] make_spectrogram: log colorbar ticks: {ticks}")

        def log_tick_formatter(value, position=None):
            if value <= 0:
                return ""
            exponent = int(np.log10(value))
            if np.isclose(value, 10**exponent):
                return f"$10^{{{exponent}}}$"
            return ""

        # Create the colorbar with custom ticks and formatter
        colorbar = fig.colorbar(
            im,
            ax=axis_object,
            label=z_axis_label if z_axis_label is not None else "Counts",
            ticks=ticks,
            format=log_tick_formatter,
        )
    else:
        # Linear colorbar: mask NaN and inf values, set vmin/vmax
        z_axis_min = float(z_axis_min)
        z_axis_max = float(z_axis_max)
        matrix_plot = np.where(np.isnan(matrix_plot), z_axis_min, matrix_plot)
        matrix_plot = np.where(np.isneginf(matrix_plot), z_axis_min, matrix_plot)
        matrix_plot = np.where(np.isposinf(matrix_plot), z_axis_max, matrix_plot)
        if not (
            np.isfinite(z_axis_min)
            and np.isfinite(z_axis_max)
            and z_axis_max > z_axis_min
        ):
            z_axis_min = float(np.nanmin(matrix_plot))
            z_axis_max = float(np.nanmax(matrix_plot))
        im = axis_object.imshow(
            matrix_plot,
            aspect="auto",
            origin="lower",
            extent=(x_axis_plot[0], x_axis_plot[-1], y_axis[0], y_axis[-1]),
            cmap=colormap,
            vmin=z_axis_min,
            vmax=z_axis_max,
        )
        # Create the colorbar for linear scale
        colorbar = fig.colorbar(
            im,
            ax=axis_object,
            label=z_axis_label if z_axis_label is not None else "Counts",
        )

    # Set axis labels and title
    axis_object.set_xlabel(x_label)
    axis_object.set_ylabel(y_axis_label if y_axis_label is not None else "Energy (eV)")
    if instrument_label is not None:
        axis_object.set_title(instrument_label)

    # Configure y-axis ticks and scale
    if len(y_axis) >= 2:
        if y_axis_scale_function != "log":
            # For linear y-axis, set ticks at reasonable intervals
            y_max_str = str(y_axis_max)
            y_max_digits = len(y_max_str)
            y_first_digit = int(y_max_str[0])
            y_second_digit = int(y_max_str[1])
            if y_second_digit >= 5:
                step_size = 10**y_max_digits
                y_max_tick = (y_first_digit) * 10 ** (y_max_digits - 1)
            else:
                step_size = 10 ** (y_max_digits - 1)
                y_max_tick = (y_first_digit + 0.5) * 10 ** (y_max_digits - 1)
            yticks = [
                i
                for i in range(y_axis_min, int(y_max_tick) + 1, step_size)
                if (i / y_max_tick) <= 1.1
            ]
            if len(yticks) > 0:
                axis_object.set_yticks(yticks)
                axis_object.set_yticklabels([f"{int(e)}" for e in yticks])
        else:
            # For log y-axis, set scale to log
            axis_object.set_yscale("log")

    # Format x-axis as time if using unix timestamps
    if x_axis_is_unix:
        x_limits = axis_object.get_xlim()
        left_datetime = mdates.num2date(x_limits[0], tz=timezone.utc)
        right_datetime = mdates.num2date(x_limits[1], tz=timezone.utc)
        displayed_time_range_seconds = (right_datetime - left_datetime).total_seconds()
        if displayed_time_range_seconds < 120:
            axis_object.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M:%S", tz=timezone.utc)
            )
        else:
            axis_object.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M", tz=timezone.utc)
            )

    # Draw vertical lines for orbit boundaries or other events if provided
    if vertical_lines_unix is not None and len(vertical_lines_unix) > 0:
        if x_axis_is_unix:
            vertical_lines_plot = date2num(
                [
                    datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    for timestamp in vertical_lines_unix
                ]
            )
            x_min_plot = x_axis_plot[0]
            x_max_plot = x_axis_plot[-1]
            vertical_lines_plot = [
                v for v in vertical_lines_plot if x_min_plot <= v <= x_max_plot
            ]
        else:
            vertical_lines_plot = [
                v for v in vertical_lines_unix if x_axis_plot[0] <= v <= x_axis_plot[-1]
            ]
        for vertical_line in vertical_lines_plot:
            # Draw a thick black line under a thinner red line for visibility
            axis_object.axvline(
                vertical_line,
                color="black",
                linestyle="-",
                linewidth=4,
                alpha=1.0,
                zorder=10,
            )
            axis_object.axvline(
                vertical_line,
                color="red",
                linestyle="-",
                linewidth=2,
                alpha=1.0,
                zorder=11,
            )

    # Set tick parameters for better readability
    axis_object.tick_params(
        axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE, length=8, width=1
    )
    axis_object.tick_params(
        axis="both", which="minor", labelsize=TICK_LABEL_FONT_SIZE, length=5, width=1
    )
    colorbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE, length=6, width=1)
    colorbar.ax.tick_params(
        which="minor", labelsize=TICK_LABEL_FONT_SIZE, length=3, width=1
    )

    # Set axis label font sizes
    axis_object.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    axis_object.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    colorbar.ax.set_ylabel("Counts", fontsize=AXIS_LABEL_FONT_SIZE)

    # Return the axis and the x-axis values used for plotting
    return axis_object, x_axis_plot


def generic_plot_spectrogram_set(
    datasets,
    collapse_axis=1,
    zoom_center=None,
    zoom_window_seconds=None,
    vertical_lines=None,
    x_is_unix=True,
    y_scale="linear",
    z_scale="linear",
    colormap="viridis",
    figure_title=None,
    show=False,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
):
    """Plot a vertical stack of generic spectrograms.

    Parameters
    ----------
    datasets : list of dict
        Each dict requires keys ``'x'``, ``'y'``, ``'data'`` and may include optional keys:
        ``'label'``, ``'y_label'``, ``'z_label'``, ``'y_min'``, ``'y_max'``, ``'z_min'``, ``'z_max'``.
    collapse_axis : int, default 1
        Axis index of the 3D array collapsed prior to plotting.
    zoom_center : float, optional
        Center (UNIX time) for zoom column when used.
    zoom_window_seconds : float, optional
        Duration of zoom window (seconds) when ``zoom_center`` provided.
    vertical_lines : list of float, optional
        UNIX timestamps to annotate with vertical lines.
    x_is_unix : bool, default True
        If ``True``, x values are treated as UNIX seconds and formatted.
    y_scale : {'linear', 'log'}, default 'linear'
        Y-axis scaling mode.
    z_scale : {'linear', 'log'}, default 'linear'
        Color (intensity) scale mode.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    figure_title : str, optional
        Figure-level title (sup-title).
    show : bool, default False
        If ``True``, display interactively (requires GUI backend).
    y_min : float, optional
        Global Y min fallback when per-row not supplied. Defaults to 0 if omitted and per-row missing.
    y_max : float, optional
        Global Y max fallback when per-row not supplied. If both global and per-row absent, inferred.
    z_min : float, optional
        Global colorbar lower bound fallback.
    z_max : float, optional
        Global colorbar upper bound fallback.

    Returns
    -------
    tuple
        ``(fig, canvas)`` or ``(None, None)`` if ``datasets`` is empty.
    """
    if not datasets:
        return None, None
    fig = Figure(figsize=(10, 3 * len(datasets)))
    canvas = FigureCanvas(fig)
    axes = []
    for row_index, dataset in enumerate(datasets):
        axis_obj = fig.add_subplot(len(datasets), 1, row_index + 1)
        axes.append(axis_obj)
        # Resolve per-dataset ranges with global fallback (row-specific wins)
        dataset_y_min = dataset.get("y_min", y_min)
        dataset_y_max = dataset.get("y_max", y_max)
        dataset_z_min = dataset.get("z_min", z_min)
        dataset_z_max = dataset.get("z_max", z_max)
        # Compute fallback y max from provided y array if not given
        inferred_y_max = (
            dataset["y"].max()
            if dataset_y_max is None and dataset.get("y") is not None
            else dataset_y_max
        )
        make_spectrogram(
            x_axis_values=dataset["x"],
            y_axis_values=dataset["y"],
            data_array_3d=dataset["data"],
            collapse_axis=collapse_axis,
            center_timestamp=zoom_center,
            window_duration_seconds=zoom_window_seconds,
            x_axis_is_unix=x_is_unix,
            y_axis_scale_function=y_scale,
            z_axis_scale_function=z_scale,
            y_axis_min=dataset_y_min if dataset_y_min is not None else 0,
            y_axis_max=inferred_y_max if inferred_y_max is not None else 4000,
            z_axis_min=dataset_z_min,
            z_axis_max=dataset_z_max,
            colormap=colormap,
            y_axis_label=dataset.get("y_label", "Energy (eV)"),
            z_axis_label=dataset.get("z_label", "Counts"),
            x_axis_label="Time (UTC)" if x_is_unix else dataset.get("x_label"),
            vertical_lines_unix=vertical_lines,
            axis_object=axis_obj,
        )
        if dataset.get("label"):
            axis_obj.set_title(dataset["label"])
    if figure_title:
        fig.suptitle(figure_title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig, canvas


def generic_batch_plot(
    items,
    output_dir,
    build_datasets_fn,
    zoom_center_fn=None,
    zoom_window_seconds=None,
    vertical_lines_fn=None,
    y_scale="linear",
    z_scale="linear",
    colormap="viridis",
    max_workers=2,
    progress_json_path: str = PLOTTING_PROGRESS_JSON_PATH,
    ignore_progress_json: bool = False,
    flush_batch_size: int = 10,
    log_flush_batch_size: int | None = None,
    install_signal_handlers: bool = True,
):
    """Generic batch runner for plotting datasets.

    Parameters
    ----------
    items : iterable
        Iterable of item identifiers (any ``repr``-able objects).
    output_dir : str
        Base output directory; plots saved under ``output_dir/<item>/generic.png``.
    build_datasets_fn : callable
        Callable returning ``list[dict]`` describing datasets for an item.
    zoom_center_fn : callable, optional
        Callable mapping item -> center UNIX time (or ``None``) for zoom.
    zoom_window_seconds : float, optional
        Duration of zoom window in seconds.
    vertical_lines_fn : callable, optional
        Callable mapping item -> list[float] UNIX timestamps (or ``None``).
    y_scale : {'linear', 'log'}, default 'linear'
        Y-axis scaling for all rows.
    z_scale : {'linear', 'log'}, default 'linear'
        Color scaling for all rows.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    max_workers : int, default 2
        Number of parallel worker processes.
    progress_json_path : str, default PLOTTING_PROGRESS_JSON_PATH
        Path to progress JSON (resumable state). Created/updated as needed.
    ignore_progress_json : bool, default False
        If ``True``, skip reading existing progress prior to execution.
    flush_batch_size : int, default 10
        Progress/log batch size; values < 1 coerced to 1. Final partial batch flushed.
    log_flush_batch_size : int, optional
        Explicit log batch size; if ``None`` reuse ``flush_batch_size``.
    install_signal_handlers : bool, default True
        When True, a temporary SIGINT handler is installed (restored on exit) to
        enable graceful interruption (progress & log flush). Set False in embedded
        environments if altering the global handler causes side-effects.

    Returns
    -------
    list of tuple
        Sequence of ``(item, status)`` with ``status`` in {``'ok'``, ``'no_data'``, ``'error'``}.

    Notes
    -----
    * Logging is buffered and force-flushed at completion.
    * Progress JSON contains simple lists of completed, error, and no-data items.
    * Items are identified via ``repr(item)`` for data-agnostic persistence.
    """
    os.makedirs(output_dir, exist_ok=True)
    previous_sigint = None
    if install_signal_handlers:
        try:
            previous_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _sigint_handler)
        except Exception as _gbp_sig_setup_exc:
            log_message(
                f"[WARN] Could not install temporary SIGINT handler: {_gbp_sig_setup_exc}"
            )
    flush_batch_size = max(1, int(flush_batch_size))
    configure_log_batch(log_flush_batch_size or flush_batch_size)

    # Step: Load prior progress (if any)
    progress_state = {
        # Ordered list of repr(item) for successfully completed items.
        "completed_items": [],
        # Items that raised exceptions inside the worker.
        "errors": [],
        # Items that returned logically empty datasets (no plots generated).
        "no_data": [],
        # Sequential index counter for processed items (0-based).
        "last_index": -1,
        # For future schema migrations.
        "schema_version": 1,
    }
    if (not ignore_progress_json) and os.path.exists(progress_json_path):
        try:
            with open(progress_json_path, "r") as progress_in:
                loaded = json.load(progress_in)
            if isinstance(loaded, dict):
                # Merge known keys
                for k in progress_state.keys():
                    if k in loaded:
                        progress_state[k] = loaded[k]
        except Exception as progress_json_read_exception:
            log_error(
                f"[PROGRESS] Failed to read existing progress JSON '{progress_json_path}': {progress_json_read_exception}"
            )

    # Step: Determine pending items (skip already completed)
    item_list = list(items)
    completed_set = set(progress_state.get("completed_items", []))
    pending_items = [it for it in item_list if repr(it) not in completed_set]
    total_pending = len(pending_items)
    log_message(
        f"[BATCH] Starting generic batch plot with {total_pending} pending / {len(item_list)} total items; flush_batch_size={flush_batch_size} log_flush_batch_size={log_flush_batch_size or flush_batch_size}"
    )

    # Step: Define progress JSON flush helper
    pending_progress_write_count = (
        0  # number of in-memory updates since last JSON flush
    )

    def _flush_progress(force: bool = False):
        """Flush progress JSON to disk using batch semantics.

        Parameters
        ----------
        force : bool, default False
            If True, flush even if the number of pending updates is below the
            configured batch size.
        """
        nonlocal pending_progress_write_count
        if pending_progress_write_count == 0 and not force:
            return
        if (pending_progress_write_count >= flush_batch_size) or force:
            try:
                with open(progress_json_path, "w") as progress_out:
                    json.dump(progress_state, progress_out, indent=2)
                pending_progress_write_count = 0
            except Exception as progress_json_write_exception:
                log_error(
                    f"[PROGRESS] Failed writing progress JSON '{progress_json_path}': {progress_json_write_exception}"
                )

    # Step: Worker wrapper

    def _worker(item):
        try:
            datasets = build_datasets_fn(item)
            if not datasets:
                return (item, "no_data")
            center = zoom_center_fn(item) if zoom_center_fn else None
            vlines = vertical_lines_fn(item) if vertical_lines_fn else None
            fig, canvas = generic_plot_spectrogram_set(
                datasets,
                zoom_center=center,
                zoom_window_seconds=zoom_window_seconds,
                vertical_lines=vlines,
                y_scale=y_scale,
                z_scale=z_scale,
                colormap=colormap,
                show=False,
            )
            if fig is not None:
                od = os.path.join(output_dir, str(item))
                os.makedirs(od, exist_ok=True)
                out_path = os.path.join(od, "generic.png")
                fig.savefig(out_path, dpi=150)
                close_all_axes_and_clear(fig)
            return (item, "ok")
        except Exception as generic_exception:
            log_error(f"[GENERIC-FAIL] Item {item}: {generic_exception}")
            return (item, "error")

    results = []
    processed_item_count = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as process_pool:
        future_map = {
            process_pool.submit(_worker, item_identifier): item_identifier
            for item_identifier in pending_items
        }
        for finished_future in concurrent.futures.as_completed(future_map):
            original_item_identifier = future_map[finished_future]
            try:
                item_identifier, status = finished_future.result()
            except Exception as generic_batch_future_exception:
                status = "error"
                item_identifier = original_item_identifier
                log_error(
                    f"[GENERIC-FAIL] Item {original_item_identifier} outer exception: {generic_batch_future_exception}"
                )
            results.append((item_identifier, status))
            # Progress classification & state update
            item_repr = repr(item_identifier)
            if status == "ok":
                progress_state["completed_items"].append(item_repr)
            elif status == "no_data":
                progress_state["no_data"].append(item_repr)
            else:
                progress_state["errors"].append(item_repr)
            processed_item_count += 1
            progress_state["last_index"] = processed_item_count - 1
            pending_progress_write_count += 1
            _flush_progress(force=False)
    # Step: Final flushes
    _flush_progress(force=True)
    _flush_log_buffer(force=True)
    log_message(
        (
            "[BATCH] Completed generic batch plot: "
            f"{processed_item_count} processed (ok={sum(1 for _, s in results if s == 'ok')} "
            f"no_data={sum(1 for _, s in results if s == 'no_data')} "
            f"error={sum(1 for _, s in results if s == 'error')})"
        ),
        force_flush=True,
    )
    # Restore prior handler if we replaced it
    if install_signal_handlers and previous_sigint is not None:
        try:
            signal.signal(signal.SIGINT, previous_sigint)
        except Exception as _gbp_sig_restore_exc:
            log_message(
                f"[WARN] Could not restore original SIGINT handler: {_gbp_sig_restore_exc}"
            )
    return results


def generic_plot_multirow_optional_zoom(
    datasets,
    vertical_lines=None,
    zoom_duration_minutes=6.25,
    y_scale="linear",
    z_scale="linear",
    colormap="viridis",
    show=False,
    title=None,
    row_label_pad=50,
    row_label_rotation=90,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
):
    """Render a multi-row spectrogram grid with an optional zoom column.

    Parameters
    ----------
    datasets : list of dict
        Each dict must contain keys:

        * ``'x'`` – 1D UNIX epoch seconds (float) array
        * ``'y'`` – 1D energy (eV) array (unfiltered, 0–4000 typical)
        * ``'data'`` – 3D ndarray that can be collapsed (time, pitch/angle, energy)

        Optional per-row keys (all honored when present):

        * ``'label'`` – Row label placed on the left (rotated)
        * ``'y_label'`` – Units label for y-axis (default: ``'Energy (eV)'``)
        * ``'z_label'`` – Color scale label (default: ``'Counts'``)
        * ``'y_min'`` / ``'y_max'`` – Energy bounds (overrides global ``y_min`` / ``y_max`` args)
        * ``'z_min'`` / ``'z_max'`` – Color bounds (overrides global ``z_min`` / ``z_max`` args)
        * ``'vmin'`` / ``'vmax'`` – Precomputed percentile (or fixed) color bounds used when
            ``z_min`` / ``z_max`` not provided. (``vmin``/``vmax`` are interpreted as the *row's*
            native bounds; global ``z_min`` / ``z_max`` will still clamp if supplied.)
    vertical_lines : list of float, optional
        UNIX timestamps defining event/selection markers and potential zoom window.
    zoom_duration_minutes : float, default 6.25
        Desired zoom window length in minutes (may auto-expand to include full marked span).
    y_scale : {'linear', 'log'}, default 'linear'
        Y-axis scaling.
    z_scale : {'linear', 'log'}, default 'linear'
        Color (intensity) scale.
    colormap : str, default 'viridis'
        Matplotlib colormap.
    show : bool, default False
        If ``True``, display interactively.
    title : str, optional
        Figure suptitle.
    row_label_pad : int, default 50
        Padding for row labels.
    row_label_rotation : int, default 90
        Rotation angle (degrees) for row labels.
    y_min, y_max, z_min, z_max : float, optional
        Global override bounds applied uniformly when provided. Any per-row
        ``y_min`` / ``y_max`` / ``z_min`` / ``z_max`` in a dataset dict take
        precedence. When neither global nor per-row color bounds are supplied
        the function relies on each dataset's ``vmin`` / ``vmax`` (if present)
        else falls back to internal percentile selection in lower-level calls.

    Returns
    -------
    tuple
        ``(fig, canvas)`` or ``(None, None)`` if ``datasets`` is empty.

    Notes
    -----
    * Determines need for a zoom column dynamically: only rendered if at least
        one dataset contains non-NaN values inside the computed zoom window.
    * Y-axis and colorbar labels default to "Energy (eV)" and "Counts" when a
        dataset omits explicit ``y_label`` / ``z_label`` (defaults originate in
        ``generic_plot_spectrogram_set`` / dataset assembly).
    """
    if not datasets:
        return None, None
    # Determine zoom window & whether needed
    zoom_needed = False
    center_value = None
    duration = None
    if vertical_lines and len(vertical_lines) > 0:
        if len(vertical_lines) == 1:
            center_value = vertical_lines[0]
            duration = zoom_duration_minutes * 60
        else:
            center_value = 0.5 * (vertical_lines[0] + vertical_lines[1])
            min_window = abs(vertical_lines[1] - vertical_lines[0]) * 1.5
            requested_window = zoom_duration_minutes * 60
            duration = max(requested_window, min_window)
        left = center_value - duration / 2
        right = center_value + duration / 2
        for ds in datasets:
            t = ds["x"]
            d = ds["data"]
            mask_zoom = (t >= left) & (t <= right)
            # Require some non-NaN data inside window
            if np.any(~np.isnan(d[mask_zoom])):
                zoom_needed = True
                break
    number_rows = len(datasets)
    number_columns = 2 if zoom_needed else 1
    fig = Figure(figsize=(12 * number_columns, 3 * number_rows))
    canvas = FigureCanvas(fig)
    axes = np.empty((number_rows, number_columns), dtype=object)
    for i in range(number_rows):
        for j in range(number_columns):
            axes[i, j] = fig.add_subplot(
                number_rows, number_columns, i * number_columns + j + 1
            )
    # Plot rows
    for i, ds in enumerate(datasets):
        times = ds["x"]
        energy = ds["y"]
        data3d = ds["data"]
        vmin = ds.get("vmin")
        vmax = ds.get("vmax")
        # Full
        make_spectrogram(
            x_axis_values=times,
            y_axis_values=energy,
            data_array_3d=data3d,
            collapse_axis=1,
            x_axis_min=times[0],
            x_axis_max=times[-1],
            x_axis_is_unix=True,
            instrument_label=None,
            y_axis_scale_function=y_scale,
            z_axis_scale_function=z_scale,
            vertical_lines_unix=vertical_lines,
            z_axis_min=vmin if z_min is None else z_min,
            z_axis_max=vmax if z_max is None else z_max,
            axis_object=axes[i, 0],
            colormap=colormap,
        )
        # Zoom
        if number_columns == 2:
            make_spectrogram(
                x_axis_values=times,
                y_axis_values=energy,
                data_array_3d=data3d,
                collapse_axis=1,
                center_timestamp=center_value,
                window_duration_seconds=duration,
                x_axis_is_unix=True,
                instrument_label=None,
                y_axis_scale_function=y_scale,
                z_axis_scale_function=z_scale,
                vertical_lines_unix=vertical_lines,
                z_axis_min=vmin if z_min is None else z_min,
                z_axis_max=vmax if z_max is None else z_max,
                axis_object=axes[i, 1],
                colormap=colormap,
            )
    # Row labels
    for i, ds in enumerate(datasets):
        axes[i, 0].set_ylabel(
            ds.get("label", ""),
            fontsize=AXIS_LABEL_FONT_SIZE,
            rotation=row_label_rotation,
            labelpad=row_label_pad,
            va="center",
        )
    # Column headers
    if number_columns == 2:
        axes[0, 0].set_title("Full", fontsize=AXIS_LABEL_FONT_SIZE)
        axes[0, 1].set_title("Zoomed", fontsize=AXIS_LABEL_FONT_SIZE)
    else:
        axes[0, 0].set_title("Full", fontsize=AXIS_LABEL_FONT_SIZE)
    # Title
    if title:
        fig.suptitle(title, fontsize=AXIS_LABEL_FONT_SIZE + 2)
    # Timespan annotation (use first dataset times)
    base_times = datasets[0]["x"]
    t0 = datetime.fromtimestamp(base_times[0], tz=timezone.utc)
    t1 = datetime.fromtimestamp(base_times[-1], tz=timezone.utc)
    data_timespan_str = f"Data timespan: {t0.strftime('%Y-%m-%d %H:%M:%S')} to {t1.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    marked_str = ""
    if vertical_lines and len(vertical_lines) > 0:
        v0 = datetime.fromtimestamp(min(vertical_lines), tz=timezone.utc)
        v1 = datetime.fromtimestamp(max(vertical_lines), tz=timezone.utc)
        marked_str = f"\nMarked range: {v0.strftime('%Y-%m-%d %H:%M:%S')} to {v1.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.01, data_timespan_str, ha="center", va="bottom", fontsize=13)
    if marked_str:
        fig.text(
            0.5,
            0.045,
            marked_str.strip(),
            ha="center",
            va="bottom",
            fontsize=13,
            color="red",
        )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig, canvas
