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
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
]

__date__: str = "2026-06-17"
__status__: str = "Development"
__version__: str = "0.0.2"
__license__: str = "GPL-3.0"

import concurrent.futures
import gc
import json
import math
import os
import signal
import sys
import traceback
import time as _time
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import cdflib
import numpy as np
from tqdm import tqdm

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

# FAST-specific paths
FAST_CDF_DATA_FOLDER_PATH = "./FAST_data/"
FAST_FILTERED_ORBITS_CSV_PATH = "./FAST_Cusp_Indices.csv"
FAST_PLOTTING_PROGRESS_JSON = "./batch_multi_plot_FAST_progress.json"
FAST_LOGFILE_PATH = "./batch_multi_plot_FAST_log.log"
FAST_OUTPUT_BASE = "./FAST_plots/"
FAST_LOGFILE_DATETIME_PATH = "./batch_multi_plot_FAST_logfile_datetime.txt"
FAST_COLLAPSE_FUNCTION = np.nansum
CDF_VARIABLES = ["time_unix", "data", "energy", "pitch_angle"]

# Colormaps for each axis-scaling combination (colorblind-friendly and visually distinct)
DEFAULT_COLORMAP_LINEAR_Y_LINEAR_Z = "viridis"
DEFAULT_COLORMAP_LINEAR_Y_LOG_Z = "cividis"
DEFAULT_COLORMAP_LOG_Y_LINEAR_Z = "plasma"
DEFAULT_COLORMAP_LOG_Y_LOG_Z = "inferno"

# Buffered logging configuration
INFO_LOG_BATCH_SIZE_DEFAULT = 10
_INFO_LOG_BATCH_SIZE = INFO_LOG_BATCH_SIZE_DEFAULT
_INFO_LOG_BUFFER: list[tuple[str, str]] = []


# Module-level helpers


def _load_or_create_logfile_datetime(path: str) -> str:
    """Read or create the persistent logfile datetime stamp."""
    if os.path.exists(path) and (text := Path(path).read_text().strip()):
        return text
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(path).write_text(stamp)
    return stamp


FAST_LOGFILE_DATETIME_STRING = _load_or_create_logfile_datetime(
    FAST_LOGFILE_DATETIME_PATH
)


def _parse_year_month(file_path: str) -> tuple[str, str]:
    """Extract (year, month) from a CDF path containing a YYYY/MM directory pair.

    Returns ``('unknown', 'unknown')`` when the pattern is not found.
    """
    parts = Path(file_path).parts
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 4:
            nxt = parts[i + 1] if i + 1 < len(parts) else ""
            month = nxt if nxt.isdigit() and len(nxt) == 2 else "unknown"
            return part, month
    return "unknown", "unknown"


def _extrema_overrides(
    global_extrema: dict | None,
    inst: str,
    y_scale: str,
    z_scale: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract and round per-instrument axis limits from a global extrema dict.

    Returns ``(y_min, y_max, z_min, z_max)`` with rounded values when keys are
    present in *global_extrema*, or ``(None, None, None, None)`` otherwise.
    """
    if not isinstance(global_extrema, dict):
        return None, None, None, None
    kb = f"{inst}_{y_scale}_{z_scale}"

    def _r(v: float | None, d: str) -> float | None:
        return round_extrema(v, d) if v is not None else None

    return (
        _r(global_extrema.get(f"{kb}_y_min"), "down"),
        _r(global_extrema.get(f"{kb}_y_max"), "up"),
        _r(global_extrema.get(f"{kb}_z_min"), "down"),
        _r(global_extrema.get(f"{kb}_z_max"), "up"),
    )


def _classify_error_reason(msg: str) -> str:
    """Map an error message to a short reason token for progress JSON keys."""
    m = msg.lower()
    if "divide" in m and "zero" in m:
        return "divide-by-zero"
    if "invalid" in m and "cdf" in m:
        return "invalid-cdf"
    if "timeout" in m:
        return "timeout"
    if "plot" in m:
        return "plotting"
    return "generic"


def _add_to_orbit_list(pdisk: dict, key: str, orbit: int) -> None:
    """Add *orbit* to the sorted list at *pdisk[key]*, creating the key if absent."""
    pdisk[key] = sorted(set(pdisk.get(key, [])) | {orbit})


def extract_orbit_and_instrument(cdf_path: str) -> tuple[int, str, str] | None:
    """Parse a CDF filename to ``(orbit_number, instrument_type, cdf_path)``.

    Returns ``None`` when the filename does not match the expected pattern, the
    orbit number cannot be parsed, or the instrument type is ``None`` or ``'orb'``.
    """
    filename = os.path.basename(cdf_path)
    parts = filename.split("_")
    if len(parts) < 5:
        return None
    try:
        orbit_number = int(parts[-2])
    except Exception as exc:
        info_logger(
            f"[ERROR] Invalid orbit number in filename: {filename}",
            exc,
            level="message",
        )
        return None
    instrument_type = get_cdf_file_type(cdf_path)
    if instrument_type is None or instrument_type == "orb":
        return None
    return (orbit_number, instrument_type, cdf_path)


# Logging


def configure_info_logger_batch(batch_size: int) -> None:
    """Set the buffered info logging batch size (values < 1 become 1)."""
    global _INFO_LOG_BATCH_SIZE
    _INFO_LOG_BATCH_SIZE = max(1, batch_size)


def flush_info_logger_buffer(force: bool = True) -> None:
    """Flush all buffered log messages immediately."""
    global _INFO_LOG_BUFFER
    for level, msg in _INFO_LOG_BUFFER:
        try:
            (log_error if level == "error" else log_message)(msg)
        except Exception:
            print(msg, file=sys.stderr if level == "error" else sys.stdout)
    _INFO_LOG_BUFFER = []


def info_logger(
    prefix: str,
    exception: BaseException | None = None,
    level: str = "error",
    include_trace: bool = False,
    force_flush: bool = False,
) -> None:
    """Log a message (optionally with an exception and traceback) to the buffered log.

    Parameters
    ----------
    prefix : str
        Human-readable message prefix.
    exception : BaseException or None
        Optional exception; if given, the class name and value are appended.
    level : {'error', 'message'}, default 'error'
        'error' routes to ``log_error``; otherwise to ``log_message``.
    include_trace : bool, default False
        If True and an exception is given, include a formatted traceback.
    force_flush : bool, default False
        Force an immediate buffer flush after this message.

    Notes
    -----
    Falls back to print on stdout/stderr if the underlying logger fails. When
    an exception is provided the message is formatted as
    ``"{prefix} [<ExceptionClass>]: {exception}"``.
    """
    name = type(exception).__name__ if exception is not None else None
    message = f"{prefix} [{name}]: {exception}" if name else str(prefix)

    trace_lines: list[str] = []
    if include_trace and exception is not None:
        try:
            trace = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
            trace_lines.append("[TRACE]\n" + trace)
        except Exception:
            pass

    try:
        _INFO_LOG_BUFFER.append((level, message))
        for tr in trace_lines:
            _INFO_LOG_BUFFER.append(("message", tr))
        if (
            force_flush
            or _INFO_LOG_BATCH_SIZE <= 1
            or len(_INFO_LOG_BUFFER) >= _INFO_LOG_BATCH_SIZE
        ):
            flush_info_logger_buffer(force=True)
    except Exception:
        try:
            (log_error if level == "error" else log_message)(message)
        except Exception:
            try:
                print(message)
            except Exception:
                pass


def _terminate_all_child_processes() -> None:
    """Best-effort terminate all child processes via psutil (errors suppressed)."""
    try:
        import psutil
    except Exception:
        return
    try:
        for child in psutil.Process().children(recursive=True):
            try:
                child.terminate()
            except Exception:
                pass
    except Exception:
        pass


# Plotting helpers


def round_extrema(value: float | int, direction: str) -> float:
    """Round an extrema value to a clean significant-digit axis limit.

    Rounds to the next significant digit in the specified direction so plot
    axis limits look consistent (e.g. 1234 → 1300 for 'up').

    Parameters
    ----------
    value : float or int
        Extrema value. Zero returns 0.0.
    direction : {'up', 'down'}
        Round up (for maxima) or down (for minima).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If direction is not ``'up'`` or ``'down'``.

    Examples
    --------
    >>> round_extrema(1234, 'up')
    1300.0
    >>> round_extrema(0.0123, 'down')
    0.012
    """
    if value == 0:
        return 0.0
    factor = 10 ** (math.floor(math.log10(abs(value))) - 1)
    if direction == "up":
        return float(math.ceil(value / factor) * factor)
    if direction == "down":
        return float(math.floor(value / factor) * factor)
    raise ValueError(f"Invalid direction: {direction}")


def FAST_plot_pitch_angle_grid(
    cdf_file_path: str,
    filtered_orbits_df=None,
    orbit_number: int | None = None,
    zoom_duration_minutes: float = 6.25,
    scale_function_y: str = "linear",
    scale_function_z: str = "linear",
    pitch_angle_categories: dict[str, list[tuple[float, float]]] | None = None,
    show: bool = True,
    colormap: str = "viridis",
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[Any, Any]:
    """Plot a grid of ESA spectrograms collapsed by pitch-angle categories.

    Each row corresponds to a pitch-angle category (e.g. downgoing, upgoing,
    perpendicular, all). If orbit boundary timestamps are available a zoom
    column is added. Data are collapsed over pitch-angle via
    ``FAST_COLLAPSE_FUNCTION`` (``np.nansum`` by default).

    Parameters
    ----------
    cdf_file_path : str
        Path to the instrument CDF file.
    filtered_orbits_df : pandas.DataFrame or None
        DataFrame used to compute vertical lines; if None, lines are omitted.
    orbit_number : int or None
        Orbit number used to label vertical lines.
    zoom_duration_minutes : float, default 6.25
        Window length (minutes) for the optional zoom column.
    scale_function_y : {'linear', 'log'}, default 'linear'
        Y-axis scaling.
    scale_function_z : {'linear', 'log'}, default 'linear'
        Color scale for intensity.
    pitch_angle_categories : dict or None
        Mapping of label -> list of (min_deg, max_deg) ranges; defaults to
        the four standard groups when None.
    show : bool, default True
        If True, display the figure interactively.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    y_min, y_max : float or None, optional
        Energy (y-axis) limits; defaults to [0, 4000] when None.
    z_min, z_max : float or None, optional
        Color scale limits; defaults to row-level 1st/99th percentiles when None.

    Returns
    -------
    tuple[Figure or None, FigureCanvasBase or None]
        Figure and Canvas, or ``(None, None)`` when no datasets are produced.
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
        if not vertical_lines:
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
    y_lower = 0 if y_min is None else y_min
    y_upper = 4000 if y_max is None else y_max
    valid_energy_mask = (energy >= y_lower) & (energy <= y_upper)

    datasets = []
    for key in pa_keys:
        mask = np.zeros_like(pitchangle, dtype=bool)
        for rng in pitch_angle_categories[key]:
            mask |= (pitchangle >= rng[0]) & (pitchangle <= rng[1])
        pa_data = data[:, mask, :]
        matrix_full = FAST_COLLAPSE_FUNCTION(pa_data, axis=1)
        nan_col_mask = ~np.all(np.isnan(matrix_full), axis=0)
        matrix_full = matrix_full[:, nan_col_mask & valid_energy_mask]
        matrix_full_plot = matrix_full.T
        if matrix_full_plot.size == 0:
            continue
        vmin = z_min if z_min is not None else np.nanpercentile(matrix_full_plot, 1)
        vmax = z_max if z_max is not None else np.nanpercentile(matrix_full_plot, 99)
        datasets.append(
            {
                "x": times,
                "y": energy,
                "data": pa_data,
                "label": key.title(),
                "y_label": "Energy (eV)",
                "z_label": "Counts",
                "vmin": vmin,
                "vmax": vmax,
                "y_min": y_lower,
                "y_max": y_upper,
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
    cdf_file_paths: dict[str, str],
    filtered_orbits_df=None,
    orbit_number: int | None = None,
    zoom_duration_minutes: float = 6.25,
    scale_function_y: str = "linear",
    scale_function_z: str = "linear",
    instrument_order: tuple[str, ...] = ("ees", "eeb", "ies", "ieb"),
    show: bool = True,
    colormap: str = "viridis",
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    global_extrema: dict[str, int | float] | None = None,
) -> tuple[Any, Any]:
    """Plot a multi-instrument ESA spectrogram grid for a single orbit.

    Loads each instrument CDF, collapses across pitch-angle, and constructs
    datasets for ``generic_plot_multirow_optional_zoom``. A zoom column is
    included when vertical lines are available for the orbit.

    Parameters
    ----------
    cdf_file_paths : dict of {str: str}
        Mapping of instrument key (``'ees'``, ``'eeb'``, ``'ies'``, ``'ieb'``)
        to CDF file path. Missing instruments are skipped.
    filtered_orbits_df : pandas.DataFrame or None
        DataFrame for vertical line computation; None omits lines.
    orbit_number : int or None
        Orbit identifier used in titles and vertical lines.
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
        Global fallback axis/color limits used when ``global_extrema`` does not
        supply an instrument-specific key.
    global_extrema : dict or None
        Precomputed extrema keyed as ``{instrument}_{y_scale}_{z_scale}_{axis}_{min|max}``
        supplying per-instrument limits. Takes precedence over the direct
        ``y_min`` / ``y_max`` / ``z_min`` / ``z_max`` arguments.

    Returns
    -------
    tuple[Figure or None, FigureCanvasBase or None]
        Figure and Canvas, or ``(None, None)`` when no datasets are produced.

    Notes
    -----
    - Files that fail to load are logged and skipped.
    - Energy bins are restricted to ``[0, 4000]`` unless overridden.
    - ``vmin``/``vmax`` per row use 1st/99th percentiles unless ``global_extrema``
      provides per-instrument ``z_min`` / ``z_max``.
    """
    datasets = []
    vertical_lines = None
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

            if (
                vertical_lines is None
                and filtered_orbits_df is not None
                and orbit_number is not None
            ):
                instrument_type = get_cdf_file_type(cdf_path)
                vertical_lines = get_timestamps_for_orbit(
                    filtered_orbits_df, orbit_number, instrument_type, times
                )
                if not vertical_lines:
                    info_logger(
                        f"No vertical lines found for orbit {orbit_number} in {cdf_path}. Skipping.",
                        level="message",
                    )

            # Determine per-instrument y and z bounds
            if isinstance(global_extrema, dict):
                key_prefix = f"{inst}_{scale_function_y}_{scale_function_z}"
                y_lower = global_extrema.get(
                    f"{key_prefix}_y_min", 0 if y_min is None else y_min
                )
                y_upper = global_extrema.get(
                    f"{key_prefix}_y_max", 4000 if y_max is None else y_max
                )
                vmin = global_extrema.get(f"{key_prefix}_z_min")
                vmax = global_extrema.get(f"{key_prefix}_z_max")
            else:
                y_lower = 0 if y_min is None else y_min
                y_upper = 4000 if y_max is None else y_max
                vmin = vmax = None

            matrix_full = FAST_COLLAPSE_FUNCTION(data, axis=1)
            nan_col_mask = ~np.all(np.isnan(matrix_full), axis=0)
            valid_energy_mask = (energy >= y_lower) & (energy <= y_upper)
            matrix_full = matrix_full[:, nan_col_mask & valid_energy_mask]
            matrix_full_plot = matrix_full.T
            if matrix_full_plot.size == 0:
                continue

            if vmin is None:
                vmin = np.nanpercentile(matrix_full_plot, 1)
            if vmax is None:
                vmax = np.nanpercentile(matrix_full_plot, 99)

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
                    **({"z_min": z_min} if z_min is not None else {}),
                    **({"z_max": z_max} if z_max is not None else {}),
                }
            )
        except Exception as exc:
            info_logger(
                f"Failed to load CDF for {inst} at {cdf_path}. Skipping.",
                exc,
                level="error",
            )

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


def FAST_process_single_orbit(
    orbit_number: int,
    instrument_file_paths: dict[str, str],
    filtered_orbits_dataframe,
    zoom_duration_minutes: float,
    y_axis_scale: str,
    z_axis_scale: str,
    instrument_order: tuple[str, ...],
    colormap: str,
    output_base_directory: str,
    orbit_timeout_seconds: int | float = 60,
    instrument_timeout_seconds: int | float = 30,
    global_extrema: dict[str, int | float] | None = None,
    override_plots: bool = True,
) -> dict[str, Any]:
    """Process and save all ESA spectrogram plots for a single orbit.

    For each available instrument, generates two plot versions:
    1. Using ``global_extrema`` (``_given_extrema`` suffix) if provided.
    2. Raw (per-file) extrema (``_raw`` suffix).
    This applies to both pitch-angle and instrument-grid plots. Figures are
    accumulated and only saved if no timeout thresholds are exceeded.

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
        Maximum wall-clock seconds for the entire orbit.
    instrument_timeout_seconds : int or float, default 30
        Per-instrument/grid timeout.
    global_extrema : dict or None
        Precomputed extrema mapping (from ``compute_global_extrema``).
    override_plots : bool, default True
        If False, skip plotting when the output file already exists.

    Returns
    -------
    dict
        Result with keys ``orbit`` (int), ``status`` (``'ok'``, ``'error'``, or
        ``'timeout'``), ``errors`` (list of str), and optionally
        ``timeout_type`` / ``timeout_instrument``.
    """
    result: dict[str, Any] = {"orbit": orbit_number, "status": "ok", "errors": []}
    orbit_start_time = _time.time()
    pending_figures: list[dict[str, Any]] = []
    timeout_triggered = False
    timeout_type = None
    timeout_instrument = None

    try:
        first_path = next(
            (
                instrument_file_paths[k]
                for k in ("ees", "eeb", "ies", "ieb")
                if k in instrument_file_paths
            ),
            None,
        )
        year, month = (
            _parse_year_month(first_path) if first_path else ("unknown", "unknown")
        )
        output_dir = os.path.join(
            output_base_directory, str(year), str(month), str(orbit_number)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Per-instrument pitch-angle plots (given_extrema and raw)
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
                cusp = bool(vertical_lines)
                cusp_tag = "_cusp" if cusp else ""
                y_min_ov, y_max_ov, z_min_ov, z_max_ov = _extrema_overrides(
                    global_extrema, inst_detected, y_axis_scale, z_axis_scale
                )

                # Given-extrema version
                fig_given, canvas_given = FAST_plot_pitch_angle_grid(
                    cdf_path,
                    filtered_orbits_df=filtered_orbits_dataframe,
                    orbit_number=orbit_number,
                    zoom_duration_minutes=zoom_duration_minutes,
                    scale_function_y=y_axis_scale,
                    scale_function_z=z_axis_scale,
                    show=False,
                    colormap=colormap,
                    y_min=y_min_ov,
                    y_max=y_max_ov,
                    z_min=z_min_ov,
                    z_max=z_max_ov,
                )
                if fig_given is not None:
                    fname = (
                        f"{orbit_number}{cusp_tag}_pitch-angle_ESA_{inst_detected}"
                        f"_y-{y_axis_scale}_z-{z_axis_scale}_given_extrema.png"
                    )
                    out_path = os.path.join(output_dir, fname)
                    if not override_plots and os.path.exists(out_path):
                        info_logger(
                            f"[SKIP] Plot already exists, skipping: {out_path}",
                            level="message",
                        )
                    else:
                        pending_figures.append(
                            {
                                "figure": fig_given,
                                "canvas": canvas_given,
                                "path": out_path,
                                "desc": f"pitch-angle {inst_detected} (given extrema)",
                            }
                        )

                # Raw version
                fig_raw, canvas_raw = FAST_plot_pitch_angle_grid(
                    cdf_path,
                    filtered_orbits_df=filtered_orbits_dataframe,
                    orbit_number=orbit_number,
                    zoom_duration_minutes=zoom_duration_minutes,
                    scale_function_y=y_axis_scale,
                    scale_function_z=z_axis_scale,
                    show=False,
                    colormap=colormap,
                )
                if fig_raw is not None:
                    fname = (
                        f"{orbit_number}{cusp_tag}_pitch-angle_ESA_{inst_detected}"
                        f"_y-{y_axis_scale}_z-{z_axis_scale}_raw.png"
                    )
                    out_path = os.path.join(output_dir, fname)
                    if not override_plots and os.path.exists(out_path):
                        info_logger(
                            f"[SKIP] Plot already exists, skipping: {out_path}",
                            level="message",
                        )
                    else:
                        pending_figures.append(
                            {
                                "figure": fig_raw,
                                "canvas": canvas_raw,
                                "path": out_path,
                                "desc": f"pitch-angle {inst_detected} (raw extrema)",
                            }
                        )

            except Exception as exc:
                err = f"[FAIL] Plotting Orbit {orbit_number} pitch angle grid for {inst_type}"
                info_logger(err, exc, level="error")
                result["status"] = "error"
                result["errors"].append(err)
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
                        f"[TIMEOUT] Instrument {inst_type} in orbit {orbit_number} exceeded "
                        f"{instrument_timeout_seconds:.0f}s ({inst_elapsed:.2f}s). Aborting.",
                        level="message",
                    )

        # Instrument-grid plots (given_extrema and raw)
        if not timeout_triggered:
            grid_start = _time.time()
            try:
                fig_grid_given, canvas_grid_given = FAST_plot_instrument_grid(
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
                if fig_grid_given is not None:
                    fname = f"{orbit_number}_instrument-grid_ESA_y-{y_axis_scale}_z-{z_axis_scale}_given_extrema.png"
                    out_path = os.path.join(output_dir, fname)
                    if not override_plots and os.path.exists(out_path):
                        info_logger(
                            f"[SKIP] Plot already exists, skipping: {out_path}",
                            level="message",
                        )
                    else:
                        pending_figures.append(
                            {
                                "figure": fig_grid_given,
                                "canvas": canvas_grid_given,
                                "path": out_path,
                                "desc": "instrument-grid (given extrema)",
                            }
                        )

                fig_grid_raw, canvas_grid_raw = FAST_plot_instrument_grid(
                    instrument_file_paths,
                    filtered_orbits_df=filtered_orbits_dataframe,
                    orbit_number=orbit_number,
                    zoom_duration_minutes=zoom_duration_minutes,
                    scale_function_y=y_axis_scale,
                    scale_function_z=z_axis_scale,
                    instrument_order=instrument_order,
                    show=False,
                    colormap=colormap,
                    global_extrema=None,
                )
                if fig_grid_raw is not None:
                    fname = f"{orbit_number}_instrument-grid_ESA_y-{y_axis_scale}_z-{z_axis_scale}_raw.png"
                    out_path = os.path.join(output_dir, fname)
                    if not override_plots and os.path.exists(out_path):
                        info_logger(
                            f"[SKIP] Plot already exists, skipping: {out_path}",
                            level="message",
                        )
                    else:
                        pending_figures.append(
                            {
                                "figure": fig_grid_raw,
                                "canvas": canvas_grid_raw,
                                "path": out_path,
                                "desc": "instrument-grid (raw extrema)",
                            }
                        )

            except Exception as exc:
                err = f"[FAIL] Plotting Orbit {orbit_number} instrument grid"
                info_logger(err, exc, level="error")
                result["status"] = "error"
                result["errors"].append(err)
            finally:
                grid_elapsed = _time.time() - grid_start
                info_logger(
                    f"[TIMING] Orbit {orbit_number} instrument-grid elapsed {grid_elapsed:.3f}s",
                    level="message",
                )
                if grid_elapsed > instrument_timeout_seconds and not timeout_triggered:
                    timeout_triggered = True
                    timeout_type = "instrument"
                    timeout_instrument = "instrument_grid"
                    info_logger(
                        f"[TIMEOUT] Instrument grid in orbit {orbit_number} exceeded "
                        f"{instrument_timeout_seconds:.0f}s ({grid_elapsed:.2f}s). Aborting.",
                        level="message",
                    )

        # Orbit total timeout check
        orbit_elapsed = _time.time() - orbit_start_time
        if orbit_elapsed > orbit_timeout_seconds and not timeout_triggered:
            timeout_triggered = True
            timeout_type = "orbit"
            info_logger(
                f"[TIMEOUT] Orbit {orbit_number} exceeded {orbit_timeout_seconds:.0f}s total "
                f"({orbit_elapsed:.2f}s). Aborting without saving.",
                level="message",
            )

        if timeout_triggered:
            for fig_item in pending_figures:
                try:
                    close_all_axes_and_clear(fig_item["figure"])
                except Exception:
                    pass
            pending_figures.clear()
            result["status"] = "timeout"
            result["timeout_type"] = timeout_type
            if timeout_instrument:
                result["timeout_instrument"] = timeout_instrument
            return result

        # Save accumulated figures
        for fig_item in pending_figures:
            fig, fpath = fig_item["figure"], fig_item["path"]
            try:
                info_logger(
                    f"[DEBUG] Saving {fig_item['desc']} plot: y_axis_scale={y_axis_scale}, "
                    f"z_axis_scale={z_axis_scale}, filename={fpath}",
                    level="message",
                )
                fig.savefig(fpath, dpi=200)
                info_logger(f"[SAVED] {fpath}", level="message")
            except Exception as exc:
                info_logger(f"[FAIL] Saving figure {fpath}", exc, level="error")
                result["status"] = "error"
                result["errors"].append(str(exc))
            finally:
                try:
                    close_all_axes_and_clear(fig)
                except Exception:
                    pass
        pending_figures.clear()

    except Exception as exc:
        err = f"[FAIL] Orbit {orbit_number} processing"
        info_logger(err, exc, level="error")
        result["status"] = "error"
        result["errors"].append(err)
    finally:
        gc.collect()

    return result


# Batch driver


def FAST_plot_spectrograms_directory(
    directory_path: str = FAST_CDF_DATA_FOLDER_PATH,
    output_base: str = FAST_OUTPUT_BASE,
    y_scale: str = "linear",
    z_scale: str = "log",
    zoom_duration_minutes: float = DEFAULT_ZOOM_WINDOW_MINUTES,
    instrument_order: tuple[str, ...] = ("ees", "eeb", "ies", "ieb"),
    verbose: bool = True,
    progress_json_path: str | None = FAST_PLOTTING_PROGRESS_JSON,
    ignore_progress_json: bool = False,
    use_tqdm: bool | None = None,
    colormap: str = "viridis",
    max_workers: int = 4,
    orbit_timeout_seconds: int | float = 60,
    instrument_timeout_seconds: int | float = 30,
    retry_timeouts: bool = True,
    flush_batch_size: int = 10,
    log_flush_batch_size: int | None = None,
    max_processing_percentile: float | None = None,
    override_plots: bool = True,
) -> list[dict[str, Any]]:
    """Batch process ESA spectrogram plots for all orbits in a directory.

    Discovers instrument CDF files (excluding ``_orb_``), groups them by orbit,
    and processes each orbit in parallel worker processes (safe for matplotlib).
    Progress is persisted to a JSON file to support resumable runs. When
    ``max_processing_percentile`` is not None, a global extrema pass runs first
    (``compute_global_extrema``) and both raw and given-extrema plots are saved;
    otherwise only raw plots are produced.

    Parameters
    ----------
    directory_path : str, default FAST_CDF_DATA_FOLDER_PATH
        Root folder containing CDF files.
    output_base : str, default FAST_OUTPUT_BASE
        Base output directory; plots are saved under ``output_base/year/month/orbit``.
    y_scale : {'linear', 'log'}, default 'linear'
        Y-axis scaling.
    z_scale : {'linear', 'log'}, default 'log'
        Color scale for intensity.
    zoom_duration_minutes : float, default DEFAULT_ZOOM_WINDOW_MINUTES
        Zoom window length for zoom columns.
    instrument_order : tuple of str, default ('ees', 'eeb', 'ies', 'ieb')
        Display order for the instrument grid.
    verbose : bool, default True
        Print additional batch messages when True.
    progress_json_path : str or None, default FAST_PLOTTING_PROGRESS_JSON
        Path to persist progress across runs; None disables persistence.
    ignore_progress_json : bool, default False
        If True, do not read existing progress before starting.
    use_tqdm : bool or None, default None
        Show a tqdm progress bar when True; defaults to False when None.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    max_workers : int, default 4
        Max number of worker processes.
    orbit_timeout_seconds : int or float, default 60
        Total per-orbit timeout (seconds).
    instrument_timeout_seconds : int or float, default 30
        Per-instrument/grid timeout (seconds).
    retry_timeouts : bool, default True
        If True, retry timed-out orbits once with a smaller pool.
    flush_batch_size : int, default 10
        Orbit completions between progress/extrema JSON writes. Values < 1
        become 1. Final partial batch always flushes.
    log_flush_batch_size : int or None, default None
        Logging buffer batch size; defaults to ``flush_batch_size`` when None.
    max_processing_percentile : float or None, default None
        Percentile (0–100] for pooled intensity (Z) maxima in
        ``compute_global_extrema``. None skips the extrema pass and raw-only
        plots are produced. Energy (Y) maxima use a fixed 99% cumulative
        coverage rule regardless.
    override_plots : bool, default True
        If False, skip plots whose output file already exists.

    Returns
    -------
    list of dict
        Result dictionaries from ``FAST_process_single_orbit`` (and retries).

    Raises
    ------
    KeyboardInterrupt
        Re-raised on SIGINT/SIGTERM so the caller can stop multi-combo loops.

    Notes
    -----
    - Progress JSON key ``f"progress_{y_scale}_{z_scale}_last_orbit"`` tracks
      the last completed orbit; error/timeout orbits are recorded under
      dedicated keys (including per-instrument).
    - Signal handlers terminate child processes and raise ``KeyboardInterrupt``
      to interrupt the main wait loop immediately.
    """
    shutdown_requested = {"flag": False}

    # Nested: compute_global_extrema

    def compute_global_extrema(
        directory_path: str,
        y_scale: str,
        z_scale: str,
        instrument_order: Iterable[str],
        extrema_json_path: str = "./FAST_calculated_extrema.json",
        compute_mins: bool = False,
        max_percentile: float = 95.0,
        log_floor_cutoff: float = 0.1,
        log_floor_value: float = -1.0,
        flush_batch_size: int = 10,
    ) -> dict[str, Any]:
        """Compute (or incrementally update) cached axis extrema per instrument.

        Performs a resumable pass over all instrument CDF files, flushing
        incremental progress to ``extrema_json_path`` after each
        ``flush_batch_size`` orbits.

        Extrema logic
        -------------
        - Y (energy) minima are fixed to 0 unless ``compute_mins`` is True.
        - Linear Y maxima: smallest energy whose cumulative positive finite
          count reaches 99% of total positive finite samples.
        - Linear Z maxima: ``max_percentile``th percentile of pooled positive
          finite intensity samples.
        - If the requested scale is log and linear_linear extrema already exist
          in the cache, they are log-transformed without re-scanning files. If
          the requested scale is linear and linear_linear extrema exist, they
          are copied directly.
        - Log transform applies a floor: values ``<= log_floor_cutoff`` or
          non-finite are replaced by ``log_floor_value``.
        - Maxima are monotonically non-decreasing across incremental updates;
          energy maxima are capped at 4000.

        Parameters
        ----------
        directory_path : str
            Root directory containing instrument CDF files.
        y_scale : {'linear', 'log'}
            Y scaling label (used for cache key names).
        z_scale : {'linear', 'log'}
            Z scaling label (used for cache key names).
        instrument_order : iterable of str
            Instruments to process (e.g., ``("ees", "eeb", "ies", "ieb")``).
        extrema_json_path : str, default './FAST_calculated_extrema.json'
            Path to the JSON cache file (created if absent).
        compute_mins : bool, default False
            If True, compute intensity minima; otherwise they are set to 0.
        max_percentile : float, default 95.0
            Percentile applied to pooled positive intensity for ``z_max``.
        log_floor_cutoff : float, default 0.1
            Values at or below this threshold map to ``log_floor_value`` in
            log space.
        log_floor_value : float, default -1.0
            Floor value substituted for invalid log-domain extrema.
        flush_batch_size : int, default 10
            Orbits with updates between JSON flushes; coerced to ≥ 1.

        Returns
        -------
        dict
            Updated extrema mapping containing values and progress entries.
        """
        # Load (or initialise) persistent extrema cache from disk.
        if os.path.exists(extrema_json_path):
            try:
                with open(extrema_json_path, "r") as file_in:
                    extrema_state: dict[str, Any] = json.load(file_in)
            except Exception as exc:
                info_logger(
                    f"[EXTREMA] Failed to read existing extrema JSON '{extrema_json_path}' (starting fresh)",
                    exc,
                    level="message",
                )
                extrema_state = {}
        else:
            extrema_state = {}

        def _safe_log_transform(linear_value: float | int | None) -> float:
            """Convert a linear-domain value to log10 with floor handling."""
            try:
                if linear_value is None:
                    return float(log_floor_value)
                v = float(linear_value)
                if not np.isfinite(v) or v <= log_floor_cutoff:
                    return float(log_floor_value)
                return float(np.log10(v))
            except Exception as exc:
                info_logger(
                    "[EXTREMA] _safe_log_transform failed; substituting floor value",
                    exc,
                    level="message",
                )
                return float(log_floor_value)

        # Discover CDF files and group by orbit/instrument (skip _orb_ files)
        orbit_map: dict[int, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for path_obj in Path(directory_path).rglob("*.[cC][dD][fF]"):
            candidate_path = str(path_obj)
            if "_orb_" in candidate_path.lower():
                continue
            instrument_name = get_cdf_file_type(candidate_path)
            if not (
                isinstance(instrument_name, str) and instrument_name in instrument_order
            ):
                continue
            file_name = os.path.basename(candidate_path)
            file_parts = file_name.split("_")
            if len(file_parts) < 5:
                continue
            try:
                orbit_number = int(file_parts[-2])
            except Exception:
                continue
            orbit_map[orbit_number][instrument_name].append(candidate_path)

        sorted_orbit_numbers = sorted(orbit_map.keys())

        energy_positive_counts_by_instrument: dict[str, dict[float, int]] = {
            inst: defaultdict(int) for inst in instrument_order
        }
        positive_sample_arrays_by_instrument: dict[str, list[np.ndarray]] = {
            inst: [] for inst in instrument_order
        }
        file_progress_index_by_instrument: dict[str, int] = {
            inst: -1 for inst in instrument_order
        }
        total_files_per_instrument: dict[str, int] = {
            inst: sum(len(orbit_map[orb].get(inst, [])) for orb in sorted_orbit_numbers)
            for inst in instrument_order
        }

        total_discovered_files = sum(total_files_per_instrument.values())
        extrema_progress_bar = tqdm(
            total=total_discovered_files,
            desc=f"Extrema {y_scale}/{z_scale}",
            unit="file",
            leave=False,
            disable=(total_discovered_files == 0),
        )

        try:
            orbits_since_last_flush = 0
            last_orbit_global_key = f"{y_scale}_{z_scale}_last_orbit"
            last_processed_orbit_val = extrema_state.get(last_orbit_global_key, -1)
            last_processed_orbit = (
                int(last_processed_orbit_val)
                if isinstance(last_processed_orbit_val, (int, float))
                else -1
            )

            for orbit_number in sorted_orbit_numbers:
                if orbit_number <= last_processed_orbit:
                    continue
                for instrument_name in instrument_order:
                    key_prefix = f"{instrument_name}_{y_scale}_{z_scale}"
                    progress_key = f"{key_prefix}_extrema_progress"
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
                        continue

                    y_is_log = y_scale == "log"
                    z_is_log = z_scale == "log"
                    ll_y_key = f"{instrument_name}_linear_linear_y_max"
                    ll_z_key = f"{instrument_name}_linear_linear_z_max"
                    ll_y_min_key = f"{instrument_name}_linear_linear_y_min"
                    ll_z_min_key = f"{instrument_name}_linear_linear_z_min"

                    # Reuse linear_linear extrema when available (copy or log-transform)
                    if not y_is_log and ll_y_key in extrema_state:
                        extrema_state[f"{key_prefix}_y_max"] = extrema_state[ll_y_key]
                        extrema_state[f"{key_prefix}_y_min"] = extrema_state.get(
                            ll_y_min_key, 0
                        )
                    elif y_is_log and ll_y_key in extrema_state:
                        lv = extrema_state[ll_y_key]
                        extrema_state[f"{key_prefix}_y_max"] = (
                            _safe_log_transform(lv)
                            if isinstance(lv, (int, float))
                            else log_floor_value
                        )
                        extrema_state[f"{key_prefix}_y_min"] = log_floor_value

                    if not z_is_log and ll_z_key in extrema_state:
                        extrema_state[f"{key_prefix}_z_max"] = extrema_state[ll_z_key]
                        extrema_state[f"{key_prefix}_z_min"] = extrema_state.get(
                            ll_z_min_key, 0
                        )
                    elif z_is_log and ll_z_key in extrema_state:
                        lv = extrema_state[ll_z_key]
                        extrema_state[f"{key_prefix}_z_max"] = (
                            _safe_log_transform(lv)
                            if isinstance(lv, (int, float))
                            else log_floor_value
                        )
                        extrema_state[f"{key_prefix}_z_min"] = log_floor_value

                    y_done = ll_y_key in extrema_state
                    z_done = ll_z_key in extrema_state
                    if y_done and z_done:
                        total_for_inst = total_files_per_instrument[instrument_name]
                        extrema_state[progress_key] = {
                            "processed_index": max(total_for_inst - 1, -1),
                            "total": total_for_inst,
                            "complete": True,
                        }
                        for inst in instrument_order:
                            extrema_state.pop(
                                f"{inst}_{y_scale}_{z_scale}_last_orbit", None
                            )
                        extrema_state[last_orbit_global_key] = (
                            max(sorted_orbit_numbers) if sorted_orbit_numbers else -1
                        )
                        try:
                            with open(extrema_json_path, "w") as file_out:
                                json.dump(extrema_state, file_out, indent=2)
                        except Exception as exc:
                            info_logger(
                                f"[EXTREMA] Failed to save extrema JSON after reuse for instrument={instrument_name}",
                                exc,
                                level="message",
                            )
                        info_logger(
                            f"[EXTREMA] Used precomputed linear_linear extrema for "
                            f"instrument={instrument_name} y_scale={y_scale} z_scale={z_scale}",
                            level="message",
                        )
                        continue

                    # Per-file ingestion — NOTE: accumulation into energy_positive_counts
                    # and positive_sample_arrays is not yet implemented; this reads files
                    # but does not yet populate the accumulators below.
                    for cdf_path in sorted(
                        orbit_map[orbit_number].get(instrument_name, [])
                    ):
                        try:
                            cdf_obj = cdflib.CDF(cdf_path)
                            _data_raw = np.asarray(cdf_obj.varget("data"))
                            _energy_raw = np.asarray(cdf_obj.varget("energy"))
                            _pitch_angle_raw = np.asarray(cdf_obj.varget("pitch_angle"))
                        except Exception as exc:
                            info_logger(
                                f"[EXTREMA] Ingest failure inst={instrument_name} orbit={orbit_number} file={cdf_path}",
                                exc,
                                level="message",
                            )

                    # Post-orbit extrema update (monotonic merge from accumulators)
                    try:
                        energy_counts_map = energy_positive_counts_by_instrument[
                            instrument_name
                        ]
                        positive_blocks = positive_sample_arrays_by_instrument[
                            instrument_name
                        ]

                        candidate_energy_max = 0.0
                        if energy_counts_map:
                            sorted_energies = sorted(energy_counts_map.keys())
                            counts_arr = np.array(
                                [energy_counts_map[e] for e in sorted_energies]
                            )
                            cumulative = np.cumsum(counts_arr)
                            target = 0.99 * cumulative[-1]
                            idx = np.searchsorted(cumulative, target, side="right")
                            idx = min(idx, len(sorted_energies) - 1)
                            candidate_energy_max = float(sorted_energies[idx])

                        candidate_intensity_max = 0.0
                        if positive_blocks:
                            aggregated = np.concatenate(positive_blocks)
                            finite_pos = aggregated[
                                np.isfinite(aggregated) & (aggregated > 0)
                            ]
                            if finite_pos.size:
                                candidate_intensity_max = float(
                                    np.nanpercentile(finite_pos, max_percentile)
                                )

                        prev_e = extrema_state.get(f"{key_prefix}_y_max")
                        prev_z = extrema_state.get(f"{key_prefix}_z_max")
                        merged_e = (
                            max(float(prev_e), candidate_energy_max)
                            if isinstance(prev_e, (int, float))
                            else candidate_energy_max
                        )
                        merged_z = (
                            max(float(prev_z), candidate_intensity_max)
                            if isinstance(prev_z, (int, float))
                            else candidate_intensity_max
                        )
                        try:
                            merged_e = int(min(4000, math.ceil(merged_e)))
                        except Exception:
                            pass
                        try:
                            merged_z = float(math.ceil(merged_z))
                        except Exception:
                            pass

                        if compute_mins and positive_blocks:
                            try:
                                aggregated = np.concatenate(positive_blocks)
                                finite_pos = aggregated[
                                    np.isfinite(aggregated) & (aggregated > 0)
                                ]
                                intensity_min_store = (
                                    float(np.nanpercentile(finite_pos, 1))
                                    if finite_pos.size
                                    else 0.0
                                )
                            except Exception:
                                intensity_min_store = 0.0
                            energy_min_store = 0
                        else:
                            energy_min_store = 0
                            intensity_min_store = 0

                        extrema_state[f"{key_prefix}_y_min"] = energy_min_store
                        extrema_state[f"{key_prefix}_y_max"] = merged_e
                        extrema_state[f"{key_prefix}_z_min"] = intensity_min_store
                        extrema_state[f"{key_prefix}_z_max"] = merged_z
                        extrema_state[progress_key] = {
                            "processed_index": file_progress_index_by_instrument[
                                instrument_name
                            ],
                            "total": total_files_per_instrument[instrument_name],
                            "complete": (
                                file_progress_index_by_instrument[instrument_name] + 1
                                >= total_files_per_instrument[instrument_name]
                            ),
                        }
                        for inst in instrument_order:
                            extrema_state.pop(
                                f"{inst}_{y_scale}_{z_scale}_last_orbit", None
                            )
                        extrema_state[last_orbit_global_key] = orbit_number

                        try:
                            extrema_progress_bar.set_postfix(
                                inst=instrument_name, orbit=orbit_number, refresh=False
                            )
                        except Exception:
                            pass

                    except Exception as exc:
                        info_logger(
                            f"[EXTREMA] Update failure inst={instrument_name} orbit={orbit_number}",
                            exc,
                            level="message",
                        )

                    # Batched flush
                    orbits_since_last_flush += 1
                    if orbits_since_last_flush >= flush_batch_size:
                        try:
                            with open(extrema_json_path, "w") as file_out:
                                json.dump(extrema_state, file_out, indent=2)
                            orbits_since_last_flush = 0
                        except Exception as exc:
                            info_logger(
                                f"[EXTREMA] Batched flush failure after orbit {orbit_number}",
                                exc,
                                level="message",
                            )

            # Final flush
            if orbits_since_last_flush > 0:
                try:
                    if last_orbit_global_key in extrema_state:
                        ordered = {
                            last_orbit_global_key: extrema_state[last_orbit_global_key]
                        }
                        ordered.update(
                            {
                                k: v
                                for k, v in extrema_state.items()
                                if k != last_orbit_global_key
                            }
                        )
                        with open(extrema_json_path, "w") as file_out:
                            json.dump(ordered, file_out, indent=2)
                    else:
                        with open(extrema_json_path, "w") as file_out:
                            json.dump(extrema_state, file_out, indent=2)
                except Exception as exc:
                    info_logger(
                        "[EXTREMA] Final batched flush failure", exc, level="message"
                    )

        finally:
            try:
                extrema_progress_bar.close()
            except Exception:
                pass

        if last_orbit_global_key in extrema_state:
            ordered = {last_orbit_global_key: extrema_state[last_orbit_global_key]}
            ordered.update(
                {k: v for k, v in extrema_state.items() if k != last_orbit_global_key}
            )
            return ordered
        return extrema_state

    # Nested: signal handler

    def _signal_handler(signum, frame):  # frame unused
        if not shutdown_requested["flag"]:
            info_logger(
                f"[INTERRUPT] Signal {signum} received. Requesting shutdown...",
                level="message",
            )
            shutdown_requested["flag"] = True
            try:
                _terminate_all_child_processes()
            finally:
                raise KeyboardInterrupt
        else:
            info_logger(
                "[INTERRUPT] Second interrupt – forcing immediate exit.",
                level="message",
            )
            try:
                _terminate_all_child_processes()
            finally:
                raise SystemExit(130)

    # Setup

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception as exc:
        info_logger("[WARN] Could not register signal handlers", exc, level="message")

    filtered_orbits_dataframe = load_filtered_orbits()

    try:
        configure_info_logger_batch(log_flush_batch_size or flush_batch_size)
    except Exception:
        pass

    global_extrema = None
    if max_processing_percentile is not None:
        global_extrema = compute_global_extrema(
            directory_path,
            y_scale,
            z_scale,
            instrument_order,
            compute_mins=False,
            max_percentile=float(max_processing_percentile),
            log_floor_cutoff=0.1,
            log_floor_value=-1.0,
            flush_batch_size=flush_batch_size,
        )

    # Gather non-orbit CDF files and build orbit → instruments mapping
    cdf_file_paths = [
        str(p)
        for p in Path(directory_path).rglob("*.[cC][dD][fF]")
        if "_orb_" not in str(p).lower()
    ]
    orbit_to_instruments: dict[int, dict[str, str]] = defaultdict(dict)
    for cdf_path in cdf_file_paths:
        parsed = extract_orbit_and_instrument(cdf_path)
        if parsed is not None:
            orbit_num, inst_type, _ = parsed
            orbit_to_instruments[orbit_num][inst_type] = cdf_path

    sorted_orbits = sorted(orbit_to_instruments.items(), key=lambda x: x[0])
    total_orbits = len(sorted_orbits)

    # Load progress
    progress_key = f"{y_scale}_{z_scale}_last_orbit"
    error_key = f"{y_scale}_{z_scale}_error_plotting"
    progress_data: dict[str, Any] = {}
    last_completed_orbit = None
    error_orbits: set[int] = set()
    if progress_json_path is not None and not ignore_progress_json:
        try:
            with open(progress_json_path, "r") as f:
                progress_data = json.load(f)
            last_completed_orbit = progress_data.get(progress_key)
            error_orbits = set(progress_data.get(error_key, []))
        except Exception as exc:
            info_logger(
                f"[ERROR] Failed to load progress JSON from {progress_json_path}. Starting fresh.",
                exc,
                level="error",
            )

    start_idx = 0
    if last_completed_orbit is not None:
        for i, (orbit, _) in enumerate(sorted_orbits):
            if orbit > last_completed_orbit:
                start_idx = i
                break
        else:
            start_idx = total_orbits
        info_logger(
            f"[RESUME] Skipping {start_idx} orbits (up to orbit {last_completed_orbit}). "
            f"{len(error_orbits)} error orbits will also be skipped.",
            level="message",
        )
    else:
        info_logger(
            f"[RESUME] No previous progress found. Starting from the first orbit. "
            f"{len(error_orbits)} error orbits will be skipped if present.",
            level="message",
        )

    use_tqdm_bar = bool(use_tqdm) if use_tqdm is not None else False
    if flush_batch_size < 1:
        flush_batch_size = 1

    # Build argument list — always include raw; add given_extrema when available
    def _orbit_args(orbit_n: int, inst_files: dict, extrema: dict | None) -> tuple:
        return (
            orbit_n,
            inst_files,
            filtered_orbits_dataframe,
            zoom_duration_minutes,
            y_scale,
            z_scale,
            instrument_order,
            colormap,
            output_base,
            orbit_timeout_seconds,
            instrument_timeout_seconds,
            extrema,
            override_plots,
        )

    orbit_args_list: list[tuple] = []
    for orbit_number, instrument_files in sorted_orbits[start_idx:]:
        if orbit_number in error_orbits:
            continue
        orbit_args_list.append(_orbit_args(orbit_number, instrument_files, None))
        if global_extrema is not None:
            orbit_args_list.append(
                _orbit_args(orbit_number, instrument_files, global_extrema)
            )

    results: list[dict[str, Any]] = []
    _batched_progress_dirty = {"count": 0}

    # Nested: save_progress_json

    def save_progress_json(data: dict[str, Any], force: bool = False) -> None:
        """Persist *data* to disk if batch threshold is met or ``force`` is True."""
        if progress_json_path is None:
            return
        if not force:
            _batched_progress_dirty["count"] += 1
            if _batched_progress_dirty["count"] < flush_batch_size:
                return
        _batched_progress_dirty["count"] = 0
        try:
            with open(progress_json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            info_logger("[FAIL] Could not write progress JSON", exc, level="error")

    # Submit futures

    executor = None
    _orbit_completions_since_flush = {"count": 0}

    # Nested: _handle_completed_future

    def _handle_completed_future(
        fut: concurrent.futures.Future, orbit_number: int
    ) -> None:
        """Consume a completed future, append its result, and update progress JSON."""
        try:
            result = fut.result()
        except Exception as exc:
            info_logger(
                f"[BATCH] Orbit {orbit_number} generated an exception",
                exc,
                level="error",
            )
            result = {"orbit": orbit_number, "status": "error", "errors": [str(exc)]}
            results.append(result)
            if progress_json_path is not None:
                try:
                    with open(progress_json_path, "r") as f:
                        pdisk = json.load(f)
                except Exception:
                    pdisk = {}
                pdisk[progress_key] = orbit_number
                _add_to_orbit_list(pdisk, error_key, orbit_number)
                reason = _classify_error_reason(str(exc))
                _add_to_orbit_list(
                    pdisk, f"unknown_{y_scale}_{z_scale}_error-{reason}", orbit_number
                )
                _add_to_orbit_list(
                    pdisk, f"{y_scale}_{z_scale}_error-{reason}", orbit_number
                )
                _orbit_completions_since_flush["count"] += 1
                if _orbit_completions_since_flush["count"] >= flush_batch_size:
                    save_progress_json(pdisk, force=True)
                    _orbit_completions_since_flush["count"] = 0
                else:
                    save_progress_json(pdisk)
            return

        results.append(result)
        status_value = result.get("status")
        if verbose and use_tqdm_bar:
            tqdm.write(f"[BATCH] Completed orbit {orbit_number}: {status_value}")
        if progress_json_path is None:
            return

        try:
            with open(progress_json_path, "r") as f:
                pdisk = json.load(f)
        except Exception:
            pdisk = {}

        pdisk[progress_key] = orbit_number
        pdisk.setdefault(error_key, [])
        orbit_timeout_key = f"orbit_{y_scale}_{z_scale}_timed_out"
        pdisk.setdefault(orbit_timeout_key, [])

        if status_value == "error":
            _add_to_orbit_list(pdisk, error_key, orbit_number)
            for err_msg in result.get("errors") or []:
                reason = _classify_error_reason(err_msg)
                lowered = err_msg.lower()
                inst = next(
                    (c for c in ("ees", "eeb", "ies", "ieb") if c in lowered), "unknown"
                )
                _add_to_orbit_list(
                    pdisk, f"{inst}_{y_scale}_{z_scale}_error-{reason}", orbit_number
                )
                _add_to_orbit_list(
                    pdisk, f"{y_scale}_{z_scale}_error-{reason}", orbit_number
                )
        elif status_value == "timeout":
            timeout_type = result.get("timeout_type")
            timeout_instrument = result.get("timeout_instrument")
            if timeout_type == "orbit":
                _add_to_orbit_list(pdisk, orbit_timeout_key, orbit_number)
                info_logger(
                    f"[TIMEOUT-LOG] Recorded orbit timeout for orbit {orbit_number} under key {orbit_timeout_key}",
                    level="message",
                )
            elif timeout_type == "instrument":
                inst_to = timeout_instrument or "unknown_instrument"
                tk = f"{inst_to}_{y_scale}_{z_scale}_timed_out"
                _add_to_orbit_list(pdisk, tk, orbit_number)
                info_logger(
                    f"[TIMEOUT-LOG] Recorded instrument timeout for orbit {orbit_number}, "
                    f"instrument {inst_to} under key {tk}",
                    level="message",
                )

        _orbit_completions_since_flush["count"] += 1
        if _orbit_completions_since_flush["count"] >= flush_batch_size:
            save_progress_json(pdisk, force=True)
            _orbit_completions_since_flush["count"] = 0
        else:
            save_progress_json(pdisk)

    # Main execution loop

    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        future_to_orbit: dict[concurrent.futures.Future, int] = {}
        for args in orbit_args_list:
            if shutdown_requested["flag"]:
                break
            future = executor.submit(FAST_process_single_orbit, *args)
            future_to_orbit[future] = args[0]
        futures = set(future_to_orbit.keys())

        total_count = len(futures)
        processed_count = 0
        progress_bar = None
        if use_tqdm_bar:
            if start_idx > 0:
                info_logger(
                    f"[RESUME] Resuming progress bar at orbit {start_idx + 1} of {total_orbits} "
                    f"for y_scale={y_scale}, z_scale={z_scale}.",
                    level="message",
                )
            progress_bar = tqdm(
                total=total_count,
                initial=0,
                desc=f"Plotting - {y_scale} / {z_scale}",
                unit="orbit",
                leave=False,
            )
        try:
            while futures:
                if shutdown_requested["flag"]:
                    break
                done, _ = concurrent.futures.wait(
                    futures, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    futures.discard(fut)
                    orbit_number = future_to_orbit[fut]
                    _handle_completed_future(fut, orbit_number)
                    processed_count += 1
                    if progress_bar is not None:
                        progress_bar.set_postfix(orbit=orbit_number)
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        # Flush progress for the last batch
        try:
            if progress_json_path is not None and os.path.exists(progress_json_path):
                with open(progress_json_path, "r") as f:
                    final_pd = json.load(f)
            else:
                final_pd = progress_data if isinstance(progress_data, dict) else {}
            save_progress_json(final_pd, force=True)
        except Exception:
            pass

        if shutdown_requested["flag"]:
            info_logger(
                "[INTERRUPT] Shutdown requested; cancelling remaining futures.",
                level="message",
            )
            for fut in list(futures):
                try:
                    fut.cancel()
                except Exception:
                    pass
            try:
                executor.shutdown(wait=False, cancel_futures=True)
                if hasattr(executor, "_processes"):
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    _time.sleep(0.05)
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            if proc.is_alive():
                                proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
            raise KeyboardInterrupt

    except KeyboardInterrupt as exc:
        info_logger(
            f"[INTERRUPT] KeyboardInterrupt caught. Terminating worker processes... Exception: {exc}",
            level="message",
        )
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
                if hasattr(executor, "_processes"):
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    _time.sleep(0.05)
                    for proc in executor._processes.values():  # type: ignore[attr-defined]
                        try:
                            if proc.is_alive():
                                proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
        raise
    finally:
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    # Final flush of progress and logs
    try:
        if progress_json_path is not None and os.path.exists(progress_json_path):
            with open(progress_json_path, "r") as f:
                final_pd = json.load(f)
        else:
            final_pd = progress_data if isinstance(progress_data, dict) else {}
        save_progress_json(final_pd, force=True)
    except Exception:
        pass
    try:
        flush_info_logger_buffer(force=True)
    except Exception:
        pass

    # Retry timed-out orbits once with a smaller pool
    if retry_timeouts and not shutdown_requested["flag"]:
        timeout_orbits = [r["orbit"] for r in results if r.get("status") == "timeout"]
        if timeout_orbits:
            info_logger(
                f"[RETRY] Retrying {len(timeout_orbits)} timed-out orbits once "
                f"(thresholds orbit={orbit_timeout_seconds}s, instrument={instrument_timeout_seconds}s).",
                level="message",
            )
            retry_args = [
                _orbit_args(o, orbit_to_instruments[o], None)
                for o in timeout_orbits
                if o in orbit_to_instruments
            ]
            retry_results: list[dict[str, Any]] = []
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
                            # On success, remove the orbit from timeout lists in the JSON
                            if (
                                progress_json_path is not None
                                and r_result.get("status") == "ok"
                            ):
                                try:
                                    with open(progress_json_path, "r") as f:
                                        pdisk_retry = json.load(f)
                                except Exception as exc:
                                    info_logger(
                                        "[WARN] Could not read progress JSON for retry cleanup",
                                        exc,
                                        level="message",
                                    )
                                    pdisk_retry = {}
                                timeout_keys = [
                                    k
                                    for k in pdisk_retry
                                    if k.endswith(f"_{y_scale}_{z_scale}_timed_out")
                                ]
                                modified = False
                                for tk in timeout_keys:
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
                                    except Exception as exc:
                                        info_logger(
                                            "[WARN] Could not write cleaned progress JSON",
                                            exc,
                                            level="message",
                                        )
                        except Exception as exc:
                            info_logger(
                                f"[RETRY] Orbit {r_orbit} retry failed",
                                exc,
                                level="error",
                            )
                            retry_results.append(
                                {
                                    "orbit": r_orbit,
                                    "status": "error",
                                    "errors": [str(exc)],
                                }
                            )
            except Exception as exc:
                info_logger(
                    "[RETRY] Failed to execute retry pool", exc, level="message"
                )

            results_map = {r["orbit"]: r for r in results}
            for rr in retry_results:
                results_map[rr["orbit"]] = rr
            results = list(results_map.values())

    return results


def main() -> None:
    """Run the FAST batch plotter for all y/z scale combinations sequentially.

    Invokes ``FAST_plot_spectrograms_directory`` for each combination of
    linear/log y and z, using colormaps tailored for each. An interrupt during
    any run stops the sequence without starting subsequent combinations.
    """
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
        info_logger("[INTERRUPT] Batch plotting aborted by user.", level="message")
        print("\n[INTERRUPT] Aborted by user.")
        sys.exit(130)
