"""Single-output FAST ESA spectrogram rendering.

Both functions here render one figure for a single CDF file or a single
orbit's worth of instrument files. Batch callers
(:mod:`configurable_spectrograms.fast.process_orbit`) call these same
functions once per orbit rather than duplicating the rendering logic.
"""

from typing import Any

import numpy as np

from configurable_spectrograms.cdf_utils import get_cdf_file_type, get_timestamps_for_orbit, load_fast_cdf_dataset
from configurable_spectrograms.fast.constants import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PITCH_ANGLE_CATEGORIES,
    FAST_COLLAPSE_FUNCTION,
)
from configurable_spectrograms.logging_utils import log_exception
from configurable_spectrograms.percentile_utils import compute_percentile_bounds
from configurable_spectrograms.plotting import generic_plot_multirow_optional_zoom

# Row order used when building the pitch-angle grid; independent of dict
# iteration order so callers passing a custom `pitch_angle_categories`
# mapping still get a stable row layout for the four standard categories.
_PITCH_ANGLE_ROW_KEYS = (
    "all\n(0, 360)",
    "downgoing\n(0, 30), (330, 360)",
    "upgoing\n(150, 210)",
    "perpendicular\n(40, 140), (210, 330)",
)


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
    cusp_marker_style: str = "both",
    cusp_marker_kwargs: dict | None = None,
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
        Color scale limits; defaults to row-level 1st/99th percentiles when
        None.
    cusp_marker_style : {'line', 'bracket', 'both'}, default 'both'
        Cusp-boundary marker style; see
        :mod:`configurable_spectrograms.cusp_marking`.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the marker-drawing function.

    Returns
    -------
    tuple[Figure or None, FigureCanvasBase or None]
        Figure and Canvas, or ``(None, None)`` when no datasets are produced.
    """
    # TODO: record orbits when error contains "is not a CDF file or a non-supported CDF!" in json log file
    if pitch_angle_categories is None:
        pitch_angle_categories = DEFAULT_PITCH_ANGLE_CATEGORIES
    instrument_type = get_cdf_file_type(cdf_file_path)
    dataset = load_fast_cdf_dataset(cdf_file_path)
    times, data, energy, pitchangle = (
        dataset["times"],
        dataset["data"],
        dataset["energy"],
        dataset["pitch_angle"],
    )

    vertical_lines = None
    if filtered_orbits_df is not None and orbit_number is not None:
        vertical_lines = get_timestamps_for_orbit(filtered_orbits_df, orbit_number, instrument_type, times)
        if not vertical_lines:
            log_exception(
                f"No vertical lines found for orbit {orbit_number} in {cdf_file_path}. Skipping.",
                level="message",
            )

    y_lower = 0 if y_min is None else y_min
    y_upper = 4000 if y_max is None else y_max
    valid_energy_mask = (energy >= y_lower) & (energy <= y_upper)

    datasets = []
    for key in _PITCH_ANGLE_ROW_KEYS:
        if key not in pitch_angle_categories:
            continue
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
        vmin, vmax = compute_percentile_bounds(matrix_full_plot, 1, 99, z_min, z_max)
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
        log_exception(f"[WARNING] No pitch angle datasets to plot for {cdf_file_path}.", level="message")
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
        cusp_marker_style=cusp_marker_style,
        cusp_marker_kwargs=cusp_marker_kwargs,
    )


def FAST_plot_instrument_grid(
    cdf_file_paths: dict[str, str],
    filtered_orbits_df=None,
    orbit_number: int | None = None,
    zoom_duration_minutes: float = 6.25,
    scale_function_y: str = "linear",
    scale_function_z: str = "linear",
    instrument_order: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER,
    show: bool = True,
    colormap: str = "viridis",
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    global_extrema: dict[str, int | float] | None = None,
    cusp_marker_style: str = "both",
    cusp_marker_kwargs: dict | None = None,
) -> tuple[Any, Any]:
    """Plot a multi-instrument ESA spectrogram grid for a single orbit.

    Loads each instrument CDF, collapses across pitch-angle, and constructs
    datasets for ``generic_plot_multirow_optional_zoom``. A zoom column is
    included when vertical lines are available for the orbit.

    Parameters
    ----------
    cdf_file_paths : dict of {str: str}
        Mapping of instrument key (``'ees'``, ``'eeb'``, ``'ies'``,
        ``'ieb'``) to CDF file path. Missing instruments are skipped.
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
    instrument_order : tuple of str, default DEFAULT_INSTRUMENT_ORDER
        Display order of instrument rows.
    show : bool, default True
        Whether to show the figure interactively.
    colormap : str, default 'viridis'
        Matplotlib colormap name.
    y_min, y_max, z_min, z_max : float or None, optional
        Global fallback axis/color limits used when ``global_extrema`` does
        not supply an instrument-specific key.
    global_extrema : dict or None
        Precomputed extrema keyed as
        ``{instrument}_{y_scale}_{z_scale}_{axis}_{min|max}`` supplying
        per-instrument limits. Takes precedence over the direct ``y_min`` /
        ``y_max`` / ``z_min`` / ``z_max`` arguments.
    cusp_marker_style : {'line', 'bracket', 'both'}, default 'both'
        Cusp-boundary marker style; see
        :mod:`configurable_spectrograms.cusp_marking`.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the marker-drawing function.

    Returns
    -------
    tuple[Figure or None, FigureCanvasBase or None]
        Figure and Canvas, or ``(None, None)`` when no datasets are produced.

    Notes
    -----
    - Files that fail to load are logged and skipped.
    - Energy bins are restricted to ``[0, 4000]`` unless overridden.
    - ``vmin``/``vmax`` per row use 1st/99th percentiles unless
      ``global_extrema`` provides per-instrument ``z_min`` / ``z_max``.
    """
    datasets = []
    vertical_lines = None
    for inst in instrument_order:
        cdf_path = cdf_file_paths.get(inst)
        if not cdf_path:
            continue
        try:
            dataset = load_fast_cdf_dataset(cdf_path)
            times, data, energy = dataset["times"], dataset["data"], dataset["energy"]

            if vertical_lines is None and filtered_orbits_df is not None and orbit_number is not None:
                instrument_type = get_cdf_file_type(cdf_path)
                vertical_lines = get_timestamps_for_orbit(filtered_orbits_df, orbit_number, instrument_type, times)
                if not vertical_lines:
                    log_exception(
                        f"No vertical lines found for orbit {orbit_number} in {cdf_path}. Skipping.",
                        level="message",
                    )

            if isinstance(global_extrema, dict):
                key_prefix = f"{inst}_{scale_function_y}_{scale_function_z}"
                y_lower = global_extrema.get(f"{key_prefix}_y_min", 0 if y_min is None else y_min)
                y_upper = global_extrema.get(f"{key_prefix}_y_max", 4000 if y_max is None else y_max)
                row_z_min = global_extrema.get(f"{key_prefix}_z_min")
                row_z_max = global_extrema.get(f"{key_prefix}_z_max")
            else:
                y_lower = 0 if y_min is None else y_min
                y_upper = 4000 if y_max is None else y_max
                row_z_min = row_z_max = None

            matrix_full = FAST_COLLAPSE_FUNCTION(data, axis=1)
            nan_col_mask = ~np.all(np.isnan(matrix_full), axis=0)
            valid_energy_mask = (energy >= y_lower) & (energy <= y_upper)
            matrix_full = matrix_full[:, nan_col_mask & valid_energy_mask]
            matrix_full_plot = matrix_full.T
            if matrix_full_plot.size == 0:
                continue

            vmin, vmax = compute_percentile_bounds(matrix_full_plot, 1, 99, row_z_min, row_z_max)

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
            log_exception(f"Failed to load CDF for {inst} at {cdf_path}. Skipping.", exc, level="error")

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
        cusp_marker_style=cusp_marker_style,
        cusp_marker_kwargs=cusp_marker_kwargs,
    )
