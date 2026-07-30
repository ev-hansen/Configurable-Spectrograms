"""Single-output spectrogram rendering.

These functions render one figure (or one panel of a figure) for a single
item -- a single CDF, orbit, or caller-supplied dataset. Batch/loop callers
(:mod:`configurable_spectrograms.generic_batch`,
:mod:`configurable_spectrograms.fast.process_orbit`) call these same
functions once per item rather than re-implementing rendering logic, so a
single-plot CLI script and a batch driver always produce identical output
for identical inputs.
"""

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for batch and headless rendering.

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import _pylab_helpers  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402
from matplotlib.dates import date2num  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from configurable_spectrograms.constants import (  # noqa: E402
    AXIS_LABEL_FONT_SIZE,
    COLLAPSE_FUNCTION,
    PLOT_FIGURE_HEIGHT_INCHES,
    PLOT_FIGURE_WIDTH_INCHES,
    TICK_LABEL_FONT_SIZE,
)
from configurable_spectrograms.cusp_marking import draw_cusp_bracket_marker, draw_cusp_line_markers  # noqa: E402
from configurable_spectrograms.logging_utils import log_message  # noqa: E402
from configurable_spectrograms.percentile_utils import compute_percentile_bounds  # noqa: E402

_CUSP_MARKER_DRAWERS = {
    "line": draw_cusp_line_markers,
    "bracket": draw_cusp_bracket_marker,
}


def close_all_axes_and_clear(fig) -> None:
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
    Ensures axes are deleted, the canvas is closed/detached, and removes the
    figure from the global Gcf registry when possible to mitigate memory
    growth during large batch operations.
    """
    for axis in list(fig.axes):
        try:
            fig.delaxes(axis)
        except Exception as axis_close_error:
            log_message(f"[WARN] Error closing axis: {axis_close_error}")
    fig.clf()
    if hasattr(fig, "canvas") and fig.canvas is not None:
        try:
            fig.canvas.close()
        except Exception as canvas_close_error:
            log_message(f"[WARN] Error closing canvas: {canvas_close_error}")
        try:
            fig.canvas.figure = None
        except Exception as canvas_figure_clear_error:
            log_message(f"[WARN] Error clearing canvas figure: {canvas_figure_clear_error}")
        fig.canvas = None
    try:
        if hasattr(fig, "number") and fig.number is not None:
            _pylab_helpers.Gcf.destroy(fig.number)
    except Exception as gcf_registry_error:
        log_message(f"[WARN] Error removing figure from Gcf registry: {gcf_registry_error}")


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
    cusp_marker_style="line",
    cusp_marker_kwargs=None,
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
        UNIX timestamps to annotate with a cusp-boundary marker.
    cusp_marker_style : {'line', 'bracket'}, default 'line'
        Marker style for ``vertical_lines_unix``: ``'line'`` reproduces the
        original double-line marker; ``'bracket'`` draws a bracket spanning
        the boundary interval below the axis instead.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the selected marker-drawing
        function (see :mod:`configurable_spectrograms.cusp_marking`).

    Returns
    -------
    axis_object : matplotlib.axes.Axes or None
        The axis object used for plotting (``None`` if no data plotted).
    x_axis_plot : numpy.ndarray or None
        X values actually used (possibly filtered / converted), or ``None``
        if skipped.
    """
    log_message(
        f"[DEBUG] make_spectrogram: y_axis_scale_function={y_axis_scale_function}, "
        f"z_axis_scale_function={z_axis_scale_function}, z_axis_min={z_axis_min}, "
        f"z_axis_max={z_axis_max}, colormap={colormap}"
    )

    x_axis = np.asarray(x_axis_values)
    y_axis = np.asarray(y_axis_values)
    data_array = np.asarray(data_array_3d)

    # Collapse the 3D data array along the specified axis (e.g., sum over pitch angle).
    collapsed_matrix = COLLAPSE_FUNCTION(data_array, axis=collapse_axis)

    # Mask out columns that are all NaN and restrict to the valid energy range.
    nan_column_mask = ~np.all(np.isnan(collapsed_matrix), axis=0)
    valid_energy_mask = (y_axis >= y_axis_min) & (y_axis <= y_axis_max)
    combined_mask = nan_column_mask & valid_energy_mask
    collapsed_matrix = collapsed_matrix[:, combined_mask]
    y_axis = y_axis[combined_mask]
    if collapsed_matrix.size == 0 or y_axis.size == 0:
        log_message("[WARNING] All energy bins were filtered out. No data to plot.")
        return None, None

    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        collapsed_matrix = collapsed_matrix[:, ::-1]

    if center_timestamp is not None and window_duration_seconds is not None:
        half_window = window_duration_seconds / 2
        left_bound = center_timestamp - half_window
        right_bound = center_timestamp + half_window
        zoom_mask = (x_axis >= left_bound) & (x_axis <= right_bound)
        x_axis = x_axis[zoom_mask]
        collapsed_matrix = collapsed_matrix[zoom_mask, :]

    if x_axis_min is not None or x_axis_max is not None:
        x_mask = np.ones_like(x_axis, dtype=bool)
        if x_axis_min is not None:
            x_mask &= x_axis >= x_axis_min
        if x_axis_max is not None:
            x_mask &= x_axis <= x_axis_max
        x_axis = x_axis[x_mask]
        collapsed_matrix = collapsed_matrix[x_mask, :]

    if x_axis_is_unix:
        x_axis_datetime = np.array([datetime.fromtimestamp(x, tz=timezone.utc) for x in x_axis])
        x_axis_plot = date2num(x_axis_datetime)
        x_label = x_axis_label if x_axis_label is not None else "Time (UTC)"
    else:
        x_axis_plot = x_axis
        x_label = x_axis_label if x_axis_label is not None else "X"

    if axis_object is None:
        fig = Figure(figsize=(PLOT_FIGURE_WIDTH_INCHES, PLOT_FIGURE_HEIGHT_INCHES))
        FigureCanvas(fig)
        axis_object = fig.add_subplot(1, 1, 1)
    else:
        fig = axis_object.figure

    matrix_plot = collapsed_matrix.T

    if center_timestamp is not None and window_duration_seconds is not None:
        if x_axis_is_unix:
            left_num = float(
                date2num(datetime.fromtimestamp(center_timestamp - window_duration_seconds / 2, tz=timezone.utc))
            )
            right_num = float(
                date2num(datetime.fromtimestamp(center_timestamp + window_duration_seconds / 2, tz=timezone.utc))
            )
            axis_object.set_xlim(left_num, right_num)
        else:
            axis_object.set_xlim(
                center_timestamp - window_duration_seconds / 2,
                center_timestamp + window_duration_seconds / 2,
            )
    else:
        axis_object.set_xlim(x_axis_plot[0], x_axis_plot[-1])

    if matrix_plot.size == 0:
        log_message("[WARNING] No data to plot after filtering. Skipping plot.")
        return None, None

    z_axis_min, z_axis_max = compute_percentile_bounds(matrix_plot, 1, 99, z_axis_min, z_axis_max)

    finite_positive = matrix_plot[np.isfinite(matrix_plot) & (matrix_plot > 0)]
    safe_vmin = np.nanmin(finite_positive) if finite_positive.size > 0 else 1e-10

    if z_axis_scale_function == "log":
        if np.any(matrix_plot <= 0) or not (
            np.isfinite(z_axis_min)
            and np.isfinite(z_axis_max)
            and z_axis_min > 0
            and z_axis_max > 0
            and z_axis_max > z_axis_min
        ):
            log_message(
                "[WARNING] Non-positive values found in matrix for log colorbar. "
                "Masking to z_axis_min and enforcing log scale."
            )
        z_axis_min = float(max(z_axis_min, safe_vmin, 1e-10))
        z_axis_max = float(z_axis_max)
        matrix_plot = np.where(~np.isfinite(matrix_plot) | (matrix_plot <= 0), z_axis_min, matrix_plot)
        norm = mcolors.LogNorm(vmin=z_axis_min, vmax=z_axis_max)
        im = axis_object.imshow(
            matrix_plot,
            aspect="auto",
            origin="lower",
            extent=(x_axis_plot[0], x_axis_plot[-1], y_axis[0], y_axis[-1]),
            cmap=colormap,
            norm=norm,
        )
        min_exponent = int(np.floor(np.log10(z_axis_min)))
        max_exponent = int(np.ceil(np.log10(z_axis_max)))
        ticks = [10**i for i in range(min_exponent, max_exponent + 1) if z_axis_min <= 10**i <= z_axis_max]

        def log_tick_formatter(value, position=None):
            if value <= 0:
                return ""
            exponent = int(np.log10(value))
            if np.isclose(value, 10**exponent):
                return f"$10^{{{exponent}}}$"
            return ""

        colorbar = fig.colorbar(
            im,
            ax=axis_object,
            label=z_axis_label if z_axis_label is not None else "Counts",
            ticks=ticks,
            format=log_tick_formatter,
        )
    else:
        z_axis_min = float(z_axis_min)
        z_axis_max = float(z_axis_max)
        matrix_plot = np.where(np.isnan(matrix_plot), z_axis_min, matrix_plot)
        matrix_plot = np.where(np.isneginf(matrix_plot), z_axis_min, matrix_plot)
        matrix_plot = np.where(np.isposinf(matrix_plot), z_axis_max, matrix_plot)
        if not (np.isfinite(z_axis_min) and np.isfinite(z_axis_max) and z_axis_max > z_axis_min):
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
        colorbar = fig.colorbar(
            im,
            ax=axis_object,
            label=z_axis_label if z_axis_label is not None else "Counts",
        )

    axis_object.set_xlabel(x_label)
    axis_object.set_ylabel(y_axis_label if y_axis_label is not None else "Energy (eV)")
    if instrument_label is not None:
        axis_object.set_title(instrument_label)

    if len(y_axis) >= 2:
        if y_axis_scale_function != "log":
            y_max_str = str(y_axis_max)
            y_max_digits = len(y_max_str)
            y_first_digit = int(y_max_str[0])
            y_second_digit = int(y_max_str[1])
            if y_second_digit >= 5:
                step_size = 10**y_max_digits
                y_max_tick = y_first_digit * 10 ** (y_max_digits - 1)
            else:
                step_size = 10 ** (y_max_digits - 1)
                y_max_tick = (y_first_digit + 0.5) * 10 ** (y_max_digits - 1)
            yticks = [i for i in range(y_axis_min, int(y_max_tick) + 1, step_size) if (i / y_max_tick) <= 1.1]
            if len(yticks) > 0:
                axis_object.set_yticks(yticks)
                axis_object.set_yticklabels([f"{int(e)}" for e in yticks])
        else:
            axis_object.set_yscale("log")

    if x_axis_is_unix:
        x_limits = axis_object.get_xlim()
        left_datetime = mdates.num2date(x_limits[0], tz=timezone.utc)
        right_datetime = mdates.num2date(x_limits[1], tz=timezone.utc)
        displayed_time_range_seconds = (right_datetime - left_datetime).total_seconds()
        if displayed_time_range_seconds < 120:
            axis_object.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
        else:
            axis_object.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))

    if vertical_lines_unix is not None and len(vertical_lines_unix) > 0:
        if x_axis_is_unix:
            vertical_lines_plot = date2num(
                [datetime.fromtimestamp(timestamp, tz=timezone.utc) for timestamp in vertical_lines_unix]
            )
            x_min_plot = x_axis_plot[0]
            x_max_plot = x_axis_plot[-1]
            vertical_lines_plot = [v for v in vertical_lines_plot if x_min_plot <= v <= x_max_plot]
        else:
            vertical_lines_plot = [v for v in vertical_lines_unix if x_axis_plot[0] <= v <= x_axis_plot[-1]]
        draw_marker = _CUSP_MARKER_DRAWERS.get(cusp_marker_style, draw_cusp_line_markers)
        draw_marker(axis_object, vertical_lines_plot, **(cusp_marker_kwargs or {}))

    axis_object.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE, length=8, width=1)
    axis_object.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_FONT_SIZE, length=5, width=1)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE, length=6, width=1)
    colorbar.ax.tick_params(which="minor", labelsize=TICK_LABEL_FONT_SIZE, length=3, width=1)

    axis_object.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    axis_object.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    colorbar.ax.set_ylabel("Counts", fontsize=AXIS_LABEL_FONT_SIZE)

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
    cusp_marker_style="line",
    cusp_marker_kwargs=None,
):
    """Plot a vertical stack of generic spectrograms.

    Parameters
    ----------
    datasets : list of dict
        Each dict requires keys ``'x'``, ``'y'``, ``'data'`` and may include
        optional keys: ``'label'``, ``'y_label'``, ``'z_label'``,
        ``'y_min'``, ``'y_max'``, ``'z_min'``, ``'z_max'``.
    collapse_axis : int, default 1
        Axis index of the 3D array collapsed prior to plotting.
    zoom_center : float, optional
        Center (UNIX time) for zoom column when used.
    zoom_window_seconds : float, optional
        Duration of zoom window (seconds) when ``zoom_center`` provided.
    vertical_lines : list of float, optional
        UNIX timestamps to annotate with a cusp-boundary marker.
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
        Global Y min fallback when per-row not supplied. Defaults to 0 if
        omitted and per-row missing.
    y_max : float, optional
        Global Y max fallback when per-row not supplied. If both global and
        per-row absent, inferred.
    z_min : float, optional
        Global colorbar lower bound fallback.
    z_max : float, optional
        Global colorbar upper bound fallback.
    cusp_marker_style : {'line', 'bracket'}, default 'line'
        Marker style forwarded to :func:`make_spectrogram`.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the marker-drawing function.

    Returns
    -------
    tuple
        ``(fig, canvas)`` or ``(None, None)`` if ``datasets`` is empty.
    """
    if not datasets:
        return None, None
    fig = Figure(figsize=(10, 3 * len(datasets)))
    canvas = FigureCanvas(fig)
    for row_index, dataset in enumerate(datasets):
        axis_obj = fig.add_subplot(len(datasets), 1, row_index + 1)
        dataset_y_min = dataset.get("y_min", y_min)
        dataset_y_max = dataset.get("y_max", y_max)
        dataset_z_min = dataset.get("z_min", z_min)
        dataset_z_max = dataset.get("z_max", z_max)
        inferred_y_max = (
            dataset["y"].max() if dataset_y_max is None and dataset.get("y") is not None else dataset_y_max
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
            cusp_marker_style=cusp_marker_style,
            cusp_marker_kwargs=cusp_marker_kwargs,
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
    cusp_marker_style="line",
    cusp_marker_kwargs=None,
):
    """Render a multi-row spectrogram grid with an optional zoom column.

    Parameters
    ----------
    datasets : list of dict
        Each dict must contain keys:

        * ``'x'`` -- 1D UNIX epoch seconds (float) array
        * ``'y'`` -- 1D energy (eV) array (unfiltered, 0-4000 typical)
        * ``'data'`` -- 3D ndarray that can be collapsed (time, pitch/angle, energy)

        Optional per-row keys (all honored when present):

        * ``'label'`` -- Row label placed on the left (rotated)
        * ``'y_label'`` -- Units label for y-axis (default: ``'Energy (eV)'``)
        * ``'z_label'`` -- Color scale label (default: ``'Counts'``)
        * ``'y_min'`` / ``'y_max'`` -- Energy bounds (overrides global ``y_min`` / ``y_max`` args)
        * ``'z_min'`` / ``'z_max'`` -- Color bounds (overrides global ``z_min`` / ``z_max`` args)
        * ``'vmin'`` / ``'vmax'`` -- Precomputed percentile (or fixed) color bounds used when
          ``z_min`` / ``z_max`` not provided.
    vertical_lines : list of float, optional
        UNIX timestamps defining the cusp boundary and potential zoom window.
    zoom_duration_minutes : float, default 6.25
        Desired zoom window length in minutes (may auto-expand to include
        full marked span).
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
        precedence.
    cusp_marker_style : {'line', 'bracket'}, default 'line'
        Marker style forwarded to :func:`make_spectrogram`.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the marker-drawing function.

    Returns
    -------
    tuple
        ``(fig, canvas)`` or ``(None, None)`` if ``datasets`` is empty.

    Notes
    -----
    Determines need for a zoom column dynamically: only rendered if at least
    one dataset contains non-NaN values inside the computed zoom window.
    """
    if not datasets:
        return None, None
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
            axes[i, j] = fig.add_subplot(number_rows, number_columns, i * number_columns + j + 1)
    for i, ds in enumerate(datasets):
        times = ds["x"]
        energy = ds["y"]
        data3d = ds["data"]
        vmin = ds.get("vmin")
        vmax = ds.get("vmax")
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
            cusp_marker_style=cusp_marker_style,
            cusp_marker_kwargs=cusp_marker_kwargs,
            z_axis_min=vmin if z_min is None else z_min,
            z_axis_max=vmax if z_max is None else z_max,
            axis_object=axes[i, 0],
            colormap=colormap,
        )
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
                cusp_marker_style=cusp_marker_style,
                cusp_marker_kwargs=cusp_marker_kwargs,
                z_axis_min=vmin if z_min is None else z_min,
                z_axis_max=vmax if z_max is None else z_max,
                axis_object=axes[i, 1],
                colormap=colormap,
            )
    for i, ds in enumerate(datasets):
        axes[i, 0].set_ylabel(
            ds.get("label", ""),
            fontsize=AXIS_LABEL_FONT_SIZE,
            rotation=row_label_rotation,
            labelpad=row_label_pad,
            va="center",
        )
    if number_columns == 2:
        axes[0, 0].set_title("Full", fontsize=AXIS_LABEL_FONT_SIZE)
        axes[0, 1].set_title("Zoomed", fontsize=AXIS_LABEL_FONT_SIZE)
    else:
        axes[0, 0].set_title("Full", fontsize=AXIS_LABEL_FONT_SIZE)
    if title:
        fig.suptitle(title, fontsize=AXIS_LABEL_FONT_SIZE + 2)
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
