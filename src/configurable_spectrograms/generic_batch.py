"""Generic (data-agnostic) batch spectrogram plotting."""

import functools
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from configurable_spectrograms.batch_runner import run_batch
from configurable_spectrograms.constants import PLOTTING_PROGRESS_JSON_PATH
from configurable_spectrograms.logging_utils import log_error
from configurable_spectrograms.plotting import close_all_axes_and_clear, generic_plot_spectrogram_set


def generic_batch_plot(
    items,
    output_dir: str,
    build_datasets_fn: Callable[[Any], list[dict]],
    zoom_center_fn: Callable[[Any], float | None] | None = None,
    zoom_window_seconds: float | None = None,
    vertical_lines_fn: Callable[[Any], list[float] | None] | None = None,
    y_scale: str = "linear",
    z_scale: str = "linear",
    colormap: str = "viridis",
    cusp_marker_style: str = "both",
    cusp_marker_kwargs: dict | None = None,
    max_workers: int = 2,
    progress_json_path: str = PLOTTING_PROGRESS_JSON_PATH,
    ignore_progress_json: bool = False,
    flush_batch_size: int = 10,
    log_flush_batch_size: int | None = None,
    install_signal_handlers: bool = True,
) -> list[tuple[Any, str]]:
    """Generic batch runner for plotting datasets across many items in parallel.

    Each item is rendered by calling
    :func:`configurable_spectrograms.plotting.generic_plot_spectrogram_set`
    exactly once, in a worker process managed by
    :func:`configurable_spectrograms.batch_runner.run_batch`, so a single
    item plotted through this batch driver produces the same output as
    calling the single-output function directly.

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
    cusp_marker_style : {'line', 'bracket', 'both'}, default 'both'
        Cusp-boundary marker style forwarded to ``generic_plot_spectrogram_set``.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the marker-drawing function.
    max_workers : int, default 2
        Number of parallel worker processes.
    progress_json_path : str, default PLOTTING_PROGRESS_JSON_PATH
        Path to progress JSON (resumable state). Created/updated as needed.
    ignore_progress_json : bool, default False
        If ``True``, skip reading existing progress prior to execution.
    flush_batch_size : int, default 10
        Progress/log batch size; values < 1 coerced to 1. Final partial
        batch flushed.
    log_flush_batch_size : int, optional
        Explicit log batch size; if ``None`` reuse ``flush_batch_size``.
    install_signal_handlers : bool, default True
        When True, a temporary SIGINT handler is installed (restored on
        exit) to enable graceful interruption (progress & log flush).

    Returns
    -------
    list of tuple
        Sequence of ``(item, status)`` with ``status`` in {``'ok'``,
        ``'no_data'``, ``'error'``}.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _worker(item):
        try:
            datasets = build_datasets_fn(item)
            if not datasets:
                return (item, "no_data")
            center = zoom_center_fn(item) if zoom_center_fn else None
            vertical_lines = vertical_lines_fn(item) if vertical_lines_fn else None
            fig, _canvas = generic_plot_spectrogram_set(
                datasets,
                zoom_center=center,
                zoom_window_seconds=zoom_window_seconds,
                vertical_lines=vertical_lines,
                y_scale=y_scale,
                z_scale=z_scale,
                colormap=colormap,
                cusp_marker_style=cusp_marker_style,
                cusp_marker_kwargs=cusp_marker_kwargs,
                show=False,
            )
            if fig is not None:
                item_output_dir = os.path.join(output_dir, str(item))
                os.makedirs(item_output_dir, exist_ok=True)
                out_path = os.path.join(item_output_dir, "generic.png")
                fig.savefig(out_path, dpi=150)
                close_all_axes_and_clear(fig)
            return (item, "ok")
        except Exception as generic_exception:
            log_error(f"[GENERIC-FAIL] Item {item}: {generic_exception}")
            return (item, "error")

    return run_batch(
        items,
        _worker,
        functools.partial(ProcessPoolExecutor, max_workers=max_workers),
        progress_json_path=progress_json_path,
        ignore_progress_json=ignore_progress_json,
        flush_batch_size=flush_batch_size,
        log_flush_batch_size=log_flush_batch_size,
        install_signal_handlers=install_signal_handlers,
    )
