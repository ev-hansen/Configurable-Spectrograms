"""Per-orbit FAST spectrogram processing (the parallel batch worker unit)."""

import gc
import os
import time as _time
from typing import Any

from configurable_spectrograms.cdf_utils import get_cdf_file_type, get_timestamps_for_orbit, load_fast_cdf_dataset
from configurable_spectrograms.fast.constants import DEFAULT_INSTRUMENT_ORDER
from configurable_spectrograms.fast.extrema import _extrema_overrides
from configurable_spectrograms.fast.orbit_discovery import _parse_year_month
from configurable_spectrograms.fast.plotting import FAST_plot_instrument_grid, FAST_plot_pitch_angle_grid
from configurable_spectrograms.logging_utils import log_exception
from configurable_spectrograms.plotting import close_all_axes_and_clear


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
    cusp_marker_style: str = "both",
    cusp_marker_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Process and save all ESA spectrogram plots for a single orbit.

    For each available instrument, generates two plot versions:
    1. Using ``global_extrema`` (``_given_extrema`` suffix) if provided.
    2. Raw (per-file) extrema (``_raw`` suffix).
    This applies to both pitch-angle and instrument-grid plots. Each figure
    is saved to disk and closed immediately after it's rendered, so at most
    one or two figures are ever held in memory at once regardless of how
    many instruments and extrema variants an orbit produces.

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
    cusp_marker_style : {'line', 'bracket', 'both'}, default 'both'
        Cusp-boundary marker style forwarded to the plotting functions.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the marker-drawing function.

    Returns
    -------
    dict
        Result with keys ``orbit`` (int), ``status`` (``'ok'``, ``'error'``,
        or ``'timeout'``), ``errors`` (list of str), and optionally
        ``timeout_type`` / ``timeout_instrument``.

    Notes
    -----
    Each figure is saved and closed as soon as it's rendered, so a
    mid-orbit timeout can leave some of an orbit's PNGs already written to
    disk. This is safe given deterministic output filenames: a subsequent
    retry of a timed-out orbit simply overwrites the partial set.
    """
    result: dict[str, Any] = {"orbit": orbit_number, "status": "ok", "errors": []}
    orbit_start_time = _time.time()
    timeout_triggered = False
    timeout_type = None
    timeout_instrument = None

    def _save_figure(fig, out_path: str, desc: str) -> None:
        """Save *fig* to *out_path* (if not skipped) and always close it afterward."""
        if not override_plots and os.path.exists(out_path):
            log_exception(f"[SKIP] Plot already exists, skipping: {out_path}", level="message")
            close_all_axes_and_clear(fig)
            return
        try:
            log_exception(
                f"[DEBUG] Saving {desc} plot: y_axis_scale={y_axis_scale}, "
                f"z_axis_scale={z_axis_scale}, filename={out_path}",
                level="message",
            )
            fig.savefig(out_path, dpi=200)
            log_exception(f"[SAVED] {out_path}", level="message")
        except Exception as exc:
            log_exception(f"[FAIL] Saving figure {out_path}", exc, level="error")
            result["status"] = "error"
            result["errors"].append(str(exc))
        finally:
            close_all_axes_and_clear(fig)

    try:
        first_path = next(
            (instrument_file_paths[k] for k in DEFAULT_INSTRUMENT_ORDER if k in instrument_file_paths),
            None,
        )
        year, month = _parse_year_month(first_path) if first_path else ("unknown", "unknown")
        output_dir = os.path.join(output_base_directory, str(year), str(month), str(orbit_number))
        os.makedirs(output_dir, exist_ok=True)

        for inst_type in DEFAULT_INSTRUMENT_ORDER:
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
                time_unix_array = load_fast_cdf_dataset(cdf_path)["times"]
                vertical_lines = get_timestamps_for_orbit(
                    filtered_orbits_dataframe, orbit_number, inst_detected, time_unix_array
                )
                cusp_tag = "_cusp" if vertical_lines else ""
                y_min_ov, y_max_ov, z_min_ov, z_max_ov = _extrema_overrides(
                    global_extrema, inst_detected, y_axis_scale, z_axis_scale
                )

                fig_given, _canvas_given = FAST_plot_pitch_angle_grid(
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
                    cusp_marker_style=cusp_marker_style,
                    cusp_marker_kwargs=cusp_marker_kwargs,
                )
                if fig_given is not None:
                    fname = (
                        f"{orbit_number}{cusp_tag}_pitch-angle_ESA_{inst_detected}"
                        f"_y-{y_axis_scale}_z-{z_axis_scale}_given_extrema-{colormap}.png"
                    )
                    _save_figure(
                        fig_given, os.path.join(output_dir, fname), f"pitch-angle {inst_detected} (given extrema)"
                    )

                fig_raw, _canvas_raw = FAST_plot_pitch_angle_grid(
                    cdf_path,
                    filtered_orbits_df=filtered_orbits_dataframe,
                    orbit_number=orbit_number,
                    zoom_duration_minutes=zoom_duration_minutes,
                    scale_function_y=y_axis_scale,
                    scale_function_z=z_axis_scale,
                    show=False,
                    colormap=colormap,
                    cusp_marker_style=cusp_marker_style,
                    cusp_marker_kwargs=cusp_marker_kwargs,
                )
                if fig_raw is not None:
                    fname = (
                        f"{orbit_number}{cusp_tag}_pitch-angle_ESA_{inst_detected}"
                        f"_y-{y_axis_scale}_z-{z_axis_scale}_raw-{colormap}.png"
                    )
                    _save_figure(fig_raw, os.path.join(output_dir, fname), f"pitch-angle {inst_detected} (raw extrema)")

            except Exception as exc:
                err = f"[FAIL] Plotting Orbit {orbit_number} pitch angle grid for {inst_type}"
                log_exception(err, exc, level="error")
                result["status"] = "error"
                result["errors"].append(err)
            finally:
                inst_elapsed = _time.time() - inst_start
                log_exception(
                    f"[TIMING] Orbit {orbit_number} instrument {inst_type} elapsed {inst_elapsed:.3f}s",
                    level="message",
                )
                if inst_elapsed > instrument_timeout_seconds and not timeout_triggered:
                    timeout_triggered = True
                    timeout_type = "instrument"
                    timeout_instrument = inst_type
                    log_exception(
                        f"[TIMEOUT] Instrument {inst_type} in orbit {orbit_number} exceeded "
                        f"{instrument_timeout_seconds:.0f}s ({inst_elapsed:.2f}s). Aborting.",
                        level="message",
                    )

        if not timeout_triggered:
            grid_start = _time.time()
            try:
                fig_grid_given, _canvas_grid_given = FAST_plot_instrument_grid(
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
                    cusp_marker_style=cusp_marker_style,
                    cusp_marker_kwargs=cusp_marker_kwargs,
                )
                if fig_grid_given is not None:
                    fname = (
                        f"{orbit_number}_instrument-grid_ESA_y-{y_axis_scale}_z-{z_axis_scale}"
                        f"_given_extrema-{colormap}.png"
                    )
                    _save_figure(fig_grid_given, os.path.join(output_dir, fname), "instrument-grid (given extrema)")

                fig_grid_raw, _canvas_grid_raw = FAST_plot_instrument_grid(
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
                    cusp_marker_style=cusp_marker_style,
                    cusp_marker_kwargs=cusp_marker_kwargs,
                )
                if fig_grid_raw is not None:
                    fname = f"{orbit_number}_instrument-grid_ESA_y-{y_axis_scale}_z-{z_axis_scale}_raw-{colormap}.png"
                    _save_figure(fig_grid_raw, os.path.join(output_dir, fname), "instrument-grid (raw extrema)")

            except Exception as exc:
                err = f"[FAIL] Plotting Orbit {orbit_number} instrument grid"
                log_exception(err, exc, level="error")
                result["status"] = "error"
                result["errors"].append(err)
            finally:
                grid_elapsed = _time.time() - grid_start
                log_exception(
                    f"[TIMING] Orbit {orbit_number} instrument-grid elapsed {grid_elapsed:.3f}s",
                    level="message",
                )
                if grid_elapsed > instrument_timeout_seconds and not timeout_triggered:
                    timeout_triggered = True
                    timeout_type = "instrument"
                    timeout_instrument = "instrument_grid"
                    log_exception(
                        f"[TIMEOUT] Instrument grid in orbit {orbit_number} exceeded "
                        f"{instrument_timeout_seconds:.0f}s ({grid_elapsed:.2f}s). Aborting.",
                        level="message",
                    )

        orbit_elapsed = _time.time() - orbit_start_time
        if orbit_elapsed > orbit_timeout_seconds and not timeout_triggered:
            timeout_triggered = True
            timeout_type = "orbit"
            log_exception(
                f"[TIMEOUT] Orbit {orbit_number} exceeded {orbit_timeout_seconds:.0f}s total ({orbit_elapsed:.2f}s).",
                level="message",
            )

        if timeout_triggered:
            result["status"] = "timeout"
            result["timeout_type"] = timeout_type
            if timeout_instrument:
                result["timeout_instrument"] = timeout_instrument
            return result

    except Exception as exc:
        err = f"[FAIL] Orbit {orbit_number} processing"
        log_exception(err, exc, level="error")
        result["status"] = "error"
        result["errors"].append(err)
    finally:
        gc.collect()

    return result
