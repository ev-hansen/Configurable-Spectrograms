"""Batch driver: process every orbit in a FAST CDF directory in parallel."""

import concurrent.futures
import json
import os
import signal
import time as _time
from typing import Any

from tqdm import tqdm

from configurable_spectrograms.cdf_utils import load_filtered_orbits
from configurable_spectrograms.constants import DEFAULT_ZOOM_WINDOW_MINUTES
from configurable_spectrograms.fast.constants import (
    FAST_CDF_DATA_FOLDER_PATH,
    FAST_OUTPUT_BASE,
    FAST_PLOTTING_PROGRESS_JSON,
)
from configurable_spectrograms.fast.extrema import compute_global_extrema
from configurable_spectrograms.fast.orbit_discovery import (
    _add_to_orbit_list,
    _classify_error_reason,
    discover_orbit_files,
)
from configurable_spectrograms.fast.process_orbit import FAST_process_single_orbit
from configurable_spectrograms.logging_utils import configure_log_batch, flush_log_buffer, log_exception
from configurable_spectrograms.process_utils import terminate_all_child_processes

_INSTRUMENT_KEYS = ("ees", "eeb", "ies", "ieb")


def FAST_plot_spectrograms_directory(
    directory_path: str = FAST_CDF_DATA_FOLDER_PATH,
    output_base: str = FAST_OUTPUT_BASE,
    y_scale: str = "linear",
    z_scale: str = "log",
    zoom_duration_minutes: float = DEFAULT_ZOOM_WINDOW_MINUTES,
    instrument_order: tuple[str, ...] = _INSTRUMENT_KEYS,
    verbose: bool = True,
    progress_json_path: str | None = FAST_PLOTTING_PROGRESS_JSON,
    ignore_progress_json: bool = False,
    use_tqdm: bool | None = None,
    colormap: str = "viridis",
    cusp_marker_style: str = "both",
    cusp_marker_kwargs: dict | None = None,
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

    Discovers instrument CDF files (excluding ``_orb_``), groups them by
    orbit, and processes each orbit in parallel worker processes (safe for
    matplotlib). Progress is persisted to a JSON file to support resumable
    runs. When ``max_processing_percentile`` is not None, a global extrema
    pass runs first (:func:`configurable_spectrograms.fast.extrema.compute_global_extrema`)
    and both raw and given-extrema plots are saved; otherwise only raw plots
    are produced.

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
    cusp_marker_style : {'line', 'bracket', 'both'}, default 'both'
        Cusp-boundary marker style forwarded to every orbit's plots.
    cusp_marker_kwargs : dict or None, optional
        Extra keyword arguments forwarded to the marker-drawing function.
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
        Percentile (0-100] for pooled intensity (Z) maxima in
        ``compute_global_extrema``. None skips the extrema pass and
        raw-only plots are produced. Energy (Y) maxima use a fixed 99%
        cumulative coverage rule regardless.
    override_plots : bool, default True
        If False, skip plots whose output file already exists.

    Returns
    -------
    list of dict
        Result dictionaries from ``FAST_process_single_orbit`` (and
        retries).

    Raises
    ------
    KeyboardInterrupt
        Re-raised on SIGINT/SIGTERM so the caller can stop multi-combo loops.

    Notes
    -----
    - Progress JSON key ``f"progress_{y_scale}_{z_scale}_last_orbit"`` tracks
      the last completed orbit; error/timeout orbits are recorded under
      dedicated keys (including per-instrument).
    - Signal handlers terminate child processes and raise
      ``KeyboardInterrupt`` to interrupt the main wait loop immediately.
    """
    shutdown_requested = {"flag": False}

    def _signal_handler(signum, frame):  # frame unused
        if not shutdown_requested["flag"]:
            log_exception(f"[INTERRUPT] Signal {signum} received. Requesting shutdown...", level="message")
            shutdown_requested["flag"] = True
            try:
                terminate_all_child_processes()
            finally:
                raise KeyboardInterrupt
        else:
            log_exception("[INTERRUPT] Second interrupt - forcing immediate exit.", level="message")
            try:
                terminate_all_child_processes()
            finally:
                raise SystemExit(130)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, OSError) as exc:
        log_exception("[WARN] Could not register signal handlers", exc, level="message")

    filtered_orbits_dataframe = load_filtered_orbits()
    configure_log_batch(log_flush_batch_size or flush_batch_size)

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

    orbit_to_instruments = discover_orbit_files(directory_path, instrument_order)
    sorted_orbits = sorted(orbit_to_instruments.items(), key=lambda x: x[0])
    total_orbits = len(sorted_orbits)

    progress_key = f"{y_scale}_{z_scale}_last_orbit"
    error_key = f"{y_scale}_{z_scale}_error_plotting"
    progress_data: dict[str, Any] = {}
    last_completed_orbit = None
    error_orbits: set[int] = set()
    if progress_json_path is not None and not ignore_progress_json:
        try:
            with open(progress_json_path) as f:
                progress_data = json.load(f)
            last_completed_orbit = progress_data.get(progress_key)
            error_orbits = set(progress_data.get(error_key, []))
        except (OSError, json.JSONDecodeError) as exc:
            log_exception(
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
        log_exception(
            f"[RESUME] Skipping {start_idx} orbits (up to orbit {last_completed_orbit}). "
            f"{len(error_orbits)} error orbits will also be skipped.",
            level="message",
        )
    else:
        log_exception(
            f"[RESUME] No previous progress found. Starting from the first orbit. "
            f"{len(error_orbits)} error orbits will be skipped if present.",
            level="message",
        )

    use_tqdm_bar = bool(use_tqdm) if use_tqdm is not None else False
    flush_batch_size = max(1, flush_batch_size)

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
            cusp_marker_style,
            cusp_marker_kwargs,
        )

    orbit_args_list: list[tuple] = []
    for orbit_number, instrument_files in sorted_orbits[start_idx:]:
        if orbit_number in error_orbits:
            continue
        orbit_args_list.append(_orbit_args(orbit_number, instrument_files, None))
        if global_extrema is not None:
            orbit_args_list.append(_orbit_args(orbit_number, instrument_files, global_extrema))

    results: list[dict[str, Any]] = []
    _batched_progress_dirty = {"count": 0}

    def save_progress_json(data: dict[str, Any], force: bool = False) -> None:
        """Persist *data* to disk if the batch threshold is met or ``force`` is True."""
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
        except OSError as exc:
            log_exception("[FAIL] Could not write progress JSON", exc, level="error")

    executor = None
    _orbit_completions_since_flush = {"count": 0}

    def _handle_completed_future(fut: concurrent.futures.Future, orbit_number: int) -> None:
        """Consume a completed future, append its result, and update progress JSON."""
        try:
            result = fut.result()
        except Exception as exc:
            log_exception(f"[BATCH] Orbit {orbit_number} generated an exception", exc, level="error")
            result = {"orbit": orbit_number, "status": "error", "errors": [str(exc)]}
            results.append(result)
            if progress_json_path is not None:
                try:
                    with open(progress_json_path) as f:
                        pdisk = json.load(f)
                except (OSError, json.JSONDecodeError):
                    pdisk = {}
                pdisk[progress_key] = orbit_number
                _add_to_orbit_list(pdisk, error_key, orbit_number)
                reason = _classify_error_reason(str(exc))
                _add_to_orbit_list(pdisk, f"unknown_{y_scale}_{z_scale}_error-{reason}", orbit_number)
                _add_to_orbit_list(pdisk, f"{y_scale}_{z_scale}_error-{reason}", orbit_number)
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
            with open(progress_json_path) as f:
                pdisk = json.load(f)
        except (OSError, json.JSONDecodeError):
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
                inst = next((c for c in _INSTRUMENT_KEYS if c in lowered), "unknown")
                _add_to_orbit_list(pdisk, f"{inst}_{y_scale}_{z_scale}_error-{reason}", orbit_number)
                _add_to_orbit_list(pdisk, f"{y_scale}_{z_scale}_error-{reason}", orbit_number)
        elif status_value == "timeout":
            timeout_type = result.get("timeout_type")
            timeout_instrument = result.get("timeout_instrument")
            if timeout_type == "orbit":
                _add_to_orbit_list(pdisk, orbit_timeout_key, orbit_number)
            elif timeout_type == "instrument":
                inst_to = timeout_instrument or "unknown_instrument"
                tk = f"{inst_to}_{y_scale}_{z_scale}_timed_out"
                _add_to_orbit_list(pdisk, tk, orbit_number)

        _orbit_completions_since_flush["count"] += 1
        if _orbit_completions_since_flush["count"] >= flush_batch_size:
            save_progress_json(pdisk, force=True)
            _orbit_completions_since_flush["count"] = 0
        else:
            save_progress_json(pdisk)

    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        future_to_orbit: dict[concurrent.futures.Future, int] = {}
        for args in orbit_args_list:
            if shutdown_requested["flag"]:
                break
            future = executor.submit(FAST_process_single_orbit, *args)
            future_to_orbit[future] = args[0]
        futures = set(future_to_orbit.keys())

        progress_bar = None
        if use_tqdm_bar:
            if start_idx > 0:
                log_exception(
                    f"[RESUME] Resuming progress bar at orbit {start_idx + 1} of {total_orbits} "
                    f"for y_scale={y_scale}, z_scale={z_scale}.",
                    level="message",
                )
            progress_bar = tqdm(
                total=len(futures),
                initial=0,
                desc=f"Plotting - {y_scale} / {z_scale}",
                unit="orbit",
                leave=False,
            )
        try:
            while futures:
                if shutdown_requested["flag"]:
                    break
                done, _ = concurrent.futures.wait(futures, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    futures.discard(fut)
                    orbit_number = future_to_orbit[fut]
                    _handle_completed_future(fut, orbit_number)
                    if progress_bar is not None:
                        progress_bar.set_postfix(orbit=orbit_number)
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        try:
            if progress_json_path is not None and os.path.exists(progress_json_path):
                with open(progress_json_path) as f:
                    final_pd = json.load(f)
            else:
                final_pd = progress_data if isinstance(progress_data, dict) else {}
            save_progress_json(final_pd, force=True)
        except OSError:
            pass

        if shutdown_requested["flag"]:
            log_exception("[INTERRUPT] Shutdown requested; cancelling remaining futures.", level="message")
            for fut in list(futures):
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            _kill_lingering_processes(executor)
            raise KeyboardInterrupt

    except KeyboardInterrupt as exc:
        log_exception(
            f"[INTERRUPT] KeyboardInterrupt caught. Terminating worker processes... Exception: {exc}",
            level="message",
        )
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            _kill_lingering_processes(executor)
        raise
    finally:
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    try:
        if progress_json_path is not None and os.path.exists(progress_json_path):
            with open(progress_json_path) as f:
                final_pd = json.load(f)
        else:
            final_pd = progress_data if isinstance(progress_data, dict) else {}
        save_progress_json(final_pd, force=True)
    except OSError:
        pass
    flush_log_buffer(force=True)

    if retry_timeouts and not shutdown_requested["flag"]:
        results = _retry_timed_out_orbits(
            results,
            orbit_to_instruments,
            _orbit_args,
            max_workers,
            progress_json_path,
            y_scale,
            z_scale,
        )

    return results


def _kill_lingering_processes(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """Best-effort terminate then kill any worker processes still alive after shutdown."""
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    for proc in processes.values():
        try:
            proc.terminate()
        except Exception:
            pass
    _time.sleep(0.05)
    for proc in processes.values():
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


def _retry_timed_out_orbits(
    results: list[dict[str, Any]],
    orbit_to_instruments: dict[int, dict[str, str]],
    orbit_args_fn,
    max_workers: int,
    progress_json_path: str | None,
    y_scale: str,
    z_scale: str,
) -> list[dict[str, Any]]:
    """Retry every orbit whose status is ``'timeout'`` once, with a smaller worker pool."""
    timeout_orbits = [r["orbit"] for r in results if r.get("status") == "timeout"]
    if not timeout_orbits:
        return results

    log_exception(f"[RETRY] Retrying {len(timeout_orbits)} timed-out orbits once.", level="message")
    retry_args = [orbit_args_fn(o, orbit_to_instruments[o], None) for o in timeout_orbits if o in orbit_to_instruments]
    retry_results: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(max_workers, 2)) as retry_executor:
            retry_future_map = {retry_executor.submit(FAST_process_single_orbit, *ra): ra[0] for ra in retry_args}
            for rfut in concurrent.futures.as_completed(retry_future_map):
                r_orbit = retry_future_map[rfut]
                try:
                    r_result = rfut.result()
                    retry_results.append(r_result)
                    log_exception(f"[RETRY] Completed orbit {r_orbit}: {r_result.get('status')}", level="message")
                    if progress_json_path is not None and r_result.get("status") == "ok":
                        _clear_timeout_flag(progress_json_path, r_orbit, y_scale, z_scale)
                except Exception as exc:
                    log_exception(f"[RETRY] Orbit {r_orbit} retry failed", exc, level="error")
                    retry_results.append({"orbit": r_orbit, "status": "error", "errors": [str(exc)]})
    except Exception as exc:
        log_exception("[RETRY] Failed to execute retry pool", exc, level="message")

    results_map = {r["orbit"]: r for r in results}
    for retry_result in retry_results:
        results_map[retry_result["orbit"]] = retry_result
    return list(results_map.values())


def _clear_timeout_flag(progress_json_path: str, orbit: int, y_scale: str, z_scale: str) -> None:
    """Remove *orbit* from every ``*_timed_out`` progress-JSON list after a successful retry."""
    try:
        with open(progress_json_path) as f:
            pdisk = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log_exception("[WARN] Could not read progress JSON for retry cleanup", exc, level="message")
        return
    timeout_keys = [k for k in pdisk if k.endswith(f"_{y_scale}_{z_scale}_timed_out")]
    modified = False
    for tk in timeout_keys:
        if isinstance(pdisk.get(tk), list) and orbit in pdisk[tk]:
            pdisk[tk] = [x for x in pdisk[tk] if x != orbit]
            modified = True
    if modified:
        try:
            with open(progress_json_path, "w") as f:
                json.dump(pdisk, f, indent=2)
        except OSError as exc:
            log_exception("[WARN] Could not write cleaned progress JSON", exc, level="message")
