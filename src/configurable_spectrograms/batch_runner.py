"""Executor-agnostic batch execution with resumable progress tracking.

:func:`run_batch` is the shared scaffolding (progress-JSON load/merge/flush,
buffered-log flush cadence, ``as_completed`` loop, SIGINT handling) used by
both the CPU-bound plotting batch driver
(:func:`configurable_spectrograms.generic_batch.generic_batch_plot`, which
supplies a ``ProcessPoolExecutor`` factory) and the I/O-bound download batch
driver (:func:`configurable_spectrograms.download.download_cdf_files_threaded`,
which supplies a ``ThreadPoolExecutor`` factory). Callers choose the
concurrency primitive that matches their workload; this module only
provides the orchestration around whichever executor they hand it.
"""

import concurrent.futures
import json
import os
import signal
import sys
from collections.abc import Callable, Iterable
from typing import Any

from configurable_spectrograms.logging_utils import configure_log_batch, flush_log_buffer, log_error, log_message
from configurable_spectrograms.process_utils import terminate_all_child_processes


def _sigint_handler(signum, frame) -> None:
    """SIGINT handler that terminates children and exits promptly."""
    log_message("[INFO] SIGINT received. Terminating all child processes and exiting.")
    terminate_all_child_processes()
    sys.exit(1)


def run_batch(
    items: Iterable[Any],
    worker_fn: Callable[[Any], tuple[Any, str]],
    executor_factory: Callable[[], concurrent.futures.Executor],
    progress_json_path: str | None = None,
    ignore_progress_json: bool = False,
    flush_batch_size: int = 10,
    log_flush_batch_size: int | None = None,
    install_signal_handlers: bool = True,
) -> list[tuple[Any, str]]:
    """Run ``worker_fn`` over ``items`` in parallel with resumable progress tracking.

    Parameters
    ----------
    items : iterable
        Iterable of item identifiers (any ``repr``-able objects).
    worker_fn : callable
        Callable taking one item and returning ``(item, status)``, where
        ``status`` is a short label such as ``'ok'``, ``'no_data'``, or
        ``'error'``.
    executor_factory : callable
        Zero-argument callable returning a fresh
        ``concurrent.futures.Executor`` to use as a context manager (e.g.
        ``functools.partial(ProcessPoolExecutor, max_workers=4)`` for
        CPU-bound work, or ``functools.partial(ThreadPoolExecutor,
        max_workers=8)`` for I/O-bound work).
    progress_json_path : str or None, optional
        Path to a JSON file used for resumable progress tracking across
        runs. ``None`` disables persistence.
    ignore_progress_json : bool, default False
        If ``True``, skip reading existing progress prior to execution.
    flush_batch_size : int, default 10
        Progress/log batch size; values less than 1 are coerced to 1. The
        final partial batch is always flushed.
    log_flush_batch_size : int or None, optional
        Explicit log batch size; if ``None``, reuses ``flush_batch_size``.
    install_signal_handlers : bool, default True
        When ``True``, a temporary SIGINT handler is installed (and restored
        on exit) to enable graceful interruption with a final progress/log
        flush.

    Returns
    -------
    list of tuple
        Sequence of ``(item, status)`` results, one per submitted item.

    Notes
    -----
    Items are identified via ``repr(item)`` for data-agnostic progress
    persistence, matching the pattern used across this codebase's batch
    drivers.
    """
    previous_sigint = None
    if install_signal_handlers:
        try:
            previous_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _sigint_handler)
        except (ValueError, OSError) as sig_setup_exc:
            log_message(f"[WARN] Could not install temporary SIGINT handler: {sig_setup_exc}")

    flush_batch_size = max(1, int(flush_batch_size))
    configure_log_batch(log_flush_batch_size or flush_batch_size)

    progress_state: dict[str, Any] = {
        "completed_items": [],
        "errors": [],
        "no_data": [],
        "last_index": -1,
        "schema_version": 1,
    }
    if progress_json_path is not None and not ignore_progress_json and os.path.exists(progress_json_path):
        try:
            with open(progress_json_path) as progress_in:
                loaded = json.load(progress_in)
            if isinstance(loaded, dict):
                for key in progress_state:
                    if key in loaded:
                        progress_state[key] = loaded[key]
        except (OSError, json.JSONDecodeError) as progress_read_exc:
            log_error(f"[PROGRESS] Failed to read existing progress JSON '{progress_json_path}': {progress_read_exc}")

    item_list = list(items)
    completed_set = set(progress_state.get("completed_items", []))
    pending_items = [item for item in item_list if repr(item) not in completed_set]
    log_message(
        f"[BATCH] Starting batch run with {len(pending_items)} pending / {len(item_list)} total items; "
        f"flush_batch_size={flush_batch_size}"
    )

    pending_progress_write_count = 0

    def _flush_progress(force: bool = False) -> None:
        nonlocal pending_progress_write_count
        if progress_json_path is None:
            return
        if pending_progress_write_count == 0 and not force:
            return
        if pending_progress_write_count < flush_batch_size and not force:
            return
        try:
            with open(progress_json_path, "w") as progress_out:
                json.dump(progress_state, progress_out, indent=2)
            pending_progress_write_count = 0
        except OSError as progress_write_exc:
            log_error(f"[PROGRESS] Failed writing progress JSON '{progress_json_path}': {progress_write_exc}")

    results: list[tuple[Any, str]] = []
    processed_item_count = 0
    with executor_factory() as executor:
        future_map = {executor.submit(worker_fn, item): item for item in pending_items}
        for finished_future in concurrent.futures.as_completed(future_map):
            original_item = future_map[finished_future]
            try:
                item_identifier, status = finished_future.result()
            except Exception as batch_future_exc:
                status = "error"
                item_identifier = original_item
                log_error(f"[BATCH-FAIL] Item {original_item} outer exception: {batch_future_exc}")
            results.append((item_identifier, status))
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

    _flush_progress(force=True)
    flush_log_buffer(force=True)
    log_message(
        "[BATCH] Completed batch run: "
        f"{processed_item_count} processed (ok={sum(1 for _, s in results if s == 'ok')} "
        f"no_data={sum(1 for _, s in results if s == 'no_data')} "
        f"error={sum(1 for _, s in results if s == 'error')})",
        force_flush=True,
    )
    if install_signal_handlers and previous_sigint is not None:
        try:
            signal.signal(signal.SIGINT, previous_sigint)
        except (ValueError, OSError) as sig_restore_exc:
            log_message(f"[WARN] Could not restore original SIGINT handler: {sig_restore_exc}")
    return results
