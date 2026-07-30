"""Buffered logging shared by the generic and FAST batch/plotting pipelines.

Log messages are queued in memory and flushed to disk in batches to avoid a
disk write per message during large batch runs. The destination file is set
explicitly via :func:`set_logfile_path` (typically once, from a CLI's
``main()``) rather than resolved as a side effect of importing this module,
so importing the library never touches the filesystem.
"""

import traceback
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

_LOG_BUFFER: list[tuple[str, str]] = []
_LOG_BATCH_SIZE = 10
_LOGFILE_PATH: str | None = None


def get_logfile_path(prefix: str, datetime_marker_path: str) -> str:
    """Return a persistent per-run log file path derived from a marker file.

    The marker file at *datetime_marker_path* holds a timestamp string that is
    created on first use and reused afterwards, so repeated runs of the same
    pipeline share one logfile instead of minting a new one on every call.

    Parameters
    ----------
    prefix : str
        Filename prefix for the resulting log path (e.g. ``'./batch_log'``).
    datetime_marker_path : str
        Path to a small text file used to persist the timestamp marker.

    Returns
    -------
    str
        Log file path of the form ``f"{prefix}_{timestamp}.log"``.
    """
    marker = Path(datetime_marker_path)
    marker_text = marker.read_text().strip() if marker.exists() else ""
    if not marker_text:
        marker_text = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        marker.write_text(marker_text)
    return f"{prefix}_{marker_text}.log"


def set_logfile_path(path: str | None) -> None:
    """Set the destination file that buffered log flushes are written to."""
    global _LOGFILE_PATH
    _LOGFILE_PATH = path


def configure_log_batch(batch_size: int) -> None:
    """Configure buffered logging batch size.

    Parameters
    ----------
    batch_size : int
        Desired number of log records to accumulate before an automatic
        flush. Values less than 1 are coerced to 1.
    """
    global _LOG_BATCH_SIZE
    _LOG_BATCH_SIZE = max(1, int(batch_size))


def _flush_log_buffer(force: bool = False) -> None:
    """Flush buffered log messages to disk.

    Parameters
    ----------
    force : bool, default False
        If True, flush even if the current buffer length is below the
        configured batch size threshold.
    """
    if not _LOG_BUFFER:
        return
    if len(_LOG_BUFFER) < _LOG_BATCH_SIZE and not force:
        return
    if _LOGFILE_PATH is None:
        _LOG_BUFFER.clear()
        return
    try:
        with open(_LOGFILE_PATH, "a") as logfile_out:
            for level, msg in _LOG_BUFFER:
                logfile_out.write(f"[ERROR] {msg}\n" if level == "error" else msg + "\n")
    except OSError as log_flush_exception:
        tqdm.write(f"[ERROR] Failed flushing log buffer: {log_flush_exception}")
    finally:
        _LOG_BUFFER.clear()


def log_message(message: str, force_flush: bool = False) -> None:
    """Queue an informational log message.

    Messages are appended to an in-memory buffer; a flush occurs
    automatically once the configured batch size is reached or
    ``force_flush`` is True.
    """
    _LOG_BUFFER.append(("info", message))
    _flush_log_buffer(force=force_flush)


def log_error(message: str, force_flush: bool = False) -> None:
    """Queue an error log message and echo it to the console immediately."""
    tqdm.write("[ERROR] " + message)
    _LOG_BUFFER.append(("error", message))
    _flush_log_buffer(force=force_flush)


def flush_log_buffer(force: bool = True) -> None:
    """Publicly flush any buffered log messages to disk (see :func:`log_message`)."""
    _flush_log_buffer(force=force)


def log_exception(
    prefix: str,
    exception: BaseException | None = None,
    level: str = "error",
    include_trace: bool = False,
    force_flush: bool = False,
) -> None:
    """Log a message, optionally with an exception and traceback.

    Parameters
    ----------
    prefix : str
        Human-readable message prefix.
    exception : BaseException or None, optional
        Optional exception; if given, its class name and value are appended.
    level : {'error', 'message'}, default 'error'
        ``'error'`` routes to :func:`log_error`; anything else routes to
        :func:`log_message`.
    include_trace : bool, default False
        If True and an exception is given, also log a formatted traceback.
    force_flush : bool, default False
        Force an immediate buffer flush after logging this message (and the
        traceback, if any).
    """
    exception_name = type(exception).__name__ if exception is not None else None
    message = f"{prefix} [{exception_name}]: {exception}" if exception_name else str(prefix)
    (log_error if level == "error" else log_message)(message, force_flush=force_flush)
    if include_trace and exception is not None:
        trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        log_message("[TRACE]\n" + trace, force_flush=force_flush)
