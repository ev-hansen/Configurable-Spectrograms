"""FAST orbit/instrument file discovery and progress-key bookkeeping."""

import os
from collections import defaultdict
from pathlib import Path

from configurable_spectrograms.cdf_utils import get_cdf_file_type
from configurable_spectrograms.fast.constants import DEFAULT_INSTRUMENT_ORDER
from configurable_spectrograms.logging_utils import log_exception


def _parse_year_month(file_path: str) -> tuple[str, str]:
    """Extract ``(year, month)`` from a CDF path containing a YYYY/MM directory pair.

    Parameters
    ----------
    file_path : str
        Path expected to contain a 4-digit year directory followed by a
        2-digit month directory (e.g. ``.../2000/01/...``).

    Returns
    -------
    tuple of str
        ``(year, month)``, or ``('unknown', 'unknown')`` when the pattern is
        not found.

    Examples
    --------
    >>> _parse_year_month("./FAST_data/2000/01/fa_esa_l2_eeb_20000101001737_13312_v02.cdf")
    ('2000', '01')
    >>> _parse_year_month("no_year_here.cdf")
    ('unknown', 'unknown')
    """
    parts = Path(file_path).parts
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 4:
            nxt = parts[i + 1] if i + 1 < len(parts) else ""
            month = nxt if nxt.isdigit() and len(nxt) == 2 else "unknown"
            return part, month
    return "unknown", "unknown"


def _classify_error_reason(msg: str) -> str:
    """Map an error message to a short reason token for progress JSON keys.

    Parameters
    ----------
    msg : str
        Raw error message text.

    Returns
    -------
    str
        One of ``'divide-by-zero'``, ``'invalid-cdf'``, ``'timeout'``,
        ``'plotting'``, or ``'generic'``.

    Examples
    --------
    >>> _classify_error_reason("divide by zero encountered")
    'divide-by-zero'
    >>> _classify_error_reason("Timeout while processing orbit")
    'timeout'
    >>> _classify_error_reason("something else entirely")
    'generic'
    """
    lowered = msg.lower()
    if "divide" in lowered and "zero" in lowered:
        return "divide-by-zero"
    if "invalid" in lowered and "cdf" in lowered:
        return "invalid-cdf"
    if "timeout" in lowered:
        return "timeout"
    if "plot" in lowered:
        return "plotting"
    return "generic"


def _add_to_orbit_list(progress_dict: dict, key: str, orbit: int) -> None:
    """Add *orbit* to the sorted list at ``progress_dict[key]``, creating the key if absent.

    Examples
    --------
    >>> progress = {}
    >>> _add_to_orbit_list(progress, "errors", 5)
    >>> _add_to_orbit_list(progress, "errors", 3)
    >>> progress["errors"]
    [3, 5]
    """
    progress_dict[key] = sorted(set(progress_dict.get(key, [])) | {orbit})


def extract_orbit_and_instrument(cdf_path: str) -> tuple[int, str, str] | None:
    """Parse a CDF filename to ``(orbit_number, instrument_type, cdf_path)``.

    Parameters
    ----------
    cdf_path : str
        Path (or bare filename) of the CDF file.

    Returns
    -------
    tuple or None
        ``(orbit_number, instrument_type, cdf_path)``, or ``None`` when the
        filename does not match the expected pattern, the orbit number
        cannot be parsed, or the instrument type is ``None`` or ``'orb'``.

    Examples
    --------
    >>> extract_orbit_and_instrument("fa_esa_l2_eeb_20000101001737_13312_v02.cdf")
    (13312, 'eeb', 'fa_esa_l2_eeb_20000101001737_13312_v02.cdf')
    >>> extract_orbit_and_instrument("fa_k0_orb_13312_v01.cdf") is None
    True
    """
    filename = os.path.basename(cdf_path)
    parts = filename.split("_")
    if len(parts) < 5:
        return None
    try:
        orbit_number = int(parts[-2])
    except ValueError as exc:
        log_exception(f"[ERROR] Invalid orbit number in filename: {filename}", exc, level="message")
        return None
    instrument_type = get_cdf_file_type(cdf_path)
    if instrument_type is None or instrument_type == "orb":
        return None
    return (orbit_number, instrument_type, cdf_path)


def discover_orbit_files(
    directory_path: str, instrument_order: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER
) -> dict[int, dict[str, str]]:
    """Discover FAST instrument CDF files and group them by orbit.

    Walks *directory_path* recursively for non-orbit-ephemeris CDF files
    (paths containing ``_orb_`` are excluded), parses each file's orbit
    number and instrument type, and groups them into
    ``{orbit: {instrument: path}}``.

    Parameters
    ----------
    directory_path : str
        Root folder containing instrument CDF files.
    instrument_order : tuple of str, default DEFAULT_INSTRUMENT_ORDER
        Instrument codes to include; files for other instruments are
        skipped.

    Returns
    -------
    dict of {int: dict of {str: str}}
        Mapping of orbit number to ``{instrument: cdf_path}``. When multiple
        files exist for the same orbit/instrument pair, the last one seen
        during the directory walk wins.
    """
    orbit_to_instruments: dict[int, dict[str, str]] = defaultdict(dict)
    for path_obj in Path(directory_path).rglob("*.[cC][dD][fF]"):
        candidate_path = str(path_obj)
        if "_orb_" in candidate_path.lower():
            continue
        parsed = extract_orbit_and_instrument(candidate_path)
        if parsed is None:
            continue
        orbit_number, instrument_type, cdf_path = parsed
        if instrument_type not in instrument_order:
            continue
        orbit_to_instruments[orbit_number][instrument_type] = cdf_path
    return dict(orbit_to_instruments)


def resolve_shared_orbit(instrument_day_files: dict[str, list[str]]) -> tuple[int | None, dict[str, str]]:
    """Pick one orbit's worth of files out of a day's downloaded/discovered CDFs.

    A single FAST day commonly spans multiple orbits per instrument, each a
    separate CDF file. Callers that plot one orbit at a time (e.g.
    :func:`configurable_spectrograms.fast.plotting.FAST_plot_instrument_grid`)
    need exactly one file per instrument, so this resolves the day down to
    the orbit number shared by the most instruments, breaking ties by
    picking the lowest orbit number.

    Parameters
    ----------
    instrument_day_files : dict of {str: list of str}
        Mapping of instrument key to every CDF file path found for one day,
        as returned by
        :func:`configurable_spectrograms.download.download_single_day_cdf`.

    Returns
    -------
    tuple[int or None, dict of {str: str}]
        The resolved orbit number (``None`` if no file parsed an orbit
        number at all) and a mapping of instrument -> single file path for
        that orbit. Instruments with no file for the resolved orbit are
        omitted from the mapping.

    Examples
    --------
    >>> resolve_shared_orbit({
    ...     "eeb": ["fa_esa_l2_eeb_20000101001737_100_v02.cdf",
    ...             "fa_esa_l2_eeb_20000101031737_101_v02.cdf"],
    ...     "ies": ["fa_esa_l2_ies_20000101001738_100_v02.cdf"],
    ... })
    (100, {'eeb': 'fa_esa_l2_eeb_20000101001737_100_v02.cdf', 'ies': 'fa_esa_l2_ies_20000101001738_100_v02.cdf'})
    >>> resolve_shared_orbit({"eeb": [], "ies": []})
    (None, {})
    """
    orbit_to_instruments: dict[int, dict[str, str]] = {}
    for file_paths in instrument_day_files.values():
        for file_path in file_paths:
            parsed = extract_orbit_and_instrument(file_path)
            if parsed is None:
                continue
            orbit_number, instrument_type, cdf_path = parsed
            orbit_to_instruments.setdefault(orbit_number, {})[instrument_type] = cdf_path
    if not orbit_to_instruments:
        return None, {}
    best_orbit = max(orbit_to_instruments, key=lambda orbit: (len(orbit_to_instruments[orbit]), -orbit))
    return best_orbit, orbit_to_instruments[best_orbit]


def resolve_orbit_from_files(instrument_files: dict[str, str]) -> int | None:
    """Best-effort orbit number for a manually-assembled instrument file mapping.

    Used for title/vertical-line labeling when a caller supplies its own
    ``{instrument: file_path}`` mapping directly rather than discovering one
    from a folder via :func:`discover_orbit_files`, so no orbit number is
    known up front.

    Parameters
    ----------
    instrument_files : dict of {str: str}
        Mapping of instrument key to CDF file path.

    Returns
    -------
    int or None
        The orbit number parsed from the first file in *instrument_files*
        whose name matches the expected FAST CDF naming pattern, or
        ``None`` if none do.

    Examples
    --------
    >>> resolve_orbit_from_files({"eeb": "fa_esa_l2_eeb_20000101001737_13312_v02.cdf"})
    13312
    >>> resolve_orbit_from_files({"eeb": "not_a_fast_file.cdf"}) is None
    True
    """
    for file_path in instrument_files.values():
        parsed = extract_orbit_and_instrument(file_path)
        if parsed is not None:
            return parsed[0]
    return None
