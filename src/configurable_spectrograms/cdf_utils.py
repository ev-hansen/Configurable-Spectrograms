"""CDF file discovery, metadata, and dataset-loading helpers.

Shared by both the generic and FAST-specific plotting/batch pipelines so
that file-type detection, orbit-boundary lookup, and CDF loading logic
exists in exactly one place.
"""

from pathlib import Path

import cdflib
import numpy as np
import pandas as pd
from tqdm import tqdm

from configurable_spectrograms.constants import CDF_DATA_DIRECTORY, CDF_VARIABLE_NAMES, FILTERED_ORBITS_CSV_PATH
from configurable_spectrograms.logging_utils import log_error

# Module-level caches to avoid repeated disk I/O / recomputation in batch routines.
filtered_orbits_cache: dict[str, pd.DataFrame | None] = {}
orbit_column_cache: dict[tuple[int, str], tuple[str, str, str]] = {}
cdf_type_cache: dict[str, str | None] = {}

INSTRUMENT_TAGS = ("ees", "eeb", "ies", "ieb")


def load_filtered_orbits(csv_path: str = FILTERED_ORBITS_CSV_PATH) -> pd.DataFrame | None:
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
    A module-level dictionary caches previously loaded DataFrames keyed by
    the path string to avoid repeated disk I/O in batch routines.
    """
    if csv_path in filtered_orbits_cache:
        return filtered_orbits_cache[csv_path]
    try:
        dataframe = pd.read_csv(csv_path, sep="\t")
    except OSError as exc:
        log_error(f"Error loading CSV {csv_path}: {exc}")
        return None
    filtered_orbits_cache[csv_path] = dataframe
    return dataframe


def get_timestamps_for_orbit(
    filtered_orbits_dataframe: pd.DataFrame | None,
    orbit_number: int,
    instrument_type: str | None,
    time_unix_array: np.ndarray | None,
) -> list[float]:
    """Compute orbit boundary UNIX timestamps from filtered indices.

    Parameters
    ----------
    filtered_orbits_dataframe : pandas.DataFrame or None
        DataFrame containing filtered orbits and min/max indices per
        instrument.
    orbit_number : int
        Orbit number to look up.
    instrument_type : str or None
        Instrument type identifier (e.g. ``'ees'``, ``'ies'``).
    time_unix_array : numpy.ndarray or None
        1D array of UNIX timestamps for the instrument.

    Returns
    -------
    list of float
        Boundary UNIX timestamps for the orbit: one value when the CSV row
        gives a degenerate (equal) min/max index, two values (start, end)
        otherwise. Returns an empty list when the orbit is not found or
        inputs are missing.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> orbits = pd.DataFrame({"orbit": [42], "ees min index": [1], "ees max index": [3]})
    >>> times = np.array([100.0, 200.0, 300.0, 400.0])
    >>> get_timestamps_for_orbit(orbits, 42, "ees", times)
    [200.0, 400.0]
    >>> get_timestamps_for_orbit(orbits, 99, "ees", times)
    []
    """
    dataframe = filtered_orbits_dataframe
    if dataframe is None or instrument_type is None or time_unix_array is None:
        return []
    cache_key = (id(dataframe), instrument_type)
    if cache_key not in orbit_column_cache:
        orbit_column = next(col for col in dataframe.columns if "orbit" in col.lower())
        min_index_column = next(
            col for col in dataframe.columns if instrument_type in col.lower() and "min index" in col.lower()
        )
        max_index_column = next(
            col for col in dataframe.columns if instrument_type in col.lower() and "max index" in col.lower()
        )
        orbit_column_cache[cache_key] = (orbit_column, min_index_column, max_index_column)
    orbit_column, min_index_column, max_index_column = orbit_column_cache[cache_key]
    row = dataframe[dataframe[orbit_column] == orbit_number]
    if row.empty:
        return []
    try:
        min_index = int(row.iloc[0][min_index_column])
        max_index = int(row.iloc[0][max_index_column])
    except (TypeError, ValueError):
        from configurable_spectrograms.logging_utils import log_message

        log_message("[WARN] Non-integer indices found in orbit row, using 0.")
        return []
    min_index = max(0, min(min_index, len(time_unix_array) - 1))
    max_index = max(0, min(max_index, len(time_unix_array) - 1))
    if min_index == max_index:
        return [float(time_unix_array[min_index])]
    return [float(time_unix_array[min_index]), float(time_unix_array[max_index])]


def get_cdf_file_type(cdf_file_path: str) -> str | None:
    """Infer instrument type from a CDF file path.

    Parameters
    ----------
    cdf_file_path : str
        Path to the CDF file.

    Returns
    -------
    str or None
        Instrument type string (e.g. ``'ees'``), ``'orb'`` for orbit files,
        or ``None`` if not recognized.

    Examples
    --------
    >>> get_cdf_file_type("fa_esa_l2_eeb_20000101001737_13312_v02.cdf")
    'eeb'
    >>> get_cdf_file_type("fa_k0_orb_13312_v01.cdf")
    'orb'
    """
    path_lower = cdf_file_path.lower()
    if "_orb_" in path_lower:
        return "orb"
    for tag in INSTRUMENT_TAGS:
        if f"_{tag}_" in path_lower:
            return tag
    log_error(f"Unknown CDF file type for path: {cdf_file_path}")
    return None


def get_variable_shape(cdf_path: str, variable_name: str) -> tuple[int, ...] | None:
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
        Variable shape tuple, or ``None`` if the variable is absent, not an
        array, or an error occurs.
    """
    instrument_type = cdf_type_cache.get(cdf_path)
    if instrument_type is None:
        instrument_type = get_cdf_file_type(cdf_path)
        cdf_type_cache[cdf_path] = instrument_type
    if instrument_type is None or instrument_type == "orb":
        return None
    try:
        with cdflib.CDF(cdf_path) as cdf:
            variable_data = cdf.varget(variable_name)
            return variable_data.shape if isinstance(variable_data, np.ndarray) else None
    except Exception as exc:
        log_error(f"Error reading {cdf_path} for variable {variable_name}: {exc}")
        return None


def get_cdf_var_shapes(
    cdf_folder_path: str = CDF_DATA_DIRECTORY,
    variable_names: list[str] = CDF_VARIABLE_NAMES,
) -> dict[str, list[tuple[int, ...] | None]]:
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
        Mapping from variable name (str) to a list of shape tuples (or
        ``None``) per file.
    """
    cdf_file_paths = [str(p) for p in Path(cdf_folder_path).rglob("*.[cC][dD][fF]")]
    shapes_by_variable = {}
    for variable_name in variable_names:
        shapes_by_variable[variable_name] = [
            get_variable_shape(cdf_path, variable_name)
            for cdf_path in tqdm(
                cdf_file_paths,
                desc=f"Processing CDF files ({variable_name})",
                unit="file",
                total=len(cdf_file_paths),
            )
        ]
    return shapes_by_variable


def load_fast_cdf_dataset(
    cdf_path: str, variable_names: tuple[str, ...] = tuple(CDF_VARIABLE_NAMES)
) -> dict[str, np.ndarray]:
    """Load and reshape a FAST CDF file's time/data/energy/pitch-angle arrays.

    Energy and pitch-angle variables are collapsed from their raw
    ``(time, angle, energy)`` or ``(time, energy, angle)`` storage down to
    1D bin arrays, and ``data`` is transposed to
    ``(time, pitch_angle, energy)`` order when needed, so the result is
    ready to collapse along pitch angle for a spectrogram.

    Parameters
    ----------
    cdf_path : str
        Path to the instrument CDF file.
    variable_names : tuple of str, default CDF_VARIABLE_NAMES
        Names of the (time, data, energy, pitch_angle) variables, in that
        order.

    Returns
    -------
    dict
        Mapping with keys ``'times'``, ``'data'``, ``'energy'``,
        ``'pitch_angle'``.
    """
    with cdflib.CDF(cdf_path) as cdf_file:
        times = np.asarray(cdf_file.varget(variable_names[0]))
        data = np.asarray(cdf_file.varget(variable_names[1]))
        energy_full = np.asarray(cdf_file.varget(variable_names[2]))
        pitch_angle_full = np.asarray(cdf_file.varget(variable_names[3]))
    energy = energy_full[0, 0, :] if energy_full.ndim == 3 else energy_full
    pitch_angle = pitch_angle_full[0, :, 0] if pitch_angle_full.ndim == 3 else pitch_angle_full
    if data.shape[1] == len(energy) and data.shape[2] == len(pitch_angle):
        data = np.transpose(data, (0, 2, 1))
    return {"times": times, "data": data, "energy": energy, "pitch_angle": pitch_angle}
