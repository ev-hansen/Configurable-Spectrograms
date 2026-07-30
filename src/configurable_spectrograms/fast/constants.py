"""FAST-instrument-specific paths, variable names, and default colormaps."""

from configurable_spectrograms.constants import (
    COLLAPSE_FUNCTION,
    COLORMAP_LINEAR_Y_LINEAR_Z,
    COLORMAP_LINEAR_Y_LOG_Z,
    COLORMAP_LOG_Y_LINEAR_Z,
    COLORMAP_LOG_Y_LOG_Z,
)

FAST_CDF_DATA_FOLDER_PATH = "./FAST_data/"
FAST_FILTERED_ORBITS_CSV_PATH = "./FAST_Cusp_Indices.csv"
FAST_PLOTTING_PROGRESS_JSON = "./batch_multi_plot_FAST_progress.json"
FAST_OUTPUT_BASE = "./FAST_plots/"
FAST_LOGFILE_PREFIX = "./batch_multi_plot_FAST_log"
FAST_LOGFILE_DATETIME_MARKER_PATH = "./batch_multi_plot_FAST_logfile_datetime.txt"
FAST_EXTREMA_JSON_PATH = "./FAST_calculated_extrema.json"

#: Same collapse function as the generic pipeline (kept as a distinct name
#: for readability at FAST call sites).
FAST_COLLAPSE_FUNCTION = COLLAPSE_FUNCTION

CDF_VARIABLES = ("time_unix", "data", "energy", "pitch_angle")

DEFAULT_INSTRUMENT_ORDER = ("ees", "eeb", "ies", "ieb")

# Colormaps for each axis-scaling combination (colorblind-friendly and visually distinct);
# aliases of the shared generic constants so both pipelines have exactly one source of truth.
DEFAULT_COLORMAP_LINEAR_Y_LINEAR_Z = COLORMAP_LINEAR_Y_LINEAR_Z
DEFAULT_COLORMAP_LINEAR_Y_LOG_Z = COLORMAP_LINEAR_Y_LOG_Z
DEFAULT_COLORMAP_LOG_Y_LINEAR_Z = COLORMAP_LOG_Y_LINEAR_Z
DEFAULT_COLORMAP_LOG_Y_LOG_Z = COLORMAP_LOG_Y_LOG_Z

#: Default pitch-angle category boundaries (degrees) used when a caller
#: doesn't supply their own mapping.
DEFAULT_PITCH_ANGLE_CATEGORIES: dict[str, list[tuple[float, float]]] = {
    "downgoing\n(0, 30), (330, 360)": [(0.0, 30.0), (330.0, 360.0)],
    "upgoing\n(150, 210)": [(150.0, 210.0)],
    "perpendicular\n(40, 140), (210, 330)": [(40.0, 140.0), (210.0, 330.0)],
    "all\n(0, 360)": [(0.0, 360.0)],
}
