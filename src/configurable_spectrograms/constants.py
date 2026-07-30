"""Shared constants for spectrogram plotting and batch processing."""

import numpy as np

#: Directory containing CDF data files.
CDF_DATA_DIRECTORY = "./FAST_data/"

#: List of variable names expected in CDF files.
CDF_VARIABLE_NAMES = ["time_unix", "data", "energy", "pitch_angle"]

#: Function used to collapse a 3D data array down to 2D (e.g. sum over pitch angle).
COLLAPSE_FUNCTION = np.nansum

# Colormaps for different axis-scaling combinations (colorblind-friendly and visually distinct).
COLORMAP_LINEAR_Y_LINEAR_Z = "viridis"
COLORMAP_LINEAR_Y_LOG_Z = "cividis"
COLORMAP_LOG_Y_LINEAR_Z = "plasma"
COLORMAP_LOG_Y_LOG_Z = "inferno"

# Plot configuration.
PLOT_FIGURE_WIDTH_INCHES = 6.25
PLOT_FIGURE_HEIGHT_INCHES = 2.0
TICK_LABEL_FONT_SIZE = 15
AXIS_LABEL_FONT_SIZE = 18
DEFAULT_ZOOM_WINDOW_MINUTES = 6  # Default zoom window duration in minutes.

#: Path to the filtered cusp orbits CSV.
FILTERED_ORBITS_CSV_PATH = "./FAST_Cusp_Indices.csv"

#: Path to JSON tracking generic batch-plotting progress across sessions.
PLOTTING_PROGRESS_JSON_PATH = "./batch_multi_plot_progress.json"

#: Parent directory for generic batch-plot output.
OUTPUT_BASE_DIRECTORY = "./plots/"
