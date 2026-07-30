# Configurable-Spectrograms
An attempt to make easily configurable spectrograms that allow batch folder processing of arbitrary data.

View documentation for the library on [readthedocs](https://configurable-spectrograms.readthedocs.io/en/latest/) or [github pages](https://ev-hansen.github.io/Configurable-Spectrograms/) (same contents)

## Notable features
- batch processing over entire folders, memory-efficient (figures are saved and closed as soon as they're rendered) and multi-threaded/multi-process (thread pool for CDF downloads, process pool for spectrogram rendering)
- single-plot scripts (and a GUI page) for rendering one file/orbit at a time without running a full batch
- different colormaps for different y- and z- axis scale combinations, including turbo
- ability to mark the auroral cusp region either as a double vertical line (default) or as a bracket spanning the region, with configurable duration for the accompanying zoomed-in plot
- configurable y and z axes
- progress saved mid-run in case scripts need to be interrupted
- example usage for [FAST](https://web.archive.org/web/20250813172018/https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1996-049A) ion and electron ESA data sourced from downloaded [CDF](https://web.archive.org/web/20250813173821/https://cdf.gsfc.nasa.gov/) files

# Download and Installation
1) Make sure you have [git](https://git-scm.com/install/) installed
2) [Install uv](https://docs.astral.sh/uv/getting-started/installation/) so that package versions can match the ones used in this repo
3) Navigate to your preferred destination folder in a command line or terminal and clone this repository with:

    git clone https://github.com/ev-hansen/Configurable-Spectrograms.git
4) In the same command line or terminal, run:
```
 uv venv --python 3.14.6; uv pip install -r requirements.in
```

5) You are now done installing and can use the python files in the specified directory.
6) If you ever need to reset the environment for debug reasons and you are on mac or linux, run
```
 sh reset_uv.sh
```

# Files
## General-use Library (`src/configurable_spectrograms/`)
All reusable plotting/batch/download logic lives in the `configurable_spectrograms` package (installed automatically by `uv run`/`uv sync` via the src-layout in `pyproject.toml`), split by concern:
- `constants.py`, `logging_utils.py`, `process_utils.py` -- shared constants, buffered logging, and child-process cleanup
- `cdf_utils.py`, `percentile_utils.py` -- CDF file/orbit discovery and axis-extrema/percentile helpers
- `cusp_marking.py` -- the two cusp-boundary marker styles (`"line"`, the original double-line marker, and `"bracket"`, which spans the region instead)
- `plotting.py` -- single-output spectrogram rendering (`make_spectrogram`, `generic_plot_spectrogram_set`, `generic_plot_multirow_optional_zoom`)
- `batch_runner.py`, `generic_batch.py` -- the executor-agnostic batch scaffold and the generic (data-agnostic) batch plotting driver built on it
- `download.py` -- FAST CDF downloading, including a thread-pool batch downloader (`download_cdf_files_threaded`)
- `fast/` -- FAST-instrument-specific single-output plotting (`fast/plotting.py`), per-orbit batch worker (`fast/process_orbit.py`), the directory-wide batch driver (`fast/batch_directory.py`), global-extrema computation (`fast/extrema.py`), and orbit/file discovery (`fast/orbit_discovery.py`)

The top-level scripts below import from this package rather than containing plotting logic themselves.

## Single-plot Scripts
- ``single_plot_spectrogram.py`` / ``single_plot_FAST_spectrograms.py``
CLIs that render exactly one spectrogram figure (one CDF file, or one FAST orbit) and exit -- no batch machinery, no progress JSON. Useful for previewing settings before committing to a full batch run.
```
 uv run single_plot_FAST_spectrograms.py --cdf-file path/to/file.cdf --output out.png
 uv run single_plot_FAST_spectrograms.py --data-folder ./FAST_data --orbit 13312 --output out.png
```

## FAST-specific Scripts
- ``batch_multi_plot_FAST_spectrograms.py``
Python script implementing the library's FAST batch driver for use with FAST EISA CDF data. Marked timestamps are when FAST is in the auroral cusp region (shown as a double line by default, or a bracket via `cusp_marker_style="bracket"`), plots spectrograms for pitch angle ranges as well as all instruments, plots for the same "instrument" should be scaled the same in terms of y and z axes (e.g. all ies plots should have same min and max for energy and counts).
This file can be run with the following, along with arguments (or by modifying the file).
```
 uv run batch_multi_plot_FAST_spectrograms.py
```

- ``FAST_CDF_download.py``
Script to download FAST EISA CDF files directly from [NASA's CDAWeb](https://cdaweb.gsfc.nasa.gov/) without manually using the web interface.
This file can be run with
```
 uv run FAST_CDF_download.py
```

For more information, use
```
 uv run FAST_CDF_download.py --help
```

- ``GUI_batch_download_plot_FAST.py``
A GUI file using pyside6 based on material design to assist with batch downloading FAST EISA CDF files, batch plotting the EISA data with spectrograms, and rendering a single spectrogram (one file or one orbit) without running a full batch. Uses ``FAST_CDF_download.py``, ``batch_multi_plot_FAST_spectrograms.py``, and ``configurable_spectrograms.fast.plotting`` directly for the single-plot page.
This file can be run with
```
 uv run GUI_batch_download_plot_FAST.py
```

### Misc Files for FAST Scripts
- [REQUIRED] ``FAST_Cusp_Indices.csv``
CSV file (currently tab-seperated) containing indicies for when FAST CDF files indicate FAST was in the auroral cusp region, currently only covers 2000 and 2001

- [EXAMPLE] ``FAST CDF variables.txt``
An example ``.txt`` file listing the CDF variables and their shape for FAST CDF files using an example orbit number. NOTE: ``time`` dimensions (``epoch``, ``time_unix``, ``unix_time``, etc) may be differently sized depending on orbit number. Additionally, data dimensions are of shape ``time`` x (either 32 or 64) x 96.
