<h1 align="center">Configurable Spectrograms</h1>

<p align="center">
Easily configurable, memory-efficient spectrogram plotting for arbitrary CDF data, with batch and single-output tooling built out for FAST ESA data.
</p>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
<img alt="License: GPL-3.0" src="https://img.shields.io/badge/license-GPL--3.0-blue.svg">
<img alt="Managed with uv" src="https://img.shields.io/badge/managed%20with-uv-de5fe9.svg">
<img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg">
</p>

Documentation is built with Sphinx and published to [Read the Docs](https://configurable-spectrograms.readthedocs.io/en/latest/) and [GitHub Pages](https://ev-hansen.github.io/Configurable-Spectrograms/) (same contents) on every push to `main` -- see [Documentation](#documentation) to build it locally.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Library](#library-srcconfigurable_spectrograms)
- [Scripts](#scripts)
- [Documentation](#documentation)

## Features

- Batch processing over entire folders -- memory-efficient (figures are saved and closed as soon as they're rendered) and concurrent (thread pool for CDF downloads, process pool for spectrogram rendering)
- Single-output scripts (and a GUI page) for rendering one file/orbit at a time without running a full batch
- Different colormaps for different y-/z-axis scale combinations
- The auroral cusp region can be marked as a double vertical line (default) or as a bracket spanning the region, with a configurable duration for the accompanying zoomed-in plot
- Configurable y and z axes
- Progress saved mid-run so batch scripts can be safely interrupted and resumed
- Example usage for [FAST](https://web.archive.org/web/20250813172018/https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1996-049A) ion and electron ESA data sourced from downloaded [CDF](https://web.archive.org/web/20250813173821/https://cdf.gsfc.nasa.gov/) files

## Installation

1. Make sure you have [git](https://git-scm.com/install/) installed.
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) so that package versions match the ones used in this repo.
3. Navigate to your preferred destination folder and clone this repository:
   ```bash
   git clone https://github.com/ev-hansen/Configurable-Spectrograms.git
   ```
4. From the cloned directory, create the environment and install dependencies:
   ```bash
   uv venv --python 3.14.6
   uv pip install -r requirements.in
   ```
5. You're now done installing, and can run the Python files described below.

If you ever need to reset the environment for debug reasons and you're on macOS or Linux, run:
```bash
sh reset_uv.sh
```

## Library (`src/configurable_spectrograms/`)

All reusable plotting/batch/download logic lives in the `configurable_spectrograms` package (installed automatically by `uv run`/`uv sync` via the src-layout in `pyproject.toml`), split by concern:

| Module(s) | What it does |
| --- | --- |
| `constants.py`, `logging_utils.py`, `process_utils.py` | Shared constants, buffered logging, and child-process cleanup |
| `cdf_utils.py`, `percentile_utils.py` | CDF file/orbit discovery and axis-extrema/percentile helpers |
| `cusp_marking.py` | The two cusp-boundary marker styles -- `"line"`, the double-line marker, and `"bracket"`, which spans the region instead |
| `plotting.py` | Single-output spectrogram rendering (`make_spectrogram`, `generic_plot_spectrogram_set`, `generic_plot_multirow_optional_zoom`) |
| `batch_runner.py`, `generic_batch.py` | The executor-agnostic batch scaffold and the generic (data-agnostic) batch plotting driver built on it |
| `download.py` | FAST CDF downloading from CDA Web -- single-day (`download_single_day_cdf`), single-year (`FAST_ESA_CDF_download`), and a thread-pool multi-year batch downloader (`download_cdf_files_threaded`) |
| `fast/` | FAST-instrument-specific single-output plotting (`fast/plotting.py`), per-orbit batch worker (`fast/process_orbit.py`), the directory-wide batch driver (`fast/batch_directory.py`), global-extrema computation (`fast/extrema.py`), and orbit/file discovery (`fast/orbit_discovery.py`) |

Each top-level script below supplies only its own event/instrument-specific constants and delegates everything else to this library.

## Scripts

### Downloading

| Script | What it does |
| --- | --- |
| `single_day_FAST_CDF_download.py` | Downloads one calendar day's FAST CDF files for the given instrument(s) and exits |
| `FAST_CDF_download.py` | Downloads a whole year of FAST CDF files, looping the single-day download above over every day of the year |

```bash
uv run single_day_FAST_CDF_download.py --date 2001-11-30 --instruments eeb ieb
uv run FAST_CDF_download.py --year 2001
```

Both accept `--instruments`, `--output_path`, and `--base_url`; run either with `--help` for the full list.

### Plotting

| Script | What it does |
| --- | --- |
| `single_plot_spectrogram.py` | Renders one generic spectrogram figure from one CDF file and exits -- no batch machinery, no progress JSON |
| `single_plot_FAST_spectrograms.py` | Renders one FAST CDF file's pitch-angle grid, or one orbit's multi-instrument grid, and exits |
| `batch_multi_plot_spectrogram.py` | Batch-plots a folder of CDF data as spectrograms; should work with FAST-formatted CDFs but is meant to stay flexible for other instruments too |
| `batch_multi_plot_FAST_spectrograms.py` | Batch-plots a whole folder of FAST ESA data as spectrograms -- pitch-angle-range and full-instrument grids, cusp-region markup, and consistent per-instrument axis scaling |

```bash
uv run single_plot_spectrogram.py --cdf-file path/to/file.cdf --output out.png
uv run single_plot_FAST_spectrograms.py --cdf-file path/to/file.cdf --output out.png
uv run single_plot_FAST_spectrograms.py --data-folder ./FAST_data --orbit 13312 --output out.png
```

The two single-output scripts take CLI arguments (`--help` for the full list); the two batch scripts are configured by editing the constants near the top of the file, then run directly:
```bash
uv run batch_multi_plot_FAST_spectrograms.py
```

### GUI

`GUI_batch_download_plot_FAST.py` is a PySide6 Material-Design interface with three pages -- Download (batch CDF downloading), Plot (batch spectrogram rendering), and Single Day (render one figure -- from a local CDF file, a local folder plus orbit number, individually-selected instrument files, or a direct CDA Web download by date -- without running a full batch):

```bash
uv run GUI_batch_download_plot_FAST.py
```

### Supporting files

- `FAST_Cusp_Indices.csv` **(required)** -- tab-separated indices for when FAST CDF files indicate FAST was in the auroral cusp region; currently only covers 2000 and 2001.
- `FAST CDF variables.txt` **(example)** -- lists the CDF variables and their shape for FAST CDF files using an example orbit number. `time` dimensions (`epoch`, `time_unix`, `unix_time`, etc.) may be differently sized depending on orbit number; data dimensions are of shape `time` x (32 or 64) x 96.

## Documentation

API docs for the `configurable_spectrograms` library are built with Sphinx from the `docs/` directory:

```bash
cd docs
uv run python -m sphinx -M html . _build
```

The built HTML is written to `docs/_build/html`. Docs are also rebuilt and published automatically to GitHub Pages (and picked up by Read the Docs) on every push to `main`.
