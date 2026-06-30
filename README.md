# Configurable-Spectrograms
An attempt to make easily configurable spectrograms that allow batch folder processing of arbitrary data.

View documentation for batch_multi_plot file functions on [readthedocs](https://configurable-spectrograms.readthedocs.io/en/latest/) or [github pages](https://ev-hansen.github.io/Configurable-Spectrograms/) (same contents)

## Notable features
- batch processing over entire folders
- different colormaps for different y- and z- axis scale combinations
- ability to mark and plot zoomed-in plots of the regions of interest, with configurable duration for said region
- configurable y and z axes
- progress saved mid-run in case scripts need to be interrupted
- example usage for [FAST](https://web.archive.org/web/20250813172018/https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1996-049A) ion and electron ESA data sourced from downloaded [CDF](https://web.archive.org/web/20250813173821/https://cdf.gsfc.nasa.gov/) files 

**TODO**: *Upload example spectrogram plots from FAST data, upload script to find when FAST CDF files are in the auroral cusp and generate the ``FAST_Cusp_Indices.csv`` file, run said script on 2001 CDF files and update said ``.csv`` file, convert counts to flux for FAST plots, create script to sort (at least) FAST CDF data into the assumed folder layout of ``{FAST_CDF_DATA_FOLDER_PATH}/year/month``, think of a clever/fun/more memorable name*

# Download and Installation
1) Make sure you have [git](https://git-scm.com/install/) installed
2) [Install uv](https://docs.astral.sh/uv/getting-started/installation/) so that package versions can match the ones used in this repo
3) Navigate to your preferred destination folder in a command line or terminal and clone this repository with ``git clone https://github.com/ev-hansen/Configurable-Spectrograms.git`` 
4) Run ``uv venv`` in the command line or terminal to create a virtual environment followed by ``uv pip install -r requirements.txt`` to install the required python packages
5) You are now done installing and can use the python files in the specified directory.


# Files
## General-use Library
- ``batch_multi_plot_spectrogram.py``
Python file providing functions to use to batch plot spectrograms in a configurable manner.
Imported into other python files with ``import batch_multi_plot_spectrogram.py``, assuming the file is in the same directory.

## FAST-specific Scripts
- ``batch_multi_plot_FAST_spectrograms.py``
Python script implementing ``batch_multi_plot_spectrogram.py`` for use with FAST EISA CDF data. Marked timestamps are when FAST is in the auroral cusp region, plots spectrograms for pitch angle ranges as well as all instruments, plots for the same "instrument" should be scaled the same in terms of y and z axes (e.g. all ies plots should have same min and max for energy and counts).
This file can be run with ``uv run batch_multi_plot_FAST_spectrograms.py`` along with arguments (or by modifying the file).

- ``FAST_CDF_download.py``
Script to download FAST EISA CDF files directly from [NASA's CDAWeb](https://cdaweb.gsfc.nasa.gov/) without manually using the web interface.
This file can be run with ``uv run FAST_CDF_download.py`` while in this directory. . Use `FAST_CDF_download.py --help` for more information.

- ``GUI_batch_download_plot_FAST.py``
A GUI file using pyside6 based on material design to assist with batch downloading FAST EISA CDF files and batch plotting the EISA data with spectrograms. Uses ``FAST_CDF_download.py`` and ``batch_multi_plot_FAST_spectrograms.py``.
This file can be run with ``uv run GUI_batch_download_plot_FAST.py`` while in this directory

### Misc Files for FAST Scripts
- [REQUIRED] ``FAST_Cusp_Indices.csv``
CSV file (currently tab-seperated) containing indicies for when FAST CDF files indicate FAST was in the auroral cusp region, currently only covers 2000 and 2001

- [EXAMPLE] ``FAST CDF variables.txt``
An example ``.txt`` file listing the CDF variables and their shape for FAST CDF files using an example orbit number. NOTE: ``time`` dimensions (``epoch``, ``time_unix``, ``unix_time``, etc) may be differently sized depending on orbit number. Additionally, data dimensions are of shape ``time`` x (either 32 or 64) x 96.