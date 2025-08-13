# Configurable-Spectrograms
An attempt to make easily configurable spectrograms that allow batch folder processing of arbitrary data.

View documentation on [readthedocs](https://configurable-spectrograms.readthedocs.io/en/latest/) or [github pages](https://ev-hansen.github.io/Configurable-Spectrograms/)

## Notable features
- batch processing over entire folders
- different colormaps for different y- and z- axis scale combinations
- ability to mark and plot zoom-ed in plots of the regions of interest, with configurable duration for said region
- configurable y and z axes
- progress saved mid-run in case scripts need to be interrupted
- example usage for [FAST](https://web.archive.org/web/20250813172018/https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1996-049A) ion and electron ESA data sourced from downloaded [CDF](https://web.archive.org/web/20250813173821/https://cdf.gsfc.nasa.gov/) files 

**TODO**: *Ensure tests are formatted correctly, upload script to find when FAST CDF files are in the auroral cusp and generate the ``FAST_Cusp_Indices.csv`` file, run said script on 2001 CDF files and update said ``.csv`` file, convert counts to flux for FAST plots, create script to sort (at least) FAST CDF data into the assumed folder layout of ``{FAST_CDF_DATA_FOLDER_PATH}/year/month`` think of a clever/fun/more memorable name*

# Files
## ``batch_multi_plot_spectrogram.py``
Python file providing functions to use to plot spectrograms in a configurable way

## ``batch_multi_plot_FAST_spectrograms.py``
Python script implementing ``batch_multi_plot_spectrogram.py`` for use with FAST CDF data
- Marked timestamps are when FAST is in the auroral cusp region
- plots spectrograms for pitch angle ranges as well as all instruments
- plots for the same "instrument" should be scaled the same in terms of y and z axes (e.g. all ies plots should have same min and max for energy and counts).

## ``FAST_CDF_download.py``
Script to download FAST CDF files directly from [NASA's CDAWeb](https://cdaweb.gsfc.nasa.gov/) without manually using the web interface. Currently configured for years 2000 and 2001.

## ``FAST_Cusp_Indices.csv``
CSV file (currently tab-seperated) containing indicies for when FAST CDF files indicate FAST was in the auroral cusp region

## ``FAST CDF variables.txt``
An example ``.txt`` file listing the CDF variables and their shape for FAST CDF files using an example orbit number. NOTE: ``time`` dimensions (``epoch``, ``time_unix``, ``unix_time``, etc) may be differently sized depending on orbit number. Additionally, data dimensions are of shape ``time`` x (either 32 or 64) x 96.