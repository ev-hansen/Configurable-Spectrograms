Batching and Buffered Logging
=============================

Overview
--------
The FAST spectrogram batch utilities implement disk I/O reduction and resiliency
through configurable batching for:

* Global extrema JSON persistence
* Plot progress JSON persistence
* Buffered info logging

Key Parameters
--------------
``flush_batch_size`` (default: 10)
    Controls how many orbits worth of successful extrema / progress updates are
    accumulated in-memory before writing JSON changes to disk. A final flush
    always occurs at normal program termination to avoid data loss.

``log_flush_batch_size`` (default: ``flush_batch_size``)
    Controls how many log records are buffered before being written. If
    unspecified, logging reuses ``flush_batch_size``. A forced flush occurs at
    shutdown.

Design Guarantees
-----------------
* At most ``flush_batch_size - 1`` orbit extrema updates are lost if the process
  is interrupted unexpectedly (e.g., SIGKILL, power loss).
* Progress JSON only writes when new orbit work was completed or on forced
  flush to minimize unnecessary disk churn.
* Buffered logging reduces filesystem sync frequency; logs are still flushed on
  completion or when the buffer reaches threshold.

Operational Notes
-----------------
* Setting ``flush_batch_size=1`` restores per-orbit persistence (maximum safety,
  higher I/O volume).
* Very large batch sizes increase risk of lost in-memory work if the process
  terminates abnormally.
* For long-running jobs, consider future enhancements like time-based flushes
  (not implemented yet) if extremely sparse orbit completion rates occur.

Error Categorization in Progress JSON
-------------------------------------
Per-instrument keys record error categories using the pattern:
``{instrument}_{y_scale}_{z_scale}_error-{reason}``

Common reasons include:
* ``timeout`` - worker exceeded allotted time
* ``invalid-cdf`` - structural or content issues in input CDF
* ``divide-by-zero`` - numerical domain error
* ``plotting`` - matplotlib/rendering failure
* ``generic`` - any uncategorized exception

Example Usage Snippet
---------------------
.. code-block:: python

    from batch_multi_plot_FAST_spectrograms import FAST_plot_spectrograms_directory

    FAST_plot_spectrograms_directory(
        input_dir="FAST_data",
        output_dir="FAST_plots",
        flush_batch_size=10,          # write extrema/progress every 10 orbits
        log_flush_batch_size=20,      # log buffer larger than JSON flush
    )

Tuning Recommendations
----------------------
* SSD / local FS: batch size 20-50 often reduces overhead further.
* Network / cloud drive: keep batch smaller (5-15) to limit at-risk data.
* Debug sessions: use ``flush_batch_size=1`` plus lower log batch for immediacy.
* Embedded / IDE environments: if keyboard shortcuts appear impaired, run
  with ``install_signal_handlers=False`` (generic) or rely on default behavior
  in FAST (handlers are now restored automatically after completion).

Related API Docstrings
----------------------
Full parameter and behavior details are in the docstrings for:
* ``compute_global_extrema``
* ``FAST_plot_spectrograms_directory``
* ``FAST_plot_instrument_grid`` (per-instrument overrides)

Per-Instrument / Per-Row Overrides (FAST & Generic)
---------------------------------------------------
Both the FAST-specific and generic batch plotting pipelines accept **row-level
axis overrides** via dataset dictionaries. For FAST multi-instrument grids this
enables distinct energy (``y_min``/``y_max``) and intensity (``z_min``/``z_max``)
ranges for ``ees``, ``eeb``, ``ies``, and ``ieb`` so that one instrument with a
high dynamic range does not wash out contrast for another.

Example applying precomputed extrema per instrument:

.. code-block:: python

  from batch_multi_plot_FAST_spectrograms import (
    compute_global_extrema, FAST_plot_spectrograms_directory
  )

  extrema = compute_global_extrema(
    directory_path="FAST_data",
    y_scale="linear",
    z_scale="log",
    flush_batch_size=20,
  )

  # Each row in multi-instrument figures now receives its own limits
  FAST_plot_spectrograms_directory(
    directory_path="FAST_data",
    y_scale="linear",
    z_scale="log",
    flush_batch_size=20,
    log_flush_batch_size=20,
  )

In addition, pitch-angle category grids now attach per-row energy and color
scale bounds (when provided) so categories with narrower relevant ranges retain
visual detail.

API Examples
------------

Minimal Plotting Run
~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

  from batch_multi_plot_FAST_spectrograms import FAST_plot_spectrograms_directory

  # Process all orbits with default batching (10) and logging
  FAST_plot_spectrograms_directory(
    input_dir="FAST_data",      # path containing CDF files
    output_dir="FAST_plots",    # destination for generated figures
  )

Custom Batching and Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

  FAST_plot_spectrograms_directory(
    input_dir="FAST_data",
    output_dir="FAST_plots",
    y_scale="linear",
    z_scale="log",
    flush_batch_size=25,          # write extrema/progress every 25 orbits
    log_flush_batch_size=50,      # less frequent log writes
    max_workers=6,                # increase parallelism
    orbit_timeout_seconds=90,     # orbit-level timeout
    instrument_timeout_seconds=45,
  )

Resuming After Interruption
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If execution is interrupted, re-running with the same parameters resumes from the
last recorded orbit per scale combination (unless ``ignore_progress_json=True``):

.. code-block:: python

  FAST_plot_spectrograms_directory(
    input_dir="FAST_data",
    output_dir="FAST_plots",
    flush_batch_size=10,
  )  # continues where previous run left off

Forcing a Fresh Pass (ignoring prior progress):

.. code-block:: python

  FAST_plot_spectrograms_directory(
    input_dir="FAST_data",
    output_dir="FAST_plots",
    ignore_progress_json=True,  # disregard existing progress JSON
  )

Computing / Refreshing Extrema Only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can call the internal extrema routine directly to precompute ranges without plotting:

.. code-block:: python

  from batch_multi_plot_FAST_spectrograms import compute_global_extrema  # if re-exported
  # (If not publicly exported, call through the plotting module or duplicate logic.)
  # When calling directly you can still specify the intensity percentile as 'max_percentile'.
  # The energy (Y) coverage threshold remains fixed at 99% of cumulative positive samples.
  state = compute_global_extrema(
    directory_path="FAST_data",
    y_scale="linear",
    z_scale="log",
    instrument_order=("ees", "eeb", "ies", "ieb"),
    extrema_json_path="FAST_calculated_extrema.json",
    max_percentile=95.0,      # matches FAST_plot_spectrograms_directory default
    flush_batch_size=20,
  )
  print("Updated extrema keys:", [k for k in state.keys() if k.endswith("_z_max")])

Tight Loop / Immediate Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set ``flush_batch_size=1`` for maximum safety (higher I/O volume):

.. code-block:: python

  FAST_plot_spectrograms_directory(
    input_dir="FAST_data",
    output_dir="FAST_plots",
    flush_batch_size=1,       # per-orbit JSON writes
    log_flush_batch_size=1,   # immediate log emission
  )

High-Throughput Bulk Mode
~~~~~~~~~~~~~~~~~~~~~~~~~
On fast local SSDs you can increase both batch sizes to reduce writes further:

.. code-block:: python

  FAST_plot_spectrograms_directory(
    input_dir="FAST_data",
    output_dir="FAST_plots",
    flush_batch_size=50,
    log_flush_batch_size=50,
    max_workers=8,
  max_processing_percentile=97.5,  # use a different intensity percentile
  )

Axis Label Conventions
----------------------
Both FAST-specific and generic plotting pipelines default to labeling the
vertical (energy) axis as ``Energy (eV)`` and the colorbar as ``Counts``. These
defaults are injected into each dataset row (``y_label`` / ``z_label``) and can
be overridden at dataset construction time to reflect alternative units (e.g.,
``'Differential Energy Flux'``). Centralizing units in the dataset dictionaries
maintains separation between physical semantics and the generic rendering code.

Intensity Percentile vs Energy Coverage
---------------------------------------
``max_processing_percentile`` (FAST) controls the intensity percentile used for
color scale maxima during global extrema computation. Energy (Y-axis) coverage
remains fixed at 99% of cumulative positive samples and is not currently user
configurable. The generic module defers to per-row ``vmin``/``vmax`` or on-the-fly
percentile selection inside ``make_spectrogram`` when bounds are omitted.


