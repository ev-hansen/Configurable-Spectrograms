Generic Batch Spectrogram Plotting
==================================

Overview
--------
The module ``batch_multi_plot_spectrogram.py`` provides a **data-agnostic** batch
pipeline for generating multi-row spectrogram figures with optional per-row
color scaling overrides, batched progress persistence, resumable execution, and
buffered logging. It mirrors the FAST-specific implementation while removing
mission-specific assumptions.

Key Public Functions
--------------------
``generic_batch_plot``
    High-level orchestration that iterates over item identifiers (any iterable)
    and invokes plotting for each data container while batching JSON progress writes
    and log flushing. Accepts ``install_signal_handlers`` to opt out of temporary
    SIGINT handler installation in embedded environments.
``generic_plot_spectrogram_set``
    Render a multi-row spectrogram grid for a single logical item (e.g., orbit),
    applying per-row y/z min/max overrides and optional percentile-based color scaling.
``make_spectrogram``
    Low-level helper to collapse a 3D data cube along one axis and render a single
    spectrogram panel with linear or logarithmic y and z scaling.

Batching Parameters
-------------------
``flush_batch_size``
    Number of successfully processed items buffered before progress JSON (and any
    extrema-like state) is persisted. Final flush always occurs at normal termination.
``log_flush_batch_size``
    Size of the in-memory log message buffer; defaults to ``flush_batch_size`` when
    unspecified.

Progress JSON Schema (Generic)
------------------------------
The progress JSON managed by ``generic_batch_plot`` contains keys (illustrative)::

    {
        "completed_items": ["itemA", "itemB", ...],
        "errors": ["itemC"],
        "no_data": ["itemD"],
        "last_index": 42,
        "schema_version": 1
    }

Notes:
    * ``errors`` is a list of item representations; detailed exception text lives
        only in the log file to keep resumable state compact and schema-stable.
    * ``last_index`` is the zero-based index in the original ``items`` iterable
        of the most recently processed entry (useful for sanity checks / debugging).

Resumability is supported by reloading this file unless ``ignore_progress_json=True``.

Examples
--------
Minimal Run
~~~~~~~~~~~
.. code-block:: python

    from batch_multi_plot_spectrogram import generic_batch_plot

    # Suppose we have a pre-built list of items and a loader that yields datasets
    items = ["orbit_001", "orbit_002", "orbit_003"]

    def load_datasets_for_item(item_id):
        # Return list[dict] each with keys: x, y, data, label (and optional vmin/vmax)
        return build_datasets(item_id)  # user-defined

    generic_batch_plot(
        items=items,
        output_dir="plots_generic",
        build_datasets_fn=load_datasets_for_item,
    )

Custom Batching and Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    generic_batch_plot(
        items=items,
        output_dir="plots_generic",
        build_datasets_fn=load_datasets_for_item,
        flush_batch_size=25,
        log_flush_batch_size=50,
    )

Resuming After Interruption
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    generic_batch_plot(
        items=items,
        output_dir="plots_generic",
        build_datasets_fn=load_datasets_for_item,
        progress_json_path="generic_progress.json",  # default
    )  # picks up where it left off

Force Fresh Processing (ignore prior progress):

.. code-block:: python

    generic_batch_plot(
        items=items,
        output_dir="plots_generic",
        build_datasets_fn=load_datasets_for_item,
        ignore_progress_json=True,
    )

Per-Row Min/Max Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~
Each dataset dict may include per-row overrides (all optional): ``y_min``, ``y_max``,
``z_min``, ``z_max`` plus optional precomputed color scale bounds ``vmin`` / ``vmax``.
When ``z_min`` / ``z_max`` are absent but ``vmin`` / ``vmax`` are present, those act
as the row's native color limits (still clamped by any global ``z_min``/``z_max``
arguments passed to ``generic_plot_spectrogram_set``).

.. code-block:: python

    def load_datasets_for_item(item_id):
        return [
            {"x": x1, "y": y1, "data": d1, "label": "EES", "y_min": 0, "y_max": 4000},
            {"x": x2, "y": y2, "data": d2, "label": "IES", "z_min": 1e-2, "z_max": 1e2},
        ]

    generic_batch_plot(items, "plots_generic", load_datasets_for_item)

Zoom Column and Event Lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provide ``vertical_lines`` (UNIX timestamps) and a ``zoom_duration_minutes`` to add an
optional right-hand zoom column when data exist inside the derived window:

.. code-block:: python

    generic_plot_spectrogram_set(
        datasets=my_datasets,
        vertical_lines=[t0, t1],
        zoom_duration_minutes=10,
        colormap="plasma",
        z_scale="log",
    )

Low-Level Single Panel
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from batch_multi_plot_spectrogram import make_spectrogram

    ax, x_used = make_spectrogram(
        x_axis_values=time_sec,
        y_axis_values=energy_ev,
        data_array_3d=data_cube,
        collapse_axis=1,
        y_axis_scale_function="log",
        z_axis_scale_function="log",
        instrument_label="EES",
        vertical_lines_unix=[event_time],
    )

Percentile-Based Color Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If ``z_axis_min`` or ``z_axis_max`` are omitted, internal logic selects robust
percentile-based bounds (see function docstring). Provide explicit bounds to
fully control the color scale. Precomputing and supplying ``vmin``/``vmax`` when
assembling datasets avoids recomputation in tight loops.

Error Handling
--------------
All exceptions are captured with descriptive variable names; errors are recorded in
``errors`` within the progress JSON while processing continues with remaining items.

Implementation Notes
--------------------
* Logging is buffered; call-level helpers flush automatically at batch boundaries.
* Figures are explicitly closed via ``close_all_axes_and_clear`` to control memory use.
* The module is intentionally stateless apart from in-memory log and progress batching.
* A temporary SIGINT handler is installed (and restored) when ``install_signal_handlers=True``.
    Disable this in notebooks / embedded shells if global handler changes cause side-effects.

Axis Label Defaults
-------------------
Unless overridden per dataset row, the following label defaults are applied:

* Y-axis: ``Energy (eV)``
* Colorbar (Z): ``Counts``

Provide ``y_label`` / ``z_label`` keys in each dataset dict to customize units
or semantics (e.g., ``'Differential Flux (eV cm^-2 s^-1 sr^-1)'``). Global
changes can be introduced at the dataset construction layer to keep the batch
orchestration code data-agnostic.

See Also
--------
* :doc:`batching_and_logging` for shared batching semantics.
* FAST-specific orchestration: ``FAST_plot_spectrograms_directory`` in the mission module.
