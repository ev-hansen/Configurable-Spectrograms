"""Global axis-extrema computation for FAST batch plotting.

:func:`compute_global_extrema` performs a resumable pass over instrument CDF
files to determine shared, sensible axis limits (so every orbit in a batch
run uses the same energy/intensity scale) before the main plotting pass
begins.
"""

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from tqdm import tqdm

from configurable_spectrograms.cdf_utils import load_fast_cdf_dataset
from configurable_spectrograms.fast.constants import FAST_COLLAPSE_FUNCTION, FAST_EXTREMA_JSON_PATH
from configurable_spectrograms.fast.orbit_discovery import discover_orbit_files
from configurable_spectrograms.logging_utils import log_exception
from configurable_spectrograms.percentile_utils import round_extrema


def _extrema_overrides(
    global_extrema: dict | None,
    inst: str,
    y_scale: str,
    z_scale: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract and round per-instrument axis limits from a global extrema dict.

    Parameters
    ----------
    global_extrema : dict or None
        Mapping produced by :func:`compute_global_extrema`, or ``None``.
    inst : str
        Instrument code (e.g. ``'ees'``).
    y_scale, z_scale : str
        Axis-scale labels used to build the lookup key prefix.

    Returns
    -------
    tuple of (float or None)
        ``(y_min, y_max, z_min, z_max)`` with rounded values when keys are
        present in *global_extrema*, or ``(None, None, None, None)``
        otherwise.

    Examples
    --------
    >>> extrema = {"ees_linear_linear_y_max": 1234, "ees_linear_linear_z_min": 0.0123}
    >>> _extrema_overrides(extrema, "ees", "linear", "linear")
    (None, 1300.0, 0.012, None)
    >>> _extrema_overrides(None, "ees", "linear", "linear")
    (None, None, None, None)
    """
    if not isinstance(global_extrema, dict):
        return None, None, None, None
    key_prefix = f"{inst}_{y_scale}_{z_scale}"

    def _rounded(value: float | None, direction: str) -> float | None:
        return round_extrema(value, direction) if value is not None else None

    return (
        _rounded(global_extrema.get(f"{key_prefix}_y_min"), "down"),
        _rounded(global_extrema.get(f"{key_prefix}_y_max"), "up"),
        _rounded(global_extrema.get(f"{key_prefix}_z_min"), "down"),
        _rounded(global_extrema.get(f"{key_prefix}_z_max"), "up"),
    )


def compute_global_extrema(
    directory_path: str,
    y_scale: str,
    z_scale: str,
    instrument_order: Iterable[str],
    extrema_json_path: str = FAST_EXTREMA_JSON_PATH,
    compute_mins: bool = False,
    max_percentile: float = 95.0,
    log_floor_cutoff: float = 0.1,
    log_floor_value: float = -1.0,
    flush_batch_size: int = 10,
) -> dict[str, Any]:
    """Compute (or incrementally update) cached axis extrema per instrument.

    Performs a resumable pass over all instrument CDF files, flushing
    incremental progress to ``extrema_json_path`` after each
    ``flush_batch_size`` orbits.

    Extrema logic
    -------------
    - Y (energy) minima are fixed to 0 unless ``compute_mins`` is True.
    - Linear Y maxima: smallest energy whose cumulative positive finite
      count reaches 99% of total positive finite samples.
    - Linear Z maxima: ``max_percentile``-th percentile of pooled positive
      finite intensity samples.
    - If the requested scale is log and linear_linear extrema already exist
      in the cache, they are log-transformed without re-scanning files. If
      the requested scale is linear and linear_linear extrema exist, they
      are copied directly.
    - Log transform applies a floor: values ``<= log_floor_cutoff`` or
      non-finite are replaced by ``log_floor_value``.
    - Maxima are monotonically non-decreasing across incremental updates;
      energy maxima are capped at 4000.

    Parameters
    ----------
    directory_path : str
        Root directory containing instrument CDF files.
    y_scale : {'linear', 'log'}
        Y scaling label (used for cache key names).
    z_scale : {'linear', 'log'}
        Z scaling label (used for cache key names).
    instrument_order : iterable of str
        Instruments to process (e.g., ``("ees", "eeb", "ies", "ieb")``).
    extrema_json_path : str, default FAST_EXTREMA_JSON_PATH
        Path to the JSON cache file (created if absent).
    compute_mins : bool, default False
        If True, compute intensity minima; otherwise they are set to 0.
    max_percentile : float, default 95.0
        Percentile applied to pooled positive intensity for ``z_max``.
    log_floor_cutoff : float, default 0.1
        Values at or below this threshold map to ``log_floor_value`` in log
        space.
    log_floor_value : float, default -1.0
        Floor value substituted for invalid log-domain extrema.
    flush_batch_size : int, default 10
        Orbits with updates between JSON flushes; coerced to >= 1.

    Returns
    -------
    dict
        Updated extrema mapping containing values and progress entries.
    """
    instrument_order = tuple(instrument_order)
    if os.path.exists(extrema_json_path):
        try:
            with open(extrema_json_path) as file_in:
                extrema_state: dict[str, Any] = json.load(file_in)
        except (OSError, json.JSONDecodeError) as exc:
            log_exception(
                f"[EXTREMA] Failed to read existing extrema JSON '{extrema_json_path}' (starting fresh)",
                exc,
                level="message",
            )
            extrema_state = {}
    else:
        extrema_state = {}

    def _safe_log_transform(linear_value: float | int | None) -> float:
        """Convert a linear-domain value to log10 with floor handling."""
        if linear_value is None:
            return float(log_floor_value)
        try:
            value = float(linear_value)
        except (TypeError, ValueError):
            return float(log_floor_value)
        if not np.isfinite(value) or value <= log_floor_cutoff:
            return float(log_floor_value)
        return float(np.log10(value))

    orbit_to_instruments = discover_orbit_files(directory_path, instrument_order)
    sorted_orbit_numbers = sorted(orbit_to_instruments.keys())

    energy_positive_counts_by_instrument: dict[str, dict[float, int]] = {
        inst: defaultdict(int) for inst in instrument_order
    }
    positive_sample_arrays_by_instrument: dict[str, list[np.ndarray]] = {inst: [] for inst in instrument_order}
    total_files_per_instrument: dict[str, int] = {
        inst: sum(1 for orb in sorted_orbit_numbers if inst in orbit_to_instruments[orb])
        for inst in instrument_order
    }

    total_discovered_files = sum(total_files_per_instrument.values())
    extrema_progress_bar = tqdm(
        total=total_discovered_files,
        desc=f"Extrema {y_scale}/{z_scale}",
        unit="file",
        leave=False,
        disable=(total_discovered_files == 0),
    )

    try:
        orbits_since_last_flush = 0
        last_orbit_global_key = f"{y_scale}_{z_scale}_last_orbit"
        last_processed_orbit_val = extrema_state.get(last_orbit_global_key, -1)
        last_processed_orbit = (
            int(last_processed_orbit_val) if isinstance(last_processed_orbit_val, (int, float)) else -1
        )

        for orbit_index, orbit_number in enumerate(sorted_orbit_numbers):
            if orbit_number <= last_processed_orbit:
                continue
            for instrument_name in instrument_order:
                key_prefix = f"{instrument_name}_{y_scale}_{z_scale}"
                progress_key = f"{key_prefix}_extrema_progress"
                progress_entry = extrema_state.get(progress_key)
                if isinstance(progress_entry, dict) and progress_entry.get("complete"):
                    continue

                y_is_log = y_scale == "log"
                z_is_log = z_scale == "log"
                ll_y_key = f"{instrument_name}_linear_linear_y_max"
                ll_z_key = f"{instrument_name}_linear_linear_z_max"
                ll_y_min_key = f"{instrument_name}_linear_linear_y_min"
                ll_z_min_key = f"{instrument_name}_linear_linear_z_min"

                if not y_is_log and ll_y_key in extrema_state:
                    extrema_state[f"{key_prefix}_y_max"] = extrema_state[ll_y_key]
                    extrema_state[f"{key_prefix}_y_min"] = extrema_state.get(ll_y_min_key, 0)
                elif y_is_log and ll_y_key in extrema_state:
                    extrema_state[f"{key_prefix}_y_max"] = _safe_log_transform(extrema_state[ll_y_key])
                    extrema_state[f"{key_prefix}_y_min"] = log_floor_value

                if not z_is_log and ll_z_key in extrema_state:
                    extrema_state[f"{key_prefix}_z_max"] = extrema_state[ll_z_key]
                    extrema_state[f"{key_prefix}_z_min"] = extrema_state.get(ll_z_min_key, 0)
                elif z_is_log and ll_z_key in extrema_state:
                    extrema_state[f"{key_prefix}_z_max"] = _safe_log_transform(extrema_state[ll_z_key])
                    extrema_state[f"{key_prefix}_z_min"] = log_floor_value

                y_done = ll_y_key in extrema_state
                z_done = ll_z_key in extrema_state
                if y_done and z_done:
                    total_for_inst = total_files_per_instrument[instrument_name]
                    extrema_state[progress_key] = {
                        "processed_index": max(total_for_inst - 1, -1),
                        "total": total_for_inst,
                        "complete": True,
                    }
                    for inst in instrument_order:
                        extrema_state.pop(f"{inst}_{y_scale}_{z_scale}_last_orbit", None)
                    extrema_state[last_orbit_global_key] = max(sorted_orbit_numbers) if sorted_orbit_numbers else -1
                    try:
                        with open(extrema_json_path, "w") as file_out:
                            json.dump(extrema_state, file_out, indent=2)
                    except OSError as exc:
                        log_exception(
                            f"[EXTREMA] Failed to save extrema JSON after reuse for instrument={instrument_name}",
                            exc,
                            level="message",
                        )
                    continue

                energy_counts_map = energy_positive_counts_by_instrument[instrument_name]
                positive_blocks = positive_sample_arrays_by_instrument[instrument_name]

                cdf_path = orbit_to_instruments.get(orbit_number, {}).get(instrument_name)
                if cdf_path is not None:
                    try:
                        cdf_dataset = load_fast_cdf_dataset(cdf_path)
                    except Exception as exc:
                        log_exception(
                            f"[EXTREMA] Ingest failure inst={instrument_name} orbit={orbit_number} file={cdf_path}",
                            exc,
                            level="message",
                        )
                    else:
                        collapsed = FAST_COLLAPSE_FUNCTION(cdf_dataset["data"], axis=1)
                        finite_positive_mask = np.isfinite(collapsed) & (collapsed > 0)
                        counts_per_bin = finite_positive_mask.sum(axis=0)
                        for energy_value, count in zip(cdf_dataset["energy"], counts_per_bin, strict=False):
                            if count:
                                energy_counts_map[float(energy_value)] += int(count)
                        positive_values = collapsed[finite_positive_mask]
                        if positive_values.size:
                            positive_blocks.append(positive_values)
                    extrema_progress_bar.update(1)

                try:
                    candidate_energy_max = 0.0
                    if energy_counts_map:
                        sorted_energies = sorted(energy_counts_map.keys())
                        counts_arr = np.array([energy_counts_map[e] for e in sorted_energies])
                        cumulative = np.cumsum(counts_arr)
                        target = 0.99 * cumulative[-1]
                        idx = min(np.searchsorted(cumulative, target, side="right"), len(sorted_energies) - 1)
                        candidate_energy_max = float(sorted_energies[idx])

                    candidate_intensity_max = 0.0
                    if positive_blocks:
                        aggregated = np.concatenate(positive_blocks)
                        finite_pos = aggregated[np.isfinite(aggregated) & (aggregated > 0)]
                        if finite_pos.size:
                            candidate_intensity_max = float(np.nanpercentile(finite_pos, max_percentile))

                    prev_e = extrema_state.get(f"{key_prefix}_y_max")
                    prev_z = extrema_state.get(f"{key_prefix}_z_max")
                    merged_e = (
                        max(float(prev_e), candidate_energy_max)
                        if isinstance(prev_e, (int, float))
                        else candidate_energy_max
                    )
                    merged_z = (
                        max(float(prev_z), candidate_intensity_max)
                        if isinstance(prev_z, (int, float))
                        else candidate_intensity_max
                    )
                    merged_e = int(min(4000, math.ceil(merged_e)))
                    merged_z = float(math.ceil(merged_z))

                    if compute_mins and positive_blocks:
                        aggregated = np.concatenate(positive_blocks)
                        finite_pos = aggregated[np.isfinite(aggregated) & (aggregated > 0)]
                        intensity_min_store = float(np.nanpercentile(finite_pos, 1)) if finite_pos.size else 0.0
                        energy_min_store = 0
                    else:
                        energy_min_store = 0
                        intensity_min_store = 0

                    extrema_state[f"{key_prefix}_y_min"] = energy_min_store
                    extrema_state[f"{key_prefix}_y_max"] = merged_e
                    extrema_state[f"{key_prefix}_z_min"] = intensity_min_store
                    extrema_state[f"{key_prefix}_z_max"] = merged_z
                    extrema_state[progress_key] = {
                        "processed_index": orbit_index,
                        "total": total_files_per_instrument[instrument_name],
                        "complete": orbit_index + 1 >= total_files_per_instrument[instrument_name],
                    }
                    for inst in instrument_order:
                        extrema_state.pop(f"{inst}_{y_scale}_{z_scale}_last_orbit", None)
                    extrema_state[last_orbit_global_key] = orbit_number

                    extrema_progress_bar.set_postfix(inst=instrument_name, orbit=orbit_number, refresh=False)

                except Exception as exc:
                    log_exception(
                        f"[EXTREMA] Update failure inst={instrument_name} orbit={orbit_number}",
                        exc,
                        level="message",
                    )

                orbits_since_last_flush += 1
                if orbits_since_last_flush >= flush_batch_size:
                    try:
                        with open(extrema_json_path, "w") as file_out:
                            json.dump(extrema_state, file_out, indent=2)
                        orbits_since_last_flush = 0
                    except OSError as exc:
                        log_exception(
                            f"[EXTREMA] Batched flush failure after orbit {orbit_number}",
                            exc,
                            level="message",
                        )

        if orbits_since_last_flush > 0:
            try:
                if last_orbit_global_key in extrema_state:
                    ordered = {last_orbit_global_key: extrema_state[last_orbit_global_key]}
                    ordered.update({k: v for k, v in extrema_state.items() if k != last_orbit_global_key})
                    with open(extrema_json_path, "w") as file_out:
                        json.dump(ordered, file_out, indent=2)
                else:
                    with open(extrema_json_path, "w") as file_out:
                        json.dump(extrema_state, file_out, indent=2)
            except OSError as exc:
                log_exception("[EXTREMA] Final batched flush failure", exc, level="message")

    finally:
        extrema_progress_bar.close()

    if last_orbit_global_key in extrema_state:
        ordered = {last_orbit_global_key: extrema_state[last_orbit_global_key]}
        ordered.update({k: v for k, v in extrema_state.items() if k != last_orbit_global_key})
        return ordered
    return extrema_state
