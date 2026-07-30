"""Axis-extrema rounding and percentile-bound computation for color scales."""

import math

import numpy as np


def round_extrema(value: float | int, direction: str) -> float:
    """Round an extrema value to a clean significant-digit axis limit.

    Rounds to the next significant digit in the specified direction so plot
    axis limits look consistent (e.g. 1234 -> 1300 for 'up').

    Parameters
    ----------
    value : float or int
        Extrema value. Zero returns 0.0.
    direction : {'up', 'down'}
        Round up (for maxima) or down (for minima).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If direction is not ``'up'`` or ``'down'``.

    Examples
    --------
    >>> round_extrema(1234, 'up')
    1300.0
    >>> round_extrema(0.0123, 'down')
    0.012
    """
    if value == 0:
        return 0.0
    factor = 10 ** (math.floor(math.log10(abs(value))) - 1)
    if direction == "up":
        return float(math.ceil(value / factor) * factor)
    if direction == "down":
        return float(math.floor(value / factor) * factor)
    raise ValueError(f"Invalid direction: {direction}")


def compute_percentile_bounds(
    matrix: np.ndarray,
    low_percentile: float = 1,
    high_percentile: float = 99,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[float, float]:
    """Return ``(z_min, z_max)`` color-scale bounds for a data matrix.

    Explicit ``z_min``/``z_max`` values are used as-is when given; otherwise
    each bound is computed independently via ``numpy.nanpercentile``. This
    unifies the vmin/vmax percentile logic that plotting functions need when
    the caller hasn't supplied fixed bounds.

    Parameters
    ----------
    matrix : numpy.ndarray
        Data array (NaNs ignored).
    low_percentile : float, default 1
        Percentile used for the lower bound when ``z_min`` is ``None``.
    high_percentile : float, default 99
        Percentile used for the upper bound when ``z_max`` is ``None``.
    z_min : float or None, optional
        Explicit lower bound; overrides ``low_percentile`` when given.
    z_max : float or None, optional
        Explicit upper bound; overrides ``high_percentile`` when given.

    Returns
    -------
    tuple of float
        ``(z_min, z_max)``.

    Examples
    --------
    >>> import numpy as np
    >>> compute_percentile_bounds(np.array([[1.0, 2.0, 3.0, 100.0]]), 0, 100)
    (1.0, 100.0)
    >>> compute_percentile_bounds(np.array([1.0, 2.0, 3.0]), z_min=-5.0, z_max=5.0)
    (-5.0, 5.0)
    """
    resolved_min = float(z_min) if z_min is not None else float(np.nanpercentile(matrix, low_percentile))
    resolved_max = float(z_max) if z_max is not None else float(np.nanpercentile(matrix, high_percentile))
    return resolved_min, resolved_max
