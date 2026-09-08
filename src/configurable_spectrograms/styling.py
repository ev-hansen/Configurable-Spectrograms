"""Shared matplotlib rcParams setup and benign third-party warning suppression.

Nothing here runs as an import-time side effect -- a caller opts in by
calling :func:`configure_matplotlib` (and/or :func:`quiet_cdflib_noise`)
once from its own ``main()``/module top level, the same way each of this
project's CLI entry-point scripts already opts into the non-interactive
``Agg`` backend explicitly rather than having it forced on every importer.
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt

from configurable_spectrograms.constants import TICK_LABEL_FONT_SIZE

DEFAULT_FONT: dict = {"family": "sans-serif", "size": TICK_LABEL_FONT_SIZE}
DEFAULT_DPI: int = 150
DEFAULT_AXES_LINEWIDTH: float = 1.5
DEFAULT_TICK_WIDTH: float = 1.5
DEFAULT_TICK_LENGTH: float = 6.0
DEFAULT_MINOR_TICK_WIDTH: float = 1.0
DEFAULT_MINOR_TICK_LENGTH: float = 3.0
DEFAULT_GRID: bool = False
DEFAULT_GRID_ALPHA: float = 0.3
DEFAULT_GRID_LINEWIDTH: float = 0.5
DEFAULT_GRID_COLOR: str = "gray"


def quiet_cdflib_noise() -> None:
    """Silence a benign cdflib ``DeprecationWarning`` about a NumPy
    generic timedelta unit that fires on every epoch conversion
    regardless of data quality -- it's cdflib's, not this project's, to
    fix, and drowns out real warnings when scanning many CDF files in a
    batch run.
    """
    warnings.filterwarnings(
        "ignore",
        message="The 'generic' unit for NumPy timedelta is deprecated",
        category=DeprecationWarning,
    )


def configure_matplotlib(
    font: dict | None = None,
    dpi: int = DEFAULT_DPI,
    usetex: bool = False,
    constrained_layout: bool = True,
    axes_linewidth: float = DEFAULT_AXES_LINEWIDTH,
    tick_width: float = DEFAULT_TICK_WIDTH,
    tick_length: float = DEFAULT_TICK_LENGTH,
    minor_tick_width: float = DEFAULT_MINOR_TICK_WIDTH,
    minor_tick_length: float = DEFAULT_MINOR_TICK_LENGTH,
    grid: bool = DEFAULT_GRID,
    grid_alpha: float = DEFAULT_GRID_ALPHA,
    grid_linewidth: float = DEFAULT_GRID_LINEWIDTH,
    grid_color: str = DEFAULT_GRID_COLOR,
) -> None:
    """Apply one consistent set of matplotlib rcParams -- bold axis/tick
    labels, uniform tick widths/lengths (major and minor), optional grid
    -- across every plot rendered after this call, instead of each
    plotting function setting its own tick/spine styling piecemeal.

    Parameters
    ----------
    font : dict or None, optional
        Passed to ``matplotlib.rc('font', **font)``. Defaults to
        :data:`DEFAULT_FONT`.
    dpi : int, optional
    usetex : bool, default False
        If True, render text via a real LaTeX install. Left off by
        default -- unlike the project this was ported from, nothing here
        assumes a LaTeX toolchain is present, and ``usetex=True`` fails
        hard (not gracefully) on a machine without one.
    constrained_layout : bool, optional
    axes_linewidth : float, optional
        Thickness of the axes box (all four spines).
    tick_width, tick_length : float, optional
        Major tick thickness/length (both axes).
    minor_tick_width, minor_tick_length : float, optional
        Minor tick thickness/length (both axes).
    grid : bool, optional
    grid_alpha, grid_linewidth, grid_color : optional
        Grid line styling, used only if ``grid=True``.
    """
    matplotlib.rc("font", **(font or DEFAULT_FONT))
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "figure.constrained_layout.use": constrained_layout,
            "figure.dpi": dpi,
            "axes.linewidth": axes_linewidth,
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.major.width": tick_width,
            "ytick.major.width": tick_width,
            "xtick.major.size": tick_length,
            "ytick.major.size": tick_length,
            "xtick.minor.width": minor_tick_width,
            "ytick.minor.width": minor_tick_width,
            "xtick.minor.size": minor_tick_length,
            "ytick.minor.size": minor_tick_length,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.grid": grid,
            "grid.alpha": grid_alpha,
            "grid.linewidth": grid_linewidth,
            "grid.color": grid_color,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
