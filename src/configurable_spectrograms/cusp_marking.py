"""Cusp-boundary markers drawn onto a spectrogram axis.

Two interchangeable styles are provided: the original double-line marker
(:func:`draw_cusp_line_markers`) and a bracket marker
(:func:`draw_cusp_bracket_marker`) that spans the cusp interval below the
axis instead of drawing lines through the data.
"""


def draw_cusp_line_markers(axis_object, marker_positions_plot, **kwargs) -> list:
    """Draw a thick black line under a thinner red line at each marker position.

    This is the original cusp-boundary marker style: for each position in
    ``marker_positions_plot`` a 4-pixel-wide black line is drawn first,
    followed by a 2-pixel-wide red line on top, so the boundary remains
    visible against both light and dark spectrogram data.

    Parameters
    ----------
    axis_object : matplotlib.axes.Axes
        Axes to draw onto.
    marker_positions_plot : list of float
        X positions, already converted to the axes' plotting units, marking
        cusp boundaries.
    **kwargs
        Accepted and ignored, so callers can pass a single ``**style_kwargs``
        dict regardless of which marker style is selected.

    Returns
    -------
    list
        The matplotlib ``Line2D`` artists created (two per marker position).
    """
    artists = []
    for position in marker_positions_plot:
        artists.append(
            axis_object.axvline(position, color="black", linestyle="-", linewidth=4, alpha=1.0, zorder=10)
        )
        artists.append(
            axis_object.axvline(position, color="red", linestyle="-", linewidth=2, alpha=1.0, zorder=11)
        )
    return artists


def draw_cusp_bracket_marker(
    axis_object,
    marker_positions_plot,
    color: str = "black",
    bracket_y: float = -0.08,
    bracket_tick_height: float = 0.02,
    caption: str | None = None,
    caption_offset: float = 0.04,
    caption_fontsize: float | None = None,
    linewidth: float = 1.5,
) -> list:
    """Draw a bracket spanning the cusp interval below the axis.

    An alternative to :func:`draw_cusp_line_markers` that brackets the cusp
    region rather than drawing lines through the plotted data, using the
    axes' x-data / y-axes-fraction transform so the bracket sits at a fixed
    relative offset below the axis regardless of the data's y-range.

    When two or more marker positions are given, the bracket spans the
    interval ``(min(marker_positions_plot), max(marker_positions_plot))``.
    When exactly one position is given (no true interval to bracket), a
    single vertical tick is drawn at that position instead.

    Parameters
    ----------
    axis_object : matplotlib.axes.Axes
        Axes to draw onto.
    marker_positions_plot : list of float
        X positions, already converted to the axes' plotting units.
    color : str, default 'black'
        Line color.
    bracket_y : float, default -0.08
        Y position of the bracket's horizontal bar, in axes-fraction
        coordinates (negative values sit below the axis).
    bracket_tick_height : float, default 0.02
        Height of the vertical end-ticks, in axes-fraction coordinates.
    caption : str or None, optional
        Caption text centered below the bracket.
    caption_offset : float, default 0.04
        Additional axes-fraction offset below ``bracket_y`` for the caption.
    caption_fontsize : float or None, optional
        Caption font size; uses the matplotlib default when ``None``.
    linewidth : float, default 1.5
        Bracket line width.

    Returns
    -------
    list
        The matplotlib artists created: one ``Line2D``, plus a ``Text`` when
        ``caption`` is given. Empty when ``marker_positions_plot`` is empty.

    Notes
    -----
    The default offsets are deliberately small so the bracket clears the
    x-axis without colliding with the tick labels or x-axis label in this
    codebase's default figure sizes, and so stacked multi-row grids don't
    have one row's bracket overlap the row below it. A caption, or a larger
    ``bracket_y`` magnitude, may need extra bottom margin reserved by the
    caller (e.g. via ``fig.subplots_adjust`` /
    ``fig.tight_layout(rect=...)``) to avoid overlapping nearby text.
    """
    if not marker_positions_plot:
        return []
    transform = axis_object.get_xaxis_transform()
    artists = []
    if len(marker_positions_plot) == 1:
        position = marker_positions_plot[0]
        (line,) = axis_object.plot(
            [position, position],
            [0, bracket_y],
            color=color,
            linewidth=linewidth,
            transform=transform,
            clip_on=False,
        )
        caption_x = position
    else:
        start, end = min(marker_positions_plot), max(marker_positions_plot)
        bracket_top = bracket_y + bracket_tick_height
        (line,) = axis_object.plot(
            [start, start, end, end],
            [bracket_top, bracket_y, bracket_y, bracket_top],
            color=color,
            linewidth=linewidth,
            transform=transform,
            clip_on=False,
        )
        caption_x = 0.5 * (start + end)
    artists.append(line)
    if caption:
        text = axis_object.text(
            caption_x,
            bracket_y - caption_offset,
            caption,
            transform=transform,
            ha="center",
            va="top",
            fontsize=caption_fontsize,
            clip_on=False,
        )
        artists.append(text)
    return artists
