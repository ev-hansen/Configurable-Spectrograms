"""Whitespace-padded, column-aligned table formatting for batch-summary reports.

A plain delimiter-separated line (even with a wide delimiter like two tabs)
only lines up in a monospace viewer when every entry happens to be about the
same length -- with entries as different in width as an instrument name and
a bare count, that assumption breaks down into a jagged, hard-to-read mess.
This instead measures each column's widest entry (header included) and pads
every field out to that width, so columns stay visually aligned regardless
of content length.
"""


def format_aligned_table(
    header: list[str],
    rows: list[list],
    column_gap: str = "  ",
    blank_line_before_last_row: bool = False,
) -> str:
    """Format ``header`` + ``rows`` as a whitespace-padded, column-aligned
    table string (no trailing newline).

    Each cell is stringified with ``str()`` before measuring/padding, then
    right-justified within its column (header cells included) -- every
    column aligned the same way, rather than numeric columns reading one
    direction and text columns the other. The last column is left unpadded
    so lines don't carry trailing whitespace; its start still lines up row
    to row since every column before it is fixed-width.

    ``blank_line_before_last_row``, if set, inserts one blank line between
    the rest of the rows and the last row -- for a table whose last row is
    a grand-total summary, to set it visually apart from the per-item rows
    above it. The blank line doesn't affect column widths, and the total
    row itself is still measured/padded together with every other row
    (its own content, e.g. a wide "TOTAL" label, can be what makes a
    column wider than any data row needs).

    Parameters
    ----------
    header : list of str
    rows : list of list
        Row values, any type -- stringified via ``str()``.
    column_gap : str, default '  '
        Separator written between adjacent columns.
    blank_line_before_last_row : bool, default False

    Returns
    -------
    str

    Examples
    --------
    >>> print(format_aligned_table(["Status", "Count"], [["ok", 12], ["error", 3]]))
    Status  Count
        ok  12
     error  3
    """
    str_rows: list[list[str]] = [[str(value) for value in row] for row in rows]

    widths: list[int] = [len(title) for title in header]
    for row in str_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        cells: list[str] = [value.rjust(widths[index]) for index, value in enumerate(values[:-1])]
        if values:
            cells.append(values[-1])
        # rstrip: a blank last cell (e.g. a TOTAL row's unused trailing
        # column) would otherwise leave the line ending in the
        # column_gap itself, silently reintroducing the trailing
        # whitespace the unpadded-last-column design is meant to avoid.
        return column_gap.join(cells).rstrip()

    formatted_rows: list[str] = [format_row(row) for row in str_rows]
    if blank_line_before_last_row and formatted_rows:
        formatted_rows.insert(-1, "")

    return "\n".join([format_row(header), *formatted_rows])


def write_aligned_table(
    path: str,
    header: list[str],
    rows: list[list],
    column_gap: str = "  ",
    blank_line_before_last_row: bool = False,
    preamble: str | None = None,
) -> None:
    """Write :func:`format_aligned_table`'s output to *path* as a
    standalone, human-readable report.

    ``preamble``, if given, is written before the table (trailing
    newlines stripped, then a blank line) -- e.g. a sentence stating what
    batch run or configuration the table's rows summarize.
    """
    table = format_aligned_table(header, rows, column_gap, blank_line_before_last_row)
    with open(path, "w", newline="") as f:
        if preamble is not None:
            f.write(preamble.rstrip("\n") + "\n\n")
        f.write(table + "\n")
