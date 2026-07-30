# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
GUI for downloading and batch-plotting FAST ESA CDF data.

Provides a PySide6 Material-Design interface with three pages:

- **Download** — select instruments, years, and output folder; delegates to
  ``FAST_CDF_download.FAST_ESA_CDF_download`` via a background QThread.
- **Plot** — select data/output folders, axis scales, colormap, cusp marker
  style, and noise percentile; delegates to
  ``batch_multi_plot_FAST_spectrograms.FAST_plot_spectrograms_directory``
  via a background QThread.
- **Single Plot** — render one CDF file's pitch-angle grid, or one orbit's
  multi-instrument grid from a data folder, without running a full batch;
  delegates to ``configurable_spectrograms.fast.plotting`` via a background
  QThread.
"""

from __future__ import annotations

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"
__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
]
__date__: str = "2025-06-17"
__status__: str = "Development"
__version__: str = "0.0.1"
__license__: str = "GPL-3.0"

import datetime as dt
import functools
import multiprocessing
import os
import sys
from pathlib import Path

from PySide6.QtCore import QDate, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QIcon, QIntValidator, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qt_material import apply_stylesheet, QtStyleTools
from qt_material_icons import MaterialIcon

# PySide6 re-exposes each scoped enum member as a flat, backward-compatible
# alias on the enclosing class (e.g. Qt.AlignCenter for
# Qt.AlignmentFlag.AlignCenter). These aliases exist and work at runtime,
# but ty's PySide6 stubs don't model them, so uses of them below carry an
# unresolved-attribute suppression directive.

# --- Constants ---
MIN_YEAR, MAX_YEAR = 1996, 2009
INSTRUMENT_OPTIONS: tuple[str, ...] = ("eeb", "ees", "esv", "ieb", "ies")
SCALE_OPTIONS: tuple[str, ...] = ("linear", "log")
COLORMAP_OPTIONS: tuple[str, ...] = (
    "viridis",
    "cividis",
    "plasma",
    "inferno",
    "turbo",
)
CUSP_STYLE_OPTIONS: tuple[str, ...] = ("line", "bracket")
PLOT_MODE_OPTIONS: tuple[str, ...] = ("Single Instrument (pitch-angle grid)", "Full Orbit (instrument grid)")
#: Full Orbit mode's three ways of supplying its instrument CDF files.
DATA_SOURCE_OPTIONS: tuple[str, ...] = (
    "Data Folder + Orbit Number",
    "Select Files (one per instrument)",
    "Download by Date",
)
#: Instruments supported by FAST_plot_instrument_grid (matches DEFAULT_INSTRUMENT_ORDER).
FULL_ORBIT_INSTRUMENTS: tuple[str, ...] = ("eeb", "ees", "ieb", "ies")
DEFAULT_NOISE_PERCENTILE: float = 90.0
THEME_DARK, THEME_LIGHT = "dark_teal.xml", "light_purple.xml"
PRIMARY: str = os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#2196F3")
ERROR: str = "#d32f2f"


# --- Subprocess target functions (module-level for multiprocessing "spawn" pickling) ---
def _download_year_in_process(instruments: set, year: int, data_folder: str) -> None:
    """Run inside a child process; imports kept local to avoid spawn overhead."""
    from FAST_CDF_download import FAST_ESA_CDF_download

    FAST_ESA_CDF_download(instruments=list(instruments), year=year, data_folder=data_folder)


def _plot_in_process(
    directory_path: str,
    output_base: str,
    y_scale: str,
    z_scale: str,
    verbose: bool,
    use_tqdm: bool,
    colormap: str,
    max_processing_percentile: float,
    cusp_marker_style: str = "line",
) -> None:
    """Run inside a child process; imports kept local to avoid spawn overhead."""
    from batch_multi_plot_FAST_spectrograms import FAST_plot_spectrograms_directory

    FAST_plot_spectrograms_directory(
        directory_path=directory_path,
        output_base=output_base,
        y_scale=y_scale,
        z_scale=z_scale,
        verbose=verbose,
        use_tqdm=use_tqdm,
        colormap=colormap,
        max_processing_percentile=max_processing_percentile,
        cusp_marker_style=cusp_marker_style,
    )


def _single_plot_in_process(
    mode: str,
    cdf_file_path: str,
    data_source: str,
    orbit_folder_path: str,
    orbit_number: int | None,
    instrument_file_paths: dict[str, str],
    download_date: dt.date | None,
    download_instruments: set[str],
    output_path: str,
    y_scale: str,
    z_scale: str,
    colormap: str,
    cusp_marker_style: str,
) -> None:
    """Run inside a child process; imports kept local to avoid spawn overhead.

    Renders exactly one figure -- either a single CDF's pitch-angle grid
    (``mode == "single_instrument"``, orbit number auto-parsed from the
    filename) or one orbit's multi-instrument grid (``mode == "full_orbit"``)
    -- and saves it to *output_path*. Full Orbit mode resolves its instrument
    CDF files according to *data_source*: ``"folder"`` discovers them from
    *orbit_folder_path* and *orbit_number*; ``"files"`` uses the
    caller-supplied *instrument_file_paths* directly; ``"download"`` fetches
    *download_date* for *download_instruments* from CDA Web first.
    """
    from configurable_spectrograms.cdf_utils import load_filtered_orbits
    from configurable_spectrograms.download import download_single_day_cdf
    from configurable_spectrograms.fast.orbit_discovery import (
        discover_orbit_files,
        extract_orbit_and_instrument,
        resolve_orbit_from_files,
        resolve_shared_orbit,
    )
    from configurable_spectrograms.fast.plotting import FAST_plot_instrument_grid, FAST_plot_pitch_angle_grid

    filtered_orbits_df = load_filtered_orbits()
    if mode == "single_instrument":
        parsed = extract_orbit_and_instrument(cdf_file_path)
        resolved_orbit = parsed[0] if parsed is not None else None
        fig, _canvas = FAST_plot_pitch_angle_grid(
            cdf_file_path,
            filtered_orbits_df=filtered_orbits_df,
            orbit_number=resolved_orbit,
            scale_function_y=y_scale,
            scale_function_z=z_scale,
            show=False,
            colormap=colormap,
            cusp_marker_style=cusp_marker_style,
        )
    else:
        if data_source == "files":
            instrument_files = instrument_file_paths
            resolved_orbit = resolve_orbit_from_files(instrument_files)
        elif data_source == "download":
            assert download_date is not None
            day_files = download_single_day_cdf(date=download_date, instruments=list(download_instruments))
            resolved_orbit, instrument_files = resolve_shared_orbit(day_files)
        else:
            resolved_orbit = orbit_number
            instrument_files = discover_orbit_files(orbit_folder_path).get(orbit_number, {})
        if not instrument_files:
            raise RuntimeError("No instrument CDF files found for the given input.")
        fig, _canvas = FAST_plot_instrument_grid(
            instrument_files,
            filtered_orbits_df=filtered_orbits_df,
            orbit_number=resolved_orbit,
            scale_function_y=y_scale,
            scale_function_z=z_scale,
            show=False,
            colormap=colormap,
            cusp_marker_style=cusp_marker_style,
        )
    if fig is None:
        raise RuntimeError("No data available to plot for the given input.")
    fig.savefig(output_path, dpi=200)


# --- Supplemental stylesheet ---
# Custom tokens resolved by _make_extra_stylesheet():
#   %(BTN_TEXT_COLOR)s     — button icon/text (white dark, #3c3c3c light)
#   %(CHIP_BORDER_COLOR)s  — semi-transparent chip outline
#   %(CONTENT_TEXT_COLOR)s — body/input text readable on the current theme
_EXTRA_CSS: str = """
QWidget#sidebar { border-right: 1px solid %(QTMATERIAL_SECONDARYDARKCOLOR)s; padding: 0px; }
QWidget#root { padding: 0px; }

QPushButton#navBtn {
    background-color: transparent; border: none; border-radius: 16px;
    font-size: 11px; padding: 4px 0px; text-align: center;
    color: %(BTN_TEXT_COLOR)s;
}
QPushButton#navBtn:hover { background-color: %(QTMATERIAL_SECONDARYLIGHTCOLOR)s; }
QPushButton#navBtn[selected="true"] {
    background-color: %(QTMATERIAL_SECONDARYLIGHTCOLOR)s;
    color: %(QTMATERIAL_PRIMARYCOLOR)s; font-weight: 600;
}

QPushButton#chip {
    border: 1px solid %(CHIP_BORDER_COLOR)s; border-radius: 8px;
    font-size: 13px; font-weight: 500; padding: 6px 18px; min-width: 52px;
    color: %(BTN_TEXT_COLOR)s; text-transform: none;
}
QPushButton#chip[selected="true"] {
    background-color: %(QTMATERIAL_PRIMARYCOLOR)s;
    color: %(QTMATERIAL_PRIMARYTEXTCOLOR)s;
    border: 1px solid %(QTMATERIAL_PRIMARYCOLOR)s;
}

QPushButton#ctaBtn, QPushButton#folderBtn { color: %(BTN_TEXT_COLOR)s; }
QPushButton#ctaBtn:disabled, QPushButton#folderBtn:disabled { color: %(DISABLED_TEXT_COLOR)s; }

QLabel#folderPath   { font-size: 12px; font-style: italic; }
QLabel#noteText     { font-size: 11px; font-style: italic; }
QFrame#divider      { border: none; max-height: 1px; }
QLabel#pageTitle    { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
QLabel#sectionTitle { font-size: 15px; font-weight: 600; }
QLabel#bodyText     { font-size: 13px; }
QLabel#statusLabel  { font-size: 12px; font-weight: 500; }

QComboBox#styledCombo, QLineEdit#percentileEntry { color: %(CONTENT_TEXT_COLOR)s; }
/* QAbstractItemView popup is a top-level window; descendant selector won't match it */
QAbstractItemView { color: %(CONTENT_TEXT_COLOR)s; }

QScrollBar:vertical { background: %(SCROLLBAR_TRACK_COLOR)s; width: 8px; margin: 0px; }
/* Override qt_material's opacity(0.1) default — one rule per line; Qt QSS drops all but
   the last selector in a comma group when sub-controls are involved */
QScrollBar::handle           { background: %(QTMATERIAL_PRIMARYCOLOR)s; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical  { background: %(QTMATERIAL_PRIMARYCOLOR)s; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:horizontal { background: %(QTMATERIAL_PRIMARYCOLOR)s; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover   { background: %(QTMATERIAL_PRIMARYCOLOR)s; }
QScrollBar::handle:horizontal:hover { background: %(QTMATERIAL_PRIMARYCOLOR)s; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


# --- Theme helpers ---
def _system_is_dark() -> bool:
    """Return ``True`` when the OS colour scheme is dark.

    Prefers the Qt 6.5+ ``styleHints().colorScheme()`` API; falls back to
    measuring the luminance of the window background colour on older builds.
    """
    hints = QApplication.styleHints()
    if hasattr(hints, "colorScheme"):
        return hints.colorScheme() == Qt.ColorScheme.Dark
    palette = QApplication.palette()
    return palette.color(palette.Window).lightnessF() < 0.5  # ty: ignore[unresolved-attribute]


def _make_extra_stylesheet(is_dark: bool) -> str:
    """Substitute dynamic colour tokens into ``_EXTRA_CSS`` for the current theme.

    Uses an explicit token dict rather than spreading all of ``os.environ`` to
    avoid collisions with arbitrary environment variable values.

    Parameters
    ----------
    is_dark : bool
        ``True`` for the dark theme variant; ``False`` for light.

    Returns
    -------
    str
        Resolved CSS string ready to append to the application stylesheet.
    """
    # light_purple secondaryDarkColor is #e6e6e6 (near-white), so pin an
    # explicit foreground rather than relying on the theme value.
    fg = "#ffffff" if is_dark else "#3c3c3c"
    return _EXTRA_CSS % {
        "QTMATERIAL_SECONDARYDARKCOLOR": os.environ.get("QTMATERIAL_SECONDARYDARKCOLOR", "#37474f"),
        "QTMATERIAL_SECONDARYLIGHTCOLOR": os.environ.get("QTMATERIAL_SECONDARYLIGHTCOLOR", "#cfd8dc"),
        "QTMATERIAL_PRIMARYCOLOR": os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#2196F3"),
        "QTMATERIAL_PRIMARYTEXTCOLOR": os.environ.get("QTMATERIAL_PRIMARYTEXTCOLOR", "#ffffff"),
        "CONTENT_TEXT_COLOR": fg,
        "BTN_TEXT_COLOR": fg,
        "DISABLED_TEXT_COLOR": "rgba(255,255,255,0.38)" if is_dark else "#9e9e9e",
        "CHIP_BORDER_COLOR": ("rgba(255,255,255,0.45)" if is_dark else "rgba(0,0,0,0.28)"),
        "SCROLLBAR_TRACK_COLOR": ("rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.10)"),
    }


def _apply_app_theme(app: QApplication, is_dark: bool) -> None:
    """Apply qt_material theme, supplemental CSS, and Roboto font to *app*.

    Parameters
    ----------
    app : QApplication
        The running application instance.
    is_dark : bool
        ``True`` selects dark_teal; ``False`` selects light_purple.
    """
    apply_stylesheet(app, theme=THEME_DARK if is_dark else THEME_LIGHT, extra={})
    app.setStyleSheet(app.styleSheet() + _make_extra_stylesheet(is_dark))
    font = QFont("Roboto", 10)
    font.setHintingPreference(QFont.PreferFullHinting)  # ty: ignore[unresolved-attribute]
    app.setFont(font)


def _colored_pixmap(icon: QIcon, size: int, hex_color: str) -> QPixmap:
    """Return a copy of *icon* recoloured to *hex_color* at *size* × *size* px.

    Copies the icon pixmap (preserving its device pixel ratio) then applies
    ``SourceIn`` composition to replace every pixel's RGB with *hex_color*
    while keeping the original alpha channel (icon shape).

    Parameters
    ----------
    icon : QIcon
        Source icon whose shape (alpha channel) is used as the mask.
    size : int
        Width and height of the output pixmap in pixels.
    hex_color : str
        Target colour as a hex string (e.g. ``'#ffffff'``).

    Returns
    -------
    QPixmap
        Recoloured pixmap with premultiplied alpha matching the source shape.
    """
    result = icon.pixmap(size, size).copy()  # .copy() preserves devicePixelRatio
    painter = QPainter(result)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(result.rect(), QColor(hex_color))
    painter.end()
    return result


# --- UI builder helpers ---
def _make_scroll_page(parent: QWidget) -> tuple[QVBoxLayout, QScrollArea]:
    """Wrap *parent* in a frameless, horizontally-locked scroll area.

    Returns the inner ``QVBoxLayout`` and the ``QScrollArea`` so callers can
    set widget-level scrollbar CSS that survives Qt style re-polish events.
    """
    outer = QVBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)  # ty: ignore[unresolved-attribute]
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # ty: ignore[unresolved-attribute]
    container = QWidget()
    scroll.setWidget(container)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(40, 32, 40, 32)
    layout.setSpacing(0)
    outer.addWidget(scroll)
    return layout, scroll


def _add_divider(layout: QVBoxLayout, before: int = 28, after: int = 24) -> None:
    """Insert a horizontal rule into *layout* with optional surrounding spacing.

    Parameters
    ----------
    layout : QVBoxLayout
        Target layout.
    before : int, default 28
        Pixels above the divider; 0 to suppress.
    after : int, default 24
        Pixels below the divider; 0 to suppress.
    """
    if before:
        layout.addSpacing(before)
    divider = QFrame()
    divider.setObjectName("divider")
    divider.setFrameShape(QFrame.HLine)  # ty: ignore[unresolved-attribute]
    layout.addWidget(divider)
    if after:
        layout.addSpacing(after)


def _section_label(
    layout: QVBoxLayout,
    text: str,
    spacing: int = 10,
    point_size: int | None = None,
) -> QLabel:
    """Add a ``sectionTitle``-styled label to *layout* and return it.

    Parameters
    ----------
    layout : QVBoxLayout
        Target layout.
    text : str
        Label text.
    spacing : int, default 10
        Pixels added below the label.
    point_size : int or None, optional
        Font point-size override; inherits the stylesheet default when ``None``.

    Returns
    -------
    QLabel
        The created label widget.
    """
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    if point_size is not None:
        font = lbl.font()
        font.setPointSize(point_size)
        lbl.setFont(font)
    layout.addWidget(lbl)
    layout.addSpacing(spacing)
    return lbl


def _page_header(layout: QVBoxLayout, title: str, body: str) -> None:
    """Insert a large page title and word-wrapped description into *layout*.

    Parameters
    ----------
    layout : QVBoxLayout
        Target layout.
    title : str
        Large page title (``pageTitle`` style).
    body : str
        Paragraph placed under a "Description" sub-heading.
    """
    title_lbl = QLabel(title)
    title_lbl.setObjectName("pageTitle")
    layout.addWidget(title_lbl)
    layout.addSpacing(20)
    _section_label(layout, "Description", spacing=8)
    body_lbl = QLabel(body)
    body_lbl.setObjectName("bodyText")
    body_lbl.setWordWrap(True)
    layout.addWidget(body_lbl)


def _folder_selector(
    layout: QVBoxLayout,
    click_fn,
    *,
    note: str | None = None,
) -> tuple[QPushButton, QLabel]:
    """Add a folder-selector button row to *layout* and return its widgets.

    An optional italicised hint note is prepended above the button. The
    returned path label is updated by the caller after a folder is chosen.

    Parameters
    ----------
    layout : QVBoxLayout
        Target layout.
    click_fn : callable
        Slot invoked when the "Select Folder" button is clicked.
    note : str or None, optional
        Hint text rendered above the button row.

    Returns
    -------
    tuple[QPushButton, QLabel]
        The folder button and the path display label.
    """
    if note:
        note_lbl = QLabel(note)
        note_lbl.setObjectName("noteText")
        note_lbl.setWordWrap(True)
        layout.addWidget(note_lbl)
        layout.addSpacing(10)
    row = QHBoxLayout()
    row.setSpacing(12)
    row.setContentsMargins(0, 0, 0, 0)
    btn = QPushButton("  Select Folder")
    btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("folder_open"), 24, "#ffffff")))
    btn.setObjectName("folderBtn")
    btn.setFixedHeight(38)
    btn.clicked.connect(click_fn)
    row.addWidget(btn)
    row.addStretch()
    layout.addLayout(row)
    layout.addSpacing(6)
    path_lbl = QLabel("No folder selected")
    path_lbl.setObjectName("folderPath")
    layout.addWidget(path_lbl)
    return btn, path_lbl


def _add_toggle_section(
    layout: QVBoxLayout,
    title: str,
    check_label: str,
    note: str,
    initial: bool,
    slot,
) -> QCheckBox:
    """Add a titled checkbox with a hint note to *layout* and return it.

    Parameters
    ----------
    layout : QVBoxLayout
        Target layout.
    title : str
        Section-title text displayed above the checkbox.
    check_label : str
        Text displayed next to the checkbox.
    note : str
        Italicised hint rendered below the checkbox.
    initial : bool
        Initial checked state.
    slot : callable
        Connected to ``QCheckBox.toggled``; receives the new ``bool`` state.

    Returns
    -------
    QCheckBox
        The created checkbox widget.
    """
    _section_label(layout, title, spacing=8)
    cb = QCheckBox(check_label)
    cb.setObjectName("toggleCheck")
    cb.setChecked(initial)
    cb.toggled.connect(slot)
    layout.addWidget(cb)
    layout.addSpacing(4)
    note_lbl = QLabel(note)
    note_lbl.setObjectName("noteText")
    note_lbl.setWordWrap(True)
    layout.addWidget(note_lbl)
    return cb


def _make_combo(options: tuple[str, ...], default: str) -> QComboBox:
    """Return a ``styledCombo`` QComboBox pre-populated with *options* and *default* selected."""
    combo = QComboBox()
    combo.setObjectName("styledCombo")
    combo.addItems(options)
    combo.setCurrentText(default)
    return combo


def _percentile_slider_section(layout: QVBoxLayout, initial: float) -> tuple[QSlider, QLineEdit, QLabel]:
    """Add a linked noise-cutoff percentile slider + entry + hint note to *layout*.

    The slider (0-100) and the numeric entry stay in sync with each other;
    callers connect to ``slider.valueChanged`` / ``entry.editingFinished``
    themselves since the note text and stored value differ per page.

    Parameters
    ----------
    layout : QVBoxLayout
        Target layout.
    initial : float
        Initial percentile value for both the slider and the entry field.

    Returns
    -------
    tuple[QSlider, QLineEdit, QLabel]
        The slider, the numeric entry field, and the hint note label (whose
        text the caller should update via its own change handlers).
    """
    pct_row = QHBoxLayout()
    pct_row.setSpacing(14)
    pct_row.setContentsMargins(0, 0, 0, 0)
    slider = QSlider(Qt.Horizontal)  # ty: ignore[unresolved-attribute]
    slider.setObjectName("percentileSlider")
    slider.setRange(0, 100)
    slider.setValue(int(initial))
    slider.setFixedHeight(32)
    slider.setMaximumWidth(400)
    entry = QLineEdit(str(int(initial)))
    entry.setObjectName("percentileEntry")
    entry.setFixedHeight(36)
    entry.setMaximumWidth(72)
    entry.setAlignment(Qt.AlignCenter)  # ty: ignore[unresolved-attribute]
    pct_row.addWidget(slider)
    pct_row.addWidget(entry)
    pct_row.addStretch()
    layout.addLayout(pct_row)
    layout.addSpacing(4)
    note_lbl = QLabel()
    note_lbl.setObjectName("noteText")
    note_lbl.setWordWrap(True)
    layout.addWidget(note_lbl)
    return slider, entry, note_lbl


def _make_stop_btn(click_fn) -> QPushButton:
    """Return a pre-styled red stop button, initially hidden, connected to *click_fn*."""
    btn = QPushButton("  Stop")
    btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("stop"), 24, "#ffffff")))
    btn.setFixedHeight(48)
    btn.setVisible(False)
    btn.clicked.connect(click_fn)
    btn.setStyleSheet(
        "QPushButton { background-color: #c62828; color: #ffffff; border: none;"
        " border-radius: 4px; padding: 0 18px; font-size: 13px; font-weight: 500; }"
        " QPushButton:hover { background-color: #b71c1c; }"
        " QPushButton:pressed { background-color: #7f0000; }"
    )
    return btn


def _scrollbar_css(primary: str, track: str) -> str:
    """Return a QScrollBar widget stylesheet pinning handle colour to *primary*."""
    return (
        f"QScrollBar:vertical {{ background: {track}; width: 8px; margin: 0px; }}"
        f" QScrollBar::handle:vertical {{ background: {primary}; border-radius: 4px; min-height: 24px; }}"
        " QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        " QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }"
    )


def _truncated_path(path: str, max_len: int = 60) -> str:
    """Return *path* with a leading ellipsis when longer than *max_len* characters."""
    return path if len(path) < max_len else "…" + path[-(max_len - 3) :]


# --- Background worker threads ---
class _BaseWorker(QThread):
    """Shared base providing the four standard signals and ``request_stop``."""

    progress: Signal = Signal(str)
    # Deliberately redeclares QThread's own no-argument `finished` signal:
    # this one fires only on confirmed success, distinct from QThread's
    # built-in `finished` (which fires whenever run() returns, including
    # after an error or a requested stop).
    finished: Signal = Signal()  # ty: ignore[invalid-attribute-override]
    stopped: Signal = Signal()
    error: Signal = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._proc: multiprocessing.Process | None = None

    def request_stop(self) -> None:
        """Set the interruption flag and kill the active subprocess immediately."""
        self.requestInterruption()
        if (proc := self._proc) is not None and proc.is_alive():
            proc.kill()


class DownloadWorker(_BaseWorker):
    """Background thread that downloads FAST ESA CDF files via CDA Web.

    Parameters
    ----------
    given_instruments : set[str]
        Instrument codes to download (e.g. ``{'eeb', 'ees'}``).
    years : list[int]
        Calendar years to fetch.
    data_folder : str
        Root directory where downloaded files are saved.
    parent : QObject or None, optional
        Qt parent object.
    """

    def __init__(
        self,
        given_instruments: set[str],
        years: list[int],
        data_folder: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.given_instruments = given_instruments
        self.years = years
        self.data_folder = data_folder

    def run(self) -> None:
        """Execute the download loop; emits ``finished``, ``stopped``, or ``error``."""
        try:
            for year in self.years:
                if self.isInterruptionRequested():
                    self.stopped.emit()
                    return
                self.progress.emit(f"Downloading year {year}…")
                self._proc = multiprocessing.Process(
                    target=_download_year_in_process,
                    args=(self.given_instruments, year, self.data_folder),
                )
                self._proc.start()
                self._proc.join()
                exitcode, self._proc = self._proc.exitcode, None
                if self.isInterruptionRequested():
                    self.stopped.emit()
                    return
                if exitcode != 0:
                    self.error.emit(f"Download process exited with code {exitcode} for year {year}")
                    return
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class PlotWorker(_BaseWorker):
    """Background thread that generates FAST ESA spectrograms.

    Parameters
    ----------
    directory_path : str
        Root directory containing FAST CDF files.
    output_base : str
        Base directory for output PNG files.
    y_scale : {'linear', 'log'}
        Energy (y-axis) scaling.
    z_scale : {'linear', 'log'}
        Intensity (color) scaling.
    verbose : bool
        Whether to emit detailed log output.
    use_tqdm : bool
        Whether to show a tqdm progress bar in the console.
    colormap : str
        Matplotlib colormap name.
    max_processing_percentile : float
        Upper noise-cutoff percentile forwarded to the batch plotter.
    cusp_marker_style : str, default 'line'
        Cusp-boundary marker style forwarded to the batch plotter.
    parent : QObject or None, optional
        Qt parent object.
    """

    def __init__(
        self,
        directory_path: str,
        output_base: str,
        y_scale: str,
        z_scale: str,
        verbose: bool,
        use_tqdm: bool,
        colormap: str,
        max_processing_percentile: float,
        cusp_marker_style: str = "line",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.directory_path = directory_path
        self.output_base = output_base
        self.y_scale = y_scale
        self.z_scale = z_scale
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.colormap = colormap
        self.max_processing_percentile = max_processing_percentile
        self.cusp_marker_style = cusp_marker_style

    def run(self) -> None:
        """Execute the plotting pass; emits ``finished``, ``stopped``, or ``error``."""
        try:
            self.progress.emit("Generating spectrograms…")
            self._proc = multiprocessing.Process(
                target=_plot_in_process,
                kwargs=dict(
                    directory_path=self.directory_path,
                    output_base=self.output_base,
                    y_scale=self.y_scale,
                    z_scale=self.z_scale,
                    verbose=self.verbose,
                    use_tqdm=self.use_tqdm,
                    colormap=self.colormap,
                    max_processing_percentile=self.max_processing_percentile,
                    cusp_marker_style=self.cusp_marker_style,
                ),
            )
            self._proc.start()
            self._proc.join()
            exitcode, self._proc = self._proc.exitcode, None
            if self.isInterruptionRequested():
                self.stopped.emit()
            elif exitcode == 0:
                self.finished.emit()
            else:
                self.error.emit(f"Plotting process exited with code {exitcode}")
        except Exception as exc:
            self.error.emit(str(exc))


class SinglePlotWorker(_BaseWorker):
    """Background thread that renders one FAST ESA spectrogram figure.

    Parameters
    ----------
    mode : {'single_instrument', 'full_orbit'}
        Which single-output plotting function to call.
    cdf_file_path : str
        Path to one CDF file (used when ``mode == 'single_instrument'``).
    data_source : {'folder', 'files', 'download'}
        How ``mode == 'full_orbit'`` obtains its instrument CDF files:
        discovered from ``orbit_folder_path`` + ``orbit_number``, supplied
        directly via ``instrument_file_paths``, or fetched from CDA Web for
        ``download_date`` / ``download_instruments``.
    orbit_folder_path : str
        Root data folder (used when ``data_source == 'folder'``).
    orbit_number : int or None
        Orbit number to resolve within ``orbit_folder_path`` (used when
        ``data_source == 'folder'``).
    instrument_file_paths : dict of {str: str}
        Instrument -> CDF file path mapping (used when
        ``data_source == 'files'``).
    download_date : datetime.date or None
        Calendar day to fetch (used when ``data_source == 'download'``).
    download_instruments : set of str
        Instrument codes to fetch (used when ``data_source == 'download'``).
    output_path : str
        Destination PNG file path.
    y_scale, z_scale : {'linear', 'log'}
        Axis scaling.
    colormap : str
        Matplotlib colormap name.
    cusp_marker_style : {'line', 'bracket'}
        Cusp-boundary marker style.
    parent : QObject or None, optional
        Qt parent object.
    """

    def __init__(
        self,
        mode: str,
        cdf_file_path: str,
        data_source: str,
        orbit_folder_path: str,
        orbit_number: int | None,
        instrument_file_paths: dict[str, str],
        download_date: dt.date | None,
        download_instruments: set[str],
        output_path: str,
        y_scale: str,
        z_scale: str,
        colormap: str,
        cusp_marker_style: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.cdf_file_path = cdf_file_path
        self.data_source = data_source
        self.orbit_folder_path = orbit_folder_path
        self.orbit_number = orbit_number
        self.instrument_file_paths = instrument_file_paths
        self.download_date = download_date
        self.download_instruments = download_instruments
        self.output_path = output_path
        self.y_scale = y_scale
        self.z_scale = z_scale
        self.colormap = colormap
        self.cusp_marker_style = cusp_marker_style

    def run(self) -> None:
        """Execute the single-plot render; emits ``finished``, ``stopped``, or ``error``."""
        try:
            self.progress.emit("Generating spectrogram…")
            self._proc = multiprocessing.Process(
                target=_single_plot_in_process,
                kwargs=dict(
                    mode=self.mode,
                    cdf_file_path=self.cdf_file_path,
                    data_source=self.data_source,
                    orbit_folder_path=self.orbit_folder_path,
                    orbit_number=self.orbit_number,
                    instrument_file_paths=self.instrument_file_paths,
                    download_date=self.download_date,
                    download_instruments=self.download_instruments,
                    output_path=self.output_path,
                    y_scale=self.y_scale,
                    z_scale=self.z_scale,
                    colormap=self.colormap,
                    cusp_marker_style=self.cusp_marker_style,
                ),
            )
            self._proc.start()
            self._proc.join()
            exitcode, self._proc = self._proc.exitcode, None
            if self.isInterruptionRequested():
                self.stopped.emit()
            elif exitcode == 0:
                self.finished.emit()
            else:
                self.error.emit(f"Plotting process exited with code {exitcode}")
        except Exception as exc:
            self.error.emit(str(exc))


# --- Custom widgets ---
class NavButton(QPushButton):
    """Sidebar navigation button with a Material icon above a text label.

    Renders as a fixed-height tile; the ``selected`` Qt property drives
    stylesheet-level highlighting via ``set_selected``. The icon is stored as
    a recoloured pixmap so it remains visible on both dark and light themes.

    Parameters
    ----------
    icon : QIcon
        Source icon displayed above the text label.
    label : str
        Short descriptive text shown below the icon.
    icon_color : str, default '#ffffff'
        Hex colour applied to the icon pixmap via ``_colored_pixmap``.
    parent : QWidget or None, optional
        Qt parent widget.
    """

    def __init__(self, icon: QIcon, label: str, icon_color: str = "#ffffff", parent=None) -> None:
        super().__init__(parent)
        self._icon_src = icon
        self._icon_color = icon_color
        self.setObjectName("navBtn")
        self.setCheckable(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # ty: ignore[unresolved-attribute]
        self.setFixedHeight(64)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(2)
        lay.setAlignment(Qt.AlignCenter)  # ty: ignore[unresolved-attribute]
        self._icon_label = QLabel()
        self._icon_label.setAlignment(Qt.AlignCenter)  # ty: ignore[unresolved-attribute]
        self._icon_label.setPixmap(_colored_pixmap(icon, 28, icon_color))
        self._icon_label.setAttribute(Qt.WA_TransparentForMouseEvents)  # ty: ignore[unresolved-attribute]
        self._text_label = QLabel(label)
        self._text_label.setAlignment(Qt.AlignCenter)  # ty: ignore[unresolved-attribute]
        self._text_label.setFont(QFont("Inter", 10, QFont.Medium))  # ty: ignore[unresolved-attribute]
        self._text_label.setAttribute(Qt.WA_TransparentForMouseEvents)  # ty: ignore[unresolved-attribute]
        lay.addWidget(self._icon_label)
        lay.addWidget(self._text_label)

    def set_selected(self, selected: bool) -> None:
        """Set the ``selected`` Qt property and repaint to apply the highlight style."""
        self.setProperty("selected", str(selected).lower())
        self.style().unpolish(self)
        self.style().polish(self)

    def update_icon(self, icon: QIcon | None = None, color: str | None = None) -> None:
        """Replace the icon source and/or tint colour, then refresh the pixmap.

        Parameters
        ----------
        icon : QIcon or None, optional
            New source icon; unchanged when ``None``.
        color : str or None, optional
            New hex tint colour; unchanged when ``None``.
        """
        if icon is not None:
            self._icon_src = icon
        if color is not None:
            self._icon_color = color
        self._icon_label.setPixmap(_colored_pixmap(self._icon_src, 28, self._icon_color))


class ToggleChip(QPushButton):
    """Checkable pill-style chip button that reflects its state via a Qt property.

    The ``selected`` property drives active styling so checked chips appear
    filled while unchecked chips show only a border.

    Parameters
    ----------
    text : str
        Chip label text.
    parent : QWidget or None, optional
        Qt parent widget.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setCheckable(True)
        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        """Sync the ``selected`` Qt property to the new toggle state."""
        self.setProperty("selected", "true" if checked else "false")
        self.style().unpolish(self)
        self.style().polish(self)


# --- Download page ---
class DownloadPage(QWidget):
    """Page widget for batch-downloading FAST ESA CDF files.

    Presents chip selectors for instruments and years, a folder picker, and
    a CTA button that spawns a ``DownloadWorker``. The CTA is only enabled
    when at least one instrument, one year, and a folder are all selected.

    Parameters
    ----------
    parent : QWidget or None, optional
        Qt parent widget.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.given_instruments: set[str] = set()
        self.years: list[int] = []
        self.data_folder: str = ""
        self._worker: DownloadWorker | None = None
        self._stop_requested: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct and lay out all child widgets for the download page."""
        layout, self._scroll = _make_scroll_page(self)
        _page_header(
            layout,
            "Batch Download FAST CDF Files",
            "By using CDA Web, this program can download CDF files for the FAST "
            "mission given the type of instrument data, target year, and output "
            "folder. Due to the current state of FAST CDF files, some instrument "
            "data may not be available (e.g. orb ephemeris), and there may be some "
            "missing or incomplete data (e.g. esv only has data through 2002).",
        )
        _add_divider(layout)
        _section_label(layout, "Settings", spacing=20, point_size=17)
        _section_label(layout, "Instrument Data")
        chip_row = QHBoxLayout()
        chip_row.setSpacing(8)
        chip_row.setContentsMargins(0, 0, 0, 0)
        self._instrument_chips: dict[str, ToggleChip] = {}
        for name in INSTRUMENT_OPTIONS:
            chip = ToggleChip(name)
            chip.toggled.connect(self._update_instruments)
            self._instrument_chips[name] = chip
            chip_row.addWidget(chip)
        chip_row.addStretch()
        layout.addLayout(chip_row)
        _add_divider(layout)
        _section_label(layout, "Years")
        year_grid = QGridLayout()
        year_grid.setSpacing(8)
        year_grid.setContentsMargins(0, 0, 0, 0)
        self._year_chips: dict[int, ToggleChip] = {}
        for idx, year in enumerate(range(MIN_YEAR, MAX_YEAR + 1)):
            chip = ToggleChip(str(year))
            chip.toggled.connect(self._update_years)
            self._year_chips[year] = chip
            year_grid.addWidget(chip, idx // 7, idx % 7)
        layout.addLayout(year_grid)
        _add_divider(layout)
        _section_label(layout, "Output Folder")
        self._folder_btn, self._folder_path_label = _folder_selector(layout, self._select_folder)
        layout.addSpacing(8)
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        layout.addWidget(self._status_label)
        layout.addStretch()
        cta_row = QHBoxLayout()
        cta_row.addStretch()
        self._stop_btn = _make_stop_btn(self._stop_download)
        cta_row.addWidget(self._stop_btn)
        cta_row.addSpacing(8)
        self._cta_btn = QPushButton("  Confirm and Download")
        self._cta_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("file_download"), 24, "#ffffff")))
        self._cta_btn.setObjectName("ctaBtn")
        self._cta_btn.setFixedHeight(48)
        self._cta_btn.setEnabled(False)
        self._cta_btn.clicked.connect(self._start_download)
        cta_row.addWidget(self._cta_btn)
        layout.addLayout(cta_row)

    def _update_instruments(self) -> None:
        """Sync ``self.given_instruments`` from chip states and refresh the CTA."""
        self.given_instruments = {n for n, c in self._instrument_chips.items() if c.isChecked()}
        self._check_ready()

    def _update_years(self) -> None:
        """Sync ``self.years`` from chip states, update the status label, and refresh CTA."""
        self.years = sorted(y for y, c in self._year_chips.items() if c.isChecked())
        if self.years:
            n = len(self.years)
            span = f"{self.years[0]}–{self.years[-1]}" if n > 1 else str(self.years[0])
            self._set_status(f"{n} year(s) selected: {span}", PRIMARY)
        else:
            self._set_status("", "")
        self._check_ready()

    def _select_folder(self) -> None:
        """Open a directory picker and store the selected output path."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", str(Path.home()))
        if folder:
            self.data_folder = folder
            self._folder_path_label.setText(_truncated_path(folder))
        self._check_ready()

    def _check_ready(self) -> None:
        """Enable the CTA only when all required inputs are set."""
        self._cta_btn.setEnabled(bool(self.given_instruments and self.years and self.data_folder))

    def _start_download(self) -> None:
        """Disable the CTA, show the stop button, and launch a ``DownloadWorker``."""
        self._cta_btn.setEnabled(False)
        self._stop_btn.setVisible(True)
        self._stop_requested = False
        self._set_status("Starting download…", PRIMARY)
        self._worker = DownloadWorker(set(self.given_instruments), list(self.years), self.data_folder)
        self._worker.progress.connect(lambda m: self._status_label.setText(m))
        self._worker.finished.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop_download(self) -> None:
        """Kill the active download subprocess; worker emits ``stopped`` on completion."""
        if self._worker and self._worker.isRunning():
            self._stop_requested = True
            self._worker.request_stop()
            self._set_status("Stopping…", "#9e9e9e")

    def _reset_controls(self) -> None:
        """Hide the stop button and re-enable the CTA after a worker completes."""
        self._stop_btn.setVisible(False)
        self._cta_btn.setEnabled(True)

    def _set_status(self, text: str, color: str = "") -> None:
        """Update the status label text and inline colour rule."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 500;" if color else "")

    def _handle_cancelled(self) -> bool:
        """Return ``True`` and show "Stopped." if the user initiated the stop.

        Resets ``_stop_requested`` as a side-effect.
        """
        if not self._stop_requested:
            return False
        self._stop_requested = False
        self._set_status("Stopped.", "#9e9e9e")
        return True

    def _on_stopped(self) -> None:
        """Handle worker-confirmed cancellation."""
        self._reset_controls()
        self._set_status("Stopped.", "#9e9e9e")

    def _on_finished(self) -> None:
        """Handle download completion; suppresses the success dialog when stopped."""
        self._reset_controls()
        if self._handle_cancelled():
            return
        self._set_status("✓ Download complete.", "#2E7D32")
        QMessageBox.information(self, "Done", "All files downloaded successfully.")

    def _on_error(self, msg: str) -> None:
        """Display a download error message (suppressed when the user stopped the worker)."""
        self._reset_controls()
        if self._handle_cancelled():
            return
        self._set_status(f"Error: {msg}", ERROR)
        QMessageBox.critical(self, "Download Error", msg)

    def apply_theme_colors(self, is_dark: bool) -> None:
        """Recolour icons and pin scrollbar colour for the current theme.

        Widget-level CSS on the scroll area takes highest precedence and
        survives Qt style re-polish events triggered by page switching.
        """
        fg = "#ffffff" if is_dark else "#3c3c3c"
        primary = os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#1de9b6" if is_dark else "#e040fb")
        track = "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.10)"
        self._scroll.setStyleSheet(_scrollbar_css(primary, track))
        self._folder_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("folder_open"), 24, fg)))
        self._cta_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("file_download"), 24, fg)))


# --- Plot page ---
class PlotPage(QWidget):
    """Page widget for generating FAST ESA spectrograms from local CDF files.

    Provides folder pickers for data and output directories, toggles for
    verbose logging and tqdm, combo boxes for axis scales and colormap, and a
    linked slider/entry pair for the noise-cutoff percentile. A ``PlotWorker``
    is spawned when the CTA is clicked.

    Parameters
    ----------
    parent : QWidget or None, optional
        Qt parent widget.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.FAST_CDF_data_folder_path: str = ""
        self.output_plots_folder_path: str = ""
        self._worker: PlotWorker | None = None
        self._stop_requested: bool = False
        self.verbose: bool = False
        self.use_tqdm: bool = True
        self.y_scale: str = SCALE_OPTIONS[1]
        self.z_scale: str = SCALE_OPTIONS[1]
        self.colormap: str = COLORMAP_OPTIONS[0]
        self.cusp_marker_style: str = CUSP_STYLE_OPTIONS[0]
        self.max_processing_percentile: float = DEFAULT_NOISE_PERCENTILE
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct and lay out all child widgets for the plot page."""
        layout, self._scroll = _make_scroll_page(self)
        _page_header(
            layout,
            "Plot FAST CDF Data",
            "Generate spectrograms from locally stored FAST ESA CDF files. "
            "The data folder must follow the expected hierarchy so that files "
            "can be located automatically. Adjust the scaling, colormap, and "
            "noise cutoff to fine-tune the output plots."
            "NOTE: only ees, eeb, ies, and ieb FAST CDF files can be plotted"
            "with this program.",
        )
        _add_divider(layout)
        _section_label(layout, "Settings", spacing=20, point_size=17)
        _section_label(layout, "FAST CDF Data Folder", spacing=6)
        self._folder_btn, self._folder_path_label = _folder_selector(
            layout,
            self._select_folder,
            note=(
                "Directory must follow the hierarchy:  /top_level_dir/year/month/  "
                "(top_level_dir may be named anything)"
            ),
        )
        _add_divider(layout)
        _section_label(layout, "Plot Output Folder", spacing=6)
        self._output_folder_btn, self._output_folder_path_label = _folder_selector(
            layout,
            self._select_output_folder,
            note=(
                "Output will follow the hierarchy:  /top_level_dir/year/month/  (top_level_dir may be named anything)"
            ),
        )
        _add_divider(layout)
        self._verbose_check = _add_toggle_section(
            layout,
            "Verbose Logging",
            "Enable verbose output",
            "Error logging will be more detailed but may be overwhelming for large batches.",
            self.verbose,
            lambda v: setattr(self, "verbose", v),
        )
        _add_divider(layout)
        self._tqdm_check = _add_toggle_section(
            layout,
            "Use tqdm Progress Bar",
            "Enable tqdm",
            "tqdm displays estimated plotting progress in the terminal, but may negatively"
            + " impact performance. Without using tqdm, the plotting progress is not displayed",
            self.use_tqdm,
            lambda v: setattr(self, "use_tqdm", v),
        )
        _add_divider(layout)
        _section_label(layout, "Energy Scaling")
        self._y_scale_combo = _make_combo(SCALE_OPTIONS, self.y_scale)
        self._y_scale_combo.currentTextChanged.connect(lambda t: setattr(self, "y_scale", t))
        layout.addWidget(self._y_scale_combo)
        _add_divider(layout)
        _section_label(layout, "Counts / Flux Scaling")
        self._z_scale_combo = _make_combo(SCALE_OPTIONS, self.z_scale)
        self._z_scale_combo.currentTextChanged.connect(lambda t: setattr(self, "z_scale", t))
        layout.addWidget(self._z_scale_combo)
        _add_divider(layout)
        _section_label(layout, "Colormap")
        self._colormap_combo = _make_combo(COLORMAP_OPTIONS, self.colormap)
        self._colormap_combo.currentTextChanged.connect(lambda t: setattr(self, "colormap", t))
        layout.addWidget(self._colormap_combo)
        _add_divider(layout)
        _section_label(layout, "Cusp Marker Style")
        self._cusp_style_combo = _make_combo(CUSP_STYLE_OPTIONS, self.cusp_marker_style)
        self._cusp_style_combo.currentTextChanged.connect(lambda t: setattr(self, "cusp_marker_style", t))
        layout.addWidget(self._cusp_style_combo)
        _add_divider(layout)
        _section_label(layout, "Noise Cutoff Percentile")
        self._percentile_slider, self._percentile_entry, self._percentile_note = _percentile_slider_section(
            layout, DEFAULT_NOISE_PERCENTILE
        )
        self._percentile_slider.valueChanged.connect(self._on_slider_changed)
        self._percentile_entry.editingFinished.connect(self._on_entry_changed)
        self._update_percentile_note()
        layout.addStretch()
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        layout.addWidget(self._status_label)
        cta_row = QHBoxLayout()
        cta_row.addStretch()
        self._stop_btn = _make_stop_btn(self._stop_plot)
        cta_row.addWidget(self._stop_btn)
        cta_row.addSpacing(8)
        self._cta_btn = QPushButton("  Confirm and Plot")
        self._cta_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("area_chart"), 24, "#ffffff")))
        self._cta_btn.setObjectName("ctaBtn")
        self._cta_btn.setFixedHeight(48)
        self._cta_btn.setEnabled(False)
        self._cta_btn.clicked.connect(self._start_plot)
        cta_row.addWidget(self._cta_btn)
        layout.addSpacing(24)
        layout.addLayout(cta_row)

    def _select_folder(self) -> None:
        """Open a directory picker and store the CDF data folder path."""
        folder = QFileDialog.getExistingDirectory(self, "Select FAST CDF Data Folder", str(Path.home()))
        if folder:
            self.FAST_CDF_data_folder_path = folder
            self._folder_path_label.setText(_truncated_path(folder))
        self._check_ready()

    def _select_output_folder(self) -> None:
        """Open a directory picker and store the plot output folder path."""
        folder = QFileDialog.getExistingDirectory(self, "Select Plot Output Folder", str(Path.home()))
        if folder:
            self.output_plots_folder_path = folder
            self._output_folder_path_label.setText(_truncated_path(folder))
        self._check_ready()

    def _on_slider_changed(self, value: int) -> None:
        """Sync the entry field and note label when the slider moves."""
        self.max_processing_percentile = float(value)
        self._percentile_entry.setText(str(value))
        self._update_percentile_note()

    def _on_entry_changed(self) -> None:
        """Validate the manual entry field, clamp it to 0–100, then sync the slider."""
        try:
            value = max(0.0, min(100.0, float(self._percentile_entry.text().strip())))
        except ValueError:
            value = self.max_processing_percentile
        self.max_processing_percentile = value
        self._percentile_slider.blockSignals(True)
        self._percentile_slider.setValue(int(round(value)))
        self._percentile_slider.blockSignals(False)
        self._percentile_entry.setText(f"{value:.1f}")
        self._update_percentile_note()

    def _update_percentile_note(self) -> None:
        """Refresh the hint label below the percentile controls."""
        self._percentile_note.setText(
            f"Current value: {self.max_processing_percentile:.1f}  — "
            "pixels above this percentile of the data range are clipped as noise. "
            "It is suggested to keep this above 90."
        )

    def _check_ready(self) -> None:
        """Enable the CTA only when both folder paths are set."""
        self._cta_btn.setEnabled(bool(self.FAST_CDF_data_folder_path and self.output_plots_folder_path))

    def _start_plot(self) -> None:
        """Disable the CTA, show the stop button, and launch a ``PlotWorker``."""
        self._cta_btn.setEnabled(False)
        self._stop_btn.setVisible(True)
        self._stop_requested = False
        self._set_status("Starting plotting…", PRIMARY)
        self._worker = PlotWorker(
            self.FAST_CDF_data_folder_path,
            self.output_plots_folder_path,
            self.y_scale,
            self.z_scale,
            self.verbose,
            self.use_tqdm,
            self.colormap,
            self.max_processing_percentile,
            self.cusp_marker_style,
        )
        self._worker.progress.connect(lambda m: self._status_label.setText(m))
        self._worker.finished.connect(self._on_plot_finished)
        self._worker.stopped.connect(self._on_plot_stopped)
        self._worker.error.connect(self._on_plot_error)
        self._worker.start()

    def _stop_plot(self) -> None:
        """Kill the active plot subprocess; worker emits ``stopped`` on completion."""
        if self._worker and self._worker.isRunning():
            self._stop_requested = True
            self._worker.request_stop()
            self._set_status("Stopping…", "#9e9e9e")

    def _reset_controls(self) -> None:
        """Hide the stop button and re-enable the CTA after a worker completes."""
        self._stop_btn.setVisible(False)
        self._cta_btn.setEnabled(True)

    def _set_status(self, text: str, color: str = "") -> None:
        """Update the status label text and inline colour rule."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 500;" if color else "")

    def _handle_cancelled(self) -> bool:
        """Return ``True`` and show "Stopped." if the user initiated the stop.

        Resets ``_stop_requested`` as a side-effect.
        """
        if not self._stop_requested:
            return False
        self._stop_requested = False
        self._set_status("Stopped.", "#9e9e9e")
        return True

    def _on_plot_stopped(self) -> None:
        """Handle worker-confirmed plot cancellation."""
        self._reset_controls()
        self._set_status("Stopped.", "#9e9e9e")

    def _on_plot_finished(self) -> None:
        """Handle successful plot generation; suppresses the dialog when stopped."""
        self._reset_controls()
        if self._handle_cancelled():
            return
        self._set_status("✓ Plot generation complete.", "#2E7D32")
        QMessageBox.information(self, "Done", "All spectrograms generated successfully.")

    def _on_plot_error(self, msg: str) -> None:
        """Display a plotting error message (suppressed when the user stopped the worker)."""
        self._reset_controls()
        if self._handle_cancelled():
            return
        self._set_status(f"Error: {msg}", ERROR)
        QMessageBox.critical(self, "Plot Error", msg)

    def apply_theme_colors(self, is_dark: bool) -> None:
        """Recolour icons, pin scrollbar colour, and update input text for the current theme."""
        fg = "#ffffff" if is_dark else "#3c3c3c"
        primary = os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#1de9b6" if is_dark else "#e040fb")
        track = "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.10)"
        self._scroll.setStyleSheet(_scrollbar_css(primary, track))
        self._folder_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("folder_open"), 24, fg)))
        self._output_folder_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("folder_open"), 24, fg)))
        self._cta_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("area_chart"), 24, fg)))
        for combo in (self._y_scale_combo, self._z_scale_combo, self._colormap_combo, self._cusp_style_combo):
            combo.setStyleSheet(f"color: {fg};")
            combo.view().setStyleSheet(f"color: {fg};")
        self._percentile_entry.setStyleSheet(f"color: {fg};")


# --- Single plot page ---
class SinglePlotPage(QWidget):
    """Page widget for rendering a single FAST ESA spectrogram figure.

    Two modes are supported: plotting one CDF file's pitch-angle grid
    (orbit number auto-parsed from the filename), or plotting one orbit's
    multi-instrument grid. Full Orbit mode's instrument CDF files can come
    from any of three sources: a data folder resolved by orbit number, CDF
    files selected individually per instrument, or a calendar day fetched
    from CDA Web on demand. A ``SinglePlotWorker`` is spawned when the CTA
    is clicked.

    Parameters
    ----------
    parent : QWidget or None, optional
        Qt parent widget.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mode: str = "single_instrument"
        self.cdf_file_path: str = ""
        self.data_source: str = "folder"
        self.orbit_folder_path: str = ""
        self.orbit_number: int | None = None
        self.instrument_file_paths: dict[str, str] = {}
        self.download_date: dt.date = dt.date(2000, 1, 1)
        self.download_instruments: set[str] = set()
        self.output_path: str = ""
        self._worker: SinglePlotWorker | None = None
        self._stop_requested: bool = False
        self.y_scale: str = SCALE_OPTIONS[1]
        self.z_scale: str = SCALE_OPTIONS[1]
        self.colormap: str = COLORMAP_OPTIONS[0]
        self.cusp_marker_style: str = CUSP_STYLE_OPTIONS[0]
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct and lay out all child widgets for the single-plot page."""
        layout, self._scroll = _make_scroll_page(self)
        _page_header(
            layout,
            "Single Spectrogram Plot",
            "Render one FAST ESA spectrogram figure without running a full batch. "
            "Choose a single CDF file to plot its pitch-angle grid, or plot one "
            "orbit's multi-instrument grid -- resolved from a data folder and "
            "orbit number, from CDF files you select yourself (one per "
            "instrument), or by downloading one calendar day directly from CDA "
            "Web (1996-08-21 through 2009-05-04).",
        )
        _add_divider(layout)
        _section_label(layout, "Settings", spacing=20, point_size=17)
        _section_label(layout, "Plot Mode")
        self._mode_combo = _make_combo(PLOT_MODE_OPTIONS, PLOT_MODE_OPTIONS[0])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_combo)
        _add_divider(layout)

        self._file_section_label = _section_label(layout, "CDF File", spacing=6)
        file_row = QHBoxLayout()
        file_row.setSpacing(12)
        file_row.setContentsMargins(0, 0, 0, 0)
        self._file_btn = QPushButton("  Select CDF File")
        self._file_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("description"), 24, "#ffffff")))
        self._file_btn.setObjectName("folderBtn")
        self._file_btn.setFixedHeight(38)
        self._file_btn.clicked.connect(self._select_file)
        file_row.addWidget(self._file_btn)
        file_row.addStretch()
        layout.addLayout(file_row)
        layout.addSpacing(6)
        self._file_path_label = QLabel("No file selected")
        self._file_path_label.setObjectName("folderPath")
        layout.addWidget(self._file_path_label)
        layout.addSpacing(8)

        self._source_section_label = _section_label(layout, "Data Source", spacing=6)
        self._source_combo = _make_combo(DATA_SOURCE_OPTIONS, DATA_SOURCE_OPTIONS[0])
        self._source_combo.currentTextChanged.connect(self._on_source_changed)
        layout.addWidget(self._source_combo)
        layout.addSpacing(8)

        # -- "folder" source: existing data-folder + orbit-number lookup --
        self._folder_section_label = _section_label(layout, "FAST CDF Data Folder", spacing=6)
        self._folder_btn, self._folder_path_label = _folder_selector(layout, self._select_folder)
        self._orbit_section_label = _section_label(layout, "Orbit Number", spacing=6)
        self._orbit_entry = QLineEdit()
        self._orbit_entry.setObjectName("percentileEntry")
        self._orbit_entry.setPlaceholderText("e.g. 13312")
        self._orbit_entry.setValidator(QIntValidator(0, 999999, self))
        self._orbit_entry.setFixedHeight(36)
        self._orbit_entry.setMaximumWidth(140)
        self._orbit_entry.textChanged.connect(self._on_orbit_number_changed)
        layout.addWidget(self._orbit_entry)
        layout.addSpacing(8)
        self._folder_source_widgets: tuple[QWidget, ...] = (
            self._folder_section_label,
            self._folder_btn,
            self._folder_path_label,
            self._orbit_section_label,
            self._orbit_entry,
        )

        # -- "files" source: one file picker per instrument --
        self._instrument_files_section_label = _section_label(layout, "Instrument CDF Files", spacing=6)
        self._instrument_file_btns: dict[str, QPushButton] = {}
        self._instrument_file_labels: dict[str, QLabel] = {}
        for instrument in FULL_ORBIT_INSTRUMENTS:
            inst_row = QHBoxLayout()
            inst_row.setSpacing(12)
            inst_row.setContentsMargins(0, 0, 0, 0)
            inst_btn = QPushButton(f"  Select {instrument.upper()} File")
            inst_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("description"), 24, "#ffffff")))
            inst_btn.setObjectName("folderBtn")
            inst_btn.setFixedHeight(38)
            inst_btn.clicked.connect(functools.partial(self._select_instrument_file, instrument))
            inst_row.addWidget(inst_btn)
            inst_row.addStretch()
            layout.addLayout(inst_row)
            layout.addSpacing(6)
            inst_label = QLabel("No file selected")
            inst_label.setObjectName("folderPath")
            layout.addWidget(inst_label)
            layout.addSpacing(6)
            self._instrument_file_btns[instrument] = inst_btn
            self._instrument_file_labels[instrument] = inst_label
        layout.addSpacing(2)
        self._files_source_widgets: tuple[QWidget, ...] = (
            self._instrument_files_section_label,
            *self._instrument_file_btns.values(),
            *self._instrument_file_labels.values(),
        )

        # -- "download" source: date + instrument chips, fetched via CDA Web --
        self._download_date_section_label = _section_label(layout, "Date", spacing=6)
        self._date_edit = QDateEdit()
        self._date_edit.setObjectName("percentileEntry")
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDisplayFormat("yyyy-MM-dd")
        self._date_edit.setMinimumDate(QDate(1996, 8, 21))
        self._date_edit.setMaximumDate(QDate(2009, 5, 4))
        self._date_edit.setDate(QDate(2000, 1, 1))
        self._date_edit.setFixedHeight(36)
        self._date_edit.setMaximumWidth(160)
        self._date_edit.dateChanged.connect(self._on_date_changed)
        layout.addWidget(self._date_edit)
        layout.addSpacing(8)
        self._download_instruments_section_label = _section_label(layout, "Instruments to Download", spacing=6)
        download_chip_row = QHBoxLayout()
        download_chip_row.setSpacing(8)
        download_chip_row.setContentsMargins(0, 0, 0, 0)
        self._download_instrument_chips: dict[str, ToggleChip] = {}
        for instrument in FULL_ORBIT_INSTRUMENTS:
            chip = ToggleChip(instrument)
            chip.toggled.connect(self._update_download_instruments)
            self._download_instrument_chips[instrument] = chip
            download_chip_row.addWidget(chip)
        download_chip_row.addStretch()
        layout.addLayout(download_chip_row)
        layout.addSpacing(8)
        self._download_source_widgets: tuple[QWidget, ...] = (
            self._download_date_section_label,
            self._date_edit,
            self._download_instruments_section_label,
            *self._download_instrument_chips.values(),
        )

        _add_divider(layout)
        _section_label(layout, "Energy Scaling")
        self._y_scale_combo = _make_combo(SCALE_OPTIONS, self.y_scale)
        self._y_scale_combo.currentTextChanged.connect(lambda t: setattr(self, "y_scale", t))
        layout.addWidget(self._y_scale_combo)
        _add_divider(layout)
        _section_label(layout, "Counts / Flux Scaling")
        self._z_scale_combo = _make_combo(SCALE_OPTIONS, self.z_scale)
        self._z_scale_combo.currentTextChanged.connect(lambda t: setattr(self, "z_scale", t))
        layout.addWidget(self._z_scale_combo)
        _add_divider(layout)
        _section_label(layout, "Colormap")
        self._colormap_combo = _make_combo(COLORMAP_OPTIONS, self.colormap)
        self._colormap_combo.currentTextChanged.connect(lambda t: setattr(self, "colormap", t))
        layout.addWidget(self._colormap_combo)
        _add_divider(layout)
        _section_label(layout, "Cusp Marker Style")
        self._cusp_style_combo = _make_combo(CUSP_STYLE_OPTIONS, self.cusp_marker_style)
        self._cusp_style_combo.currentTextChanged.connect(lambda t: setattr(self, "cusp_marker_style", t))
        layout.addWidget(self._cusp_style_combo)
        _add_divider(layout)
        _section_label(layout, "Output PNG File", spacing=6)
        out_row = QHBoxLayout()
        out_row.setSpacing(12)
        out_row.setContentsMargins(0, 0, 0, 0)
        self._output_btn = QPushButton("  Choose Output File")
        self._output_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("save"), 24, "#ffffff")))
        self._output_btn.setObjectName("folderBtn")
        self._output_btn.setFixedHeight(38)
        self._output_btn.clicked.connect(self._select_output_file)
        out_row.addWidget(self._output_btn)
        out_row.addStretch()
        layout.addLayout(out_row)
        layout.addSpacing(6)
        self._output_path_label = QLabel("No output file selected")
        self._output_path_label.setObjectName("folderPath")
        layout.addWidget(self._output_path_label)
        layout.addSpacing(8)

        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        layout.addWidget(self._status_label)
        layout.addStretch()
        cta_row = QHBoxLayout()
        cta_row.addStretch()
        self._stop_btn = _make_stop_btn(self._stop_plot)
        cta_row.addWidget(self._stop_btn)
        cta_row.addSpacing(8)
        self._cta_btn = QPushButton("  Confirm and Plot")
        self._cta_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("area_chart"), 24, "#ffffff")))
        self._cta_btn.setObjectName("ctaBtn")
        self._cta_btn.setFixedHeight(48)
        self._cta_btn.setEnabled(False)
        self._cta_btn.clicked.connect(self._start_plot)
        cta_row.addWidget(self._cta_btn)
        layout.addSpacing(24)
        layout.addLayout(cta_row)

        self._update_mode_visibility()

    def _on_mode_changed(self, text: str) -> None:
        """Switch between single-instrument and full-orbit input widgets."""
        self.mode = "single_instrument" if text == PLOT_MODE_OPTIONS[0] else "full_orbit"
        self._update_mode_visibility()
        self._check_ready()

    def _update_mode_visibility(self) -> None:
        """Show only the input widgets relevant to the current mode."""
        is_single = self.mode == "single_instrument"
        for widget in (self._file_section_label, self._file_btn, self._file_path_label):
            widget.setVisible(is_single)
        for widget in (self._source_section_label, self._source_combo):
            widget.setVisible(not is_single)
        if is_single:
            for widget in (*self._folder_source_widgets, *self._files_source_widgets, *self._download_source_widgets):
                widget.setVisible(False)
        else:
            self._update_source_visibility()

    def _on_source_changed(self, text: str) -> None:
        """Switch Full Orbit mode's data-acquisition widgets to match the selected source."""
        self.data_source = {
            DATA_SOURCE_OPTIONS[0]: "folder",
            DATA_SOURCE_OPTIONS[1]: "files",
            DATA_SOURCE_OPTIONS[2]: "download",
        }[text]
        self._update_source_visibility()
        self._check_ready()

    def _update_source_visibility(self) -> None:
        """Show only the data-acquisition widgets for the current Full Orbit data source."""
        for widget in self._folder_source_widgets:
            widget.setVisible(self.data_source == "folder")
        for widget in self._files_source_widgets:
            widget.setVisible(self.data_source == "files")
        for widget in self._download_source_widgets:
            widget.setVisible(self.data_source == "download")

    def _select_file(self) -> None:
        """Open a file picker and store the selected CDF file path."""
        path, _filter = QFileDialog.getOpenFileName(self, "Select CDF File", str(Path.home()), "CDF files (*.cdf)")
        if path:
            self.cdf_file_path = path
            self._file_path_label.setText(_truncated_path(path))
        self._check_ready()

    def _select_folder(self) -> None:
        """Open a directory picker and store the CDF data folder path."""
        folder = QFileDialog.getExistingDirectory(self, "Select FAST CDF Data Folder", str(Path.home()))
        if folder:
            self.orbit_folder_path = folder
            self._folder_path_label.setText(_truncated_path(folder))
        self._check_ready()

    def _on_orbit_number_changed(self, text: str) -> None:
        """Parse the orbit-number entry field, tolerating an empty/invalid value."""
        text = text.strip()
        self.orbit_number = int(text) if text.isdigit() else None
        self._check_ready()

    def _select_instrument_file(self, instrument: str) -> None:
        """Open a file picker for one instrument and store its selected CDF path."""
        path, _filter = QFileDialog.getOpenFileName(
            self, f"Select {instrument.upper()} CDF File", str(Path.home()), "CDF files (*.cdf)"
        )
        if path:
            self.instrument_file_paths[instrument] = path
            self._instrument_file_labels[instrument].setText(_truncated_path(path))
        self._check_ready()

    def _on_date_changed(self, qdate: QDate) -> None:
        """Sync ``self.download_date`` from the date-edit widget."""
        self.download_date = dt.date(qdate.year(), qdate.month(), qdate.day())
        self._check_ready()

    def _update_download_instruments(self) -> None:
        """Sync ``self.download_instruments`` from the instrument chip states."""
        self.download_instruments = {n for n, c in self._download_instrument_chips.items() if c.isChecked()}
        self._check_ready()

    def _select_output_file(self) -> None:
        """Open a save-file picker and store the destination PNG path."""
        path, _filter = QFileDialog.getSaveFileName(
            self, "Choose Output PNG File", str(Path.home()), "PNG files (*.png)"
        )
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            self.output_path = path
            self._output_path_label.setText(_truncated_path(path))
        self._check_ready()

    def _check_ready(self) -> None:
        """Enable the CTA only when the current mode/source's required inputs are set."""
        if self.mode == "single_instrument":
            ready = bool(self.cdf_file_path and self.output_path)
        elif self.data_source == "files":
            ready = bool(self.instrument_file_paths and self.output_path)
        elif self.data_source == "download":
            ready = bool(self.download_instruments and self.output_path)
        else:
            ready = bool(self.orbit_folder_path and self.orbit_number is not None and self.output_path)
        self._cta_btn.setEnabled(ready)

    def _start_plot(self) -> None:
        """Disable the CTA, show the stop button, and launch a ``SinglePlotWorker``."""
        self._cta_btn.setEnabled(False)
        self._stop_btn.setVisible(True)
        self._stop_requested = False
        self._set_status("Starting plotting…", PRIMARY)
        self._worker = SinglePlotWorker(
            self.mode,
            self.cdf_file_path,
            self.data_source,
            self.orbit_folder_path,
            self.orbit_number,
            dict(self.instrument_file_paths),
            self.download_date,
            set(self.download_instruments),
            self.output_path,
            self.y_scale,
            self.z_scale,
            self.colormap,
            self.cusp_marker_style,
        )
        self._worker.progress.connect(lambda m: self._status_label.setText(m))
        self._worker.finished.connect(self._on_plot_finished)
        self._worker.stopped.connect(self._on_plot_stopped)
        self._worker.error.connect(self._on_plot_error)
        self._worker.start()

    def _stop_plot(self) -> None:
        """Kill the active plot subprocess; worker emits ``stopped`` on completion."""
        if self._worker and self._worker.isRunning():
            self._stop_requested = True
            self._worker.request_stop()
            self._set_status("Stopping…", "#9e9e9e")

    def _reset_controls(self) -> None:
        """Hide the stop button and re-enable the CTA after a worker completes."""
        self._stop_btn.setVisible(False)
        self._cta_btn.setEnabled(True)

    def _set_status(self, text: str, color: str = "") -> None:
        """Update the status label text and inline colour rule."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 500;" if color else "")

    def _handle_cancelled(self) -> bool:
        """Return ``True`` and show "Stopped." if the user initiated the stop.

        Resets ``_stop_requested`` as a side-effect.
        """
        if not self._stop_requested:
            return False
        self._stop_requested = False
        self._set_status("Stopped.", "#9e9e9e")
        return True

    def _on_plot_stopped(self) -> None:
        """Handle worker-confirmed plot cancellation."""
        self._reset_controls()
        self._set_status("Stopped.", "#9e9e9e")

    def _on_plot_finished(self) -> None:
        """Handle successful plot generation; suppresses the dialog when stopped."""
        self._reset_controls()
        if self._handle_cancelled():
            return
        self._set_status("✓ Plot generation complete.", "#2E7D32")
        QMessageBox.information(self, "Done", "Spectrogram generated successfully.")

    def _on_plot_error(self, msg: str) -> None:
        """Display a plotting error message (suppressed when the user stopped the worker)."""
        self._reset_controls()
        if self._handle_cancelled():
            return
        self._set_status(f"Error: {msg}", ERROR)
        QMessageBox.critical(self, "Plot Error", msg)

    def apply_theme_colors(self, is_dark: bool) -> None:
        """Recolour icons, pin scrollbar colour, and update input text for the current theme."""
        fg = "#ffffff" if is_dark else "#3c3c3c"
        primary = os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#1de9b6" if is_dark else "#e040fb")
        track = "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.10)"
        self._scroll.setStyleSheet(_scrollbar_css(primary, track))
        self._file_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("description"), 24, fg)))
        self._folder_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("folder_open"), 24, fg)))
        self._output_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("save"), 24, fg)))
        self._cta_btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("area_chart"), 24, fg)))
        for btn in self._instrument_file_btns.values():
            btn.setIcon(QIcon(_colored_pixmap(MaterialIcon("description"), 24, fg)))
        for combo in (
            self._mode_combo,
            self._source_combo,
            self._y_scale_combo,
            self._z_scale_combo,
            self._colormap_combo,
            self._cusp_style_combo,
        ):
            combo.setStyleSheet(f"color: {fg};")
            combo.view().setStyleSheet(f"color: {fg};")
        self._orbit_entry.setStyleSheet(f"color: {fg};")
        self._date_edit.setStyleSheet(f"color: {fg};")


# --- Main application window ---
class MainWindow(QMainWindow, QtStyleTools):
    """Top-level application window with a Material sidebar and a page stack.

    The sidebar contains ``NavButton`` tiles for the Download and Plot pages
    and a theme-toggle button at the bottom. The content area hosts a
    ``QStackedWidget`` switching between ``DownloadPage`` and ``PlotPage``.
    The initial theme is inferred from the OS colour scheme.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FAST CDF File Download And Spectrogram Plotting Tool")
        self.setMinimumSize(820, 600)
        self.resize(1000, 700)
        self.current_theme: str = THEME_DARK if _system_is_dark() else THEME_LIGHT
        is_dark = self.current_theme == THEME_DARK
        nav_color = "#ffffff" if is_dark else "#3c3c3c"

        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(88)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(8, 24, 8, 24)
        sb.setSpacing(4)
        self._nav_download = NavButton(MaterialIcon("file_download"), "Download", icon_color=nav_color)
        self._nav_plot = NavButton(MaterialIcon("area_chart"), "Plot", icon_color=nav_color)
        self._nav_single_plot = NavButton(MaterialIcon("insert_chart"), "Single Plot", icon_color=nav_color)
        self._nav_download.clicked.connect(lambda: self._switch_page(0))
        self._nav_plot.clicked.connect(lambda: self._switch_page(1))
        self._nav_single_plot.clicked.connect(lambda: self._switch_page(2))
        sb.addWidget(self._nav_download)
        sb.addWidget(self._nav_plot)
        sb.addWidget(self._nav_single_plot)
        sb.addStretch()
        self._theme_btn = NavButton(
            MaterialIcon("light_mode" if is_dark else "dark_mode"),
            "Theme",
            icon_color=nav_color,
        )
        self._theme_btn.clicked.connect(self._toggle_theme)
        sb.addWidget(self._theme_btn)
        root_layout.addWidget(sidebar)

        content = QWidget()
        content.setObjectName("root")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(16, 16, 16, 16)
        self._stack = QStackedWidget()
        self._stack.setObjectName("content")
        self._download_page = DownloadPage()
        self._plot_page = PlotPage()
        self._single_plot_page = SinglePlotPage()
        self._stack.addWidget(self._download_page)
        self._stack.addWidget(self._plot_page)
        self._stack.addWidget(self._single_plot_page)
        cl.addWidget(self._stack)
        root_layout.addWidget(content, 1)

        self._download_page.apply_theme_colors(is_dark)
        self._plot_page.apply_theme_colors(is_dark)
        self._single_plot_page.apply_theme_colors(is_dark)
        self._switch_page(0)

    def _toggle_theme(self) -> None:
        """Swap between dark and light Material themes and recolour all nav icons."""
        self.current_theme = THEME_LIGHT if self.current_theme == THEME_DARK else THEME_DARK
        new_is_dark = self.current_theme == THEME_DARK
        app_instance = QApplication.instance()
        assert isinstance(app_instance, QApplication)
        _apply_app_theme(app_instance, new_is_dark)
        new_color = "#ffffff" if new_is_dark else "#3c3c3c"
        self._nav_download.update_icon(color=new_color)
        self._nav_plot.update_icon(color=new_color)
        self._nav_single_plot.update_icon(color=new_color)
        self._theme_btn.update_icon(
            icon=MaterialIcon("light_mode" if new_is_dark else "dark_mode"),
            color=new_color,
        )
        self._download_page.apply_theme_colors(new_is_dark)
        self._plot_page.apply_theme_colors(new_is_dark)
        self._single_plot_page.apply_theme_colors(new_is_dark)
        # Re-polish nav buttons so property-based selected state re-evaluates
        # against the freshly applied stylesheet.
        self._switch_page(self._stack.currentIndex())

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop any running workers before closing so the OS sees a clean Python exit."""
        for page in (self._download_page, self._plot_page, self._single_plot_page):
            worker = page._worker
            if worker is None or not worker.isRunning():
                continue
            # Disconnect all cross-thread signals first to prevent a deadlock where the
            # worker thread emits a signal while the main thread is blocked in wait().
            for sig in (worker.progress, worker.finished, worker.stopped, worker.error):
                try:
                    sig.disconnect()
                except RuntimeError:
                    pass
            worker.request_stop()
            worker.wait(3000)
        event.accept()

    def _switch_page(self, index: int) -> None:
        """Navigate to the page at *index* (0 = Download, 1 = Plot, 2 = Single Plot)."""
        self._stack.setCurrentIndex(index)
        self._nav_download.set_selected(index == 0)
        self._nav_plot.set_selected(index == 1)
        self._nav_single_plot.set_selected(index == 2)


# --- Entry point ---
def main() -> None:
    """Initialise the QApplication, apply theme, and launch the main window."""
    app = QApplication(sys.argv)
    _apply_app_theme(app, _system_is_dark())
    window = MainWindow()
    window.show()
    app.exec()
    # os._exit bypasses multiprocessing's atexit handler, which blocks waiting for
    # child processes that closeEvent already killed; sys.exit would hang here.
    os._exit(0)


if __name__ == "__main__":
    main()
