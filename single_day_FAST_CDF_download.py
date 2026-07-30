#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI to download one day's FAST CDF files from CDA Web.

Companion to ``FAST_CDF_download.py``: where that script downloads a whole
year, this script downloads exactly one calendar day and exits. All
downloading logic lives in :mod:`configurable_spectrograms.download`; this
script only parses arguments and calls it -- the same function the GUI's
Single Plot page calls for its "Download by Date" data source.
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
]

__date__: str = "2026-07-30"
__status__: str = "Development"
__version__: str = "0.0.1"
__license__: str = "GPL-3.0"

import argparse
import datetime as dt
import sys

from configurable_spectrograms.download import (
    DEFAULT_FOLDER,
    DEFAULT_INSTRUMENT_LIST,
    FAST_ESA_BASE_URL,
    FAST_MAX_DATE,
    FAST_MIN_DATE,
    INSTRUMENT_OPTIONS,
    download_single_day_cdf,
)


def _parse_date(text: str) -> dt.date:
    """Parse a ``YYYY-MM-DD`` CLI argument and check it against FAST's coverage span."""
    try:
        parsed = dt.date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {text!r}; expected YYYY-MM-DD") from exc
    if not (FAST_MIN_DATE <= parsed <= FAST_MAX_DATE):
        raise argparse.ArgumentTypeError(
            f"{parsed.isoformat()} is outside FAST ESA CDF coverage "
            f"({FAST_MIN_DATE.isoformat()} through {FAST_MAX_DATE.isoformat()})"
        )
    return parsed


def main() -> int:
    """Parse CLI arguments and download one day of FAST ESA CDF files."""
    parser = argparse.ArgumentParser(description="Script to download one day of FAST CDF files from CDA Web")

    parser.add_argument(
        "--date",
        help="calendar day to download, YYYY-MM-DD",
        required=True,
        type=_parse_date,
    )

    parser.add_argument(
        "--base_url",
        help="base URL to get the files",
        default=FAST_ESA_BASE_URL,
    )

    parser.add_argument(
        "--output_path",
        help="path to save the files",
        default=DEFAULT_FOLDER,
    )

    parser.add_argument(
        "--instruments",
        nargs="+",
        help="instruments to download",
        default=DEFAULT_INSTRUMENT_LIST,
        choices=list(INSTRUMENT_OPTIONS),
    )

    args = parser.parse_args()

    day_files = download_single_day_cdf(
        date=args.date,
        instruments=args.instruments,
        base_url=args.base_url,
        data_folder=args.output_path,
    )

    total_files = sum(len(files) for files in day_files.values())
    if total_files == 0:
        print(f"[WARNING] No FAST CDF files found for {args.date.isoformat()}.")
        return 1
    for instrument, files in day_files.items():
        print(f"[{instrument}] {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Aborted by user.")
        sys.exit(130)
