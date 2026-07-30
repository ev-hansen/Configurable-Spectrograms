#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI to download FAST CDF files from CDA Web since the web interface can have issues.

NOTE: can't get orb ephemeris files to my knowledge :(

All downloading logic lives in :mod:`configurable_spectrograms.download`;
this script only parses arguments and calls it.
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
]

__date__: str = "2025-06-17"
__status__: str = "Development"
__version__: str = "0.0.2"
__license__: str = "GPL-3.0"

import argparse
import sys

from configurable_spectrograms.download import (
    DEFAULT_FOLDER,
    DEFAULT_INSTRUMENT_LIST,
    DEFAULT_YEAR,
    FAST_ESA_BASE_URL,
    INSTRUMENT_OPTIONS,
    FAST_ESA_CDF_download,
)


def main() -> None:
    """Parse CLI arguments and download one year of FAST ESA CDF files."""
    parser = argparse.ArgumentParser(description="Script to download FAST CDF files from CDA Web")

    parser.add_argument(
        "--base_url",
        help="base URL to get the files",
        default=FAST_ESA_BASE_URL,
    )

    parser.add_argument(
        "--year",
        help="year of data to download",
        default=DEFAULT_YEAR,
        choices=list(range(1996, 2009)),
        type=int,
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

    FAST_ESA_CDF_download(
        base_url=args.base_url,
        year=args.year,
        data_folder=args.output_path,
        instruments=args.instruments,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Aborted by user.")
        sys.exit(130)
