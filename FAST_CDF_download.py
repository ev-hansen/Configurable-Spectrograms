#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to download FAST CDF files from CDA Web since the web interface can have issues
NOTE: can't get orb ephemeris files to my knowledge :(
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

import requests
from bs4 import BeautifulSoup
import os
import sys
from tqdm import tqdm
import argparse

FAST_ESA_BASE_URL: str = "https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2"
INSTRUMENT_OPTIONS: set[str] = {
    "eeb",
    "ees",
    "ieb",
    "ies",
}  # "esv" also an option for FAST
DEFAULT_YEAR: int = 2000
DEFAULT_FOLDER: str = "./FAST_data/"


def FAST_ESA_CDF_download(
    base_url: str = FAST_ESA_BASE_URL,
    year: int = DEFAULT_YEAR,
    data_folder: str = DEFAULT_FOLDER,
    instruments: set[str] = INSTRUMENT_OPTIONS,
) -> None:
    """Automates process of downloading CDF files of FAST data

    Args:
        base_url (str, optional): _description_. Defaults to "https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2".
        instruments (set[str], optional): _description_. Defaults to {"eeb", "ees", "ieb", "ies"}.
        year (int, optional): _description_. Defaults to 2000.
        data_folder (str, optional): _description_. Defaults to "./FAST_data".
    """
    # based on code by scrapingbee and Amjad Hussain Syed
    # archived page on scrapingbee:
    #     https://web.archive.org/web/20250630144357/https://www.scrapingbee.com/webscraping-questions/beautifulsoup/how-to-find-all-links-using-beautifulsoup-and-python/
    # archived code by Amjad Hussain Syed's on stack overflow:
    #     https://web.archive.org/web/20250630134956/https://stackoverflow.com/questions/68969647/download-all-files-with-extension-from-a-page

    for i in range(1, 13):
        for instrument in instruments:
            web_folder: str = str(i).zfill(2)
            fast_data_folder = f"{data_folder}/{year}/{web_folder}"
            os.makedirs(fast_data_folder, exist_ok=True)

            # ees: https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2/ees/2000/01/
            # eeb: https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2/eeb/2000/01/
            # ies: https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2/ies/2000/01/
            # ieb: https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2/ieb/2000/01/
            page: str = f"{base_url}/{instrument}/{year}/{web_folder}"

            response = requests.get(page)
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a")  # Find all elements with the tag <a>

            print(f"{i}/{len(instruments)*12} | downloading files from: {page}")

            for link in tqdm(links):
                file_name = link.get("href")
                if ".cdf" in file_name:
                    download_link = f"{page}/{file_name}"
                    output_file = f"{fast_data_folder}/{file_name}"

                    if not os.path.exists(output_file):
                        r = requests.get(download_link, stream=True)
                        total_length = r.headers.get("content-length")

                        with open(output_file, "wb") as f:
                            if total_length is None:  # no content length header
                                f.write(r.content)
                            else:
                                total_length = int(total_length)
                                for data in r.iter_content(chunk_size=4096):
                                    f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to download FAST CDF files from CDA Web"
    )

    parser.add_argument(
        "--base_url",
        help=f"base URL to get the files",
        default=FAST_ESA_BASE_URL,
    )

    parser.add_argument(
        "--year",
        help=f"year of data to download.",
        default={DEFAULT_YEAR},
        choices=list(range(1996, 2009)),
        type=int,
    )

    parser.add_argument(
        "--output_path",
        help=f"path to save the files",
        default=DEFAULT_FOLDER,
    )

    parser.add_argument(
        "--instruments",
        nargs="+",
        help=f"instruments to download",
        default=INSTRUMENT_OPTIONS,
        choices=list(INSTRUMENT_OPTIONS),
    )

    args = parser.parse_args()

    try:
        FAST_ESA_CDF_download(
            base_url=args.base_url,
            year=args.year,
            data_folder=args.output_path,
            instruments=args.instruments,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Aborted by user.")
        sys.exit(130)
