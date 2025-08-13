#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Script to download FAST CDF files from CDA Web since the web interface can have issues
    NOTE: can't get orb ephemeris files to my knowledge :(
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Emma Mirizio", "Co-Mentor"],
    ["Marilia Samara", "Co-Mentor"],
    ]

__date__: str = "2025-08-13"
__status__: str = "Development"
__version__: str = "0.0.1"
__license__: str = "GPL-3.0"

import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm


def FAST_ESA_CDF_download(
    base_url: str = "https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2",
    instruments: set[str] = {
        "ees",
        "ies",
        "eeb",
        "ieb",
    },  # consider eeb, ieb - higher res, larger files
    year: int = 2000,
    data_folder: str = "./FAST_data/",
) -> None:
    """Automates process of downloading CDF files of FAST data

    Args:
        base_url (str, optional): _description_. Defaults to "https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2".
        instruments (set[str], optional): _description_. Defaults to {"ees", "ies"}.
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

FAST_ESA_CDF_download(year=2000)
FAST_ESA_CDF_download(year=2001)
