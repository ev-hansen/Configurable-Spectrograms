"""FAST ESA CDF file downloading from CDA Web: single-day, single-year, and threaded batch."""

import calendar
import datetime as dt
import functools
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from configurable_spectrograms.batch_runner import run_batch
from configurable_spectrograms.logging_utils import log_error

FAST_ESA_BASE_URL: str = "https://cdaweb.gsfc.nasa.gov/pub/data/fast/esa/l2"
INSTRUMENT_OPTIONS: set[str] = {
    "eeb",
    "ees",
    "ieb",
    "ies",
}  # "esv" also an option for FAST
DEFAULT_INSTRUMENT_LIST: list[str] = sorted(INSTRUMENT_OPTIONS)
DEFAULT_YEAR: int = 2000
DEFAULT_FOLDER: str = "./FAST_data/"
#: Earliest and latest calendar days with any FAST ESA CDF coverage on CDA Web.
FAST_MIN_DATE: dt.date = dt.date(1996, 8, 21)
FAST_MAX_DATE: dt.date = dt.date(2009, 5, 4)


def _download_single_cdf_file(download_link: str, output_file: str) -> None:
    """Stream one CDF file from *download_link* to *output_file*.

    Falls back to a single non-streamed write only when the server omits a
    ``Content-Length`` header, which avoids buffering the whole response in
    memory for every ordinary download.
    """
    response = requests.get(download_link, stream=True)
    if response.headers.get("content-length") is None:
        with open(output_file, "wb") as f:
            f.write(response.content)
        return
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=4096):
            f.write(chunk)


def _list_instrument_page_files(page: str) -> list[str]:
    """Scrape one CDA Web month-listing page and return its linked ``.cdf`` file names."""
    # based on code by scrapingbee and Amjad Hussain Syed
    # archived page on scrapingbee:
    #     https://web.archive.org/web/20250630144357/https://www.scrapingbee.com/webscraping-questions/beautifulsoup/how-to-find-all-links-using-beautifulsoup-and-python/
    # archived code by Amjad Hussain Syed's on stack overflow:
    #     https://web.archive.org/web/20250630134956/https://stackoverflow.com/questions/68969647/download-all-files-with-extension-from-a-page
    response = requests.get(page)
    soup = BeautifulSoup(response.content, "html.parser")
    return [
        href for a in soup.find_all("a") if isinstance(href := a.get("href"), str) and ".cdf" in href
    ]


def _matches_day(file_name: str, instrument: str, date_str: str) -> bool:
    """Return whether *file_name* is a FAST ESA CDF for *instrument* on *date_str*.

    Parameters
    ----------
    file_name : str
        Bare CDF file name (e.g. from a CDA Web directory listing link).
    instrument : str
        Instrument code (e.g. ``'eeb'``).
    date_str : str
        Calendar day as ``'YYYYMMDD'``.

    Examples
    --------
    >>> _matches_day("fa_esa_l2_eeb_20000101001737_13312_v02.cdf", "eeb", "20000101")
    True
    >>> _matches_day("fa_esa_l2_eeb_20000102001737_13313_v02.cdf", "eeb", "20000101")
    False
    >>> _matches_day("fa_esa_l2_ies_20000101001737_13312_v02.cdf", "eeb", "20000101")
    False
    """
    return file_name.endswith(".cdf") and f"_{instrument}_{date_str}" in file_name


def download_single_day_cdf(
    date: dt.date,
    instruments: list[str] = DEFAULT_INSTRUMENT_LIST,
    base_url: str = FAST_ESA_BASE_URL,
    data_folder: str = DEFAULT_FOLDER,
    _page_file_names: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Download every FAST ESA CDF file for one calendar day, per instrument.

    Parameters
    ----------
    date : datetime.date
        Calendar day to download. FAST ESA CDF coverage spans roughly
        :data:`FAST_MIN_DATE` through :data:`FAST_MAX_DATE`; a date outside
        that range simply returns empty lists.
    instruments : list of str, default DEFAULT_INSTRUMENT_LIST
        Instrument codes to download (e.g. ``['eeb', 'ees']``); the ones
        desired can be specified explicitly, e.g. when called from a CLI's
        ``--instruments`` argument.
    base_url : str, default FAST_ESA_BASE_URL
        Base CDA Web URL for FAST ESA level-2 data.
    data_folder : str, default DEFAULT_FOLDER
        Root output directory; files are saved under
        ``{data_folder}/{year}/{month}/``.
    _page_file_names : dict of {str: list of str} or None, optional
        Internal use only. Pre-scraped ``{instrument: [file_name, ...]}``
        month listing, letting :func:`FAST_ESA_CDF_download` reuse one page
        fetch across every day of the month instead of re-requesting it for
        each day. ``None`` (the default) fetches a fresh listing here.

    Returns
    -------
    dict of {str: list of str}
        Local CDF file paths for *date*, keyed by instrument -- downloaded
        just now, or already present from an earlier run. A single day
        commonly spans several FAST orbits, so an instrument may map to more
        than one file; an instrument with no data that day maps to an empty
        list.
    """
    web_folder = f"{date.month:02d}"
    date_str = date.strftime("%Y%m%d")
    fast_data_folder = f"{data_folder}/{date.year}/{web_folder}"
    os.makedirs(fast_data_folder, exist_ok=True)

    results: dict[str, list[str]] = {}
    for instrument in instruments:
        page = f"{base_url}/{instrument}/{date.year}/{web_folder}"
        if _page_file_names is not None:
            file_names = _page_file_names.get(instrument, [])
        else:
            file_names = _list_instrument_page_files(page)
        output_files = []
        for file_name in file_names:
            if _matches_day(file_name, instrument, date_str):
                output_file = f"{fast_data_folder}/{file_name}"
                if not os.path.exists(output_file):
                    _download_single_cdf_file(f"{page}/{file_name}", output_file)
                output_files.append(output_file)
        results[instrument] = sorted(output_files)
    return results


def FAST_ESA_CDF_download(
    base_url: str = FAST_ESA_BASE_URL,
    year: int = DEFAULT_YEAR,
    data_folder: str = DEFAULT_FOLDER,
    instruments: list[str] = DEFAULT_INSTRUMENT_LIST,
) -> None:
    """Download one year of FAST ESA CDF files from CDA Web.

    Scrapes each month/instrument listing page once, then calls
    :func:`download_single_day_cdf` for every calendar day of *year* against
    that cached listing, so every day is downloaded through the same
    single-day logic used for one-off single-day downloads elsewhere in
    this module, without re-requesting the same month page once per day.

    Parameters
    ----------
    base_url : str, default FAST_ESA_BASE_URL
        Base CDA Web URL for FAST ESA level-2 data.
    year : int, default DEFAULT_YEAR
        Calendar year to download.
    data_folder : str, default DEFAULT_FOLDER
        Root output directory; files are saved under
        ``{data_folder}/{year}/{month}/``.
    instruments : list of str, default DEFAULT_INSTRUMENT_LIST
        Instrument codes to download (e.g. ``['eeb', 'ees']``).

    Notes
    -----
    For downloading many years at once with thread-pool parallelism, see
    :func:`download_cdf_files_threaded`.
    """
    for month_index in range(1, 13):
        web_folder = str(month_index).zfill(2)
        print(f"STATUS: Loading month listing pages for {year}-{web_folder}")
        page_file_names: dict[str, list[str]] = {
            instrument: _list_instrument_page_files(f"{base_url}/{instrument}/{year}/{web_folder}")
            for instrument in instruments
        }
        days_in_month = calendar.monthrange(year, month_index)[1]
        print(f"{month_index}/12 | downloading files for {year}-{web_folder}")
        for day_index in tqdm(range(1, days_in_month + 1)):
            download_single_day_cdf(
                date=dt.date(year, month_index, day_index),
                instruments=instruments,
                base_url=base_url,
                data_folder=data_folder,
                _page_file_names=page_file_names,
            )


def _discover_download_targets(
    base_url: str, years: list[int], instruments: set[str], data_folder: str
) -> list[tuple[str, str]]:
    """Scrape CDA Web listings for every (year, month, instrument) combination.

    Returns
    -------
    list of tuple
        ``(download_link, output_file)`` pairs for files not already present
        on disk.
    """
    targets: list[tuple[str, str]] = []
    for year in years:
        for month_index in range(1, 13):
            web_folder = str(month_index).zfill(2)
            for instrument in instruments:
                fast_data_folder = f"{data_folder}/{year}/{web_folder}"
                os.makedirs(fast_data_folder, exist_ok=True)
                page = f"{base_url}/{instrument}/{year}/{web_folder}"
                try:
                    response = requests.get(page)
                except requests.RequestException as exc:
                    log_error(f"[DOWNLOAD] Failed to load listing page {page}: {exc}")
                    continue
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all("a"):
                    file_name = link.get("href")
                    if file_name and ".cdf" in file_name:
                        output_file = f"{fast_data_folder}/{file_name}"
                        if not os.path.exists(output_file):
                            targets.append((f"{page}/{file_name}", output_file))
    return targets


def download_cdf_files_threaded(
    base_url: str = FAST_ESA_BASE_URL,
    years: list[int] | None = None,
    data_folder: str = DEFAULT_FOLDER,
    instruments: set[str] = INSTRUMENT_OPTIONS,
    max_workers: int = 8,
    progress_json_path: str | None = None,
    ignore_progress_json: bool = False,
    flush_batch_size: int = 25,
) -> list[tuple[tuple[str, str], str]]:
    """Download many years of FAST ESA CDF files in parallel using a thread pool.

    Listing pages are scraped sequentially first (cheap -- one small HTML
    page per year/month/instrument combination), then every individual file
    download is dispatched to a ``ThreadPoolExecutor`` via
    :func:`configurable_spectrograms.batch_runner.run_batch`: downloading is
    I/O-bound, so thread-level concurrency is used here instead of the
    process-level concurrency the plotting batch drivers use for their
    CPU-bound rendering work.

    Parameters
    ----------
    base_url : str, default FAST_ESA_BASE_URL
        Base CDA Web URL for FAST ESA level-2 data.
    years : list of int or None, optional
        Calendar years to download; defaults to ``[DEFAULT_YEAR]`` when
        ``None``.
    data_folder : str, default DEFAULT_FOLDER
        Root output directory; files are saved under
        ``{data_folder}/{year}/{month}/``.
    instruments : set of str, default INSTRUMENT_OPTIONS
        Instrument codes to download.
    max_workers : int, default 8
        Number of download threads.
    progress_json_path : str or None, optional
        Path to a JSON file used for resumable progress tracking. ``None``
        disables persistence.
    ignore_progress_json : bool, default False
        If ``True``, skip reading existing progress prior to execution.
    flush_batch_size : int, default 25
        Progress/log batch size passed through to ``run_batch``.

    Returns
    -------
    list of tuple
        Sequence of ``((download_link, output_file), status)`` results,
        where ``status`` is ``'ok'`` or ``'error'``.
    """
    resolved_years = years if years is not None else [DEFAULT_YEAR]
    targets = _discover_download_targets(base_url, resolved_years, instruments, data_folder)

    def _worker(target: tuple[str, str]) -> tuple[tuple[str, str], str]:
        download_link, output_file = target
        try:
            _download_single_cdf_file(download_link, output_file)
            return (target, "ok")
        except Exception as exc:
            log_error(f"[DOWNLOAD-FAIL] {download_link}: {exc}")
            return (target, "error")

    return run_batch(
        targets,
        _worker,
        functools.partial(ThreadPoolExecutor, max_workers=max_workers),
        progress_json_path=progress_json_path,
        ignore_progress_json=ignore_progress_json,
        flush_batch_size=flush_batch_size,
    )
