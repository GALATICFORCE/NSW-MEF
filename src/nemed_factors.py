"""Monthly emission factors per DUID, obtained the same way as in nemed.get_marginal_emissions"""
import pandas as pd

import aemo_archive_fix
aemo_archive_fix.apply()

import nemed.process as nemed_process
from nemed.downloader import download_genset_map

CACHE = "data/raw/nemed_cache"
GENSET_ASOF = "2026/07/01 00:00"   # same as fetch_mef.py


def monthly_factors(first, last, genset_asof=GENSET_ASOF):
    """Returns file_year, file_month, DUID, CO2E_EMISSIONS_FACTOR ...

    Calls nemed's function month by month, as fetch_mef.py does. With genset_asof=None, nemed's default asof_date is used.
    """
    original = nemed_process.download_genset_map
    nemed_process.download_genset_map = (
        lambda cache, asof_date=None: download_genset_map(cache, asof_date=genset_asof))
    try:
        out = []
        for m in pd.period_range(first, last, freq="M"):
            s, e = m.start_time, (m + 1).start_time
            out.append(nemed_process._get_duid_emissions_intensities(
                s.strftime("%Y/%m/%d %H:%M"), e.strftime("%Y/%m/%d %H:%M"), CACHE))
    finally:
        nemed_process.download_genset_map = original
    return pd.concat(out).drop_duplicates(["file_year", "file_month", "DUID"])
