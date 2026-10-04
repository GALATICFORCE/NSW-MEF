"""DUID 的月度排放因子，取法和 nemed.get_marginal_emissions 一致"""
import pandas as pd

import aemo_archive_fix
aemo_archive_fix.apply()

import nemed.process as nemed_process
from nemed.downloader import download_genset_map

CACHE = "data/raw/nemed_cache"
GENSET_ASOF = "2026/07/01 00:00"   # 和 fetch_mef.py 一致


def monthly_factors(first, last, genset_asof=GENSET_ASOF):
    """返回 file_year, file_month, DUID, CO2E_EMISSIONS_FACTOR ...

    和 fetch_mef.py 一样逐月调用 nemed 的函数。genset_asof=None 时用 nemed 默认的 asof_date。
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
