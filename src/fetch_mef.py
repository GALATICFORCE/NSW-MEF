import sys
import time
from pathlib import Path

import pandas as pd
import aemo_archive_fix
aemo_archive_fix.apply()

import nemed
import nemed.process as nemed_process
from nemed.downloader import download_genset_map

GENSET_ASOF = "2026/07/01 00:00"
CACHE = Path("data/raw/nemed_cache")
OUT_DIR = Path("data/raw/mef_5min")
CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def genset_map_fixed(cache, asof_date=None):
    return download_genset_map(cache, asof_date=GENSET_ASOF)

nemed_process.download_genset_map = genset_map_fixed


def fetch_month(month_start):
    month_end = month_start + pd.offsets.MonthBegin(1)
    out = OUT_DIR / f"mef_nsw1_5min_{month_start:%Y-%m}.csv"
    if out.exists():
        print("skip", out.name)
        return
    t0 = time.time()
    df = nemed.get_marginal_emissions(
        start_time=month_start.strftime("%Y/%m/%d %H:%M"),
        end_time=month_end.strftime("%Y/%m/%d %H:%M"),
        cache=str(CACHE),
    )
    nsw = df[df["Region"] == "NSW1"]
    nsw.to_csv(out, index=False)
    print(f"{month_start:%Y-%m}: {len(nsw)} rows, {time.time() - t0:.0f}s")

if __name__ == "__main__":
    first, last = sys.argv[1], sys.argv[2]   # 例如 2025-07 2025-07
    for m in pd.date_range(first, last, freq="MS"):
        fetch_month(m)
