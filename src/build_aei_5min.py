"""NSW1 full-year 5-minute AEI = total emissions ÷ sent-out energy (decided 10-11; nemed's Intensity_Index column is not used)

Calls nemed's internal _total_emissions_process month by month: the window starts at the last interval of the previous
month (so the first interval of the month has a previous SCADA reading), only the month's own intervals are kept, and
nothing is written to the cache as processed_co2_total_*.parquet.
(Checked: the 24 intervals 2025-07-31 23:05 → 2025-08-01 01:00 are identical to the outer get_total_emissions_by_DI_DUID.)

Only NSW1 in-region GENERATOR units (nemed default, excluding bidirectional units such as batteries); units with an empty
emission factor are left out of AEI, but their output is counted separately. The genset map and DUDETAILSUMMARY are both
fixed at GENSET_ASOF.

Usage: python src\\build_aei_5min.py
"""
import time
from pathlib import Path

import pandas as pd

from nemed_factors import GENSET_ASOF   # importing it already runs aemo_archive_fix.apply()

import nemed.process as nemed_process
from nemed.downloader import download_dudetailsummary, download_genset_map

CACHE = "data/aei_cache"
REGION = "NSW1"
OUT = "data/processed/aei_nsw1_5min_full.csv"
REPORT = "outputs/aei_build_v1.txt"
TRIAL = "data/processed/aei_trial_nsw1_5min_2025-07-15.csv"
FMT = "%Y/%m/%d %H:%M"

nemed_process.download_genset_map = lambda cache, asof_date=None: download_genset_map(cache, asof_date=GENSET_ASOF)
nemed_process.download_dudetailsummary = (
    lambda cache, asof_date=None: download_dudetailsummary(cache, asof_date=GENSET_ASOF))

lines = []


def note(text=""):
    print(text, flush=True)
    lines.append(str(text))


def month_units(month):
    """Unit-level table for one month (including units with an empty emission factor), Time = interval end"""
    lo, hi = month.start_time, (month + 1).start_time
    unit = nemed_process._total_emissions_process(
        (lo - pd.Timedelta("5min")).strftime(FMT), hi.strftime(FMT), CACHE,
        filter_regions=[REGION], dropna_co2factors=False)
    unit = unit.drop_duplicates(["Time", "DUID"])
    return unit[(unit["Time"] > lo) & (unit["Time"] <= hi)]


if __name__ == "__main__":
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    totals, noef = [], []
    for month in pd.period_range("2025-07", "2026-06", freq="M"):
        t0 = time.time()
        unit = month_units(month)
        has_ef = unit["Plant_Emissions_Intensity"].notna()
        totals.append(unit[has_ef].groupby("Time")[["Energy_SO", "Total_Emissions"]].sum())
        noef.append(unit.loc[~has_ef, ["Time", "DUID", "Energy_SO"]])
        note(f"{month}: {unit['Time'].nunique()} intervals, {unit['DUID'].nunique()} units "
             f"({unit.loc[~has_ef, 'DUID'].nunique()} without EF), {time.time() - t0:.0f} s")

    full_index = pd.date_range("2025-07-01 00:05", "2026-07-01 00:00", freq="5min", name="TimeEnding")
    aei = pd.concat(totals).rename_axis("TimeEnding").reindex(full_index)
    aei["aei"] = aei["Total_Emissions"] / aei["Energy_SO"]
    aei.reset_index().to_csv(OUT, index=False)

    note(f"\n== {OUT}")
    note(f"total runtime {time.time() - t_all:.0f} s")
    note(f"rows {len(aei)} | NaN per column {aei.isna().sum().to_dict()} | Energy_SO <= 0: {(aei['Energy_SO'] <= 0).sum()}")
    month_start = aei.index[(aei.index.day == 1) & (aei.index.hour == 0) & (aei.index.minute == 5)]
    note(f"first interval of each month (00:05) present with energy: "
         f"{int((aei.loc[month_start, 'Energy_SO'] > 0).sum())} of {len(month_start)}")
    note(f"aei (tCO2e/MWh): min {aei['aei'].min():.3f}, p1 {aei['aei'].quantile(0.01):.3f}, "
         f"median {aei['aei'].median():.3f}, p99 {aei['aei'].quantile(0.99):.3f}, max {aei['aei'].max():.3f}")

    # Regression check: total emissions on 2025-07-15 must match the nemed output of the d62f074 trial run, interval by interval
    trial = pd.read_csv(TRIAL, parse_dates=["TimeEnding"]).set_index("TimeEnding")["Total_Emissions"]
    diff = (aei["Total_Emissions"].reindex(trial.index) - trial).abs().max()
    note(f"regression vs trial 2025-07-15 Total_Emissions: max abs diff {diff:.2e} t over {len(trial)} intervals")

    # Units with an empty emission factor: number of intervals with output, and share of energy
    noef = pd.concat(noef)
    noef = noef[noef["Energy_SO"] > 0]
    all_energy = aei["Energy_SO"].sum() + noef["Energy_SO"].sum()
    note(f"\n== units without emissions factor (excluded from AEI)")
    note(f"intervals where any such unit has output: {noef['Time'].nunique()} of {len(aei)}; "
         f"their energy {noef['Energy_SO'].sum():.1f} MWh = {noef['Energy_SO'].sum() / all_energy:.5%} "
         f"of NSW1 generator sent-out energy")
    by_unit = noef.groupby("DUID").agg(intervals=("Time", "nunique"), energy_mwh=("Energy_SO", "sum"))
    by_unit["share"] = by_unit["energy_mwh"] / all_energy
    by_unit["months"] = noef.groupby("DUID")["Time"].agg(
        lambda s: ",".join(sorted((s - pd.Timedelta("5min")).dt.strftime("%Y-%m").unique())))
    note(by_unit.sort_values("energy_mwh", ascending=False).to_string() if len(by_unit) else "none")
    for d in ["CG2", "CG4"]:
        if d not in by_unit.index:
            note(f"{d}: no output in any interval")

    size_gb = sum(f.stat().st_size for f in Path(CACHE).rglob("*") if f.is_file()) / 1e9
    note(f"\ncache {CACHE}: {size_gb:.2f} GB "
         f"(SCADA csv {sum(f.stat().st_size for f in Path(CACHE).glob('*SCADA*.CSV')) / 1e9:.2f} GB, "
         f"feather {sum(f.stat().st_size for f in Path(CACHE).glob('*.feather')) / 1e9:.2f} GB)")
    Path(REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}, {REPORT}")
