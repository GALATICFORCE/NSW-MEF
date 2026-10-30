"""AEI trial run (hand-off doc §7b): use nemed to get one day of NSW1 5-minute total emissions and intensity,
and check 3 intervals by hand

Only NSW1 in-region generation; emission factors are the same set as for MEF (GENUNITS + DUALLOC, as of 2026-07-01).
Usage: python src\\aei_trial.py
"""
import time
from pathlib import Path

import pandas as pd

from nemed_factors import GENSET_ASOF   # importing it already runs aemo_archive_fix.apply()

import nemed
import nemed.process as nemed_process
from nemed.downloader import download_dudetailsummary, download_genset_map, read_plant_auxload_csv
from nemosis import dynamic_data_compiler

CACHE = "data/aei_cache"
DAY = pd.Timestamp("2025-07-15")
CHECK_ENDS = [pd.Timestamp("2025-07-15 04:00"), pd.Timestamp("2025-07-15 12:00"), pd.Timestamp("2025-07-15 19:00")]
REGION = "NSW1"
OUT_CSV = "data/processed/aei_trial_nsw1_5min_2025-07-15.csv"
REPORT = "outputs/aei_trial_2025-07-15.txt"
FMT = "%Y/%m/%d %H:%M"

# Fix asof as for MEF, so nemed's default ("90 days before today") does not drift with the run date
nemed_process.download_genset_map = lambda cache, asof_date=None: download_genset_map(cache, asof_date=GENSET_ASOF)
nemed_process.download_dudetailsummary = (
    lambda cache, asof_date=None: download_dudetailsummary(cache, asof_date=GENSET_ASOF))

lines = []


def note(text=""):
    print(text)
    lines.append(str(text))


def run_nemed():
    # Start at 23:55 the day before: nemed computes energy from the mean of two consecutive SCADA readings,
    # so the first interval in the window has no previous reading
    start, end = DAY - pd.Timedelta("5min"), DAY + pd.Timedelta(days=1)
    res = nemed.get_total_emissions(start.strftime(FMT), end.strftime(FMT), CACHE, filter_regions=[REGION])
    return res[(res["TimeEnding"] > DAY) & (res["TimeEnding"] <= end)].reset_index(drop=True)


def hand_check(nemed_res):
    """Recompute directly from SCADA, DUDETAILSUMMARY, emission factors and the auxiliary-load table,
    bypassing nemed's aggregation code"""
    scada = dynamic_data_compiler((DAY - pd.Timedelta("5min")).strftime("%Y/%m/%d %H:%M:%S"),
                                  (DAY + pd.Timedelta(days=1, minutes=5)).strftime("%Y/%m/%d %H:%M:%S"),
                                  "DISPATCH_UNIT_SCADA", CACHE,
                                  select_columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE"], fformat="feather")
    scada = scada.drop_duplicates(["SETTLEMENTDATE", "DUID"])
    scada["MW"] = scada["SCADAVALUE"].clip(lower=0)
    info = download_dudetailsummary(CACHE, asof_date=GENSET_ASOF)
    nsw_gen = set(info.loc[(info["REGIONID"] == REGION) & (info["DISPATCHTYPE"] == "GENERATOR"), "DUID"])
    factors = nemed_process._get_duid_emissions_intensities(
        (DAY - pd.Timedelta("5min")).strftime(FMT), (DAY + pd.Timedelta(days=1)).strftime(FMT), CACHE)
    ef = factors[(factors["file_year"] == DAY.year) & (factors["file_month"] == DAY.month)].set_index("DUID")[
        "CO2E_EMISSIONS_FACTOR"]
    aux = read_plant_auxload_csv().drop_duplicates("DUID").set_index("DUID")["PCT_AUXILIARY_LOAD"]

    note("\n== hand check (3 intervals)")
    note("nemed convention: energy = mean(SCADA at SETTLEMENTDATE T, T+5) x 5/60, x (1 - aux%), "
         "NSW1 GENERATOR units with an emissions factor only")
    note("simple: sum(MW x EF) / sum(MW) using SCADA at SETTLEMENTDATE T, same units, no ramp, no aux")
    for t in CHECK_ENDS:
        a = scada[scada["SETTLEMENTDATE"] == t].set_index("DUID")["MW"]
        b = scada[scada["SETTLEMENTDATE"] == t + pd.Timedelta("5min")].set_index("DUID")["MW"]
        df = pd.DataFrame({"a": a, "b": b})
        df = df[df.index.isin(nsw_gen)]
        df["ef"] = ef.reindex(df.index)
        df["so"] = 1 - aux.reindex(df.index).fillna(0) / 100
        no_ef = df[df["ef"].isna()]
        df = df[df["ef"].notna()]
        energy = (0.5 * (df["a"] + df["b"]) * 5 / 60 * df["so"])
        aei_hand = (energy * df["ef"]).sum() / energy.sum()
        aei_simple = (df["a"] * df["ef"]).sum() / df["a"].sum()
        row = nemed_res[nemed_res["TimeEnding"] == t].iloc[0]
        note(f"TimeEnding {t:%Y-%m-%d %H:%M}: nemed {row['Intensity_Index']:.6f} | hand (nemed convention) "
             f"{aei_hand:.6f}, diff {aei_hand - row['Intensity_Index']:+.2e} | simple {aei_simple:.6f}, "
             f"diff {aei_simple - row['Intensity_Index']:+.4f} ({(aei_simple / row['Intensity_Index'] - 1) * 100:+.2f}%)")
        note(f"    energy: nemed {row['Energy']:.3f} MWh, hand {energy.sum():.3f} MWh | units: {len(df)} with EF, "
             f"{len(no_ef)} NSW1 generators without EF: "
             + (", ".join(f"{d} {m:.1f} MW" for d, m in no_ef["b"].fillna(0).items()) or "none"))
    return info, ef, nsw_gen


if __name__ == "__main__":
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_nemed()
    t_first = time.time() - t0
    t0 = time.time()
    res = run_nemed()
    t_cached = time.time() - t0
    res.to_csv(OUT_CSV, index=False)

    size_mb = sum(f.stat().st_size for f in Path(CACHE).rglob("*") if f.is_file()) / 1e6
    note(f"== nemed.get_total_emissions, {REGION}, {DAY:%Y-%m-%d}")
    note(f"runtime: first call (with downloads) {t_first:.0f} s, second call (cached) {t_cached:.0f} s")
    note(f"cache {CACHE}: {size_mb:.0f} MB")
    for f in sorted(Path(CACHE).glob("*")):
        note(f"    {f.name}  {f.stat().st_size / 1e6:.1f} MB")
    note(f"rows: {len(res)} | TimeEnding {res['TimeEnding'].min()} -> {res['TimeEnding'].max()}")
    note(f"columns: {list(res.columns)} | dtypes: {dict(res.dtypes.astype(str))}")
    note(f"NaN per column: {res.isna().sum().to_dict()} | Intensity_Index == 0: {(res['Intensity_Index'] == 0).sum()}")
    note(f"Intensity_Index (tCO2e/MWh): min {res['Intensity_Index'].min():.3f}, median "
         f"{res['Intensity_Index'].median():.3f}, max {res['Intensity_Index'].max():.3f} | "
         f"day energy {res['Energy'].sum():.0f} MWh, emissions {res['Total_Emissions'].sum():.0f} t")

    info, ef, nsw_gen = hand_check(res)

    # When aggregating, nemed picks the energy column with columns.str.contains('Energy')[0], which selects as-generated Energy,
    # while Total_Emissions is computed on sent-out Energy_SO, so the numerator and denominator of Intensity_Index differ in basis
    start, end = DAY - pd.Timedelta("5min"), DAY + pd.Timedelta(days=1)
    duid = nemed_process.get_total_emissions_by_DI_DUID(
        start.strftime(FMT), end.strftime(FMT), CACHE, filter_regions=[REGION], return_all=True)
    duid = duid.drop_duplicates(["Time", "DUID"])
    duid = duid[(duid["Time"] > DAY) & (duid["Time"] <= end)]
    g = duid.assign(em_gen=duid["Energy"] * duid["Plant_Emissions_Intensity"]).groupby("Time")
    sent_out = g["Total_Emissions"].sum() / g["Energy_SO"].sum()
    as_gen = g["em_gen"].sum() / g["Energy"].sum()
    nemed_ii = res.set_index("TimeEnding")["Intensity_Index"]
    note("\n== all 288 intervals: nemed Intensity_Index vs consistent recomputation from nemed's own unit-level table")
    for name, s in [("sent-out (emissions / Energy_SO)", sent_out), ("as-generated (Energy x EF / Energy)", as_gen)]:
        ratio = nemed_ii / s.reindex(nemed_ii.index)
        note(f"{name}: nemed / this = min {ratio.min():.4f}, median {ratio.median():.4f}, max {ratio.max():.4f}")
    so_share = g["Energy_SO"].sum() / g["Energy"].sum()
    note(f"Energy_SO / Energy per interval: min {so_share.min():.4f}, median {so_share.median():.4f}, "
         f"max {so_share.max():.4f}")

    scada = dynamic_data_compiler(DAY.strftime("%Y/%m/%d %H:%M:%S"),
                                  (DAY + pd.Timedelta(days=1)).strftime("%Y/%m/%d %H:%M:%S"),
                                  "DISPATCH_UNIT_SCADA", CACHE,
                                  select_columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE"], fformat="feather")
    scada = scada[scada["DUID"].isin(nsw_gen)].drop_duplicates(["SETTLEMENTDATE", "DUID"])
    mwh = (scada["SCADAVALUE"].clip(lower=0) * 5 / 60).groupby(scada["DUID"].map(lambda d: pd.isna(ef.get(d)))).sum()
    note(f"NSW1 generator SCADA energy on the day (MWh, no ramp): with EF {mwh.get(False, 0):.0f}, "
         f"without EF {mwh.get(True, 0):.0f}")
    nsw_info = info[info["REGIONID"] == REGION]
    note("\n== NSW1 units in DUDETAILSUMMARY (as of " + GENSET_ASOF[:10] + ") by DISPATCHTYPE: "
         + str(nsw_info["DISPATCHTYPE"].value_counts().to_dict()))
    no_ef = sorted(d for d in nsw_gen if pd.isna(ef.get(d)))
    note(f"NSW1 GENERATOR units with no emissions factor in 2025-07 (dropped by nemed from energy AND emissions): {no_ef}")

    Path(REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {REPORT}, {OUT_CSV}")
