import json

import pandas as pd

from nemed_factors import CACHE, GENSET_ASOF, monthly_factors
from nemed.downloader import download_genset_map, download_plant_emissions_factors

SRC = "data/processed/mef_nsw1_5min_full.csv"
REPORT_DIR = "data/processed"

df = pd.read_csv(SRC, parse_dates=["Time"])
# Assign hour, day and month by interval start time
start = df["Time"] - pd.Timedelta("5min")
df["month"] = start.dt.to_period("M")
df["hour"] = start.dt.hour
df["half"] = start.dt.year.map({2025: "2025H2", 2026: "2026H1"})

print("==== 1. status counts")
print(df["status"].value_counts().to_string())

print("\n==== 2. batt_dir")
print(pd.crosstab(df["status"], df["batt_dir"]).to_string())

print("\n==== 3. has_ldof / ef_missing × status")
print(pd.crosstab(df["has_ldof"], df["status"], margins=True).to_string())
print(pd.crosstab(df["ef_missing"], df["status"], margins=True).to_string())
print("has_ldof and ef_missing:", (df["has_ldof"] & df["ef_missing"]).sum())

print("\n==== 4. where ef_missing comes from")
rows = []
for day in pd.date_range("2025-07-01", "2026-07-01"):
    for r in json.load(open(f"{CACHE}/NEMED_PS_DAILY_{day:%Y-%m-%d}.json")):
        if r["@Market"] == "Energy" and r["@RegionID"] == "NSW1" and r["@DispatchedMarket"] == "ENOF":
            rows.append((r["@PeriodID"], r["@Unit"]))
enof = pd.DataFrame(rows, columns=["PeriodID", "DUID"])
enof["Time"] = pd.to_datetime(enof["PeriodID"].str[:19])
enof = enof[enof["Time"].isin(df.loc[df["ef_missing"], "Time"]) & ~enof["DUID"].str.contains("T-V-MNSP1")]
enof["file_year"], enof["file_month"] = enof["Time"].dt.year, enof["Time"].dt.month

fixed = monthly_factors("2025-07", "2026-06")
default = monthly_factors("2025-07", "2026-06", genset_asof=None)
key = ["DUID", "file_year", "file_month"]
miss = enof.merge(fixed[key + ["CO2E_EMISSIONS_FACTOR"]], on=key, how="left")
miss = miss[miss["CO2E_EMISSIONS_FACTOR"].isna()].drop(columns="CO2E_EMISSIONS_FACTOR")
miss = miss.merge(default[key + ["CO2E_EMISSIONS_FACTOR"]].rename(columns={"CO2E_EMISSIONS_FACTOR": "ef_default_asof"}),
                  on=key, how="left")

map_fixed = download_genset_map(CACHE, asof_date=GENSET_ASOF)
map_default = download_genset_map(CACHE)
genunits = download_plant_emissions_factors("2025/07/01 00:00", "2026/07/01 00:00", CACHE)
for duid, sub in miss.groupby("DUID"):
    months = sorted({f"{y}-{m:02d}" for y, m in zip(sub["file_year"], sub["file_month"])})
    gs_fixed = map_fixed.loc[map_fixed["DUID"] == duid, "GENSETID"].tolist()
    gs_default = map_default.loc[map_default["DUID"] == duid, "GENSETID"].tolist()
    gu = genunits[genunits["GENSETID"].isin(set(gs_fixed + gs_default) | {duid})]
    gu = gu[[f"{y}-{m:02d}" in months for y, m in zip(gu["file_year"], gu["file_month"])]]
    print(f"{duid}: {sub['Time'].nunique()} intervals, months {months}")
    print(f"   DUALLOC GENSETID  asof {GENSET_ASOF[:7]}: {gs_fixed} | nemed default asof: {gs_default}")
    if gu.empty:
        print("   GENUNITS: no matching GENSETID in these months")
    else:
        print("   GENUNITS:", gu.groupby("GENSETID")["CO2E_EMISSIONS_FACTOR"]
              .agg(lambda s: f"{s.notna().sum()}/{len(s)} months with a factor").to_dict())
    print(f"   intervals with a factor under nemed's default asof: {sub.loc[sub['ef_default_asof'].notna(), 'Time'].nunique()}")

print("\n==== 5. mef_nemed full-year max and min")
for name, idx in [("max", df["mef_nemed"].idxmax()), ("min", df["mef_nemed"].idxmin())]:
    r = df.loc[idx]
    print(f"{name}: {r['mef_nemed']:.4f} at {r['Time']} | status {r['status']} | w_enof {r['w_enof']:.4f} "
          f"| n_tbslack {r['n_tbslack']} | {r['dispatch_types']}")
print("mef_nemed quantiles:", df["mef_nemed"].quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(4).to_dict())

print("\n==== 6. w_enof + w_gen + w_load, by status")
w = df["w_enof"] + df["w_gen"] + df["w_load"]
print(w.groupby(df["status"]).quantile([0.01, 0.5, 0.99]).unstack().round(4).to_string())

print("\n==== 8 (new definition) hour-of-day × status × batt_dir")
df["cell"] = df["status"] + df["batt_dir"].map(lambda d: f"_{d}" if isinstance(d, str) else "")
order = ["nonbatt", "batt_charge", "batt_discharge", "batt_both",
         "mixed_charge", "mixed_discharge", "mixed_both", "other", "norecord"]
for half, sub in df.groupby("half"):
    counts = pd.crosstab(sub["hour"], sub["cell"]).reindex(columns=order, fill_value=0)
    denom = counts.sum(axis=1)
    counts.to_csv(f"{REPORT_DIR}/check8_counts_{half}.csv")
    print(f"\n-- {half} counts (denominator per hour = {denom.unique().tolist()} intervals = days × 12)")
    print(counts.to_string())
    b = counts[[c for c in order if c.startswith(("batt", "mixed"))]]
    batt_only = counts[["batt_charge", "batt_discharge", "batt_both"]]
    share = pd.DataFrame({
        "(batt+mixed) / all": b.sum(axis=1) / denom,
        "charge / (batt+mixed)": (b["batt_charge"] + b["mixed_charge"]) / b.sum(axis=1),
        "discharge / (batt+mixed)": (b["batt_discharge"] + b["mixed_discharge"]) / b.sum(axis=1),
        "both / (batt+mixed)": (b["batt_both"] + b["mixed_both"]) / b.sum(axis=1),
        "charge / batt": batt_only["batt_charge"] / batt_only.sum(axis=1),
    }).round(3)
    share.to_csv(f"{REPORT_DIR}/check8_share_{half}.csv")
    print(f"\n-- {half} shares (denominator in the column name)")
    print(share.to_string())
