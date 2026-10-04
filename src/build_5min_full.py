import glob
import json

import pandas as pd

from nemed_factors import CACHE, monthly_factors

MEF_DIR = "data/raw/mef_5min"
OUT = "data/processed/mef_nsw1_5min_full.csv"
START, END = pd.Timestamp("2025-07-01 00:05"), pd.Timestamp("2026-07-01 00:00")
BASSLINK = "T-V-MNSP1"
FCAS = r"^[LR](?:1SE|6SE|60S|5MI|5RE)(?:,GEN|,LOAD)?$"


def read_price_setters():
    """NSW1 能量市场的 price setter 记录，每条记录一行"""
    rows = []
    for day in pd.date_range(START.normalize(), END.normalize()):
        for r in json.load(open(f"{CACHE}/NEMED_PS_DAILY_{day:%Y-%m-%d}.json")):
            if r["@Market"] == "Energy" and r["@RegionID"] == "NSW1":
                rows.append((r["@PeriodID"], r["@Unit"], r["@DispatchedMarket"], float(r["@Increase"])))
    ps = pd.DataFrame(rows, columns=["PeriodID", "Unit", "DispatchedMarket", "Increase"])
    # PeriodID 形如 2025-07-01T00:45:00+10:00，即 NEM 时间的区间结束时刻
    ps["Time"] = pd.to_datetime(ps["PeriodID"].str[:19])
    ps = ps[ps["Time"].between(START, END)].drop(columns="PeriodID")

    dm = ps["DispatchedMarket"]
    kind = pd.Series("other", index=ps.index)   # GenericConstraintSurplus、DROF 等
    kind[dm == "ENOF"] = "enof"
    kind[dm == "BDOF,GEN"] = "gen"
    kind[dm == "BDOF,LOAD"] = "load"
    kind[dm == "LDOF"] = "ldof"
    kind[dm.str.match(FCAS)] = "fcas"
    kind[dm.str.startswith("tbslack")] = "tbslack"
    ps["kind"] = kind
    return ps


def flag_ef_missing(ps):
    """ENOF 定价机组查不到排放因子的记录。和 nemed 一样去掉 Basslink，按 Time（区间结束）所在月份匹配"""
    factors = monthly_factors("2025-07", "2026-06")
    enof = ps[(ps["kind"] == "enof") & ~ps["Unit"].str.contains(BASSLINK)].copy()
    enof["file_year"], enof["file_month"] = enof["Time"].dt.year, enof["Time"].dt.month
    enof = enof.merge(factors[["file_year", "file_month", "DUID", "CO2E_EMISSIONS_FACTOR"]],
                      left_on=["Unit", "file_year", "file_month"], right_on=["DUID", "file_year", "file_month"],
                      how="left")
    return set(enof.loc[enof["CO2E_EMISSIONS_FACTOR"].isna(), "Time"])


def summarise(ps):
    def w(k):
        return ps["Increase"].where(ps["kind"] == k, 0.0)

    ps = ps.assign(w_enof=w("enof"), w_gen=w("gen"), w_load=w("load"), w_ldof=w("ldof"), w_fcas=w("fcas"),
                   is_tbslack=ps["kind"] == "tbslack",
                   neg_batt=ps["kind"].isin(["gen", "load"]) & (ps["Increase"] < 0))
    for k in ["enof", "gen", "load", "ldof"]:
        ps[f"has_{k}"] = ps["kind"] == k
    g = ps.groupby("Time")
    out = g[["w_enof", "w_gen", "w_load", "w_ldof", "w_fcas"]].sum()
    out["n_tbslack"] = g["is_tbslack"].sum()
    out["n_neg_batt"] = g["neg_batt"].sum()
    out[["has_enof", "has_gen", "has_load", "has_ldof"]] = g[["has_enof", "has_gen", "has_load", "has_ldof"]].any()
    out["dispatch_types"] = g["DispatchedMarket"].agg(lambda s: "|".join(sorted(s.unique())))
    return out


def read_nemed_mef():
    df = pd.concat(pd.read_csv(f, parse_dates=["Time"]) for f in sorted(glob.glob(f"{MEF_DIR}/*.csv")))
    assert not df["Time"].duplicated().any()
    return df.set_index("Time")["Intensity_Index"].rename("mef_nemed")


ps = read_price_setters()
full_index = pd.date_range(START, END, freq="5min", name="Time")
df = summarise(ps).reindex(full_index)
num = ["w_enof", "w_gen", "w_load", "w_ldof", "w_fcas", "n_tbslack", "n_neg_batt"]
df[num] = df[num].fillna(0)
df[["n_tbslack", "n_neg_batt"]] = df[["n_tbslack", "n_neg_batt"]].astype(int)
flags = ["has_enof", "has_gen", "has_load", "has_ldof"]
df[flags] = df[flags].fillna(False).astype(bool)
df["dispatch_types"] = df["dispatch_types"].fillna("")
df["mef_nemed"] = read_nemed_mef().reindex(full_index)
df["ef_missing"] = df.index.isin(flag_ef_missing(ps))

# status 按顺序判断，先命中的优先
has_bdof = df["has_gen"] | df["has_load"]
df["status"] = "other"
df.loc[df["has_enof"] & ~has_bdof & df["mef_nemed"].notna(), "status"] = "nonbatt"
df.loc[has_bdof & df["has_enof"], "status"] = "mixed"
df.loc[has_bdof & ~df["has_enof"], "status"] = "batt"
df.loc[df["dispatch_types"] == "", "status"] = "norecord"

batt_dir = pd.Series(pd.NA, index=df.index, dtype="object")
batt_dir[df["has_load"] & ~df["has_gen"]] = "charge"
batt_dir[df["has_gen"] & ~df["has_load"]] = "discharge"
batt_dir[df["has_gen"] & df["has_load"]] = "both"
df["batt_dir"] = batt_dir.where(df["status"].isin(["batt", "mixed"]))

out = df.reset_index()[["Time", "mef_nemed", "status", "batt_dir", "w_enof", "w_gen", "w_load", "w_ldof",
                        "w_fcas", "n_tbslack", "n_neg_batt", "has_ldof", "ef_missing", "dispatch_types"]]
assert len(out) == 105_120 and out["Time"].is_unique
out.to_csv(OUT, index=False)
print("rows:", len(out), "->", OUT)
