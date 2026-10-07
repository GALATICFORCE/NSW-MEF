"""5 分钟 MEF 标签，规则见 docs/label_rules_v1.md（tag label-rules-v1）

档位列（规则第 5 节）：1 / 2 / 3 = 实际用了第几档；0 = 三档都不足 N 个来源区间，对应的标签为 NaN；
空 = 不适用（该区间没有这一侧的电池记录，或已被第 7 节置为 NaN）。
"""
import numpy as np
import pandas as pd

FULL = "data/processed/mef_nsw1_5min_full.csv"
OUT = "data/processed/mef_5min_labels_rules-v1.csv"
LABELS = ("A", "B", "C", "Cfwd")
W_RANGE = (0.5, 1.5)
ETA = 0.85
N_MIN = 6
# 规则第 2 节：全年统一用 2026H1 这一套，NEM 小时 [start, end)
WINDOWS = [("night", "charge", 0, 5), ("morning", "discharge", 5, 7),
           ("day", "charge", 7, 16), ("evening", "discharge", 16, 24)]


def load_full(path=FULL):
    return pd.read_csv(path, parse_dates=["Time"])


def w_total(df):
    return df["w_enof"] + df["w_gen"] + df["w_load"]


def nan_reason(df):
    """规则第 7 节，四个标签共用。空字符串 = 不置 NaN；多个原因同时成立时取排在前面的"""
    w = w_total(df)
    reason = pd.Series("", index=df.index)
    # 从优先级最低的开始写，后写的覆盖先写的
    reason[(w < W_RANGE[0]) | (w > W_RANGE[1])] = "w_total"
    reason[df["ef_missing"]] = "ef_missing"
    reason[df["has_ldof"]] = "has_ldof"
    status_nan = df["status"].isin(["other", "norecord"])
    reason[status_nan] = "status_" + df.loc[status_nan, "status"]
    return reason


def nan_mask(df):
    return nan_reason(df) != ""


def is_source(df):
    """规则第 3 节：窗口值只从这些区间取"""
    return (df["status"] == "nonbatt") & ~nan_mask(df)


def segments(df):
    """给每个区间标上预测日、窗口和窗口段；返回 (每区间的段号, 段表)

    窗口段 = 连续的、预测日和窗口都相同的一段区间。夏令时期间 NEM 23 点属于下一个悉尼日期，
    所以那一天开头会多出一个单独的晚间放电段（规则第 1 节）。
    """
    start = df["Time"] - pd.Timedelta("5min")
    hour = start.dt.hour
    day = start.dt.tz_localize("Australia/Brisbane").dt.tz_convert("Australia/Sydney").dt.date
    window = pd.Series(pd.NA, index=df.index, dtype="object")
    kind = window.copy()
    for name, k, h0, h1 in WINDOWS:
        in_w = (hour >= h0) & (hour < h1)
        window[in_w], kind[in_w] = name, k
    assert window.notna().all()

    key = day.astype(str) + "|" + window
    seg = (key != key.shift()).cumsum()
    src = is_source(df)
    value = df["mef_nemed"].where(src)
    table = pd.DataFrame({"day": day, "kind": kind, "start": start, "src": src, "value": value}).groupby(seg).agg(
        day=("day", "first"), kind=("kind", "first"), start=("start", "min"),
        n_src=("src", "sum"), mean=("value", "mean"))

    # 配对用的段号（规则第 4 节）。所在窗口类型符合就用所在段，否则往前 / 往后找
    charge_id = table.index.to_series().where(table["kind"] == "charge")
    discharge_id = table.index.to_series().where(table["kind"] == "discharge")
    table["prev_charge_same_day"] = charge_id.groupby(table["day"]).ffill()
    table["next_discharge_same_day"] = discharge_id.groupby(table["day"]).bfill()
    table["next_charge_any_day"] = charge_id.bfill()
    return seg, table


def resolve(paired, day, kind, table):
    """规则第 5 节的三档兜底；paired = 配对段号（可为 NaN），day / kind = 第 2、3 档用的日期和窗口类型"""
    by_kind = table.assign(s=table["mean"] * table["n_src"]).groupby(["day", "kind"])[["n_src", "s"]].sum()
    by_day = by_kind.groupby(level="day").sum()

    t1 = table.reindex(paired.to_numpy())
    t2 = by_kind.reindex(pd.MultiIndex.from_arrays([day.to_numpy(), kind.to_numpy()]))
    t3 = by_day.reindex(day.to_numpy())
    ok1 = (t1["n_src"] >= N_MIN).to_numpy()
    ok2 = (t2["n_src"] >= N_MIN).to_numpy()
    ok3 = (t3["n_src"] >= N_MIN).to_numpy()
    value = np.select([ok1, ok2, ok3], [t1["mean"].to_numpy(), (t2["s"] / t2["n_src"]).to_numpy(),
                                        (t3["s"] / t3["n_src"]).to_numpy()], default=np.nan)
    tier = np.select([ok1, ok2, ok3], [1, 2, 3], default=0)
    return pd.Series(value, index=paired.index), pd.Series(tier, index=paired.index)


def battery_fills(df, name, eta):
    """返回 (fill_gen, fill_load, 档位列字典)：电池 GEN / LOAD 记录的反事实排放因子（规则第 6 节）"""
    seg, table = segments(df)
    own = table.reindex(seg.to_numpy()).set_index(df.index)
    has_gen = df["batt_dir"].isin(["discharge", "both"]) & ~nan_mask(df)
    has_load = df["batt_dir"].isin(["charge", "both"]) & ~nan_mask(df)

    def tier_col(tier, applies):
        return tier.where(applies).astype("Int64")

    if name == "B":
        own_seg = pd.Series(seg.to_numpy(), index=df.index, dtype=float)
        value, tier = resolve(own_seg, own["day"], own["kind"], table)
        return value, value, {"tier_B": tier_col(tier, has_gen | has_load)}

    charge = pd.Series("charge", index=df.index)
    discharge = pd.Series("discharge", index=df.index)
    # 充电记录：C 和 C-fwd 相同，同一预测日内所在或之后的放电段
    load_value, load_tier = resolve(own["next_discharge_same_day"], own["day"], discharge, table)
    if name == "C":
        gen_value, gen_tier = resolve(own["prev_charge_same_day"], own["day"], charge, table)
        tiers = {"tier_C_gen": tier_col(gen_tier, has_gen), "tier_C_load": tier_col(load_tier, has_load)}
    else:
        # C-fwd 放电记录：所在或之后的充电段，可以跨日；第 2、3 档用配对段那一天，没有配对段（数据最后一天）就用当天
        paired = own["next_charge_any_day"]
        paired_day = table["day"].reindex(paired.to_numpy()).set_axis(df.index)
        gen_value, gen_tier = resolve(paired, paired_day.fillna(own["day"]), charge, table)
        tiers = {"tier_Cfwd_gen": tier_col(gen_tier, has_gen)}
    return gen_value / eta, eta * load_value, tiers


def compute_label(df, name, eta=ETA):
    """返回 Time、label_<name> 和对应的档位列"""
    if name == "A":
        fill_gen = fill_load = pd.Series(0.0, index=df.index)
        tiers = {}
    elif name in LABELS:
        fill_gen, fill_load, tiers = battery_fills(df, name, eta)
    else:
        raise ValueError(f"unknown label {name!r}, expected one of {LABELS}")

    # 规则第 6 节：base 是 nemed 的非电池部分，batt 区间没有非电池部分；
    # 没有这一侧电池记录的区间，这一侧贡献为 0（不让 NaN 的填充值传进来）
    base = df["mef_nemed"].where(df["status"] != "batt", 0.0)
    gen_part = (df["w_gen"] * fill_gen).where(df["batt_dir"].isin(["discharge", "both"]), 0.0)
    load_part = (df["w_load"] * fill_load).where(df["batt_dir"].isin(["charge", "both"]), 0.0)
    label = base + gen_part + load_part
    out = pd.DataFrame({"Time": df["Time"], f"label_{name}": label.mask(nan_mask(df))})
    for col, tier in tiers.items():
        out[col] = tier
    return out


if __name__ == "__main__":
    df = load_full()
    out = df[["Time"]].copy()
    for name in LABELS:
        out = out.merge(compute_label(df, name), on="Time")
    out["status"] = df["status"]
    out["batt_dir"] = df["batt_dir"]
    out["w_total"] = w_total(df)
    out["nan_reason"] = nan_reason(df)
    out.to_csv(OUT, index=False)
    print(f"rows: {len(out)} | rule-based NaN intervals: {nan_mask(df).sum()} -> {OUT}")
    for name in LABELS:
        print(f"label_{name} NaN: {out[f'label_{name}'].isna().sum()}")
