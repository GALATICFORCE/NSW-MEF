"""5 分钟 MEF 标签，规则见 docs/label_rules_v1.md（tag label-rules-v1）"""
import pandas as pd

FULL = "data/processed/mef_nsw1_5min_full.csv"
OUT = "data/processed/mef_5min_labels_rules-v1.csv"
LABELS = ("A", "B", "C", "Cfwd")
W_RANGE = (0.5, 1.5)


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


def compute_label(df, name):
    """返回 Time 和 label_<name>；B/C/Cfwd 实现时再加 fill_tier_<name> 列"""
    if name == "A":
        fill_gen = fill_load = pd.Series(0.0, index=df.index)
        tiers = {}
    elif name in LABELS:
        raise NotImplementedError(f"label {name} is not implemented yet")
    else:
        raise ValueError(f"unknown label {name!r}, expected one of {LABELS}")

    # 规则第 6 节：base 是 nemed 的非电池部分，batt 区间没有非电池部分
    base = df["mef_nemed"].where(df["status"] != "batt", 0.0)
    label = base + df["w_gen"] * fill_gen + df["w_load"] * fill_load
    out = pd.DataFrame({"Time": df["Time"], f"label_{name}": label.mask(nan_mask(df))})
    for col, tier in tiers.items():
        out[col] = tier
    return out


if __name__ == "__main__":
    df = load_full()
    out = df[["Time"]].copy()
    for name in ["A"]:   # B / C / Cfwd 实现后加进来
        label = compute_label(df, name)
        # 标签只能在规则规定的位置为 NaN
        assert label[f"label_{name}"].isna().equals(nan_mask(df)), f"label_{name}: unexpected NaN"
        out = out.merge(label, on="Time")
    out["status"] = df["status"]
    out["batt_dir"] = df["batt_dir"]
    out["w_total"] = w_total(df)
    out["nan_reason"] = nan_reason(df)
    out.to_csv(OUT, index=False)
    print(f"rows: {len(out)} | NaN intervals: {nan_mask(df).sum()} -> {OUT}")
