"""5 分钟标签 → 小时表，格式是 MEF / AEI / 电价三方共用的小时表格式

用法：python src\\build_hourly_mef.py --label A --rules label-rules-v1
"""
import argparse

import pandas as pd

from labels import LABELS, compute_label, load_full

MIN_INTERVALS = 6
FIRST_HOUR, LAST_HOUR = pd.Timestamp("2025-07-01 00:00"), pd.Timestamp("2026-06-30 23:00")
FMT = "%Y-%m-%d %H:%M"


def build_hourly(labels, name):
    # 1. Time 是区间结束时刻：减 5 分钟变成区间开始，再向下取整到整点（NEM 时间）
    hour = (labels["Time"] - pd.Timedelta("5min")).dt.floor("H")

    # 2. 每小时取有效区间的均值，n_intervals = 有效区间数
    g = labels[f"label_{name}"].groupby(hour)
    hourly = pd.DataFrame({"value": g.mean(), "n_intervals": g.count()})
    hourly = hourly.reindex(pd.date_range(FIRST_HOUR, LAST_HOUR, freq="H"))
    hourly["n_intervals"] = hourly["n_intervals"].fillna(0).astype(int)

    # 3. 有效区间不足的小时设为缺失
    hourly.loc[hourly["n_intervals"] < MIN_INTERVALS, "value"] = float("nan")

    # 4. NEM 时间 = AEST（UTC+10，无夏令时），和布里斯班相同；悉尼有夏令时
    nem = hourly.index
    sydney = nem.tz_localize("Australia/Brisbane").tz_convert("Australia/Sydney").tz_localize(None)
    return pd.DataFrame({
        "hour_start_nem": nem.strftime(FMT),
        "hour_start_sydney": sydney.strftime(FMT),
        "sydney_date": sydney.strftime("%Y-%m-%d"),
        "value": hourly["value"].to_numpy(),
        "n_intervals": hourly["n_intervals"].to_numpy(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--rules", default="label-rules-v1", choices=["label-rules-v1"])
    args = parser.parse_args()

    out = build_hourly(compute_label(load_full(), args.label), args.label)
    path = f"data/processed/mef_hourly_{args.label}_{args.rules.removeprefix('label-')}.csv"
    out.to_csv(path, index=False)
    print(f"hours: {len(out)} | NaN hours: {out['value'].isna().sum()} -> {path}")
