"""两个 naive 基线（交接文档 §4.3），只在 train / val 夜上计算和评估

- yesterday（B1）：D-1 的同一悉尼小时。对 D+1 00:00-07:00 的小时，"昨天"也是 D-1（48 小时前），见 §3.3
- hourmean（B2）：训练夜里每个悉尼小时的 C 标签均值

用法：python src\\baselines.py
"""
import pandas as pd

from metrics import evaluate
from nights import load_hourly, night_hours, split_nights

FULL = "data/processed/mef_nsw1_5min_full.csv"
LABEL = "C"
OUT = "outputs/metrics_{split}_{model}-v2.csv"
SPLITS = ("train", "val")


def hour_status_counts():
    """NEM 小时 -> 12 个 5 分钟里 status = nonbatt 的个数、status ∈ {batt, mixed} 的个数"""
    full = pd.read_csv(FULL, usecols=["Time", "status"], parse_dates=["Time"])
    hour = (full["Time"] - pd.Timedelta("5min")).dt.floor("H")
    return pd.DataFrame({
        "n_nonbatt": (full["status"] == "nonbatt").groupby(hour).sum(),
        "n_battfill": full["status"].isin(["batt", "mixed"]).groupby(hour).sum(),
    })


def yesterday(df, hourly):
    """B1。D-1 的这个悉尼小时出现两次（夏令时结束）取均值；不存在（夏令时开始）改用 D-2 的同一小时。

    返回 (预测, 用了 D-2 的行, 用了"两次取均值"的行)
    """
    key_cols = [hourly["sydney_date"], hourly["hour_start_sydney"].dt.hour]
    lut = hourly.groupby(key_cols)["value"].mean()
    n_rows = hourly.groupby(key_cols).size()

    src_date = df["night_of"] - pd.Timedelta(days=1)
    hour = df["hour_start_sydney"].dt.hour
    # 防泄漏：只用 D-1（悉尼日期）的数据，整天都早于预测发出时刻 D 00:00
    assert (src_date + pd.Timedelta(days=1) <= df["night_of"]).all()
    key = pd.MultiIndex.from_arrays([src_date, hour])
    pred = pd.Series(lut.reindex(key).to_numpy(), index=df.index)
    doubled = pd.Series(n_rows.reindex(key).to_numpy() == 2, index=df.index)
    fallback = pred.isna()
    if fallback.any():
        d2 = pd.MultiIndex.from_arrays([src_date[fallback] - pd.Timedelta(days=1), hour[fallback]])
        pred[fallback] = lut.reindex(d2).to_numpy()
    assert pred.notna().all()
    return pred, fallback, doubled


def hourmean(df):
    """B2：只用训练夜"""
    train = df[df["split"] == "train"]
    lut = train.groupby(train["hour_start_sydney"].dt.hour)["truth"].mean()
    return df["hour_start_sydney"].dt.hour.map(lut)


if __name__ == "__main__":
    df = night_hours(LABEL, SPLITS).rename(columns={"value": "truth"})
    assert set(df["split"]) == set(SPLITS), "test nights must not be loaded before Step 7"
    counts = hour_status_counts()
    df["nonbatt_majority"] = df["hour_start_nem"].map(counts["n_nonbatt"]) >= 7
    df["n_battfill"] = df["hour_start_nem"].map(counts["n_battfill"])

    # B1 只需要测试期之前的小时
    first_test = split_nights(load_hourly(LABEL)).query("split == 'test'")["night_of"].min()
    hourly = load_hourly(LABEL)
    hourly = hourly[hourly["sydney_date"] < first_test]
    pred_b1, fallback, doubled = yesterday(df, hourly)
    preds = {"yesterday": pred_b1, "hourmean": hourmean(df)}

    for split in SPLITS:
        sub = df[df["split"] == split]
        print(f"{split}: {sub['night_of'].nunique()} nights, {len(sub)} hours, "
              f"truth NaN hours dropped: {int(sub['truth'].isna().sum())}, "
              f"nonbatt-majority hours: {int(sub['nonbatt_majority'].sum())}")
    print(f"yesterday: D-2 fallback {int(fallback.sum())} "
          f"{df.loc[fallback, 'hour_start_sydney'].dt.strftime('%Y-%m-%d %H:%M').tolist()}, "
          f"two-hour mean {int(doubled.sum())} "
          f"{df.loc[doubled, 'hour_start_sydney'].dt.strftime('%Y-%m-%d %H:%M').tolist()}")

    pd.set_option("display.width", 200)
    for model, pred in preds.items():
        for split in SPLITS:
            sub = df[df["split"] == split].assign(pred=pred[df["split"] == split])
            summary, nightly = evaluate(sub)
            summary.to_csv(OUT.format(split=split, model=model), index=False)
            share = nightly.drop_duplicates("night_of")["batt_share"]
            halves = summary.query("month == 'all' and group.str.startswith('batt_share')", engine="python")
            print(f"\n== {model} | {split} | battery-share median {share.median():.3f}, nights per half "
                  f"{dict(zip(halves['group'], halves['n_nights']))}")
            print(summary.round(3).to_string(index=False))
