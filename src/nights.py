"""夜的定义和切分（交接文档 §4.2）

night_of = D：悉尼本地 D 18:00 → D+1 07:00，正常 13 小时；2025-10-04 夜 12 小时，2026-04-04 夜 14 小时。
按夜、按时间顺序切分，边界是固定日期（10-11 定，和已提交的 outputs\\nights_split_v1.csv 相同）。
以后加更长的滞后特征时只从 train 前端裁，val / test 的日期不动。

用法：python src\\nights.py   （核对切分和 outputs\\nights_split_v1.csv 逐行相同；不改写该文件）
"""
import pandas as pd

HOURLY = "data/processed/mef_hourly_{label}_rules-v1.csv"
SPLIT_OUT = "outputs/nights_split_v1.csv"
NIGHT_START, NIGHT_END = 18, 7               # 悉尼本地小时
MAX_LAG_DAYS = 7                              # Step 4 的特征最远用到 D-7
FIRST_DAY = pd.Timestamp("2025-07-01")        # 第一个完整的悉尼日期
LAST_NIGHT = pd.Timestamp("2026-06-29")       # 2026-06-30 夜缺 07-01 00:00-07:00
FIRST_NIGHT = FIRST_DAY + pd.Timedelta(days=MAX_LAG_DAYS)
VAL_START = pd.Timestamp("2026-03-15")
TEST_START = pd.Timestamp("2026-05-08")


def load_hourly(label="C"):
    h = pd.read_csv(HOURLY.format(label=label))
    for col in ["hour_start_nem", "hour_start_sydney", "sydney_date"]:
        h[col] = pd.to_datetime(h[col])
    return h


def add_night(h):
    """加 night_of 列；不在 18:00-07:00 的小时为 NaT"""
    hour = h["hour_start_sydney"].dt.hour
    night = pd.Series(pd.NaT, index=h.index, dtype="datetime64[ns]")
    night[hour >= NIGHT_START] = h.loc[hour >= NIGHT_START, "sydney_date"]
    night[hour < NIGHT_END] = h.loc[hour < NIGHT_END, "sydney_date"] - pd.Timedelta(days=1)
    return h.assign(night_of=night)


def split_nights(h):
    """可用夜按固定日期边界切分；返回 night_of, split, n_hours"""
    nights = add_night(h).dropna(subset=["night_of"]).groupby("night_of").size().rename("n_hours")
    nights = nights[(nights.index >= FIRST_NIGHT) & (nights.index <= LAST_NIGHT)].reset_index()
    nights["split"] = "train"
    nights.loc[nights["night_of"] >= VAL_START, "split"] = "val"
    nights.loc[nights["night_of"] >= TEST_START, "split"] = "test"
    return nights[["night_of", "split", "n_hours"]]


def night_hours(label="C", splits=("train", "val")):
    """可用夜里的小时（默认只含 train 和 val，测试集在 Step 7 之前不参与任何计算）"""
    h = add_night(load_hourly(label))
    nights = split_nights(h)
    out = h.merge(nights[["night_of", "split"]], on="night_of")
    return out[out["split"].isin(splits)].reset_index(drop=True)


if __name__ == "__main__":
    nights = split_nights(load_hourly("C"))
    expected = {pd.Timestamp("2025-10-04"): 12, pd.Timestamp("2026-04-04"): 14}
    bad = nights[nights["n_hours"] != nights["night_of"].map(expected).fillna(13)]
    assert bad.empty, f"nights with unexpected hour counts:\n{bad}"
    committed = pd.read_csv(SPLIT_OUT, parse_dates=["night_of"])
    assert nights.reset_index(drop=True).equals(committed[nights.columns]), f"split differs from {SPLIT_OUT}"
    print(f"usable nights: {len(nights)} ({nights['night_of'].min():%Y-%m-%d} -> {nights['night_of'].max():%Y-%m-%d}), "
          f"hours: {nights['n_hours'].sum()} | identical to {SPLIT_OUT}")
    for split, g in nights.groupby("split", sort=False):
        print(f"  {split}: {len(g)} nights, {g['n_hours'].sum()} hours, "
              f"{g['night_of'].min():%Y-%m-%d} -> {g['night_of'].max():%Y-%m-%d}")
