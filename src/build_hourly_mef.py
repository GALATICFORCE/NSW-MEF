import pandas as pd

SRC = "data/raw/mef_nsw1_5min_2025-07-01_7d.csv"
OUT = "data/processed/mef_nsw1_hourly_v0.1.csv"
MIN_INTERVALS = 6
DATA_VERSION = "v0.1"

df = pd.read_csv(SRC, parse_dates=["Time"])

# 1. Time 是区间结束时刻：减 5 分钟变成区间开始，再向下取整到整点
df["hour_nem"] = (df["Time"] - pd.Timedelta("5min")).dt.floor("H")

# 2. 每小时取平均，同时记录有几个 5 分钟区间参与
hourly = (df.groupby("hour_nem")["Intensity_Index"]
            .agg(mef_target="mean", n_intervals="size")
            .reset_index())

# 3. 补齐时间段内的每一个小时，完全没有数据的小时也要占一行
full = pd.date_range(hourly["hour_nem"].min(), hourly["hour_nem"].max(), freq="H")
hourly = (hourly.set_index("hour_nem").reindex(full)
                .rename_axis("hour_nem").reset_index())
hourly["n_intervals"] = hourly["n_intervals"].fillna(0).astype(int)

# 4. 覆盖不足的小时设为缺失
hourly.loc[hourly["n_intervals"] < MIN_INTERVALS, "mef_target"] = float("nan")

# 5. NEM 时间 = AEST（UTC+10，无夏令时），和布里斯班相同
nem = hourly["hour_nem"].dt.tz_localize("Australia/Brisbane")
hourly["timestamp_utc"] = nem.dt.tz_convert("UTC")
hourly["timestamp_local"] = nem.dt.tz_convert("Australia/Sydney")
hourly["region"] = "NSW1"
hourly["data_version"] = DATA_VERSION

out = hourly[["timestamp_utc", "timestamp_local", "region",
              "mef_target", "n_intervals", "data_version"]]
out.to_csv(OUT, index=False)

print(out.head())
print("hours:", len(out), "| missing mef:", out["mef_target"].isna().sum())
print(out["n_intervals"].value_counts().sort_index())
