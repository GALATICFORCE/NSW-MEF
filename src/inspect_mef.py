import pandas as pd

df = pd.read_csv("data/raw/mef_nsw1_5min_2025-07-01_7d.csv", parse_dates=["Time"])

# ---- 1. Which 5-minute intervals are missing ----
expected = pd.date_range("2025-07-01 00:05", "2025-07-08 00:00", freq="5min")
missing = expected.difference(df["Time"])
print("expected:", len(expected), "present:", df["Time"].nunique(), "missing:", len(missing))
print("duplicated Time:", df["Time"].duplicated().sum())

# Merge consecutive missing intervals into gaps
s = pd.Series(missing)
gap_id = s.diff().ne(pd.Timedelta("5min")).cumsum()
gaps = s.groupby(gap_id).agg(start="first", end="last", n="size")
print("number of gaps:", len(gaps))
print(gaps["n"].describe())
print(gaps.sort_values("n", ascending=False).head(5))

# Missing intervals by hour of day
print(pd.Series(missing.hour).value_counts().sort_index())

# ---- 2. Negative and unusually high values ----
cols = ["Time", "Intensity_Index", "DUID", "CO2E_ENERGY_SOURCE"]
neg = df[df["Intensity_Index"] < 0]
print("negative rows:", len(neg))
print(neg[cols].head(10))

high = df[df["Intensity_Index"] > 1.2]
print("> 1.2 rows:", len(high))
print(high[cols].head(10))
