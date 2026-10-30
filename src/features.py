"""Step 4 feature table (hand-off doc §4.4, §7): one row per hour of night D, train and val nights only

The forecast is issued at Sydney local D 00:00; every lag feature uses only hours with Sydney date ≤ D-1.
lag1_window_mean is the mean of the hours 00-06 and 18-23 on Sydney date D-1,
not "the night of D-1": that night's 00:00-07:00 falls on day D, which is in the future.

Usage: python src\\features.py
"""
import numpy as np
import pandas as pd

from baselines import yesterday
from nights import TEST_START, load_hourly, night_hours

LABEL = "C"
OUT = "data/processed/features_C_rules-v1_trainval.csv"
NIGHT_HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
WINDOW_HOURS = list(range(0, 7)) + list(range(18, 24))
HOUR_COLS = [f"hour_{h:02d}" for h in NIGHT_HOURS]
DOW_COLS = [f"dow_{d}" for d in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]]
LAG_COLS = ["lag1_same_hour", "lag1_window_mean", "lag1_day_mean", "lag2to7_same_hour_mean"]
FEATURES = HOUR_COLS + DOW_COLS + LAG_COLS
DAY = pd.Timedelta(days=1)


def lookup_hourly():
    """Lookup table: keeps only hours before the test period (as in baselines.py)"""
    h = load_hourly(LABEL)
    return h[h["sydney_date"] < TEST_START]


def assert_before_issue(src_latest, night_of, name):
    """No leakage: the latest source hour used by a feature (Sydney start time) must have fully ended before D 00:00"""
    assert (pd.Series(src_latest, index=night_of.index) + pd.Timedelta(hours=1) <= night_of).all(), \
        f"{name} uses data from D 00:00 or later"


def build(df, hourly=None):
    """df = output of nights.night_hours (column value is label C); returns the feature table and the record of
    daylight-saving handling for lag1_same_hour

    hourly = lookup table for the lag features, by default only hours before the test period; Step 5 passes the
    full-year hourly table when building features for test nights.
    """
    if hourly is None:
        hourly = lookup_hourly()
    d1 = df["night_of"] - DAY
    hour = df["hour_start_sydney"].dt.hour

    out = df[["night_of", "hour_start_nem", "hour_start_sydney", "split"]].copy()
    out["y"] = df["value"]
    for h, col in zip(NIGHT_HOURS, HOUR_COLS):
        out[col] = (hour == h).astype(int)
    for i, col in enumerate(DOW_COLS):
        out[col] = (df["night_of"].dt.dayofweek == i).astype(int)

    # lag1_same_hour: same Sydney hour on D-1; if D-1 lacks that hour use D-2, if it has two take the mean (as in B1)
    out["lag1_same_hour"], fallback, doubled = yesterday(df, hourly)
    src = d1.where(~fallback, d1 - DAY)
    assert_before_issue(src + pd.to_timedelta(hour, unit="H"), df["night_of"], "lag1_same_hour")

    # lag1_window_mean / lag1_day_mean: only hours with Sydney date = D-1
    src_hour = hourly["hour_start_sydney"].dt.hour
    window = hourly[src_hour.isin(WINDOW_HOURS)].groupby("sydney_date")
    day = hourly.groupby("sydney_date")
    assert_before_issue(d1.map(window["hour_start_sydney"].max()), df["night_of"], "lag1_window_mean")
    out["lag1_window_mean"] = d1.map(window["value"].mean()).to_numpy()
    assert_before_issue(d1.map(day["hour_start_sydney"].max()), df["night_of"], "lag1_day_mean")
    out["lag1_day_mean"] = d1.map(day["value"].mean()).to_numpy()

    # lag2to7_same_hour_mean: same Sydney hour on D-2…D-7; a day without that hour is skipped, a day with two is averaged first
    per_day_hour = hourly.groupby([hourly["sydney_date"], src_hour])["value"].mean()
    lags = []
    for k in range(2, 8):
        src = df["night_of"] - k * DAY
        assert_before_issue(src + pd.to_timedelta(hour, unit="H"), df["night_of"], "lag2to7_same_hour_mean")
        lags.append(per_day_hour.reindex(pd.MultiIndex.from_arrays([src, hour])).to_numpy())
    lags = np.vstack(lags)
    assert (~np.isnan(lags)).any(axis=0).all()
    out["lag2to7_same_hour_mean"] = np.nanmean(lags, axis=0)
    return out, fallback, doubled


if __name__ == "__main__":
    df = night_hours(LABEL, ("train", "val"))
    assert (df["night_of"] < TEST_START).all(), "no test nights in the feature table"
    feats, fallback, doubled = build(df)
    assert feats[FEATURES].notna().all().all(), "NaN in features"
    assert feats["y"].notna().all()
    feats.assign(night_of=feats["night_of"].dt.strftime("%Y-%m-%d"),
                 hour_start_nem=feats["hour_start_nem"].dt.strftime("%Y-%m-%d %H:%M"),
                 hour_start_sydney=feats["hour_start_sydney"].dt.strftime("%Y-%m-%d %H:%M")).to_csv(OUT, index=False)
    print(f"rows: {dict(feats['split'].value_counts())} | feature columns: {len(FEATURES)} | "
          f"NaN in features: {int(feats[FEATURES].isna().sum().sum())} -> {OUT}")
    print(f"lag1_same_hour: D-2 fallback {int(fallback.sum())}, two-hour mean {int(doubled.sum())}")
    nights_per_row = feats.groupby("night_of").size()
    print(f"hours per night: {dict(nights_per_row.value_counts().sort_index())}")
