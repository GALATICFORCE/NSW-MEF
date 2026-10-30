"""Night definition and split (hand-off doc §4.2)

night_of = D: Sydney local D 18:00 → D+1 07:00, normally 13 hours; the 2025-10-04 night has 12 hours, the 2026-04-04 night 14.
Split by night, in time order, at fixed date boundaries (decided 10-11, identical to the committed outputs\\nights_split_v1.csv).
Longer lag features added later are only trimmed from the front of train; the val / test dates do not move.

Usage: python src\\nights.py   (checks the split against outputs\\nights_split_v1.csv row by row; does not rewrite that file)
"""
import pandas as pd

HOURLY = "data/processed/mef_hourly_{label}_rules-v1.csv"
SPLIT_OUT = "outputs/nights_split_v1.csv"
NIGHT_START, NIGHT_END = 18, 7               # Sydney local hours
MAX_LAG_DAYS = 7                              # Step 4 features reach back to D-7 at most
FIRST_DAY = pd.Timestamp("2025-07-01")        # first complete Sydney date
LAST_NIGHT = pd.Timestamp("2026-06-29")       # the 2026-06-30 night lacks 07-01 00:00-07:00
FIRST_NIGHT = FIRST_DAY + pd.Timedelta(days=MAX_LAG_DAYS)
VAL_START = pd.Timestamp("2026-03-15")
TEST_START = pd.Timestamp("2026-05-08")


def load_hourly(label="C"):
    h = pd.read_csv(HOURLY.format(label=label))
    for col in ["hour_start_nem", "hour_start_sydney", "sydney_date"]:
        h[col] = pd.to_datetime(h[col])
    return h


def add_night(h):
    """Adds a night_of column; hours outside 18:00-07:00 get NaT"""
    hour = h["hour_start_sydney"].dt.hour
    night = pd.Series(pd.NaT, index=h.index, dtype="datetime64[ns]")
    night[hour >= NIGHT_START] = h.loc[hour >= NIGHT_START, "sydney_date"]
    night[hour < NIGHT_END] = h.loc[hour < NIGHT_END, "sydney_date"] - pd.Timedelta(days=1)
    return h.assign(night_of=night)


def split_nights(h):
    """Splits the usable nights at fixed date boundaries; returns night_of, split, n_hours"""
    nights = add_night(h).dropna(subset=["night_of"]).groupby("night_of").size().rename("n_hours")
    nights = nights[(nights.index >= FIRST_NIGHT) & (nights.index <= LAST_NIGHT)].reset_index()
    nights["split"] = "train"
    nights.loc[nights["night_of"] >= VAL_START, "split"] = "val"
    nights.loc[nights["night_of"] >= TEST_START, "split"] = "test"
    return nights[["night_of", "split", "n_hours"]]


def night_hours(label="C", splits=("train", "val")):
    """Hours of the usable nights (by default train and val only; the test set is not used in any computation
    before Step 7)"""
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
