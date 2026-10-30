"""Structural checks: 5-minute full table, 5-minute labels, hourly tables. Reports only row counts, NaN counts and
PASS/FAIL, no MEF value statistics

Usage: python src\\validate.py --labels A
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from build_hourly_mef import FIRST_HOUR, FMT, LAST_HOUR, MIN_INTERVALS
from labels import LABELS, OUT as LABELS_FILE, is_source, load_full, nan_mask, nan_reason

EXPECTED_NAN_5MIN = 647
EXPECTED_SOURCES = 70_731
# tier column -> battery direction it applies to (rules section 5)
TIER_COLS = {"tier_B": ["charge", "discharge", "both"], "tier_C_gen": ["discharge", "both"],
             "tier_C_load": ["charge", "both"], "tier_Cfwd_gen": ["discharge", "both"]}
TIER_FOR = {"B": ["tier_B"], "C": ["tier_C_gen", "tier_C_load"], "Cfwd": ["tier_Cfwd_gen", "tier_C_load"]}
REPORT = "outputs/validate_rules-v1.txt"
DST_DAYS = {"2025-10-05": 23, "2026-04-05": 25}

lines, failures = [], []


def note(text=""):
    print(text)
    lines.append(text)


def check(name, ok, detail=""):
    note(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def check_full(full):
    note("== 5-min full table")
    start = full["Time"] - pd.Timedelta("5min")
    expected = pd.date_range("2025-07-01 00:05", "2026-07-01 00:00", freq="5min")
    check("rows = 105,120", len(full) == 105_120, f"{len(full)}")
    check("Time unique, continuous, 2025-07-01 00:05 -> 2026-07-01 00:00",
          full["Time"].is_unique and full["Time"].equals(pd.Series(expected, name="Time")))
    per_month = start.dt.to_period("M").value_counts().sort_index()
    bad = [str(m) for m, n in per_month.items() if n != m.days_in_month * 288]
    check("rows per month (by interval start) = days x 288", not bad, f"mismatched months: {bad}" if bad else "")
    per_hour = start.dt.floor("H").value_counts()
    check("every NEM hour has exactly 12 intervals", (per_hour == 12).all() and len(per_hour) == 8_760,
          f"{len(per_hour)} hours, interval counts {sorted(per_hour.unique().tolist())}")


def check_labels_5min(full, names):
    note("\n== 5-min labels: " + LABELS_FILE)
    lab = pd.read_csv(LABELS_FILE, parse_dates=["Time"])
    check("labels file Time matches full table", lab["Time"].equals(full["Time"]))
    mask = nan_mask(full)
    for name in names:
        col = f"label_{name}"
        if col not in lab:
            check(f"{col} present", False)
            continue
        n_nan = int(lab[col].isna().sum())
        check(f"{col} NaN = {EXPECTED_NAN_5MIN}", n_nan == EXPECTED_NAN_5MIN, f"{n_nan}")
        check(f"{col} NaN positions = rule-based nan_mask", lab[col].isna().equals(mask))
    reasons = nan_reason(full)
    note("NaN intervals by reason (first matching rule): "
         + ", ".join(f"{r} {n}" for r, n in reasons[reasons != ""].value_counts().items()))
    month = (full["Time"] - pd.Timedelta("5min")).dt.to_period("M")
    note("NaN intervals by month: " + ", ".join(f"{m} {n}" for m, n in mask.groupby(month).sum().items()))

    n_src = int(is_source(full).sum())
    check(f"source intervals = {EXPECTED_SOURCES:,}", n_src == EXPECTED_SOURCES, f"{n_src}")
    present = [n for n in names if f"label_{n}" in lab]
    if len(present) > 1:
        same_nan = all(lab[f"label_{n}"].isna().equals(lab[f"label_{present[0]}"].isna()) for n in present)
        check("NaN positions identical across " + "/".join(present), same_nan)
        nonbatt = (full["status"] == "nonbatt") & ~mask
        same = all((lab.loc[nonbatt, f"label_{n}"] == lab.loc[nonbatt, f"label_{present[0]}"]).all() for n in present)
        check("nonbatt intervals: all labels equal (= mef_nemed)", same)
    if "label_A" in lab:
        batt = (full["status"] == "batt") & ~mask
        check("batt intervals: label_A = 0", (lab.loc[batt, "label_A"] == 0).all())

    for col in sorted({c for n in names for c in TIER_FOR.get(n, [])}):
        if col not in lab:
            check(f"{col} present", False)
            continue
        applies = full["batt_dir"].isin(TIER_COLS[col]) & ~mask
        tier = lab[col]
        check(f"{col} filled exactly where it applies", tier.notna().equals(applies),
              f"{int(applies.sum())} applicable intervals")
        check(f"{col} values in {{0,1,2,3}}", tier.dropna().isin([0, 1, 2, 3]).all())
        dist = pd.crosstab(month[applies], tier[applies].fillna(-1).astype(int)).reindex(
            columns=[1, 2, 3, 0], fill_value=0)
        dist.columns = ["tier1", "tier2", "tier3", "none(NaN)"]
        dist.loc["total"] = dist.sum()
        note(f"{col} by month (counts of applicable intervals):")
        note(dist.to_string())
    return lab


def check_hourly(lab, name):
    path = f"data/processed/mef_hourly_{name}_rules-v1.csv"
    note(f"\n== hourly table: {path}")
    if not Path(path).exists():
        check(f"{path} exists", False)
        return
    h = pd.read_csv(path, dtype={"hour_start_nem": str, "hour_start_sydney": str, "sydney_date": str})
    cols = ["hour_start_nem", "hour_start_sydney", "sydney_date", "value", "n_intervals"]
    check("columns in shared order", list(h.columns) == cols, f"{list(h.columns)}")
    check("rows = 8,760", len(h) == 8_760, f"{len(h)}")
    nem = pd.to_datetime(h["hour_start_nem"], format=FMT)
    check("hour_start_nem unique, continuous, 2025-07-01 00:00 -> 2026-06-30 23:00",
          nem.is_unique and nem.equals(pd.Series(pd.date_range(FIRST_HOUR, LAST_HOUR, freq="H"))))
    sydney = nem.dt.tz_localize("Australia/Brisbane").dt.tz_convert("Australia/Sydney").dt.tz_localize(None)
    check("hour_start_sydney = NEM converted to Australia/Sydney", h["hour_start_sydney"].equals(sydney.dt.strftime(FMT)))
    check("sydney_date = date of hour_start_sydney", h["sydney_date"].equals(h["hour_start_sydney"].str[:10]))
    per_day = h.groupby("sydney_date").size()
    bad = {d: n for d, n in per_day.items() if n != DST_DAYS.get(d, 24)}
    check("hours per sydney_date: 2025-10-05 = 23, 2026-04-05 = 25, others 24", not bad and len(per_day) == 365,
          f"{len(per_day)} dates, " + (f"mismatches {bad}" if bad else
                                      f"2025-10-05 {per_day['2025-10-05']}, 2026-04-05 {per_day['2026-04-05']}"))

    hour = (lab["Time"] - pd.Timedelta("5min")).dt.floor("H")
    n_valid = lab[f"label_{name}"].notna().groupby(hour).sum().reindex(nem).to_numpy()
    check("n_intervals = valid 5-min labels in the hour", (h["n_intervals"].to_numpy() == n_valid).all())
    check(f"value NaN <=> n_intervals < {MIN_INTERVALS}",
          h["value"].isna().equals(h["n_intervals"] < MIN_INTERVALS))

    note(f"NaN hours: {int(h['value'].isna().sum())} | hours with n_intervals < 12: {int((h['n_intervals'] < 12).sum())}")
    note("n_intervals distribution: " + ", ".join(f"{k}:{v}" for k, v in h["n_intervals"].value_counts().sort_index().items()))
    month = nem.dt.to_period("M")
    note("NaN hours by month: " + ", ".join(f"{m} {n}" for m, n in h["value"].isna().groupby(month).sum().items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", default=["A"], choices=LABELS)
    args = parser.parse_args()

    full = load_full()
    check_full(full)
    lab = check_labels_5min(full, args.labels)
    for name in args.labels:
        check_hourly(lab, name)

    note(f"\nSUMMARY: {'ALL PASS' if not failures else 'FAILED: ' + '; '.join(failures)}")
    Path(REPORT).parent.mkdir(exist_ok=True)
    Path(REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {REPORT}")
    sys.exit(1 if failures else 0)
