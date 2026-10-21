"""Build and validate the shared-format hourly NSW1 price table."""

import argparse
from pathlib import Path

import pandas as pd


DATA_PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = DATA_PREP_DIR / ".local" / "price_5min_nsw1_raw.csv"
OUTPUT_DIR = DATA_PREP_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "price_hourly_nsw1_v1.csv"
VALIDATION_FILE = OUTPUT_DIR / "price_hourly_validation_v1.txt"

EXPECTED_RAW_ROWS = 105_120
EXPECTED_HOURLY_ROWS = 8_760


def monthly_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    summary = (
        hourly.assign(month=hourly["hour_start_nem"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            rows=("value", "size"),
            missing_values=("value", lambda values: int(values.isna().sum())),
            minimum=("value", "min"),
            median=("value", "median"),
            maximum=("value", "max"),
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"five-minute DISPATCHPRICE CSV (default: {DEFAULT_INPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, parse_dates=["SETTLEMENTDATE"])
    raw = raw.sort_values("SETTLEMENTDATE").reset_index(drop=True)

    # AEMO timestamps identify the interval end. Subtract five minutes before
    # assigning each observation to an hourly period.
    raw["interval_start_nem"] = raw["SETTLEMENTDATE"] - pd.Timedelta(minutes=5)
    raw["hour_start_nem"] = raw["interval_start_nem"].dt.floor("h")

    raw_hour_counts = raw.groupby("hour_start_nem").size()
    incomplete_raw_hours = raw_hour_counts[raw_hour_counts != 12]

    hourly = (
        raw.groupby("hour_start_nem", as_index=False)
        .agg(value=("RRP", "mean"), n_intervals=("RRP", "count"))
        .sort_values("hour_start_nem")
        .reset_index(drop=True)
    )

    # Shared missing-data rule: fewer than six valid intervals leaves the
    # hourly value empty. The interval count is retained to expose coverage.
    hourly.loc[hourly["n_intervals"] < 6, "value"] = pd.NA

    nem_aware = hourly["hour_start_nem"].dt.tz_localize("Australia/Brisbane")
    sydney_aware = nem_aware.dt.tz_convert("Australia/Sydney")
    hourly["hour_start_sydney"] = sydney_aware.dt.tz_localize(None)
    hourly["sydney_date"] = hourly["hour_start_sydney"].dt.strftime("%Y-%m-%d")

    hourly = hourly[
        [
            "hour_start_nem",
            "hour_start_sydney",
            "sydney_date",
            "value",
            "n_intervals",
        ]
    ]

    expected_hours = pd.date_range(
        "2025-07-01 00:00:00", "2026-06-30 23:00:00", freq="h"
    )
    missing_hours = expected_hours.difference(hourly["hour_start_nem"])
    unexpected_hours = pd.DatetimeIndex(hourly["hour_start_nem"]).difference(
        expected_hours
    )
    duplicate_nem_hours = int(hourly["hour_start_nem"].duplicated().sum())

    date_counts = hourly.groupby("sydney_date").size()
    unusual_date_counts = date_counts[date_counts != 24]
    expected_unusual_counts = {
        "2025-10-05": 23,
        "2026-04-05": 25,
    }

    missing_value_count = int(hourly["value"].isna().sum())
    invalid_nan_rule_count = int(
        (hourly["value"].isna() & (hourly["n_intervals"] >= 6)).sum()
    )
    monthly = monthly_summary(hourly)

    validation_lines = [
        f"Raw rows: {len(raw):,} (expected {EXPECTED_RAW_ROWS:,})",
        f"Hourly rows: {len(hourly):,} (expected {EXPECTED_HOURLY_ROWS:,})",
        f"First NEM hour: {hourly['hour_start_nem'].min()}",
        f"Last NEM hour: {hourly['hour_start_nem'].max()}",
        f"Duplicate NEM hours: {duplicate_nem_hours}",
        f"Missing expected NEM hours: {len(missing_hours)}",
        f"Unexpected NEM hours: {len(unexpected_hours)}",
        f"Raw hours not containing 12 intervals: {len(incomplete_raw_hours)}",
        f"Sydney dates: {len(date_counts)} (expected 365)",
        "Unusual Sydney-date hour counts: "
        + ", ".join(f"{date}={count}" for date, count in unusual_date_counts.items()),
        f"Missing hourly values: {missing_value_count}",
        f"Empty values with six or more valid intervals: {invalid_nan_rule_count}",
        "",
        "Monthly price summary (AUD/MWh):",
        monthly.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "",
        "Outlier note: Negative and high prices were flagged for review but retained,",
        "because both can be valid outcomes in the NEM wholesale market.",
    ]
    validation_text = "\n".join(validation_lines)
    print(validation_text)

    assert len(raw) == EXPECTED_RAW_ROWS
    assert len(hourly) == EXPECTED_HOURLY_ROWS
    assert duplicate_nem_hours == 0
    assert len(missing_hours) == 0
    assert len(unexpected_hours) == 0
    assert len(incomplete_raw_hours) == 0
    assert len(date_counts) == 365
    assert unusual_date_counts.to_dict() == expected_unusual_counts
    assert missing_value_count == 0
    assert invalid_nan_rule_count == 0
    assert set(hourly["n_intervals"]) == {12}

    output = hourly.copy()
    output["hour_start_nem"] = output["hour_start_nem"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    output["hour_start_sydney"] = output["hour_start_sydney"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    output.to_csv(OUTPUT_FILE, index=False)
    VALIDATION_FILE.write_text(validation_text + "\n", encoding="utf-8")

    print(f"\nSaved hourly table: {OUTPUT_FILE}")
    print(f"Saved validation output: {VALIDATION_FILE}")


if __name__ == "__main__":
    main()
