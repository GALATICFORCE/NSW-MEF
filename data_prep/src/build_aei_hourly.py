"""Validate the five-minute AEI data and build the shared hourly table."""

import argparse
from pathlib import Path

import pandas as pd


DATA_PREP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = DATA_PREP_DIR.parent
DEFAULT_INPUT_FILE = REPO_DIR / "data" / "processed" / "aei_nsw1_5min_full.csv"
PRICE_FILE = DATA_PREP_DIR / "outputs" / "price_hourly_nsw1_v1.csv"
OUTPUT_DIR = DATA_PREP_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "aei_hourly_nsw1_v1.csv"
VALIDATION_FILE = OUTPUT_DIR / "aei_hourly_validation_v1.txt"

EXPECTED_RAW_ROWS = 105_120
EXPECTED_HOURLY_ROWS = 8_760
EXPECTED_COLUMNS = {"TimeEnding", "Energy_SO", "Total_Emissions", "aei"}


def monthly_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    return (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"five-minute AEI CSV (default: {DEFAULT_INPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, parse_dates=["TimeEnding"])
    raw = raw.sort_values("TimeEnding").reset_index(drop=True)

    missing_columns = EXPECTED_COLUMNS.difference(raw.columns)
    duplicate_timestamps = int(raw["TimeEnding"].duplicated().sum())
    missing_by_column = raw[list(EXPECTED_COLUMNS)].isna().sum().astype(int)

    expected_timestamps = pd.date_range(
        "2025-07-01 00:05:00", "2026-07-01 00:00:00", freq="5min"
    )
    missing_timestamps = expected_timestamps.difference(raw["TimeEnding"])
    unexpected_timestamps = pd.DatetimeIndex(raw["TimeEnding"]).difference(
        expected_timestamps
    )

    calculated_aei = raw["Total_Emissions"] / raw["Energy_SO"]
    formula_difference = (raw["aei"] - calculated_aei).abs()
    max_formula_difference = float(formula_difference.max())
    formula_mismatch_count = int((formula_difference > 1e-12).sum())
    nonpositive_energy_count = int((raw["Energy_SO"] <= 0).sum())

    # TimeEnding is the end of each five-minute interval. Assign using the
    # interval start so 00:05 belongs to the hour beginning at 00:00.
    raw["interval_start_nem"] = raw["TimeEnding"] - pd.Timedelta(minutes=5)
    raw["hour_start_nem"] = raw["interval_start_nem"].dt.floor("h")

    raw_hour_counts = raw.groupby("hour_start_nem").size()
    incomplete_raw_hours = raw_hour_counts[raw_hour_counts != 12]

    hourly = (
        raw.groupby("hour_start_nem", as_index=False)
        .agg(value=("aei", "mean"), n_intervals=("aei", "count"))
        .sort_values("hour_start_nem")
        .reset_index(drop=True)
    )
    hourly.loc[hourly["n_intervals"] < 6, "value"] = pd.NA

    ratio_of_hourly_totals = (
        raw.groupby("hour_start_nem")["Total_Emissions"].sum()
        / raw.groupby("hour_start_nem")["Energy_SO"].sum()
    )
    simple_hourly_average = hourly.set_index("hour_start_nem")["value"]
    hourly_method_difference = (
        simple_hourly_average - ratio_of_hourly_totals
    ).abs()
    max_hourly_method_difference = float(hourly_method_difference.max())
    median_hourly_method_difference = float(hourly_method_difference.median())

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
    expected_unusual_counts = {"2025-10-05": 23, "2026-04-05": 25}

    missing_value_count = int(hourly["value"].isna().sum())
    invalid_nan_rule_count = int(
        (hourly["value"].isna() & (hourly["n_intervals"] >= 6)).sum()
    )
    implausible_hour_count = int(
        ((hourly["value"] < 0) | (hourly["value"] > 1.2)).sum()
    )

    price = pd.read_csv(PRICE_FILE)
    output_nem_keys = hourly["hour_start_nem"].dt.strftime("%Y-%m-%d %H:%M")
    price_nem_keys = price["hour_start_nem"].astype(str)
    keys_match_price = output_nem_keys.reset_index(drop=True).equals(
        price_nem_keys.reset_index(drop=True)
    )

    monthly = monthly_summary(hourly)
    validation_lines = [
        f"Raw rows: {len(raw):,} (expected {EXPECTED_RAW_ROWS:,})",
        f"Input columns: {raw.columns[:4].tolist()}",
        f"Missing required columns: {sorted(missing_columns)}",
        f"First timestamp: {raw['TimeEnding'].min()}",
        f"Last timestamp: {raw['TimeEnding'].max()}",
        f"Duplicate raw timestamps: {duplicate_timestamps}",
        f"Missing expected raw timestamps: {len(missing_timestamps)}",
        f"Unexpected raw timestamps: {len(unexpected_timestamps)}",
        "Missing raw values: "
        + ", ".join(f"{column}={count}" for column, count in missing_by_column.items()),
        f"Non-positive Energy_SO rows: {nonpositive_energy_count}",
        f"Maximum |aei - Total_Emissions/Energy_SO|: {max_formula_difference:.3e}",
        f"Formula mismatches above 1e-12: {formula_mismatch_count}",
        f"Raw AEI range: {raw['aei'].min():.6f} to {raw['aei'].max():.6f} tCO2-e/MWh",
        "Maximum absolute difference between simple hourly mean and "
        f"hourly ratio-of-totals: {max_hourly_method_difference:.6e}",
        "Median absolute difference between simple hourly mean and "
        f"hourly ratio-of-totals: {median_hourly_method_difference:.6e}",
        "",
        f"Hourly rows: {len(hourly):,} (expected {EXPECTED_HOURLY_ROWS:,})",
        f"First NEM hour: {hourly['hour_start_nem'].min()}",
        f"Last NEM hour: {hourly['hour_start_nem'].max()}",
        f"Duplicate NEM hours: {duplicate_nem_hours}",
        f"Missing expected NEM hours: {len(missing_hours)}",
        f"Unexpected NEM hours: {len(unexpected_hours)}",
        f"Raw hours not containing 12 rows: {len(incomplete_raw_hours)}",
        f"Sydney dates: {len(date_counts)} (expected 365)",
        "Unusual Sydney-date hour counts: "
        + ", ".join(f"{date}={count}" for date, count in unusual_date_counts.items()),
        f"Missing hourly values: {missing_value_count}",
        f"Empty values with six or more valid intervals: {invalid_nan_rule_count}",
        f"Hourly values outside 0 to 1.2 tCO2-e/MWh: {implausible_hour_count}",
        f"Hourly NEM keys exactly match price table: {keys_match_price}",
        "",
        "Monthly AEI summary (tCO2-e/MWh):",
        monthly.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
    ]
    validation_text = "\n".join(validation_lines)
    print(validation_text)

    assert len(raw) == EXPECTED_RAW_ROWS
    assert not missing_columns
    assert duplicate_timestamps == 0
    assert len(missing_timestamps) == 0
    assert len(unexpected_timestamps) == 0
    assert int(missing_by_column.sum()) == 0
    assert nonpositive_energy_count == 0
    assert formula_mismatch_count == 0
    assert len(hourly) == EXPECTED_HOURLY_ROWS
    assert duplicate_nem_hours == 0
    assert len(missing_hours) == 0
    assert len(unexpected_hours) == 0
    assert len(incomplete_raw_hours) == 0
    assert len(date_counts) == 365
    assert unusual_date_counts.to_dict() == expected_unusual_counts
    assert missing_value_count == 0
    assert invalid_nan_rule_count == 0
    assert implausible_hour_count == 0
    assert set(hourly["n_intervals"]) == {12}
    assert keys_match_price

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
