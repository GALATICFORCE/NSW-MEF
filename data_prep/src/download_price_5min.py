"""Download and validate NSW1 five-minute wholesale prices for the study year.

Run from any directory with ``python data_prep/src/download_price_5min.py``.
The raw filtered file and AEMO cache are local-only files ignored by Git.
"""

from pathlib import Path

import pandas as pd
from nemosis import dynamic_data_compiler


START_TIME = "2025/07/01 00:00:00"
END_TIME = "2026/07/01 00:00:00"
EXPECTED_ROWS = 105_120

DATA_PREP_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = DATA_PREP_DIR / ".cache" / "aemo"
LOCAL_DATA_DIR = DATA_PREP_DIR / ".local"
OUTPUT_FILE = LOCAL_DATA_DIR / "price_5min_nsw1_raw.csv"


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    price = dynamic_data_compiler(
        start_time=START_TIME,
        end_time=END_TIME,
        table_name="DISPATCHPRICE",
        raw_data_location=str(CACHE_DIR),
        select_columns=["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "RRP"],
        filter_cols=["REGIONID", "INTERVENTION"],
        filter_values=[["NSW1"], [0]],
        fformat="csv",
        keep_csv=True,
    )

    price["SETTLEMENTDATE"] = pd.to_datetime(price["SETTLEMENTDATE"])
    price = price.sort_values("SETTLEMENTDATE").reset_index(drop=True)

    duplicate_count = int(price.duplicated("SETTLEMENTDATE").sum())
    missing_rrp_count = int(price["RRP"].isna().sum())
    expected_timestamps = pd.date_range(
        "2025-07-01 00:05:00",
        "2026-07-01 00:00:00",
        freq="5min",
    )
    missing_timestamps = expected_timestamps.difference(price["SETTLEMENTDATE"])
    unexpected_timestamps = pd.DatetimeIndex(price["SETTLEMENTDATE"]).difference(
        expected_timestamps
    )

    print(f"Rows: {len(price):,} (expected {EXPECTED_ROWS:,})")
    print(f"First timestamp: {price['SETTLEMENTDATE'].min()}")
    print(f"Last timestamp: {price['SETTLEMENTDATE'].max()}")
    print(f"Regions: {sorted(price['REGIONID'].unique())}")
    print(f"Intervention values: {sorted(price['INTERVENTION'].unique())}")
    print(f"Duplicate timestamps: {duplicate_count}")
    print(f"Missing RRP values: {missing_rrp_count}")
    print(f"Missing expected timestamps: {len(missing_timestamps)}")
    print(f"Unexpected timestamps: {len(unexpected_timestamps)}")
    print(f"RRP range: {price['RRP'].min():.5f} to {price['RRP'].max():.5f} AUD/MWh")

    assert len(price) == EXPECTED_ROWS, "Unexpected number of price records"
    assert duplicate_count == 0, "Duplicate SETTLEMENTDATE values found"
    assert missing_rrp_count == 0, "Missing RRP values found"
    assert len(missing_timestamps) == 0, "Expected timestamps are missing"
    assert len(unexpected_timestamps) == 0, "Unexpected timestamps found"
    assert set(price["REGIONID"]) == {"NSW1"}, "Unexpected region found"
    assert set(price["INTERVENTION"]) == {0}, "Unexpected intervention value found"

    price.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
