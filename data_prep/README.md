# Data preparation — NSW AEI and wholesale-price data

Otis owns this data-preparation component of the ENGG2112 NSW MEF project.
It contains:
validated hourly average emissions intensity (AEI) and NSW wholesale-price
tables for July 2025 to June 2026.

## Layout

- `src/download_price_5min.py`: downloads and validates five-minute NSW1
  `DISPATCHPRICE` data from AEMO using NEMOSIS.
- `src/build_price_hourly.py`: converts five-minute prices to the shared hourly
  format and writes the price validation report.
- `src/build_aei_hourly.py`: validates Kunyu's five-minute AEI file, converts it
  to the shared hourly format and checks that its keys match the price table.
- `outputs/`: the two frozen hourly CSVs and their validation output.
- `docs/data_dictionary_v1.md`: sources, fields, units and transformation rules.
- `docs/data_quality_v1.md`: validation results and limitations.

Raw downloads, caches and virtual environments are deliberately excluded.

## Environment

From the repository root:

```bash
python3 -m venv data_prep/.venv
source data_prep/.venv/bin/activate
python -m pip install "nemosis==3.8.1" "pandas>=2.2.3,<3"
```

## Reproduce the price table

```bash
python data_prep/src/download_price_5min.py
python data_prep/src/build_price_hourly.py
```

The first command writes the raw filtered file and AEMO cache into ignored
local directories under `data_prep/`. The second command writes the hourly CSV
and validation report to `data_prep/outputs/`.

## Reproduce the AEI table

Kunyu's upstream build writes the five-minute input to
`data/processed/aei_nsw1_5min_full.csv`. With that local file present:

```bash
python data_prep/src/build_aei_hourly.py
```

Alternatively, supply another location explicitly:

```bash
python data_prep/src/build_aei_hourly.py --input /path/to/aei_nsw1_5min_full.csv
```

## Shared output schema

Both CSVs use:

```text
hour_start_nem,hour_start_sydney,sydney_date,value,n_intervals
```

Join tables using `hour_start_nem`. Sydney `02:00` occurs twice when daylight
saving ends, so `hour_start_sydney` is not a unique key.

## AI use

An AI assistant drafted and ran the download, aggregation and validation code
under Otis's direction. Otis reviewed the methods, sources and outputs. This use
is recorded in the team's AI declaration and contributions log.
