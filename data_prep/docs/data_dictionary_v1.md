# Data dictionary v1 — NSW AEI and wholesale price

## Common scope and schema

- Study period: five-minute intervals ending `2025-07-01 00:05` through
  `2026-07-01 00:00`, giving 105,120 intervals and 8,760 hours.
- NEM time is AEST (UTC+10) throughout the year. Raw timestamps mark the end
  of each interval.
- Hour assignment: `(interval_end - 5 minutes).floor("h")`.
- Sydney time: localise NEM time with `Australia/Brisbane`, then convert to
  `Australia/Sydney`.
- Missing-data rule: `value` is the mean of valid five-minute values;
  `n_intervals` records their number; fewer than six leaves `value` empty.
- Unique join key: `hour_start_nem`.

| Output column | Definition |
|---|---|
| `hour_start_nem` | Beginning of the hour in NEM time, `YYYY-MM-DD HH:MM`. |
| `hour_start_sydney` | Same instant in Sydney local time; not unique at the daylight-saving fallback. |
| `sydney_date` | Sydney calendar date used to form charging nights. |
| `value` | Hourly AEI or wholesale price. |
| `n_intervals` | Valid five-minute values used in the hourly mean. |

## `price_hourly_nsw1_v1.csv`

- Source: AEMO MMS `DISPATCHPRICE`, retrieved with NEMOSIS 3.8.1.
- Fields: `SETTLEMENTDATE`, `REGIONID`, `INTERVENTION`, `RRP`.
- Filters: `REGIONID = NSW1`; `INTERVENTION = 0`.
- Why filter intervention: the non-intervention solution represents the normal
  market outcome and avoids alternative solutions duplicating an interval.
- Hourly value: arithmetic mean of the twelve equal-duration five-minute RRP
  observations.
- Units: AUD/MWh.
- Negative and high prices are retained because they can be genuine wholesale
  outcomes.

AEMO's five-minute-settlement specification identifies interval 1 as the
trading interval ending at 00:05 AEST, supporting the interval-end convention:
https://www.aemo.com.au/-/media/files/electricity/nem/5ms/systems-workstream/2020/emms-technical-specification---five-minute-settlement---marked-up.pdf

## `aei_hourly_nsw1_v1.csv`

- Input: `data/processed/aei_nsw1_5min_full.csv`, built by Kunyu with NEMED
  from AEMO `DISPATCH_UNIT_SCADA` and generator emissions factors.
- Fields: `TimeEnding`, `Energy_SO`, `Total_Emissions`, `aei`.
- Five-minute formula: `aei = Total_Emissions / Energy_SO`.
- Emissions factors: AEMO `GENUNITS.CO2E_EMISSIONS_FACTOR`, frozen as at
  1 July 2026; the same factors are used by the MEF workflow.
- Scope: NSW1 generators only. Interstate imports, batteries and other
  bidirectional units are excluded, making the measure production-based.
- Missing factors: CG2, CG4 and ERGT01 produced energy but lacked emissions
  factors. They were excluded from both numerator and denominator rather than
  silently assigned zero emissions. They affected 1,536 intervals and supplied
  13,680.8 MWh (0.02282% of NSW sent-out generation).
- Hourly value: simple mean of twelve five-minute AEIs. This matches a
  constant-power load and the shared MEF/price aggregation rule.
- Units: tCO2-e/MWh, numerically equal to kg CO2-e/kWh.

Otis verified the input's completeness and the aggregate AEI formula. Kunyu's
unit-level build and missing-factor statistics are documented in
`src/build_aei_5min.py`, `docs/aei_method_note.md` and
`outputs/aei_build_v1.txt`.

## Sources

1. **Price timestamp convention:** AEMO's *EMMS Technical Specification —
   Five-Minute Settlement* identifies interval 1 as the interval ending at
   00:05 AEST. Therefore, each `SETTLEMENTDATE` was treated as the end of its
   five-minute interval. Source:
   https://www.aemo.com.au/-/media/files/electricity/nem/5ms/systems-workstream/2020/emms-technical-specification---five-minute-settlement---marked-up.pdf
2. **Price source and units:** the price input came from the public AEMO MMS
   `DISPATCHPRICE` table. `RRP` is the regional reference price; NSW1,
   non-intervention rows were retained and values are in AUD/MWh. Sources:
   https://visualisations.aemo.com.au/aemo/nemweb/mmsdatamodelreport/electricity/mms%20data%20model%20report_files/MMS_130.htm
   and https://tech-specs.docs.public.aemo.com.au/Content/TOC_common-topics-DOnotChange/Glossary.htm
3. **Input coverage and vintage:** both hourly tables cover five-minute
   intervals ending 1 July 2025 00:05 through 1 July 2026 00:00. Price was
   downloaded from `DISPATCHPRICE` with NEMOSIS 3.8.1. The AEI input supplied
   by Kunyu was built from AEMO `DISPATCH_UNIT_SCADA` and
   `GENUNITS.CO2E_EMISSIONS_FACTOR`, with the generator-information snapshot
   frozen at 1 July 2026. Build details:
   https://github.com/GALATICFORCE/NSW-MEF/blob/master/docs/aei_method_note.md

## Versioning

These are v1 files prepared before the 21 October result freeze. Any change
after the freeze must be written as a new v2 file rather than overwriting v1.
