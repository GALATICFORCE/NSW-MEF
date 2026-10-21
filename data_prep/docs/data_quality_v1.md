# Data quality report v1 — NSW AEI and wholesale price

## Validation summary

| Check | Price | AEI |
|---|---:|---:|
| Five-minute rows | 105,120 | 105,120 |
| Hourly rows | 8,760 | 8,760 |
| Duplicate raw timestamps | 0 | 0 |
| Missing expected raw timestamps | 0 | 0 |
| Raw hours without 12 rows | 0 | 0 |
| Duplicate NEM hours | 0 | 0 |
| Missing hourly values | 0 | 0 |
| Hourly keys match | Yes | Yes |

Both outputs cover `2025-07-01 00:00` through `2026-06-30 23:00` NEM time.
There are 365 Sydney dates: 5 October 2025 contains 23 hours, 5 April 2026
contains 25, and every other date contains 24. Downstream joins must therefore
use `hour_start_nem`.

## AEI checks

- No missing values in `TimeEnding`, `Energy_SO`, `Total_Emissions` or `aei`.
- No non-positive `Energy_SO` records.
- Maximum `|aei - Total_Emissions/Energy_SO|`: 2.22 × 10^-16.
- Raw AEI range: 0.213904–0.863899 tCO2-e/MWh.
- Hourly values outside 0–1.2 tCO2-e/MWh: 0.
- Missing-factor units CG2, CG4 and ERGT01 were excluded from both emissions
  and energy. They operated in 1,536 intervals and contributed 0.02282% of NSW
  sent-out generation. This small excluded share should still be stated as a
  limitation rather than treated as zero-emission generation.

The validation script also reports the difference between the selected simple
hourly AEI mean and the alternative hourly ratio of total emissions to total
energy whenever the raw input is available.

## Price checks

- No missing five-minute or hourly price observations.
- Raw range: -999.99093 to 20,300.00000 AUD/MWh.
- Negative prices and price spikes were flagged but retained as potentially
  genuine NEM outcomes.
- Monthly hourly medians range from 56.81040 to 87.26766 AUD/MWh; extreme
  monthly hourly maxima occur primarily from October 2025 to February 2026.

## Limitations affecting interpretation

1. AEI is production-based and excludes interstate imports, interconnector
   losses and imported electricity's emissions.
2. The marginal unit used for MEF may be outside NSW, while AEI includes only
   NSW generation. Spatial boundaries can therefore contribute to differences.
3. AEI and MEF answer different questions; disagreement is not automatically a
   data error and is one of the project's research questions.
4. Hourly averaging intentionally hides five-minute variation because the load
   is scheduled in one-hour blocks.
5. Wholesale price is not a household bill; retail margins, network charges
   and tariff structures are excluded.
6. Genuine price spikes may dominate annual cost totals. Report totals together
   with typical-night statistics.

Full monthly min/median/max tables and executable checks are preserved in
`data_prep/outputs/price_hourly_validation_v1.txt` and
`data_prep/outputs/aei_hourly_validation_v1.txt`.
