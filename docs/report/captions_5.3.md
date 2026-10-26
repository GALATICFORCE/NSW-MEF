# Captions and file guide for Section 5.3 (Kunyu)

All files are in the public repository https://github.com/GALATICFORCE/NSW-MEF at tag `results-v1` (commit `0c6ca64`, 20 Oct 2026). Nothing in `outputs/` has changed since the tag. Period and labels are the same for every item below unless stated.

- **Evaluation period:** test nights only, 2026-05-08 to 2026-06-29 (53 nights, 689 hours; each night = 18:00–07:00 Sydney time).
- **Truth:** hourly MEF label C (battery-priced intervals filled under the daily-balance assumption). Regret is also re-scored under labels B and C-fwd.
- **Units:** MAE in tCO2e/MWh; regret in kg CO2e per night (4 charging hours × 2.3 kWh).

## Table — frozen test-set model comparison
File: `outputs/model_compare_test-v1.csv`
Caption: *Test-set comparison of day-ahead forecasts, 53 test nights (8 May – 29 June 2026), truth = label C. MAE over all hours and over hours with no battery-priced interval (tCO2e/MWh); mean nightly scheduling regret under labels B, C and C-fwd (kg CO2e per night, 4 × 2.3 kWh); mean within-night Spearman ρ (label C); mean within-night range of the forecast and of the actual label. hourmean uses the training + validation mean per hour and selects the same hours as fixed charging. hgb-v1 is shown for completeness; it was not selected (rule fixed before training). Repository tag results-v1.*

Columns to keep in the report: model, MAE all, MAE pure (no battery-priced interval, 55 hours), regret B / C / C-fwd, ρ(C), forecast range. Rows: hourmean, yesterday, weekmean, ridge-v1 (selected), hgb-v1 (not selected).

## Figure — paired nightly scheduling comparison
File: `outputs/fig_test_paired_regret-v1.png` (numbers in `outputs/paired_regret_test-v1.csv`; per-night values in `outputs/nightly_regret_test-v1.csv`)
Caption: *Per-night difference in scheduling regret, forecast strategy minus fixed charging (18:00–22:00), 53 test nights, truth = label C unless labelled; kg CO2e per night. Points: mean difference; bars: 95% bootstrap interval (10,000 resamples of nights). A strategy is better than fixed charging only if the whole interval lies below zero. ridge-v1 vs fixed: −0.010 [−0.112, 0.106] under C; +0.052 [−0.027, 0.152] under B; −0.040 [−0.155, 0.085] under C-fwd. On 30 of 53 nights ridge-v1 selected exactly the same four hours as fixed charging. Repository tag results-v1.*

## Optional second figure (appendix or in place of the table)
File: `outputs/fig_test_model_compare-v1.png` — bar chart of the table above (MAE and regret per model). Same caption fields as the table.

## Supporting diagnostics (appendix only)
- `outputs/fig_error_heatmap-v1.png`, `outputs/fig_error_by_fill-v1.png`, `outputs/fig_train_val_gap-v1.png` — **validation** nights (2026-03-15 to 2026-05-07, 54 nights), not test. Label the period if used.
- `outputs/error_by_fill_test-v1.csv`, `outputs/bias_by_hour_test-v1.csv` — test-night versions of the error diagnostics.
- `docs/model_selection.md` — the rules fixed in advance (selection, "better than fixed", one-shot test).

## Numbers for the 5.3 text (so the text and the table agree)
- Test MAE: ridge-v1 0.1506, hgb-v1 0.1511 (tie), hourmean 0.1654, weekmean 0.1688, yesterday 0.1919.
- Regret under C: ridge-v1 1.061, fixed/hourmean 1.072, hgb-v1 1.085, weekmean 1.110, yesterday 1.348.
- Within-night range: forecast 0.18 vs actual 0.45 tCO2e/MWh (validation: actual 0.56).
- Perfect knowledge of MEF (oracle) would cut charging emissions by about one fifth relative to fixed charging: validation 4.63 vs 5.86 kg per night (−21%); test 4.63 vs 5.70 kg per night (−19%, from Oscar's `scheduler/schedule_results_v1.csv` at commit `879aa3a`, label C, 53 test nights — the same 1.072 kg/night gap as `outputs/model_compare_test-v1.csv`).
