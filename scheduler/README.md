# scheduler/ — owner: Oscar Naglitsch

Scheduler and emissions/cost scoring for a home EV charger (9.2 kWh per night = 4 hours × 2.3 kWh).
Each night D runs 18:00–07:00 Sydney time (13 hours; the last hour starts at 06:00). Each strategy
picks the 4 hours with the lowest signal (ties go to the earlier hour); fixed charging means plug in
at 18:00 and charge 18:00–22:00. Every schedule is scored with the actual hourly MEF under labels
B, C and C-fwd and the actual NSW wholesale price, never with the signal used to choose it.

## Files

| File | What it holds | Made by | Commit |
|---|---|---|---|
| `scheduler.ipynb` | The scheduler: pick rule + unit test, loader and night slicer, round 1, round 2 | — | 879aa3a |
| `schedule_results_v1.csv` | One row per night × strategy: hours chosen, kWh, cost (AUD), emissions under B, C, C-fwd (kg CO2e), AEI-accounted kg. `fixed`, `mef_oracle`, `aei`, `price` on all 364 nights; `mef_forecast` on the 53 test nights | notebook cells 5 and 7 | 6db090c, 879aa3a |
| `schedule_test_ridge_vs_fixed_v1.csv` | Ridge forecast vs fixed on the 53 test nights (2026-05-08 to 2026-06-29) | notebook cell 8 | 0355c46 |
| `schedule_test_oracle_v1.csv` | True-MEF (label C) schedule on the same 53 test nights | notebook cell 8 | 0355c46 |
| `make_figures.py` | Makes the two figures and the totals table below from `schedule_results_v1.csv` | `py scheduler/make_figures.py` (from the repository root) | this branch |
| `fig_5_1_hours_overlap_v1.png` | Figure for Section 5.1 | `make_figures.py` | this branch |
| `fig_5_2_emissions_cost_v1.png` | Figure for Section 5.2 | `make_figures.py` | this branch |
| `fig_5_2_totals_v1.csv` | The numbers behind Figure 5.2: totals per strategy and % change vs fixed | `make_figures.py` | this branch |

The input tables (MEF, AEI, price, forecast) are not stored here; they are shared separately.
All numbers come from the frozen results; nothing was recomputed for the figures.

## Note on `hours_chosen`

`hours_chosen` lists Sydney start hours, e.g. `22;23;1;2`. On the 14-hour night of 2026-04-04,
daylight saving ends and 02:00 Sydney time happens twice, so a schedule can show `2` twice
(`mef_oracle` that night is `18;1;2;2`: both 02:00 hours were chosen). This is correct, not a
duplicate row. On the 12-hour night of 2025-10-04, 02:00 does not exist.

## Figure captions

**Figure 5.1.** Charging hours chosen by each strategy and their overlap with the true-MEF schedule.
(a) Share of nights on which each Sydney start hour is chosen. (b) Number of the 4 chosen hours
shared with the schedule picked by true MEF (label C) on the same night; legend gives the mean.
Full year 2025-07-01 to 2026-06-29, 364 nights, actual signals; MEF schedule chosen with label C.
Source: `scheduler/schedule_results_v1.csv` (commit 879aa3a).

**Figure 5.2.** Estimated emissions and wholesale cost of each strategy relative to fixed charging
(18:00–22:00). Emissions are estimated with actual hourly MEF under labels B, C and C-fwd (kg CO2e);
cost is at wholesale prices (AUD). All strategies charge 9.2 kWh per night. Full year 2025-07-01 to
2026-06-29, 364 nights, actual signals. Source: `scheduler/schedule_results_v1.csv` (commit 879aa3a);
numbers in `scheduler/fig_5_2_totals_v1.csv`.

## Headline numbers (364 nights, from `fig_5_2_totals_v1.csv`)

| Strategy | kg CO2e, B | kg CO2e, C | kg CO2e, C-fwd | Cost AUD |
|---|---|---|---|---|
| fixed | 1646.25 | 1696.28 | 2072.73 | 426.36 |
| mef_oracle | 1448.80 (−12.0%) | 1345.03 (−20.7%) | 1722.46 (−16.9%) | 368.95 (−13.5%) |
| aei | 1904.79 (+15.7%) | 1952.40 (+15.1%) | 2151.36 (+3.8%) | 388.08 (−9.0%) |
| price | 2321.67 (+41.0%) | 2274.11 (+34.1%) | 2306.70 (+11.3%) | 250.20 (−41.3%) |

Overlap with the true-MEF schedule (mean hours shared out of 4): fixed 2.10, AEI 1.51, price 0.84.
Same 4 hours on 11.5% (fixed), 3.0% (AEI) and 0% (price) of nights.

Cross-check with Section 5.3 (label C, 53 test nights): fixed − oracle = 1.0715 kg per night;
mef_forecast − fixed = +0.0517 (B), −0.0103 (C), −0.0398 (C-fwd) kg per night. These match Kunyu's numbers.
