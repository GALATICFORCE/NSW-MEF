# Pre-specified rules for model selection and evaluation

This note collects, in one place, the rules used to select models and read results, and says for each one **when it was fixed** relative to the result it governs. It was compiled on 21 October 2026, after the results freeze, from the step-by-step instructions given to the coding assistant at the time (kept as part of the AI-use record). Rules that were added or formalised after a result was seen are listed separately at the end.

## Split (set 7 October; applied in `02ab8dc`, revised in `c73573f`)
- Nights are split by date, in time order: training 2025-07-08 to 2026-03-14 (250 nights), validation 2026-03-15 to 2026-05-07 (54), test 2026-05-08 to 2026-06-29 (53). Stored in `outputs/nights_split_v1.csv`; `src/nights.py` asserts the split has not changed.
- No metric was computed on test nights before the single test evaluation. Before that, test nights were loaded only by `src/forecast.py` (Step 5) on 17 October, to build their forecasts (run twice that evening, with byte-identical outputs: `b17f595`, `e4968f4`): their own labels were blanked before features were built, and no metric was computed. In the model-development scripts (`src/ridge.py`, `src/hgb.py`, `src/label_diagnostics.py`, `src/error_analysis.py`), features are read through `ridge.load()`, which asserts that no test night is present; the B and C-fwd labels used for cross-label scoring are read through `night_hours(label, ("train", "val"))`, which selects training and validation nights only (no separate assertion there).

## Ridge (set 11 October, before the first fit; applied in `0a4f709`)
- Feature set fixed in advance: calendar (Sydney hour, day of week) plus four lagged-MEF features using data up to the day before (`cal+lag`). Month, season and holidays not used.
- Alpha grid 0.01 to 10,000; choose the alpha with the lowest validation MAE (all hours); within 1e-6, take the larger alpha. Result: alpha = 100.
- "Beats the baseline" defined in advance as: validation MAE **and** validation regret (label C) both lower than the hourly-average baseline. Result: MAE lower, regret about 1% higher, so "partly".

## Gradient boosting (set 12 October, before the first fit; applied in `22ca5b4`)
- Same 24 features as Ridge, fitted on training nights only. Fixed settings: learning rate 0.05, at least 40 samples per leaf, no early stopping, random seed 0.
- Grid: depth {2, 3, 5} x iterations {50, 100, 200, 400}; choose the lowest validation MAE; within 1e-6, take the simpler model. Result: depth 3, 100 iterations.
- **Replace Ridge only if gradient boosting has both lower validation MAE and lower validation regret (label C) than Ridge.** Result: MAE 0.1788 vs 0.1775 and regret 1.274 vs 1.236, so Ridge was kept.

## Paired comparisons (method set 12 October, applied in `22ca5b4`; reading completed 20 October)
- Per-night difference in regret between two strategies on the same nights; 95% interval by bootstrap over nights (10,000 resamples, seed 0, percentile interval).
- Fixed on 12 October, before the comparison was run: a forecast counts as "better than fixed charging" only if the mean difference is below 0 **and** the whole interval is below 0; it counts as robust only if this holds under all three battery assumptions (B, C, C-fwd). Otherwise it reads "no clear difference".
- Regret under each label is measured against that label's own best four hours (defined with the label diagnostics, 11 October).

## Test-set protocol (set 17 and 20 October; applied in `b17f595`, `8cffa41`, `0c6ca64`)
- Validation-night forecasts use the model fitted on training nights. Test-night forecasts use the same settings refitted on training plus validation nights (decided 17 October); no test observation is used for fitting.
- The model delivered for scheduling is Ridge regardless of test results; gradient boosting is evaluated on test nights for completeness only (set 20 October, before the run).
- The full reading rule for paired comparisons was written down on 20 October, before the test run: interval entirely below 0 = "better"; entirely above 0 = "worse"; contains 0 = "no clear difference".
- The evaluation script was committed before it was run (`8cffa41`), run once on 20 October 2026, and its outputs committed as `0c6ca64` with the tag `results-v1`. The script refuses to run again while `outputs/test_run_log-v1.txt` exists.

## Added or formalised after seeing results (labelled as such)
- **`weekmean` reference baseline** (same hour, 2 to 7 days earlier) and the **label diagnostics**: added after the first Ridge results (11 October).
- **`mae_pure`** (MAE on hours with no battery-priced interval): added on 12 October after the label-diagnostic results, as a supplement; the `nonbatt` group definition was kept unchanged.
- **"Worse" verdict**: the 12 October rule defined only "better" and "no clear difference". The label `higher` (interval entirely above 0) was added by the coding assistant when implementing Step 4c, and became a written rule only on 20 October (test protocol above). On 20 October Kunyu decided that the one validation case with an interval entirely above 0 for Ridge (Ridge vs fixed under label B, +0.089 [0.017, 0.177] kg/night) is reported as "**no better than** fixed charging", not "worse".
- **Night-averaged interval** for the gradient boosting vs Ridge comparison on battery-free hours (`outputs/pure_hours_paired_nightavg-v1.csv`): computed after the hour-averaged result was seen; its reading rule was fixed before computing it.
