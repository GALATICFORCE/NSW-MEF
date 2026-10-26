## 4.2 Forecast Model Development and Selection

At 00:00 Sydney time on day D the model forecasts the hourly MEF (label C,
Section 4.1) for that night's charging window, D 18:00 to D+1 07:00 — 13 hours,
a lead time of 18 to 31 hours — using only data available by 23:59 on D−1.

All models share 24 features: one-hot indicators for the 13 hours and the 7 days
of the week, and four lagged-MEF features (the same hour on D−1; the mean of the
D−1 hours inside the charging window; the D−1 daily mean; the same-hour mean over
D−2 to D−7). Month, season and holiday indicators were omitted because one year of
data contains each season only once.

Three benchmarks were used: *yesterday* (same hour on D−1), *hourmean* (the
training-period mean for each hour, which always selects 18:00–22:00 and so equals
the fixed schedule) and *weekmean* (same-hour mean over D−2 to D−7).

Ridge regression (lag features standardised on training nights; α chosen from
0.01–10,000 by validation MAE, giving 100) was the primary model because its
coefficients can be read directly. A gradient-boosted tree ensemble
(HistGradientBoostingRegressor; depth and tree count chosen from a 3 × 4 grid by
validation MAE, giving depth 3, 100 trees) tested whether a non-linear model would
capture the step-like changes in MEF between hydro- and coal-priced hours. By a
rule fixed before it was trained, it would replace Ridge only if both validation
MAE and validation regret were lower; it achieved neither (MAE 0.1788 vs 0.1775
tCO2e/MWh; regret 1.274 vs 1.236 kg/night), so Ridge was carried forward.

## 4.3 Training, Validation and Evaluation

The 357 usable nights (2025-07-08 to 2026-06-29) were split chronologically into
250 training nights (to 2026-03-14), 54 validation nights (2026-03-15 to
2026-05-07) and 53 test nights (2026-05-08 to 2026-06-29). The split was committed
before any model was fitted and not changed afterwards.

Leakage was controlled in three ways: every lag feature asserts that no value
later than D−1 23:59 is used (for the hours after midnight, "yesterday" is 48
hours earlier); test-night labels were removed before features were built, and
every training and diagnostic script asserts that no test night is loaded; all
hyper-parameters were chosen on validation nights only.

For the test forecasts, the selected Ridge model was refitted with the same
hyper-parameters on the combined training and validation nights (304 nights,
3,952 hours). The evaluation script was committed before it was run, was run once,
and refuses to run again; its outputs are tagged `results-v1`.

Numerical accuracy and scheduling value are reported separately. Accuracy is the
MAE (tCO2e/MWh), over all hours and over hours with no battery-priced interval.
Scheduling value is the nightly regret: the actual emissions (label C, 2.3 kWh per
hour) of the four hours a method selects, minus those of the four best hours that
night, in kg CO2e per night; the within-night Spearman rank correlation is also
reported. Regret is re-scored under labels B and C-fwd. Methods are compared on
the same nights: each pair's mean nightly difference is given with a 95%
bootstrap interval (10,000 resamples), and a method is called better than fixed
charging only if the whole interval lies below zero under all three labels.
