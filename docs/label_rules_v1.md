# MEF label rules v1 (frozen version, for review on Wednesday)

> Translated from the Chinese original on 2026-10-30; no rule was changed. The original text is at tag `label-rules-v1`.

## 0. Declaration

- **MEF values the drafter had seen beforehand**:
  - hourly mean MEF for 2026H1 under the two treatments "drop" and "set to 0";
  - monthly mean MEF from 2025-07 to 2026-06 under the two treatments;
  - MEF values of individual intervals while debugging;
  - the full-year maximum, minimum and quantiles of mef_nemed reported by item 5 of the 5-minute table acceptance checks, and the raw price setter records of those two intervals;
  - quality counts computed before freezing to settle open items: the number of intervals with mef_nemed < 0 and > 1.5 among nonbatt intervals with normal w.
- **Basis for drafting**: only the direction distribution tables from acceptance item 8 (`data/processed/check8_share_2025H2.csv`, `check8_share_2026H1.csv`), and counts computed only from status, batt_dir, the flag columns, the w columns and timestamps. No A/B/C/C-fwd label was computed.
- **The window boundaries were set after seeing the full-year direction distribution.**
- **The w filter in section 3 was proposed after seeing the extreme values of mef_nemed.** The decision is based only on the w columns, not on MEF values; the sensitivity analysis keeps a version "without the w filter".

## 1. Inputs and time conventions

- Input: `data/processed/mef_nsw1_5min_full.csv`, produced by `src/build_5min_full.py`.
- Interval start = Time − 5 min, in NEM time (AEST, UTC+10, no daylight saving). Hours and windows are determined in NEM time.
- **Forecast day**: the interval start converted to the Sydney local date (Australia/Sydney).
  - Within the study period, daylight saving (AEDT) covers NEM time 2025-10-05 02:00 to 2026-04-05 01:55.
  - During this period, intervals in NEM 23:00–24:00 belong to the next Sydney date: 2,184 intervals, 411 of them battery-priced.
- **Window segment** = (forecast day, window). During daylight saving, the evening discharge window is cut by the day boundary: the NEM 23:00 hour becomes a separate window segment at the start of the next forecast day.
- **Basslink**: as in nemed, the ENOF records of Basslink (`T-V-MNSP1`) do not count as ENOF price-setting units.
  - not included in w_enof;
  - not counted as "has ENOF" when determining status;
  - recorded only in the separate diagnostic column `w_basslink`.
- **w_total** = w_enof + w_gen + w_load (w_enof excludes Basslink).
- Four labels:

| Label | Meaning | Use |
|---|---|---|
| `mef_A` | battery set to 0 | reference only, not used to grade conclusions |
| `mef_B` | same-window counterfactual | forecast target + ex-post evaluation |
| `mef_C` | daily-balance counterfactual | forecast target + ex-post evaluation |
| `mef_Cfwd` | recharge-later counterfactual | **ex-post evaluation only** (hour selection and emissions evaluation under perfect foresight), not a forecast target |

C-fwd pairing can reach into the next day. If it were used as a forecast target, lag features such as "same hour yesterday" would carry information from the next day, causing leakage.

## 2. Windows

**Rule**: in each NEM hour, check whether charging or discharging is the majority (>50%) among battery-priced intervals (status batt or mixed); consecutive hours with the same majority direction form one window.

The `charge` share = (batt_charge + mixed_charge) / (batt + mixed). Computing it from batt intervals only does not change the majority direction of any hour.

| Window | Type | 2025H2 | 2026H1 |
|---|---|---|---|
| Early morning | charge | 01:00–04:00 | 00:00–05:00 |
| Morning peak | discharge | 04:00–08:00 | 05:00–07:00 |
| Daytime | charge | 08:00–16:00 | 07:00–16:00 |
| Evening | discharge | 16:00–01:00 (across midnight) | 16:00–24:00 |

The two sets differ only in three hours, 0, 4 and 7:

| Hour | 2025H2 charge share | 2026H1 charge share | Battery intervals (H2 / H1) |
|---|---|---|---|
| 0 | 0.390 | 0.538 | 213 / 398 |
| 4 | 0.404 | 0.582 | 228 / 498 |
| 7 | 0.420 | 0.574 | 462 / 1,114 |

All three hours are charge/discharge transitions, with shares between 0.39 and 0.58.

**Main specification: the 2026H1 set is used for the whole year.** Three reasons:

1. 2026H1 has 23,021 battery-priced intervals and 2025H2 has 10,845, so about two thirds of the intervals whose labels depend on the battery treatment are in H1.
2. 2026H1 is closer to the current battery fleet size.
3. Applying the H1 set to H2 puts only 903 intervals in hours with the opposite majority direction; applying the H2 set to H1 would put 2,010 there.

**Sensitivity analysis**: each half-year uses its own set.

The four windows cover all 24 hours of the day; no interval falls outside a window.

## 3. Source intervals and window values

- **Source intervals**: intervals meeting all four conditions below, 70,731 in total:
  - `status == nonbatt`
  - `ef_missing == False`
  - `has_ldof == False`
  - 0.5 ≤ w_total ≤ 1.5
- **Window value** S(segment) = mean `mef_nemed` of all source intervals in that window segment.

## 4. Pairing

### 4.1 Three counterfactuals

Example: an evening-peak interval priced by battery discharge, with 1 MWh more consumed:

| Label | Load type | Battery state | How the extra discharge is made up |
|---|---|---|---|
| B | unexpected load | energy-limited | 1 MWh less is discharged in the same window, made up by the non-battery units at that time |
| C | expected load | daily balance | 1/η MWh extra was already charged for it earlier that day |
| C-fwd | unexpected load | not energy-limited | 1/η MWh extra is charged later to make it up, possibly not until the next day |

### 4.2 Pairing rules

- **B**: a battery record takes the value of the window segment it is in, for both charge and discharge. Values are taken only within the same forecast day.
- **C, discharge records (BDOF,GEN)**:
  - if the interval falls in a charge window, take that window segment;
  - otherwise, take the nearest charge window segment in the same forecast day that ends before it.
- **C-fwd, discharge records (BDOF,GEN)**:
  - if the interval falls in a charge window, take that window segment;
  - otherwise, take the nearest charge window segment that starts after it, **which may be on the next day**.
- **C and C-fwd, charge records (BDOF,LOAD)**, identical for both:
  - if the interval falls in a discharge window, take that window segment;
  - otherwise, take the nearest discharge window segment in the same forecast day that starts after it.

## 5. Fallback

N = 6, i.e. 30 minutes of source intervals. Try the three tiers below in order; the first tier with ≥ N source intervals is used, and the value is its mean:

| Tier | Source for B | Source for C, C-fwd |
|---|---|---|
| 1 | the window segment the interval is in | the window segment paired under section 4 |
| 2 | all window segments on the same day of the same type as the interval's window | all window segments on the same day of the required type (all charge segments for discharge records, all discharge segments for charge records) |
| 3 | all source intervals on the same day | all source intervals on the same day |

- **"Same day"**: for B and C, the interval's own forecast day. For C-fwd discharge records, **the day of the paired window segment**, i.e. the next day when the pairing crosses days.
- **The last day of the data** (Sydney date 2026-06-30) has no next day. C-fwd discharge records that would need to cross into the next day start directly from tier 2 on the same day.
- If none of the three tiers has ≥ N, the value is NaN.
- Each value gets its own column recording which tier was actually used: `tier_B`, `tier_C_gen`, `tier_C_load`, `tier_Cfwd_gen`. The charge side of C-fwd is the same as C and reuses `tier_C_load`.
- Typical case: during daylight saving, a C discharge record in the first hour of a forecast day finds no "earlier" charge segment on that day and goes straight to tier 2.

## 6. Label formula

Follows nemed's composition: intensity = Σ (Increase × emission factor), not normalised. Only the emission factors of battery records are replaced with counterfactual values:

```
base   = mef_nemed     (status nonbatt or mixed)
       = 0             (status batt)
mef_X  = base + w_gen × EF_gen^X + w_load × EF_load^X
```

| Label | EF_gen | EF_load |
|---|---|---|
| A | 0 | 0 |
| B | S(own segment) | S(own segment) |
| C | S(earlier charge segment) ÷ η | η × S(later discharge segment) |
| C-fwd | S(later charge segment, may cross days) ÷ η | η × S(later discharge segment) |

- For nonbatt intervals w_gen = w_load = 0, so all four labels equal mef_nemed.
- Intervals with batt_dir = both need no special handling: the GEN and LOAD sides each use their own emission factor. If the value needed on either side is NaN, the whole label is NaN.
- Battery records with negative Increase (`n_neg_batt`) are included with their original sign, consistent with how nemed treats negative ENOF values.
- η = 0.85, a fixed value in v1. The sensitivity analysis uses 0.80 and 0.90, which affects C and C-fwd.

## 7. Set to NaN

All A/B/C/C-fwd labels are NaN for the following intervals, 647 in total:

- status is other or norecord;
- has_ldof is true;
- ef_missing is true;
- w_total is not within [0.5, 1.5].

In addition, values for which none of the three fallback tiers of section 5 is met are also NaN.

## 8. Grading conclusions

- **Robust conclusion**: holds under all three settings B, C and C-fwd.
- **Depends on the storage assumption**: holds under only some of the settings. Which ones must be stated.
- A is a reference only and is not used to grade conclusions.

## 9. Sensitivity analyses

1. η = 0.80 and 0.90 (affects C and C-fwd).
2. Windows: each half-year uses its own set.
3. No w filter: drop the w_total condition from section 3 and the corresponding NaN rule from section 7.

## 10. How the open items of the draft are handled in v1

No open items remain after freezing.

1. **w_enof includes Basslink**: changed, see section 1. With Basslink removed, the status counts and batt_dir counts do not change; the 1% quantile of w_total for mixed and nonbatt goes from 0.075 and 0.000 back to 0.787 and 0.871.

   **Whether to use a w filter**: adopted in v1. Intervals with w_total outside [0.5, 1.5] have all four labels set to NaN and are not used as source intervals.
   - Intervals affected: 151 nonbatt, 95 batt, 25 mixed.
   - Source intervals therefore drop from 70,875 to 70,731.
   - Reason: a sum of Increase far from 1 indicates ill-conditioned price-setter sensitivities (units at the same price counted more than once, or large values in opposite directions cancelling out). nemed's formula is not normalised and carries this distortion straight into the MEF.
2. **Whether to add a `has_basslink` flag**: not handled in v1.
   - Intervals with a large Basslink share have w_enof well below 1 and are removed by the w filter of item 1 (11 of them among nonbatt).
   - The remaining nonbatt intervals with Basslink records (2,684 of the 2,695) keep nemed's value, which amounts to treating the Basslink contribution as 0. This is a known limitation of v1. The `w_basslink` column is kept for diagnostics.
3. **Extreme values in nonbatt intervals themselves**: apart from the w filter of item 1, not handled in v1.
   - Among nonbatt intervals with normal w, 248 still have mef_nemed < 0 and 54 have mef_nemed > 1.5.
   - The cause is Increase values in opposite directions cancelling out (for example, a hydro unit with positive Increase and a coal unit with negative Increase). This can occur in its own right in a price-setter-based method, so the original values are kept and the counts are stated.
4. **Pairing at 7:00 and 15:00**: not handled in v1; the rules of section 4 apply with no special treatment. The sensitivity analysis with separate half-year windows (section 9, item 2) partly reflects the effect of the boundary choice.

## 11. Revision history

**Changes from draft `8bf7280`:**

1. Basslink is no longer included in w_enof and does not count as "has ENOF"; new diagnostic column `w_basslink`. (`src/build_5min_full.py`)
2. New C-fwd variant:
   - pairing and fallback rules in sections 4 and 5;
   - used for ex-post evaluation only;
   - the meaning of the three counterfactuals B, C and C-fwd is stated;
   - a robust conclusion now has to hold under all three settings.
3. Each of the four open items in the former section 9 is given its v1 treatment, see section 10. New w filter (sections 3 and 7) and the corresponding sensitivity analysis (section 9, item 3).
4. η fixed at 0.85; the sensitivity analysis uses 0.80 and 0.90.
5. The declaration now also lists the values seen before this freeze.
