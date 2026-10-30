"""Per-night evaluation metrics (hand-off doc §4.5): MAE, regret, within-night Spearman ρ; summarised by month, in four groups

Input has one row per hour, with at least the columns night_of, hour_start_nem, pred, truth, nonbatt_majority, n_battfill;
pass one split at a time (the high and low battery-share halves are split at the median within that split).
Computed only on hours where truth is not null; pred must not be null on those hours.

Groups (decided 10-11):
- all: all hours; MAE, regret, Spearman
- nonbatt: MAE only, on hours where nonbatt is the majority (at least 7 of the 12 intervals are nonbatt); these hours
  can still contain battery-filled intervals (regret / Spearman on a subset of hours would be a different scheduling problem)
- batt_share_high / batt_share_low: nights split into two halves at the within-split median of the battery price-setting
  share (share of 5-minute intervals with status ∈ {batt, mixed}); MAE, regret, Spearman on whole nights; nights at
  the median go to low
"""
import numpy as np
import pandas as pd

K_HOURS = 4            # charge for 4 hours each night
KWH_PER_HOUR = 2.3     # 2.3 kW × 1 h; MEF in t/MWh = kg/kWh, so the product is in kg
INTERVALS_PER_HOUR = 12


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def per_night(df, whole_night=True):
    """One row per night: n_hours, mae, regret_kg, spearman.

    regret = extra true emissions (kg) of the 4 hours picked by the forecast compared with the 4 hours picked by the truth.
    With whole_night=False (nonbatt group, only some hours) only MAE is computed.
    """
    rows = []
    for night, g in df.sort_values("hour_start_nem").groupby("night_of"):
        g = g[g["truth"].notna()]
        if g["pred"].isna().any():
            raise ValueError(f"NaN prediction on night {night:%Y-%m-%d}")
        n = len(g)
        mae = (g["pred"] - g["truth"]).abs().mean() if n else np.nan
        regret = rho = np.nan
        if whole_night:
            # ties go to the earlier hour
            picked = g.nsmallest(K_HOURS, "pred", keep="first")["truth"].sum()
            best = g.nsmallest(K_HOURS, "truth", keep="first")["truth"].sum()
            regret = KWH_PER_HOUR * (picked - best)
            # when both pick the same 4 hours, a different summation order can differ by a rounding error
            assert regret >= -1e-9, f"negative regret on night {night:%Y-%m-%d}: {regret}"
            rho = spearman(g["pred"], g["truth"])
        rows.append((night, n, mae, regret, rho))
    return pd.DataFrame(rows, columns=["night_of", "n_hours", "mae", "regret_kg", "spearman"])


def batt_share_half(df):
    """Battery price-setting share per night, split into high / low at the median within the given split"""
    g = df.groupby("night_of")
    share = g["n_battfill"].sum() / (INTERVALS_PER_HOUR * g.size())
    return pd.Series(np.where(share > share.median(), "high", "low"), index=share.index), share


def _summary(nightly, whole_night=True):
    """With whole_night=False (nonbatt group), regret and Spearman are not summarised and are written as NaN"""
    nan = float("nan")
    return pd.Series({
        "n_nights": int(nightly["mae"].notna().sum()),
        "n_hours": int(nightly["n_hours"].sum()),
        "mae": nightly["mae"].mean(),
        "regret_kg_mean": nightly["regret_kg"].mean() if whole_night else nan,
        "regret_kg_median": nightly["regret_kg"].median() if whole_night else nan,
        "regret_kg_p90": nightly["regret_kg"].quantile(0.9) if whole_night else nan,
        "n_nights_regret": int(nightly["regret_kg"].notna().sum()) if whole_night else 0,
        "spearman_mean": nightly["spearman"].mean() if whole_night else nan,
        "spearman_median": nightly["spearman"].median() if whole_night else nan,
        "n_nights_spearman": int(nightly["spearman"].notna().sum()) if whole_night else 0,
    })


def evaluate(df):
    """Four groups × (each month + all); returns (summary table, per-night detail)"""
    assert df["split"].nunique() == 1, "pass one split at a time"
    half, share = batt_share_half(df)
    groups = {
        "all": (df, True),
        "nonbatt": (df[df["nonbatt_majority"]], False),
        "batt_share_high": (df[df["night_of"].map(half) == "high"], True),
        "batt_share_low": (df[df["night_of"].map(half) == "low"], True),
    }
    summaries, nightly_all = [], []
    for group, (sub, whole_night) in groups.items():
        nightly = per_night(sub, whole_night).assign(group=group)
        month = nightly["night_of"].dt.to_period("M").astype(str)
        s = nightly.groupby(month).apply(_summary, whole_night=whole_night)
        s.loc["all"] = _summary(nightly, whole_night)
        summaries.append(s.rename_axis("month").reset_index().assign(group=group))
        nightly_all.append(nightly)
    summary = pd.concat(summaries)[["group", "month"] + list(_summary(nightly_all[0]).index)]
    nightly = pd.concat(nightly_all, ignore_index=True)
    nightly["batt_share"] = nightly["night_of"].map(share)
    return summary.reset_index(drop=True), nightly
