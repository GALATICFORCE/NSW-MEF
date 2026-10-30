"""5-minute MEF labels; rules in docs/label_rules_v1.md (tag label-rules-v1)

Tier columns (rules section 5): 1 / 2 / 3 = the tier actually used; 0 = none of the three tiers has N source intervals,
and the label is NaN; empty = not applicable (the interval has no battery record on this side, or section 7 set it to NaN).
"""
import numpy as np
import pandas as pd

FULL = "data/processed/mef_nsw1_5min_full.csv"
OUT = "data/processed/mef_5min_labels_rules-v1.csv"
LABELS = ("A", "B", "C", "Cfwd")
W_RANGE = (0.5, 1.5)
ETA = 0.85
N_MIN = 6
# Rules section 2: the 2026H1 set is used for the whole year; NEM hours [start, end)
WINDOWS = [("night", "charge", 0, 5), ("morning", "discharge", 5, 7),
           ("day", "charge", 7, 16), ("evening", "discharge", 16, 24)]


def load_full(path=FULL):
    return pd.read_csv(path, parse_dates=["Time"])


def w_total(df):
    return df["w_enof"] + df["w_gen"] + df["w_load"]


def nan_reason(df):
    """Rules section 7, shared by all four labels. Empty string = not set to NaN;
    if several reasons apply, the one listed first wins"""
    w = w_total(df)
    reason = pd.Series("", index=df.index)
    # Write from the lowest priority up; later writes overwrite earlier ones
    reason[(w < W_RANGE[0]) | (w > W_RANGE[1])] = "w_total"
    reason[df["ef_missing"]] = "ef_missing"
    reason[df["has_ldof"]] = "has_ldof"
    status_nan = df["status"].isin(["other", "norecord"])
    reason[status_nan] = "status_" + df.loc[status_nan, "status"]
    return reason


def nan_mask(df):
    return nan_reason(df) != ""


def is_source(df):
    """Rules section 3: window values are taken only from these intervals"""
    return (df["status"] == "nonbatt") & ~nan_mask(df)


def segments(df):
    """Tag each interval with its forecast day, window and window segment; returns (segment id per interval, segment table)

    A window segment = a run of consecutive intervals with the same forecast day and window. During daylight saving, NEM hour 23
    belongs to the next Sydney date, so that day starts with an extra, separate evening discharge segment (rules section 1).
    """
    start = df["Time"] - pd.Timedelta("5min")
    hour = start.dt.hour
    day = start.dt.tz_localize("Australia/Brisbane").dt.tz_convert("Australia/Sydney").dt.date
    window = pd.Series(pd.NA, index=df.index, dtype="object")
    kind = window.copy()
    for name, k, h0, h1 in WINDOWS:
        in_w = (hour >= h0) & (hour < h1)
        window[in_w], kind[in_w] = name, k
    assert window.notna().all()

    key = day.astype(str) + "|" + window
    seg = (key != key.shift()).cumsum()
    src = is_source(df)
    value = df["mef_nemed"].where(src)
    table = pd.DataFrame({"day": day, "kind": kind, "start": start, "src": src, "value": value}).groupby(seg).agg(
        day=("day", "first"), kind=("kind", "first"), start=("start", "min"),
        n_src=("src", "sum"), mean=("value", "mean"))

    # Segment id used for pairing (rules section 4). If the interval's own window type matches, use its own segment;
    # otherwise search backward / forward
    charge_id = table.index.to_series().where(table["kind"] == "charge")
    discharge_id = table.index.to_series().where(table["kind"] == "discharge")
    table["prev_charge_same_day"] = charge_id.groupby(table["day"]).ffill()
    table["next_discharge_same_day"] = discharge_id.groupby(table["day"]).bfill()
    table["next_charge_any_day"] = charge_id.bfill()
    return seg, table


def resolve(paired, day, kind, table):
    """Three-tier fallback of rules section 5; paired = paired segment id (may be NaN),
    day / kind = date and window type used by tiers 2 and 3"""
    by_kind = table.assign(s=table["mean"] * table["n_src"]).groupby(["day", "kind"])[["n_src", "s"]].sum()
    by_day = by_kind.groupby(level="day").sum()

    t1 = table.reindex(paired.to_numpy())
    t2 = by_kind.reindex(pd.MultiIndex.from_arrays([day.to_numpy(), kind.to_numpy()]))
    t3 = by_day.reindex(day.to_numpy())
    ok1 = (t1["n_src"] >= N_MIN).to_numpy()
    ok2 = (t2["n_src"] >= N_MIN).to_numpy()
    ok3 = (t3["n_src"] >= N_MIN).to_numpy()
    value = np.select([ok1, ok2, ok3], [t1["mean"].to_numpy(), (t2["s"] / t2["n_src"]).to_numpy(),
                                        (t3["s"] / t3["n_src"]).to_numpy()], default=np.nan)
    tier = np.select([ok1, ok2, ok3], [1, 2, 3], default=0)
    return pd.Series(value, index=paired.index), pd.Series(tier, index=paired.index)


def battery_fills(df, name, eta):
    """Returns (fill_gen, fill_load, dict of tier columns): counterfactual emission factors for battery GEN / LOAD
    records (rules section 6)"""
    seg, table = segments(df)
    own = table.reindex(seg.to_numpy()).set_index(df.index)
    has_gen = df["batt_dir"].isin(["discharge", "both"]) & ~nan_mask(df)
    has_load = df["batt_dir"].isin(["charge", "both"]) & ~nan_mask(df)

    def tier_col(tier, applies):
        return tier.where(applies).astype("Int64")

    if name == "B":
        own_seg = pd.Series(seg.to_numpy(), index=df.index, dtype=float)
        value, tier = resolve(own_seg, own["day"], own["kind"], table)
        return value, value, {"tier_B": tier_col(tier, has_gen | has_load)}

    charge = pd.Series("charge", index=df.index)
    discharge = pd.Series("discharge", index=df.index)
    # Charge records: same for C and C-fwd; the discharge segment the interval is in, or the next one, within the same
    # forecast day
    load_value, load_tier = resolve(own["next_discharge_same_day"], own["day"], discharge, table)
    if name == "C":
        gen_value, gen_tier = resolve(own["prev_charge_same_day"], own["day"], charge, table)
        tiers = {"tier_C_gen": tier_col(gen_tier, has_gen), "tier_C_load": tier_col(load_tier, has_load)}
    else:
        # C-fwd discharge records: the charge segment the interval is in, or the next one, possibly on a later day; tiers 2
        # and 3 use the paired segment's day, or the same day if there is no paired segment (last day of the data)
        paired = own["next_charge_any_day"]
        paired_day = table["day"].reindex(paired.to_numpy()).set_axis(df.index)
        gen_value, gen_tier = resolve(paired, paired_day.fillna(own["day"]), charge, table)
        tiers = {"tier_Cfwd_gen": tier_col(gen_tier, has_gen)}
    return gen_value / eta, eta * load_value, tiers


def compute_label(df, name, eta=ETA):
    """Returns Time, label_<name> and the matching tier columns"""
    if name == "A":
        fill_gen = fill_load = pd.Series(0.0, index=df.index)
        tiers = {}
    elif name in LABELS:
        fill_gen, fill_load, tiers = battery_fills(df, name, eta)
    else:
        raise ValueError(f"unknown label {name!r}, expected one of {LABELS}")

    # Rules section 6: base is nemed's non-battery part; batt intervals have no non-battery part;
    # an interval with no battery record on a side gets 0 from that side (so NaN fill values do not propagate)
    base = df["mef_nemed"].where(df["status"] != "batt", 0.0)
    gen_part = (df["w_gen"] * fill_gen).where(df["batt_dir"].isin(["discharge", "both"]), 0.0)
    load_part = (df["w_load"] * fill_load).where(df["batt_dir"].isin(["charge", "both"]), 0.0)
    label = base + gen_part + load_part
    out = pd.DataFrame({"Time": df["Time"], f"label_{name}": label.mask(nan_mask(df))})
    for col, tier in tiers.items():
        out[col] = tier
    return out


if __name__ == "__main__":
    df = load_full()
    out = df[["Time"]].copy()
    for name in LABELS:
        out = out.merge(compute_label(df, name), on="Time")
    out["status"] = df["status"]
    out["batt_dir"] = df["batt_dir"]
    out["w_total"] = w_total(df)
    out["nan_reason"] = nan_reason(df)
    out.to_csv(OUT, index=False)
    print(f"rows: {len(out)} | rule-based NaN intervals: {nan_mask(df).sum()} -> {OUT}")
    for name in LABELS:
        print(f"label_{name} NaN: {out[f'label_{name}'].isna().sum()}")
