"""Step 4c: model 2, HistGradientBoosting (hand-off doc §7), train and val nights only

Inputs are exactly the same 24 feature columns as ridge-v1 (no scaling, no re-encoding); the target is label C; fitted
on train only; forecasts are not clipped.
Grid max_depth ∈ {2, 3, 5} × max_iter ∈ {50, 100, 200, 400}: each depth is fitted for 400 iterations and staged_predict
gives the intermediate iteration counts.
Selection: lowest validation MAE in group all; within 1e-6, take the simpler one (smaller max_depth first, then smaller max_iter).

Usage: python src\\hgb.py   (run features.py first)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from baselines import hourmean
from features import FEATURES
from metrics import evaluate, per_night
from nights import night_hours
from ridge import headline, load, make_model

DEPTHS = [2, 3, 5]
ITERS = [50, 100, 200, 400]
FIXED_PARAMS = dict(loss="squared_error", learning_rate=0.05, min_samples_leaf=40, l2_regularization=0.0,
                    early_stopping=False, random_state=0)
RIDGE_ALPHA = 100
FIXED_HOURS = [18, 19, 20, 21]
SPLITS = ("train", "val")
TRUTHS = ["B", "C", "Cfwd"]
N_BOOT = 10_000
PAIRS = [("yesterday", "fixed"), ("weekmean", "fixed"), ("ridge-v1", "fixed"), ("hgb-v1", "fixed"),
         ("hgb-v1", "ridge-v1")]
OUT_SCAN = "outputs/hgb_scan-v1.csv"
OUT_METRICS = "outputs/metrics_{split}_hgb-v1.csv"
OUT_COMPARE = "outputs/model_compare-v1.csv"
OUT_NIGHTLY = "outputs/nightly_regret-v1.csv"
OUT_PAIRED = "outputs/paired_regret-v1.csv"


def fit_hgb(df, depth, n_iter):
    train = df["split"] == "train"
    return HistGradientBoostingRegressor(max_depth=depth, max_iter=n_iter, **FIXED_PARAMS).fit(
        df.loc[train, FEATURES], df.loc[train, "y"])


def split_eval(df, pred, split):
    return evaluate(df[df["split"] == split].assign(pred=pred[df["split"] == split]))[0]


def scan(df):
    rows, preds = [], {}
    for depth in DEPTHS:
        model = fit_hgb(df, depth, max(ITERS))
        for n_iter, p in enumerate(model.staged_predict(df[FEATURES]), start=1):
            if n_iter not in ITERS:
                continue
            pred = pd.Series(p, index=df.index)
            preds[(depth, n_iter)] = pred
            for split in SPLITS:
                mae, regret, rho = headline(split_eval(df, pred, split))
                rows.append({"max_depth": depth, "max_iter": n_iter, "split": split, "mae": mae,
                             "regret_kg_mean": regret, "spearman_mean": rho})
    return pd.DataFrame(rows), preds


def pick(scan_df):
    val = scan_df[scan_df["split"] == "val"]
    cand = val[val["mae"] <= val["mae"].min() + 1e-6].sort_values(["max_depth", "max_iter"])
    return int(cand.iloc[0]["max_depth"]), int(cand.iloc[0]["max_iter"])


def nightly(df, preds, truths):
    """Regret per night: split × truth label × strategy"""
    rows = []
    for split in SPLITS:
        sub = df[df["split"] == split]
        for truth in TRUTHS:
            for name, pred in preds.items():
                n = per_night(sub.assign(truth=truths[truth][sub.index], pred=pred[sub.index]))
                rows.append(n[["night_of", "regret_kg"]].assign(split=split, truth=truth, strategy=name))
    out = pd.concat(rows, ignore_index=True)
    return out[["split", "night_of", "truth", "strategy", "regret_kg"]]


def paired(nightly_df):
    rows = []
    for (split, truth), g in nightly_df.groupby(["split", "truth"], sort=False):
        wide = g.pivot(index="night_of", columns="strategy", values="regret_kg")
        for strat, ref in PAIRS:
            d = (wide[strat] - wide[ref]).to_numpy()
            rng = np.random.default_rng(0)
            boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            verdict = "lower" if hi < 0 else ("higher" if lo > 0 else "indistinguishable")
            rows.append({"split": split, "truth": truth, "strategy": strat, "reference": ref, "n_nights": len(d),
                         "diff_mean": d.mean(), "ci_low": lo, "ci_high": hi,
                         "n_lower": int((d < -1e-9).sum()), "n_same": int((np.abs(d) < 1e-9).sum()),
                         "n_higher": int((d > 1e-9).sum()), "verdict": verdict})
    return pd.DataFrame(rows)


def compare(df, preds, truths, pure):
    rows = []
    for split in SPLITS:
        sub = df[df["split"] == split]
        for name in ["hourmean", "yesterday", "weekmean", "ridge-v1", "hgb-v1"]:
            pred = preds[name]
            summ = split_eval(df, pred, split).set_index(["group", "month"])
            s = sub.assign(pred=pred[sub.index])
            p = s[pure[s.index]]
            nights = s.groupby("night_of")
            row = {"split": split, "model": name,
                   "mae_all": summ.loc[("all", "all"), "mae"], "n_all": int(summ.loc[("all", "all"), "n_hours"]),
                   "mae_nonbatt_majority": summ.loc[("nonbatt", "all"), "mae"],
                   "n_nonbatt_majority": int(summ.loc[("nonbatt", "all"), "n_hours"]),
                   "mae_pure": per_night(p, whole_night=False)["mae"].mean(), "n_pure": len(p)}
            for truth in TRUTHS:
                n = per_night(sub.assign(truth=truths[truth][sub.index], pred=pred[sub.index]))
                row[f"regret_{truth}"] = n["regret_kg"].mean()
                if truth == "C":
                    row["spearman_C"] = n["spearman"].mean()
            row["pred_range_mean"] = (nights["pred"].max() - nights["pred"].min()).mean()
            row["truth_range_mean"] = (nights["truth"].max() - nights["truth"].min()).mean()
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df = load()
    assert set(df["split"]) == set(SPLITS), "test nights must not be loaded"

    scan_df, scan_preds = scan(df)
    scan_df.to_csv(OUT_SCAN, index=False)
    depth, n_iter = pick(scan_df)
    print("== HGB grid (all group, all months)")
    print(scan_df.pivot_table(index=["max_depth", "max_iter"], columns="split",
                              values=["mae", "regret_kg_mean", "spearman_mean"]).round(4).to_string())
    print(f"chosen hgb-v1: max_depth {depth}, max_iter {n_iter}")

    # Reproducibility: fit the same parameters twice, forecasts identical value by value; also compared with
    # staged_predict from the grid
    p1 = fit_hgb(df, depth, n_iter).predict(df[FEATURES])
    p2 = fit_hgb(df, depth, n_iter).predict(df[FEATURES])
    print(f"refit twice identical: {np.array_equal(p1, p2)} | "
          f"max diff vs staged_predict in the grid: {np.abs(p1 - scan_preds[(depth, n_iter)].to_numpy()).max():.2e}")
    assert np.array_equal(p1, p2)
    hgb = pd.Series(p1, index=df.index)

    for split in SPLITS:
        summary = split_eval(df, hgb, split)
        summary.to_csv(OUT_METRICS.format(split=split), index=False)
        print(f"\n== hgb-v1 | {split}")
        print(summary.query("month == 'all'").round(3).to_string(index=False))

    train = df["split"] == "train"
    ridge = make_model(FEATURES, RIDGE_ALPHA).fit(df.loc[train, FEATURES], df.loc[train, "y"])
    preds = {
        "fixed": pd.Series(np.where(df["hour_start_sydney"].dt.hour.isin(FIXED_HOURS), 0.0, 1.0), index=df.index),
        "hourmean": hourmean(df),
        "yesterday": df["lag1_same_hour"],
        "weekmean": df["lag2to7_same_hour_mean"],
        "ridge-v1": pd.Series(ridge.predict(df[FEATURES]), index=df.index),
        "hgb-v1": hgb,
    }
    truths = {"C": df["y"]}
    for lab in ["B", "Cfwd"]:
        h = night_hours(lab, SPLITS).set_index("hour_start_nem")["value"]
        truths[lab] = df["hour_start_nem"].map(h)
        assert truths[lab].notna().all()
    pure = df["n_battfill"] == 0

    comp = compare(df, preds, truths, pure)
    comp.to_csv(OUT_COMPARE, index=False)
    print("\n== model compare")
    print(comp.round(4).to_string(index=False))

    night = nightly(df, {k: preds[k] for k in ["fixed", "yesterday", "weekmean", "ridge-v1", "hgb-v1"]}, truths)
    night.assign(night_of=night["night_of"].dt.strftime("%Y-%m-%d")).to_csv(OUT_NIGHTLY, index=False)
    print(f"\nnightly_regret rows: {len(night)}")

    pr = paired(night)
    pr.to_csv(OUT_PAIRED, index=False)
    print("\n== paired regret (strategy - reference, kg/night)")
    print(pr.round(4).to_string(index=False))
