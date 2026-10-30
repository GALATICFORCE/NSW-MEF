"""Step 4: Ridge (hand-off doc §4.4, §7), train and val nights only

ridge-v1 = feature set cal+lag (fixed in advance, not chosen from validation results); cal is used only as a diagnostic.
The 4 lag features use StandardScaler (fitted on train only), one-hot columns are not scaled; Ridge has an intercept and is
fitted on train only; forecasts are not clipped.
alpha: lowest validation MAE in group all; within 1e-6, take the larger alpha.

Usage: python src\\ridge.py   (run features.py first)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines import OUT as BASELINE_OUT, hour_status_counts, hourmean, yesterday
from features import DOW_COLS, FEATURES, HOUR_COLS, LAG_COLS, OUT as FEATURES_FILE, lookup_hourly
from metrics import K_HOURS, KWH_PER_HOUR, evaluate
from nights import TEST_START

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]
FEATURE_SETS = {"cal": HOUR_COLS + DOW_COLS, "cal+lag": FEATURES}
MODEL, MODEL_SET = "ridge-v1", "cal+lag"
SPLITS = ("train", "val")
OUT_METRICS = "outputs/metrics_{split}_ridge-v1.csv"
OUT_SCAN = "outputs/ridge_alpha_scan-v1.csv"
OUT_COEFS = "outputs/ridge_coefs-v1.csv"
OUT_MAE_HOUR = "outputs/mae_by_hour_val-v1.csv"
OUT_CONTEXT = "outputs/regret_context-v1.csv"
FIXED_HOURS = [18, 19, 20, 21]   # plug in at 18:00 and charge until 22:00


def make_model(cols, alpha):
    lag = [c for c in cols if c in LAG_COLS]
    pre = ColumnTransformer([("lag", StandardScaler(), lag)], remainder="passthrough",
                            verbose_feature_names_out=False) if lag else "passthrough"
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha, fit_intercept=True))])


def load():
    df = pd.read_csv(FEATURES_FILE, parse_dates=["night_of", "hour_start_nem", "hour_start_sydney"])
    assert set(df["split"]) == set(SPLITS) and (df["night_of"] < TEST_START).all(), "test nights must not be loaded"
    counts = hour_status_counts()
    df["nonbatt_majority"] = df["hour_start_nem"].map(counts["n_nonbatt"]) >= 7
    df["n_battfill"] = df["hour_start_nem"].map(counts["n_battfill"])
    df["truth"] = df["y"]
    df["sydney_date"] = df["hour_start_sydney"].dt.normalize()
    return df


def evaluate_split(df, pred, split):
    sub = df[df["split"] == split].assign(pred=pred[df["split"] == split])
    return evaluate(sub)


def headline(summary):
    row = summary.query("group == 'all' and month == 'all'").iloc[0]
    return row["mae"], row["regret_kg_mean"], row["spearman_mean"]


def pick_alpha(scan, feature_set):
    """Lowest validation MAE in group all; within 1e-6, take the larger alpha"""
    val = scan[(scan["feature_set"] == feature_set) & (scan["split"] == "val")]
    best = val["mae"].min()
    return val.loc[val["mae"] <= best + 1e-6, "alpha"].max()


def regret_context(df):
    rows = []
    for split in SPLITS:
        opt, fixed = [], []
        for _, g in df[df["split"] == split].groupby("night_of"):
            best = g.nsmallest(K_HOURS, "truth", keep="first")["truth"].sum()
            opt.append(KWH_PER_HOUR * best)
            fixed_sum = g.loc[g["hour_start_sydney"].dt.hour.isin(FIXED_HOURS), "truth"].sum()
            fixed.append(KWH_PER_HOUR * (fixed_sum - best))
        rows.append({"split": split, "n_nights": len(opt), "optimal_kg_mean": pd.Series(opt).mean(),
                     "fixed_18_22_regret_kg_mean": pd.Series(fixed).mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df = load()
    train = df["split"] == "train"

    # Baselines are recomputed in the same evaluation and must be identical to the v2 files
    preds = {"yesterday": yesterday(df, lookup_hourly())[0], "hourmean": hourmean(df)}
    for model, pred in preds.items():
        for split in SPLITS:
            summary, _ = evaluate_split(df, pred, split)
            ref = pd.read_csv(BASELINE_OUT.format(split=split, model=model))
            num = summary.select_dtypes("number").columns
            new = summary[num].reset_index(drop=True)
            diff = (new - ref[num]).abs().max().max()
            assert summary[["group", "month"]].equals(ref[["group", "month"]]) and diff < 1e-12 \
                and new.isna().equals(ref[num].isna()), f"{model} {split} differs from {BASELINE_OUT}"
    print("baselines recomputed: identical to v2 files")

    scan, fitted = [], {}
    for fs, cols in FEATURE_SETS.items():
        for alpha in ALPHAS:
            model = make_model(cols, alpha).fit(df.loc[train, cols], df.loc[train, "y"])
            pred = pd.Series(model.predict(df[cols]), index=df.index)
            fitted[(fs, alpha)] = (model, pred)
            for split in SPLITS:
                mae, regret, rho = headline(evaluate_split(df, pred, split)[0])
                scan.append({"feature_set": fs, "alpha": alpha, "split": split, "mae": mae,
                             "regret_kg_mean": regret, "spearman_mean": rho})
    scan = pd.DataFrame(scan)
    scan.to_csv(OUT_SCAN, index=False)
    chosen = {fs: pick_alpha(scan, fs) for fs in FEATURE_SETS}
    print("\n== alpha scan (all group, all months)")
    print(scan.pivot_table(index=["feature_set", "alpha"], columns="split",
                           values=["mae", "regret_kg_mean", "spearman_mean"]).round(4).to_string())
    print(f"chosen alpha: {chosen}")

    model, pred = fitted[(MODEL_SET, chosen[MODEL_SET])]
    preds[MODEL] = pred
    for split in SPLITS:
        summary, _ = evaluate_split(df, pred, split)
        summary.to_csv(OUT_METRICS.format(split=split), index=False)
        print(f"\n== {MODEL} (alpha {chosen[MODEL_SET]}) | {split}")
        print(summary.round(3).to_string(index=False))

    names = model.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": list(names) + ["(intercept)"],
                          "coef": list(model.named_steps["ridge"].coef_) + [model.named_steps["ridge"].intercept_]})
    coefs.to_csv(OUT_COEFS, index=False)
    print("\n== top 10 |coef| (lag features standardised; one-hot on 0/1 scale)")
    top = coefs[coefs["feature"] != "(intercept)"].reindex(
        coefs["coef"].abs().sort_values(ascending=False).index).dropna().head(10)
    print(top.round(4).to_string(index=False))

    val = df[df["split"] == "val"]
    by_hour = pd.DataFrame({m: (p[val.index] - val["truth"]).abs().groupby(val["hour_start_sydney"].dt.hour).mean()
                            for m, p in [(MODEL, preds[MODEL]), ("hourmean", preds["hourmean"]),
                                         ("yesterday", preds["yesterday"])]})
    by_hour.insert(0, "n_hours", val.groupby(val["hour_start_sydney"].dt.hour).size())
    by_hour = by_hour.reindex([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]).rename_axis("hour_sydney")
    by_hour.to_csv(OUT_MAE_HOUR)
    print("\n== val MAE by Sydney hour")
    print(by_hour.round(3).to_string())

    context = regret_context(df)
    context.to_csv(OUT_CONTEXT, index=False)
    print("\n== regret context")
    print(context.round(3).to_string(index=False))
