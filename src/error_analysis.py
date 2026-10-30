"""Step 6: error analysis (hand-off doc 10-19 version §7), diagnosis only, no fixes; train and val nights only

The models are fitted on train only and are the ones evaluated in Step 4 / 4c: ridge-v1 (alpha 100), hgb-v1 (depth 3,
100 iterations), hourmean, weekmean (feature column lag2to7_same_hour_mean, as in hgb.py). First asserts that mae_all and
mae_pure equal outputs/model_compare-v1.csv, then does three things:
- Q1 ridge-v1 errors by Sydney hour × month (month taken from night_of; 2026-03 is split into two columns by split)
- Q2 four bins by the number of battery-filled intervals in the hour; mae_hour_adj = |error| − that model's mean |error|
  in the same split and Sydney hour, averaged per bin
- Q3 train / val gap; on val, per-hour d = |err_hgb| − |err_ridge|, 95% interval by resampling whole nights

The mae of heatmap cells and fill bins, and mean_d in pure_hours_paired, are plain averages over hours (a cell / a bin is
not a whole night); train_val_gap and pure_hours_paired_nightavg average per night first and then over nights, as in
model_compare (nightavg was added on 10-20 after seeing the results; both versions are reported).

Usage: python src\\error_analysis.py   (run features.py first)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from baselines import hourmean  # noqa: E402
from features import FEATURES  # noqa: E402
from hgb import fit_hgb  # noqa: E402
from metrics import per_night  # noqa: E402
from ridge import load, make_model  # noqa: E402

RIDGE_ALPHA = 100
HGB_DEPTH, HGB_ITER = 3, 100
MODELS = ["ridge-v1", "hgb-v1", "hourmean", "weekmean"]
SPLITS = ("train", "val")
NIGHT_HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
FILL_BINS = ["0", "1-4", "5-8", "9-12"]
N_BOOT = 10_000
COMPARE = "outputs/model_compare-v1.csv"
OUT_HOUR_MONTH = "outputs/error_by_hour_month-v1.csv"
OUT_FILL = "outputs/error_by_fill-v1.csv"
OUT_FILL_COUNTS = "outputs/fill_by_hour_counts-v1.csv"
OUT_GAP = "outputs/train_val_gap-v1.csv"
OUT_PAIRED = "outputs/pure_hours_paired-v1.csv"
OUT_PAIRED_NIGHT = "outputs/pure_hours_paired_nightavg-v1.csv"
FIG_HEATMAP = "outputs/fig_error_heatmap-v1.png"
FIG_FILL = "outputs/fig_error_by_fill-v1.png"
FIG_GAP = "outputs/fig_train_val_gap-v1.png"

# Colours: categorical colours go to the four models in a fixed order, the same in all three figures;
# the heatmap uses a single blue, light to dark
COLORS = {"ridge-v1": "#2a78d6", "hgb-v1": "#eb6834", "hourmean": "#1baf7a", "weekmean": "#eda100"}
LIGHT = {"ridge-v1": "#86b6ef", "hgb-v1": "#f5b498"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 150, "savefig.facecolor": "white"})


def fit_predictions(df):
    train = df["split"] == "train"
    ridge = make_model(FEATURES, RIDGE_ALPHA).fit(df.loc[train, FEATURES], df.loc[train, "y"])
    return {"ridge-v1": pd.Series(ridge.predict(df[FEATURES]), index=df.index),
            "hgb-v1": pd.Series(fit_hgb(df, HGB_DEPTH, HGB_ITER).predict(df[FEATURES]), index=df.index),
            "hourmean": hourmean(df),
            "weekmean": df["lag2to7_same_hour_mean"]}


def night_mae(sub):
    """MAE per night first, then averaged over nights (same as mae_all / mae_pure in model_compare)"""
    return per_night(sub, whole_night=False)["mae"].mean()


def regression_check(df, preds):
    ref = pd.read_csv(COMPARE).set_index(["split", "model"])
    rows = []
    for split in SPLITS:
        for model in MODELS:
            sub = df[df["split"] == split].assign(pred=preds[model])
            pure = sub[sub["n_battfill"] == 0]
            r = ref.loc[(split, model)]
            row = {"split": split, "model": model, "mae_all": night_mae(sub), "mae_pure": night_mae(pure),
                   "n_all": len(sub), "n_pure": len(pure)}
            diff = max(abs(row["mae_all"] - r["mae_all"]), abs(row["mae_pure"] - r["mae_pure"]))
            assert diff < 1e-12 and row["n_all"] == r["n_all"] and row["n_pure"] == r["n_pure"], \
                f"{model} {split} differs from {COMPARE} (max diff {diff:.2e})"
            rows.append({**row, "max_diff": diff})
    return pd.DataFrame(rows)


def long_errors(df, preds):
    """One row = one hour × one model"""
    base = df[["split", "night_of", "hour_start_nem", "truth", "n_battfill"]].assign(
        hour_sydney=df["hour_start_sydney"].dt.hour, month=df["night_of"].dt.strftime("%Y-%m"),
        fill_bin=pd.cut(df["n_battfill"], [-0.5, 0.5, 4.5, 8.5, 12.5], labels=FILL_BINS).astype(str))
    out = pd.concat([base.assign(model=m, pred=preds[m]) for m in MODELS], ignore_index=True)
    out["err"] = out["pred"] - out["truth"]
    out["abs_err"] = out["err"].abs()
    # Remove the hour-of-day confounder: subtract the mean |error| of the same model, split and Sydney hour
    hour_mean = out.groupby(["model", "split", "hour_sydney"])["abs_err"].transform("mean")
    out["abs_err_hour_adj"] = out["abs_err"] - hour_mean
    return out


def hour_order(s):
    return s.map({h: i for i, h in enumerate(NIGHT_HOURS)})


# ---------- Q1
def q1_table(long):
    r = long[long["model"] == "ridge-v1"]
    t = r.groupby(["split", "month", "hour_sydney"]).agg(
        n_hours=("err", "size"), mae=("abs_err", "mean"), bias=("err", "mean")).reset_index()
    t["split"] = pd.Categorical(t["split"], list(SPLITS))
    return t.sort_values(["split", "month", "hour_sydney"], key=lambda s: hour_order(s) if s.name == "hour_sydney" else s) \
        .reset_index(drop=True).assign(split=lambda x: x["split"].astype(str))


def q1_figure(t, long):
    cols = t[["split", "month"]].drop_duplicates().reset_index(drop=True)
    grid = t.pivot_table(index="hour_sydney", columns=["split", "month"], values="mae").reindex(NIGHT_HOURS)
    grid = grid[[(s, m) for s, m in zip(cols["split"], cols["month"])]]
    r = long[long["model"] == "ridge-v1"]
    nights = r.groupby(["split", "month"])["night_of"].agg(["nunique", "min", "max"])

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[4.2, 0.09, 0.5, 1.25], wspace=0.05)
    ax, cax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    # The colour scale starts at the data minimum rounded down to a multiple of 0.05, so differences between cells
    # are visible; the colour bar shows the scale
    vmin = float(np.floor(grid.min().min() * 20) / 20)
    vmax = float(np.ceil(grid.max().max() * 20) / 20)
    im = ax.imshow(grid.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.55 else INK)
    labels = []
    for s, m in grid.columns:
        n = nights.loc[(s, m)]
        labels.append(f"{m}\n{n['min']:%d}-{n['max']:%d}\n{n['nunique']} nights")
    ax.set_xticks(range(grid.shape[1]), labels, fontsize=7)
    ax.set_yticks(range(len(NIGHT_HOURS)), [f"{h:02d}:00" for h in NIGHT_HOURS])
    ax.set_ylabel("Sydney hour (start)")
    split_at = int((cols["split"] == "train").sum()) - 0.5
    ax.axvline(split_at, color=INK, lw=2)
    ax.text(split_at - 0.3, -0.9, "fitted period (in-sample)  ", ha="right", va="bottom", color=INK2)
    ax.text(split_at + 0.3, -0.9, "  validation (held-out)", ha="left", va="bottom", color=INK2)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_title("Ridge-v1 forecast error by Sydney hour and month of night\n"
                 "MAE, tCO2e/MWh (training nights in-sample, validation nights held out)", loc="left", pad=22)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("MAE (tCO2e/MWh)")
    cb.outline.set_visible(False)

    vb = t[t["split"] == "val"].assign(w=lambda x: x["bias"] * x["n_hours"]).groupby("hour_sydney") \
        .agg(w=("w", "sum"), n=("n_hours", "sum"))
    bias = (vb["w"] / vb["n"]).reindex(NIGHT_HOURS)
    axb.barh(range(len(NIGHT_HOURS)), bias.to_numpy(), color=COLORS["ridge-v1"], height=0.6)
    axb.axvline(0, color=INK, lw=1)
    axb.set_ylim(len(NIGHT_HOURS) - 0.5, -0.5)
    axb.set_yticks(range(len(NIGHT_HOURS)), [])
    axb.grid(axis="x", color=GRID, lw=0.8)
    axb.set_axisbelow(True)
    axb.set_xlabel("forecast − actual (tCO2e/MWh)")
    axb.set_title("Validation bias by hour\n(negative = forecast too low)", loc="left", pad=22)
    fig.savefig(FIG_HEATMAP, bbox_inches="tight")
    plt.close(fig)
    return grid, bias


# ---------- Q2
def q2_tables(long):
    t = long.groupby(["split", "fill_bin", "model"]).agg(
        n_hours=("abs_err", "size"), n_nights=("night_of", "nunique"),
        mae=("abs_err", "mean"), mae_hour_adj=("abs_err_hour_adj", "mean")).reset_index()
    t["split"] = pd.Categorical(t["split"], list(SPLITS))
    t["fill_bin"] = pd.Categorical(t["fill_bin"], FILL_BINS)
    t["model"] = pd.Categorical(t["model"], MODELS)
    t = t.sort_values(["split", "fill_bin", "model"]).reset_index(drop=True)
    for c in ["split", "fill_bin", "model"]:
        t[c] = t[c].astype(str)
    v = long[(long["split"] == "val") & (long["model"] == "ridge-v1")]
    counts = pd.crosstab(v["hour_sydney"], v["fill_bin"]).reindex(index=NIGHT_HOURS, columns=FILL_BINS, fill_value=0)
    counts["n_hours"] = counts.sum(axis=1)
    return t, counts.rename_axis("hour_sydney").reset_index()


def q2_figure(t):
    v = t[t["split"] == "val"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    width, x = 0.19, np.arange(len(FILL_BINS))
    for ax, col, title in [(axes[0], "mae", "MAE (tCO2e/MWh)"),
                           (axes[1], "mae_hour_adj", "MAE relative to the same Sydney hour (tCO2e/MWh)")]:
        for k, m in enumerate(MODELS):
            vals = v[v["model"] == m].set_index("fill_bin").reindex(FILL_BINS)[col]
            ax.bar(x + (k - 1.5) * width, vals.to_numpy(), width * 0.9, color=COLORS[m], label=m)
        ax.set_xticks(x, [f"{b}" for b in FILL_BINS])
        ax.set_xlabel("battery-filled 5-min intervals in the hour (of 12)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    n = v[v["model"] == "ridge-v1"].set_index("fill_bin").reindex(FILL_BINS)["n_hours"]
    top = axes[0].get_ylim()[1]
    for i, b in enumerate(FILL_BINS):
        grp = v[v["fill_bin"] == b]["mae"].max()
        axes[0].text(i, grp + top * 0.015, f"n = {n[b]} h", ha="center", va="bottom", color=INK2, fontsize=8)
    axes[0].set_ylim(0, top * 1.08)
    axes[1].axhline(0, color=INK, lw=1)
    axes[0].legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.suptitle("Validation nights: forecast error by amount of battery filling in the hour", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_FILL, bbox_inches="tight")
    plt.close(fig)


# ---------- Q3
def q3_gap(check):
    rows = []
    for model in ["ridge-v1", "hgb-v1"]:
        c = check[check["model"] == model].set_index("split")
        for subset, col in [("all", "mae_all"), ("pure", "mae_pure")]:
            tr, va = c.loc["train", col], c.loc["val", col]
            rows.append({"model": model, "subset": subset, "mae_train": tr, "mae_val": va, "gap": va - tr})
    return pd.DataFrame(rows)


def q3_paired(long):
    """Returns two tables with the same columns, from the same set of whole-night resamples:
    pooled = hours of the drawn nights averaged together; nightavg = mean per night, then over nights (same basis as
    mae_pure in model_compare; added on 10-20 by Kunyu's decision after seeing the Step 6 results)"""
    v = long[long["split"] == "val"]
    wide = v.pivot_table(index=["night_of", "hour_start_nem", "n_battfill"], columns="model",
                         values="abs_err").reset_index()
    wide["d"] = wide["hgb-v1"] - wide["ridge-v1"]
    rows = {"pooled": [], "nightavg": []}
    for subset, sub in [("pure", wide[wide["n_battfill"] == 0]), ("all", wide)]:
        per = sub.groupby("night_of")["d"].agg(["sum", "size"])
        s, n = per["sum"].to_numpy(), per["size"].to_numpy()
        night_mean = s / n
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
        boots = {"pooled": s[idx].sum(axis=1) / n[idx].sum(axis=1),      # after resampling whole nights, average over hours
                 "nightavg": night_mean[idx].mean(axis=1)}               # after resampling whole nights, average over nights
        d = sub["d"].to_numpy()
        centre = {"pooled": d.mean(), "nightavg": night_mean.mean()}
        for kind, boot in boots.items():
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows[kind].append({"subset": subset, "n_hours": len(sub), "n_nights": len(per), "mean_d": centre[kind],
                               "ci_low": lo, "ci_high": hi, "n_hgb_worse": int((d > 1e-9).sum()),
                               "n_equal": int((np.abs(d) < 1e-9).sum()), "n_hgb_better": int((d < -1e-9).sum())})
    return pd.DataFrame(rows["pooled"]), pd.DataFrame(rows["nightavg"])


def q3_figure(gap):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = [("ridge-v1", "train"), ("ridge-v1", "val"), ("hgb-v1", "train"), ("hgb-v1", "val")]
    width = 0.2
    for g, subset in enumerate(["all", "pure"]):
        row = gap[gap["subset"] == subset].set_index("model")
        for k, (m, split) in enumerate(bars):
            val = row.loc[m, f"mae_{split}"]
            xpos = g + (k - 1.5) * width
            train = split == "train"
            ax.bar(xpos, val, width * 0.9, color=LIGHT[m] if train else COLORS[m],
                   hatch="///" if train else None, edgecolor="white", linewidth=0,
                   label=f"{m} {'training (in-sample)' if train else 'validation'}" if g == 0 else None)
            ax.text(xpos, val + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks([0, 1], ["all hours", "battery-free hours (pure)"])
    ax.set_ylabel("MAE (tCO2e/MWh)")
    ax.set_ylim(0, gap[["mae_train", "mae_val"]].max().max() * 1.18)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.set_title("Training vs validation MAE, Ridge-v1 and HGB-v1\n"
                 "(per-night MAE averaged over nights, tCO2e/MWh)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_GAP, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load()                                  # load() already asserts there are no test nights
    assert set(df["split"]) == set(SPLITS)
    assert df["n_battfill"].between(0, 12).all()
    preds = fit_predictions(df)

    check = regression_check(df, preds)
    print(f"== regression check vs {COMPARE}: max diff {check['max_diff'].max():.1e} (all < 1e-12)")
    print(check.drop(columns="max_diff").round(4).to_string(index=False))

    long = long_errors(df, preds)

    t1 = q1_table(long)
    t1.to_csv(OUT_HOUR_MONTH, index=False)
    grid, val_bias = q1_figure(t1, long)
    print(f"\n== Q1 ({OUT_HOUR_MONTH}: {len(t1)} rows; cells with n_hours < 10: {int((t1['n_hours'] < 10).sum())})")
    print("worst 3 cells overall:")
    print(t1.nlargest(3, "mae").round(3).to_string(index=False))
    print("worst 3 validation cells:")
    print(t1[t1["split"] == "val"].nlargest(3, "mae").round(3).to_string(index=False))
    print("validation MAE, hours 18-21, by month:")
    print(t1[(t1["split"] == "val") & t1["hour_sydney"].isin([18, 19, 20, 21])]
          .pivot_table(index="hour_sydney", columns="month", values="mae").reindex([18, 19, 20, 21]).round(3).to_string())
    print("validation bias by hour (forecast - actual):")
    print(val_bias.round(3).to_string())

    t2, counts = q2_tables(long)
    t2.to_csv(OUT_FILL, index=False)
    counts.to_csv(OUT_FILL_COUNTS, index=False)
    q2_figure(t2)
    print(f"\n== Q2 ({OUT_FILL})")
    print(t2.round(4).to_string(index=False))
    print(f"\nvalidation hour x fill-bin counts ({OUT_FILL_COUNTS}):")
    print(counts.to_string(index=False))
    for split in SPLITS:
        r = t2[(t2["split"] == split) & (t2["model"] == "ridge-v1")]
        print(f"ridge-v1 {split}: MAE range across bins {r['mae'].max() - r['mae'].min():.4f}, "
              f"mae_hour_adj range {r['mae_hour_adj'].max() - r['mae_hour_adj'].min():.4f}")

    gap = q3_gap(check)
    gap.to_csv(OUT_GAP, index=False)
    paired, paired_night = q3_paired(long)
    paired.to_csv(OUT_PAIRED, index=False)
    paired_night.to_csv(OUT_PAIRED_NIGHT, index=False)
    q3_figure(gap)
    print(f"\n== Q3 ({OUT_GAP}, {OUT_PAIRED})")
    print(gap.round(4).to_string(index=False))
    print(paired.round(4).to_string(index=False))
    print(f"same d averaged per night first, then over nights ({OUT_PAIRED_NIGHT}; "
          f"same definition as the mae_pure difference in {COMPARE}):")
    print(paired_night.round(4).to_string(index=False))
