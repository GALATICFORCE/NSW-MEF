"""Step 7：冻结——测试集只跑一次（交接文档 10-20 版 §7）

整个项目里只有这个脚本对 test 夜的真实标签算指标。没有命令行参数，模型、特征、alpha、对照全部写死：
- ridge-v1：直接读交给 Oscar 的 outputs/mef_forecast_C_rules-v1_ridge-v1.csv 的 test 行；
  另按 Step 5（train + val 重拟合、alpha 100）重算一遍，断言逐值相同
- hgb-v1（未被选中，只为报告完整）：同一组参数在 train + val 上重拟合
- hourmean：train + val 夜的同小时均值；fixed：18–21 点；yesterday、weekmean：现成特征列；oracle_C：真实 C 标签

顺序：
1. 保护：日志或任何输出已存在、src/ 和 HEAD 不一致、HEAD 不是 commit A、预测记录文件为空 —— 直接退出，什么都不写
2. 解盲前的回归检查（只用 train + val）：同一套评估函数在 val 上必须逐值复现已提交的 val 输出；
   对不上就退出，此时还没读 test 标签，也还没建日志
3. 独占创建日志：运行时间、commit A、预测文件 SHA-256、Kunyu 的预测原文；然后解盲、算指标、写输出和图
4. 日志末尾写 "single run completed" 和每个输出文件的 SHA-256

用法：python src\\test_eval.py   （只运行一次；先把 Kunyu 的预测原文写进 outputs\\test_run_predictions-v1.txt）
"""
import hashlib
import os
import subprocess
import sys
from datetime import datetime

import error_analysis as ea  # noqa: I001  先导入：带来 Step 6 的画图设置（Agg）、颜色、long_errors()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import HistGradientBoostingRegressor

from baselines import hour_status_counts, hourmean
from features import FEATURES, build
from hgb import FIXED_PARAMS, PAIRS, fit_hgb, paired
from metrics import K_HOURS, KWH_PER_HOUR, evaluate, per_night
from nights import TEST_START, load_hourly, night_hours, split_nights
from ridge import load, make_model

RIDGE_ALPHA = 100
HGB_DEPTH, HGB_ITER = 3, 100
FIXED_HOURS = [18, 19, 20, 21]
TRUTHS = ["B", "C", "Cfwd"]
TOL = 1e-12
COMMIT_A_SUBJECT = "Step 7 script, not yet run"
N_TEST_NIGHTS, N_TEST_HOURS = 53, 689

DEPLOYABLE = ["hourmean", "yesterday", "weekmean", "ridge-v1", "hgb-v1"]          # model_compare 的行
NIGHTLY_STRATS = ["fixed", "hourmean", "yesterday", "weekmean", "ridge-v1", "hgb-v1"]
CROSS_STRATS = NIGHTLY_STRATS + ["oracle_C"]
FIG_STRATS = ["fixed", "hourmean", "yesterday", "weekmean", "ridge-v1", "hgb-v1"]

FORECAST = "outputs/mef_forecast_C_rules-v1_ridge-v1.csv"
PRED_FILE = "outputs/test_run_predictions-v1.txt"
LOG = "outputs/test_run_log-v1.txt"
OUT_METRICS = {"ridge-v1": "outputs/metrics_test_ridge-v1.csv", "hgb-v1": "outputs/metrics_test_hgb-v1.csv",
               "hourmean": "outputs/metrics_test_hourmean-v1.csv", "yesterday": "outputs/metrics_test_yesterday-v1.csv",
               "weekmean": "outputs/metrics_test_weekmean-v1.csv"}
OUT_COMPARE = "outputs/model_compare_test-v1.csv"
OUT_CROSS = "outputs/crosslabel_eval_test-v1.csv"
OUT_NIGHTLY = "outputs/nightly_regret_test-v1.csv"
OUT_PAIRED = "outputs/paired_regret_test-v1.csv"
OUT_FILL = "outputs/error_by_fill_test-v1.csv"
OUT_BIAS = "outputs/bias_by_hour_test-v1.csv"
FIG_COMPARE = "outputs/fig_test_model_compare-v1.png"
FIG_PAIRED = "outputs/fig_test_paired_regret-v1.png"
OUTPUTS = [LOG, *OUT_METRICS.values(), OUT_COMPARE, OUT_CROSS, OUT_NIGHTLY, OUT_PAIRED, OUT_FILL, OUT_BIAS,
           FIG_COMPARE, FIG_PAIRED]

# 已提交的 val 输出：解盲前的回归检查用，也是 _test 表里 val 对照行的来源（复制，不重算）
REF_METRICS_VAL = {"ridge-v1": "outputs/metrics_val_ridge-v1.csv", "hgb-v1": "outputs/metrics_val_hgb-v1.csv",
                   "hourmean": "outputs/metrics_val_hourmean-v2.csv",
                   "yesterday": "outputs/metrics_val_yesterday-v2.csv",
                   "weekmean": "outputs/metrics_val_weekmean-v2.csv"}
REF_COMPARE = "outputs/model_compare-v1.csv"
REF_CROSS = "outputs/crosslabel_eval-v1.csv"
REF_NIGHTLY = "outputs/nightly_regret-v1.csv"
REF_PAIRED = "outputs/paired_regret-v1.csv"
REF_FILL = "outputs/error_by_fill-v1.csv"
REF_HOUR_MONTH = "outputs/error_by_hour_month-v1.csv"

COLORS = {**ea.COLORS, "yesterday": "#e87ba4", "fixed": "#008300"}   # 分类色第 5、6 槽给 yesterday、fixed


class RunLog:
    """打印到屏幕；建日志之前先缓存，建日志时一起写进去，之后边打印边写"""

    def __init__(self):
        self.buffer, self.fh = [], None

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text, flush=True)
        if self.fh is None:
            self.buffer.append(text)
        else:
            self.fh.write(text + "\n")
            self.fh.flush()

    def open(self, header):
        self.fh = open(LOG, "x", encoding="utf-8", newline="\n")       # "x"：文件已存在就报错，绝不覆盖
        self.fh.write(header + "\n")
        self.fh.write("\n".join(self.buffer) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def guard():
    """任何一条不满足就退出，什么都不写"""
    if os.path.exists(LOG):
        sys.exit(f"STOP: {LOG} already exists. Step 7 runs once; nothing was written.")
    existing = [p for p in OUTPUTS if os.path.exists(p)]
    if existing:
        sys.exit(f"STOP: output files already exist: {existing}. Nothing was written.")
    if not os.path.exists(PRED_FILE) or not open(PRED_FILE, encoding="utf-8").read().strip():
        sys.exit(f"STOP: {PRED_FILE} is missing or empty. Write Kunyu's predictions first.")
    if git("ls-files", "src/test_eval.py") != "src/test_eval.py":
        sys.exit("STOP: src/test_eval.py is not committed.")
    if git("status", "--porcelain", "--", "src"):
        sys.exit("STOP: src/ differs from HEAD; the script must be exactly commit A.")
    head, subject = git("rev-parse", "HEAD"), git("log", "-1", "--format=%s")
    if subject != COMMIT_A_SUBJECT:
        sys.exit(f"STOP: HEAD is '{subject}', expected commit A '{COMMIT_A_SUBJECT}'.")
    return head


def with_status(df):
    counts = hour_status_counts()
    df["nonbatt_majority"] = df["hour_start_nem"].map(counts["n_nonbatt"]) >= 7
    df["n_battfill"] = df["hour_start_nem"].map(counts["n_battfill"])
    return df


def label_truths(df, splits):
    """真实值 C 就是 truth 列；B、C-fwd 按 hour_start_nem 取对应的小时表（模型不重训，只换计分的尺子）"""
    truths = {"C": df["truth"]}
    for lab in ["B", "Cfwd"]:
        h = night_hours(lab, splits).set_index("hour_start_nem")["value"]
        truths[lab] = df["hour_start_nem"].map(h)
        assert truths[lab].notna().all(), f"missing {lab} labels"
    return truths


def fixed_pred(df):
    return pd.Series(np.where(df["hour_start_sydney"].dt.hour.isin(FIXED_HOURS), 0.0, 1.0), index=df.index)


# ---------- 评估函数（val 回归检查和 test 用的是同一组）
def compare_rows(sub, preds, truths, split):
    """同 hgb.compare，一次一个 split"""
    rows = []
    pure = sub["n_battfill"] == 0
    for name in DEPLOYABLE:
        pred = preds[name]
        summ = evaluate(sub.assign(pred=pred))[0].set_index(["group", "month"])
        s = sub.assign(pred=pred)
        nights = s.groupby("night_of")
        row = {"split": split, "model": name,
               "mae_all": summ.loc[("all", "all"), "mae"], "n_all": int(summ.loc[("all", "all"), "n_hours"]),
               "mae_nonbatt_majority": summ.loc[("nonbatt", "all"), "mae"],
               "n_nonbatt_majority": int(summ.loc[("nonbatt", "all"), "n_hours"]),
               "mae_pure": per_night(s[pure], whole_night=False)["mae"].mean(), "n_pure": int(pure.sum())}
        for truth in TRUTHS:
            n = per_night(sub.assign(truth=truths[truth], pred=pred))
            row[f"regret_{truth}"] = n["regret_kg"].mean()
            if truth == "C":
                row["spearman_C"] = n["spearman"].mean()
        row["pred_range_mean"] = (nights["pred"].max() - nights["pred"].min()).mean()
        row["truth_range_mean"] = (nights["truth"].max() - nights["truth"].min()).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def nightly_rows(sub, preds, truths, split, strategies):
    """同 hgb.nightly：每夜 regret，split × 真实标签 × 策略"""
    rows = []
    for truth in TRUTHS:
        for name in strategies:
            n = per_night(sub.assign(truth=truths[truth], pred=preds[name]))
            rows.append(n[["night_of", "regret_kg"]].assign(split=split, truth=truth, strategy=name))
    return pd.concat(rows, ignore_index=True)[["split", "night_of", "truth", "strategy", "regret_kg"]]


def crosslabel_rows(sub, preds, truths, split, strategies):
    """同 label_diagnostics.crosslabel：每夜最优排放参照 + 各策略 regret"""
    rows = []
    for truth in TRUTHS:
        optimal = sub.assign(t=truths[truth]).groupby("night_of")["t"].apply(
            lambda s: KWH_PER_HOUR * s.nsmallest(K_HOURS).sum()).mean()
        for name in strategies:
            nightly = per_night(sub.assign(truth=truths[truth], pred=preds[name]))
            rho = nightly["spearman"].mean() if name not in ("fixed", "oracle_C") else np.nan
            rows.append({"split": split, "truth": truth, "strategy": name, "n_nights": len(nightly),
                         "optimal_kg_mean": optimal, "regret_kg_mean": nightly["regret_kg"].mean(),
                         "spearman_mean": rho})
    return pd.DataFrame(rows)


def fill_rows(long, split):
    """同 Step 6 Q2，只取 ridge-v1"""
    r = long[(long["split"] == split) & (long["model"] == "ridge-v1")]
    g = r.groupby("fill_bin").agg(n_hours=("abs_err", "size"), n_nights=("night_of", "nunique"),
                                  mae=("abs_err", "mean"), mae_hour_adj=("abs_err_hour_adj", "mean"))
    g = g.reindex(ea.FILL_BINS)
    g[["n_hours", "n_nights"]] = g[["n_hours", "n_nights"]].fillna(0).astype(int)
    return g.rename_axis("fill_bin").reset_index().assign(split=split, model="ridge-v1")[
        ["split", "fill_bin", "model", "n_hours", "n_nights", "mae", "mae_hour_adj"]]


def bias_rows(long, split):
    r = long[(long["split"] == split) & (long["model"] == "ridge-v1")]
    return r.groupby("hour_sydney").agg(n_hours=("err", "size"), bias=("err", "mean")).reindex(ea.NIGHT_HOURS)


def val_bias_from_step6():
    """Step 6 的 val bias（按月的表合并成按小时）：复制已有结果，不重算"""
    t = pd.read_csv(REF_HOUR_MONTH)
    v = t[t["split"] == "val"].assign(w=lambda x: x["bias"] * x["n_hours"])
    g = v.groupby("hour_sydney").agg(w=("w", "sum"), n_hours=("n_hours", "sum"))
    return pd.DataFrame({"n_hours": g["n_hours"], "bias": g["w"] / g["n_hours"]}).reindex(ea.NIGHT_HOURS)


def same(new, ref, keys, what):
    """逐值比较：键相同、数值列差 < TOL、空值位置相同、文字列相同"""
    new = new.sort_values(keys).reset_index(drop=True)
    ref = ref.sort_values(keys).reset_index(drop=True)
    assert len(new) == len(ref) and new[keys].astype(str).equals(ref[keys].astype(str)), f"{what}: rows differ"
    worst = 0.0
    for col in ref.columns:
        if col in keys or col not in new.columns:
            assert col in new.columns, f"{what}: missing column {col}"
            continue
        if pd.api.types.is_numeric_dtype(ref[col]):
            a, b = new[col].astype(float).to_numpy(), ref[col].astype(float).to_numpy()
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"{what}: NaN pattern differs in {col}"
            d = np.nanmax(np.abs(a - b)) if (~np.isnan(a)).any() else 0.0
            assert d < TOL, f"{what}: {col} differs by {d:.2e}"
            worst = max(worst, d)
        else:
            assert new[col].astype(str).equals(ref[col].astype(str)), f"{what}: {col} differs"
    return worst


def val_checks(tv, log):
    """解盲前：同一组评估函数在 val 上逐值复现已提交的输出"""
    train = tv["split"] == "train"
    ridge = make_model(FEATURES, RIDGE_ALPHA).fit(tv.loc[train, FEATURES], tv.loc[train, "y"])
    preds = {"ridge-v1": pd.Series(ridge.predict(tv[FEATURES]), index=tv.index),
             "hgb-v1": pd.Series(fit_hgb(tv, HGB_DEPTH, HGB_ITER).predict(tv[FEATURES]), index=tv.index),
             "hourmean": hourmean(tv), "yesterday": tv["lag1_same_hour"], "weekmean": tv["lag2to7_same_hour_mean"],
             "fixed": fixed_pred(tv), "oracle_C": tv["truth"]}
    truths = label_truths(tv, ("train", "val"))
    v = tv[tv["split"] == "val"]
    pv = {k: s[v.index] for k, s in preds.items()}
    tvv = {k: s[v.index] for k, s in truths.items()}
    checks = {}
    for m in DEPLOYABLE:
        checks[f"metrics_val {m}"] = same(evaluate(v.assign(pred=pv[m]))[0], pd.read_csv(REF_METRICS_VAL[m]),
                                          ["group", "month"], m)
    ref = pd.read_csv(REF_COMPARE)
    checks["model_compare val"] = same(compare_rows(v, pv, tvv, "val"), ref[ref["split"] == "val"],
                                       ["split", "model"], "model_compare")
    ref = pd.read_csv(REF_CROSS)
    checks["crosslabel val"] = same(crosslabel_rows(v, pv, tvv, "val", ["fixed", "yesterday", "ridge-v1", "weekmean",
                                                                          "oracle_C"]),
                                    ref[ref["split"] == "val"], ["split", "truth", "strategy"], "crosslabel")
    night = nightly_rows(v, pv, tvv, "val", ["fixed", "yesterday", "weekmean", "ridge-v1", "hgb-v1"])
    ref = pd.read_csv(REF_NIGHTLY)
    checks["nightly_regret val"] = same(night.assign(night_of=night["night_of"].dt.strftime("%Y-%m-%d")),
                                        ref[ref["split"] == "val"], ["split", "night_of", "truth", "strategy"],
                                        "nightly_regret")
    ref = pd.read_csv(REF_PAIRED)
    checks["paired_regret val"] = same(paired(night), ref[ref["split"] == "val"],
                                       ["split", "truth", "strategy", "reference"], "paired_regret")
    long = ea.long_errors(tv, {m: preds[m] for m in ea.MODELS})
    ref = pd.read_csv(REF_FILL, dtype={"fill_bin": str})
    checks["error_by_fill val ridge"] = same(fill_rows(long, "val"),
                                             ref[(ref["split"] == "val") & (ref["model"] == "ridge-v1")],
                                             ["split", "fill_bin", "model"], "error_by_fill")
    b_new, b_ref = bias_rows(long, "val"), val_bias_from_step6()
    d = float(np.abs(b_new["bias"] - b_ref["bias"]).max())
    assert (b_new["n_hours"] == b_ref["n_hours"]).all() and d < TOL, "val bias by hour differs from Step 6"
    checks["bias_by_hour val"] = d
    log("== pre-unblinding regression checks on validation nights (same functions as the test evaluation)")
    for k, d in checks.items():
        log(f"  {k:28s} max diff {d:.1e}  PASS")


# ---------- 图
def light(hex_color, t=0.55):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(rgb + (1 - rgb) * t)


def fig_compare(comp_test, comp_val, cross_test, cross_val, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    width = 0.38
    panels = [(axes[0], FIG_STRATS[1:], "mae_all", "MAE (tCO2e/MWh), true label = C", "{:.3f}"),
              (axes[1], FIG_STRATS, "regret_C", "Regret (kg per night), true label = C", "{:.2f}")]
    for ax, strats, col, title, fmt in panels:
        for i, s in enumerate(strats):
            for k, (split, comp, cross) in enumerate([("val", comp_val, cross_val), ("test", comp_test, cross_test)]):
                if s == "fixed":
                    val = cross[(cross["truth"] == "C") & (cross["strategy"] == "fixed")]["regret_kg_mean"].iloc[0]
                else:
                    val = comp[comp["model"] == s][col].iloc[0]
                x = i + (k - 0.5) * width
                ax.bar(x, val, width * 0.92, color=light(COLORS[s]) if split == "val" else COLORS[s],
                       hatch="///" if split == "val" else None, edgecolor="white", linewidth=0)
                ax.text(x, val, fmt.format(val), ha="center", va="bottom", fontsize=6.5, color=ea.INK2)
        ax.set_xticks(range(len(strats)), strats)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=ea.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    handles = [Patch(facecolor=light("#52514e"), hatch="///", edgecolor="white", label="validation, 54 nights"),
               Patch(facecolor="#52514e", label="test, 53 nights")]
    axes[0].legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    fig.suptitle("Validation vs test nights: ridge-v1 and reference strategies (each colour = one strategy)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_paired(pr, n_nights, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, sharex=True)   # 三种标签同一刻度，可直接比
    labels = [f"{s} − {r}" for s, r in PAIRS]
    for ax, truth in zip(axes, TRUTHS):
        sub = pr[pr["truth"] == truth].set_index(["strategy", "reference"])
        for y, (s, r) in enumerate(PAIRS):
            row = sub.loc[(s, r)]
            m, lo, hi = row["diff_mean"], row["ci_low"], row["ci_high"]
            c = COLORS[s]
            ax.errorbar(m, y, xerr=[[max(m - lo, 0.0)], [max(hi - m, 0.0)]], fmt="o", color=c, ecolor=c,
                        elinewidth=2, capsize=3,
                        markersize=7, markerfacecolor=c if r == "fixed" else "white", markeredgewidth=2)
        ax.axvline(0, color=ea.INK, lw=1)
        ax.set_title(f"true label = {truth}", loc="left")
        ax.grid(axis="x", color=ea.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("difference in regret (kg per night)")
    axes[0].set_yticks(range(len(PAIRS)), labels)
    axes[0].set_ylim(len(PAIRS) - 0.5, -0.5)
    fig.suptitle(f"Test nights ({n_nights}): per-night regret difference, mean and 95% per-night bootstrap interval"
                 "\nleft of 0 = the first strategy emits less", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------- 解盲之后
def test_frame(tv, log):
    hourly = load_hourly("C")
    test_hours = night_hours("C", ("test",))
    feats, fallback, doubled = build(test_hours, hourly=hourly)        # 四条防泄漏 assert 照样逐行跑
    expected = split_nights(hourly).query("split == 'test'")
    assert (feats["night_of"] >= TEST_START).all() and set(feats["split"]) == {"test"}
    assert len(feats) == N_TEST_HOURS == expected["n_hours"].sum()
    assert feats["night_of"].nunique() == N_TEST_NIGHTS == len(expected)
    assert feats[FEATURES].notna().all().all() and feats["y"].notna().all()
    te = with_status(feats)
    te["truth"] = te["y"]
    log(f"test nights {te['night_of'].nunique()}, hours {len(te)}, "
        f"{te['night_of'].min():%Y-%m-%d} -> {te['night_of'].max():%Y-%m-%d}; NaN in features 0; "
        f"lag1_same_hour D-2 fallback {int(fallback.sum())}, two-hour mean {int(doubled.sum())}")
    return te


def test_predictions(tv, te, log):
    fc = pd.read_csv(FORECAST, parse_dates=["hour_start_nem"])
    fc = fc[fc["split"] == "test"].set_index("hour_start_nem")
    assert len(fc) == len(te) and set(fc.index) == set(te["hour_start_nem"])
    ridge_file = te["hour_start_nem"].map(fc["mef_pred"])
    ridge = make_model(FEATURES, RIDGE_ALPHA).fit(tv[FEATURES], tv["y"])      # Step 5：train + val 重拟合
    d = float(np.abs(ridge.predict(te[FEATURES]) - ridge_file.to_numpy()).max())
    assert d < TOL, f"ridge-v1 recomputed differs from {FORECAST} by {d:.2e}"
    log(f"ridge-v1: read from {FORECAST} (test rows); Step 5 refit recomputed, max diff {d:.1e}")

    def hgb_fit():
        return HistGradientBoostingRegressor(max_depth=HGB_DEPTH, max_iter=HGB_ITER, **FIXED_PARAMS).fit(
            tv[FEATURES], tv["y"]).predict(te[FEATURES])
    p1, p2 = hgb_fit(), hgb_fit()
    assert np.array_equal(p1, p2)
    lut = tv.groupby(tv["hour_start_sydney"].dt.hour)["y"].mean()
    picks = sorted(lut.nsmallest(K_HOURS).index)
    log(f"hgb-v1: refit on train + val ({len(tv)} hours), two fits identical")
    log(f"hourmean: train + val same-hour means; lowest four hours = {picks} (fixed = {FIXED_HOURS})")
    return {"ridge-v1": ridge_file, "hgb-v1": pd.Series(p1, index=te.index),
            "hourmean": te["hour_start_sydney"].dt.hour.map(lut), "yesterday": te["lag1_same_hour"],
            "weekmean": te["lag2to7_same_hour_mean"], "fixed": fixed_pred(te), "oracle_C": te["truth"]}


def main():
    pd.set_option("display.width", 220)
    sys.stdout.reconfigure(errors="backslashreplace")
    head = guard()
    log = RunLog()
    tv = load()                                   # train + val；load() 断言没有 test 夜
    val_checks(tv, log)

    # ---- 解盲点：从这里起读 test 夜的真实标签
    predictions = open(PRED_FILE, encoding="utf-8").read().strip()
    header = "\n".join([
        "Step 7 test-set evaluation: single run",
        f"run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"commit A (script fixed before the run): {head}  '{COMMIT_A_SUBJECT}'",
        f"src/test_eval.py SHA-256: {sha256('src/test_eval.py')}",
        f"{FORECAST} SHA-256: {sha256(FORECAST)}",
        f"Kunyu's predictions, written before the run ({PRED_FILE}, verbatim):",
        predictions,
        "", "---- script stdout ----"])
    log.open(header)
    log("\n== unblinding: test nights")
    te = test_frame(tv, log)
    truths = label_truths(te, ("test",))
    preds = test_predictions(tv, te, log)

    for m, path in OUT_METRICS.items():
        evaluate(te.assign(pred=preds[m]))[0].to_csv(path, index=False)
    ridge_month = pd.read_csv(OUT_METRICS["ridge-v1"]).query("group == 'all'")

    comp = compare_rows(te, preds, truths, "test")
    ref = pd.read_csv(REF_COMPARE)
    comp_val = ref[ref["split"] == "val"].reset_index(drop=True)
    pd.concat([comp, comp_val], ignore_index=True).to_csv(OUT_COMPARE, index=False)

    cross = crosslabel_rows(te, preds, truths, "test", CROSS_STRATS)
    cross.to_csv(OUT_CROSS, index=False)
    ref = pd.read_csv(REF_CROSS)
    cross_val = ref[ref["split"] == "val"].reset_index(drop=True)

    night = nightly_rows(te, preds, truths, "test", NIGHTLY_STRATS)
    night.assign(night_of=night["night_of"].dt.strftime("%Y-%m-%d")).to_csv(OUT_NIGHTLY, index=False)
    pr = paired(night)
    pr.to_csv(OUT_PAIRED, index=False)

    long = ea.long_errors(te, {m: preds[m] for m in ea.MODELS})
    ref = pd.read_csv(REF_FILL, dtype={"fill_bin": str})
    fill_val = ref[(ref["split"] == "val") & (ref["model"] == "ridge-v1")].reset_index(drop=True)
    fill = pd.concat([fill_rows(long, "test"), fill_val], ignore_index=True)
    fill.to_csv(OUT_FILL, index=False)
    b_test, b_val = bias_rows(long, "test"), val_bias_from_step6()
    bias = pd.DataFrame({"hour_sydney": ea.NIGHT_HOURS, "n_hours_val": b_val["n_hours"].to_numpy(),
                         "bias_val": b_val["bias"].to_numpy(), "n_hours_test": b_test["n_hours"].to_numpy(),
                         "bias_test": b_test["bias"].to_numpy()})
    bias.to_csv(OUT_BIAS, index=False)

    fig_compare(comp, comp_val, cross, cross_val, FIG_COMPARE)
    fig_paired(pr, te["night_of"].nunique(), FIG_PAIRED)

    log("\n== main table: test rows (val rows in the same file are copied from model_compare-v1.csv)")
    log(pd.concat([comp, comp_val]).round(4).to_string(index=False))
    log("\n== regret by true label, incl. fixed and oracle_C (test; val copied from crosslabel_eval-v1.csv)")
    log(pd.concat([cross, cross_val]).round(4).to_string(index=False))
    log("\n== paired per-night regret difference (test)")
    log(pr.round(4).to_string(index=False))
    log("\n== ridge-v1 by month (test, all group)")
    log(ridge_month[["month", "n_nights", "n_hours", "mae", "regret_kg_mean", "spearman_mean"]].round(4)
        .to_string(index=False))
    log("\n== ridge-v1 by battery-fill bin (test computed; val copied from error_by_fill-v1.csv)")
    log(fill.round(4).to_string(index=False))
    log("\n== ridge-v1 bias by Sydney hour (forecast - actual)")
    log(bias.round(4).to_string(index=False))

    log("\n== outputs (SHA-256)")
    for path in OUTPUTS[1:]:
        log(f"  {sha256(path)}  {path}")
    log("\nsingle run completed")
    log.close()


if __name__ == "__main__":
    main()
