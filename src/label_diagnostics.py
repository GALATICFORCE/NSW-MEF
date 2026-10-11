"""Step 4b 标签诊断（交接文档 §7）：只做评估，不训练新模型，只用 train 和 val 的夜

(a) 夜里各小时的平均 MEF 高低在 A / B / C / C-fwd 下是否一致，在不含电池填补的小时里是否成立
(b) 这个高低关系随时间怎么变（P1 / P2 / val 三段）
(c) 换一种标签当真实值，现有预测和策略表现如何

ridge-v1 的预测用 Step 4 同样的数据和 alpha = 100 重新得到（Ridge 是确定性的，结果和 Step 4 相同）。
weekmean = 现成特征 lag2to7_same_hour_mean，不拟合参数；它是看了 Step 4 结果后加的参照基线。

用法：python src\\label_diagnostics.py   （先运行 features.py）
"""
import numpy as np
import pandas as pd

from baselines import hour_status_counts
from features import FEATURES
from metrics import K_HOURS, KWH_PER_HOUR, evaluate, per_night, spearman
from nights import night_hours
from ridge import MODEL_SET, load, make_model, pick_alpha

FULL = "data/processed/mef_nsw1_5min_full.csv"
LABELS = ["A", "B", "C", "Cfwd"]
PAIRS = [("C", "B"), ("C", "Cfwd"), ("B", "Cfwd")]
NIGHT_HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
FIXED_HOURS = [18, 19, 20, 21]
RIDGE_ALPHA = 100
PERIODS = [("P1", "2025-07-08", "2025-11-30"), ("P2", "2025-12-01", "2026-03-14"), ("val", "2026-03-15", "2026-05-07")]
OUT_PROFILE = "outputs/label_hour_profile-v1.csv"
OUT_AGREE = "outputs/label_agreement-v1.csv"
OUT_CROSS = "outputs/crosslabel_eval-v1.csv"
OUT_WEEKMEAN = "outputs/metrics_{split}_weekmean-v2.csv"


def period_of(night):
    out = pd.Series(pd.NA, index=night.index, dtype="object")
    for name, lo, hi in PERIODS:
        out[(night >= lo) & (night <= hi)] = name
    assert out.notna().all()
    return out


def hours_table():
    """train + val 夜的每个小时：四种标签的值、电池计数、时期"""
    base = night_hours("C", ("train", "val")).rename(columns={"value": "C"})
    for lab in ["A", "B", "Cfwd"]:
        base[lab] = base["hour_start_nem"].map(night_hours(lab, ("train", "val")).set_index("hour_start_nem")["value"])
    counts = hour_status_counts()
    base["nonbatt_majority"] = base["hour_start_nem"].map(counts["n_nonbatt"]) >= 7
    base["n_battfill"] = base["hour_start_nem"].map(counts["n_battfill"])
    base["hour_syd"] = base["hour_start_sydney"].dt.hour
    base["period"] = period_of(base["night_of"])
    assert base[LABELS].notna().all().all()
    return base


def profile(h):
    """表 1：时期 × 标签 × 悉尼小时的均值和小时数，外加 5 分钟电池定价构成"""
    rows = []
    variants = {lab: (h, lab) for lab in LABELS}
    variants["nonbatt_only"] = (h[h["nonbatt_majority"]], "C")      # 交接文档的定义；这些小时里标签并不都相同
    variants["nonbatt_pure"] = (h[h["n_battfill"] == 0], "C")       # 没有任何电池定价区间，四种标签完全相同
    pure = h[h["n_battfill"] == 0]
    assert (pure[LABELS].max(axis=1) - pure[LABELS].min(axis=1)).max() == 0
    for name, (sub, col) in variants.items():
        g = sub.groupby(["period", "hour_syd"])[col]
        rows.append(pd.DataFrame({"label": name, "mean": g.mean(), "n_hours": g.size()}))
    prof = pd.concat(rows).reset_index()

    # 5 分钟构成：这些夜的小时里，电池放电定价 / 电池充电定价 / 非电池（status 不是 batt 或 mixed）各占多少
    full = pd.read_csv(FULL, usecols=["Time", "status", "batt_dir"], parse_dates=["Time"])
    full["hour_start_nem"] = (full["Time"] - pd.Timedelta("5min")).dt.floor("H")
    full = full.merge(h[["hour_start_nem", "period", "hour_syd"]], on="hour_start_nem")
    batt = full["status"].isin(["batt", "mixed"])
    full["discharge"] = batt & (full["batt_dir"] == "discharge")
    full["charge"] = batt & (full["batt_dir"] == "charge")
    full["nonbatt"] = ~batt
    share = full.groupby(["period", "hour_syd"])[["discharge", "charge", "nonbatt"]].mean().add_prefix("share_")
    prof = prof.merge(share.reset_index(), on=["period", "hour_syd"])
    order = {p: i for i, (p, _, _) in enumerate(PERIODS)}
    hour_order = {hh: i for i, hh in enumerate(NIGHT_HOURS)}
    prof = prof.sort_values(["period", "label", "hour_syd"],
                            key=lambda s: s.map(order) if s.name == "period" else (
                                s.map(hour_order) if s.name == "hour_syd" else s))
    return prof[["period", "label", "hour_syd", "mean", "n_hours", "share_discharge", "share_charge", "share_nonbatt"]]


def best4(g, col):
    return set(g.sort_values("hour_start_nem").nsmallest(K_HOURS, col, keep="first")["hour_start_nem"])


def agreement(h):
    """表 2：时期 × 标签对的夜内 Spearman 和最优 4 小时的重合"""
    rows = []
    for (period, night), g in h.groupby(["period", "night_of"]):
        for a, b in PAIRS:
            overlap = len(best4(g, a) & best4(g, b))
            rows.append((period, f"{a}-{b}", spearman(g[a], g[b]), overlap))
    df = pd.DataFrame(rows, columns=["period", "pair", "spearman", "overlap"])
    out = df.groupby(["period", "pair"]).agg(
        n_nights=("overlap", "size"), spearman_mean=("spearman", "mean"),
        best4_overlap_mean=("overlap", "mean"), share_overlap_4=("overlap", lambda s: (s == 4).mean()))
    order = {p: i for i, (p, _, _) in enumerate(PERIODS)}
    return out.reset_index().sort_values(["period", "pair"], key=lambda s: s.map(order) if s.name == "period" else s)


def strategies(h, feats):
    """现成的、按 C 得到的预测（不重训）；fixed 和 oracle_C 是挑时段的规则"""
    model = make_model(FEATURES, RIDGE_ALPHA).fit(feats.loc[feats["split"] == "train", FEATURES],
                                                  feats.loc[feats["split"] == "train", "y"])
    by_nem = feats.set_index("hour_start_nem")
    return {
        "fixed": np.where(h["hour_syd"].isin(FIXED_HOURS), 0.0, 1.0),
        "yesterday": h["hour_start_nem"].map(by_nem["lag1_same_hour"]).to_numpy(),
        "ridge-v1": h["hour_start_nem"].map(pd.Series(model.predict(by_nem[FEATURES]), index=by_nem.index)).to_numpy(),
        "weekmean": h["hour_start_nem"].map(by_nem["lag2to7_same_hour_mean"]).to_numpy(),
        "oracle_C": h["C"].to_numpy(),
    }


def crosslabel(h, preds):
    """表 3：split × 真实标签 × 策略"""
    rows = []
    for split in ["train", "val"]:
        sub = h[h["split"] == split]
        for truth in ["B", "C", "Cfwd"]:
            optimal = sub.groupby("night_of")[truth].apply(
                lambda s: KWH_PER_HOUR * s.nsmallest(K_HOURS).sum()).mean()
            for name, pred in preds.items():
                nightly = per_night(sub.assign(truth=sub[truth], pred=pred[sub.index]))
                rho = nightly["spearman"].mean() if name not in ("fixed", "oracle_C") else np.nan
                rows.append({"split": split, "truth": truth, "strategy": name, "n_nights": len(nightly),
                             "optimal_kg_mean": optimal, "regret_kg_mean": nightly["regret_kg"].mean(),
                             "spearman_mean": rho})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    h = hours_table()
    feats = load()
    assert set(feats["split"]) == {"train", "val"}
    assert pick_alpha(pd.read_csv("outputs/ridge_alpha_scan-v1.csv"), MODEL_SET) == RIDGE_ALPHA
    print(f"hours: {len(h)} | nonbatt-majority {int(h['nonbatt_majority'].sum())} | "
          f"no battery-set interval {int((h['n_battfill'] == 0).sum())}")

    prof = profile(h)
    prof.to_csv(OUT_PROFILE, index=False)
    wide = prof.pivot_table(index=["period", "label"], columns="hour_syd", values="mean")[NIGHT_HOURS]
    print("\n== (a)(b) mean label by Sydney hour")
    print(wide.round(3).to_string())
    comp = prof[prof["label"] == "C"].pivot_table(index="period", columns="hour_syd",
                                                  values=["share_discharge", "share_charge"])
    print("\n== 5-min composition (share of battery discharge-set / charge-set intervals)")
    print(comp.round(2).to_string())
    ranks = prof.assign(rank=prof.groupby(["period", "label"])["mean"].rank(method="first"))
    low4 = ranks[ranks["rank"] <= 4].groupby(["period", "label"])["hour_syd"].apply(lambda s: sorted(s.tolist()))
    print("\n== four lowest-mean hours per period and label")
    print(low4.to_string())

    agree = agreement(h)
    agree.to_csv(OUT_AGREE, index=False)
    print("\n== label agreement")
    print(agree.round(3).to_string(index=False))

    preds = strategies(h, feats)
    cross = crosslabel(h, preds)
    cross.to_csv(OUT_CROSS, index=False)
    print("\n== cross-label evaluation")
    print(cross.round(3).to_string(index=False))

    for split in ["train", "val"]:
        sub = feats[feats["split"] == split].assign(pred=feats["lag2to7_same_hour_mean"])
        summary, _ = evaluate(sub)
        summary.to_csv(OUT_WEEKMEAN.format(split=split), index=False)
        print(f"\n== weekmean | {split}")
        print(summary.query("month == 'all'").round(3).to_string(index=False))
