"""按夜计算的评估指标（交接文档 §4.5）：MAE、regret、夜内 Spearman ρ；按月汇总，分四组

输入每行一个小时，至少有 night_of、hour_start_nem、pred、truth、nonbatt_majority、n_battfill 列；
每次只传一个 split（电池占比的高、低两半按这个 split 内的中位数划分）。
只在 truth 非空的小时上算；pred 在这些小时不能为空。

分组（10-11 定）：
- all：全部小时，MAE、regret、Spearman
- nonbatt：只算 MAE，只用 nonbatt 占多数的小时（regret / Spearman 去掉部分小时就成了另一个调度问题）
- batt_share_high / batt_share_low：按夜的电池定价占比（status ∈ {batt, mixed} 的 5 分钟占比）
  在 split 内按中位数分两半，整夜算 MAE、regret、Spearman；占比等于中位数的夜归 low
"""
import numpy as np
import pandas as pd

K_HOURS = 4            # 每晚充电 4 个小时
KWH_PER_HOUR = 2.3     # 2.3 kW × 1 h；MEF 单位 t/MWh = kg/kWh，乘出来是 kg
INTERVALS_PER_HOUR = 12


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def per_night(df, whole_night=True):
    """每夜一行：n_hours、mae、regret_kg、spearman。

    regret = 按预测挑的 4 个小时与按真实值挑的 4 个小时相比，真实排放多出的 kg。
    whole_night=False（nonbatt 组，只含部分小时）时只算 MAE。
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
            # 同值时取较早的小时
            picked = g.nsmallest(K_HOURS, "pred", keep="first")["truth"].sum()
            best = g.nsmallest(K_HOURS, "truth", keep="first")["truth"].sum()
            regret = KWH_PER_HOUR * (picked - best)
            # 两组挑到同样 4 个小时时，求和顺序不同可能差一个舍入误差
            assert regret >= -1e-9, f"negative regret on night {night:%Y-%m-%d}: {regret}"
            rho = spearman(g["pred"], g["truth"])
        rows.append((night, n, mae, regret, rho))
    return pd.DataFrame(rows, columns=["night_of", "n_hours", "mae", "regret_kg", "spearman"])


def batt_share_half(df):
    """每夜的电池定价占比，在传入的这个 split 内按中位数分成 high / low"""
    g = df.groupby("night_of")
    share = g["n_battfill"].sum() / (INTERVALS_PER_HOUR * g.size())
    return pd.Series(np.where(share > share.median(), "high", "low"), index=share.index), share


def _summary(nightly, whole_night=True):
    """whole_night=False（nonbatt 组）时不汇总 regret 和 Spearman，直接写 NaN"""
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
    """四组 × (每月 + 全部)；返回 (汇总表, 每夜明细)"""
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
