"""按夜计算的评估指标（交接文档 §4.5）：MAE、regret、夜内 Spearman ρ；按月汇总，分两组

输入每行一个小时，至少有 night_of、hour_start_nem、pred、truth、nonbatt_majority 列。
只在 truth 非空的小时上算；pred 在这些小时不能为空。
"""
import numpy as np
import pandas as pd

K_HOURS = 4            # 每晚充电 4 个小时
KWH_PER_HOUR = 2.3     # 2.3 kW × 1 h；MEF 单位 t/MWh = kg/kWh，乘出来是 kg
GROUPS = {"all": None, "nonbatt": "nonbatt_majority"}


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def per_night(df):
    """每夜一行：n_hours、mae、regret_kg、spearman。

    regret = 按预测挑的 4 个小时与按真实值挑的 4 个小时相比，真实排放多出的 kg；
    小时数不超过 4 的夜（只会出现在 nonbatt 组）不算 regret，少于 3 的不算 Spearman。
    """
    rows = []
    for night, g in df.sort_values("hour_start_nem").groupby("night_of"):
        g = g[g["truth"].notna()]
        if g["pred"].isna().any():
            raise ValueError(f"NaN prediction on night {night:%Y-%m-%d}")
        n = len(g)
        mae = (g["pred"] - g["truth"]).abs().mean() if n else np.nan
        regret = np.nan
        if n > K_HOURS:
            # 同值时取较早的小时
            picked = g.nsmallest(K_HOURS, "pred", keep="first")["truth"].sum()
            best = g.nsmallest(K_HOURS, "truth", keep="first")["truth"].sum()
            regret = KWH_PER_HOUR * (picked - best)
        rho = spearman(g["pred"], g["truth"]) if n >= 3 else np.nan
        rows.append((night, n, mae, regret, rho))
    return pd.DataFrame(rows, columns=["night_of", "n_hours", "mae", "regret_kg", "spearman"])


def _summary(nightly):
    return pd.Series({
        "n_nights": int(nightly["mae"].notna().sum()),
        "n_hours": int(nightly["n_hours"].sum()),
        "mae": nightly["mae"].mean(),
        "regret_kg_mean": nightly["regret_kg"].mean(),
        "regret_kg_median": nightly["regret_kg"].median(),
        "regret_kg_p90": nightly["regret_kg"].quantile(0.9),
        "n_nights_regret": int(nightly["regret_kg"].notna().sum()),
        "spearman_mean": nightly["spearman"].mean(),
        "spearman_median": nightly["spearman"].median(),
        "n_nights_spearman": int(nightly["spearman"].notna().sum()),
    })


def evaluate(df):
    """两组 × (每月 + 全部)；返回 (汇总表, 每夜明细)"""
    summaries, nightly_all = [], []
    for group, col in GROUPS.items():
        sub = df if col is None else df[df[col]]
        nightly = per_night(sub).assign(group=group)
        month = nightly["night_of"].dt.to_period("M").astype(str)
        s = nightly.groupby(month).apply(_summary)
        s.loc["all"] = _summary(nightly)
        summaries.append(s.rename_axis("month").reset_index().assign(group=group))
        nightly_all.append(nightly)
    summary = pd.concat(summaries)[["group", "month"] + list(_summary(nightly_all[0]).index)]
    return summary.reset_index(drop=True), pd.concat(nightly_all, ignore_index=True)
