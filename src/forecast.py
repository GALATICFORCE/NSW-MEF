"""Step 5：给 Oscar 的预测文件（交接文档 §5.2），模型 ridge-v1（Step 4c 按事先定的规则选定）

- val 夜：只在 train 上拟合的 ridge-v1，就是 Step 4 评估过的那个；指标和系数逐值复现 Step 4 的输出文件。
- test 夜：同一组超参数（cal+lag，alpha 100）在 train + val 上重新拟合一次（10-17 Kunyu 定，选项 b）。
  滞后特征用到测试期内悉尼日期 ≤ D-1 的 C 标签（预测发出时已可得）；test 夜本身的标签在建特征前清空。
  对测试期的标签不算任何指标、不报任何统计量。

用法：python src\\forecast.py   （先运行 features.py 和 ridge.py）
"""
import numpy as np
import pandas as pd

from features import FEATURES, build
from nights import TEST_START, load_hourly, night_hours
from ridge import MODEL, MODEL_SET, OUT_COEFS, OUT_METRICS, OUT_SCAN, evaluate_split, load, make_model, pick_alpha

LABEL = "C"
ALPHA = 100        # ridge-v1 的 alpha，Step 4 在验证集上选定
OUT = "data/processed/mef_forecast_C_rules-v1_ridge-v1.csv"
OUT_COEFS_TRAINVAL = "outputs/ridge_coefs_trainval-v1.csv"
COLUMNS = ["night_of", "hour_start_sydney", "hour_start_nem", "mef_pred", "label", "rules", "model", "split"]
KEYS = ["night_of", "hour_start_nem", "hour_start_sydney", "split"]


def max_diff(a, b):
    return float(np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float)).max())


def same_as_file(summary, path):
    ref = pd.read_csv(path)
    num = summary.select_dtypes("number").columns
    new = summary[num].reset_index(drop=True)
    return summary[["group", "month"]].equals(ref[["group", "month"]]) and new.isna().equals(ref[num].isna()) \
        and (new - ref[num]).abs().max().max() < 1e-12


def coefs(model):
    names = list(model.named_steps["pre"].get_feature_names_out()) + ["(intercept)"]
    return pd.Series(list(model.named_steps["ridge"].coef_) + [model.named_steps["ridge"].intercept_], index=names)


def fmt(df):
    return df.assign(night_of=df["night_of"].dt.strftime("%Y-%m-%d"),
                     hour_start_sydney=df["hour_start_sydney"].dt.strftime("%Y-%m-%d %H:%M"),
                     hour_start_nem=df["hour_start_nem"].dt.strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    assert pick_alpha(pd.read_csv(OUT_SCAN), MODEL_SET) == ALPHA, f"alpha differs from {OUT_SCAN}"
    df = load()                       # train + val 的特征表，不含 test 夜
    train = df["split"] == "train"
    hourly = load_hourly(LABEL)       # 全年小时表：test 夜的滞后特征要用到测试期内 D-1 及以前的小时

    # 1. 用全年查找表重建 train + val 特征，应和已提交的特征表逐值相同
    rebuilt, _, _ = build(night_hours(LABEL, ("train", "val")), hourly=hourly)
    assert rebuilt[KEYS].equals(df[KEYS]), "rebuilt rows differ from the feature file"
    diff = max_diff(rebuilt[FEATURES + ["y"]], df[FEATURES + ["y"]])
    assert diff < 1e-12
    print(f"train+val features rebuilt with the full-year lookup: {len(rebuilt)} rows, max diff {diff:.1e}")

    # 2. val 夜：只在 train 上拟合，复现 Step 4
    m_train = make_model(FEATURES, ALPHA).fit(df.loc[train, FEATURES], df.loc[train, "y"])
    pred_trval = pd.Series(m_train.predict(df[FEATURES]), index=df.index)
    for split in ("train", "val"):
        summary, _ = evaluate_split(df, pred_trval, split)
        assert same_as_file(summary, OUT_METRICS.format(split=split)), f"{split} metrics differ from Step 4"
    ref_coefs = pd.read_csv(OUT_COEFS).set_index("feature")["coef"]
    c_train = coefs(m_train)
    assert c_train.index.equals(ref_coefs.index) and max_diff(c_train, ref_coefs) < 1e-12
    print(f"train-only fit: metrics and coefficients identical to Step 4 ({OUT_METRICS.format(split='val')})")

    # 3. test 夜特征：先清空 test 夜本身的标签，滞后特征只从查找表取
    test = night_hours(LABEL, ("test",)).assign(value=np.nan)
    assert (test["night_of"] >= TEST_START).all() and set(test["split"]) == {"test"}
    feats_test, fallback, doubled = build(test, hourly=hourly)
    feats_test = feats_test.drop(columns="y")
    n_nan = int(feats_test[FEATURES].isna().sum().sum())
    assert n_nan == 0, "NaN in test-night features"
    print(f"test-night features: {len(feats_test)} rows, {feats_test['night_of'].nunique()} nights, "
          f"NaN in features {n_nan}, lag1_same_hour D-2 fallback {int(fallback.sum())}, "
          f"two-hour mean {int(doubled.sum())}")

    # 4. test 夜：同一组超参数在 train + val 上重新拟合；拟合两次确认可复现
    assert set(df["split"]) == {"train", "val"}
    m_all = make_model(FEATURES, ALPHA).fit(df[FEATURES], df["y"])
    pred_test = m_all.predict(feats_test[FEATURES])
    again = make_model(FEATURES, ALPHA).fit(df[FEATURES], df["y"]).predict(feats_test[FEATURES])
    assert np.array_equal(pred_test, again), "refit is not reproducible"
    print(f"train+val refit: {len(df)} training rows ({int(train.sum())} train + {int((~train).sum())} val), "
          f"0 test rows; two fits give identical predictions")

    c_all = coefs(m_all)
    coef_table = pd.DataFrame({"feature": c_all.index, "coef_train": c_train.to_numpy(),
                               "coef_trainval": c_all.to_numpy()})
    coef_table.to_csv(OUT_COEFS_TRAINVAL, index=False)
    print("\n== top 10 |coef| of the train+val refit (lag features standardised on train+val)")
    top = coef_table[coef_table["feature"] != "(intercept)"]
    print(top.reindex(top["coef_trainval"].abs().sort_values(ascending=False).index).head(10)
          .round(4).to_string(index=False))

    # 5. 预测文件
    val = df.loc[df["split"] == "val", KEYS].assign(mef_pred=pred_trval[df["split"] == "val"])
    out = pd.concat([val, feats_test[KEYS].assign(mef_pred=pred_test)], ignore_index=True)
    out = out.assign(label=LABEL, rules="label-rules-v1", model=MODEL).sort_values("hour_start_nem")
    out = fmt(out)[COLUMNS]
    assert out["hour_start_nem"].is_unique and out["mef_pred"].notna().all()
    out.to_csv(OUT, index=False)

    per_night = out.groupby(["split", "night_of"], sort=False).size()
    print(f"\n{OUT}: {len(out)} rows | NaN mef_pred {int(out['mef_pred'].isna().sum())} | "
          f"hour_start_nem unique {out['hour_start_nem'].is_unique}")
    for split, g in out.groupby("split", sort=False):
        n = per_night[split]
        print(f"  {split}: {g['night_of'].nunique()} nights, {len(g)} rows, "
              f"{g['night_of'].min()} -> {g['night_of'].max()}, hours per night {dict(n.value_counts().sort_index())}")
    print(f"  columns: {list(out.columns)}")
