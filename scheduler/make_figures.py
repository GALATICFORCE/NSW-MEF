"""Figures for report Sections 5.1 and 5.2 (owner: Oscar).

Reads only scheduler/schedule_results_v1.csv (commit 879aa3a) and uses the
full-year comparison: 364 complete nights, 2025-07-01 to 2026-06-29, actual signals.
mef_forecast is left out here because it only exists on the 53 test nights (Section 5.3).

Run from the repository root:  py scheduler/make_figures.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS = "scheduler/schedule_results_v1.csv"
OUT = "scheduler/"

# Full-year strategies, in the order they are shown
STRATS = ["fixed", "mef_oracle", "aei", "price"]
NAMES = {"fixed": "Fixed (18:00-22:00)", "mef_oracle": "MEF (true, label C)",
         "aei": "AEI", "price": "Price"}
COLOURS = {"fixed": "#8c8c8c", "mef_oracle": "#2a7f62", "aei": "#d08c2a", "price": "#3b6fb6"}
# Sydney start hours of the charging window 18:00-07:00, in night order
HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]

res = pd.read_csv(RESULTS)
res = res[res["strategy"].isin(STRATS)]
n_nights = res["night_of"].nunique()
assert n_nights == 364, n_nights
# Sanity check from B4: every strategy charges the same energy every night
assert res["energy_kwh"].nunique() == 1


def hour_set(text):
    """'22;23;1;2' -> [22, 23, 1, 2]. On the 14-hour night of 2026-04-04 hour 2 can appear twice."""
    return [int(h) for h in str(text).split(";")]


# ---------- Figure 5.1: which hours each strategy picks, and the overlap with MEF ----------

# (a) share of nights on which each Sydney hour was chosen
share = pd.DataFrame(0.0, index=STRATS, columns=HOURS)
for s in STRATS:
    for hours in res.loc[res["strategy"] == s, "hours_chosen"]:
        for h in hours.split(";"):
            share.loc[s, int(h)] += 1
share = share / n_nights * 100

# (b) number of hours each strategy shares with mef_oracle on the same night (0 to 4)
wide = res.pivot(index="night_of", columns="strategy", values="hours_chosen")
overlap = pd.DataFrame({
    s: [len(set(hour_set(a)) & set(hour_set(b))) for a, b in zip(wide[s], wide["mef_oracle"])]
    for s in ["fixed", "aei", "price"]
}, index=wide.index)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})

im = ax1.imshow(share.values, cmap="Greens", vmin=0, vmax=100, aspect="auto")
ax1.set_xticks(range(len(HOURS)))
ax1.set_xticklabels([f"{h:02d}" for h in HOURS])
ax1.set_yticks(range(len(STRATS)))
ax1.set_yticklabels([NAMES[s] for s in STRATS])
ax1.set_xlabel("Start of charging hour (Sydney time)")
for i in range(len(STRATS)):
    for j in range(len(HOURS)):
        v = share.values[i, j]
        ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                 color="white" if v > 60 else "black")
ax1.set_title("(a) % of nights each hour is chosen")
fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02, label="% of nights")

width = 0.27
for k, s in enumerate(["fixed", "aei", "price"]):
    counts = overlap[s].value_counts().reindex(range(5), fill_value=0) / n_nights * 100
    ax2.bar(np.arange(5) + (k - 1) * width, counts.values, width,
            color=COLOURS[s], label=f"{NAMES[s]} (mean {overlap[s].mean():.2f})")
ax2.set_xticks(range(5))
ax2.set_xlabel("Hours shared with the MEF (label C) choice, out of 4")
ax2.set_ylabel("% of nights")
ax2.set_title("(b) Overlap with the true-MEF schedule")
ax2.legend(fontsize=8, frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT + "fig_5_1_hours_overlap_v1.png", dpi=200)
plt.close(fig)

# ---------- Figure 5.2: emissions and cost relative to fixed charging ----------

cols = {"emissions_kg_B": "Emissions, label B", "emissions_kg_C": "Emissions, label C",
        "emissions_kg_Cfwd": "Emissions, label C-fwd", "cost_aud": "Wholesale cost"}
totals = res.groupby("strategy")[list(cols)].sum().loc[STRATS]
change = (totals / totals.loc["fixed"] - 1) * 100          # % change vs fixed

# Table behind the figure, so the report can quote exact numbers
table = totals.round(2)
for c in cols:
    table[c + "_pct_vs_fixed"] = change[c].round(1)
table.insert(0, "nights", n_nights)
table.to_csv(OUT + "fig_5_2_totals_v1.csv")

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(cols))
width = 0.26
for k, s in enumerate(["mef_oracle", "aei", "price"]):
    vals = change.loc[s].values
    bars = ax.bar(x + (k - 1) * width, vals, width, color=COLOURS[s], label=NAMES[s])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(list(cols.values()))
ax.set_ylabel("Change vs fixed charging (%)")
ax.set_title("Estimated emissions and wholesale cost relative to fixed charging, 364 nights")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
lo, hi = change.drop(index="fixed").values.min(), change.drop(index="fixed").values.max()
ax.set_ylim(min(lo, 0) - 6, max(hi, 0) + 6)
fig.tight_layout()
fig.savefig(OUT + "fig_5_2_emissions_cost_v1.png", dpi=200)
plt.close(fig)

# Print the numbers so they can be checked against the figures
print("Nights:", n_nights)
print("\nShare of nights each hour is chosen (%):\n", share.round(0))
print("\nMean hours shared with mef_oracle:\n", overlap.mean().round(2))
print("Nights with identical 4 hours (%):\n", (overlap == 4).mean().mul(100).round(1))
print("\nTotals:\n", totals.round(2))
print("\n% change vs fixed:\n", change.round(1))
