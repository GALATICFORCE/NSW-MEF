"""Project Gantt chart (planned vs actual) from project_timeline.csv.

Writes project_timeline.png (for the report appendix) and the Mermaid block in project_timeline.md.
Run from the repository root:  python docs/timeline/make_timeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
TODAY = pd.Timestamp("2026-10-21")          # snapshot date; update when the CSV is updated
DAY = pd.Timedelta(days=1)

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR = {"done": "#2a78d6", "done_late": "#2a78d6", "in_progress": "#fab219",
         "late": "#d03b3b", "planned": "#86b6ef"}
LABEL = {"done": "done", "done_late": "done (late)", "in_progress": "in progress",
         "late": "late", "planned": "planned"}


def load():
    df = pd.read_csv(HERE / "project_timeline.csv", dtype=str).fillna("")
    for c in ["planned_start", "planned_end", "actual_start", "actual_end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def plot(df):
    # one header row per section, then its tasks; y runs top to bottom
    layout, ypos = [], 0
    for sec, g in df.groupby("section", sort=False):
        layout.append(("header", sec, ypos)); ypos += 1
        for i in g.index:
            layout.append(("task", i, ypos)); ypos += 1
    n_rows = ypos
    fig, ax = plt.subplots(figsize=(13, 0.32 * n_rows + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    yt, yl = [], []
    for kind, key, row in layout:
        yi = n_rows - 1 - row
        if kind == "header":
            ax.text(-0.005, yi, key, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=9, fontweight="bold", color=INK)
            if row > 0:
                ax.axhline(yi + 0.5, color=GRID, linewidth=0.8, zorder=1)
            continue
        r = df.loc[key]
        st = r["status"]
        yt.append(yi); yl.append(r["task"])
        if st.startswith("milestone"):
            when = r["actual_end"] if st == "milestone_done" else r["planned_end"]
            ax.scatter(when + DAY / 2, yi, marker="D", s=70, zorder=3,
                       color=COLOR["done"] if st == "milestone_done" else "white",
                       edgecolor=COLOR["done"], linewidth=1.5)
            note = ("done - " if st == "milestone_done" else "planned - ") + when.strftime("%d %b")
        else:
            if pd.notna(r["planned_start"]):           # planned window: grey bar above the actual bar
                ax.barh(yi + 0.2, (r["planned_end"] - r["planned_start"]) + DAY, left=r["planned_start"],
                        height=0.18, color=AXIS, zorder=2)
            if st == "planned":
                ax.barh(yi - 0.08, (r["planned_end"] - r["planned_start"]) + DAY, left=r["planned_start"],
                        height=0.36, color="white", edgecolor=COLOR["planned"], linewidth=1.5,
                        hatch="////", zorder=2)
            else:
                end = r["actual_end"] if pd.notna(r["actual_end"]) else TODAY
                ax.barh(yi - 0.08, (end - r["actual_start"]) + DAY, left=r["actual_start"],
                        height=0.36, color=COLOR[st], zorder=2)
            note = LABEL[st] + (f" - {r['evidence']}" if r["evidence"] else "")
        ax.text(pd.Timestamp("2026-11-06"), yi, "  " + note, va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8, color=INK)
    y = [n_rows - 1]
    n = n_rows
    ax.axvline(TODAY + DAY / 2, color=INK2, linestyle="--", linewidth=1, zorder=4)
    ax.text(TODAY + DAY / 2, y[0] + 0.9, f" today ({TODAY:%d %b})", fontsize=8, color=INK2, va="bottom")
    ax.set_xlim(pd.Timestamp("2026-10-03"), pd.Timestamp("2026-11-06"))
    ax.set_ylim(-0.8, n - 0.2 + 0.9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    handles = [Patch(color=AXIS, label="planned window"),
               Patch(color=COLOR["done"], label="done"),
               Patch(color=COLOR["in_progress"], label="in progress"),
               Patch(color=COLOR["late"], label="late (bar runs to today)"),
               Patch(facecolor="white", edgecolor=COLOR["planned"], hatch="////", label="planned (future)"),
               Line2D([], [], marker="D", color=COLOR["done"], linestyle="", label="checkpoint (filled = done)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=6, frameon=False,
              fontsize=8, labelcolor=INK2)
    ax.set_title("NSW MEF project: planned vs actual timeline (Wednesdays = team meetings)",
                 loc="left", fontsize=11, color=INK, pad=22)
    fig.tight_layout()
    fig.savefig(HERE / "project_timeline.png", facecolor=SURFACE, bbox_inches="tight")


def mermaid(df):
    tag = {"done": "done", "done_late": "done", "in_progress": "active", "late": "crit, active", "planned": ""}
    lines = ["```mermaid", "gantt", "    title NSW MEF project timeline (actual dates where known)",
             "    dateFormat YYYY-MM-DD", "    axisFormat %d %b", f"    todayMarker on"]
    for sec, g in df.groupby("section", sort=False):
        lines.append(f"    section {sec.replace(':', ' -')}")
        for _, r in g.iterrows():
            name = r["task"].replace(":", " -")
            st = r["status"]
            if st.startswith("milestone"):
                when = r["actual_end"] if st == "milestone_done" else r["planned_end"]
                flag = "milestone, done" if st == "milestone_done" else "milestone"
                lines.append(f"    {name} :{flag}, {when:%Y-%m-%d}, 0d")
                continue
            if st == "planned":
                start, end = r["planned_start"], r["planned_end"]
            elif st == "late":
                start, end = r["actual_start"], max(TODAY, r["planned_end"])
            else:
                start = r["actual_start"]
                end = r["actual_end"] if pd.notna(r["actual_end"]) else TODAY
            flags = (tag[st] + ", ") if tag[st] else ""
            lines.append(f"    {name} :{flags}{start:%Y-%m-%d}, {(end + DAY):%Y-%m-%d}")
    lines.append("```")
    return "\n".join(lines)


def write_md(df):
    md = HERE / "project_timeline.md"
    text = md.read_text(encoding="utf-8")
    start, end = "<!-- MERMAID:START -->", "<!-- MERMAID:END -->"
    a, b = text.index(start) + len(start), text.index(end)
    md.write_text(text[:a] + "\n" + mermaid(df) + "\n" + text[b:], encoding="utf-8")


if __name__ == "__main__":
    df = load()
    plot(df)
    write_md(df)
    print(f"{len(df)} rows; wrote project_timeline.png and the Mermaid block in project_timeline.md")
