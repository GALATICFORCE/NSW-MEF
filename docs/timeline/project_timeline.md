# Project timeline (Gantt chart)

Planned versus actual progress for the whole team, from kick-off to submission. Snapshot: **21 October 2026** (the results-freeze checkpoint). Wednesdays are team meetings.

- **Source of truth:** [`project_timeline.csv`](project_timeline.csv): one row per task, with planned dates, actual dates, status and evidence (a commit, tag or document).
- **Regenerate** after editing the CSV: `python docs/timeline/make_timeline.py` (rewrites the PNG and the chart below; update `TODAY` in the script first).
- **For the report appendix:** [`project_timeline.png`](project_timeline.png) shows each task's planned window (grey) above its actual bar.

![Planned vs actual timeline](project_timeline.png)

## Interactive view (renders on GitHub)
Bars use actual dates where known; late items run to today; future items use planned dates.

<!-- MERMAID:START -->
```mermaid
gantt
    title NSW MEF project timeline (actual dates where known)
    dateFormat YYYY-MM-DD
    axisFormat %d %b
    todayMarker on
    section Team
    Proposal AI declarations submitted :milestone, done, 2026-10-06, 0d
    Kick-off - design and roles agreed :milestone, done, 2026-10-07, 0d
    Checkpoint - label tables and round-1 inputs :milestone, done, 2026-10-14, 0d
    Checkpoint - ML results frozen :milestone, done, 2026-10-20, 0d
    Checkpoint - full draft review :milestone, 2026-10-28, 0d
    Final report due :milestone, 2026-11-01, 0d
    Individual Q&A :milestone, 2026-11-04, 0d
    section Kunyu - MEF labels and forecast
    Environment, MEF download, 5-minute table :done, 2026-10-04, 2026-10-06
    Label rules v1 frozen :done, 2026-10-05, 2026-10-06
    Hourly MEF label tables (A/B/C/C-fwd) :done, 2026-10-06, 2026-10-08
    Night split, metrics, baselines :done, 2026-10-07, 2026-10-12
    Full-year 5-minute AEI for Otis :done, 2026-10-11, 2026-10-12
    Ridge model (validation) :done, 2026-10-11, 2026-10-12
    Label diagnostics :done, 2026-10-11, 2026-10-13
    Gradient-boosting model (validation) :done, 2026-10-12, 2026-10-13
    Forecast file to Oscar :done, 2026-10-15, 2026-10-18
    Error analysis (validation) :done, 2026-10-19, 2026-10-21
    Test-set evaluation (run once) :done, 2026-10-20, 2026-10-21
    5.3 table and figures to report :active, 2026-10-21, 2026-10-22
    Report 4.1 (MEF labels), 4.2, 4.3 drafts :2026-10-22, 2026-10-27
    section Otis - AEI and price data
    Price data download and validation :done, 2026-10-10, 2026-10-13
    AEI and price hourly tables :done, 2026-10-13, 2026-10-14
    Data dictionary and quality report :crit, active, 2026-10-12, 2026-10-22
    Push files to data_prep/ :active, 2026-10-21, 2026-10-22
    Report 4.1 (AEI and price) draft :2026-10-22, 2026-10-27
    section Oscar - scheduler and evaluation
    Scheduler and night slicer :done, 2026-10-12, 2026-10-15
    Round 1 - actual signals, 364 nights :done, 2026-10-14, 2026-10-15
    Round 2 - forecast on test nights (B, C, C-fwd) :done, 2026-10-17, 2026-10-22
    Report 4.4, figures for 5.1-5.2 :2026-10-23, 2026-10-27
    section Ciyue - report
    Report requirements and outline v1 :done, 2026-10-06, 2026-10-12
    Review of mechanism note and limitations :done, 2026-10-13, 2026-10-15
    Outline revised after freeze :done, 2026-10-21, 2026-10-22
    Sections 1-3, 4.5, 5 integration, 6-7 :2026-10-22, 2026-10-28
    section All - final report
    Full draft review and revision :2026-10-28, 2026-11-01
    Executive summary, references, AI-use appendix :2026-10-28, 2026-11-01
```
<!-- MERMAID:END -->

## Notes
- **Dates after 21 October are proposed**, pending team agreement (in particular Oscar's round-2 deadline of 23 October).
- **Status meanings:** *done (late)* = finished after its planned end; *late* = planned end has passed and the item is not yet received; *in progress* = started, planned end not yet passed.
- ML results are frozen under the git tag `results-v1`; anything run after the freeze goes to the report appendix only.
