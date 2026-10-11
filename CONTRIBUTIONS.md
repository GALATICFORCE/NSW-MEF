## Week of 6 Oct 2026 — Kunyu
- Did: set up the reproducible NEMED environment, downloaded the full year of MEF data, built and checked the 5-minute table, froze label rules v1, set the research design and the per-night scheduling with the team.
- Where: repo nsw-mef (tag label-rules-v1); handoff docs in the "Project plan & Hand-off docs" tab.
- Changed for others: shared hourly format and night_of forecast format (handoff docs A5); load scenario, AEI scope, fixed baseline and dates decided 7 Oct (A6).
- AI used for: drafting the label-rule document and the data-processing code (reviewed and accepted by me); drafting the handoff docs and meeting slides; checking methods.

## Week of 12 Oct 2026 — Kunyu
- Did: built the four hourly MEF label tables (A, B, C, C-fwd) with an automatic validation script; fixed the night definition and the train/validation/test split (250/54/53 nights); wrote the forecast metrics (MAE, regret, rank correlation) and two naive baselines; tested NEMED for AEI, found that its intensity column mixes sent-out emissions with as-generated energy (4 to 5% low), and built the full-year 5-minute AEI from unit-level data instead.
- Where: repo github.com/GALATICFORCE/NSW-MEF (now public), commits c67ffbe, b8eeb14, 02ab8dc, d62f074, a98ad68; hand-off docs updated 10 and 11 Oct.
- Changed for others: Oscar: the four MEF hourly files are ready, tables are joined on hour_start_nem, and the forecast file gets two extra columns (hour_start_nem, split) and covers validation and test nights. Otis: gets the 5-minute AEI file, so he only aggregates it to hourly. Everyone: split dates and current status are in hand-off docs A4 to A6, and the report can cite the public repo link.
- AI used for: writing and running the label, baseline and AEI code in the repo under my step-by-step instructions (I reviewed each report and made the decisions); checking methods and reviewing results; updating the hand-off docs; drafting this entry.
