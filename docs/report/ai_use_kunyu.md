# AI use — Kunyu (MEF labels and forecasting)

**What AI was used for**
- Drafting the label-rule document (`docs/label_rules_v1.md`) and the data-processing, modelling and evaluation code in `src/`, step by step under my written instructions; I reviewed each step's report before deciding the next step.
- Explaining concepts (ridge penalty, gradient boosting, leakage) using numbers from this repository.
- Drafting the hand-off documents for teammates and the weekly contribution entries.
- Checking methods and reviewing results; a second AI session, with no access to the code, independently re-computed headline numbers from the public repository.

**What AI was not used for**
- Deciding the research questions, the load scenario, the label assumptions, the split, the selection rule or the conclusions. These were team or my decisions and are recorded with dates in `docs/model_selection.md` and the commit history.

**How outputs were checked**
- Every script reports expected vs actual results before being accepted; mismatches were reported, not silently fixed (examples: the 0 NaN hours in Step 1, the battery-filled hours being hardest to predict on validation but easiest on test).
- Test-set evaluation: script committed before running (`8cffa41`), run once (`0c6ca64`, tag `results-v1`), independent recomputation of test MAE 0.1506 and regret 1.061 / 1.072 agreed.

**Full record**: "Kunyu Guo — AI Background Records" tab in the proposal document (prompts and outputs, chronological).
