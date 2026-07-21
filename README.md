# March Machine Learning Mania 2026

A project from the LeverX internship program, working through [Kaggle's March Machine Learning
Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) — forecasting
the 2026 NCAA Men's and Women's basketball tournaments (Brier score / MSE evaluation).

## Notebooks

- `01-eda.ipynb`, `02-eda-plots.ipynb` — exploring the raw data and visualizing what predicts
  tournament outcomes
- `03-baseline-model.ipynb` — a minimal logistic regression baseline (seed, win rate, scoring
  margin)
- `04-detailed-model.ipynb`, `05-ordinal-model.ipynb` — testing detailed box-score stats and
  power rankings against the baseline
- `06-gbm-model.ipynb` — testing LightGBM/XGBoost against the same baseline
- `07-gender-split-model.ipynb` — testing separate men's/women's models against the pooled
  baseline
- `08-data-scope-test.ipynb` — distributional tests (Levene's, Welch's t-test, KS) deciding
  whether training should widen from tournament-only to tournament + regular season, per gender
- `09-combined-decisions-model.ipynb` — testing the gender split and data-scope decisions
  stacked together
- `10-momentum-model.ipynb` — testing last-10-game recent-form features against the gender-split
  baseline
- `11-elo-model.ipynb` — testing a margin-of-victory-weighted Elo rating feature; new best model

Not every notebook was successful — the detailed stats, power rankings, and tree-based models
all underperformed the simple baseline. Those results are kept rather than removed, since they
ruled out real possibilities and shaped what came next. See `docs/training_data_plan.docx` for
the reasoning behind the feature and modeling choices, and `docs/project_notes.md` for the full
data schema, weekly plan, and EDA notes.

## Setup

```
pip install -r requirements.txt
```

Competition data isn't included — download it from the competition's Data tab and unzip into
`data/`.
