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
- `12-elo-conference-model.ipynb` — testing whether extending Elo with conference tournament
  games improves on notebook 11; it didn't, notebook 11's version was kept
- `13-conference-strength-model.ipynb` — testing a conference-strength feature; new best model
- `14-elo-k-tuning-model.ipynb` — tuning the Elo K-factor per gender and fixing a conference-
  strength bug found along the way; neither improved on notebook 13, which remains the best model
- `15-interaction-features-model.ipynb` — testing feature interaction terms (SeedDiff x EloDiff,
  SeedDiff x ConfStrengthDiff); no meaningful change, notebook 13 remains the best model
- `16-generate-submission.ipynb` — retrains the winning model on all historical data and
  generates `submissions/submission.csv` for the actual competition submission
- `17-team-clustering.ipynb` — unsupervised k-means clustering of teams by seed, win rate,
  scoring margin, Elo, and conference strength, exploring whether meaningful tiers emerge
- `18-cluster-prediction-model.ipynb` — testing whether clustering can improve the prediction: a
  cluster-only lookup table performed poorly, but adding cluster rank as a 6th feature to the
  existing model gave a small improvement; new best model
- `19-generate-submission-v2.ipynb` — retrains the new best model (notebook 18) on all historical
  data and generates `submissions/submission_v2.csv`

Not every notebook was successful — the detailed stats, power rankings, and tree-based models
all underperformed the simple baseline. Those results are kept rather than removed, since they
ruled out real possibilities and shaped what came next. See `docs/training_data_plan.docx` for
the reasoning behind the feature and modeling choices, and `docs/project_notes.md` for the full
data schema, weekly plan, and EDA notes.

## Result

First model: gender-split logistic regression with 5 features (seed, win rate, scoring margin,
Elo rating, conference strength) — 0.1675 average Brier score across 2021-2025 walk-forward
validation (see `notebooks/13-conference-strength-model.ipynb`).

Submitted to the actual 2026 competition leaderboard: **0.1302747** Brier score. That's notably
better than the walk-forward estimate, which is expected rather than a sign the model is secretly
stronger than measured — 2026 was, by other competitors' accounts, an unusually low-upset
("chalky") tournament, and with only ~126 total scored games, a favorite-heavy year plus a small
sample size can swing the actual score substantially in either direction. The walk-forward number
remains the more honest estimate of this model's typical performance; the leaderboard score
reflects how it did on the one tournament that actually counted.

Second model: the same 5 features plus a per-gender cluster rank (`notebooks/18-cluster-prediction-model.ipynb`)
— 0.1669 average walk-forward Brier, a small improvement. Submitted as `submissions/submission_v2.csv`
(see `notebooks/19-generate-submission-v2.ipynb`): **rank 482** on the leaderboard, an improvement
over the first submission.

## Setup

```
pip install -r requirements.txt
```

Competition data isn't included — download it from the competition's Data tab and unzip into
`data/`.
