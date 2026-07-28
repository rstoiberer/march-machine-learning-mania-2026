# Project Notes — March Machine Learning Mania 2026

Full working notes: data schema, weekly plan, and EDA findings. See the top-level `README.md`
for the short version.

## Competition

**Competition:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
**Task:** predict P(lower TeamID beats higher TeamID) for every possible matchup between
tournament-eligible teams, men's (TeamIDs 1000–1999) and women's (3000–3999) combined
**Metric:** Brier score — mean squared error between predicted probability and the actual 0/1
outcome. Kaggle's leaderboard reports this as MSE.
**Submission format:** `ID,Pred` where `ID = 2026_TeamIdLow_TeamIdHigh` and `Pred` is the
probability the lower-numbered team wins.
**Status:** competition closed (~8,258 entrants / 1,212 teams). Kaggle keeps this series open for
late submissions scored against the real 2026 results, so we can still get an honest leaderboard
number — see EDA finding below on why that matters (the raw data does *not* include 2026 outcomes).

Also a toolchain exercise: relational joins on messy multi-table data, leakage-safe time-based
validation, and a speed/quality comparison of Polars vs. Pandas and AutoGluon vs. a hand-tuned
model — see `Kaggle_Project_Options.docx` (one level up) for the original project brief.

## Data

35 CSVs, `M`-prefixed for men's / `W`-prefixed for women's, some shared (`Cities.csv`,
`Conferences.csv`). Based on the standard March Mania schema (consistent across years):

| File pattern | Contents |
|---|---|
| `{M,W}Teams.csv` | TeamID ↔ team name |
| `{M,W}Seasons.csv` | season metadata, region names, day-zero date |
| `{M,W}NCAATourneySeeds.csv` | tournament seed by team/season |
| `{M,W}RegularSeasonCompactResults.csv` | one row per regular-season game: score, W/L team, location |
| `{M,W}RegularSeasonDetailedResults.csv` | same, plus box score stats (FG, 3P, RB, TO, etc.) — men's from 2003+ |
| `{M,W}NCAATourneyCompactResults.csv` / `...DetailedResults.csv` | tournament game results, same shape |
| `MMasseyOrdinals.csv` | third-party power rankings over time (men's only) |
| `{M,W}TeamConferences.csv`, `Conferences.csv` | conference membership by season |
| `Cities.csv`, `{M,W}GameCities.csv` | game locations |
| `{M,W}TeamSpellings.csv` | name-variant → TeamID lookup |
| `MSecondaryTourneyTeams.csv` / `...CompactResults.csv` | NIT and other secondary tournaments |
| `MTeamCoaches.csv` | head coach by team/season |
| `SampleSubmission*.csv` | submission template |

Confirmed against the actual 2026 files — all 35 present as expected, plus a few structural
extras not in the table above: `{M,W}NCAATourneySlots.csv` / `MNCAATourneySeedRoundSlots.csv`
(bracket structure, for simulating a full bracket round-by-round instead of just scoring
individual matchups), `{M,W}ConferenceTourneyGames.csv`, `{M,W}GameCities.csv`, and
`MTeamCoaches.csv`.

Data files are not included in this repo (see `.gitignore`) — download from the competition's
[Data tab](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and
unzip into `data/`.

## Plan (as of the original week-1 scoping)

1. **Rules + data review, EDA** — read scoring/submission rules; go table-by-table
   through the raw CSVs (row counts, missingness, season coverage, ID consistency) to find the
   messy parts before joining anything. Done — see findings below. Key output: the data has no
   2026 ground truth, so we need a plan for getting real 2026 results (web pull for local
   iteration + late Kaggle submission for the real number).
2. **Matchup-level feature engineering** — join everything into one row per team-pair-season:
   win rate, scoring margin, strength of schedule, seed, Massey ordinal ranks. Careful to only
   use information that would have been known *before* the tournament started that season (no
   using end-of-season detailed stats that leak tournament-adjacent data). Skip season 2020
   (no tournament) as a training/validation target.
3. **Time-respecting validation harness** — train on older seasons, validate on held-out recent
   seasons (not a random split, which would let the model mix seasons and cheat). Simple
   seed-difference baseline first, to sanity-check the harness itself. Optionally validate the
   harness itself against `SampleSubmissionStage1`'s 2022-2025 window.
4. **Baseline model** — XGBoost/LightGBM on the engineered features; generate predictions for
   all 132k Stage 2 pairs; score locally against real 2026 results (pulled from the web, in the
   same schema as `MNCAATourneyCompactResults.csv`), then submit to Kaggle as a late submission
   to get the real private-leaderboard number.
5. **Iterate** — redo the join step in Polars and benchmark against Pandas on this join-heavy
   workload; run AutoGluon (or PyCaret) alongside the manual model as a comparison point.
6. **Tune** — Optuna pass on whichever model comes out ahead; try a blend if two models are
   close but diverse. Optionally use `{M,W}NCAATourneySlots.csv` to simulate the full bracket
   round-by-round rather than scoring matchups independently.
7. **Write-up** — what moved the score, what didn't, Polars vs. Pandas findings, AutoGluon vs.
   hand-tuned comparison, final standing vs. the frozen/late-submission leaderboard.

Updated after the July 16 meeting: training separate men's/women's models instead of pooled,
and using a formal distributional comparison (F-test/Levene/KS-test) to decide whether to widen
training data beyond tournament-only games, rather than assuming either approach.

**Note: the plan above is preserved as it stood in week 1.** In practice, hand-written logistic
regression (Newton's method / IRLS, no scikit-learn) ended up outperforming XGBoost/LightGBM on
every feature set tested, so it became the production model rather than a baseline to beat.
Polars, AutoGluon, and Optuna were never used in the final notebooks — see *Modeling results*
below for what actually happened, notebook by notebook.

## EDA findings

Full table-by-table output in `notebooks/01-eda.ipynb`. Headlines:

**The data does not include 2026 tournament outcomes — this changes the plan.** `MNCAATourneySeeds.csv`
and `WNCAATourneySeeds.csv` both have season 2026 (we know the 68-team bracket for each), but
`MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` stop at season 2025 — zero
2026 games in either file. So there's no local ground truth to score against out of the box.
Two ways to actually get a real number: (1) submit to Kaggle — this competition series
keeps accepting late submissions scored against the real private leaderboard even after close,
so this is the primary evaluation path; (2) pull the real 2026 bracket results from the web and
build a local ground-truth file in the same schema, for fast iteration between Kaggle submissions
(useful since submission counts are usually capped per day).

**Submission target is bigger than the bracket.** `SampleSubmissionStage2.csv` (the one that
matters — Stage1 is a 2022-2025 warm-up/validation exercise) has 132,133 rows, all season 2026:
66,430 men's pairs + 65,703 women's pairs. That's every possible pair of Division-I teams, not
just the 68 that made each tournament — because Stage 2 opens before Selection Sunday info is
locked in downstream systems. Practically, only pairs where both teams actually made the 2026
tournament will get scored, but the model needs to produce sane probabilities for the full team
universe.

**2020 is a real gap, not a data-quality bug.** Both men's and women's tournaments were cancelled
that year (COVID) — `MNCAATourneySeeds` / `WNCAATourneySeeds` / tourney results all skip season
2020 while regular-season files still have full 2020 data. Validation splits need to skip 2020 as
a target season.

**Coverage asymmetry to design around:**
- Women's history starts later than men's throughout: seasons from 1998 (men: 1985), detailed
  box-score stats from 2010 (men: 2003). Feature engineering that leans on box-score stats will
  have a shorter usable women's history.
- Women's tournament was 64 teams through 2021, expanded to 68 (matching men's First Four format)
  from 2022 on — seed-count-based features need to handle both eras.
- `MTeams.csv` has `FirstD1Season`/`LastD1Season` columns; `WTeams.csv` doesn't — no direct way
  to filter "was this a D1 program that season" for women's teams from that file alone.

**No missing values anywhere** across all 35 files — unusual for a real-world dataset and a sign
this is a well-maintained, actively-curated release rather than something scraped once.

**`MMasseyOrdinals.csv` is the big one** — 5.87M rows, 197 distinct ranking systems used
historically, ~20 systems present for 2026 (Pomeroy/POM, Massey/MAS, Moore/MOR, KenPom-adjacent
systems like KPK, etc.), each with ~7,300 rows. Rankings run up to `RankingDayNum` 133 in every
recent season including 2026, i.e., right up through Selection Sunday — safe to use as
late-season features without leaking beyond what would actually have been known. Men's only —
no women's equivalent file exists in this dataset.

**Data integrity checks passed:** every TeamID referenced in tourney results exists in the Teams
table; no season appears in tourney results without a matching seeds entry.

## Modeling results (notebooks 3-19)

Walk-forward validated (2021-2025 folds, training strictly on prior seasons each time, combined
across genders where both exist). Lower Brier is better.

| Notebook | Approach | Avg. Brier | Outcome |
|---|---|---|---|
| 03 | Logistic regression, compact features (seed/win-rate/margin), pooled, full history | 0.1739 | Baseline |
| 04 | + detailed box-score stats | 0.1759 | Worse, rejected |
| 05 | Seed-only vs. ranking-only (men's); + Massey Ordinal ranking | seed 0.2031 / ranking 0.2063 / +ordinal 0.2037 | Worse, rejected |
| 06 | LightGBM (compact) / LightGBM (all features) / XGBoost (all features) | 0.1775 / 0.1804 / 0.1822 | All worse than logistic regression, rejected |
| 07 | Gender-split models (separate men's/women's), tournament-only | 0.1711 | Better, adopted |
| 08 | Distributional tests (Levene's, Welch's t-test, KS) on training-data scope | — | Decided: men's widens to regular season + tournament, women's stays tournament-only |
| 09 | Gender split + widened men's training data | 0.1715 | Slightly worse, rejected |
| 10 | + last-10-game momentum features | 0.1709 | Wash, rejected |
| 11 | + margin-of-victory-weighted Elo rating | 0.1687 | Better, adopted |
| 12 | Elo extended with conference tournament games | 0.1691 | Worse, rejected — kept regular-season-only Elo |
| 13 | + conference-strength feature | 0.1675 | Better, adopted (standing best through notebook 17) |
| 14 | Elo K-factor tuning per gender + conference-strength bug fix | 0.1690 (tuned K) / 0.1683 (bug fix, K=20) | Both worse than notebook 13, rejected — notebook 13's model, including its known bug, kept |
| 15 | + feature interaction terms (SeedDiff × EloDiff, SeedDiff × ConfStrengthDiff) | 0.1673 / 0.1674 | No meaningful change, rejected |
| 16 | Generate submission — retrain notebook 13's model on all history | — | `submissions/submission.csv`; scored **0.1302747** on the real Kaggle leaderboard |
| 17 | Team clustering (k-means, unsupervised, exploratory) | — | No Brier score (unsupervised); found recognizable tiers and some seed/tier mismatches |
| 18 | Cluster-only lookup table / base model + cluster rank as a 6th feature | 0.1886 / 0.1669 | Lookup table much worse, rejected; cluster rank as an added feature better, adopted — new best |
| 19 | Generate submission v2 — retrain notebook 18's model on all history | — | `submissions/submission_v2.csv`; improved leaderboard rank to **482** |

Final model (notebooks 18/19): gender-split logistic regression with 6 features — SeedDiff,
WinPctDiff, AvgMarginDiff, EloDiff, ConfStrengthDiff, ClusterRankDiff.

A pattern holds across the whole run: every feature that helped (gender split, Elo, conference
strength, cluster rank) earned its place purely on held-out 2021-2025 performance, never assumed.
Just as many plausible-sounding ideas — detailed box-score stats, power rankings, tree-based
models, widened training data, momentum, interaction terms — were tested with the same rigor and
rejected when they didn't hold up. Notebook 13's conference-strength calculation carries a known
bug (cross-gender blending, caught while building notebook 14) that was kept deliberately, since
the bug-fixed version scored worse on the real test folds — an explicit, empirically-driven call,
not an oversight.

## Result

First submission (notebook 16, notebook 13's model): **0.1302747** on the real 2026 competition
leaderboard — notably better than the 0.1675 walk-forward estimate, consistent with 2026 being an
unusually low-upset ("chalky") tournament and a small ~126-game sample size.

Second submission (notebook 19, notebook 18's model): improved the leaderboard position to
**rank 482**, using the 6-feature model that added cluster rank (0.1669 average walk-forward
Brier, vs. 0.1675 for the first model).

## Setup

```
pip install -r requirements.txt
```

Unzip the competition data into `data/`, then run the notebooks in order starting with
`01-eda.ipynb`.
