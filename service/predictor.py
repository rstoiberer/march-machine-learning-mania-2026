"""Core prediction logic, kept separate from the web/FastAPI layer so it can be tested and
understood on its own. Loads model_export.json once and exposes one function: predict_matchup.
"""
import json
import math
from pathlib import Path

_EXPORT_PATH = Path(__file__).parent / "model_export.json"

with open(_EXPORT_PATH) as f:
    _MODEL = json.load(f)

FEATURES_ORDER = _MODEL["features_order"]
WEIGHTS = _MODEL["weights"]        # {"M": [...7 numbers...], "W": [...7 numbers...]}
TEAMS = _MODEL["teams"]            # {"1181": {...}, "3125": {...}, ...}


def _sigmoid(z: float) -> float:
    # squashes any real number into a probability between 0 and 1
    z = max(-30.0, min(30.0, z))  # avoid overflow for extreme inputs
    return 1.0 / (1.0 + math.exp(-z))


class UnknownTeamError(Exception):
    pass


class MismatchedGenderError(Exception):
    pass


def _predict_canonical(lower_id: str, higher_id: str) -> float:
    """Returns P(lower_id beats higher_id), evaluating the model in the exact same direction
    it was trained in throughout this project: Team1 is always the lower TeamID."""
    t1 = TEAMS[lower_id]
    t2 = TEAMS[higher_id]
    gender = t1["gender"]
    w = WEIGHTS[gender]  # [intercept, w_SeedDiff, w_WinPctDiff, w_AvgMarginDiff, w_EloDiff, w_ConfStrengthDiff, w_ClusterRankDiff]

    diffs = {
        "SeedDiff": t2["seed"] - t1["seed"],
        "WinPctDiff": t1["win_pct"] - t2["win_pct"],
        "AvgMarginDiff": t1["avg_score_margin"] - t2["avg_score_margin"],
        "EloDiff": t1["elo"] - t2["elo"],
        "ConfStrengthDiff": t1["conf_strength"] - t2["conf_strength"],
        "ClusterRankDiff": t1["cluster_rank"] - t2["cluster_rank"],
    }
    x = [diffs[name] for name in FEATURES_ORDER]

    # z = intercept + (weight_1 * diff_1) + (weight_2 * diff_2) + ... + (weight_6 * diff_6)
    z = w[0] + sum(wi * xi for wi, xi in zip(w[1:], x))
    return _sigmoid(z)


def predict_matchup(team1: int, team2: int) -> float:
    """Returns P(team1 beats team2), for team1/team2 in ANY order. Internally, this always
    evaluates the model with the lower TeamID first (matching how it was trained), then
    returns either that probability or its complement (1 - p) depending on which side the
    caller's team1 actually is. This guarantees P(A beats B) + P(B beats A) == 1 exactly,
    every time, rather than running the model twice in two slightly different directions."""
    t1 = TEAMS.get(str(team1))
    t2 = TEAMS.get(str(team2))
    if t1 is None:
        raise UnknownTeamError(f"team {team1} is not a 2026 tournament team")
    if t2 is None:
        raise UnknownTeamError(f"team {team2} is not a 2026 tournament team")
    if t1["gender"] != t2["gender"]:
        raise MismatchedGenderError(f"team {team1} ({t1['gender']}) and team {team2} "
                                     f"({t2['gender']}) are from different tournaments")

    lower_id, higher_id = sorted([team1, team2])
    p_lower_wins = _predict_canonical(str(lower_id), str(higher_id))
    return p_lower_wins if team1 == lower_id else 1 - p_lower_wins


if __name__ == "__main__":
    # a few sanity checks, run directly with: python3 predictor.py
    import itertools

    men_ids = [tid for tid, t in TEAMS.items() if t["gender"] == "M"][:5]
    print("sample men's teams:", [(tid, TEAMS[tid]["team_name"]) for tid in men_ids])

    t1, t2 = men_ids[0], men_ids[1]
    p = predict_matchup(int(t1), int(t2))
    print(f"\nP({TEAMS[t1]['team_name']} beats {TEAMS[t2]['team_name']}) = {p:.4f}")

    # symmetry check: predicting the reverse matchup should give (1 - p)
    p_rev = predict_matchup(int(t2), int(t1))
    print(f"P({TEAMS[t2]['team_name']} beats {TEAMS[t1]['team_name']}) = {p_rev:.4f}  "
          f"(should be close to {1 - p:.4f})")

    # error-handling checks
    try:
        predict_matchup(999999, int(t2))
    except UnknownTeamError as e:
        print(f"\nunknown-team check passed: {e}")

    try:
        w_id = [tid for tid, t in TEAMS.items() if t["gender"] == "W"][0]
        predict_matchup(int(t1), int(w_id))
    except MismatchedGenderError as e:
        print(f"mismatched-gender check passed: {e}")
