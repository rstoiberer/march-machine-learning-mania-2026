"""Builds a Kaggle submission file by calling the LIVE /predict endpoint instead of running the
model locally in a notebook. Goal: prove the deployed API reproduces submissions/submission_v2.csv
exactly (same model, same weights, same features -- just reached over HTTP instead of in-process).

Run this on your own machine (not in a sandboxed environment), since it needs real internet
access to reach Railway. Requires the `requests` library: pip install requests

Usage:
    python3 generate_submission_via_api.py
"""
import csv
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENDPOINT = "https://march-machine-learning-mania-2026-production-36e2.up.railway.app/predict"
BATCH_SIZE = 128  # matches the server's max_length=128 limit on the `matchups` list

SAMPLE_SUBMISSION = Path("data/SampleSubmissionStage2.csv")
MODEL_EXPORT = Path("service/model_export.json")  # only used to know which team IDs are "real"
OUTPUT_PATH = Path("submissions/submission_via_api.csv")
COMPARE_AGAINST = Path("submissions/submission_v2.csv")  # the file already submitted, rank 482


def load_valid_team_ids() -> set[int]:
    """The 136 team IDs (68 men's + 68 women's) the 2026 field actually contains. Any submission
    row that references a team NOT in this set can't get a real prediction -- neither the notebook
    nor the API can say anything about a team with no seed/stats/Elo on record."""
    with open(MODEL_EXPORT) as f:
        model = json.load(f)
    return {int(tid) for tid in model["teams"].keys()}


def load_submission_rows() -> list[dict]:
    """Every row Kaggle wants a Pred for: 132,133 of them, format `2026_TeamLow_TeamHigh`."""
    with open(SAMPLE_SUBMISSION) as f:
        return list(csv.DictReader(f))


def chunked(seq: list, size: int):
    """Yields successive `size`-length slices of `seq`. Plain Python, no library needed --
    this is the same idea as itertools.batched (Python 3.12+) but works on older versions too."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    valid_teams = load_valid_team_ids()
    print(f"loaded {len(valid_teams)} valid 2026 team IDs")

    rows = load_submission_rows()
    print(f"loaded {len(rows)} total submission rows")

    # Split each row's ID ("2026_1101_1102") into (Season, Team1, Team2), and only keep the ones
    # where BOTH teams are real 2026 tournament teams -- those are the only rows the API can
    # answer. Everything else stays at the sample submission's default Pred (0.5).
    real_rows = []
    for row in rows:
        _, t1, t2 = row["ID"].split("_")
        t1, t2 = int(t1), int(t2)
        if t1 in valid_teams and t2 in valid_teams:
            real_rows.append((row["ID"], t1, t2))
    print(f"{len(real_rows)} rows are real matchups the API can predict")

    # --- Call the live endpoint in batches of 128 ---------------------------------------------
    predictions: dict[str, float] = {}  # ID -> Pred
    batches = list(chunked(real_rows, BATCH_SIZE))
    print(f"calling the API in {len(batches)} batches of up to {BATCH_SIZE}...")

    for i, batch in enumerate(batches, start=1):
        payload = {
            "matchups": [{"team1": t1, "team2": t2} for _, t1, t2 in batch]
        }
        resp = requests.post(ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()  # crash loudly if the server errors, rather than silently continue
        results = resp.json()["predictions"]

        # Zip the batch's IDs back onto the results, in the same order we sent them -- the API
        # returns results in request order, so this is safe.
        for (id_, _, _), result in zip(batch, results):
            if result["error"]:
                # shouldn't happen since we pre-filtered to known, same-gender, distinct teams --
                # but fail loudly instead of silently writing a bad row if it ever does
                raise RuntimeError(f"unexpected error for {id_}: {result['error']}")
            predictions[id_] = result["prediction"]

        print(f"  batch {i}/{len(batches)} done ({len(batch)} matchups)")
        time.sleep(0.1)  # small pause so we're not hammering the free-tier server

    # --- Reassemble the full 132,133-row submission file ---------------------------------------
    for row in rows:
        if row["ID"] in predictions:
            row["Pred"] = predictions[row["ID"]]

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Pred"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT_PATH} ({len(rows)} rows)")

    # --- Compare against the file that was actually submitted (rank 482) -----------------------
    with open(COMPARE_AGAINST) as f:
        reference = {r["ID"]: float(r["Pred"]) for r in csv.DictReader(f)}

    diffs = []
    for row in rows:
        ref = reference.get(row["ID"])
        if ref is not None:
            diffs.append(abs(float(row["Pred"]) - ref))

    print(f"\ncompared {len(diffs)} rows against {COMPARE_AGAINST}")
    print(f"  max difference:  {max(diffs):.10f}")
    print(f"  mean difference: {sum(diffs) / len(diffs):.10f}")
    # Note: model_export.json stores weights truncated to 8 significant digits, not Python's full
    # float64 precision -- so expect tiny (~1e-6) differences on rows with large feature values
    # like EloDiff. That's export rounding, not a bug, and far too small to move a Brier score.
    if max(diffs) < 1e-4:
        print("  MATCH: the live API reproduces submission_v2.csv (differences are just")
        print("  floating-point export rounding, not a real discrepancy)")
    else:
        print("  MISMATCH: investigate before submitting")


if __name__ == "__main__":
    main()
