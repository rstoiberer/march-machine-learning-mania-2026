"""Builds a Kaggle submission file by calling LEV's live /predict endpoint (his own deployment
of his own model, for the same March Machine Learning Mania 2026 competition).

This is a variant of generate_submission_via_api.py, adapted for a THIRD PARTY's server instead
of my own. That distinction drives every difference below:

  - My own server: I trust its behavior completely, so the original script fails loudly (crashes)
    on anything unexpected, since an unexpected response there would mean *my own* code has a bug.
  - Lev's server: I don't control it and haven't tested it, so this version is defensive instead --
    retries on transient network hiccups, matches responses back to requests by content (team1/
    team2) rather than assuming his server preserves request order, and tolerates real per-row
    errors by recording them instead of crashing the whole run.

One thing that does NOT change: the universe of "real" 2026 matchups (4,556 of them, both teams
in the 136-team tournament field). That's public competition data -- the same for every model,
mine or his -- not something specific to whoever is answering the predictions.

One thing this script deliberately does NOT do: compare Lev's output against submission_v2.csv.
That comparison only makes sense for MY OWN model (proving the API matches the notebook). Lev's
model presumably has different weights/features, so his predictions are *expected* to differ from
mine -- a mismatch here would be normal, not a bug. Instead this script reports basic sanity
diagnostics: how many matchups got a real prediction vs. an error, and whether the predictions
look like valid probabilities.

Run this on a machine with real internet access. Requires: pip install requests

Usage:
    python3 generate_submission_via_lev_api.py
"""
import csv
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEV_ENDPOINT = "<lev's-endpoint-url>/predict"  # TODO: fill in Lev's actual Railway (or other) URL
SEASON = 2026  # hardcoded -- Lev confirmed his API is for the 2026 field only, same as mine
BATCH_SIZE = 128  # per Lev's confirmed limit, matching my own server's cap

SAMPLE_SUBMISSION = Path("data/SampleSubmissionStage2.csv")
MODEL_EXPORT = Path("service/model_export.json")  # only used to know which team IDs are "real" --
                                                   # this is the shared competition field, not my
                                                   # model's opinion, so it's valid to reuse here
OUTPUT_PATH = Path("submissions/submission_via_lev_api.csv")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s


def load_valid_team_ids() -> set[int]:
    """The 136 team IDs (68 men's + 68 women's) the 2026 field actually contains."""
    with open(MODEL_EXPORT) as f:
        model = json.load(f)
    return {int(tid) for tid in model["teams"].keys()}


def load_submission_rows() -> list[dict]:
    """Every row Kaggle wants a Pred for: 132,133 of them, format `2026_TeamLow_TeamHigh`."""
    with open(SAMPLE_SUBMISSION) as f:
        rows = list(csv.DictReader(f))
    # sanity check on the SEASON assumption, rather than silently trusting it
    seasons = {row["ID"].split("_")[0] for row in rows}
    if seasons != {str(SEASON)}:
        raise ValueError(f"expected every row to be season {SEASON}, found seasons: {seasons}")
    return rows


def chunked(seq: list, size: int):
    """Yields successive `size`-length slices of `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def call_endpoint_with_retries(payload: dict) -> dict:
    """POSTs to Lev's endpoint, retrying a few times on network/timeout errors before giving up --
    a third-party server (especially a free-tier deployment) is more likely to have a transient
    hiccup than my own, so a single failed request shouldn't kill the whole run."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(LEV_ENDPOINT, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"    attempt {attempt} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts: {last_error}")


def main():
    valid_teams = load_valid_team_ids()
    print(f"loaded {len(valid_teams)} valid {SEASON} team IDs")

    rows = load_submission_rows()
    print(f"loaded {len(rows)} total submission rows")

    real_rows = []
    for row in rows:
        _, t1, t2 = row["ID"].split("_")
        t1, t2 = int(t1), int(t2)
        if t1 in valid_teams and t2 in valid_teams:
            real_rows.append((row["ID"], t1, t2))
    print(f"{len(real_rows)} rows are real matchups Lev's API should be able to predict")

    # --- Call Lev's endpoint in batches of 128 ------------------------------------------------
    predictions: dict[str, float] = {}   # ID -> Pred, only for rows that came back clean
    row_errors: dict[str, str] = {}      # ID -> error message, for rows Lev's API rejected
    batches = list(chunked(real_rows, BATCH_SIZE))
    print(f"calling Lev's API in {len(batches)} batches of up to {BATCH_SIZE}...")

    for i, batch in enumerate(batches, start=1):
        payload = {
            "matchups": [{"team1": t1, "team2": t2} for _, t1, t2 in batch]
        }
        data = call_endpoint_with_retries(payload)

        if "predictions" not in data:
            raise RuntimeError(f"batch {i}: response missing 'predictions' key: {data}")
        results = data["predictions"]

        if len(results) != len(batch):
            print(f"    WARNING: batch {i} sent {len(batch)} matchups but got back "
                  f"{len(results)} results -- matching by (team1, team2) instead of position")

        # Match responses back to request IDs by CONTENT (team1, team2), not by list position.
        # My own server is documented to preserve order, but I haven't verified that for Lev's,
        # so this doesn't assume it -- safer given the format was shared secondhand, not tested.
        id_by_pair = {(t1, t2): id_ for id_, t1, t2 in batch}
        for result in results:
            pair = (result.get("team1"), result.get("team2"))
            id_ = id_by_pair.get(pair)
            if id_ is None:
                print(f"    WARNING: batch {i} returned a result for {pair}, "
                      f"which wasn't in the request -- skipping")
                continue
            if result.get("error"):
                row_errors[id_] = result["error"]
            elif result.get("prediction") is not None:
                predictions[id_] = result["prediction"]
            else:
                row_errors[id_] = "no prediction and no error in response"

        print(f"  batch {i}/{len(batches)} done ({len(batch)} matchups)")
        time.sleep(0.1)  # small pause so we're not hammering someone else's free-tier server

    # --- Reassemble the full 132,133-row submission file ----------------------------------------
    for row in rows:
        if row["ID"] in predictions:
            row["Pred"] = predictions[row["ID"]]
        # rows with an error, or rows outside the real-matchup universe, keep the sample's
        # default Pred (0.5) -- same convention as the notebook and my own script

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Pred"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUTPUT_PATH} ({len(rows)} rows)")

    # --- Sanity diagnostics (NOT a comparison to submission_v2.csv -- see module docstring) -----
    print(f"\n--- diagnostics ---")
    print(f"real matchups attempted:     {len(real_rows)}")
    print(f"got a real prediction:       {len(predictions)}")
    print(f"came back as an error:       {len(row_errors)}")
    if row_errors:
        print("sample errors:")
        for id_, msg in list(row_errors.items())[:5]:
            print(f"    {id_}: {msg}")

    if predictions:
        values = list(predictions.values())
        print(f"prediction range:            {min(values):.4f} to {max(values):.4f}")
        out_of_bounds = [v for v in values if not (0.0 <= v <= 1.0)]
        if out_of_bounds:
            print(f"  WARNING: {len(out_of_bounds)} predictions fall outside [0, 1] -- "
                  f"that would indicate a bug in Lev's model or response format")
        else:
            print("  all predictions are valid probabilities in [0, 1]")


if __name__ == "__main__":
    main()
