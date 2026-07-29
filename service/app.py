"""FastAPI web layer. This file only handles receiving requests and sending responses --
the actual prediction math lives in predictor.py and is untouched here."""
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predictor import predict_matchup, UnknownTeamError, MismatchedGenderError

app = FastAPI(title="March Mania Prediction Service")


# ---------------------------------------------------------------------------
# Request / response shapes ("data contracts"). FastAPI uses these to validate
# incoming JSON automatically, before our own code ever runs.
# ---------------------------------------------------------------------------

class Matchup(BaseModel):
    team1: int
    team2: int


class PredictRequest(BaseModel):
    # max_length=128 means FastAPI will automatically reject any request with more than
    # 128 matchups in it, exactly matching the batch size your supervisors suggested.
    matchups: List[Matchup] = Field(..., max_length=128)


class PredictionResult(BaseModel):
    team1: int
    team2: int
    prediction: Optional[float] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """A simple GET endpoint with no input needed -- just confirms the service is alive.
    Useful for Railway itself, and for a quick manual check that things are running."""
    return {"status": "ok", "service": "march-mania-prediction", "season": 2026}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    results = []
    for m in request.matchups:
        try:
            p = predict_matchup(m.team1, m.team2)
            results.append(PredictionResult(team1=m.team1, team2=m.team2, prediction=p))
        except (UnknownTeamError, MismatchedGenderError) as e:
            # one bad matchup in a batch of 128 shouldn't fail the other 127 --
            # record the error for this one row and keep going
            results.append(PredictionResult(team1=m.team1, team2=m.team2, error=str(e)))
    return PredictResponse(predictions=results)
