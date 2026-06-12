from __future__ import annotations

import csv
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from worldcup_predictor.predictor import MatchPredictor

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"
PREDICTION_LOG_PATH = ROOT / "artifacts" / "prediction_log.tsv"
STATIC_DIR = ROOT / "static"
PREDICTION_LOG_FIELDS = [
    "generated_at",
    "model_name",
    "model_latest_match_date",
    "team_a",
    "team_b",
    "match_date",
    "tournament",
    "neutral",
    "team_a_win_probability",
    "draw_probability",
    "team_b_win_probability",
    "predicted_outcome",
    "predicted_winner",
    "actual_team_a_score",
    "actual_team_b_score",
    "actual_outcome",
]


class PredictionRequest(BaseModel):
    team_a: str = Field(min_length=1)
    team_b: str = Field(min_length=1)
    neutral: bool = True
    match_date: date = Field(default_factory=date.today)
    tournament: str = "FIFA World Cup"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="World Cup Match Predictor", version="0.1.0", lifespan=lifespan)
predictor: MatchPredictor | None = None


def load_model() -> None:
    global predictor
    if ARTIFACT_PATH.exists():
        predictor = MatchPredictor.from_path(ARTIFACT_PATH)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/api/teams")
def teams() -> dict[str, list[str]]:
    require_predictor()
    return {"teams": predictor.teams}  # type: ignore[union-attr]


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    service = require_predictor()
    try:
        result = service.predict(
            team_a=request.team_a.strip(),
            team_b=request.team_b.strip(),
            neutral=request.neutral,
            match_date=request.match_date.isoformat(),
            tournament=request.tournament.strip() or "FIFA World Cup",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_prediction(request, result, service)
    return result.__dict__


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def require_predictor() -> MatchPredictor:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is missing. Run `python scripts/train_model.py` first.",
        )
    return predictor


def log_prediction(request: PredictionRequest, result, service: MatchPredictor) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PREDICTION_LOG_PATH.exists()
    row = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": service.artifact.get("model_name", ""),
        "model_latest_match_date": service.artifact.get("latest_match_date", ""),
        "team_a": request.team_a.strip(),
        "team_b": request.team_b.strip(),
        "match_date": request.match_date.isoformat(),
        "tournament": request.tournament.strip() or "FIFA World Cup",
        "neutral": str(request.neutral),
        "team_a_win_probability": f"{result.team_a_win_probability:.12f}",
        "draw_probability": f"{result.draw_probability:.12f}",
        "team_b_win_probability": f"{result.team_b_win_probability:.12f}",
        "predicted_outcome": result.predicted_outcome,
        "predicted_winner": result.predicted_winner,
        "actual_team_a_score": "",
        "actual_team_b_score": "",
        "actual_outcome": "",
    }
    with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_LOG_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
