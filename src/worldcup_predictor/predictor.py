"""Runtime prediction service backed by a trained model artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from worldcup_predictor.features import FEATURE_COLUMNS, matchup_features
from worldcup_predictor.model import load_artifact, predict_probabilities


@dataclass
class PredictionResult:
    """Prediction probabilities and explanatory signals."""

    team_a_win_probability: float
    draw_probability: float
    team_b_win_probability: float
    predicted_outcome: str
    predicted_winner: str
    top_feature_signals: list[dict[str, float | str]]


class MatchPredictor:
    """Generate predictions from a persisted model artifact."""

    def __init__(self, artifact: dict):
        """Initialize the predictor from an artifact dictionary."""
        self.artifact = artifact
        self.model = artifact["model"]
        self.teams = artifact["teams"]
        self.states = artifact["states"]

    @classmethod
    def from_path(cls, path: Path) -> MatchPredictor:
        """Load a predictor from a serialized artifact path."""
        return cls(load_artifact(path))

    def predict(
        self,
        *,
        team_a: str,
        team_b: str,
        neutral: bool,
        match_date: str,
        tournament: str,
    ) -> PredictionResult:
        """Predict match outcome probabilities for two teams."""
        self._validate_teams(team_a, team_b)
        features = matchup_features(
            team_a=team_a,
            team_b=team_b,
            match_date=match_date,
            tournament=tournament,
            neutral=neutral,
            home_team=None if neutral else team_a,
            states=self.states,
        )
        frame = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
        probabilities = predict_probabilities(self.model, frame)
        outcome = max(probabilities, key=probabilities.get)
        winner = team_a if probabilities["team_a_win"] >= probabilities["team_b_win"] else team_b
        return PredictionResult(
            team_a_win_probability=probabilities["team_a_win"],
            draw_probability=probabilities["draw"],
            team_b_win_probability=probabilities["team_b_win"],
            predicted_outcome=outcome,
            predicted_winner=winner,
            top_feature_signals=_top_signals(features),
        )

    def _validate_teams(self, team_a: str, team_b: str) -> None:
        known = set(self.teams)
        if team_a == team_b:
            raise ValueError("Choose two different teams.")
        unknown = sorted(team for team in [team_a, team_b] if team not in known)
        if unknown:
            raise ValueError(f"Unknown team(s): {', '.join(unknown)}")


def _top_signals(features: dict[str, float | int]) -> list[dict[str, float | str]]:
    names = {
        "elo_diff": "Elo rating gap",
        "form_points_diff": "Recent form gap",
        "goals_for_diff": "Recent scoring gap",
        "goals_against_diff": "Recent defensive gap",
        "days_rest_diff": "Rest gap",
        "importance": "Tournament importance",
    }
    ranked = sorted(names, key=lambda key: abs(float(features[key])), reverse=True)
    return [{"name": names[key], "value": round(float(features[key]), 3)} for key in ranked[:4]]
