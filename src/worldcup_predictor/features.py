"""Chronological feature engineering for international match prediction."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

BASE_ELO = 1500.0
K_FACTOR = 28.0
FEATURE_COLUMNS = [
    "elo_a",
    "elo_b",
    "elo_diff",
    "form_points_diff",
    "goals_for_diff",
    "goals_against_diff",
    "days_rest_diff",
    "neutral",
    "team_a_home",
    "team_b_home",
    "importance",
]

HIGH_IMPORTANCE = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "Oceania Nations Cup",
}


@dataclass
class TeamState:
    """Mutable pre-match state tracked for each team."""

    elo: float = BASE_ELO
    recent_points: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    recent_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    recent_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    last_played: pd.Timestamp | None = None


def build_training_frame(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, TeamState]]:
    """Build model-ready rows while updating team state chronologically."""
    states: dict[str, TeamState] = defaultdict(TeamState)
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for match in matches.sort_values("date").itertuples(index=False):
        home = str(match.home_team)
        away = str(match.away_team)
        home_state = states[home]
        away_state = states[away]
        feature_values = matchup_features(
            team_a=home,
            team_b=away,
            match_date=match.date,
            tournament=str(match.tournament),
            neutral=bool(match.neutral),
            home_team=home,
            states=states,
        )
        target = result_label(int(match.home_score), int(match.away_score))
        rows.append(
            {
                "date": match.date,
                "team_a": home,
                "team_b": away,
                "target": target,
                **feature_values,
            }
        )
        update_states(
            home_state, away_state, int(match.home_score), int(match.away_score), match.date
        )

    return pd.DataFrame(rows), dict(states)


def matchup_features(
    *,
    team_a: str,
    team_b: str,
    match_date: str | date | pd.Timestamp,
    tournament: str,
    neutral: bool,
    home_team: str | None,
    states: dict[str, TeamState],
) -> dict[str, float | int]:
    """Create pre-match features for a team pairing."""
    when = pd.to_datetime(match_date)
    state_a = states.get(team_a, TeamState())
    state_b = states.get(team_b, TeamState())
    return {
        "elo_a": state_a.elo,
        "elo_b": state_b.elo,
        "elo_diff": state_a.elo - state_b.elo,
        "form_points_diff": _mean(state_a.recent_points, 1.0) - _mean(state_b.recent_points, 1.0),
        "goals_for_diff": _mean(state_a.recent_for, 1.0) - _mean(state_b.recent_for, 1.0),
        "goals_against_diff": _mean(state_a.recent_against, 1.0)
        - _mean(state_b.recent_against, 1.0),
        "days_rest_diff": _days_rest(state_a, when) - _days_rest(state_b, when),
        "neutral": int(neutral),
        "team_a_home": int(home_team == team_a and not neutral),
        "team_b_home": int(home_team == team_b and not neutral),
        "importance": tournament_importance(tournament),
    }


def update_states(
    home_state: TeamState,
    away_state: TeamState,
    home_score: int,
    away_score: int,
    match_date: pd.Timestamp,
) -> None:
    """Update Elo, form, goals, and rest state after a completed match."""
    home_result = score_result(home_score, away_score)
    away_result = 1.0 - home_result
    expected_home = expected_score(home_state.elo, away_state.elo)
    expected_away = 1.0 - expected_home
    home_state.elo += K_FACTOR * (home_result - expected_home)
    away_state.elo += K_FACTOR * (away_result - expected_away)
    home_points, away_points = points_for_match(home_score, away_score)
    home_state.recent_points.append(home_points)
    away_state.recent_points.append(away_points)
    home_state.recent_for.append(float(home_score))
    home_state.recent_against.append(float(away_score))
    away_state.recent_for.append(float(away_score))
    away_state.recent_against.append(float(home_score))
    home_state.last_played = pd.to_datetime(match_date)
    away_state.last_played = pd.to_datetime(match_date)


def result_label(team_a_score: int, team_b_score: int) -> str:
    """Return the multiclass label for a score line."""
    if team_a_score > team_b_score:
        return "team_a_win"
    if team_a_score < team_b_score:
        return "team_b_win"
    return "draw"


def score_result(team_a_score: int, team_b_score: int) -> float:
    """Return a numeric result from team A's perspective."""
    if team_a_score > team_b_score:
        return 1.0
    if team_a_score < team_b_score:
        return 0.0
    return 0.5


def points_for_match(team_a_score: int, team_b_score: int) -> tuple[float, float]:
    """Return table points awarded to both teams."""
    if team_a_score > team_b_score:
        return 3.0, 0.0
    if team_a_score < team_b_score:
        return 0.0, 3.0
    return 1.0, 1.0


def expected_score(elo_a: float, elo_b: float) -> float:
    """Return the Elo expected score for team A."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def tournament_importance(tournament: str) -> int:
    """Map a tournament name to a coarse importance score."""
    if tournament in HIGH_IMPORTANCE:
        return 3
    lowered = tournament.lower()
    if "qualification" in lowered or "cup" in lowered or "championship" in lowered:
        return 2
    if "friendly" in lowered:
        return 0
    return 1


def _mean(values: deque[float], default: float) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _days_rest(state: TeamState, when: pd.Timestamp) -> float:
    if state.last_played is None:
        return 30.0
    return float(max(0, min(365, (when - state.last_played).days)))
