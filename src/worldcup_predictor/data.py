"""Data download and cleaning utilities for international results."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)

REQUIRED_RESULT_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}


def download_data(raw_dir: Path) -> None:
    """Download raw match and shootout data into a directory."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(RESULTS_URL, raw_dir / "results.csv")
    urlretrieve(SHOOTOUTS_URL, raw_dir / "shootouts.csv")


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate completed match results from a CSV file."""
    frame = pd.read_csv(path)
    missing = REQUIRED_RESULT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {sorted(missing)}")
    return clean_results(frame)


def clean_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw results and drop rows that cannot train the model."""
    cleaned = frame.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["home_score"] = pd.to_numeric(cleaned["home_score"], errors="coerce")
    cleaned["away_score"] = pd.to_numeric(cleaned["away_score"], errors="coerce")
    cleaned["neutral"] = cleaned["neutral"].map(_coerce_bool)
    cleaned["tournament"] = cleaned["tournament"].fillna("Unknown").astype(str).str.strip()
    cleaned["home_team"] = cleaned["home_team"].astype(str).str.strip()
    cleaned["away_team"] = cleaned["away_team"].astype(str).str.strip()
    cleaned = cleaned.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
    cleaned = cleaned[cleaned["home_team"] != cleaned["away_team"]]
    cleaned = cleaned.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    )
    cleaned = cleaned.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    cleaned["home_score"] = cleaned["home_score"].astype(int)
    cleaned["away_score"] = cleaned["away_score"].astype(int)
    return cleaned


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def team_names(frame: pd.DataFrame) -> list[str]:
    """Return sorted unique team names from match results."""
    teams = set(frame["home_team"].dropna()) | set(frame["away_team"].dropna())
    return sorted(str(team) for team in teams)
