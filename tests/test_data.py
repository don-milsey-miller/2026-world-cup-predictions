import pandas as pd

from worldcup_predictor.data import clean_results


def test_clean_results_drops_invalid_and_duplicates():
    raw = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": " A ",
                "away_team": "B",
                "home_score": "1",
                "away_score": "0",
                "tournament": None,
                "city": "X",
                "country": "Y",
                "neutral": "FALSE",
            },
            {
                "date": "2020-01-01",
                "home_team": " A ",
                "away_team": "B",
                "home_score": "1",
                "away_score": "0",
                "tournament": None,
                "city": "X",
                "country": "Y",
                "neutral": "FALSE",
            },
            {
                "date": "bad",
                "home_team": "C",
                "away_team": "D",
                "home_score": None,
                "away_score": "0",
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": "TRUE",
            },
        ]
    )
    cleaned = clean_results(raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["home_team"] == "A"
    assert not cleaned.iloc[0]["neutral"]
