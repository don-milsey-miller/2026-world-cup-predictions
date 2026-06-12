import pandas as pd

from worldcup_predictor.features import BASE_ELO, build_training_frame


def test_features_use_pre_match_elo():
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            },
            {
                "date": pd.Timestamp("2020-02-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 0,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": True,
            },
        ]
    )
    frame, _ = build_training_frame(matches)
    assert frame.iloc[0]["elo_a"] == BASE_ELO
    assert frame.iloc[0]["elo_b"] == BASE_ELO
    assert frame.iloc[1]["elo_a"] > BASE_ELO
    assert frame.iloc[1]["elo_b"] < BASE_ELO
