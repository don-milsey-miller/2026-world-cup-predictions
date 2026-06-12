from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from worldcup_predictor import app as app_module


@pytest.fixture()
def client(monkeypatch, tmp_path):
    model = DummyClassifier(strategy="prior")
    model.fit([[0] * 11, [1] * 11, [2] * 11], ["team_a_win", "draw", "team_b_win"])
    artifact = {
        "model": model,
        "model_name": "dummy",
        "teams": ["Argentina", "Brazil"],
        "states": {},
        "feature_columns": [],
        "labels": ["team_a_win", "draw", "team_b_win"],
        "latest_match_date": "2026-06-11",
    }
    app_module.predictor = app_module.MatchPredictor(artifact)
    monkeypatch.setattr(app_module, "PREDICTION_LOG_PATH", tmp_path / "prediction_log.tsv")
    return TestClient(app_module.app)


def test_health_and_teams(client):
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/api/teams").json()["teams"] == ["Argentina", "Brazil"]


def test_predict_rejects_same_team(client):
    response = client.post(
        "/api/predict",
        json={"team_a": "Brazil", "team_b": "Brazil", "neutral": True, "match_date": "2026-06-11"},
    )
    assert response.status_code == 400


def test_predict_probabilities_sum_to_one(client):
    response = client.post(
        "/api/predict",
        json={"team_a": "Brazil", "team_b": "Argentina", "neutral": True, "match_date": "2026-06-11"},
    )
    assert response.status_code == 200
    data = response.json()
    total = data["team_a_win_probability"] + data["draw_probability"] + data["team_b_win_probability"]
    assert total == pytest.approx(1.0)


def test_predict_logs_request(client):
    response = client.post(
        "/api/predict",
        json={
            "team_a": "Brazil",
            "team_b": "Argentina",
            "neutral": False,
            "match_date": "2026-06-11",
            "tournament": "FIFA World Cup",
        },
    )

    assert response.status_code == 200
    log_lines = app_module.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 2
    header = log_lines[0].split("\t")
    row = dict(zip(header, log_lines[1].split("\t")))
    assert row["team_a"] == "Brazil"
    assert row["team_b"] == "Argentina"
    assert row["model_name"] == "dummy"
    assert row["model_latest_match_date"] == "2026-06-11"
    assert row["actual_outcome"] == ""
