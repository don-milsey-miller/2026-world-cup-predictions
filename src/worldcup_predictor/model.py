"""Model training, selection, persistence, and probability helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from worldcup_predictor.data import team_names
from worldcup_predictor.evaluation import evaluate_probabilities
from worldcup_predictor.features import FEATURE_COLUMNS, build_training_frame

LABELS = ["draw", "team_a_win", "team_b_win"]
SELECTION_METRIC = "log_loss"


def train_and_select(matches: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train candidate models and select the best validation log-loss model."""
    feature_frame, states = build_training_frame(matches)
    if len(feature_frame) < 50:
        raise ValueError("Need at least 50 matches to train a useful model.")

    cutoff = feature_frame["date"].quantile(0.8)
    train = feature_frame[feature_frame["date"] <= cutoff]
    valid = feature_frame[feature_frame["date"] > cutoff]
    if valid.empty:
        raise ValueError("Validation split is empty.")

    candidates = candidate_models()
    metrics: dict[str, Any] = {}
    best_name = ""
    best_model: Any = None
    best_score = float("inf")

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    x_valid = valid[FEATURE_COLUMNS]
    y_valid = valid["target"]

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        probabilities = _aligned_probabilities(model, x_valid)
        model_metrics = evaluate_probabilities(y_valid, probabilities, LABELS)
        metrics[name] = model_metrics
        if model_metrics[SELECTION_METRIC] < best_score:
            best_name = name
            best_model = model
            best_score = model_metrics[SELECTION_METRIC]

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "states": states,
        "teams": team_names(matches),
        "latest_match_date": str(matches["date"].max().date()),
    }
    metrics["selected_model"] = best_name
    metrics["selection_metric"] = SELECTION_METRIC
    metrics["training_rows"] = len(train)
    metrics["validation_rows"] = len(valid)
    metrics["train_start_date"] = str(train["date"].min().date())
    metrics["train_end_date"] = str(train["date"].max().date())
    metrics["validation_start_date"] = str(valid["date"].min().date())
    metrics["validation_end_date"] = str(valid["date"].max().date())
    metrics["latest_match_date"] = artifact["latest_match_date"]
    return artifact, metrics


def candidate_models() -> dict[str, Any]:
    """Return the standard candidate model set for training."""
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    calibrated_logistic = CalibratedClassifierCV(
        estimator=Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    gradient_boosting = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.045,
        l2_regularization=0.02,
        random_state=42,
    )
    calibrated_gradient_boosting = CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=0.02,
            random_state=42,
        ),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    return {
        "logistic_regression": logistic,
        "calibrated_logistic_regression": calibrated_logistic,
        "hist_gradient_boosting": gradient_boosting,
        "calibrated_hist_gradient_boosting": calibrated_gradient_boosting,
    }


def save_artifact(artifact: dict[str, Any], path: Path) -> None:
    """Serialize a trained model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> dict[str, Any]:
    """Load a serialized model artifact."""
    return joblib.load(path)


def predict_probabilities(model: Any, features: pd.DataFrame) -> dict[str, float]:
    """Predict probabilities aligned to the application label order."""
    probabilities = _aligned_probabilities(model, features)[0]
    return {label: float(probabilities[index]) for index, label in enumerate(LABELS)}


def _aligned_probabilities(model: Any, features: pd.DataFrame):
    raw = model.predict_proba(features)
    classes = list(model.classes_)
    aligned = []
    for label in LABELS:
        aligned.append(raw[:, classes.index(label)] if label in classes else 0.0)
    return pd.DataFrame(
        {label: values for label, values in zip(LABELS, aligned, strict=True)}
    ).to_numpy()
