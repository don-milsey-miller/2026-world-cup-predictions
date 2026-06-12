"""Evaluation metrics for multiclass probability predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate_probabilities(
    y_true: pd.Series, probabilities, labels: Sequence[str]
) -> dict[str, float]:
    """Evaluate predicted class probabilities against true labels."""
    predictions = [labels[index] for index in probabilities.argmax(axis=1)]
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=list(labels))),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "brier_multiclass": float(multiclass_brier(y_true, probabilities, labels)),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "expected_calibration_error": float(
            expected_calibration_error(y_true, probabilities, labels)
        ),
    }


def multiclass_brier(y_true: pd.Series, probabilities, labels: Sequence[str]) -> float:
    """Calculate the mean one-vs-rest Brier score across classes."""
    total = 0.0
    for index, label in enumerate(labels):
        binary_true = (y_true == label).astype(int)
        total += brier_score_loss(binary_true, probabilities[:, index])
    return total / len(labels)


def expected_calibration_error(
    y_true: pd.Series,
    probabilities,
    labels: Sequence[str],
    *,
    bins: int = 10,
) -> float:
    """Calculate confidence calibration error over equal-width bins."""
    confidences = probabilities.max(axis=1)
    predictions = [labels[index] for index in probabilities.argmax(axis=1)]
    correct = pd.Series(predictions, index=y_true.index) == y_true
    total = len(y_true)
    if total == 0:
        return math.nan

    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_accuracy = float(correct[in_bin].mean())
        bin_confidence = float(confidences[in_bin].mean())
        ece += (count / total) * abs(bin_accuracy - bin_confidence)
    return ece
