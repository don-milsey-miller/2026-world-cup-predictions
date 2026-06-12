from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from worldcup_predictor.data import load_results, team_names
from worldcup_predictor.evaluation import evaluate_probabilities
from worldcup_predictor.features import FEATURE_COLUMNS, build_training_frame
from worldcup_predictor.model import LABELS, SELECTION_METRIC, _aligned_probabilities, save_artifact
from run_experiment import current_commit

SEARCH_LOG_PATH = Path("artifacts/autoresearch.tsv")
METRICS_PATH = Path("artifacts/metrics.json")
MODEL_PATH = Path("artifacts/model.joblib")


def main() -> None:
    args = parse_args()
    started_at = time.time()
    deadline = started_at + args.duration_minutes * 60

    matches = load_results(Path("data/raw/results.csv"))
    feature_frame, states = build_training_frame(matches)
    train, valid = chronological_split(feature_frame)
    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    x_valid = valid[FEATURE_COLUMNS]
    y_valid = valid["target"]
    teams = team_names(matches)
    best_score = load_current_score()

    print(f"Starting auto research for {args.duration_minutes} minutes.")
    print(f"Current best {SELECTION_METRIC}: {best_score:.6f}")

    cycle = 1
    candidate_index = 0
    candidates = candidate_factories()
    while time.time() < deadline and candidate_index < len(candidates):
        cycle_end = min(time.time() + args.interval_minutes * 60, deadline)
        attempts = 0
        print(f"Cycle {cycle} started.")

        while (
            time.time() < cycle_end
            and attempts < args.max_candidates_per_cycle
            and candidate_index < len(candidates)
        ):
            name, description, factory = candidates[candidate_index]
            candidate_index += 1
            attempts += 1
            run_started = time.perf_counter()
            status = "reject"
            metrics: dict[str, Any]
            try:
                model = factory()
                model.fit(x_train, y_train)
                probabilities = _aligned_probabilities(model, x_valid)
                model_metrics = evaluate_probabilities(y_valid, probabilities, LABELS)
                score = model_metrics[SELECTION_METRIC]
                metrics = {
                    name: model_metrics,
                    "selected_model": name,
                    "selection_metric": SELECTION_METRIC,
                    "training_rows": int(len(train)),
                    "validation_rows": int(len(valid)),
                    "train_start_date": str(train["date"].min().date()),
                    "train_end_date": str(train["date"].max().date()),
                    "validation_start_date": str(valid["date"].min().date()),
                    "validation_end_date": str(valid["date"].max().date()),
                    "latest_match_date": str(matches["date"].max().date()),
                }
                if score + args.min_improvement < best_score:
                    status = "would_promote" if args.no_promote else "promote"
                    best_score = score
                    if not args.no_promote:
                        artifact = {
                            "model": model,
                            "model_name": name,
                            "feature_columns": FEATURE_COLUMNS,
                            "labels": LABELS,
                            "states": states,
                            "teams": teams,
                            "latest_match_date": str(matches["date"].max().date()),
                        }
                        save_artifact(artifact, MODEL_PATH)
                        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                append_search_result(name, description, status, metrics, time.perf_counter() - run_started, cycle)
                print(f"{status}: {name} {score:.6f}")
            except Exception as exc:
                metrics = {"error": str(exc), "selected_model": name}
                append_search_result(name, description, "crash", metrics, time.perf_counter() - run_started, cycle)
                print(f"crash: {name}: {exc}")

        if time.time() < deadline and candidate_index < len(candidates):
            sleep_seconds = max(0, cycle_end - time.time())
            if sleep_seconds:
                print(f"Cycle {cycle} complete. Sleeping {sleep_seconds:.1f} seconds.")
                time.sleep(sleep_seconds)
        cycle += 1

    print(f"Auto research complete. Best observed {SELECTION_METRIC}: {best_score:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded local model-search cycles.")
    parser.add_argument("--duration-minutes", type=float, default=240.0)
    parser.add_argument("--interval-minutes", type=float, default=5.0)
    parser.add_argument("--max-candidates-per-cycle", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--no-promote", action="store_true", help="Log results without replacing artifacts.")
    return parser.parse_args()


def chronological_split(feature_frame):
    cutoff = feature_frame["date"].quantile(0.8)
    train = feature_frame[feature_frame["date"] <= cutoff]
    valid = feature_frame[feature_frame["date"] > cutoff]
    if valid.empty:
        raise ValueError("Validation split is empty.")
    return train, valid


def load_current_score() -> float:
    if not METRICS_PATH.exists():
        return math.inf
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selected = metrics.get("selected_model")
    if not selected:
        return math.inf
    return float(metrics[selected][SELECTION_METRIC])


def candidate_factories():
    candidates = []
    for max_iter in [80, 120, 160, 220, 300]:
        for learning_rate in [0.03, 0.045, 0.06, 0.08]:
            for l2 in [0.0, 0.02, 0.08]:
                name = f"hgb_iter{max_iter}_lr{learning_rate:g}_l2{l2:g}"
                description = f"HistGradientBoosting max_iter={max_iter} learning_rate={learning_rate} l2={l2}"
                candidates.append(
                    (
                        name,
                        description,
                        lambda max_iter=max_iter, learning_rate=learning_rate, l2=l2: HistGradientBoostingClassifier(
                            max_iter=max_iter,
                            learning_rate=learning_rate,
                            l2_regularization=l2,
                            random_state=42,
                        ),
                    )
                )

    for depth in [4, 6, 8, None]:
        for n_estimators in [200, 400]:
            name = f"extra_trees_{n_estimators}_depth{depth or 'none'}"
            description = f"ExtraTrees n_estimators={n_estimators} max_depth={depth}"
            candidates.append(
                (
                    name,
                    description,
                    lambda depth=depth, n_estimators=n_estimators: ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=depth,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                )
            )

    for depth in [5, 8, None]:
        name = f"random_forest_300_depth{depth or 'none'}"
        description = f"RandomForest n_estimators=300 max_depth={depth}"
        candidates.append(
            (
                name,
                description,
                lambda depth=depth: RandomForestClassifier(
                    n_estimators=300,
                    max_depth=depth,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            )
        )

    for c_value in [0.3, 0.7, 1.0, 1.5, 3.0]:
        name = f"logistic_c{c_value:g}"
        description = f"LogisticRegression C={c_value}"
        candidates.append(
            (
                name,
                description,
                lambda c_value=c_value: Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", C=c_value)),
                    ]
                ),
            )
        )

    candidates.append(
        (
            "calibrated_hgb_iter160_lr0045_l2002",
            "Calibrated current HistGradientBoosting baseline",
            lambda: CalibratedClassifierCV(
                estimator=HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.045,
                    l2_regularization=0.02,
                    random_state=42,
                ),
                method="sigmoid",
                cv=TimeSeriesSplit(n_splits=3),
            ),
        )
    )
    return candidates


def append_search_result(
    name: str,
    description: str,
    status: str,
    metrics: dict[str, Any],
    elapsed_seconds: float,
    cycle: int,
) -> None:
    selected = metrics.get("selected_model", name)
    selected_metrics = metrics.get(selected, {}) if selected else {}
    row = {
        "commit": current_commit(),
        "cycle": cycle,
        "candidate": name,
        "log_loss": f"{selected_metrics.get('log_loss', 0.0):.6f}",
        "accuracy": f"{selected_metrics.get('accuracy', 0.0):.6f}",
        "brier": f"{selected_metrics.get('brier_multiclass', 0.0):.6f}",
        "ece": f"{selected_metrics.get('expected_calibration_error', 0.0):.6f}",
        "seconds": f"{elapsed_seconds:.1f}",
        "status": status,
        "description": description.replace("\t", " "),
    }
    SEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SEARCH_LOG_PATH.exists()
    with SEARCH_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
