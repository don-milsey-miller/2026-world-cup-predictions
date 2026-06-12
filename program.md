# World Cup Predictor Research Protocol

This project borrows the useful discipline from `karpathy/autoresearch`: fixed data, fixed evaluation, compact experiment logs, and a bias toward simple changes that improve the primary metric.

## In-Scope Files

- `src/worldcup_predictor/features.py` contains feature engineering and Elo state updates.
- `src/worldcup_predictor/model.py` contains candidate models and model selection.
- `src/worldcup_predictor/evaluation.py` contains the fixed evaluation metrics. Treat this as the ground truth harness; change it only when deliberately changing the objective.

## Objective

Optimize validation `log_loss` first. Use `brier_multiclass`, expected calibration error, and accuracy as secondary diagnostics. Lower log loss is better because the app exposes probabilities, not only hard winners.

## Experiment Loop

1. Start from a clean git state.
2. Make one small model or feature change.
3. Run `python scripts/run_experiment.py <tag> "<short description>"`.
4. Inspect `artifacts/metrics.json` and the appended `results.tsv` row.
5. Keep changes only when validation log loss improves enough to justify the added complexity.
6. Prefer simplification when metrics are equal.

`results.tsv` is intentionally ignored by git. It is local lab notebook state, not application code.
