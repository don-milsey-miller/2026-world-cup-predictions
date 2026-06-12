from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup_predictor.data import load_results
from worldcup_predictor.model import save_artifact, train_and_select

RESULTS_PATH = Path("results.tsv")


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "manual"
    description = " ".join(sys.argv[2:]).strip() or "train current model candidates"
    started = time.perf_counter()
    status = "keep"
    metrics: dict = {}
    try:
        matches = load_results(Path("data/raw/results.csv"))
        artifact, metrics = train_and_select(matches)
        save_artifact(artifact, Path("artifacts/model.joblib"))
        Path("artifacts/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception as exc:
        status = "crash"
        metrics = {"error": str(exc), "selected_model": "", "validation_rows": 0}
        raise
    finally:
        elapsed = time.perf_counter() - started
        append_result(tag, description, status, metrics, elapsed)


def append_result(
    tag: str, description: str, status: str, metrics: dict, elapsed_seconds: float
) -> None:
    commit = current_commit()
    selected = metrics.get("selected_model", "")
    selected_metrics = metrics.get(selected, {}) if selected else {}
    row = {
        "commit": commit,
        "tag": tag,
        "log_loss": f"{selected_metrics.get('log_loss', 0.0):.6f}",
        "accuracy": f"{selected_metrics.get('accuracy', 0.0):.6f}",
        "brier": f"{selected_metrics.get('brier_multiclass', 0.0):.6f}",
        "ece": f"{selected_metrics.get('expected_calibration_error', 0.0):.6f}",
        "seconds": f"{elapsed_seconds:.1f}",
        "status": status,
        "selected_model": selected,
        "description": description.replace("\t", " "),
    }
    write_header = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
