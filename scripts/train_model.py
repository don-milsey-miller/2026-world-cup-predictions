from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup_predictor.data import load_results
from worldcup_predictor.model import save_artifact, train_and_select


def main() -> None:
    data_path = Path("data/raw/results.csv")
    if not data_path.exists():
        raise SystemExit("Missing data/raw/results.csv. Run `python scripts/download_data.py` first.")
    matches = load_results(data_path)
    artifact, metrics = train_and_select(matches)
    save_artifact(artifact, Path("artifacts/model.joblib"))
    Path("artifacts/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
