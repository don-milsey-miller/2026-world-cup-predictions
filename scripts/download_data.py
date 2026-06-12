from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup_predictor.data import download_data


if __name__ == "__main__":
    download_data(Path("data/raw"))
    print("Downloaded results.csv and shootouts.csv to data/raw")
