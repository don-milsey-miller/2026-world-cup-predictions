from __future__ import annotations

import csv
from html import escape
from pathlib import Path

RESULTS_PATH = Path("results.tsv")
AUTORESEARCH_PATH = Path("artifacts/autoresearch.tsv")
HISTORY_PATH = Path("artifacts/training_history.tsv")
CHART_PATH = Path("artifacts/training_progress.svg")


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("No training results found in results.tsv or artifacts/autoresearch.tsv.")
    write_history(runs)
    write_chart(runs)
    print(f"Wrote {HISTORY_PATH}")
    print(f"Wrote {CHART_PATH}")


def load_runs() -> list[dict[str, str | float | int]]:
    runs: list[dict[str, str | float | int]] = []
    sequence = 1
    best = float("inf")

    for row in read_tsv(RESULTS_PATH):
        log_loss = parse_float(row.get("log_loss", ""))
        if log_loss is None:
            continue
        best = min(best, log_loss)
        runs.append(
            {
                "sequence": sequence,
                "source": "experiment",
                "cycle": "",
                "name": row.get("tag", ""),
                "log_loss": log_loss,
                "cumulative_best_log_loss": best,
                "accuracy": row.get("accuracy", ""),
                "brier": row.get("brier", ""),
                "ece": row.get("ece", ""),
                "seconds": row.get("seconds", ""),
                "status": row.get("status", ""),
                "description": row.get("description", ""),
            }
        )
        sequence += 1

    for row in read_tsv(AUTORESEARCH_PATH):
        log_loss = parse_float(row.get("log_loss", ""))
        if log_loss is None:
            continue
        best = min(best, log_loss)
        runs.append(
            {
                "sequence": sequence,
                "source": "autoresearch",
                "cycle": row.get("cycle", ""),
                "name": row.get("candidate", ""),
                "log_loss": log_loss,
                "cumulative_best_log_loss": best,
                "accuracy": row.get("accuracy", ""),
                "brier": row.get("brier", ""),
                "ece": row.get("ece", ""),
                "seconds": row.get("seconds", ""),
                "status": row.get("status", ""),
                "description": row.get("description", ""),
            }
        )
        sequence += 1

    return runs


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def write_history(runs: list[dict[str, str | float | int]]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "source",
        "cycle",
        "name",
        "log_loss",
        "cumulative_best_log_loss",
        "accuracy",
        "brier",
        "ece",
        "seconds",
        "status",
        "description",
    ]
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    **run,
                    "log_loss": f"{float(run['log_loss']):.6f}",
                    "cumulative_best_log_loss": f"{float(run['cumulative_best_log_loss']):.6f}",
                }
            )


def write_chart(runs: list[dict[str, str | float | int]]) -> None:
    width = 1200
    height = 640
    left = 92
    right = 42
    top = 82
    bottom = 112
    plot_width = width - left - right
    plot_height = height - top - bottom
    losses = [float(run["log_loss"]) for run in runs]
    bests = [float(run["cumulative_best_log_loss"]) for run in runs]
    low = min(losses + bests)
    high = max(losses + bests)
    if high == low:
        high += 0.001
        low -= 0.001
    padding = (high - low) * 0.12
    y_min = low - padding
    y_max = high + padding

    points = [
        (x_for(i + 1, len(runs), left, plot_width), y_for(loss, y_min, y_max, top, plot_height))
        for i, loss in enumerate(losses)
    ]
    best_points = [
        (x_for(i + 1, len(runs), left, plot_width), y_for(best, y_min, y_max, top, plot_height))
        for i, best in enumerate(bests)
    ]

    y_ticks = [y_min + (y_max - y_min) * step / 4 for step in range(5)]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Training Log Loss Over Time</title>',
        '<desc id="desc">Per-run validation log loss and cumulative best validation log loss for experiment and auto-research runs.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1f2933">Training Log Loss Over Time</text>',
        f'<text x="{width / 2}" y="62" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#52616f">{len(runs)} saved training runs from results.tsv and artifacts/autoresearch.tsv</text>',
    ]

    for tick in y_ticks:
        y = y_for(tick, y_min, y_max, top, plot_height)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d6dce2" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#52616f">{tick:.4f}</text>'
        )

    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#4b5563" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#4b5563" stroke-width="1.5"/>',
            f'<text x="{width / 2}" y="{height - 42}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">Saved training run sequence</text>',
            f'<text x="28" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 28 {top + plot_height / 2})" font-family="Arial, sans-serif" font-size="13" fill="#374151">Validation log loss, lower is better</text>',
        ]
    )

    if len(runs) > 1:
        lines.append(polyline(points, "#8aa1b4", 2, "none"))
        lines.append(polyline(best_points, "#16a34a", 3, "none"))

    for point, run in zip(points, runs, strict=True):
        color = "#2563eb" if run["source"] == "experiment" else "#8aa1b4"
        if run["status"] in {"promote", "would_promote"}:
            color = "#16a34a"
        lines.append(
            f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="4" fill="{color}"><title>{tooltip(run)}</title></circle>'
        )

    last_best = min(runs, key=lambda run: float(run["cumulative_best_log_loss"]))
    best_value = float(runs[-1]["cumulative_best_log_loss"])
    lines.extend(
        [
            f'<rect x="{left}" y="{height - 92}" width="18" height="4" fill="#8aa1b4"/>',
            f'<text x="{left + 28}" y="{height - 86}" font-family="Arial, sans-serif" font-size="12" fill="#374151">run log loss</text>',
            f'<rect x="{left + 150}" y="{height - 92}" width="18" height="4" fill="#16a34a"/>',
            f'<text x="{left + 178}" y="{height - 86}" font-family="Arial, sans-serif" font-size="12" fill="#374151">cumulative best</text>',
            f'<text x="{width - right}" y="{height - 86}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#166534">Best so far: {best_value:.6f} ({escape(str(last_best["name"]))})</text>',
            "</svg>",
        ]
    )
    CHART_PATH.write_text("\n".join(lines), encoding="utf-8")


def x_for(sequence: int, count: int, left: int, plot_width: int) -> float:
    if count == 1:
        return left + plot_width / 2
    return left + (sequence - 1) * plot_width / (count - 1)


def y_for(value: float, y_min: float, y_max: float, top: int, plot_height: int) -> float:
    return top + (y_max - value) * plot_height / (y_max - y_min)


def polyline(points: list[tuple[float, float]], color: str, width: int, fill: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def tooltip(run: dict[str, str | float | int]) -> str:
    return escape(
        f"#{run['sequence']} {run['name']} | log_loss={float(run['log_loss']):.6f} | "
        f"best={float(run['cumulative_best_log_loss']):.6f} | status={run['status']}"
    )


if __name__ == "__main__":
    main()
