#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALL=0
DATA=0
ARTIFACTS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/cleanup_linux.sh [options]

Options:
  --all        Remove default cleanup targets, ignored data/logs, and generated artifacts.
  --data       Remove ignored raw data, results.tsv, and log files.
  --artifacts  Remove generated model/research artifacts while keeping artifacts/.gitkeep.
  --dry-run    Print what would be removed without deleting anything.
  -h, --help   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL=1 ;;
    --data) DATA=1 ;;
    --artifacts) ARTIFACTS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

remove_project_path() {
  local relative_path="$1"
  local target="$PROJECT_ROOT/$relative_path"
  local full_target
  full_target="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$target")"

  case "$full_target" in
    "$PROJECT_ROOT"/*) ;;
    *)
      echo "Refusing to remove path outside project root: $full_target" >&2
      exit 1
      ;;
  esac

  if [[ -e "$full_target" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "Would remove $relative_path"
    else
      echo "Removing $relative_path"
      rm -rf -- "$full_target"
    fi
  fi
}

cd "$PROJECT_ROOT"

default_targets=(
  ".venv"
  ".pytest_cache"
  "build"
  "dist"
  "src/worldcup_predictor.egg-info"
)

for target in "${default_targets[@]}"; do
  remove_project_path "$target"
done

for search_path in scripts src tests; do
  if [[ -d "$search_path" ]]; then
    while IFS= read -r cache_dir; do
      remove_project_path "$cache_dir"
    done < <(find "$search_path" -type d -name "__pycache__" -print)
  fi
done

if [[ "$ALL" -eq 1 || "$DATA" -eq 1 ]]; then
  remove_project_path "data/raw"
  remove_project_path "results.tsv"
  while IFS= read -r log_file; do
    remove_project_path "$log_file"
  done < <(find . -path "./.venv" -prune -o -type f -name "*.log" -print | sed 's#^\./##')
fi

if [[ "$ALL" -eq 1 || "$ARTIFACTS" -eq 1 ]]; then
  artifact_files=(
    "artifacts/autoresearch.tsv"
    "artifacts/metrics.json"
    "artifacts/model.joblib"
    "artifacts/prediction_log.tsv"
    "artifacts/training_cycles.svg"
    "artifacts/training_history.tsv"
    "artifacts/training_progress.svg"
  )
  for target in "${artifact_files[@]}"; do
    remove_project_path "$target"
  done
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete."
else
  echo "Cleanup complete."
fi
