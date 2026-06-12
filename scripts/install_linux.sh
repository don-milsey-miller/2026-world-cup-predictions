#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
DEV=0
SYSTEM_DEPS=0
DOWNLOAD_DATA=0
TRAIN_MODEL=0
RUN_CHECKS=0

usage() {
  cat <<'EOF'
Usage: scripts/install_linux.sh [options]

Options:
  --dev            Install development tools.
  --system-deps    Install Python/build dependencies with apt, dnf, or yum.
  --download-data  Refresh data/raw from the configured source.
  --train-model    Retrain artifacts/model.joblib after install.
  --run-checks     Run formatting, lint, and tests after install.
  -h, --help       Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) DEV=1 ;;
    --system-deps) SYSTEM_DEPS=1 ;;
    --download-data) DOWNLOAD_DATA=1 ;;
    --train-model) TRAIN_MODEL=1 ;;
    --run-checks) RUN_CHECKS=1 ;;
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

install_system_deps() {
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip build-essential
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip python3-devel gcc gcc-c++ make redhat-rpm-config
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y python3 python3-pip python3-devel gcc gcc-c++ make redhat-rpm-config
  else
    echo "No supported package manager found. Install Python 3.11+, venv, pip, and build tools manually." >&2
    exit 1
  fi
}

find_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "Python 3.11+ was not found. Re-run with --system-deps or install Python manually." >&2
    exit 1
  fi
}

if [[ "$SYSTEM_DEPS" -eq 1 ]]; then
  install_system_deps
fi

cd "$PROJECT_ROOT"
PYTHON_BIN="$(find_python)"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "Creating virtual environment at $VENV_PATH"
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

VENV_PYTHON="$VENV_PATH/bin/python"

echo "Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip

INSTALL_TARGET="."
if [[ "$DEV" -eq 1 ]]; then
  INSTALL_TARGET=".[dev]"
fi

echo "Installing project: $INSTALL_TARGET"
"$VENV_PYTHON" -m pip install -e "$INSTALL_TARGET"

if [[ "$DOWNLOAD_DATA" -eq 1 ]]; then
  echo "Downloading latest source data"
  "$VENV_PYTHON" scripts/download_data.py
fi

if [[ "$TRAIN_MODEL" -eq 1 ]]; then
  echo "Training model artifact"
  "$VENV_PYTHON" scripts/train_model.py
fi

if [[ "$RUN_CHECKS" -eq 1 ]]; then
  if [[ "$DEV" -ne 1 ]]; then
    echo "Installing dev tools for checks"
    "$VENV_PYTHON" -m pip install -e ".[dev]"
  fi
  "$VENV_PYTHON" -m ruff format --check .
  "$VENV_PYTHON" -m ruff check .
  "$VENV_PYTHON" -m pytest
fi

cat <<EOF

Install complete.
Run the app with:
  .venv/bin/python -m uvicorn worldcup_predictor.app:app --reload
Then open http://127.0.0.1:8000
EOF
