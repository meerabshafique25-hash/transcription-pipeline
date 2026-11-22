#!/usr/bin/env bash
# One-command start for macOS/Linux. Shows progress for every step.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VER=$($PY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
case "$VER" in
  3.10|3.11|3.12) ;;
  *) echo "Python $VER found, but this needs 3.10, 3.11 or 3.12."
     echo "Install one (e.g. 'brew install python@3.12') and run:  PYTHON=python3.12 ./run.sh"
     exit 1 ;;
esac

echo "==> [1/3] Setting up Python $VER environment and installing packages..."
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --progress-bar on | tail -1
pip install -r requirements.txt --progress-bar on

echo "==> [2/3] Downloading the Whisper 'small' model (~500 MB, only the first time)..."
hf download Systran/faster-whisper-small

echo "==> [3/3] Starting server. Open http://localhost:8000/docs  (Ctrl+C to stop)"
exec uvicorn main:app --port 8000
