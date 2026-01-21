#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP=0
RUN=0
TEST=0
FORMAT=0

for arg in "$@"; do
  case "$arg" in
    --bootstrap) BOOTSTRAP=1 ;;
    --run) RUN=1 ;;
    --test) TEST=1 ;;
    --format) FORMAT=1 ;;
  esac
done

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ "$BOOTSTRAP" -eq 1 ]; then
  python -m pip install -U pip
  python -m pip install -r requirements.txt
  python -m pip install -e .
  cp -n .env.example .env || true
  echo "Bootstrap done. Edit .env and set OPENAI_API_KEY."
  exit 0
fi

if [ "$TEST" -eq 1 ]; then
  pytest -q
  exit 0
fi

if [ "$FORMAT" -eq 1 ]; then
  ruff check .
  ruff format .
  exit 0
fi

if [ "$RUN" -eq 1 ]; then
  streamlit run app.py
  exit 0
fi

echo "Usage: ./scripts/dev.sh --bootstrap | --run | --test | --format"
