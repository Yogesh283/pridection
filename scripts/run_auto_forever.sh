#!/usr/bin/env bash
# Auto next prediction forever (writes prediction.json for the website).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source venv/bin/activate
export PYTHONUNBUFFERED=1
exec python -u main.py --live 0
