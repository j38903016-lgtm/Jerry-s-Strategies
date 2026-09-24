#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"
set -a
source "$BASE_DIR/.env"
set +a
exec "$BASE_DIR/.venv/bin/python" -m streamlit run "$BASE_DIR/dashboard.py" --server.address 127.0.0.1 --server.port "${DASHBOARD_PORT:-8502}" --server.headless true --browser.gatherUsageStats false
