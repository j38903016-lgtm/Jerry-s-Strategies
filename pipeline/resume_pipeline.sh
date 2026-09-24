#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"
set -a
source "$BASE_DIR/.env"
set +a

RUN_ID="${1:?usage: resume_pipeline.sh RUN_ID}"
LOCK_FILE="${PIPELINE_LOCK_FILE:-/data/us_stock_pipeline/pipeline.lock}"
SNAPSHOT_PATH="${DASHBOARD_SNAPSHOT:-$BASE_DIR/data/dashboard.sqlite3}"
PYTHON="$BASE_DIR/.venv/bin/python"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "pipeline already running; targeted resume skipped"
    exit 75
fi

"$PYTHON" "$BASE_DIR/pipeline.py" --mode full --resume-run-id "$RUN_ID"
"$PYTHON" "$BASE_DIR/export_dashboard_snapshot.py" --output "$SNAPSHOT_PATH"
if [[ -n "${DASHBOARD_REPO_DIR:-}" ]]; then
    "$BASE_DIR/publish_dashboard_snapshot.sh"
fi
