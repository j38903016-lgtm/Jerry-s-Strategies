#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

log_event() {
    printf '[%s] %s\n' "$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

START_EPOCH="$(date +%s)"
on_exit() {
    status=$?
    elapsed=$(( $(date +%s) - START_EPOCH ))
    log_event "END status=${status} elapsed_seconds=${elapsed}"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
log_event "START scheduled market refresh"

set -a
source "$BASE_DIR/.env"
set +a
LOCK_FILE="${PIPELINE_LOCK_FILE:-/data/us_stock_pipeline/pipeline.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log_event "SKIP pipeline already running"
    exit 75
fi
PYTHON="$BASE_DIR/.venv/bin/python"
SNAPSHOT_PATH="${DASHBOARD_SNAPSHOT:-$BASE_DIR/data/dashboard.sqlite3}"
PIPELINE_RUN_ATTEMPTS="${PIPELINE_RUN_ATTEMPTS:-2}"
PIPELINE_RERUN_DELAY_SECONDS="${PIPELINE_RERUN_DELAY_SECONDS:-300}"

pipeline_succeeded=0
resume_run_id=""
for ((attempt=1; attempt<=PIPELINE_RUN_ATTEMPTS; attempt++)); do
    log_event "PIPELINE attempt=${attempt}/${PIPELINE_RUN_ATTEMPTS}"
    pipeline_args=(--mode full)
    if [[ -n "$resume_run_id" ]]; then
        pipeline_args+=(--resume-run-id "$resume_run_id")
        log_event "PIPELINE targeted resume run_id=${resume_run_id}"
    fi
    if "$PYTHON" "$BASE_DIR/pipeline.py" "${pipeline_args[@]}"; then
        pipeline_succeeded=1
        break
    fi
    if (( attempt < PIPELINE_RUN_ATTEMPTS )); then
        resume_run_id="$("$PYTHON" - "$PIPELINE_DB" <<'PY'
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as conn:
    row = conn.execute(
        "SELECT run_id FROM pipeline_runs "
        "WHERE mode='full' AND status IN ('partial','failed','running') "
        "ORDER BY julianday(started_at) DESC LIMIT 1"
    ).fetchone()
print(row[0] if row else "")
PY
)"
        log_event "PIPELINE attempt=${attempt} failed; automatic targeted resume in ${PIPELINE_RERUN_DELAY_SECONDS}s"
        sleep "$PIPELINE_RERUN_DELAY_SECONDS"
    fi
done

if (( pipeline_succeeded == 0 )); then
    log_event "PIPELINE failed after ${PIPELINE_RUN_ATTEMPTS} attempts; snapshot publication blocked"
    exit 1
fi

"$PYTHON" "$BASE_DIR/export_dashboard_snapshot.py" --output "$SNAPSHOT_PATH"

if [[ -n "${DASHBOARD_REPO_DIR:-}" ]]; then
    "$BASE_DIR/publish_dashboard_snapshot.sh"
fi
