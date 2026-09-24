#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
CRON_TZ_LINE="CRON_TZ=America/New_York"
CRON_LINE="0 18,19 * * 1-5 $BASE_DIR/run_pipeline_at_6_et.sh >> $BASE_DIR/pipeline.log 2>&1"
(
    crontab -l 2>/dev/null \
        | grep -v -F "$BASE_DIR/run_pipeline.sh" \
        | grep -v -F "$BASE_DIR/run_pipeline_at_6_et.sh" \
        | grep -v -F "$CRON_TZ_LINE" \
        || true
    echo "$CRON_LINE"
) | crontab -
echo "Installed DST-safe weekday 06:00 America/New_York cron for $BASE_DIR"
