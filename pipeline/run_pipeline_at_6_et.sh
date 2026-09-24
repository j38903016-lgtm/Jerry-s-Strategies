#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
NY_HOUR="$(TZ=America/New_York date '+%H')"
NY_WEEKDAY="$(TZ=America/New_York date '+%u')"
NY_NOW="$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"

# The host cron runs in Asia/Shanghai and does not honor CRON_TZ. It therefore
# calls this guard at both possible Shanghai equivalents of 06:00 New York:
# 18:00 during EDT and 19:00 during EST. Exactly one call passes this check.
if [[ "$NY_HOUR" != "06" || "$NY_WEEKDAY" -gt 5 ]]; then
    printf '[%s] SKIP candidate trigger; not 06:00 on a New York weekday\n' "$NY_NOW"
    exit 0
fi

printf '[%s] ACCEPT 06:00 New York trigger\n' "$NY_NOW"
exec "$BASE_DIR/run_pipeline.sh"
