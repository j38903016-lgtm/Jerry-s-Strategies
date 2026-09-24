#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"
set -a
source "$BASE_DIR/.env"
set +a
REPO_DIR="${DASHBOARD_REPO_DIR:?Set DASHBOARD_REPO_DIR to the cloned dashboard repository}"
PRIMARY_BRANCH="${DASHBOARD_PRIMARY_BRANCH:-main}"
MIRROR_BRANCH="${DASHBOARD_MIRROR_BRANCH:-us-stock-dashboard}"
PUSH_ATTEMPTS="${DASHBOARD_PUSH_ATTEMPTS:-5}"
PUSH_RETRY_BASE_SECONDS="${DASHBOARD_PUSH_RETRY_BASE_SECONDS:-15}"
PUSH_TIMEOUT_SECONDS="${DASHBOARD_PUSH_TIMEOUT_SECONDS:-90}"
GIT_SSH_KEY="${DASHBOARD_GIT_SSH_KEY:-/home/jerry/.ssh/a_share_dashboard_deploy}"
GITHUB_SSH_COMMAND="ssh -i $GIT_SSH_KEY -o IdentitiesOnly=yes -o HostName=ssh.github.com -p 443 -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"

"$BASE_DIR/.venv/bin/python" "$BASE_DIR/export_dashboard_snapshot.py" \
    --output "$REPO_DIR/data/dashboard.sqlite3"
cp "$BASE_DIR/dashboard.py" "$REPO_DIR/dashboard.py"
cp "$BASE_DIR/streamlit_app.py" "$REPO_DIR/streamlit_app.py"
cp "$BASE_DIR/requirements-dashboard.txt" "$REPO_DIR/requirements.txt"
mkdir -p "$REPO_DIR/.streamlit"
cp "$BASE_DIR/.streamlit/config.toml" "$REPO_DIR/.streamlit/config.toml"

cd "$REPO_DIR"
git add \
    .streamlit/config.toml \
    dashboard.py \
    data/dashboard.sqlite3 \
    requirements.txt \
    streamlit_app.py
if git diff --cached --quiet; then
    echo "dashboard snapshot unchanged; checking for an unpushed local commit"
else
    git commit -m "Update US dashboard snapshot $(TZ=America/New_York date '+%Y-%m-%d')"
fi

for ((attempt=1; attempt<=PUSH_ATTEMPTS; attempt++)); do
    push_refs=("HEAD:$PRIMARY_BRANCH")
    if [[ -n "$MIRROR_BRANCH" && "$MIRROR_BRANCH" != "$PRIMARY_BRANCH" ]]; then
        push_refs+=("HEAD:$MIRROR_BRANCH")
    fi
    if timeout "$PUSH_TIMEOUT_SECONDS" env GIT_SSH_COMMAND="$GITHUB_SSH_COMMAND" git push origin "${push_refs[@]}"; then
        echo "dashboard push succeeded on attempt ${attempt}/${PUSH_ATTEMPTS}"
        exit 0
    fi
    if (( attempt < PUSH_ATTEMPTS )); then
        wait=$(( PUSH_RETRY_BASE_SECONDS * (2 ** (attempt - 1)) ))
        echo "dashboard push attempt ${attempt}/${PUSH_ATTEMPTS} failed; retrying in ${wait}s" >&2
        sleep "$wait"
    fi
done

echo "dashboard push failed after ${PUSH_ATTEMPTS} attempts" >&2
exit 1
