#!/usr/bin/env bash
#
# Wrapper for the scheduled daily run. Designed to be called by cron, launchd,
# systemd or Task Scheduler, none of which give you a login shell — so it
# resolves its own paths and activates its own virtualenv.
#
#   ./scripts/daily_scan.sh                    # uses the defaults below
#   QSCAN_UNIVERSE=us_all ./scripts/daily_scan.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UNIVERSE="${QSCAN_UNIVERSE:-sample}"
PRESET="${QSCAN_PRESET:-default}"
PROVIDER="${QSCAN_PROVIDER:-yfinance}"
CHARTS="${QSCAN_CHARTS:-25}"
WORKERS="${QSCAN_WORKERS:-8}"
DATA_DIR="${QSCAN_DATA:-$REPO_ROOT/data}"
OUT_DIR="${QSCAN_OUT:-$REPO_ROOT/out}"
LOG_DIR="${QSCAN_LOGS:-$REPO_ROOT/logs}"
PYTHON="${QSCAN_PYTHON:-python3}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily-$(date +%F).log"

# Never let a slow run overlap the next one. flock where available, a PID
# directory elsewhere (macOS ships no flock).
LOCK="$REPO_ROOT/.qscan.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK"
  if ! flock -n 9; then
    echo "$(date -Is) another run is in progress; exiting" >>"$LOG_FILE"
    exit 0
  fi
else
  if ! mkdir "$LOCK.d" 2>/dev/null; then
    echo "$(date -Is) another run is in progress; exiting" >>"$LOG_FILE"
    exit 0
  fi
  trap 'rmdir "$LOCK.d" 2>/dev/null || true' EXIT
fi

# Use the project virtualenv if there is one.
for candidate in "$REPO_ROOT/.venv/bin/activate" "$REPO_ROOT/venv/bin/activate"; do
  if [[ -f "$candidate" ]]; then
    # shellcheck disable=SC1090
    source "$candidate"
    PYTHON=python
    break
  fi
done

{
  echo "=============================================================="
  echo "$(date -Is)  qscan daily  universe=$UNIVERSE preset=$PRESET provider=$PROVIDER"
} >>"$LOG_FILE"

set +e
"$PYTHON" -m qscan daily \
  --universe "$UNIVERSE" \
  --preset "$PRESET" \
  --provider "$PROVIDER" \
  --repo "$DATA_DIR" \
  --out "$OUT_DIR" \
  --charts "$CHARTS" \
  --workers "$WORKERS" \
  >>"$LOG_FILE" 2>&1
STATUS=$?
set -e

echo "$(date -Is)  exit status $STATUS" >>"$LOG_FILE"

# Keep two months of logs.
find "$LOG_DIR" -name 'daily-*.log' -mtime +60 -delete 2>/dev/null || true

if [[ $STATUS -eq 0 ]]; then
  echo "report: $OUT_DIR/latest.html"
  # Uncomment to have the report pop open on the desktop when it finishes:
  # command -v open    >/dev/null && open    "$OUT_DIR/latest.html"   # macOS
  # command -v xdg-open >/dev/null && xdg-open "$OUT_DIR/latest.html" # Linux
else
  echo "qscan daily FAILED (status $STATUS) — see $LOG_FILE" >&2
fi

exit $STATUS
