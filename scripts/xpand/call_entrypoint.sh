#!/usr/bin/env bash
# scripts/xpand/call_entrypoint.sh, role `call` (VOZ-AC-B3-59/60): uvicorn never receives the
# orchestrator's TERM directly; it is stopped only after the drain finishes or fails.
# Why: uvicorn closes every live websocket with 1012 on TERM, which cuts the phone calls on it.
set -uo pipefail
# WEB_PORT is the port drain_web.sh polls: one knob, so the server and its drain cannot diverge.
CALL_CMD="${CALL_CMD:-uvicorn api.app:app --host 0.0.0.0 --port ${WEB_PORT:-8000} --workers 1}"
DRAIN_CMD="${DRAIN_CMD:-DRAIN_FAIL_CLOSED=true scripts/drain_web.sh}"

# VOZ-AC-B5-44: refuse to serve calls on a database whose schema is behind this image.
"$(dirname "${BASH_SOURCE[0]}")/require_db_head.sh" "${CELL_ROLE:-call}" || exit 1

# /api/v1/health/active-calls reports draining = this file exists. Cleared at start so a restart is not stuck draining.
DRAIN_FLAG_FILE="${DRAIN_FLAG_FILE:-/tmp/xpand_draining}"
export DRAIN_FLAG_FILE
rm -f "$DRAIN_FLAG_FILE"

bash -c "$CALL_CMD" &
child=$!

on_term() {
  # Reentrancy: a repeated TERM must neither re-run the drain nor re-signal uvicorn (a second TERM forces its exit
  # and skips the lifespan shutdown).
  trap '' TERM INT
  touch "$DRAIN_FLAG_FILE"
  bash -c "$DRAIN_CMD"
  local drain_status=$?
  kill -TERM "$child" 2>/dev/null
  wait "$child"
  local child_status=$?
  if [[ "$drain_status" -ne 0 ]]; then
    # VOZ-AC-B0-28 shape; a failed drain is never reported as a clean stop.
    echo "code=drain_failed where=call_entrypoint reason=drain command exited with status ${drain_status} hint=check DOGRAH_DEVOPS_SECRET and /api/v1/health/active-calls" >&2
    exit 1
  fi
  exit "$child_status"
}
trap on_term TERM INT
wait "$child"
