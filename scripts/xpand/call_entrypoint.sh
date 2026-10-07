#!/usr/bin/env bash
# scripts/xpand/call_entrypoint.sh, role `call` (VOZ-AC-B3-59/60): uvicorn never receives the
# orchestrator's TERM directly; it is stopped only after the drain finishes or fails.
# Why: uvicorn closes every live websocket with 1012 on TERM, which cuts the phone calls on it.
set -uo pipefail
CALL_CMD="${CALL_CMD:-uvicorn api.app:app --host 0.0.0.0 --port ${UVICORN_PORT:-8000} --workers 1}"
DRAIN_CMD="${DRAIN_CMD:-DRAIN_FAIL_CLOSED=true scripts/drain_web.sh}"

bash -c "$CALL_CMD" &
child=$!

on_term() {
  bash -c "$DRAIN_CMD"
  local drain_status=$?
  kill -TERM "$child" 2>/dev/null
  wait "$child"
  local child_status=$?
  if [[ "$drain_status" -ne 0 ]]; then
    # VOZ-AC-B0-28 shape; a failed drain is never reported as a clean stop.
    echo "code=drain_failed where=call_entrypoint reason=drain command exited with status ${drain_status} hint=check DOGRAH_DEVOPS_SECRET and /health/active-calls" >&2
    exit 1
  fi
  exit "$child_status"
}
trap on_term TERM INT
wait "$child"
