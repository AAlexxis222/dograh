#!/usr/bin/env bash
# scripts/xpand/require_db_head.sh <where>, gate of every cell role (VOZ-AC-B5-44). It is also the startup gate for
# the cell durations (VOZ-AC-B3-55, see below): the name predates that check and is kept so the entrypoints do not change.
# Migrating is an explicit step run only by cellctl; a role never migrates. It refuses to start on a database
# BEHIND this image, so a stale schema stops here, named, and not in a query at runtime. A database AHEAD of the
# image (alembic cannot locate its revision) starts with a warning: during a rolling update the old call keeps
# draining against the new schema (VOZ-AC-B5-43), and a rolled-back image keeps running on an expanded schema
# (VOZ-AC-B5-45). ALEMBIC_CMD is injectable for tests.
# It also asserts the cell durations (VOZ-AC-B3-55) first: the one place every role passes through at startup, so an
# incoherent CELL_CALL_DURATION_CEILING_S stops the role, named, before it serves. DURATIONS_CMD is injectable.
set -uo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHERE="${1:-unknown}"
ALEMBIC_CMD="${ALEMBIC_CMD:-alembic -c api/alembic.ini}"
DURATIONS_CMD="${DURATIONS_CMD:-python -m api.services.runtime.durations}"
CELL_STARTUP_CMD="${CELL_STARTUP_CMD:-python -m api.services.runtime.cell_startup}"
cd "$BASE_DIR"

bash -c "$DURATIONS_CMD $WHERE" || exit 1 # the module prints its own VOZ-AC-B0-28 line
# The other startup checks of a cell (devops secret, LOG_LEVEL, CALL_K_P; VOZ-AC-B3-61, B6-52): arq and coordinators
# never run the api lifespan, which makes the same call. A no-op outside a cell.
bash -c "$CELL_STARTUP_CMD $WHERE" || exit 1

# VOZ-AC-B0-28 shape, one line.
report() { echo "code=$1 where=$WHERE reason=$2 hint=$3" >&2; } # <code> <reason> <hint>
fail() { report "$@"; exit 1; }

# Revision ids in $out (the output of `alembic <current|heads>`), one per line, sorted. Log lines never have this shape.
revision_ids() { { grep -E '^[0-9A-Za-z_]+( \(.*\))?$' <<<"$out" || true; } | awk '{print $1}' | sort -u; } # no match = unmigrated DB
last_line() { tail -n 1 <<<"$out" | tr -d '\r'; }

unreadable_hint="check that the database is reachable: DATABASE_URL, postgres is up, the network between this container and it"

out="$(bash -c "$ALEMBIC_CMD heads" 2>&1)" || fail db_schema_unreadable "alembic heads failed: $(last_line)" "$unreadable_hint"
heads="$(revision_ids)"
[[ "$(wc -l <<<"$heads")" -eq 1 && -n "$heads" ]] ||
  fail db_schema_behind "the code has no single migration head (heads: $(tr '\n' ' ' <<<"$heads"))" "fix the migration tree of this image"

if ! out="$(bash -c "$ALEMBIC_CMD current" 2>&1)"; then
  # alembic cannot place the database's revision in this image's tree: the database is ahead.
  if unknown="$(grep -oE "Can't locate revision identified by '[^']*'" <<<"$out")"; then
    report db_schema_ahead "database revision ${unknown##* } is unknown to this image, the database is ahead of it" \
      "expected while an older image keeps running after a migration or a rollback; if unplanned, deploy the image that ships that revision"
    exit 0
  fi
  fail db_schema_unreadable "alembic current failed: $(last_line)" "$unreadable_hint"
fi
current="$(revision_ids)"

[[ "$current" == "$heads" ]] ||
  fail db_schema_behind "database revision '$(tr '\n' ' ' <<<"$current")' is not the head '${heads}'" \
    "run the migration step (cellctl migrate; until it exists: docker compose -f docker-compose.yaml -f docker-compose.cell.yaml run --rm api ./scripts/run_migrate.sh)"
