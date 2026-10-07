#!/usr/bin/env bash
# scripts/xpand/require_db_head.sh <where>, gate of every cell role (VOZ-AC-B5-44).
# Migrating is an explicit step run only by cellctl; a role never migrates. It refuses to start on a database
# whose alembic revision is not the single head of this image, so a stale schema stops here, named, and not in a
# query at runtime. ALEMBIC_CMD is injectable for tests.
set -uo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHERE="${1:-unknown}"
ALEMBIC_CMD="${ALEMBIC_CMD:-alembic -c api/alembic.ini}"
cd "$BASE_DIR"

# VOZ-AC-B0-28 shape, one line.
fail() { # <code> <reason>
  echo "code=$1 where=$WHERE reason=$2 hint=run the migration step (cellctl migrate; until it exists: docker compose run --rm api ./scripts/run_migrate.sh)" >&2
  exit 1
}

# Revision ids printed by `alembic <current|heads>`, one per line, sorted. Log lines never have this shape.
revisions() { # <subcommand>
  local out
  out="$(bash -c "$ALEMBIC_CMD $1" 2>&1)" || fail db_schema_unreadable "alembic $1 failed: $(tail -n 1 <<<"$out" | tr -d '\r')"
  { grep -E '^[0-9A-Za-z_]+( \(.*\))?$' <<<"$out" || true; } | awk '{print $1}' | sort -u # no match = unmigrated DB, not an error
}

current="$(revisions current)" || exit 1
heads="$(revisions heads)" || exit 1

[[ "$(wc -l <<<"$heads")" -eq 1 && -n "$heads" ]] || fail db_schema_behind "the code has no single migration head (heads: $(tr '\n' ' ' <<<"$heads"))"
[[ "$current" == "$heads" ]] || fail db_schema_behind "database revision '$(tr '\n' ' ' <<<"$current")' is not the head '${heads}'"
