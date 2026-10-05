"""VOZ-BUG-18: one turn start strategy for migrated and unmigrated rows

Upstream's c4e21b7f80a9 rewrites `provisional_vad` to "default" only in `workflows` and
`workflow_definitions`. Everything else (the frozen `workflow_runs.effective_configurations`
snapshot and the org defaults document) still carries the retired value, which the runtime
reads as the min_words fallback — so migrated and unmigrated rows started turns differently.

This runs after c4e21b7f80a9 (the merge revision below descends from it) and applies the same
mapping to all four stores through `api.services.workflow.voz_bug_18`, the module the schema
reader uses too. It also pins `turn_start_strategy: "default"` in every
WORKFLOW_CONFIGURATION_DEFAULTS org document that lacks it (VOZ-AC-B2-30).

Row counts are taken per table before and after; any difference aborts the migration
(Postgres DDL/DML is transactional, so the abort rolls everything back).

Revision ID: 5be1d27c9a43
Revises: 966eb3a3309b
Create Date: 2026-10-05 08:10:00.000000

"""

import json
import logging
from typing import Callable, Sequence, Union

import sqlalchemy as sa
from alembic import op

from api.services.workflow.voz_bug_18 import backfill_turn_start, pin_org_default

revision: str = "5be1d27c9a43"
down_revision: Union[str, None] = "966eb3a3309b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

BATCH_SIZE = 500
_LEGACY = "provisional_vad"
_ORG_DEFAULTS_KEY = "WORKFLOW_CONFIGURATION_DEFAULTS"


def _org_defaults(doc: dict) -> dict:
    return pin_org_default(backfill_turn_start(doc))


# (table, json column, extra SQL filter selecting candidate rows, row transform).
# The columns are `json`, not `jsonb`: `->>` extracts text and renders a JSON null as SQL NULL.
# Org documents are few, so every one of them is a candidate (the pin may apply to a missing key).
_TARGETS: tuple[tuple[str, str, str, Callable[[dict], dict]], ...] = (
    (
        "organization_configurations",
        "value",
        f"key = '{_ORG_DEFAULTS_KEY}'",
        _org_defaults,
    ),
    (
        "workflow_runs",
        "effective_configurations",
        f"effective_configurations->>'turn_start_strategy' = '{_LEGACY}'",
        backfill_turn_start,
    ),
    (
        "workflows",
        "workflow_configurations",
        f"workflow_configurations->>'turn_start_strategy' = '{_LEGACY}'",
        backfill_turn_start,
    ),
    (
        "workflow_definitions",
        "workflow_configurations",
        f"workflow_configurations->>'turn_start_strategy' = '{_LEGACY}'",
        backfill_turn_start,
    ),
)


def _count(conn, table: str) -> int:
    return conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one()


def _backfill(conn, table: str, column: str, where: str, transform) -> int:
    """Keyset-paged rewrite; returns the number of rows changed."""
    changed = 0
    last_id = 0
    while True:
        rows = conn.execute(
            sa.text(
                f"SELECT id, {column} FROM {table} "
                f"WHERE id > :last_id AND {column} IS NOT NULL AND ({where}) "
                f"ORDER BY id LIMIT :limit"
            ),
            {"last_id": last_id, "limit": BATCH_SIZE},
        ).fetchall()
        if not rows:
            return changed
        for row_id, raw in rows:
            last_id = row_id
            doc = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(doc, dict):
                continue
            before = json.dumps(doc, sort_keys=True)
            after = transform(doc)
            if json.dumps(after, sort_keys=True) == before:
                continue
            conn.execute(
                sa.text(f"UPDATE {table} SET {column} = :doc WHERE id = :id"),
                {"doc": json.dumps(after), "id": row_id},
            )
            changed += 1


def upgrade() -> None:
    conn = op.get_bind()
    for table, column, where, transform in _TARGETS:
        before = _count(conn, table)
        changed = _backfill(conn, table, column, where, transform)
        after = _count(conn, table)
        logger.info(
            "VOZ-BUG-18 %s.%s: rows before=%d after=%d rewritten=%d",
            table,
            column,
            before,
            after,
            changed,
        )
        if before != after:
            raise RuntimeError(
                f"VOZ-BUG-18 aborted: {table} row count changed {before} -> {after}"
            )


def downgrade() -> None:
    # One-way and deliberate: the retired value no longer exists in the application, and the
    # pre-migration values are not recoverable from the rewritten documents. Rollback is the
    # pg_dump taken before this migration (VOZ-G0-02).
    pass
