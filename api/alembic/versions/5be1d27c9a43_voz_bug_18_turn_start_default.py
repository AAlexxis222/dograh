"""VOZ-BUG-18: one turn start strategy for migrated and unmigrated rows

Upstream's c4e21b7f80a9 rewrites `provisional_vad` to "default" only in `workflows` and
`workflow_definitions`. Everything else (the frozen `workflow_runs.effective_configurations`
snapshot and the org defaults document) still carries the retired value, which the runtime
reads as the min_words fallback — so migrated and unmigrated rows started turns differently.

This runs after c4e21b7f80a9 (the merge revision below descends from it) and applies the same
mapping to all four stores (a frozen copy of `api.services.workflow.voz_bug_18`, the module the
schema reader uses; migrations do not import application code). It also pins `turn_start_strategy: "default"` in every
WORKFLOW_CONFIGURATION_DEFAULTS org document that lacks it (VOZ-AC-B2-30).

After each table's pass the candidate rows are read again and any row still holding the
retired value (or, for org documents, still lacking the pin) aborts the migration (Postgres
DDL/DML is transactional, so the abort rolls everything back).

Revision ID: 5be1d27c9a43
Revises: 966eb3a3309b
Create Date: 2026-10-05 08:10:00.000000

"""

import json
import logging
from typing import Callable, Iterator, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "5be1d27c9a43"
down_revision: Union[str, None] = "966eb3a3309b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

BATCH_SIZE = 500
_LEGACY = "provisional_vad"
_ORG_DEFAULTS_KEY = "WORKFLOW_CONFIGURATION_DEFAULTS"


# Frozen copy of api.services.workflow.voz_bug_18 as of this revision.
_LEGACY_TO_CURRENT = {_LEGACY: "default"}
_ORG_DEFAULT = "default"  # VOZ-AC-B2-30


def backfill_turn_start(doc: dict) -> dict:
    value = doc.get("turn_start_strategy")
    if value in _LEGACY_TO_CURRENT:
        doc["turn_start_strategy"] = _LEGACY_TO_CURRENT[value]
    return doc


def _org_defaults(doc: dict) -> dict:
    """Org defaults document: also pin the strategy when absent or an explicit null."""
    backfill_turn_start(doc)
    if doc.get("turn_start_strategy") is None:
        doc["turn_start_strategy"] = _ORG_DEFAULT
    return doc


def _legacy(column: str) -> str:
    # The columns are `json`, not `jsonb`: `->>` extracts text and renders a JSON null as SQL NULL.
    # A document stored double-encoded is a JSON string, where `->>` sees no key: it is a
    # candidate too and the transform decides on the decoded document.
    return (
        f"{column}->>'turn_start_strategy' = '{_LEGACY}' "
        f"OR json_typeof({column}) = 'string'"
    )


# (table, json column, extra SQL filter selecting candidate rows, row transform).
# Org documents are few, so every one of them is a candidate (the pin may apply to a missing key).
def _is_legacy(doc: dict) -> bool:
    return doc.get("turn_start_strategy") == _LEGACY


def _org_unpinned(doc: dict) -> bool:
    return doc.get("turn_start_strategy") in (None, _LEGACY)


# (table, json column, extra SQL filter selecting candidate rows, row transform, "still pending"
# predicate checked after the pass — written independently of the transform so the guard also
# catches a transform that leaves the value in place).
_TARGETS: tuple[
    tuple[str, str, str, Callable[[dict], dict], Callable[[dict], bool]], ...
] = (
    (
        "organization_configurations",
        "value",
        f"key = '{_ORG_DEFAULTS_KEY}'",
        _org_defaults,
        _org_unpinned,
    ),
    (
        "workflow_runs",
        "effective_configurations",
        _legacy("effective_configurations"),
        backfill_turn_start,
        _is_legacy,
    ),
    (
        "workflows",
        "workflow_configurations",
        _legacy("workflow_configurations"),
        backfill_turn_start,
        _is_legacy,
    ),
    (
        "workflow_definitions",
        "workflow_configurations",
        _legacy("workflow_configurations"),
        backfill_turn_start,
        _is_legacy,
    ),
)


def _documents(conn, table: str, column: str, where: str) -> Iterator[tuple[int, dict]]:
    """Keyset-paged ``(id, document)`` of the candidate rows that hold a JSON object."""
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
            return
        for row_id, raw in rows:
            last_id = row_id
            doc = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(doc, dict):
                yield row_id, doc


def _backfill(conn, table: str, column: str, where: str, transform) -> int:
    """Returns the number of rows changed."""
    changed = 0
    for row_id, doc in _documents(conn, table, column, where):
        before = json.dumps(doc, sort_keys=True)
        after = transform(doc)
        if json.dumps(after, sort_keys=True) == before:
            continue
        conn.execute(
            sa.text(f"UPDATE {table} SET {column} = :doc WHERE id = :id"),
            {"doc": json.dumps(after), "id": row_id},
        )
        changed += 1
    return changed


def upgrade() -> None:
    conn = op.get_bind()
    for table, column, where, transform, pending in _TARGETS:
        changed = _backfill(conn, table, column, where, transform)
        left = sum(
            1 for _, doc in _documents(conn, table, column, where) if pending(doc)
        )
        logger.info(
            "VOZ-BUG-18 %s.%s: rewritten=%d still pending=%d",
            table,
            column,
            changed,
            left,
        )
        if left:
            raise RuntimeError(
                f"VOZ-BUG-18 aborted: {left} {table}.{column} rows still need the rewrite"
            )


def downgrade() -> None:
    # One-way and deliberate: the retired value no longer exists in the application, and the
    # pre-migration values are not recoverable from the rewritten documents. Rollback is the
    # pg_dump taken before this migration (VOZ-G0-02).
    pass
