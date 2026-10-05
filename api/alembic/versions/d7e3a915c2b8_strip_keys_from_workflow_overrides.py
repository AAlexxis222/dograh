"""VOZ-G0-11: strip organization API keys copied into workflow overrides

`enrich_overrides_with_api_keys` stamped the org's provider keys into
`model_overrides` / `model_configuration_v2_override` on every save. The org already holds the
key and the credential object does not exist yet, so in G0 the copies are deleted. Which leaves
are secret is a frozen copy of `api.services.configuration.secrets_registry` (override sections).

Logs the number of leaves removed per row (workflows, definitions and the frozen
workflow_runs.effective_configurations snapshots) and aborts (rolling back, Postgres DDL/DML
is transactional) unless the rewritten rows then hold zero secret leaves.

Revision ID: d7e3a915c2b8
Revises: 5be1d27c9a43
Create Date: 2026-10-05 10:00:00.000000

"""

import copy
import json
import logging
from collections.abc import Iterator
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d7e3a915c2b8"
down_revision: Union[str, None] = "5be1d27c9a43"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

BATCH_SIZE = 500

# Frozen copy, as of this revision, of the override-shaped secret paths of
# api.services.configuration.secrets_registry and of its leaf walker: migrations do not import
# application code, which keeps changing after the revision ships.
OVERRIDE_KEYS: tuple[str, ...] = ("model_configuration_v2_override", "model_overrides")
_SECRET_LEAF_NAMES: tuple[str, ...] = (
    "api_key",
    "credentials",
    "aws_access_key",
    "aws_secret_key",
    "aws_session_token",
)
_MODEL_OVERRIDE_SECTIONS: tuple[str, ...] = ("llm", "tts", "stt", "realtime")
_OVERRIDE_PATTERNS: tuple[tuple[str, ...], ...] = tuple(
    [
        ("model_overrides", section, leaf)
        for section in _MODEL_OVERRIDE_SECTIONS
        for leaf in _SECRET_LEAF_NAMES
    ]
    + [("model_configuration_v2_override", "**", leaf) for leaf in _SECRET_LEAF_NAMES]
)


def _iter_leaves(
    node: Any, pattern: tuple[str, ...], path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], dict[str, Any], str]]:
    """Yield ``(path, container, key)`` for every leaf ``container[key]``
    matched by ``pattern`` under ``node``. A list encountered anywhere before
    the leaf is broadcast over (each item walked with the same remaining
    pattern and the same path)."""
    if not pattern:
        return
    if isinstance(node, list):
        for item in node:
            yield from _iter_leaves(item, pattern, path)
        return
    if not isinstance(node, dict):
        return

    head, rest = pattern[0], pattern[1:]
    if head == "**":
        for key, child in node.items():
            if rest and key == rest[0]:
                if len(rest) == 1:
                    yield (path + (key,), node, key)
                else:
                    yield from _iter_leaves(child, rest[1:], path + (key,))
            # "**" keeps matching at any depth below this key too.
            yield from _iter_leaves(child, pattern, path + (key,))
    elif head == "*":
        if not rest:
            for key in node:
                yield (path + (key,), node, key)
        else:
            for key, child in node.items():
                yield from _iter_leaves(child, rest, path + (key,))
    elif head in node:
        if not rest:
            yield (path + (head,), node, head)
        else:
            yield from _iter_leaves(node[head], rest, path + (head,))


def strip_secret_leaves(doc: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Return ``(copy of doc without secret leaves in both override shapes, leaves removed)``."""
    out = copy.deepcopy(doc)
    targets = [
        (container, key)
        for pattern in _OVERRIDE_PATTERNS
        for _path, container, key in _iter_leaves(out, pattern, ())
    ]
    removed = 0
    for container, key in targets:
        if key in container:  # a wildcard may reach the same leaf twice
            del container[key]
            removed += 1
    return out, removed


# Runs frozen since a7c2e9d41b06 copied the overrides, key included, into their snapshot.
_TARGETS = (
    ("workflows", "workflow_configurations"),
    ("workflow_definitions", "workflow_configurations"),
    ("workflow_runs", "effective_configurations"),
)


def _candidate(column: str) -> str:
    # The columns are `json`, not `jsonb`: `->` renders a JSON null as SQL NULL.
    return " OR ".join(
        [f"{column}->'{k}' IS NOT NULL" for k in OVERRIDE_KEYS]
        + [f"json_typeof({column}) = 'string'"]  # stored double-encoded
    )


def _load(raw):
    """The driver may hand back text; a document stored double-encoded needs a second decode."""
    for _ in range(2):
        if not isinstance(raw, str):
            break
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def _strip_table(conn, table: str, column: str) -> int:
    total = 0
    last_id = 0
    while True:
        rows = conn.execute(
            sa.text(
                f"SELECT id, {column} FROM {table} "
                f"WHERE id > :last_id AND {column} IS NOT NULL "
                f"AND ({_candidate(column)}) ORDER BY id LIMIT :limit"
            ),
            {"last_id": last_id, "limit": BATCH_SIZE},
        ).fetchall()
        if not rows:
            return total
        for row_id, raw in rows:
            last_id = row_id
            doc = _load(raw)
            if doc is None:
                continue
            out, removed = strip_secret_leaves(doc)
            _, left = strip_secret_leaves(out)
            if left:
                raise RuntimeError(
                    f"VOZ-G0-11 aborted: {table} id={row_id} still has secrets"
                )
            logger.info(
                "VOZ-G0-11 %s id=%d: secret leaves removed=%d", table, row_id, removed
            )
            if removed == 0:
                continue
            conn.execute(
                sa.text(f"UPDATE {table} SET {column} = :doc WHERE id = :id"),
                {"doc": json.dumps(out), "id": row_id},
            )
            total += removed
    return total


def upgrade() -> None:
    conn = op.get_bind()
    for table, column in _TARGETS:
        total = _strip_table(conn, table, column)
        logger.info(
            "VOZ-G0-11 %s.%s: total secret leaves removed=%d", table, column, total
        )


def downgrade() -> None:
    # One-way and deliberate: the removed keys are copies of the organization's own and are not
    # recoverable from the documents. Rollback is the pg_dump taken before this migration (VOZ-G0-02).
    pass
