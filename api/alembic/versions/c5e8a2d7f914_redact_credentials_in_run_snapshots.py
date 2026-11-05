"""VOZ-N0-24b: historic runs keep credential references, not credential values

`workflow_runs.effective_configurations` froze the resolved configuration with its credentials in
clear (VOZ-AC-B6-15, VOZ-AC-B1-07). Every non-empty secret leaf of a frozen document (the
registered secret paths of a workflow configuration document) becomes
`<credential_ref:workflow_definitions:ID>` when the run's pinned definition holds the same value
at the same path, and `<redacted>` otherwise. The app resolves the references against the pinned
definition when it reads a run (api/services/configuration/credential_refs.py); this file keeps a
frozen copy of that rewrite because migrations do not import application code.

A document stored double-encoded (a JSON string) is rewritten with the same encoding: the
runtime reads it as no document at all, and decoding it would switch its settings on (VOZ-G0-11).
A pinned definition stored that way counts as no document, as it does for the app resolver, so
its values are redacted.

After the pass every row is read again and the migration aborts (rolling back) if any secret leaf
still holds a value. Run it with the API stopped: a run created by an older release meanwhile
would carry values and abort it.

Downgrade resolves the references back from the pinned definitions. ONE-WAY for the redacted
leaves: `<redacted>` stays, and only the logical backup taken before the upgrade restores those
values (VOZ-AC-B9-38).

Revision ID: c5e8a2d7f914
Revises: a4f1c8e27b35
Create Date: 2026-10-11 12:00:00.000000

"""

import json
import logging
import re
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c5e8a2d7f914"
down_revision: Union[str, None] = "a4f1c8e27b35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

BATCH_SIZE = 500

# Frozen copies, as of this revision: migrations do not import application code.
_SECRET_LEAF_NAMES: tuple[str, ...] = (  # secret_surfaces.SECRET_LEAF_NAMES
    "api_key",
    "credentials",
    "aws_access_key",
    "aws_secret_key",
    "aws_session_token",
)
_MODEL_OVERRIDE_SECTIONS: tuple[str, ...] = ("llm", "tts", "stt", "realtime")
# secret_surfaces.WORKFLOW_CONFIGURATION_SECRET_PATHS
_SECRET_PATHS: tuple[tuple[str, ...], ...] = tuple(
    [
        ("model_overrides", section, leaf)
        for section in _MODEL_OVERRIDE_SECTIONS
        for leaf in _SECRET_LEAF_NAMES
    ]
    + [("model_configuration_v2_override", "**", leaf) for leaf in _SECRET_LEAF_NAMES]
    + [("voicemail_detection", "api_key")]
    + [("service_tuning", "*", "*", "ctor", "url")]
)
_REDACTED = "<redacted>"
_REF = re.compile(r"<credential_ref:workflow_definitions:(\d+)>")
_EMPTY: tuple[Any, ...] = (None, "", [], {})

_SELECT = (
    "SELECT r.id, r.definition_id, r.effective_configurations, "
    "json_typeof(r.effective_configurations) = 'string', "
    "CASE WHEN json_typeof(d.workflow_configurations) = 'object' "
    "THEN d.workflow_configurations END "
    "FROM workflow_runs r LEFT JOIN workflow_definitions d ON d.id = r.definition_id "
    "WHERE r.id > :last_id AND r.effective_configurations IS NOT NULL "
    "ORDER BY r.id LIMIT :limit"
)


def _iter_leaves(node, pattern, path):
    """``(path, container, key)`` of each leaf ``pattern`` matches; a frozen copy of
    secret_walk.iter_leaves(..., indexed=True)."""
    if not pattern:
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _iter_leaves(item, pattern, path + (index,))
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


def _value_at(document, path):
    node = document
    for part in path:
        in_dict = isinstance(part, str) and isinstance(node, dict) and part in node
        in_list = (
            isinstance(part, int) and isinstance(node, list) and 0 <= part < len(node)
        )
        if not (in_dict or in_list):
            return False, None
        node = node[part]
    return True, node


def _live_secret_leaves(document):
    """Outermost first, read again at each step: a replaced leaf takes its nested ones along."""
    leaves = sorted(
        (
            leaf
            for pattern in _SECRET_PATHS
            for leaf in _iter_leaves(document, pattern, ())
        ),
        key=lambda leaf: len(leaf[0]),
    )
    for path, container, key in leaves:
        found, value = _value_at(document, path)
        if found:
            yield path, container, key, value


def _is_marker(value) -> bool:
    return isinstance(value, str) and (
        value == _REDACTED or _REF.fullmatch(value) is not None
    )


def _to_refs(doc: dict, pinned: dict, definition_id: int | None) -> int:
    """credential_refs.to_credential_refs, in place; returns the leaves replaced."""
    replaced = 0
    for path, container, key, value in _live_secret_leaves(doc):
        if value in _EMPTY or _is_marker(value):
            continue
        found, pinned_value = _value_at(pinned, path)
        same = definition_id is not None and found and pinned_value == value
        container[key] = (
            f"<credential_ref:workflow_definitions:{definition_id}>"
            if same
            else _REDACTED
        )
        replaced += 1
    return replaced


def _from_refs(doc: dict, pinned: dict, definition_id: int | None) -> int:
    """Each reference back to the pinned definition's value; one that no longer resolves
    becomes a redaction. Returns the leaves rewritten."""
    rewritten = 0
    for path, container, key, value in _live_secret_leaves(doc):
        match = _REF.fullmatch(value) if isinstance(value, str) else None
        if match is None:
            continue
        found, real = _value_at(pinned, path)
        resolves = int(match.group(1)) == definition_id and found and real not in _EMPTY
        container[key] = real if resolves else _REDACTED
        rewritten += 1
    return rewritten


def _values_left(doc: dict) -> int:
    return sum(
        1
        for _path, _container, _key, value in _live_secret_leaves(doc)
        if value not in _EMPTY and not _is_marker(value)
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


def _rows(conn):
    """Keyset-paged ``(run id, definition id, document, double-encoded, pinned document)``."""
    last_id = 0
    while True:
        rows = conn.execute(
            sa.text(_SELECT), {"last_id": last_id, "limit": BATCH_SIZE}
        ).fetchall()
        if not rows:
            return
        for run_id, definition_id, raw, double_encoded, raw_pinned in rows:
            last_id = run_id
            doc = _load(raw)
            if doc is None:
                logger.warning(
                    "VOZ-N0-24b workflow_runs id=%d: not a JSON object, skipped", run_id
                )
                continue
            yield run_id, definition_id, doc, double_encoded, _load(raw_pinned) or {}


def _write(conn, run_id: int, doc: dict, double_encoded: bool) -> None:
    stored = json.dumps(json.dumps(doc)) if double_encoded else json.dumps(doc)
    conn.execute(
        sa.text(
            "UPDATE workflow_runs SET effective_configurations = CAST(:doc AS json) "
            "WHERE id = :id"
        ),
        {"doc": stored, "id": run_id},
    )


def _rewrite(conn, step) -> tuple[int, int]:
    rows = leaves = 0
    for run_id, definition_id, doc, double_encoded, pinned in _rows(conn):
        count = step(doc, pinned, definition_id)
        if count:
            _write(conn, run_id, doc, double_encoded)
            rows += 1
            leaves += count
    return rows, leaves


def upgrade() -> None:
    conn = op.get_bind()
    rows, leaves = _rewrite(conn, _to_refs)
    left = [run_id for run_id, _, doc, _, _ in _rows(conn) if _values_left(doc)]
    logger.info(
        "VOZ-N0-24b workflow_runs: rows rewritten=%d leaves replaced=%d rows with values left=%d",
        rows,
        leaves,
        len(left),
    )
    if left:
        raise RuntimeError(
            "code=run_credentials_left where=workflow_runs.effective_configurations "
            f"ids={left[:5]} reason={len(left)} rows still hold a credential value after "
            "the rewrite hint=stop every API and worker process (an older release is "
            "creating runs) and run the upgrade again"
        )


def downgrade() -> None:
    conn = op.get_bind()
    rows, leaves = _rewrite(conn, _from_refs)
    logger.info(
        "VOZ-N0-24b downgrade workflow_runs: rows rewritten=%d references resolved or "
        "redacted=%d; redacted values stay redacted (restore the logical backup for them)",
        rows,
        leaves,
    )
