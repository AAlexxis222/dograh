"""VOZ-N0-24c: encrypt every registered secret leaf at rest (expand + backfill + verify)

Every leaf of the registry of secret surfaces (api/services/security/secret_surfaces.py, frozen below as
`_SURFACES`) becomes a credential_box token: AES-256-GCM over the compact JSON of the value, AAD
`table:id:column:<concrete path as compact JSON>`, token `v1:<key_id>:<nonce_b64>:<ct_b64>`. The app reads
both forms until the contraction (VOZ-FU-N0-24-contract) and writes tokens from now on
(api/db/credentials_at_rest.py); this file keeps frozen copies of the envelope, the AAD, the walk and the
surfaces because migrations do not import application code. A test checks that the app decrypts what this
writes and that every surface frozen here is still registered. A surface registered later needs its own
migration: never edit e2b7c4d9a613 once it has shipped.

Leaves are walked outermost first (an encrypted `credentials` object takes its nested `api_key` along).
Empty values (null, "", {}, []), tokens and run credential markers (references, `<redacted>`) are left as
they are. A document stored double-encoded (a JSON string holding an object) has its leaves encrypted and
stays double-encoded: decoding it would switch its settings on (VOZ-G0-11). A whole-column value is
encrypted as it is, a JSON string included.

Verification (VOZ-AC-B9-38): every token written is decrypted again and compared with the value it
replaced; after each column's pass every row is read again, every token on it is decrypted, and the
migration aborts (rolling back) unless the tokens equal the secret leaves counted before and no plaintext
leaf is left.

Without CREDENTIALS_MASTER_KEY: in a cell (CELL_ROLE set) the upgrade refuses; anywhere else it changes
nothing, stamps the revision and logs how many leaves stay in clear, with the way back (set the key, then
`alembic downgrade c5e8a2d7f914 && alembic upgrade head`).

Every process of an older release READS a token as if it were the credential (it would send it to a
provider as a key, or to a customer's webhook as a header). So with the key, upgrade and downgrade refuse
while any other client session is connected to the database: stop api, call, arq and coordinators first.
Downgrade decrypts every token back (needs the key; refuses without it when a token exists). Roll the API
image back only after the downgrade, with the key.

Revision ID: e2b7c4d9a613
Revises: c5e8a2d7f914
Create Date: 2026-10-11 14:00:00.000000

"""

import base64
import copy
import json
import logging
import os
import re
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

revision: str = "e2b7c4d9a613"
down_revision: Union[str, None] = "c5e8a2d7f914"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

BATCH_SIZE = 500

# --- Frozen copies, as of this revision ------------------------------------------------------------------------------

_SECRET_LEAF_NAMES = (
    "api_key",
    "credentials",
    "aws_access_key",
    "aws_secret_key",
    "aws_session_token",
)
_WORKFLOW_CONFIGURATION_SECRET_PATHS = tuple(
    [
        ("model_overrides", section, leaf)
        for section in ("llm", "tts", "stt", "realtime")
        for leaf in _SECRET_LEAF_NAMES
    ]
    + [("model_configuration_v2_override", "**", leaf) for leaf in _SECRET_LEAF_NAMES]
    + [("voicemail_detection", "api_key")]
    + [("service_tuning", "*", "*", "ctor", "url")]
)
_WORKFLOW_GRAPH_SECRET_PATHS = tuple(
    [
        ("nodes", "data", field)
        for field in (
            "qa_api_key",
            "noveum_api_key",
            "paygent_api_key",
            "tuner_api_key",
        )
    ]
    + [("nodes", "data", "custom_headers", "value")]
)
# (table, column, json_path, row_key): secret_surfaces.SURFACES
_SURFACES: tuple[tuple[str, str, tuple[str, ...], str | None], ...] = (
    *(
        ("organization_configurations", "value", (), key)
        for key in (
            "MODEL_CONFIGURATION_V2",
            "TELEPHONY_CONFIGURATION",
            "TWILIO_CONFIGURATION",
            "LANGFUSE_CREDENTIALS",
        )
    ),
    ("organization_configurations", "value", ("config", "private_key"), "CALL_EVENTS"),
    ("telephony_configurations", "credentials", (), None),
    ("external_credentials", "credential_data", (), None),
    ("user_configurations", "configuration", (), "MODEL_CONFIGURATION"),
    ("integrations", "connection_details", (), None),
    ("webhook_deliveries", "custom_headers", (), None),
    *(
        (table, column, path, None)
        for table, column in (
            ("workflow_definitions", "workflow_configurations"),
            ("workflows", "workflow_configurations"),
            ("workflow_runs", "effective_configurations"),
        )
        for path in _WORKFLOW_CONFIGURATION_SECRET_PATHS
    ),
    *(
        (table, column, path, None)
        for table, column in (
            ("workflow_definitions", "workflow_json"),
            ("workflows", "workflow_definition"),
            ("workflow_templates", "template_json"),
        )
        for path in _WORKFLOW_GRAPH_SECRET_PATHS
    ),
    ("tools", "definition", ("config", "headers", "*"), None),
    ("tools", "definition", ("config", "resolver", "headers", "*"), None),
)
# The keyed stores name each row's document in this column.
_ROW_KEY_COLUMN = "key"

_ENV = "CREDENTIALS_MASTER_KEY"
_KEY_ID = re.compile(r"[A-Za-z0-9_.-]+")
_TOKEN = re.compile(r"v1:[A-Za-z0-9_.\-]+:[A-Za-z0-9+/]{16}:[A-Za-z0-9+/]{22,}={0,2}")
_MARKER = re.compile(r"<redacted>|<credential_ref:workflow_definitions:\d+>")
_EMPTY: tuple[Any, ...] = (None, "", [], {})
_ROOT = "$"


class _Abort(RuntimeError):
    """VOZ-AC-B0-28 shape; never carries a credential value or key material."""

    def __init__(self, code: str, where: str, reason: str, hint: str):
        super().__init__(f"code={code} where={where} reason={reason} hint={hint}")


def _keys() -> list[tuple[str, bytes]] | None:
    """credential_box.validate_master_key: ``None`` when the variable is unset or empty."""
    items = [
        item.strip() for item in os.environ.get(_ENV, "").split(",") if item.strip()
    ]
    if not items:
        return None
    keys = []
    for index, item in enumerate(items):
        key_id, _, b64 = (part.strip() for part in item.partition("="))
        try:
            key = base64.b64decode(b64, validate=True)
        except ValueError:
            key = b""
        if not _KEY_ID.fullmatch(key_id) or len(key) != 32 or key_id in dict(keys):
            raise _Abort(
                "credentials_master_key_invalid",
                f"{_ENV}[{index}]",
                "item is not 'key_id=base64(32 bytes)' with a distinct id",
                f"write {_ENV} as 'id=base64(32 bytes)', the active key first",
            )
        keys.append((key_id, key))
    return keys


def _aad(table: str, pk: int, column: str, path: tuple) -> bytes:
    return f"{table}:{pk}:{column}:{json.dumps(list(path), separators=(',', ':'))}".encode()


def _encrypt(keys, value: Any, aad: bytes) -> str:
    key_id, key = keys[0]
    nonce = os.urandom(12)
    plaintext = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    return f"v1:{key_id}:{base64.b64encode(nonce).decode()}:{base64.b64encode(ct).decode()}"


def _decrypt(keys, token: str, aad: bytes) -> Any:
    where = aad.decode("utf-8", "replace")
    _, key_id, nonce_b64, ct_b64 = token.split(":")
    key = dict(keys).get(key_id)
    if key is None:
        raise _Abort(
            "credential_unreadable",
            where,
            "token key id is not in CREDENTIALS_MASTER_KEY",
            "set the key that wrote the token in CREDENTIALS_MASTER_KEY",
        )
    try:
        plaintext = AESGCM(key).decrypt(
            base64.b64decode(nonce_b64), base64.b64decode(ct_b64), aad
        )
    except (InvalidTag, ValueError):
        raise _Abort(
            "credential_unreadable",
            where,
            "authentication failed (wrong row or key)",
            "restore the logical backup taken before the upgrade",
        ) from None
    return json.loads(plaintext)


def _is_token(value) -> bool:
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


def _iter_leaves(node, pattern, path):
    """``(path, container, key)`` of each leaf ``pattern`` matches; secret_walk.iter_leaves(..., indexed=True)."""
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
        node = node[part]
    return node


def _changes(root, patterns, step) -> dict:
    """credentials_at_rest._changes: outermost first, a changed leaf takes its nested ones along."""
    paths = sorted(
        {
            path
            for pattern in patterns
            for path, _c, _k in _iter_leaves(root, (_ROOT, *pattern), ())
        },
        key=len,
    )
    changes: dict = {}
    for path in paths:
        if any(path[:end] in changes for end in range(1, len(path))):
            continue
        value = _value_at(root, path)
        new = step(path[1:], value)
        if new is not value:
            changes[path] = new
    return changes


def _apply(document, changes):
    root = {_ROOT: copy.deepcopy(document)}
    for path, new in changes.items():
        _value_at(root, path[:-1])[path[-1]] = new
    return root[_ROOT]


def _decoded(document):
    if not isinstance(document, str) or _is_token(document):
        return None
    try:
        inner = json.loads(document)
    except ValueError:
        return None
    return inner if isinstance(inner, (dict, list)) else None


def _rewrite(document, patterns, step):
    """credentials_at_rest._rewrite: the same object when nothing changes; a double-encoded document is walked
    decoded and stays encoded."""
    changes = _changes({_ROOT: document}, patterns, step)
    if changes:
        return _apply(document, changes)
    inner = _decoded(document)
    leaf_patterns = [p for p in patterns if p]
    if inner is None or not leaf_patterns:
        return document
    changes = _changes({_ROOT: inner}, leaf_patterns, step)
    return json.dumps(_apply(inner, changes)) if changes else document


# --- The pass --------------------------------------------------------------------------------------------------------


def _groups():
    """``(table, column) -> [(json_path, row_key)]`` in registry order."""
    groups: dict = {}
    for table, column, path, row_key in _SURFACES:
        groups.setdefault((table, column), []).append((path, row_key))
    return groups


def _rows(conn, table, column, surfaces):
    """Keyset-paged ``(id, row key, stored value)``; only rows of a registered key in a keyed store."""
    row_keys = {row_key for _path, row_key in surfaces}
    keyed = None not in row_keys
    query = (
        f"SELECT id, {column}::text, {_ROW_KEY_COLUMN if keyed else 'NULL'} FROM {table} "
        f"WHERE {column} IS NOT NULL AND id > :last_id"
        + (f" AND {_ROW_KEY_COLUMN} IN :row_keys" if keyed else "")
        + " ORDER BY id LIMIT :limit"
    )
    statement = sa.text(query)
    params: dict = {"limit": BATCH_SIZE}
    if keyed:
        statement = statement.bindparams(sa.bindparam("row_keys", expanding=True))
        params["row_keys"] = sorted(row_keys)
    last_id = 0
    while True:
        rows = conn.execute(statement, {**params, "last_id": last_id}).fetchall()
        if not rows:
            return
        for row_id, raw, row_key in rows:
            last_id = row_id
            yield row_id, row_key, json.loads(raw)


def _patterns_for(surfaces, row_key, *, any_row_key=False):
    return [path for path, key in surfaces if any_row_key or key in (None, row_key)]


def _write(conn, table, column, row_id, value) -> None:
    try:
        conn.execute(
            sa.text(f"UPDATE {table} SET {column} = CAST(:doc AS json) WHERE id = :id"),
            {"doc": json.dumps(value), "id": row_id},
        )
    except sa.exc.DBAPIError as e:
        # The driver error prints the bound parameters: on a downgrade those are decrypted credentials. Only the
        # error class, the SQLSTATE and the constraint name leave this function.
        raise _Abort(
            "credentials_at_rest_write_failed",
            f"{table}:{row_id}:{column}",
            f"the database refused the write ({_db_error_names(e)})",
            "check the row's constraints and triggers, then run the migration again",
        ) from None


def _db_error_names(error) -> str:
    """``<class> sqlstate=<code> constraint=<name>`` of a driver error (asyncpg through SQLAlchemy, or psycopg)."""
    orig = error.orig
    cause = getattr(orig, "__cause__", None)
    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    constraint = getattr(cause, "constraint_name", None) or getattr(
        getattr(orig, "diag", None), "constraint_name", None
    )
    names = [type(orig).__name__]
    if sqlstate:
        names.append(f"sqlstate={sqlstate}")
    if constraint:
        names.append(f"constraint={constraint}")
    return " ".join(names)


def _refuse_other_sessions(conn) -> None:
    """Fail closed at the only point that sees every process: an older release still connected would read the
    tokens as credentials (upgrade), or a newer one would keep writing tokens (downgrade)."""
    others = _other_sessions(conn)
    if others:
        raise _Abort(
            "credentials_at_rest_sessions_open",
            "pg_stat_activity",
            f"{others} other client sessions are connected to this database, and a process of another "
            "release would read the stored credentials in the wrong form",
            "stop api, call, arq and coordinator processes (and close psql or monitoring sessions on this "
            "database), then run the migration again",
        )


def _other_sessions(conn) -> int:
    # pg_stat_activity is a snapshot taken once per transaction: clear it, or a re-check before commit sees the
    # sessions of the first check.
    conn.execute(sa.text("SELECT pg_stat_clear_snapshot()"))
    rows = conn.execute(
        sa.text(
            "SELECT backend_type, usesysid FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        )
    ).fetchall()
    return _count_other_sessions(rows)


def _count_other_sessions(rows) -> int:
    """Client backends, plus the sessions this role cannot classify: without pg_read_all_stats another role's
    rows show backend_type NULL, and failing open there would let an older release keep running."""
    return sum(
        1
        for backend_type, usesysid in rows
        if backend_type == "client backend"
        or (backend_type is None and usesysid is not None)
    )


def _census(conn, keys, table, column, surfaces) -> tuple[int, int]:
    """``(plaintext secret leaves, tokens)`` of a column; with ``keys`` every token is decrypted on the way."""
    plain = tokens = 0
    for row_id, row_key, value in _rows(conn, table, column, surfaces):

        def count(path, leaf, row_id=row_id):
            nonlocal plain, tokens
            if _is_token(leaf):
                tokens += 1
                if keys is not None:
                    _decrypt(keys, leaf, _aad(table, row_id, column, path))
            elif leaf not in _EMPTY and not (
                isinstance(leaf, str) and _MARKER.fullmatch(leaf)
            ):
                plain += 1
                # A new object reads as a change, so the leaves nested in this one are not counted again.
                return []
            return leaf

        _rewrite(value, _patterns_for(surfaces, row_key), count)
    return plain, tokens


def _encrypt_column(conn, keys, table, column, surfaces) -> None:
    secrets = rows = 0
    for row_id, row_key, value in _rows(conn, table, column, surfaces):

        def seal(path, leaf, row_id=row_id):
            nonlocal secrets
            if leaf in _EMPTY or (isinstance(leaf, str) and _MARKER.fullmatch(leaf)):
                return leaf
            secrets += 1
            aad = _aad(table, row_id, column, path)
            if _is_token(leaf):
                _decrypt(keys, leaf, aad)
                return leaf
            token = _encrypt(keys, leaf, aad)
            if _decrypt(keys, token, aad) != leaf:
                raise _Abort(
                    "credentials_at_rest_verify_failed",
                    aad.decode(),
                    "a token does not decrypt to the value it replaced",
                    "do not deploy; report it with this message",
                )
            return token

        new = _rewrite(value, _patterns_for(surfaces, row_key), seal)
        if new is not value:
            _write(conn, table, column, row_id, new)
            rows += 1
    plain, tokens = _census(conn, keys, table, column, surfaces)
    logger.info(
        "VOZ-N0-24c %s.%s: secret leaves=%d rows rewritten=%d tokens after=%d plaintext left=%d",
        table,
        column,
        secrets,
        rows,
        tokens,
        plain,
    )
    if plain or tokens != secrets:
        raise _Abort(
            "credentials_at_rest_verify_failed",
            f"{table}.{column}",
            f"after the pass {tokens} tokens and {plain} plaintext leaves, expected {secrets} tokens and none",
            "stop every API and worker process (an older release is writing) and run the upgrade again",
        )


def upgrade() -> None:
    conn = op.get_bind()
    keys = _keys()
    if keys is None:
        if os.environ.get("CELL_ROLE"):
            raise _Abort(
                "credentials_master_key_invalid",
                _ENV,
                f"{_ENV} is missing in a cell",
                "inject it from the cell Secrets Manager and run the upgrade again",
            )
        plain = sum(
            _census(conn, None, table, column, surfaces)[0]
            for (table, column), surfaces in _groups().items()
        )
        logger.warning(
            "code=credentials_at_rest_skipped where=alembic e2b7c4d9a613 reason=%s is not set, so %d secret "
            "leaves stay in clear and the revision is stamped hint=set %s, then run `alembic downgrade "
            "c5e8a2d7f914 && alembic upgrade head`",
            _ENV,
            plain,
            _ENV,
        )
        return
    _refuse_other_sessions(conn)
    for (table, column), surfaces in _groups().items():
        _encrypt_column(conn, keys, table, column, surfaces)
    _refuse_other_sessions(conn)  # again before commit: the pass can take minutes


def downgrade() -> None:
    conn = op.get_bind()
    keys = _keys()
    groups = _groups()
    if keys is None:
        tokens = sum(
            _census(conn, None, table, column, surfaces)[1]
            for (table, column), surfaces in groups.items()
        )
        if tokens:
            raise _Abort(
                "credentials_master_key_invalid",
                _ENV,
                f"{tokens} encrypted leaves cannot be decrypted without {_ENV}",
                f"set {_ENV} to the key that encrypted them and run the downgrade again",
            )
        return
    _refuse_other_sessions(conn)
    for (table, column), surfaces in groups.items():
        rows = 0
        for row_id, _row_key, value in _rows(conn, table, column, surfaces):

            def unseal(path, leaf, row_id=row_id, table=table, column=column):
                if not _is_token(leaf):
                    return leaf
                return _decrypt(keys, leaf, _aad(table, row_id, column, path))

            new = _rewrite(
                value, _patterns_for(surfaces, None, any_row_key=True), unseal
            )
            if new is not value:
                _write(conn, table, column, row_id, new)
                rows += 1
        _plain, tokens = _census(conn, keys, table, column, surfaces)
        logger.info(
            "VOZ-N0-24c downgrade %s.%s: rows decrypted=%d tokens left=%d",
            table,
            column,
            rows,
            tokens,
        )
        if tokens:
            raise _Abort(
                "credentials_at_rest_verify_failed",
                f"{table}.{column}",
                f"{tokens} tokens left after the downgrade",
                "run the downgrade again with the API stopped",
            )
    _refuse_other_sessions(conn)  # again before commit: the pass can take minutes
