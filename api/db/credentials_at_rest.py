"""Credentials at rest (VOZ-AC-B6-15, VOZ-AC-B6-18): every registered secret leaf is stored as a credential_box token.

Which leaves: ``secret_surfaces.SURFACES``. Each is stored as the token of the compact JSON of its value, whatever
its type, bound by ``credential_box.aad_for`` to its table, row id, column and concrete path: a token moved to
another row or list position does not decrypt. Empty values, tokens and run credential markers (references,
``<redacted>``) are never encrypted.

The seam is the ORM. ``load``/``refresh`` decrypt into the instance with ``set_committed_value`` (the row is not
dirtied); ``before_insert``/``before_update`` encrypt the changed secret columns from the in-memory plaintext, so a
value copied to another row is always encrypted again for that row; ``after_insert``/``after_update`` put the
plaintext back. An INSERT without an id takes it from the table's sequence first, so no plaintext row reaches the
database. SQL that bypasses the unit of work calls ``seal``/``unseal`` itself; api/tests/test_credentials_at_rest_guard.py
lists those sites and fails on a new one.

Dual reader until the contraction (``VOZ-FU-N0-24-contract``): a token is decrypted, a plaintext leaf passes as it
is. A token that cannot be read raises ``CredentialUnreadable`` (fail closed, never an empty value). Without
``CREDENTIALS_MASTER_KEY`` values are written in clear, which only happens outside a cell (the cell start check
refuses to start without the key); the start gate of every role (``cell_startup``) says so. Alembic revision
e2b7c4d9a613 encrypts the rows written before this seam with a frozen copy of these rules.
"""

import copy
import json
import os
import re
from collections.abc import Callable
from functools import partial
from typing import Any

from sqlalchemy import event, inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import (
    InstrumentedAttribute,
    get_history,
    set_committed_value,
)

from api.db.models import Base
from api.services.configuration.credential_refs import is_credential_marker
from api.services.security import credential_box
from api.services.security.secret_surfaces import SURFACES, SecretSurface
from api.services.security.secret_walk import iter_leaves, value_at

# The keyed stores (organization_configurations, user_configurations) name each row's document in this column.
ROW_KEY_COLUMN = "key"
_EMPTY: tuple[Any, ...] = (None, "", [], {})
# A document is walked as ``{_ROOT: document}``, so a whole-column surface (``json_path=()``) is a leaf like any other.
_ROOT = "$"
_TOKEN = re.compile(credential_box.TOKEN_PATTERN)
_STASH = "credentials_at_rest_plaintext"

Step = Callable[[tuple[str | int, ...], Any], Any]


def encryption_enabled() -> bool:
    return bool(credential_box.master_key_items(os.environ.get(credential_box.ENV, "")))


def is_token(value: Any) -> bool:
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


def _surfaces(table: str, column: str) -> tuple[SecretSurface, ...]:
    return tuple(s for s in SURFACES if (s.table, s.column) == (table, column))


def _changes(root: dict, surfaces: tuple[SecretSurface, ...], step: Step) -> dict:
    """``{path: new value}`` of every registered leaf ``step`` changes, outermost first: a changed leaf takes the
    leaves nested in it along (an encrypted ``credentials`` object hides its ``api_key``)."""
    paths = sorted(
        {
            path
            for surface in surfaces
            for path, _container, _key in iter_leaves(
                root, (_ROOT, *surface.json_path), (), indexed=True
            )
        },
        key=len,
    )
    changes: dict = {}
    for path in paths:
        if any(path[:end] in changes for end in range(1, len(path))):
            continue
        value = value_at(root, path)[1]
        new = step(path[1:], value)
        if new is not value:
            changes[path] = new
    return changes


def _apply(document: Any, changes: dict) -> Any:
    root = {_ROOT: copy.deepcopy(document)}
    for path, new in changes.items():
        value_at(root, path[:-1])[1][path[-1]] = new
    return root[_ROOT]


def _decoded(document: Any) -> Any:
    """The document inside a double-encoded one (a JSON string holding an object or a list), else ``None``."""
    if not isinstance(document, str) or is_token(document):
        return None
    try:
        inner = json.loads(document)
    except ValueError:
        return None
    return inner if isinstance(inner, (dict, list)) else None


def _rewrite(document: Any, surfaces: tuple[SecretSurface, ...], step: Step) -> Any:
    """``document`` with ``step(path, value)`` applied to its registered leaves; the same object when nothing changes.
    A double-encoded document is walked decoded and stays encoded."""
    changes = _changes({_ROOT: document}, surfaces, step)
    if changes:
        return _apply(document, changes)
    inner = _decoded(document)
    leaf_surfaces = tuple(s for s in surfaces if s.json_path)
    if inner is None or not leaf_surfaces:
        return document
    changes = _changes({_ROOT: inner}, leaf_surfaces, step)
    return json.dumps(_apply(inner, changes)) if changes else document


def _open(token: str, aad: bytes) -> Any:
    try:
        return json.loads(credential_box.decrypt(token, aad))
    except credential_box.MasterKeyInvalid as e:
        raise credential_box.CredentialUnreadable(
            aad.decode("utf-8", "replace"), f"{e.where}: {e.reason}"
        ) from None
    except ValueError:  # authenticated, but not the JSON this module writes
        raise credential_box.CredentialUnreadable(
            aad.decode("utf-8", "replace"), "decrypted value is not JSON"
        ) from None


def _seal(table: str, pk: int, column: str, document: Any, row_key: str | None) -> Any:
    surfaces = tuple(
        s for s in _surfaces(table, column) if s.row_key in (None, row_key)
    )
    if not surfaces or not encryption_enabled():
        return document

    def step(path, value):
        if value in _EMPTY or is_credential_marker(value):
            return value
        aad = credential_box.aad_for(table, pk, column, path)
        if is_token(value):
            # A token copied from another row or position fails here, not at the next read.
            _open(value, aad)
            return value
        return credential_box.encrypt(
            json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(), aad
        )

    return _rewrite(document, surfaces, step)


def _unseal(table: str, pk: int, column: str, document: Any) -> Any:
    """Token-driven, so it needs no row key: a token is authenticated by its AAD wherever the registry allows it."""

    def step(path, value):
        if not is_token(value):
            return value
        return _open(value, credential_box.aad_for(table, pk, column, path))

    return _rewrite(document, _surfaces(table, column), step)


def _table_column(column: InstrumentedAttribute) -> tuple[str, str]:
    (sql_column,) = column.property.columns
    return sql_column.table.name, sql_column.name


def seal(
    column: InstrumentedAttribute, pk: int, document: Any, row_key: str | None = None
) -> Any:
    """What to store in ``column`` of row ``pk`` for ``document``: for SQL that bypasses the ORM events. A keyed
    store (organization/user configurations) encrypts by row key, so leaving it out is a programming error."""
    table, name = _table_column(column)
    if row_key is None and any(s.row_key for s in _surfaces(table, name)):
        raise ValueError(
            f"code=credential_seal_row_key_missing where={table}.{name} reason=this store encrypts by row key "
            f"and none was given, so the document would be stored in clear hint=pass row_key (the row's "
            f"{ROW_KEY_COLUMN} column)"
        )
    return _seal(table, pk, name, document, row_key)


def unseal(column: InstrumentedAttribute, pk: int, document: Any) -> Any:
    """The plaintext of a ``column`` value of row ``pk`` read by SQL that bypasses the ORM events."""
    table, name = _table_column(column)
    return _unseal(table, pk, name, document)


# --- ORM events ---------------------------------------------------------------------------------------------------

_ATTRIBUTES: dict[type, tuple[tuple[str, str], ...]] = {}


def _unseal_instance(target, attrs) -> None:
    state = inspect(target)
    table = state.mapper.local_table.name
    for key, column in _ATTRIBUTES[state.class_]:
        if key not in state.dict or (attrs is not None and key not in attrs):
            continue
        value = state.dict[key]
        plain = _unseal(table, state.identity[0], column, value)
        if plain is not value:
            set_committed_value(target, key, plain)


def _on_load(target, _context) -> None:
    _unseal_instance(target, None)


def _on_refresh(target, _context, attrs) -> None:
    _unseal_instance(target, attrs)


def _before_write(mapper, connection, target, *, inserting: bool) -> None:
    if not encryption_enabled():
        return
    state = inspect(target)
    pending = [
        (key, column)
        for key, column in _ATTRIBUTES[state.class_]
        if key in state.dict and (inserting or get_history(target, key).has_changes())
    ]
    if not pending:
        return
    table = mapper.local_table.name
    if target.id is None:
        target.id = connection.execute(
            text("SELECT nextval(pg_get_serial_sequence(:table, 'id'))"),
            {"table": table},
        ).scalar_one()
    row_key = (
        getattr(target, ROW_KEY_COLUMN)
        if any(s.row_key for s in SURFACES if s.table == table)
        else None
    )
    stash = {}
    for key, column in pending:
        plain = state.dict[key]
        sealed = _seal(table, target.id, column, plain, row_key)
        if sealed is not plain:
            stash[key] = plain
            setattr(target, key, sealed)
    if stash:
        state.info[_STASH] = stash


def _after_write(_mapper, _connection, target) -> None:
    for key, plain in inspect(target).info.pop(_STASH, {}).items():
        set_committed_value(target, key, plain)


def _give_back_plaintext(_session, target) -> None:
    """A rolled-back INSERT expunges the object: it leaves holding its plaintext again, not the tokens."""
    _after_write(None, None, target)


def _drop_stale_plaintext(session, _previous_transaction) -> None:
    """A rolled-back UPDATE expires the row, which reloads from the database: the plaintext kept for it is stale."""
    for state in session.identity_map.all_states():
        state.info.pop(_STASH, None)


def register() -> None:
    """Attach the events to every mapped class with a registered secret column (api/db/__init__.py, once)."""
    if _ATTRIBUTES:
        return
    event.listen(Session, "pending_to_transient", _give_back_plaintext)
    event.listen(Session, "persistent_to_transient", _give_back_plaintext)
    event.listen(Session, "after_soft_rollback", _drop_stale_plaintext)
    columns_by_table: dict[str, set[str]] = {}
    for surface in SURFACES:
        columns_by_table.setdefault(surface.table, set()).add(surface.column)
    for mapper in Base.registry.mappers:
        table = mapper.local_table
        if table.name not in columns_by_table:
            continue
        _ATTRIBUTES[mapper.class_] = tuple(
            (mapper.get_property_by_column(table.c[column]).key, column)
            for column in sorted(columns_by_table[table.name])
        )
        event.listen(mapper.class_, "load", _on_load)
        event.listen(mapper.class_, "refresh", _on_refresh)
        event.listen(
            mapper.class_, "before_insert", partial(_before_write, inserting=True)
        )
        event.listen(
            mapper.class_, "before_update", partial(_before_write, inserting=False)
        )
        event.listen(mapper.class_, "after_insert", _after_write)
        event.listen(mapper.class_, "after_update", _after_write)
