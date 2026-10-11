"""Credentials at rest through the ORM (VOZ-AC-B6-15, VOZ-AC-B6-18). The rule itself (which leaves, the AAD, the
encoding, fail closed) is api/services/security/credential_documents.py; this module wires it to the models.

``load``/``refresh`` decrypt into the instance with ``set_committed_value`` (the row is not dirtied);
``before_insert``/``before_update`` encrypt the changed secret columns from the in-memory plaintext, so a value copied
to another row is always encrypted again for that row; ``after_insert``/``after_update`` put the plaintext back, and a
rollback gives it back or drops it. An INSERT without an id takes it from the table's sequence first, so no plaintext
row reaches the database. SQL that bypasses the unit of work calls ``seal``/``unseal`` itself;
api/tests/test_credentials_at_rest_guard.py lists those sites and fails on a new one.
"""

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
from api.services.security import credential_box
from api.services.security.credential_documents import (
    seal_document,
    secret_row_keys,
    unseal_document,
)
from api.services.security.secret_surfaces import SURFACES

# The keyed stores (organization_configurations, user_configurations) name each row's document in this column.
ROW_KEY_COLUMN = "key"
_STASH = "credentials_at_rest_plaintext"


def _table_column(column: InstrumentedAttribute) -> tuple[str, str]:
    (sql_column,) = column.property.columns
    return sql_column.table.name, sql_column.name


def seal(
    column: InstrumentedAttribute, pk: int, document: Any, row_key: str | None = None
) -> Any:
    """What to store in ``column`` of row ``pk`` for ``document``: for SQL that bypasses the ORM events. A keyed
    store (organization/user configurations) needs the row's ``key``."""
    table, name = _table_column(column)
    return seal_document(table, pk, name, document, row_key)


def unseal(column: InstrumentedAttribute, pk: int, document: Any) -> Any:
    """The plaintext of a ``column`` value of row ``pk`` read by SQL that bypasses the ORM events."""
    table, name = _table_column(column)
    return unseal_document(table, pk, name, document)


# --- ORM events ---------------------------------------------------------------------------------------------------

_ATTRIBUTES: dict[type, tuple[tuple[str, str], ...]] = {}


def _unseal_instance(target, attrs) -> None:
    state = inspect(target)
    table = state.mapper.local_table.name
    for key, column in _ATTRIBUTES[state.class_]:
        if key not in state.dict or (attrs is not None and key not in attrs):
            continue
        value = state.dict[key]
        plain = unseal_document(table, state.identity[0], column, value)
        if plain is not value:
            set_committed_value(target, key, plain)


def _on_load(target, _context) -> None:
    _unseal_instance(target, None)


def _on_refresh(target, _context, attrs) -> None:
    _unseal_instance(target, attrs)


def _before_write(mapper, connection, target, *, inserting: bool) -> None:
    if not credential_box.key_configured():
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
    row_key = getattr(target, ROW_KEY_COLUMN) if secret_row_keys(table) else None
    stash = {}
    for key, column in pending:
        plain = state.dict[key]
        sealed = seal_document(table, target.id, column, plain, row_key)
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
