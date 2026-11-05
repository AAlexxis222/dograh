"""The encryption-at-rest rule for one stored document (VOZ-AC-B6-15, VOZ-AC-B6-18). Pure: no database, no side effects.

Which leaves: ``secret_surfaces.SURFACES``. Each is stored as the credential_box token of the compact JSON of its
value, whatever its type, bound by ``credential_box.aad_for`` to its table, row id, column and concrete path: a token
moved to another row or list position does not decrypt. Empty values, tokens and run credential markers (references,
``<redacted>``) are never encrypted. Leaves are walked outermost first (an encrypted ``credentials`` object takes its
nested ``api_key`` along); a double-encoded document (a JSON string holding an object) is walked decoded and stays
encoded.

``unseal_document`` is token-driven and needs no row key: a token is authenticated by its AAD wherever the registry
allows it. A plaintext leaf passes as it is (dual reader until ``VOZ-FU-N0-24-contract``); a token that cannot be read
raises ``CredentialUnreadable``, never an empty value. ``seal_document`` writes in clear when no master key is set
(outside a cell only: the start gate refuses a cell without it).

Callers: the ORM seam (api/db/credentials_at_rest.py) and readers outside the ORM (scripts/xpand/deprecation_sweep.py).
Alembic revision e2b7c4d9a613 keeps a frozen copy of these rules.
"""

import copy
import json
from collections.abc import Callable
from functools import cache
from typing import Any

from api.services.configuration.credential_refs import is_credential_marker
from api.services.security import credential_box
from api.services.security.secret_surfaces import SURFACES, SecretSurface
from api.services.security.secret_walk import iter_leaves, value_at

_EMPTY: tuple[Any, ...] = (None, "", [], {})
# A document is walked as ``{_ROOT: document}``, so a whole-column surface (``json_path=()``) is a leaf like any other.
_ROOT = "$"

Step = Callable[[tuple[str | int, ...], Any], Any]


@cache
def secret_row_keys(table: str) -> frozenset[str]:
    """The row keys whose document holds secrets in a keyed store (empty for a table that is not keyed)."""
    return frozenset(s.row_key for s in SURFACES if s.table == table and s.row_key)


def _surfaces(table: str, column: str) -> tuple[SecretSurface, ...]:
    return tuple(s for s in SURFACES if (s.table, s.column) == (table, column))


def _changes(root: dict, surfaces: tuple[SecretSurface, ...], step: Step) -> dict:
    """``{path: new value}`` of every registered leaf ``step`` changes, outermost first: a changed leaf takes the
    leaves nested in it along."""
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
    if not isinstance(document, str) or credential_box.is_token(document):
        return None
    try:
        inner = json.loads(document)
    except ValueError:
        return None
    return inner if isinstance(inner, (dict, list)) else None


def _rewrite(document: Any, surfaces: tuple[SecretSurface, ...], step: Step) -> Any:
    """``document`` with ``step(path, value)`` applied to its registered leaves; the same object when nothing changes."""
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


def seal_document(
    table: str, pk: int, column: str, document: Any, row_key: str | None = None
) -> Any:
    """What to store in ``table.column`` of row ``pk`` for ``document``; the same object when nothing is encrypted.
    A keyed store encrypts by row key, so leaving it out is a programming error."""
    if row_key is None and secret_row_keys(table):
        raise ValueError(
            f"code=credential_seal_row_key_missing where={table}.{column} reason=this store encrypts by row key "
            "and none was given, so the document would be stored in clear hint=pass row_key (the row's key column)"
        )
    surfaces = tuple(
        s for s in _surfaces(table, column) if s.row_key in (None, row_key)
    )
    if not surfaces or not credential_box.key_configured():
        return document

    def step(path, value):
        if value in _EMPTY or is_credential_marker(value):
            return value
        aad = credential_box.aad_for(table, pk, column, path)
        if credential_box.is_token(value):
            # A token copied from another row or position fails here, not at the next read.
            _open(value, aad)
            return value
        return credential_box.encrypt(
            json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(), aad
        )

    return _rewrite(document, surfaces, step)


def unseal_document(table: str, pk: int, column: str, document: Any) -> Any:
    """The plaintext of a stored ``table.column`` value of row ``pk``; the same object when it holds no token."""

    def step(path, value):
        if not credential_box.is_token(value):
            return value
        return _open(value, credential_box.aad_for(table, pk, column, path))

    return _rewrite(document, _surfaces(table, column), step)
