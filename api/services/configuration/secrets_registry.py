"""Where secrets live inside a workflow configuration document.

Masking (API responses), the organization-level PUT (422 on any secret) and,
eventually, secret merging read from this one list. Paths use ``*`` for one
level and ``**`` for any depth; either may pass through a list (broadcast
over every item, no index recorded in the path — ``find_secret_paths``
dedupes; ``iter_concrete_leaves`` in api/services/security/secret_walk.py records it,
one path per leaf). A schema field marked ``json_schema_extra={"secret": True}`` must
appear in ``SECRET_PATHS``, which derives from the single registry in
api/services/security/secret_surfaces.py (guarded by
test_workflow_configuration_secrets_registry.py).

``find_secret_paths`` and ``mask_secrets`` disagree on non-scalar values by
design: a registered path holding a dict or a number is reported as a secret
present, but left untouched by the masking walk, which only rewrites strings
and lists of strings.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from api.services.configuration.credential_refs import is_credential_marker
from api.services.security.secret_surfaces import (
    SECRET_LEAF_NAMES,
    WORKFLOW_CONFIGURATION_SECRET_PATHS,
)
from api.services.security.secret_walk import iter_leaves

# Key names the write surfaces refuse on top of the registered secret leaves.
# Configuration documents accept unknown keys, so a credential can arrive under
# any name; these are the names a credential is usually given.
REJECTED_SECRET_NAMES: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "api_keys",
)

# Paths inside a workflow configuration document: the ones the single registry of
# secret surfaces declares for every column that stores that document.
SECRET_PATHS: tuple[tuple[str, ...], ...] = WORKFLOW_CONFIGURATION_SECRET_PATHS


def _find_non_empty(
    document: dict[str, Any] | None, patterns: tuple[tuple[str, ...], ...]
) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    if isinstance(document, dict):
        for pattern in patterns:
            for path, container, key in iter_leaves(document, pattern, ()):
                if container[key] not in (None, "", [], {}):
                    found.append(path)
    return sorted(set(found))


def find_secret_paths(document: dict[str, Any] | None) -> list[tuple[str, ...]]:
    """Paths of non-empty secret values present in ``document``."""
    return _find_non_empty(document, SECRET_PATHS)


def normalize_secret_name(name: str) -> str:
    """Comparison form of a key name: case, whitespace and word separators
    dropped, so ``apiKey``, ``API_KEY``, ``Api-Key`` and ``api_key `` are one
    name. Callers that reject secret-named keys must compare in this form —
    a JSON document can spell a key however its writer likes."""
    return "".join(name.split()).casefold().replace("_", "").replace("-", "")


def _iter_named_leaves(
    node: Any, names: frozenset[str], path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(path, value)`` for every key under ``node`` whose normalised
    name is in ``names``, at any depth, broadcasting over lists."""
    if isinstance(node, list):
        for item in node:
            yield from _iter_named_leaves(item, names, path)
        return
    if not isinstance(node, dict):
        return
    for key, child in node.items():
        if isinstance(key, str) and normalize_secret_name(key) in names:
            yield (path + (key,), child)
        yield from _iter_named_leaves(child, names, path + (key,))


def find_secret_named_paths(
    document: dict[str, Any] | None, *, extra_names: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    """Paths of non-empty values held by a secret-named key at any depth,
    registered or not. A document that accepts unknown keys can hide a secret
    under a section this registry has never heard of, and masking would then
    return it in clear; the writer that rejects secrets must use this walk
    rather than the registered paths. Names are compared normalised, and
    ``extra_names`` adds names a specific writer refuses on top of
    ``SECRET_LEAF_NAMES``. Every registered path ends in one of
    ``SECRET_LEAF_NAMES``, so this result is a superset of
    ``find_secret_paths``."""
    names = frozenset(
        normalize_secret_name(name) for name in SECRET_LEAF_NAMES + extra_names
    )
    found = [
        path
        for path, value in _iter_named_leaves(document, names, ())
        if value not in (None, "", [], {})
    ]
    return sorted(set(found))


def find_unregistered_secret_named_paths(
    document: dict[str, Any] | None, *, extra_names: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    """Secret-named paths a writer must refuse: everything
    ``find_secret_named_paths`` reports that this registry does not declare.

    A registered path is legitimate — it is masked on read and restored from
    storage on write — so it stays accepted. Anything else would be stored in
    clear and read back in clear, because the masking walk only knows the
    registered paths. Masking the unregistered names on read instead is not an
    option: clients send the whole document back, so a mask string would be
    persisted over the real value on the next save. Rejecting on write is the
    defence.
    """
    registered = set(find_secret_paths(document))
    return [
        path
        for path in find_secret_named_paths(document, extra_names=extra_names)
        if path not in registered
    ]


def mask_secrets(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """Copy of ``document`` with every registered secret masked. A secret
    value that is not a string, or a list of strings, is left untouched
    (e.g. a malformed ``api_key: 123``) rather than raised on, and so is a
    credential reference or redaction a run stores instead of a value
    (credential_refs.py). An empty or ``None`` document is returned unchanged
    rather than copied."""
    from api.services.configuration.masking import mask_key  # circular at import time

    if not document:
        return document
    masked = copy.deepcopy(document)
    for pattern in SECRET_PATHS:
        for _path, container, key in iter_leaves(masked, pattern, ()):
            value = container[key]
            if is_credential_marker(value):
                continue
            if isinstance(value, str):
                container[key] = mask_key(value)
            elif isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                container[key] = [mask_key(item) for item in value]
    return masked
