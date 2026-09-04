"""Where secrets live inside a workflow configuration document.

Masking (API responses), the organization-level PUT (422 on any secret) and,
eventually, secret merging read from this one list. Paths use ``*`` for one
level and ``**`` for any depth. A schema field marked
``json_schema_extra={"secret": True}`` must appear here (guarded by
test_workflow_configuration_secrets_registry.py).
"""

from __future__ import annotations

import copy
from typing import Any

SECRET_LEAF_NAMES: tuple[str, ...] = (
    "api_key",
    "credentials",
    "aws_access_key",
    "aws_secret_key",
)

SECRET_PATHS: tuple[tuple[str, ...], ...] = tuple(
    [("model_overrides", "*", leaf) for leaf in SECRET_LEAF_NAMES]
    + [("model_configuration_v2_override", "**", leaf) for leaf in SECRET_LEAF_NAMES]
    + [("voicemail_detection", "api_key")]
    # Later PRs append: ("turn", "analyzer", "url"), ("service_tuning", "*", "ctor", "url")
)


def _walk(node: Any, pattern: tuple[str, ...], path: tuple[str, ...], found: list):
    if not pattern:
        if node not in (None, "", [], {}):
            found.append(path)
        return
    head, rest = pattern[0], pattern[1:]
    if not isinstance(node, dict):
        return
    if head == "**":
        for key, child in node.items():
            if rest and key == rest[0]:
                _walk(child, rest[1:], path + (key,), found)
            _walk(child, pattern, path + (key,), found)
    elif head == "*":
        for key, child in node.items():
            _walk(child, rest, path + (key,), found)
    elif head in node:
        _walk(node[head], rest, path + (head,), found)


def find_secret_paths(document: dict[str, Any] | None) -> list[tuple[str, ...]]:
    """Paths of non-empty secret values present in ``document``."""
    found: list[tuple[str, ...]] = []
    if isinstance(document, dict):
        for pattern in SECRET_PATHS:
            _walk(document, pattern, (), found)
    return sorted(set(found))


def mask_secrets(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """Copy of ``document`` with every registered secret masked. A secret
    value that is not a string, or a list of strings, is left untouched
    (e.g. a malformed ``api_key: 123``) rather than raised on."""
    from api.services.configuration.masking import mask_key  # circular at import time

    if not document:
        return document
    masked = copy.deepcopy(document)
    for path in find_secret_paths(masked):
        parent = masked
        for part in path[:-1]:
            parent = parent[part]
        value = parent[path[-1]]
        if isinstance(value, str):
            parent[path[-1]] = mask_key(value)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            parent[path[-1]] = [mask_key(item) for item in value]
    return masked
