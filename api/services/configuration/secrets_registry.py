"""Where secrets live inside a workflow configuration document.

Masking (API responses), the organization-level PUT (422 on any secret) and,
eventually, secret merging read from this one list. Paths use ``*`` for one
level and ``**`` for any depth; either may pass through a list (broadcast
over every item, no index recorded in the path — ``find_secret_paths``
dedupes). A schema field marked ``json_schema_extra={"secret": True}`` must
appear here (guarded by test_workflow_configuration_secrets_registry.py).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

SECRET_LEAF_NAMES: tuple[str, ...] = (
    "api_key",
    "credentials",
    "aws_access_key",
    "aws_secret_key",
)

# The only sections merge.py restores real secrets into (MODEL_OVERRIDE_FIELDS
# in masking.py is an alias of this); a section outside this list (e.g.
# "embeddings") is never unmasked on merge, so it must never be masked either.
MODEL_OVERRIDE_SECTIONS: tuple[str, ...] = ("llm", "tts", "stt", "realtime")

SECRET_PATHS: tuple[tuple[str, ...], ...] = tuple(
    [
        ("model_overrides", section, leaf)
        for section in MODEL_OVERRIDE_SECTIONS
        for leaf in SECRET_LEAF_NAMES
    ]
    + [("model_configuration_v2_override", "**", leaf) for leaf in SECRET_LEAF_NAMES]
    + [("voicemail_detection", "api_key")]
    # Later PRs append: ("turn", "analyzer", "url"), ("service_tuning", "*", "ctor", "url")
)


def _iter_leaves(
    node: Any, pattern: tuple[str, ...], path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], dict, str]]:
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


def find_secret_paths(document: dict[str, Any] | None) -> list[tuple[str, ...]]:
    """Paths of non-empty secret values present in ``document``."""
    found: list[tuple[str, ...]] = []
    if isinstance(document, dict):
        for pattern in SECRET_PATHS:
            for path, container, key in _iter_leaves(document, pattern, ()):
                if container[key] not in (None, "", [], {}):
                    found.append(path)
    return sorted(set(found))


def mask_secrets(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """Copy of ``document`` with every registered secret masked. A secret
    value that is not a string, or a list of strings, is left untouched
    (e.g. a malformed ``api_key: 123``) rather than raised on."""
    from api.services.configuration.masking import mask_key  # circular at import time

    if not document:
        return document
    masked = copy.deepcopy(document)
    for pattern in SECRET_PATHS:
        for _path, container, key in _iter_leaves(masked, pattern, ()):
            value = container[key]
            if isinstance(value, str):
                container[key] = mask_key(value)
            elif isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                container[key] = [mask_key(item) for item in value]
    return masked
