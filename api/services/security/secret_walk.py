"""The walk over the secret leaves a registry pattern names (api/services/security/secret_surfaces.py).

It lives on the security side because encryption at rest (VOZ-N0-24) decides which leaves are encrypted, and with which
AAD (``credential_box.aad_for``), from this walk: a change to how API responses are masked must never be able to change
that. Nothing here imports configuration; api/services/configuration/secrets_registry.py (masking, the 422 writer) reads
it from here.

A pattern is a tuple of keys: ``*`` matches one level and ``**`` any depth, and either may pass through a list (broadcast
over every item).
"""

from collections.abc import Iterator
from typing import Any


def iter_leaves(
    node: Any,
    pattern: tuple[str, ...],
    path: tuple[str | int, ...],
    *,
    indexed: bool = False,
) -> Iterator[tuple[tuple[str | int, ...], dict[str, Any], str]]:
    """Yield ``(path, container, key)`` for every leaf ``container[key]``
    matched by ``pattern`` under ``node``. A list encountered anywhere before
    the leaf is broadcast over (each item walked with the same remaining
    pattern); ``indexed`` records each item's position in ``path``, otherwise
    the item shares its list's path."""
    if not pattern:
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            item_path = path + (index,) if indexed else path
            yield from iter_leaves(item, pattern, item_path, indexed=indexed)
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
                    yield from iter_leaves(
                        child, rest[1:], path + (key,), indexed=indexed
                    )
            # "**" keeps matching at any depth below this key too.
            yield from iter_leaves(child, pattern, path + (key,), indexed=indexed)
    elif head == "*":
        if not rest:
            for key in node:
                yield (path + (key,), node, key)
        else:
            for key, child in node.items():
                yield from iter_leaves(child, rest, path + (key,), indexed=indexed)
    elif head in node:
        if not rest:
            yield (path + (head,), node, head)
        else:
            yield from iter_leaves(node[head], rest, path + (head,), indexed=indexed)


def iter_concrete_leaves(
    document: Any, pattern: tuple[str, ...]
) -> Iterator[tuple[tuple[str | int, ...], Any]]:
    """Yield ``(path, value)`` for every leaf of ``document`` matched by the
    registry ``pattern``, with each list position recorded as an int: one path
    per leaf, the shape ``credential_box.aad_for`` binds a token to."""
    for path, container, key in iter_leaves(document, pattern, (), indexed=True):
        yield path, container[key]
