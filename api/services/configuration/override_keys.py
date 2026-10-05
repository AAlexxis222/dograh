"""Strip provider secrets from a workflow's model override documents (VOZ-G0-11).

Workflows used to receive a copy of the organization's API keys in their overrides
(``enrich_overrides_with_api_keys``). The organization already holds the key, so the copy is
removed. Which leaves are secret is decided by ``secrets_registry`` (the override-shaped
patterns of ``SECRET_PATHS``); this module keeps no list of its own.
"""

from __future__ import annotations

import copy
from typing import Any

from api.services.configuration.secrets_registry import SECRET_PATHS, _iter_leaves

OVERRIDE_KEYS: tuple[str, ...] = ("model_configuration_v2_override", "model_overrides")

_OVERRIDE_PATTERNS = tuple(p for p in SECRET_PATHS if p[0] in OVERRIDE_KEYS)


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
