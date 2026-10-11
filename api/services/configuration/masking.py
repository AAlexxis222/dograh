from __future__ import annotations

"""Utilities for masking API keys before they are sent to the client.

The rules are simple:
1. Only expose the last *visible* characters (default 4) of a key, behind a
   fixed run of mask characters: the mask never encodes the key's length
   (VOZ-AC-B6-19).
2. Incoming masked keys are considered a placeholder – if they equal the mask of
   the already-stored key, we treat them as *unchanged* and keep the real value
   in storage.
"""

import copy
from collections import defaultdict
from typing import Any, Dict, Optional

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration.credential_refs import is_credential_marker
from api.services.configuration.registry import ServiceConfig
from api.services.configuration.secrets_registry import mask_secrets
from api.services.security import secret_surfaces
from api.services.security.secret_surfaces import (
    MODEL_OVERRIDE_SECTIONS,
    SECRET_LEAF_NAMES,
)
from api.services.security.secret_walk import iter_leaves

VISIBLE_CHARS = 4  # number of trailing characters to reveal
MASK_CHAR = "*"
MASK_WIDTH = 4  # mask characters before the visible tail, whatever the key's length
MASK_MARKER = "***"  # substring that indicates a masked key
SERVICE_SECRET_FIELDS = SECRET_LEAF_NAMES
# The ``model_overrides`` sections merge.py restores real secrets into: a section outside this list (e.g.
# "embeddings") is never unmasked on merge, so it must never be masked either. The rule holds for ``model_overrides``
# only: ``model_configuration_v2_override`` is masked too and merge.py does not restore it, so a masked value sent back
# on that section would be stored as the mask string. What stops that today is the explicit masked-value check in
# ai_model_configuration.py, not the merger. Unifying the two descriptions of the secret set is a tracked follow-up.
MODEL_OVERRIDE_FIELDS = MODEL_OVERRIDE_SECTIONS
VOICEMAIL_DETECTION_KEY = "voicemail_detection"


def contains_masked_key(value: str | list[str] | None) -> bool:
    """Return True if *value* looks like a masked placeholder."""
    if value is None:
        return False
    keys = value if isinstance(value, list) else [value]
    return any(MASK_MARKER in k for k in keys)


def check_for_masked_keys(config: "EffectiveAIModelConfiguration") -> None:
    """Raise ValueError if any service in *config* still has a masked secret."""
    for field in ("llm", "tts", "stt", "embeddings", "realtime"):
        service = getattr(config, field, None)
        if service is None:
            continue
        for secret_field in SERVICE_SECRET_FIELDS:
            if not hasattr(service, secret_field):
                continue
            if secret_field == "api_key" and hasattr(service, "get_all_api_keys"):
                secret_value = service.get_all_api_keys()
            else:
                secret_value = getattr(service, secret_field, None)
            if contains_masked_key(secret_value):
                raise ValueError(
                    f"The {field} {secret_field} appears to be masked. "
                    "Please provide the actual value, not the masked value."
                )


class MaskedSecretUnmatched(Exception):
    """A masked secret came back with no stored value it masks (VOZ-AC-B0-28 shape).

    Storing it would replace the real secret with its mask. Never carries a
    value: ``where`` names the field."""

    def __init__(self, where: str):
        self.code, self.where = "masked_secret_unmatched", where
        self.reason = "a masked value was sent where no stored value matches it"
        self.hint = (
            "type the full value again; a mask only keeps the value it was read with"
        )
        super().__init__(
            f"code={self.code} where={where} reason={self.reason} hint={self.hint}"
        )


def mask_key(real_key: str, visible: int = VISIBLE_CHARS) -> str:
    """Return a masked representation of *real_key*.

    Example:
        >>> mask_key("sk-1234567890abcdef")
        '****cdef'
    """
    if not real_key:
        return ""

    if visible <= 0 or visible >= len(real_key):
        # too short to show any of it
        return MASK_CHAR * MASK_WIDTH

    return f"{MASK_CHAR * MASK_WIDTH}{real_key[-visible:]}"


def _length_revealing_mask(real_key: str) -> str:
    """The mask shape before VOZ-AC-B6-19 (one mask character per hidden one).

    A client holding a response from before the change still sends it back;
    follow-up VOZ-FU-N0-24-contract drops it."""
    if len(real_key) <= VISIBLE_CHARS:
        return MASK_CHAR * len(real_key)
    hidden = MASK_CHAR * (len(real_key) - VISIBLE_CHARS)
    return f"{hidden}{real_key[-VISIBLE_CHARS:]}"


def _mask_secret_value(value: str | list[str]) -> str | list[str]:
    if isinstance(value, list):
        return [mask_key(k) for k in value]
    return mask_key(value)


def is_mask_of(masked: str, real_key: str) -> bool:
    """Return *True* if *masked* equals the mask of *real_key*, current or previous shape."""
    if not real_key:
        return masked == ""
    return masked in (mask_key(real_key), _length_revealing_mask(real_key))


def resolve_masked_api_keys(
    incoming: str | list[str], existing: str | list[str]
) -> str | list[str]:
    """Resolve masked API keys against existing real keys.

    For each incoming key, if it matches the mask of an existing key, the real
    key is restored.  New (unmasked) keys are kept as-is.  This handles adds,
    removes, reorders, and partial replacements correctly.
    """
    if isinstance(incoming, str) and isinstance(existing, str):
        return existing if is_mask_of(incoming, existing) else incoming

    existing_list = existing if isinstance(existing, list) else [existing]
    incoming_list = incoming if isinstance(incoming, list) else [incoming]

    resolved: list[str] = []
    used: set[int] = set()
    for key in incoming_list:
        matched = False
        for i, real in enumerate(existing_list):
            if i not in used and is_mask_of(key, real):
                resolved.append(real)
                used.add(i)
                matched = True
                break
        if not matched:
            resolved.append(key)
    return resolved


# ---------------------------------------------------------------------------
# High-level helpers for EffectiveAIModelConfiguration objects
# ---------------------------------------------------------------------------


def _mask_service(service_cfg: Optional[ServiceConfig]) -> Optional[Dict[str, Any]]:
    if service_cfg is None:
        return None

    # Work on a dict copy so we don't mutate original models
    data = service_cfg.model_dump()
    for secret_field in SERVICE_SECRET_FIELDS:
        if secret_field not in data or not data[secret_field]:
            continue
        raw = data[secret_field]
        data[secret_field] = _mask_secret_value(raw)
    return data


def mask_user_config(config: EffectiveAIModelConfiguration) -> Dict[str, Any]:
    """Return a JSON-serialisable dict of *config* with every api_key masked."""

    return {
        "llm": _mask_service(config.llm),
        "tts": _mask_service(config.tts),
        "stt": _mask_service(config.stt),
        "embeddings": _mask_service(config.embeddings),
        "realtime": _mask_service(config.realtime),
        "is_realtime": config.is_realtime,
        "test_phone_number": config.test_phone_number,
        "timezone": config.timezone,
    }


def mask_workflow_configurations(config: Optional[Dict]) -> Optional[Dict]:
    """Mask every secret the registry declares (model overrides, v2 override,
    voicemail detection) for API responses."""
    return mask_secrets(config)


# ---------------------------------------------------------------------------
# Registry-driven mask / merge-back (workflow graphs, tool definitions)
# ---------------------------------------------------------------------------

# How a list item keeps its identity between a read and a save, by the list's field name: a node by its id, a custom
# header by its key (a key may repeat: each stored value is used once). Items of any other list match by position.
_LIST_ITEM_IDENTITY: dict[str, str] = {"nodes": "id", "custom_headers": "key"}


def _masked(value: Any) -> Any:
    """The mask of a secret leaf: strings and lists of strings; anything else, and a
    run's credential reference or redaction (not a secret), is returned as it is."""
    if is_credential_marker(value):
        return value
    if isinstance(value, str):
        return mask_key(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [mask_key(item) for item in value]
    return value


def mask_paths(document: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    """Copy of *document* with every leaf at a registered *paths* pattern masked.
    An empty or ``None`` document is returned unchanged rather than copied."""
    if not document:
        return document
    masked = copy.deepcopy(document)
    for pattern in paths:
        for _path, container, key in iter_leaves(masked, pattern, ()):
            container[key] = _masked(container[key])
    return masked


def _identity(document: Any, path: tuple[str | int, ...]) -> tuple[Any, ...]:
    """*path* with each list position replaced by the item's identity (``_LIST_ITEM_IDENTITY``)."""
    identity: list[Any] = []
    node: Any = document
    field = ""
    for part in path:
        node = node[part]
        if isinstance(part, int):
            key = _LIST_ITEM_IDENTITY.get(field) if isinstance(node, dict) else None
            identity.append((key, node.get(key)) if key else part)
        else:
            identity.append(part)
            field = part
    return tuple(identity)


def _describe(identity: tuple[Any, ...]) -> str:
    text = ""
    for part in identity:
        if isinstance(part, tuple):
            text += f"[{part[1]}]"
        elif isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text


def merge_masked_paths(
    incoming: Any,
    stored: Any,
    paths: tuple[tuple[str, ...], ...],
    *,
    where: str,
    keep_unmatched: frozenset[tuple[str, ...]] = frozenset(),
) -> Any:
    """Restore, in place, each masked leaf of *incoming* from the *stored* leaf with the
    same identity, each stored value used once. A leaf left out of *incoming* is gone. A
    mask with no stored value it masks is refused (``MaskedSecretUnmatched``) rather than
    stored, except under the patterns in *keep_unmatched*, where it is kept as sent."""
    if not incoming:
        return incoming
    for pattern in paths:
        candidates: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
        if stored:
            for path, container, key in iter_leaves(stored, pattern, (), indexed=True):
                candidates[_identity(stored, path)].append(container[key])
        for path, container, key in iter_leaves(incoming, pattern, (), indexed=True):
            value = container[key]
            if not isinstance(value, str):
                continue
            identity = _identity(incoming, path)
            pool = candidates.get(identity, [])
            match = next(
                (
                    index
                    for index, real in enumerate(pool)
                    if isinstance(real, str) and real and is_mask_of(value, real)
                ),
                None,
            )
            if match is not None:
                container[key] = pool.pop(match)
            elif contains_masked_key(value) and pattern not in keep_unmatched:
                raise MaskedSecretUnmatched(f"{where}:{_describe(identity)}")
    return incoming


# Node secret fields (``nodes.data.<field>``) keep the behaviour they had before VOZ-N0-24b: an unmatched mask is
# stored as sent. Every other graph path (custom headers, any path registered later) refuses it.
def _node_secret_field_paths() -> frozenset[tuple[str, ...]]:
    return frozenset(
        ("nodes", "data", field) for field in secret_surfaces.NODE_SECRET_FIELDS
    )


def mask_workflow_definition(workflow_definition: dict | None) -> dict | None:
    """Return a copy of *workflow_definition* with every registered graph secret masked."""
    return mask_paths(workflow_definition, secret_surfaces.WORKFLOW_GRAPH_SECRET_PATHS)


def merge_workflow_api_keys(
    incoming_definition: dict | None, existing_definition: dict | None
) -> dict | None:
    """Restore masked graph secrets in *incoming_definition* from *existing_definition*."""
    return merge_masked_paths(
        incoming_definition,
        existing_definition,
        secret_surfaces.WORKFLOW_GRAPH_SECRET_PATHS,
        where="workflow_definition",
        keep_unmatched=_node_secret_field_paths(),
    )


def refuse_masked_secrets(workflow_definition: dict | None) -> dict | None:
    """The create/import guard: a masked secret in a client-sent graph has nothing to keep
    (e.g. a downloaded workflow uploaded again), so it is refused rather than stored."""
    return merge_workflow_api_keys(workflow_definition, None)


def _tool_paths() -> tuple[tuple[str, ...], ...]:
    """A tool's static headers and its transfer resolver's, and any tool path registered later."""
    return tuple(
        surface.json_path
        for surface in secret_surfaces.SURFACES
        if surface.table == "tools"
    )


def mask_tool_definition(definition: dict | None) -> dict | None:
    """Return a copy of a tool *definition* with every registered secret masked."""
    return mask_paths(definition, _tool_paths())


def merge_tool_headers(incoming: dict, stored: dict | None) -> dict:
    """Restore masked secrets in a tool *incoming* definition from the *stored* one."""
    return merge_masked_paths(incoming, stored, _tool_paths(), where="tools.definition")


def refuse_masked_tool_headers(definition: dict) -> dict:
    """The tool create guard: a masked secret copied from another tool's response has nothing to keep."""
    return merge_tool_headers(definition, None)
