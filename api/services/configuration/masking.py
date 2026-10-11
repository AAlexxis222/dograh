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
from typing import Any, Dict, Optional

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration.registry import ServiceConfig
from api.services.configuration.secrets_registry import mask_secrets
from api.services.integrations import get_node_secret_fields
from api.services.security.secret_surfaces import (
    MODEL_OVERRIDE_SECTIONS,
    SECRET_LEAF_NAMES,
    SURFACES,
)
from api.services.security.secret_walk import iter_leaves, value_at

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
# Workflow definition helpers – mask / merge node API keys
# ---------------------------------------------------------------------------

_NODE_SECRET_FIELDS: dict[str, tuple[str, ...]] = {
    "qa": ("qa_api_key",),
}


def _secret_fields_for_node_type(node_type: str | None) -> tuple[str, ...]:
    if not node_type:
        return ()
    return _NODE_SECRET_FIELDS.get(node_type, ()) or get_node_secret_fields(node_type)


def _header_items(data: Any) -> list[dict]:
    """A node's ``custom_headers`` entries (``[{key, value}]``); every value is a registered secret."""
    headers = data.get("custom_headers") if isinstance(data, dict) else None
    if not isinstance(headers, list):
        return []
    return [header for header in headers if isinstance(header, dict)]


def mask_workflow_definition(workflow_definition: Optional[Dict]) -> Optional[Dict]:
    """Return a copy of *workflow_definition* with node secret fields and custom header values masked."""
    if not workflow_definition:
        return workflow_definition

    masked = copy.deepcopy(workflow_definition)
    for node in masked.get("nodes", []):
        for header in _header_items(node.get("data")):
            if isinstance(header.get("value"), str):
                header["value"] = mask_key(header["value"])
        secret_fields = _secret_fields_for_node_type(node.get("type"))
        if not secret_fields:
            continue
        data = node.get("data", {})
        for field in secret_fields:
            raw_key = data.get(field)
            if raw_key:
                data[field] = mask_key(raw_key)
    return masked


def _merge_custom_headers(data: Any, old_data: dict | None, node_id: Any) -> None:
    """Restore each masked header value from the stored header of the same key in
    the same node; an omitted header is simply gone. A mask with no stored value
    it masks is refused rather than stored."""
    stored = {
        header.get("key"): header.get("value") for header in _header_items(old_data)
    }
    for header in _header_items(data):
        value = header.get("value")
        if not isinstance(value, str) or not contains_masked_key(value):
            continue
        real = stored.get(header.get("key"))
        if not (isinstance(real, str) and is_mask_of(value, real)):
            raise MaskedSecretUnmatched(
                f"workflow_definition:nodes[{node_id}].data"
                f".custom_headers[{header.get('key')}]"
            )
        header["value"] = real


def merge_workflow_api_keys(
    incoming_definition: Optional[Dict], existing_definition: Optional[Dict]
) -> Optional[Dict]:
    """Preserve real node secret fields and custom header values when the incoming value is masked."""
    if not incoming_definition:
        return incoming_definition

    existing_nodes: dict[Any, dict] = {
        node.get("id"): node.get("data") or {}
        for node in (existing_definition or {}).get("nodes", [])
        if isinstance(node, dict)
    }

    for node in incoming_definition.get("nodes", []):
        _merge_custom_headers(
            node.get("data"), existing_nodes.get(node.get("id")), node.get("id")
        )
        secret_fields = _secret_fields_for_node_type(node.get("type"))
        if not secret_fields:
            continue
        data = node.get("data", {})

        old_data = existing_nodes.get(node["id"])
        if not old_data:
            continue

        for field in secret_fields:
            incoming_key = data.get(field)
            if not incoming_key:
                continue

            old_key = old_data.get(field, "")
            if old_key and is_mask_of(incoming_key, old_key):
                data[field] = old_key

    return incoming_definition


# ---------------------------------------------------------------------------
# Tool definition helpers – mask / merge static headers
# ---------------------------------------------------------------------------

# A tool's static headers and its transfer resolver's (``dict[str, str]``), from the registry.
TOOL_HEADER_PATHS: tuple[tuple[str, ...], ...] = tuple(
    surface.json_path for surface in SURFACES if surface.table == "tools"
)


def mask_tool_definition(definition: dict | None) -> dict | None:
    """Return a copy of a tool *definition* with every static header value masked."""
    if not definition:
        return definition
    masked = copy.deepcopy(definition)
    for pattern in TOOL_HEADER_PATHS:
        for _path, container, key in iter_leaves(masked, pattern, ()):
            if isinstance(container[key], str):
                container[key] = mask_key(container[key])
    return masked


def merge_tool_headers(incoming: dict, stored: dict | None) -> dict:
    """Restore each masked header value in *incoming* from the *stored*
    definition's header of the same name.

    A header left out of *incoming* is gone (the definition is replaced whole);
    a mask with no stored value it masks is refused rather than stored."""
    for pattern in TOOL_HEADER_PATHS:
        for path, container, key in iter_leaves(incoming, pattern, ()):
            value = container[key]
            if not isinstance(value, str) or not contains_masked_key(value):
                continue
            _found, real = value_at(stored or {}, path)
            if not (isinstance(real, str) and is_mask_of(value, real)):
                raise MaskedSecretUnmatched(
                    f"tools.definition:{'.'.join(map(str, path))}"
                )
            container[key] = real
    return incoming
