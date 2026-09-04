"""Organization → workflow configuration cascade.

Layers (spec §2.1): schema defaults → organization defaults → the definition
pinned by the run. Resolved once when the run is created and frozen on the
run row (§2.2-bis); nothing at runtime resolves again.

Merge semantics (§2.2): dicts key by key, scalars replace, lists replace
unless a rule declared *by path* opts into append-unique; the two AI-model
sections pass through untouched because they have their own cascade.
Validation is two-regime: the PUT keeps raising 422 through the schema; here
stored documents are clamped with a warning, never rejected.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from api.constants import (
    MAX_TEXT_CHAT_INACTIVITY_TIMEOUT_SECONDS,
    MIN_TEXT_CHAT_INACTIVITY_TIMEOUT_SECONDS,
)
from api.schemas.workflow_configurations import (
    MAX_CALL_DURATION_SECONDS,
    WorkflowConfigurationDefaults,
    schema_defaults_document,
)

# Owned by api/services/configuration/ai_model_configuration.py: replaced
# whole by the lowest layer that carries them, never deep-merged.
PASSTHROUGH_KEYS: frozenset[str] = frozenset(
    {"model_overrides", "model_configuration_v2_override"}
)
# Guarded by apply_external_pbx_mapping_policy (workflow PUT only); an org
# layer would bypass that gate. The AI-model sections already have their
# own organization level (MODEL_CONFIGURATION_V2).
ORGANIZATION_FORBIDDEN_KEYS: frozenset[str] = (
    frozenset({"external_pbx_field_mappings", "external_pbx_lead_headers"})
    | PASSTHROUGH_KEYS
)


@dataclass(frozen=True)
class ListMergeRule:
    """``flag`` is a sibling key of the list in the *overlay* document; when
    true, overlay items are appended to the base list (dedupe by ``dedupe_key``
    for lists of dicts, by value otherwise). An overlay item whose key matches
    a base item replaces it in place."""

    flag: str
    dedupe_key: str | None


LIST_MERGE_RULES: dict[tuple[str, ...], ListMergeRule] = {
    ("call_dispositions",): ListMergeRule(
        flag="call_dispositions_extend_org", dedupe_key="code"
    ),
    ("turn", "ignore_terms", "terms"): ListMergeRule(
        flag="merge_defaults", dedupe_key=None
    ),
}

# (path, min, max) — schema bounds only. Provider ranges (§5) arrive with the
# service_tuning PR and are appended here.
NUMERIC_BOUNDS: tuple[tuple[tuple[str, ...], float, float], ...] = (
    (("max_call_duration",), 1, MAX_CALL_DURATION_SECONDS),
    (
        ("text_chat_inactivity_timeout_seconds",),
        MIN_TEXT_CHAT_INACTIVITY_TIMEOUT_SECONDS,
        MAX_TEXT_CHAT_INACTIVITY_TIMEOUT_SECONDS,
    ),
)


@dataclass(frozen=True)
class ResolvedWorkflowConfigurations:
    effective: dict[str, Any]
    warnings: list[str]


def normalize_root_nulls(document: dict[str, Any] | None) -> dict[str, Any]:
    """Root ``null`` means "unset" (§1.6); nested nulls are kept verbatim."""
    if not isinstance(document, dict):
        return {}
    return {key: value for key, value in document.items() if value is not None}


def _item_key(item: Any, dedupe_key: str | None) -> Any:
    if dedupe_key is None:
        return item
    if isinstance(item, dict):
        value = item.get(dedupe_key)
        return value.casefold() if isinstance(value, str) else value
    return item


def _append_unique(base: list, overlay: list, dedupe_key: str | None) -> list:
    merged = copy.deepcopy(base)
    index = {
        _item_key(item, dedupe_key): position for position, item in enumerate(merged)
    }
    for item in overlay:
        key = _item_key(item, dedupe_key)
        if key in index:
            merged[index[key]] = copy.deepcopy(item)
        else:
            index[key] = len(merged)
            merged.append(copy.deepcopy(item))
    return merged


def _merge(base: dict, overlay: dict, path: tuple[str, ...]) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if not path and key in PASSTHROUGH_KEYS:
            merged[key] = copy.deepcopy(value)
        elif isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(current, value, path + (key,))
        elif isinstance(current, list) and isinstance(value, list):
            rule = LIST_MERGE_RULES.get(path + (key,))
            if rule is not None and overlay.get(rule.flag) is True:
                merged[key] = _append_unique(current, value, rule.dedupe_key)
            else:
                merged[key] = copy.deepcopy(value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_configuration_documents(base: dict, overlay: dict) -> dict:
    """Pure deep merge; neither input is mutated."""
    return _merge(base, overlay, ())


def _get_path(document: dict, path: tuple[str, ...]) -> Any:
    node: Any = document
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_path(document: dict, path: tuple[str, ...], value: Any) -> None:
    node = document
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value


def clamp_effective_configurations(
    effective: dict[str, Any], *, provenance: Callable[[str], str]
) -> tuple[dict[str, Any], list[str]]:
    """Clamp numeric bounds in place of raising. ``provenance(root_key)`` names
    the layer that set the key, for the warning. Values are logged only for
    numeric scalars; sections are never dumped (§1.5)."""
    clamped = copy.deepcopy(effective)
    warnings: list[str] = []
    for path, minimum, maximum in NUMERIC_BOUNDS:
        value = _get_path(clamped, path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        bounded = min(max(value, minimum), maximum)
        if bounded != value:
            _set_path(clamped, path, bounded)
            dotted = ".".join(path)
            warnings.append(
                f"{dotted}: {value} clamped to {bounded} (source={provenance(path[0])})"
            )
    return clamped, warnings


def _call_dispositions_valid(dispositions: Any) -> bool:
    try:
        WorkflowConfigurationDefaults.model_validate(
            {"call_dispositions": dispositions}
        )
    except ValidationError:
        return False
    return True


def resolve_effective_workflow_configurations(
    *,
    organization_defaults: dict[str, Any] | None,
    definition_configurations: dict[str, Any] | None,
) -> ResolvedWorkflowConfigurations:
    """Schema defaults ← organization defaults ← pinned definition."""
    organization_layer = normalize_root_nulls(organization_defaults)
    definition_layer = normalize_root_nulls(definition_configurations)

    effective = merge_configuration_documents(
        schema_defaults_document(), organization_layer
    )
    effective = merge_configuration_documents(effective, definition_layer)

    def provenance(root_key: str) -> str:
        if root_key in definition_layer:
            return "workflow"
        if root_key in organization_layer:
            return "organization"
        return "defaults"

    effective, warnings = clamp_effective_configurations(
        effective, provenance=provenance
    )

    # List invariants are not clamped: the lower layer wins whole (§2.2).
    if "call_dispositions" in effective and not _call_dispositions_valid(
        effective["call_dispositions"]
    ):
        fallback_layer = (
            definition_layer
            if "call_dispositions" in definition_layer
            else organization_layer
        )
        effective["call_dispositions"] = copy.deepcopy(
            fallback_layer.get("call_dispositions", [])
        )
        warnings.append(
            "call_dispositions: merged catalog violates the schema invariants "
            f"(unique codes / description budget); kept the {provenance('call_dispositions')} layer only"
        )
    return ResolvedWorkflowConfigurations(effective=effective, warnings=warnings)
