"""The three documents the builder needs to edit a workflow without merging
anything itself: what runs (effective), what the workflow stores (own) and
what an absent leaf inherits (base). Merge semantics live in cascade.py only."""

from dataclasses import dataclass
from typing import Any

from api.db import db_client
from api.enums import OrganizationConfigurationKey
from api.services.configuration.cascade import (
    normalize_root_nulls,
    resolve_effective_workflow_configurations,
)
from api.services.configuration.masking import mask_workflow_configurations

_KEY = OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value


@dataclass(frozen=True)
class WorkflowEffectiveConfigurations:
    effective: dict[str, Any]
    own: dict[str, Any]
    base: dict[str, Any]
    warnings: list[str]


def resolve_workflow_effective_configurations(
    *,
    organization_defaults: dict[str, Any] | None,
    definition_configurations: dict[str, Any] | None,
) -> WorkflowEffectiveConfigurations:
    own = normalize_root_nulls(definition_configurations)
    base = resolve_effective_workflow_configurations(
        organization_defaults=organization_defaults, definition_configurations={}
    )
    resolved = resolve_effective_workflow_configurations(
        organization_defaults=organization_defaults, definition_configurations=own
    )
    return WorkflowEffectiveConfigurations(
        effective=resolved.effective,
        own=own,
        base=base.effective,
        warnings=resolved.warnings,
    )


async def load_workflow_effective_configurations(
    *, organization_id: int | None, definition_configurations: dict[str, Any] | None
) -> WorkflowEffectiveConfigurations:
    """Masked in every layer: this feeds API responses only."""
    organization_defaults = (
        await db_client.get_configuration_value(organization_id, _KEY, {})
        if organization_id is not None
        else {}
    )
    layers = resolve_workflow_effective_configurations(
        organization_defaults=organization_defaults,
        definition_configurations=definition_configurations,
    )
    return WorkflowEffectiveConfigurations(
        effective=mask_workflow_configurations(layers.effective) or {},
        own=mask_workflow_configurations(layers.own) or {},
        base=mask_workflow_configurations(layers.base) or {},
        warnings=layers.warnings,
    )
