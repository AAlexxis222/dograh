"""Workflow-configuration policies shared by workflow update surfaces."""

from __future__ import annotations

from typing import Any

from api.db import db_client
from api.enums import OrganizationConfigurationKey
from api.services.configuration.cascade import resolve_effective_workflow_configurations
from api.services.organization_preferences import external_pbx_integrations_enabled

# Configuration keys owned by the External PBX UI section, guarded while the
# integration is disabled for the organization.
_EXTERNAL_PBX_CONFIGURATION_KEYS = (
    "external_pbx_field_mappings",
    "external_pbx_lead_headers",
)


class WorkflowConfigurationNotFoundError(LookupError):
    """Raised when configuration policy cannot resolve the target workflow."""


class ExternalPBXConfigurationDisabledError(PermissionError):
    """Raised when a disabled organization changes external-PBX mappings."""


async def apply_external_pbx_mapping_policy(
    workflow_configurations: dict[str, Any] | None,
    *,
    workflow_id: int,
    organization_id: int,
) -> dict[str, Any] | None:
    """Reject external-PBX edits while the integration is disabled.

    The stored document is sparse: a key the request does not send is
    inherited, so nothing is copied back in. A key the request does send is
    compared with the effective value (organization base + what the user was
    editing) and rejected if it differs.
    """
    if workflow_configurations is None or not any(
        key in workflow_configurations for key in _EXTERNAL_PBX_CONFIGURATION_KEYS
    ):
        return workflow_configurations
    if await external_pbx_integrations_enabled(organization_id):
        return workflow_configurations

    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is None:
        raise WorkflowConfigurationNotFoundError(
            f"Workflow with id {workflow_id} not found"
        )
    draft = await db_client.get_draft_version(workflow_id)
    released = getattr(workflow, "released_definition", None)
    stored = (
        draft.workflow_configurations
        if draft
        else (released.workflow_configurations if released else {})
    )
    organization_defaults = await db_client.get_configuration_value(
        organization_id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {},
    )
    effective = resolve_effective_workflow_configurations(
        organization_defaults=organization_defaults, definition_configurations=stored
    ).effective

    for key in _EXTERNAL_PBX_CONFIGURATION_KEYS:
        if key not in workflow_configurations:
            continue
        if workflow_configurations[key] != effective.get(key, []):
            raise ExternalPBXConfigurationDisabledError(
                "External PBX integrations are disabled for this organization. "
                "Enable them in Platform Settings before changing external PBX "
                "settings."
            )
    return workflow_configurations
