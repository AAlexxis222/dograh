"""Workflow-configuration policies shared by workflow update surfaces."""

from __future__ import annotations

from typing import Any

from api.db import db_client
from api.enums import OrganizationConfigurationKey
from api.services.configuration.cascade import resolve_effective_workflow_configurations
from api.services.organization_preferences import external_pbx_integrations_enabled

# Configuration keys owned by the External PBX UI section. They cannot be
# inherited from the organization layer, so the workflow document is the only
# place they live.
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
    """Protect the external-PBX keys of a saved workflow document.

    Two rules, both about keys the request did or did not send:

    - A key the request omits means "leave it as it is". These keys have no
      organization-level value to fall back on, so the stored value is copied
      back into the outgoing document instead of being dropped.
    - A key the request sends while the integration is disabled for the
      organization may not differ from the value already in effect.
    """
    if workflow_configurations is None:
        return workflow_configurations

    sent_keys = [
        key
        for key in _EXTERNAL_PBX_CONFIGURATION_KEYS
        if key in workflow_configurations
    ]
    absent_keys = [
        key
        for key in _EXTERNAL_PBX_CONFIGURATION_KEYS
        if key not in workflow_configurations
    ]
    integration_enabled = await external_pbx_integrations_enabled(organization_id)
    if not absent_keys and integration_enabled:
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
    ) or {}

    if sent_keys and not integration_enabled:
        organization_defaults = await db_client.get_configuration_value(
            organization_id,
            OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
            {},
        )
        effective = resolve_effective_workflow_configurations(
            organization_defaults=organization_defaults,
            definition_configurations=stored,
        ).effective
        for key in sent_keys:
            # The [] default is unreachable while the schema declares both PBX
            # keys with a list default, so the effective document always has
            # them. Kept because this comparison must never raise on a document
            # that lost a key.
            if workflow_configurations[key] != effective.get(key, []):
                raise ExternalPBXConfigurationDisabledError(
                    "External PBX integrations are disabled for this organization. "
                    "Enable them in Platform Settings before changing external PBX "
                    "settings."
                )

    restored = {key: stored[key] for key in absent_keys if key in stored}
    if not restored:
        return workflow_configurations
    return {**workflow_configurations, **restored}
