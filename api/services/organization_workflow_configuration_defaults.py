"""Organization-level workflow configuration base (spec §2.1).

The stored document is sparse: only the keys the organization actually set,
so a later change to a schema default still reaches workflows that never
overrode it. Secrets, external-PBX keys and the AI-model sections are
rejected on write — they have their own gates and their own cascade.
"""

from loguru import logger

from api.db import db_client
from api.enums import OrganizationConfigurationKey
from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.configuration.cascade import (
    ORGANIZATION_FORBIDDEN_KEYS,
    ResolvedWorkflowConfigurations,
    normalize_root_nulls,
    resolve_effective_workflow_configurations,
)
from api.services.configuration.secrets_registry import find_secret_paths, mask_secrets

_KEY = OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value


class OrganizationWorkflowConfigurationRejected(ValueError):
    """The organization document may not carry secrets, PBX keys or AI-model sections."""


async def get_organization_workflow_configuration_defaults(
    organization_id: int,
) -> dict:
    """Stored sparse document, secrets masked (defence in depth: the PUT rejects them)."""
    stored = await db_client.get_configuration_value(organization_id, _KEY, {})
    return mask_secrets(normalize_root_nulls(stored)) or {}


def validate_organization_workflow_configuration_document(document: dict) -> None:
    forbidden = sorted(ORGANIZATION_FORBIDDEN_KEYS & document.keys())
    if forbidden:
        raise OrganizationWorkflowConfigurationRejected(
            f"keys not allowed at organization level: {', '.join(forbidden)}"
        )
    secrets = find_secret_paths(document)
    if secrets:
        raise OrganizationWorkflowConfigurationRejected(
            "secrets are not allowed at organization level: "
            + ", ".join(".".join(path) for path in secrets)
        )


async def upsert_organization_workflow_configuration_defaults(
    organization_id: int, *, user_id: int, configurations: WorkflowConfigurationDefaults
) -> dict:
    document = normalize_root_nulls(
        configurations.model_dump(mode="json", exclude_unset=True)
    )
    validate_organization_workflow_configuration_document(document)
    await db_client.upsert_configuration(organization_id, _KEY, document)
    # Audit: keys only, never values (§1.5). No org role model exists yet (§2.5).
    logger.info(
        "organization {} workflow configuration defaults updated by user {}: keys={}",
        organization_id,
        user_id,
        sorted(document),
    )
    return document


async def get_effective_organization_workflow_configuration_defaults(
    organization_id: int,
) -> ResolvedWorkflowConfigurations:
    stored = await db_client.get_configuration_value(organization_id, _KEY, {})
    return resolve_effective_workflow_configurations(
        organization_defaults=stored, definition_configurations={}
    )
