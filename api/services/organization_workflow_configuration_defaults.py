"""Organization-level workflow configuration base: the middle layer of the
configuration cascade.

The stored document is sparse: only the keys the organization actually set,
so a later change to a schema default still reaches workflows that never
overrode it. Secrets, external-PBX keys and the AI-model sections are
rejected on write — they have their own gates and their own cascade.
"""

import json
from dataclasses import replace

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
from api.services.configuration.secrets_registry import (
    REJECTED_SECRET_NAMES,
    find_secret_named_paths,
    mask_secrets,
)

_KEY = OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value

# The document is one row of configuration, not a payload store. The cap keeps
# a single organization from parking megabytes behind every configuration read.
MAX_DOCUMENT_BYTES = 65536


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
    # Any depth, registered or not: the document accepts unknown keys, so a
    # secret under an unknown section would be stored and read back in clear.
    secrets = find_secret_named_paths(document, extra_names=REJECTED_SECRET_NAMES)
    if secrets:
        raise OrganizationWorkflowConfigurationRejected(
            "secrets are not allowed at organization level: "
            + ", ".join(".".join(path) for path in secrets)
            + ". Drop these keys from the request; a value read back from the "
            "GET is masked, and sending a masked value back is refused the "
            "same way."
        )
    size = len(json.dumps(document).encode("utf-8"))
    if size > MAX_DOCUMENT_BYTES:
        raise OrganizationWorkflowConfigurationRejected(
            f"document is {size} bytes, over the {MAX_DOCUMENT_BYTES} byte limit"
        )


async def upsert_organization_workflow_configuration_defaults(
    organization_id: int, *, user_id: int, configurations: WorkflowConfigurationDefaults
) -> dict:
    document = normalize_root_nulls(
        configurations.model_dump(mode="json", exclude_unset=True)
    )
    validate_organization_workflow_configuration_document(document)
    await db_client.upsert_configuration(organization_id, _KEY, document)
    # Audit trail: keys only, never values, because the document can hold
    # settings an operator should not see in a log. There is no organization
    # role model yet, so this line is the only trace of who changed the base.
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
    resolved = resolve_effective_workflow_configurations(
        organization_defaults=stored, definition_configurations={}
    )
    # Same defence in depth as the sibling GET: a secret the PUT never accepted
    # can still be in the store, and the effective document carries it through.
    return replace(resolved, effective=mask_secrets(resolved.effective) or {})
