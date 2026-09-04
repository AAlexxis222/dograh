from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.services.workflow import configuration_policy


def _stored_workflow(mappings: list[dict], lead_headers: list[str] | None = None):
    return SimpleNamespace(
        released_definition=SimpleNamespace(
            workflow_configurations={
                "external_pbx_field_mappings": mappings,
                "external_pbx_lead_headers": lead_headers or [],
            }
        )
    )


@pytest.mark.asyncio
async def test_disabled_policy_leaves_absent_keys_absent_and_does_not_load_workflow(
    monkeypatch,
):
    """Absent now means "inherited", not "the UI hid the section"; the server
    must not re-materialise stored PBX keys into the document."""
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=False),
    )
    get_workflow = AsyncMock()
    monkeypatch.setattr(configuration_policy.db_client, "get_workflow", get_workflow)

    incoming = {"max_call_duration": 600}
    prepared = await configuration_policy.apply_external_pbx_mapping_policy(
        incoming, workflow_id=12, organization_id=7
    )

    assert prepared is incoming
    get_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_policy_compares_changes_against_effective_document(
    monkeypatch,
):
    """Unchanged value (equal to org+stored effective) passes; a change is 403."""
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=False),
    )
    stored = [{"context_path": "qualified", "destination_field": "address3"}]
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_workflow",
        AsyncMock(return_value=_stored_workflow(stored)),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_draft_version",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_configuration_value",
        AsyncMock(return_value={}),
    )

    same = await configuration_policy.apply_external_pbx_mapping_policy(
        {"external_pbx_field_mappings": stored}, workflow_id=12, organization_id=7
    )
    assert same == {"external_pbx_field_mappings": stored}

    with pytest.raises(configuration_policy.ExternalPBXConfigurationDisabledError):
        await configuration_policy.apply_external_pbx_mapping_policy(
            {"external_pbx_field_mappings": []}, workflow_id=12, organization_id=7
        )


@pytest.mark.asyncio
async def test_disabled_policy_rejects_mapping_changes(monkeypatch):
    stored = [{"context_path": "qualified", "destination_field": "address3"}]
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_workflow",
        AsyncMock(return_value=_stored_workflow(stored)),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_draft_version",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_configuration_value",
        AsyncMock(return_value={}),
    )

    with pytest.raises(configuration_policy.ExternalPBXConfigurationDisabledError):
        await configuration_policy.apply_external_pbx_mapping_policy(
            {
                "external_pbx_field_mappings": [
                    {"context_path": "qualified", "destination_field": "comments"}
                ]
            },
            workflow_id=12,
            organization_id=7,
        )


@pytest.mark.asyncio
async def test_disabled_policy_rejects_lead_header_changes(monkeypatch):
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_workflow",
        AsyncMock(return_value=_stored_workflow([], ["first_name"])),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_draft_version",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_configuration_value",
        AsyncMock(return_value={}),
    )

    with pytest.raises(configuration_policy.ExternalPBXConfigurationDisabledError):
        await configuration_policy.apply_external_pbx_mapping_policy(
            {"external_pbx_lead_headers": ["first_name", "address1"]},
            workflow_id=12,
            organization_id=7,
        )


@pytest.mark.asyncio
async def test_disabled_policy_handles_workflow_without_released_definition(
    monkeypatch,
):
    """Duplicating a workflow can leave released_definition None until publish."""
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_workflow",
        AsyncMock(return_value=SimpleNamespace(released_definition=None)),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_draft_version",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_configuration_value",
        AsyncMock(return_value={}),
    )
    with pytest.raises(configuration_policy.ExternalPBXConfigurationDisabledError):
        await configuration_policy.apply_external_pbx_mapping_policy(
            {"external_pbx_lead_headers": ["first_name"]},
            workflow_id=12,
            organization_id=7,
        )


@pytest.mark.asyncio
async def test_enabled_external_pbx_policy_does_not_load_workflow(monkeypatch):
    get_workflow = AsyncMock()
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(configuration_policy.db_client, "get_workflow", get_workflow)
    incoming = {"external_pbx_field_mappings": []}

    prepared = await configuration_policy.apply_external_pbx_mapping_policy(
        incoming,
        workflow_id=12,
        organization_id=7,
    )

    assert prepared is incoming
    get_workflow.assert_not_awaited()
