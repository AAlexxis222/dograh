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


def _patch_storage(
    monkeypatch,
    *,
    workflow,
    draft=None,
    organization_defaults: dict | None = None,
):
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_workflow",
        AsyncMock(return_value=workflow),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_draft_version",
        AsyncMock(return_value=draft),
    )
    monkeypatch.setattr(
        configuration_policy.db_client,
        "get_configuration_value",
        AsyncMock(return_value=organization_defaults or {}),
    )


@pytest.mark.asyncio
async def test_disabled_policy_restores_absent_keys_from_storage(monkeypatch):
    """PBX keys have no organization layer, so an omitted key means "unchanged":
    the stored value is copied back instead of being dropped by the save."""
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=False),
    )
    stored = [{"context_path": "qualified", "destination_field": "address3"}]
    _patch_storage(monkeypatch, workflow=_stored_workflow(stored, ["first_name"]))

    prepared = await configuration_policy.apply_external_pbx_mapping_policy(
        {"max_call_duration": 600}, workflow_id=12, organization_id=7
    )

    assert prepared == {
        "max_call_duration": 600,
        "external_pbx_field_mappings": stored,
        "external_pbx_lead_headers": ["first_name"],
    }


@pytest.mark.asyncio
async def test_enabled_policy_restores_absent_keys_from_storage(monkeypatch):
    """The restore is about the save being partial, not about the gate."""
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=True),
    )
    stored = [{"context_path": "qualified", "destination_field": "address3"}]
    _patch_storage(monkeypatch, workflow=_stored_workflow(stored, ["first_name"]))

    prepared = await configuration_policy.apply_external_pbx_mapping_policy(
        {"max_call_duration": 600}, workflow_id=12, organization_id=7
    )

    assert prepared == {
        "max_call_duration": 600,
        "external_pbx_field_mappings": stored,
        "external_pbx_lead_headers": ["first_name"],
    }


@pytest.mark.asyncio
async def test_policy_prefers_the_draft_over_the_released_definition(monkeypatch):
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=True),
    )
    draft_headers = ["address1"]
    _patch_storage(
        monkeypatch,
        workflow=_stored_workflow([], ["first_name"]),
        draft=SimpleNamespace(
            workflow_configurations={"external_pbx_lead_headers": draft_headers}
        ),
    )

    prepared = await configuration_policy.apply_external_pbx_mapping_policy(
        {"max_call_duration": 600}, workflow_id=12, organization_id=7
    )

    assert prepared == {
        "max_call_duration": 600,
        "external_pbx_lead_headers": draft_headers,
    }


@pytest.mark.asyncio
async def test_policy_leaves_absent_keys_absent_when_nothing_is_stored(monkeypatch):
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=True),
    )
    _patch_storage(
        monkeypatch, workflow=SimpleNamespace(released_definition=None), draft=None
    )

    incoming = {"max_call_duration": 600}
    prepared = await configuration_policy.apply_external_pbx_mapping_policy(
        incoming, workflow_id=12, organization_id=7
    )

    assert prepared is incoming


@pytest.mark.asyncio
async def test_policy_raises_when_the_workflow_is_missing(monkeypatch):
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=True),
    )
    _patch_storage(monkeypatch, workflow=None)

    with pytest.raises(configuration_policy.WorkflowConfigurationNotFoundError):
        await configuration_policy.apply_external_pbx_mapping_policy(
            {"max_call_duration": 600}, workflow_id=12, organization_id=7
        )


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
    assert same == {
        "external_pbx_field_mappings": stored,
        "external_pbx_lead_headers": [],
    }

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
async def test_enabled_policy_with_every_key_sent_does_not_load_workflow(monkeypatch):
    """Nothing to restore and nothing to compare, so storage is never touched."""
    get_workflow = AsyncMock()
    monkeypatch.setattr(
        configuration_policy,
        "external_pbx_integrations_enabled",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(configuration_policy.db_client, "get_workflow", get_workflow)
    incoming = {
        "external_pbx_field_mappings": [],
        "external_pbx_lead_headers": [],
    }

    prepared = await configuration_policy.apply_external_pbx_mapping_policy(
        incoming,
        workflow_id=12,
        organization_id=7,
    )

    assert prepared is incoming
    get_workflow.assert_not_awaited()
