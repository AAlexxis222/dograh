"""Effective configuration is resolved once at run creation and frozen."""

import pytest

from api.db.models import OrganizationModel, UserModel
from api.enums import CallType, OrganizationConfigurationKey, WorkflowRunMode
from api.services.configuration.cascade import (
    WorkflowDefinitionMissingError,
    WorkflowDefinitionNotVisibleError,
    load_effective_workflow_configurations,
    run_configurations_for,
)
from api.services.workflow.run_creation import prepare_workflow_run_inputs

GRAPH = {
    "nodes": [
        {"id": "1", "type": "startCall", "data": {"name": "Start", "prompt": "Hello"}},
        {"id": "2", "type": "endCall", "data": {"name": "End", "prompt": "Bye"}},
    ],
    "edges": [
        {
            "id": "e",
            "source": "1",
            "target": "2",
            "data": {"label": "end", "condition": "x"},
        }
    ],
}


@pytest.fixture
async def org_user(async_session):
    org = OrganizationModel(provider_id="test-org-effective")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(provider_id="test-user-effective", selected_organization_id=org.id)
    async_session.add(user)
    await async_session.flush()
    return org, user


async def _published_workflow(db_session, org, user, configurations):
    workflow = await db_session.create_workflow(
        name="W", workflow_definition=GRAPH, user_id=user.id, organization_id=org.id
    )
    await db_session.save_workflow_draft(
        workflow.id, workflow_configurations=configurations
    )
    await db_session.publish_workflow_draft(workflow.id)
    return await db_session.get_workflow(workflow.id, organization_id=org.id)


async def _create_run(db_session, workflow, user, org):
    run_inputs = await prepare_workflow_run_inputs(db_session, workflow)
    return await db_session.create_workflow_run(
        name="run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        call_type=CallType.INBOUND,
        organization_id=org.id,
        definition_id=run_inputs.definition_id,
        initial_context=run_inputs.initial_context,
        effective_configurations=run_inputs.effective_configurations,
    )


async def test_run_freezes_effective_at_creation(db_session, org_user):
    org, user = org_user
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {"max_call_duration": 600, "dictionary": "org words"},
    )
    workflow = await _published_workflow(db_session, org, user, {"dictionary": "mine"})

    run = await _create_run(db_session, workflow, user, org)

    frozen = await db_session.get_workflow_run_configurations(run.id, org.id)
    assert frozen["max_call_duration"] == 600
    assert frozen["dictionary"] == "mine"
    assert frozen["max_user_idle_timeout"] == 10.0  # schema layer materialised

    # A later change of the organization base must not reach this run.
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {"max_call_duration": 900},
    )
    assert (await db_session.get_workflow_run_configurations(run.id, org.id))[
        "max_call_duration"
    ] == 600
    reloaded = await db_session.get_workflow_run(run.id, organization_id=org.id)
    assert run_configurations_for(reloaded)["max_call_duration"] == 600


async def test_legacy_run_without_snapshot_reads_pinned(db_session, org_user):
    org, user = org_user
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {"max_call_duration": 600},
    )
    workflow = await _published_workflow(db_session, org, user, {"dictionary": "mine"})
    run_inputs = await prepare_workflow_run_inputs(db_session, workflow)
    legacy = await db_session.create_workflow_run(
        name="legacy",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        organization_id=org.id,
        definition_id=run_inputs.definition_id,
        effective_configurations=None,
    )
    pinned = await db_session.get_workflow_run_configurations(legacy.id, org.id)
    assert pinned == {"dictionary": "mine"}  # no org layer retroactively
    reloaded = await db_session.get_workflow_run(legacy.id, organization_id=org.id)
    assert run_configurations_for(reloaded) == {"dictionary": "mine"}


async def test_definition_null_is_400_and_cross_tenant_is_403(
    db_session, org_user, async_session
):
    org, user = org_user
    with pytest.raises(WorkflowDefinitionMissingError):
        await load_effective_workflow_configurations(
            db_session, organization_id=org.id, definition_id=None
        )

    other_org = OrganizationModel(provider_id="test-org-other")
    async_session.add(other_org)
    await async_session.flush()
    workflow = await _published_workflow(db_session, org, user, {})
    with pytest.raises(WorkflowDefinitionNotVisibleError):
        await load_effective_workflow_configurations(
            db_session,
            organization_id=other_org.id,
            definition_id=workflow.released_definition_id,
        )
    with pytest.raises(WorkflowDefinitionMissingError):
        await load_effective_workflow_configurations(
            db_session, organization_id=org.id, definition_id=10**9
        )


async def test_run_detail_exposes_masked_effective_configurations(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    workflow = await _published_workflow(
        db_session,
        org,
        user,
        {"voicemail_detection": {"enabled": True, "api_key": "vm-1234567890"}},
    )
    run = await _create_run(db_session, workflow, user, org)
    async with test_client_factory(user) as client:
        response = await client.get(f"/api/v1/workflow/{workflow.id}/runs/{run.id}")
    body = response.json()["effective_configurations"]
    assert body["max_call_duration"] == 300
    assert body["voicemail_detection"]["api_key"].endswith("7890")
    assert "*" in body["voicemail_detection"]["api_key"]
