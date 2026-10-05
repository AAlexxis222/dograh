"""A traffic-split variant freezes the configuration of its own definition."""

import pytest

from api.db.models import OrganizationModel, UserModel
from api.services.campaign import campaign_call_dispatcher
from api.services.campaign.campaign_call_dispatcher import (
    resolve_run_configurations_for_variant,
)

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


async def _publish(db_session, workflow, voice):
    await db_session.save_workflow_draft(
        workflow.id, workflow_configurations={"tts": {"voice": voice}}
    )
    return await db_session.publish_workflow_draft(workflow.id)


@pytest.fixture
async def org_with_two_definitions(db_session, async_session, monkeypatch):
    org = OrganizationModel(provider_id="test-org-variant-config")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-variant-config", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    workflow = await db_session.create_workflow(
        name="W", workflow_definition=GRAPH, user_id=user.id, organization_id=org.id
    )
    base_def = await _publish(db_session, workflow, "base-voice")
    variant_def = await _publish(db_session, workflow, "variant-voice")
    assert base_def.id != variant_def.id
    # The dispatcher reads through the module-level client; point it at the
    # test transaction so it sees the definitions created above.
    monkeypatch.setattr(campaign_call_dispatcher, "db_client", db_session)
    return org, base_def, variant_def


@pytest.mark.asyncio
async def test_effective_config_comes_from_the_variant_definition(
    org_with_two_definitions,
):
    org, base_def, variant_def = (
        org_with_two_definitions  # variant sets tts.voice = "variant-voice"
    )
    cfg = await resolve_run_configurations_for_variant(
        org.id, definition_id=variant_def.id
    )
    assert cfg["tts"]["voice"] == "variant-voice"
