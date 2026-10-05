"""A traffic-split variant freezes the configuration of its own definition."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.db.models import OrganizationModel, UserModel
from api.services.campaign import campaign_call_dispatcher
from api.services.campaign.campaign_call_dispatcher import CampaignCallDispatcher

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


class _StopAfterRunCreation(Exception):
    pass


@pytest.fixture
async def variant_campaign(db_session, async_session, monkeypatch):
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
    # The variant runs an older definition; the workflow's published one differs,
    # so a run that froze the published definition would carry the other voice.
    variant_def = await _publish(db_session, workflow, "variant-voice")
    published_def = await _publish(db_session, workflow, "published-voice")
    assert variant_def.id != published_def.id
    # The dispatcher reads through the module-level client; point it at the
    # test transaction so it sees the definitions created above.
    monkeypatch.setattr(campaign_call_dispatcher, "db_client", db_session)
    return SimpleNamespace(
        org=org, user=user, workflow=workflow, variant_def=variant_def
    )


@pytest.mark.asyncio
async def test_dispatch_call_freezes_the_variant_definitions_configuration(
    variant_campaign, db_session, monkeypatch
):
    ctx = variant_campaign
    created = []

    async def record_run(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(id=1)

    async def stop(*_args, **_kwargs):
        raise _StopAfterRunCreation

    module = campaign_call_dispatcher
    monkeypatch.setattr(db_session, "create_workflow_run", record_run)
    monkeypatch.setattr(module.call_concurrency, "bind_workflow_run", stop)
    monkeypatch.setattr(
        module.rate_limiter, "select_from_number", AsyncMock(return_value="+100")
    )
    monkeypatch.setattr(module, "campaign_split", lambda campaign: {"revision": 1})
    monkeypatch.setattr(
        module,
        "pick_variant",
        lambda split, phone: {
            "id": "b",
            "workflow_id": ctx.workflow.id,
            "workflow_definition_id": ctx.variant_def.id,
            "weight": 50,
        },
    )
    dispatcher = CampaignCallDispatcher()
    monkeypatch.setattr(
        dispatcher,
        "get_provider_for_campaign",
        AsyncMock(
            return_value=SimpleNamespace(PROVIDER_NAME="twilio", from_numbers=["+100"])
        ),
    )
    monkeypatch.setattr(dispatcher, "_finish_interrupted_dispatch", AsyncMock())
    campaign = SimpleNamespace(
        id=1,
        organization_id=ctx.org.id,
        telephony_configuration_id=None,
        created_by=ctx.user.id,
    )
    queued_run = SimpleNamespace(
        id=1, context_variables={"phone_number": "+34600000000"}, source_uuid="s"
    )

    with pytest.raises(_StopAfterRunCreation):
        await dispatcher.dispatch_call(queued_run, campaign, concurrency_slot=None)

    (run,) = created
    assert run["definition_id"] == ctx.variant_def.id
    assert run["effective_configurations"]["tts"]["voice"] == "variant-voice"
