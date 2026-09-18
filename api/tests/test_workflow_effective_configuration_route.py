import pytest

from api.db.models import OrganizationModel, UserModel
from api.enums import OrganizationConfigurationKey

GRAPH = {
    "nodes": [{"id": "1", "type": "startCall", "data": {"name": "S", "prompt": "p"}}],
    "edges": [],
}
KEY = OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value


@pytest.fixture
async def org_user(async_session):
    org = OrganizationModel(provider_id="test-org-wf-effective")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-wf-effective", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    return org, user


async def _workflow(db_session, org, user, *, published: dict, draft: dict | None):
    workflow = await db_session.create_workflow(
        name="W",
        workflow_definition=GRAPH,
        user_id=user.id,
        organization_id=org.id,
        workflow_configurations=published,
    )
    if draft is not None:
        await db_session.save_workflow_draft(workflow.id, workflow_configurations=draft)
    return workflow


async def test_returns_three_layers_for_the_draft_being_edited(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    await db_session.upsert_configuration(
        org.id, KEY, {"max_call_duration": 600, "dictionary": "house"}
    )
    workflow = await _workflow(
        db_session,
        org,
        user,
        published={"max_call_duration": 900},
        draft={"dictionary": "mine", "ambient_noise_configuration": {"volume": 0.7}},
    )
    async with test_client_factory(user) as client:
        response = await client.get(
            f"/api/v1/workflow/{workflow.id}/configuration-effective"
        )
    assert response.status_code == 200, response.text
    body = response.json()
    # own = the draft, exactly as stored (sparse)
    assert body["own"] == {
        "dictionary": "mine",
        "ambient_noise_configuration": {"volume": 0.7},
    }
    # effective = schema <- org <- draft, leaf by leaf
    assert body["effective"]["max_call_duration"] == 600  # org, not the published 900
    assert body["effective"]["dictionary"] == "mine"  # draft wins
    assert body["effective"]["ambient_noise_configuration"] == {
        "enabled": False,
        "volume": 0.7,
    }
    # base = schema <- org only
    assert body["base"]["max_call_duration"] == 600
    assert body["base"]["dictionary"] == "house"
    assert body["base"]["ambient_noise_configuration"] == {
        "enabled": False,
        "volume": 0.3,
    }
    assert body["warnings"] == []
    assert body["definition_status"] == "draft"


async def test_falls_back_to_released_definition_without_draft(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    workflow = await _workflow(
        db_session, org, user, published={"max_call_duration": 900}, draft=None
    )
    async with test_client_factory(user) as client:
        body = (
            await client.get(f"/api/v1/workflow/{workflow.id}/configuration-effective")
        ).json()
    assert body["own"] == {"max_call_duration": 900}
    assert body["effective"]["max_call_duration"] == 900
    assert body["definition_status"] == "published"


async def test_masks_secrets_in_every_layer(test_client_factory, db_session, org_user):
    org, user = org_user
    workflow = await _workflow(
        db_session,
        org,
        user,
        published={},
        draft={"voicemail_detection": {"enabled": True, "api_key": "vm-1234567890"}},
    )
    async with test_client_factory(user) as client:
        body = (
            await client.get(f"/api/v1/workflow/{workflow.id}/configuration-effective")
        ).json()
    for layer in ("own", "effective"):
        key = body[layer]["voicemail_detection"]["api_key"]
        assert key.endswith("7890") and "*" in key


async def test_cross_tenant_workflow_is_404(
    test_client_factory, db_session, org_user, async_session
):
    org, user = org_user
    other_org = OrganizationModel(provider_id="test-org-other")
    async_session.add(other_org)
    await async_session.flush()
    other_user = UserModel(
        provider_id="test-user-other", selected_organization_id=other_org.id
    )
    async_session.add(other_user)
    await async_session.flush()
    foreign = await _workflow(
        db_session,
        other_org,
        other_user,
        published={"max_call_duration": 900},
        draft=None,
    )
    async with test_client_factory(user) as client:
        response = await client.get(
            f"/api/v1/workflow/{foreign.id}/configuration-effective"
        )
    assert response.status_code == 404
    assert "own" not in response.text


async def test_root_null_in_stored_document_is_absent_from_own(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    workflow = await _workflow(
        db_session,
        org,
        user,
        published={},
        draft={"dictionary": None, "max_call_duration": 400},
    )
    async with test_client_factory(user) as client:
        body = (
            await client.get(f"/api/v1/workflow/{workflow.id}/configuration-effective")
        ).json()
    assert body["own"] == {"max_call_duration": 400}
