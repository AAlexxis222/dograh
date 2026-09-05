import pytest

from api.db.models import OrganizationModel, UserModel
from api.enums import OrganizationConfigurationKey
from api.services.configuration.cascade import (
    load_effective_workflow_configurations,
)

GRAPH = {
    "nodes": [{"id": "1", "type": "startCall", "data": {"name": "S", "prompt": "p"}}],
    "edges": [],
}


@pytest.fixture
async def org_user(async_session):
    org = OrganizationModel(provider_id="test-org-wf-defaults")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id="test-user-wf-defaults", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    return org, user


async def test_put_stores_only_own_keys(test_client_factory, db_session, org_user):
    org, user = org_user
    async with test_client_factory(user) as client:
        response = await client.put(
            "/api/v1/organizations/workflow-configuration-defaults",
            json={
                "max_call_duration": 600,
                "ambient_noise_configuration": {"enabled": True},
            },
        )
    assert response.status_code == 200, response.text
    stored = await db_session.get_configuration_value(
        org.id, OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value
    )
    assert stored == {
        "max_call_duration": 600,
        "ambient_noise_configuration": {"enabled": True},
    }
    assert response.json() == {"workflow_configurations": stored}


async def test_org_base_change_propagates_to_existing_workflow(
    test_client_factory, db_session, org_user
):
    """The point of the base: a workflow published before the change, whose own
    document never mentions the key, resolves to the new value."""
    org, user = org_user
    workflow = await db_session.create_workflow(
        name="W", workflow_definition=GRAPH, user_id=user.id, organization_id=org.id
    )
    await db_session.save_workflow_draft(
        workflow.id, workflow_configurations={"dictionary": "mine"}
    )
    await db_session.publish_workflow_draft(workflow.id)
    workflow = await db_session.get_workflow(workflow.id, organization_id=org.id)

    async def _workflow_effective():
        resolved = await load_effective_workflow_configurations(
            db_session,
            organization_id=org.id,
            definition_id=workflow.released_definition_id,
        )
        return resolved.effective

    async with test_client_factory(user) as client:
        before = await client.get(
            "/api/v1/organizations/workflow-configuration-effective-defaults"
        )
        assert before.json()["workflow_configurations"]["max_call_duration"] == 300
        assert before.json()["warnings"] == []
        assert (await _workflow_effective())["max_call_duration"] == 300
        await client.put(
            "/api/v1/organizations/workflow-configuration-defaults",
            json={"max_call_duration": 600},
        )
        after = await client.get(
            "/api/v1/organizations/workflow-configuration-effective-defaults"
        )
    assert after.json()["workflow_configurations"]["max_call_duration"] == 600
    workflow_effective = await _workflow_effective()
    assert workflow_effective["max_call_duration"] == 600
    assert workflow_effective["dictionary"] == "mine"  # its own key survives
    assert set(after.json()) >= {
        "llm",
        "tts",
        "stt",
        "default_call_dispositions",
        "widget_text_defaults",
    }


# Which family a body is rejected for; each case pins one so a rejection for
# the wrong reason still fails.
SECRET = "secrets are not allowed at organization level"
KEY = "keys not allowed at organization level"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"voicemail_detection": {"enabled": True, "api_key": "sk-secret"}}, SECRET),
        # Unknown section: the schema accepts extra keys, so the rejection may
        # not be limited to the paths the secrets registry knows about.
        ({"my_integration": {"api_key": "sk-supersecret"}}, SECRET),
        ({"model_overrides": {"llm": {"provider": "openai", "api_key": "sk"}}}, KEY),
        (
            {
                "external_pbx_field_mappings": [
                    {"context_path": "a", "destination_field": "b"}
                ]
            },
            KEY,
        ),
        ({"external_pbx_lead_headers": ["first_name"]}, KEY),
        ({"model_configuration_v2_override": {}}, KEY),
        ({"max_call_duration": 99999}, "max_call_duration"),
        # A credential is rejected however its key is spelled, and under any of
        # the names one is usually given.
        ({"my_integration": {"apiKey": "sk-secret"}}, SECRET),
        ({"my_integration": {"API_KEY": "sk-secret"}}, SECRET),
        ({"my_integration": {"Api-Key": "sk-secret"}}, SECRET),
        ({"my_integration": {"api_key ": "sk-secret"}}, SECRET),
        ({"my_integration": {"password": "hunter2"}}, SECRET),
        ({"my_integration": {"token": "t-secret"}}, SECRET),
        ({"my_integration": {"secret": "s-secret"}}, SECRET),
        ({"my_integration": {"apiKeys": ["sk-secret"]}}, SECRET),
    ],
)
async def test_org_defaults_reject_secrets_pbx_and_model_keys(
    test_client_factory, org_user, body, expected
):
    _, user = org_user
    async with test_client_factory(user) as client:
        response = await client.put(
            "/api/v1/organizations/workflow-configuration-defaults", json=body
        )
    assert response.status_code == 422, response.text
    assert expected in response.text


async def test_get_masks_stored_secret_defensively(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {"voicemail_detection": {"api_key": "vm-1234567890"}},
    )
    async with test_client_factory(user) as client:
        response = await client.get(
            "/api/v1/organizations/workflow-configuration-defaults"
        )
        effective = await client.get(
            "/api/v1/organizations/workflow-configuration-effective-defaults"
        )
    # Both reads of the stored base mask it: the effective document inlines the
    # organization layer, so an unmasked one there leaks just as much.
    for payload in (response.json(), effective.json()):
        api_key = payload["workflow_configurations"]["voicemail_detection"]["api_key"]
        assert api_key.endswith("7890")
        assert "*" in api_key


async def test_user_defaults_endpoint_unchanged_and_anonymous(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {"max_call_duration": 600},
    )
    from httpx import ASGITransport, AsyncClient

    from api.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as anonymous:
        anon = await anonymous.get("/api/v1/user/configurations/defaults")
    async with test_client_factory(user) as client:
        authed = await client.get("/api/v1/user/configurations/defaults")
    assert anon.status_code == authed.status_code == 200
    assert anon.json() == authed.json()
    assert anon.json()["workflow_configurations"]["max_call_duration"] == 300
    assert "warnings" not in anon.json()


async def test_disposition_codes_include_org_base_catalog(
    test_client_factory, db_session, org_user
):
    org, user = org_user
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {"call_dispositions": [{"code": "house_code", "description": "x"}]},
    )
    async with test_client_factory(user) as client:
        response = await client.get("/api/v1/organizations/disposition-codes")
    assert "house_code" in response.json()["codes"]


async def test_org_defaults_reject_an_oversized_document(test_client_factory, org_user):
    """The base is one configuration row, not a payload store."""
    _, user = org_user
    async with test_client_factory(user) as client:
        response = await client.put(
            "/api/v1/organizations/workflow-configuration-defaults",
            json={"dictionary": "x" * 70000},
        )
    assert response.status_code == 422, response.text


async def test_effective_defaults_report_422_when_the_stored_document_is_invalid(
    test_client_factory, db_session, org_user
):
    """A document written around the PUT (or by an older schema) must not turn
    every read of the effective base into a 500."""
    org, user = org_user
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.WORKFLOW_CONFIGURATION_DEFAULTS.value,
        {"max_call_duration": "not-a-number"},
    )
    async with test_client_factory(user) as client:
        response = await client.get(
            "/api/v1/organizations/workflow-configuration-effective-defaults"
        )
    assert response.status_code == 422, response.text
    assert "no longer validate" in response.json()["detail"]
