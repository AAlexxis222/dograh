import pytest

from api.db.models import OrganizationModel, UserModel
from api.enums import OrganizationConfigurationKey


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
    _, user = org_user
    async with test_client_factory(user) as client:
        before = await client.get(
            "/api/v1/organizations/workflow-configuration-effective-defaults"
        )
        assert before.json()["workflow_configurations"]["max_call_duration"] == 300
        assert before.json()["warnings"] == []
        await client.put(
            "/api/v1/organizations/workflow-configuration-defaults",
            json={"max_call_duration": 600},
        )
        after = await client.get(
            "/api/v1/organizations/workflow-configuration-effective-defaults"
        )
    assert after.json()["workflow_configurations"]["max_call_duration"] == 600
    assert set(after.json()) >= {
        "llm",
        "tts",
        "stt",
        "default_call_dispositions",
        "widget_text_defaults",
    }


@pytest.mark.parametrize(
    "body",
    [
        {"voicemail_detection": {"enabled": True, "api_key": "sk-secret"}},
        # Unknown section: the schema accepts extra keys, so the rejection may
        # not be limited to the paths the secrets registry knows about.
        {"my_integration": {"api_key": "sk-supersecret"}},
        {"model_overrides": {"llm": {"provider": "openai", "api_key": "sk"}}},
        {
            "external_pbx_field_mappings": [
                {"context_path": "a", "destination_field": "b"}
            ]
        },
        {"external_pbx_lead_headers": ["first_name"]},
        {"model_configuration_v2_override": {}},
        {"max_call_duration": 99999},
    ],
)
async def test_org_defaults_reject_secrets_pbx_and_model_keys(
    test_client_factory, org_user, body
):
    _, user = org_user
    async with test_client_factory(user) as client:
        response = await client.put(
            "/api/v1/organizations/workflow-configuration-defaults", json=body
        )
    assert response.status_code == 422, response.text


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
