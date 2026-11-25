"""VOZ-N0-29 (RUL-r1-6): Cloudonix is disabled as a carrier in every deployment until its signing is verified.

Every call start resolves its provider before it takes a concurrency slot or creates a run, so the refusal lives
in the provider lookups and in the agent-stream route's provider-name check. "Refused" means: the named code
``cloudonix_signing_unverified`` reaches whoever asked (the caller hears the Spanish unavailable sentence), and no
slot, run or campaign claim is taken. Other carriers resolve exactly as before.
"""

import secrets
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.errors.telephony_errors import UNAVAILABLE_MESSAGE_ES
from api.services.telephony.base import ProviderSyncResult
from api.services.telephony.providers.cloudonix.provider import CloudonixProvider

CODE = "cloudonix_signing_unverified"
ORG_ID = 11
WORKFLOW_ID = 33
CONFIG_ID = 5
# Generated per run: the tests need values, not literals that look like credentials.
API_KEY = secrets.token_hex(8)
CLOUDONIX_CONFIG = {
    "provider": "cloudonix",
    "bearer_token": API_KEY,
    "domain_id": "acme.cloudonix.net",
    "from_numbers": ["+34911000111"],
}
TWILIO_CONFIG = {
    "provider": "twilio",
    "account_sid": "AC123",
    "auth_token": secrets.token_hex(16),
    "from_numbers": ["+34911000111"],
}
WORKFLOW = SimpleNamespace(id=WORKFLOW_ID, organization_id=ORG_ID, user_id=1)


def _loaded(config: dict):
    """Patch both config loaders of the factory, so every lookup builds ``config``."""
    return (
        patch(
            "api.services.telephony.factory.load_telephony_config_by_id",
            new=AsyncMock(return_value=dict(config)),
        ),
        patch(
            "api.services.telephony.factory.load_default_telephony_config",
            new=AsyncMock(return_value=dict(config)),
        ),
    )


@pytest.fixture
def cloudonix_config():
    by_id, default = _loaded(CLOUDONIX_CONFIG)
    with by_id, default:
        yield


def _slots() -> Mock:
    """A concurrency service that records a slot request and refuses it, so a test sees whether one was asked."""
    concurrency = Mock()
    concurrency.acquire_org_slot = AsyncMock(
        side_effect=RuntimeError("test: a concurrency slot was requested")
    )
    return concurrency


# ---------------------------------------------------------------------------------------------------------
# The provider lookups
# ---------------------------------------------------------------------------------------------------------


async def _by_id():
    from api.services.telephony.factory import get_telephony_provider_by_id

    return await get_telephony_provider_by_id(CONFIG_ID, ORG_ID)


async def _default():
    from api.services.telephony.factory import get_default_telephony_provider

    return await get_default_telephony_provider(ORG_ID)


async def _for_stamped_run():
    from api.services.telephony.factory import get_telephony_provider_for_run

    run = SimpleNamespace(initial_context={"telephony_configuration_id": CONFIG_ID})
    return await get_telephony_provider_for_run(run, ORG_ID)


async def _for_legacy_run():
    from api.services.telephony.factory import get_telephony_provider_for_run

    return await get_telephony_provider_for_run(
        SimpleNamespace(initial_context={}), ORG_ID
    )


async def _for_inbound():
    from api.services.telephony.factory import get_telephony_provider_for_inbound

    with patch(
        "api.services.telephony.factory.find_telephony_config_for_inbound",
        new=AsyncMock(return_value=(CONFIG_ID, dict(CLOUDONIX_CONFIG))),
    ):
        return await get_telephony_provider_for_inbound(
            ORG_ID, "cloudonix", "acme.cloudonix.net"
        )


LOOKUPS = [_by_id, _default, _for_stamped_run, _for_legacy_run, _for_inbound]


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", LOOKUPS, ids=lambda f: f.__name__)
async def test_every_call_path_lookup_refuses_a_cloudonix_configuration(
    cloudonix_config, lookup
):
    with pytest.raises(Exception, match=CODE) as refused:
        await lookup()

    # Not a ValueError: the routes map ValueError to telephony_configuration_not_found.
    assert not isinstance(refused.value, ValueError)


@pytest.mark.asyncio
async def test_lookups_of_another_carrier_are_unchanged():
    by_id, default = _loaded(TWILIO_CONFIG)
    with by_id, default:
        assert (await _by_id()).PROVIDER_NAME == "twilio"
        assert (await _default()).PROVIDER_NAME == "twilio"


# ---------------------------------------------------------------------------------------------------------
# Inbound: /inbound/run and the legacy /inbound/{workflow_id}
# ---------------------------------------------------------------------------------------------------------


def _inbound_db() -> Mock:
    db = Mock()
    route = (
        SimpleNamespace(id=CONFIG_ID, organization_id=ORG_ID, name="Cloudonix"),
        SimpleNamespace(id=9, address="+34911000111", inbound_workflow_id=WORKFLOW_ID),
    )
    db.find_inbound_route_by_account = AsyncMock(return_value=route)
    db.get_workflow = AsyncMock(return_value=WORKFLOW)
    db.get_workflow_by_id = AsyncMock(return_value=WORKFLOW)
    return db


NORMALIZED = SimpleNamespace(
    provider="cloudonix",
    direction="inbound",
    to_number="+34911000111",
    from_number="+34612345678",
    to_country="ES",
    from_country="ES",
    account_id="acme.cloudonix.net",
    call_id="session-1",
    raw_data={},
    sip_headers=None,
    sip_call_id=None,
)


@pytest.mark.parametrize(
    "path", ["/telephony/inbound/run", f"/telephony/inbound/{WORKFLOW_ID}"]
)
def test_inbound_cloudonix_call_hears_unavailable_and_takes_no_slot(
    cloudonix_config, path
):
    from api.routes.telephony import router

    app = FastAPI()
    app.include_router(router)
    concurrency = _slots()
    create_run = AsyncMock()
    with (
        patch("api.routes.telephony.db_client", _inbound_db()),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=CloudonixProvider),
        ),
        patch("api.routes.telephony.normalize_webhook_data", return_value=NORMALIZED),
        patch("api.routes.telephony.call_concurrency", concurrency),
        patch("api.routes.telephony._create_inbound_workflow_run", new=create_run),
    ):
        # The API key a genuine Cloudonix webhook carries: without the refusal the call would start.
        response = TestClient(app).post(
            path, data={"Session": "session-1"}, headers={"x-cx-apikey": API_KEY}
        )

    assert UNAVAILABLE_MESSAGE_ES in response.text
    concurrency.acquire_org_slot.assert_not_awaited()
    create_run.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------
# Outbound: test call, public agent trigger, campaign batch
# ---------------------------------------------------------------------------------------------------------


def test_test_call_on_cloudonix_is_400_with_the_code_and_takes_no_slot(
    cloudonix_config,
):
    from api.routes.telephony import router
    from api.services.auth.depends import get_user

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_user] = lambda: SimpleNamespace(
        id=1, selected_organization_id=ORG_ID
    )
    concurrency = _slots()
    db = Mock()
    db.get_workflow = AsyncMock(return_value=None)
    with (
        patch(
            "api.services.organization_preferences.get_organization_preferences",
            new=AsyncMock(return_value=SimpleNamespace(test_phone_number=None)),
        ),
        patch(
            "api.routes.telephony.resolve_outbound_configuration_id",
            new=AsyncMock(return_value=CONFIG_ID),
        ),
        patch("api.routes.telephony.db_client", db),
        patch("api.routes.telephony.call_concurrency", concurrency),
    ):
        response = TestClient(app).post(
            "/telephony/initiate-call",
            json={"workflow_id": WORKFLOW_ID, "phone_number": "+34600000001"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == CODE
    concurrency.acquire_org_slot.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_agent_trigger_on_cloudonix_is_400_with_the_code_and_takes_no_slot(
    cloudonix_config,
):
    from api.routes.public_agent import (
        ResolvedAgentTarget,
        TriggerCallRequest,
        _execute_resolved_target,
    )

    concurrency = _slots()
    target = ResolvedAgentTarget(
        workflow=SimpleNamespace(user_id=1, workflow_uuid="agent-uuid"),
        organization_id=ORG_ID,
        identifier_type="workflow_uuid",
        identifier_value="agent-uuid",
    )
    with (
        patch(
            "api.routes.public_agent.resolve_outbound_configuration_id",
            new=AsyncMock(return_value=CONFIG_ID),
        ),
        patch("api.routes.public_agent.call_concurrency", concurrency),
        pytest.raises(HTTPException) as refused,
    ):
        await _execute_resolved_target(
            target,
            TriggerCallRequest(phone_number="+34600000001"),
            use_draft=False,
            api_key_id=None,
            api_key_created_by=None,
        )

    assert refused.value.status_code == 400
    assert refused.value.detail == CODE
    concurrency.acquire_org_slot.assert_not_awaited()


@pytest.mark.asyncio
async def test_campaign_batch_on_cloudonix_fails_with_the_code_before_claiming(
    cloudonix_config,
):
    from api.services.campaign.campaign_call_dispatcher import CampaignCallDispatcher

    db = Mock()
    db.get_campaign_by_id = AsyncMock(
        return_value=SimpleNamespace(
            id=1,
            state="running",
            organization_id=ORG_ID,
            telephony_configuration_id=CONFIG_ID,
        )
    )
    db.claim_queued_runs_for_processing = AsyncMock(return_value=[])
    with (
        patch("api.services.campaign.campaign_call_dispatcher.db_client", db),
        patch(
            "api.services.telephony.outbound_readiness.resolve_outbound_configuration_id",
            new=AsyncMock(return_value=CONFIG_ID),
        ),
        # The campaign task records str(error) as the campaign's failure reason (campaign_tasks.py).
        pytest.raises(Exception, match=CODE),
    ):
        await CampaignCallDispatcher().process_batch(campaign_id=1)

    db.claim_queued_runs_for_processing.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------
# Agent stream: refused on the provider name, before the workflow lookup and the slot
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_stream_on_cloudonix_closes_1008_with_the_code_and_takes_no_slot():
    from api.routes.agent_stream import agent_stream_websocket

    websocket = SimpleNamespace(query_params={}, accept=AsyncMock(), close=AsyncMock())
    concurrency = _slots()
    db = Mock()
    db.get_workflow_by_uuid_unscoped = AsyncMock(return_value=WORKFLOW)
    db.create_workflow_run = AsyncMock()
    with (
        patch("api.routes.agent_stream.db_client", db),
        patch("api.routes.agent_stream.call_concurrency", concurrency),
    ):
        await agent_stream_websocket(websocket, "cloudonix", "agent-uuid")

    websocket.close.assert_awaited_once()
    assert websocket.close.await_args.kwargs["code"] == 1008
    assert CODE in websocket.close.await_args.kwargs["reason"]
    db.get_workflow_by_uuid_unscoped.assert_not_awaited()
    concurrency.acquire_org_slot.assert_not_awaited()
    db.create_workflow_run.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------
# Number validation and inbound sync resolve the provider, so they are refused with the code too
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloudonix_number_validation_is_400_with_the_code(cloudonix_config):
    from api.routes.organization import _ensure_provider_phone_number

    ok = AsyncMock(return_value=ProviderSyncResult(ok=True))
    with (
        patch.object(CloudonixProvider, "provision_phone_number", ok),
        patch.object(CloudonixProvider, "validate_phone_number", ok),
        pytest.raises(HTTPException) as refused,
    ):
        await _ensure_provider_phone_number(CONFIG_ID, ORG_ID, "+34911000111")

    assert refused.value.status_code == 400
    assert refused.value.detail == CODE
    ok.assert_not_awaited()


@pytest.mark.asyncio
async def test_cloudonix_inbound_sync_reports_the_code(cloudonix_config):
    from api.routes.organization import _sync_inbound_for_phone_number

    ok = AsyncMock(return_value=ProviderSyncResult(ok=True))
    with patch.object(CloudonixProvider, "configure_inbound", ok):
        status = await _sync_inbound_for_phone_number(
            CONFIG_ID, ORG_ID, "+34911000111", attach=False
        )

    assert status.ok is False
    assert CODE in status.message
    ok.assert_not_awaited()

# ---------------------------------------------------------------------------------------------------------
# Deleting a phone number of a disabled carrier (RUL-r1-31 a): the row goes, the carrier is never touched
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deleting_a_cloudonix_phone_number_deletes_the_row_and_reports_the_code(
    cloudonix_config,
):
    from api.routes.organization import delete_phone_number

    configure_inbound = AsyncMock(return_value=ProviderSyncResult(ok=True))
    delete_row = AsyncMock(return_value=True)
    with (
        patch(
            "api.routes.organization._ensure_config_belongs_to_org",
            new=AsyncMock(
                return_value=SimpleNamespace(id=CONFIG_ID, provider="cloudonix")
            ),
        ),
        patch(
            "api.routes.organization.db_client.get_phone_number_for_config",
            new=AsyncMock(return_value=SimpleNamespace(address="+34911000111")),
        ),
        patch("api.routes.organization.db_client.delete_phone_number", delete_row),
        patch.object(CloudonixProvider, "configure_inbound", configure_inbound),
    ):
        result = await delete_phone_number(
            config_id=CONFIG_ID,
            phone_number_id=9,
            user=SimpleNamespace(selected_organization_id=ORG_ID),
        )

    assert result["message"] == "Phone number deleted"
    assert result["provider_sync"]["ok"] is False
    assert CODE in result["provider_sync"]["message"]
    delete_row.assert_awaited_once_with(9, CONFIG_ID)
    configure_inbound.assert_not_awaited()
