"""VOZ-N0-29 (VOZ-AC-B4-46b): the legacy POST /inbound/{workflow_id} refuses before authenticating with the 401 and no run, and the
Vonage events webhook refuses a bad signature.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.tests.support.carrier_callbacks import (
    CALLER,
    ORG_ID,
    RUN_ID,
    UNAVAILABLE_ES,
    WORKFLOW_ID,
    _assert_b028,
    _route_for_twilio,
)

# ---------------------------------------------------------------------------------------------------------
# /inbound/{workflow_id} (legacy) and Vonage events
# ---------------------------------------------------------------------------------------------------------


LEGACY_REFUSALS = [
    "workflow_not_found",
    "no_configuration_of_the_carrier",
    "account_not_matched",
    "phone_number_not_configured",
    "signature_did_not_verify",
    "verifier_raised",
    "provider_cannot_be_built",
]


@pytest.mark.parametrize("refusal", LEGACY_REFUSALS)
def test_legacy_inbound_refused_before_authentication_is_401_and_creates_no_run(
    client, world, refusal
):
    # An unsigned request must not tell an existing workflow or number from a missing one: every refusal
    # before the request is authenticated is the same 401, never a TwiML document.
    normalized = SimpleNamespace(
        provider="twilio",
        direction="inbound",
        to_number="+34911000111",
        from_number=CALLER,
        to_country="ES",
        from_country="ES",
        account_id="AC123",
        call_id="CA-legacy",
        raw_data={},
    )
    workflow_id = 404 if refusal == "workflow_not_found" else WORKFLOW_ID
    world.db.get_workflow_by_id = AsyncMock(
        side_effect=lambda wid: (
            SimpleNamespace(id=WORKFLOW_ID, organization_id=ORG_ID, user_id=1)
            if wid == WORKFLOW_ID
            else None
        )
    )
    if refusal in ("no_configuration_of_the_carrier", "account_not_matched"):
        configs = [] if refusal == "no_configuration_of_the_carrier" else [object()]
        world.db.list_telephony_configurations_by_provider = AsyncMock(
            return_value=configs
        )
    elif refusal == "phone_number_not_configured":
        world.db.list_telephony_configurations_by_provider = AsyncMock(
            return_value=[object()]
        )
    else:
        _route_for_twilio(world)
    verify = AsyncMock(
        side_effect=RuntimeError("verifier broke")
        if refusal == "verifier_raised"
        else None,
        return_value=False,
    )
    provider = SimpleNamespace(
        verify_inbound_signature=verify, start_inbound_stream=AsyncMock()
    )
    build = AsyncMock(
        side_effect=ValueError("config parked")
        if refusal == "provider_cannot_be_built"
        else None,
        return_value=provider,
    )
    created = AsyncMock()
    before = world.snapshot()
    with (
        patch("api.routes.telephony.normalize_webhook_data", return_value=normalized),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=TwilioProvider),
        ),
        patch("api.routes.telephony.get_telephony_provider_by_id", new=build),
        patch(
            "api.services.telephony.factory.find_telephony_config_for_inbound",
            new=AsyncMock(
                return_value=None if refusal == "account_not_matched" else (5, None)
            ),
        ),
        patch(
            "api.routes.telephony._verify_organization_phone_number",
            new=AsyncMock(return_value=None),
        ),
        patch("api.routes.telephony._create_inbound_workflow_run", new=created),
        patch("api.routes.telephony.call_concurrency") as concurrency,
    ):
        concurrency.acquire_org_slot = AsyncMock()
        response = client.post(
            f"/telephony/inbound/{workflow_id}",
            data={"CallSid": "CA-legacy", "From": CALLER},
        )

    detail = _assert_b028(response, code="carrier_callback_unauthenticated")
    assert detail["where"] == "POST /telephony/inbound/{workflow_id}"
    assert "<Response" not in response.text and UNAVAILABLE_ES not in response.text
    provider.start_inbound_stream.assert_not_awaited()
    created.assert_not_awaited()
    concurrency.acquire_org_slot.assert_not_awaited()
    assert world.snapshot() == before
    if refusal in ("signature_did_not_verify", "verifier_raised"):
        verify.assert_awaited_once()


def test_vonage_event_bad_signature_is_401_and_processes_nothing(client, world):
    world.provider = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=False),
        parse_status_callback=Mock(),
    )
    before = world.snapshot()

    response = client.post(
        f"/telephony/vonage/events/{RUN_ID}", json={"uuid": "u1", "status": "answered"}
    )

    assert response.status_code == 401
    world.provider.parse_status_callback.assert_not_called()
    assert world.snapshot() == before
