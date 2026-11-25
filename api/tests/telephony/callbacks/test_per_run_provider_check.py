"""VOZ-N0-29 (VOZ-AC-B4-46b), RUL-r1-27: a per-run carrier callback refuses a run whose configured provider is not
the route's own provider, and a run of a disabled carrier, before verifying or processing anything.

The sharp case is a run on ARI: ARIProvider.verify_inbound_signature returns True for any request, so a route that
verifies with the run's provider without comparing its name accepts an unsigned request and moves the run. Each
route refuses with the answer it already gives to an invalid signature, and the state stays as it was.

The fakes are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import constants
from api.routes.telephony import router
from api.services.telephony.carrier_policy import CarrierDisabledError
from api.services.telephony.providers.ari.provider import ARIProvider
from api.services.telephony.providers.exotel.provider import ExotelProvider
from api.services.telephony.providers.plivo.provider import PlivoProvider
from api.services.telephony.providers.telnyx.provider import TelnyxProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.vobiz.provider import VobizProvider
from api.services.telephony.providers.vonage.provider import VonageProvider
from api.tests.support.carrier_callbacks import (
    PUBLIC_BACKEND,
    RUN_ID,
    TRANSFER_ID,
    WORKFLOW_ID,
)

ROUTES_PKG = "api.services.telephony.providers"

INVALID_SIGNATURE = {"detail": "Invalid webhook signature"}

PLIVO_ERROR = {"status": "error", "reason": "invalid_signature"}

PLIVO_HANGUP_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
)

# What an attacker sends: an ARI event that ARIProvider.parse_status_callback maps to COMPLETED, bound to the run's
# own call id so the Exotel CallSid binding would not stop it either.
ARI_EVENT = {"type": "StasisEnd", "channel": {"id": "CA-original"}}

ARI_FORM = {"type": "StasisEnd", "CallUUID": "CA-original", "CallSid": "CA-original"}

# route -> (provider of the route, method and path, body kind and body, status and body of its refusal)
ROUTES = {
    "twilio status-callback": (
        TwilioProvider,
        f"/telephony/twilio/status-callback/{RUN_ID}",
        ("data", ARI_FORM),
        (401, INVALID_SIGNATURE),
    ),
    "plivo hangup-callback": (
        PlivoProvider,
        f"/telephony/plivo/hangup-callback/{RUN_ID}",
        ("data", ARI_FORM),
        (200, PLIVO_ERROR),
    ),
    "plivo ring-callback": (
        PlivoProvider,
        f"/telephony/plivo/ring-callback/{RUN_ID}",
        ("data", ARI_FORM),
        (200, PLIVO_ERROR),
    ),
    "plivo transfer-xml": (
        PlivoProvider,
        f"/telephony/plivo/transfer-xml/conf-1/{TRANSFER_ID}",
        ("data", {"CallUUID": "CA-chosen"}),
        (200, PLIVO_HANGUP_XML),
    ),
    "plivo transfer-result": (
        PlivoProvider,
        f"/telephony/plivo/transfer-result/{TRANSFER_ID}",
        ("data", {"CallUUID": "CA-chosen", "HangupCause": "USER_BUSY"}),
        (200, PLIVO_ERROR),
    ),
    "vobiz hangup-callback": (
        VobizProvider,
        f"/telephony/vobiz/hangup-callback/{RUN_ID}",
        ("data", ARI_FORM),
        (403, INVALID_SIGNATURE),
    ),
    "vobiz ring-callback": (
        VobizProvider,
        f"/telephony/vobiz/ring-callback/{RUN_ID}",
        ("data", ARI_FORM),
        (403, INVALID_SIGNATURE),
    ),
    "vobiz hangup-callback by workflow": (
        VobizProvider,
        f"/telephony/vobiz/hangup-callback/workflow/{WORKFLOW_ID}",
        ("data", ARI_FORM),
        (403, INVALID_SIGNATURE),
    ),
    "vonage events": (
        VonageProvider,
        f"/telephony/vonage/events/{RUN_ID}",
        ("json", ARI_EVENT),
        (401, INVALID_SIGNATURE),
    ),
    "vonage events by call id": (
        VonageProvider,
        "/telephony/vonage/events",
        ("json", {**ARI_EVENT, "uuid": "CA-original"}),
        (401, INVALID_SIGNATURE),
    ),
    "telnyx events": (
        TelnyxProvider,
        f"/telephony/telnyx/events/{RUN_ID}",
        ("json", {**ARI_EVENT, "data": {}}),
        (404, {"detail": "Workflow run not found"}),
    ),
    "exotel status-callback": (
        ExotelProvider,
        f"/telephony/exotel/status-callback/{RUN_ID}",
        ("json", ARI_EVENT),
        (401, INVALID_SIGNATURE),
    ),
}

MODULES = ["twilio", "plivo", "vobiz", "vonage", "telnyx", "exotel"]


@pytest.fixture
def routes(world, monkeypatch):
    """Every per-run route module resolving through the world: its db, its provider lookup and the processor."""
    # No secret, the default outside a cell: the Telnyx events capability token is not checked.
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", "")
    world.db.get_workflow_run_by_call_id = AsyncMock(return_value=world.db.run)
    record = AsyncMock(side_effect=lambda run_id, update: world.status_updates.append(run_id))
    get_manager = AsyncMock(return_value=world.manager)
    patches = [
        patch(
            "api.services.telephony.status_processor._process_status_update", record
        ),
        patch(
            f"{ROUTES_PKG}.vobiz.routes.get_backend_endpoints",
            new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
        ),
        patch(f"{ROUTES_PKG}.twilio.routes.get_call_transfer_manager", get_manager),
        patch(f"{ROUTES_PKG}.plivo.routes.get_call_transfer_manager", get_manager),
    ]
    for name in MODULES:
        module = f"{ROUTES_PKG}.{name}.routes"
        patches.append(patch(f"{module}.db_client", world.db))
        patches.append(
            patch(
                f"{module}.get_telephony_provider_for_run",
                side_effect=world.provider_for_run,
            )
        )
        if name != "vonage":  # vonage imports the processor inside its handler
            patches.append(patch(f"{module}._process_status_update", record))
    for p in patches:
        p.start()
    app = FastAPI()
    app.include_router(router)
    yield TestClient(app, raise_server_exceptions=False)
    for p in reversed(patches):
        p.stop()


def _post(client, route):
    _, path, (kind, body), _ = ROUTES[route]
    return client.post(path, **{kind: body})


def _ari_provider(name: str | None = None) -> ARIProvider:
    provider = ARIProvider({})
    if name:
        provider.PROVIDER_NAME = name
    provider.verify_inbound_signature = AsyncMock(return_value=True)
    return provider


def _is_refusal(response, route) -> bool:
    status, body = ROUTES[route][3]
    if response.status_code != status:
        return False
    return response.text == body if isinstance(body, str) else response.json() == body


@pytest.mark.parametrize("route", ROUTES)
def test_unsigned_callback_against_a_run_of_another_provider_is_refused_and_moves_nothing(
    routes, world, route
):
    world.provider = _ari_provider()
    before = world.snapshot()

    response = _post(routes, route)

    assert _is_refusal(response, route), (response.status_code, response.text)
    world.provider.verify_inbound_signature.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize("route", ROUTES)
def test_callback_against_a_run_of_a_disabled_carrier_is_refused_not_a_500(
    routes, world, route
):
    world.provider_error = CarrierDisabledError("cloudonix")
    before = world.snapshot()

    response = _post(routes, route)

    assert _is_refusal(response, route), (response.status_code, response.text)
    assert world.snapshot() == before


@pytest.mark.parametrize("route", ROUTES)
def test_verified_callback_against_a_run_of_the_route_provider_passes_the_check(
    routes, world, route
):
    # Control: the same request against a run on the route's own provider (verifier accepting) is not refused.
    world.provider = _ari_provider(ROUTES[route][0].PROVIDER_NAME)

    response = _post(routes, route)

    assert response.status_code == 200, response.text
    assert not _is_refusal(response, route)
    world.provider.verify_inbound_signature.assert_awaited_once()
