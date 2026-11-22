"""VOZ-N0-29 (VOZ-AC-B4-46b): the Cloudonix callbacks are refused, and every route of the closed list refuses a forged request with the
401, the VOZ-AC-B0-28 body and no state change.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import pytest

from api.tests.support.carrier_callbacks import (
    ANSWER_ROUTES,
    ANSWER_URLS,
    NCCO_QUERY,
    PLIVO_TOKEN,
    TRANSFER_ID,
    _answer_url_patches,
    _assert_b028,
    _fallback_form,
    _get_ncco,
    _post_fallback,
    _post_telnyx,
    _post_transfer_result,
    _post_vobiz_xml,
    _route_for_twilio,
    _telnyx,
    _telnyx_body,
)

# ---------------------------------------------------------------------------------------------------------
# Cloudonix: its signing is UNVERIFIED (verify_inbound_signature only compares x-cx-apikey with the bearer token),
# so every deployment refuses its callbacks
# ---------------------------------------------------------------------------------------------------------


CLOUDONIX_ROUTES = [
    ("transfer-result", f"/telephony/cloudonix/transfer-result/{TRANSFER_ID}", "json"),
    ("status-callback", "/telephony/cloudonix/status-callback/999", "form"),
    ("cdr", "/telephony/cloudonix/cdr", "json"),
]


def _post_cloudonix(client, path, kind):
    if kind == "json":
        return client.post(path, json={"domain": "acme.cloudonix.net"})
    return client.post(path, data={"CallSid": "x", "status": "answer"})


@pytest.mark.parametrize(("name", "path", "kind"), CLOUDONIX_ROUTES)
def test_cloudonix_callback_in_a_cell_is_401_and_moves_nothing(
    client, world, monkeypatch, log_lines, name, path, kind
):
    monkeypatch.setenv("CELL_ROLE", "call")
    before = world.snapshot()

    response = _post_cloudonix(client, path, kind)

    _assert_b028(response, code="cloudonix_signing_unverified")
    assert world.snapshot() == before
    assert any("cloudonix_signing_unverified" in line for line in log_lines)


@pytest.mark.parametrize(("name", "path", "kind"), CLOUDONIX_ROUTES)
def test_cloudonix_callback_outside_a_cell_is_401_and_moves_nothing(
    client, world, monkeypatch, log_lines, name, path, kind
):
    # Cloudonix is disabled as a carrier in every deployment until its signing is verified (VOZ-N0-40):
    # outside a cell its callbacks are refused exactly as inside one.
    monkeypatch.delenv("CELL_ROLE", raising=False)
    before = world.snapshot()

    response = _post_cloudonix(client, path, kind)

    _assert_b028(response, code="cloudonix_signing_unverified")
    assert world.snapshot() == before
    assert any("cloudonix_signing_unverified" in line for line in log_lines)


# ---------------------------------------------------------------------------------------------------------
# The closed route list: one forged request per route, the real provider verifier behind it
# ---------------------------------------------------------------------------------------------------------


def _forge_fallback(client, world):
    _route_for_twilio(world)
    with patch(
        "api.services.telephony.callback_auth.get_telephony_provider_by_id",
        new=AsyncMock(return_value=world.provider),
    ):
        return _post_fallback(client, _fallback_form(), signature="forged")


def _forge_telnyx_transfer_result(client, world):
    world.provider = _telnyx()
    return _post_telnyx(client, _telnyx_body(), {"content-type": "application/json"})


def _forge_cloudonix(path, kind):
    return lambda client, world: _post_cloudonix(client, path, kind)


def _forge_answer_url(route):
    """The answer route, unsigned, for the run's own consistent query."""

    def forge(client, world):
        if route == "vobiz-xml":
            return _post_vobiz_xml(client, world, headers={})[0]
        if route == "ncco":
            return _get_ncco(client, world, headers={})[0]
        spec = ANSWER_URLS[route]
        world.provider = spec["provider"](PLIVO_TOKEN)
        world.provider.get_webhook_response = AsyncMock(return_value="<Response/>")
        db, route_lookup, auth_lookup = _answer_url_patches(
            spec["module"], world, world.provider_for_run
        )
        with db, route_lookup, auth_lookup:
            return client.post(
                f"/telephony/{route}?{urlencode(NCCO_QUERY)}", data=spec["form"]
            )

    return forge


# The legacy POST /inbound/{workflow_id} is covered refusal by refusal in
# test_legacy_inbound_refused_before_authentication_is_401_and_creates_no_run.
CLOSED_ROUTES = {
    "POST /inbound/fallback": _forge_fallback,
    "POST /transfer-result/{transfer_id}": lambda client, world: _post_transfer_result(
        client, signed=False
    ),
    "POST /telnyx/transfer-result/{transfer_id}": _forge_telnyx_transfer_result,
    **{
        f"POST /cloudonix/{name}": _forge_cloudonix(path, kind)
        for name, path, kind in CLOUDONIX_ROUTES
    },
    **{f"answer URL {route}": _forge_answer_url(route) for route in ANSWER_ROUTES},
}


@pytest.mark.parametrize("route", sorted(CLOSED_ROUTES))
def test_closed_route_refuses_a_forged_request_with_the_401_and_moves_nothing(
    client, world, monkeypatch, route
):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    before = world.snapshot()

    response = CLOSED_ROUTES[route](client, world)

    _assert_b028(response)
    assert world.snapshot() == before
