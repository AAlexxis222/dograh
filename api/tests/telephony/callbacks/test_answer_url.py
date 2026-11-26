"""VOZ-N0-29 (VOZ-AC-B4-46b): the run-bound answer-URL token: a carrier signature binds no run, so the four answer handlers check the token.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

import asyncio
import secrets
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import pytest

from api import constants
from api.services.telephony import ws_auth
from api.tests.support.carrier_callbacks import (
    ANSWER_ROUTES,
    ANSWER_URLS,
    BASE,
    NCCO_QUERY,
    ORG_ID,
    PLIVO_TOKEN,
    RUN_ID,
    WORKFLOW_ID,
    _answer_url_patches,
    _assert_b028,
    _get_ncco,
    _post_vobiz_xml,
    _twilio,
)

# ---------------------------------------------------------------------------------------------------------
# Answer-URL token: a carrier signature binds no run (Vobiz signs the bare URL, the Vonage GET JWT has no
# payload hash and no freshness), so a captured one replays against every run of its org with a consistent
# query. The answer URL carries a run-bound token that the four answer handlers check.
# ---------------------------------------------------------------------------------------------------------


OTHER_RUN = 8


@pytest.fixture
def answer_secret(monkeypatch):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", secrets.token_hex(16))
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_ENFORCE", True)
    monkeypatch.delenv("CELL_ROLE", raising=False)


def _call_answer_url(client, world, route, *, t=None):
    """The answer route, validly signed by the run's own org, for the run's own consistent query."""
    query = {**NCCO_QUERY, **({"t": t} if t is not None else {})}
    if route == "vobiz-xml":
        return _post_vobiz_xml(client, world, query=query)
    if route == "ncco":
        return _get_ncco(client, world, query=query)
    spec = ANSWER_URLS[route]
    world.provider = spec["provider"](PLIVO_TOKEN)
    world.provider.get_webhook_response = AsyncMock(return_value="<Response/>")
    path = f"/telephony/{route}?{urlencode(query)}"
    db, route_lookup, auth_lookup = _answer_url_patches(
        spec["module"], world, world.provider_for_run
    )
    with db, route_lookup, auth_lookup:
        response = client.post(
            path,
            data=spec["form"],
            headers=spec["sign"](BASE + path, spec["form"], world.provider),
        )
    return response, world.provider.get_webhook_response


@pytest.mark.parametrize("route", ANSWER_ROUTES)
@pytest.mark.parametrize("token", ["missing", "other_run", "events_token", "garbage"])
def test_replayed_answer_signature_without_this_runs_token_is_401(
    client, world, answer_secret, route, token
):
    # A captured signature is valid for any run of its org: only the run-bound token tells the runs apart.
    t = {
        "missing": None,
        "other_run": lambda: ws_auth.mint_answer_url_token(OTHER_RUN),
        "events_token": lambda: ws_auth.mint_events_token(RUN_ID),
        "garbage": "ñ-not-a-token",
    }[token]
    before = world.snapshot()

    response, build = _call_answer_url(
        client, world, route, t=t() if callable(t) else t
    )

    detail = _assert_b028(response, code="answer_url_token_invalid")
    assert detail["where"].endswith(route)
    build.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize("route", ANSWER_ROUTES)
def test_answer_url_with_this_runs_token_returns_the_document(
    client, world, answer_secret, route
):
    response, build = _call_answer_url(
        client, world, route, t=ws_auth.mint_answer_url_token(RUN_ID)
    )

    assert response.status_code == 200, response.text
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


@pytest.mark.parametrize("route", ANSWER_ROUTES)
@pytest.mark.parametrize("token", ["missing", "other_run"])
def test_answer_url_without_this_runs_token_is_served_and_logged_while_enforcement_is_off(
    client, world, answer_secret, monkeypatch, log_lines, route, token
):
    # Soft rollout (secret set, TELEPHONY_WS_TOKEN_ENFORCE off): an answer URL dialed by pre-token code carries no
    # token, and the carrier signature still holds. The miss is logged, never the token.
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_ENFORCE", False)
    t = None if token == "missing" else ws_auth.mint_answer_url_token(OTHER_RUN)

    response, build = _call_answer_url(client, world, route, t=t)

    assert response.status_code == 200, response.text
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)
    (line,) = [line for line in log_lines if "answer-URL token" in line]
    assert "WARNING" in line and route in line
    assert ("missing" if t is None else "wrong") in line
    if t is not None:
        assert t not in line


@pytest.mark.parametrize("route", ANSWER_ROUTES)
def test_answer_url_without_this_runs_token_in_a_cell_is_401_even_with_enforcement_off(
    client, world, answer_secret, monkeypatch, route
):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_ENFORCE", False)
    monkeypatch.setenv("CELL_ROLE", "api")
    before = world.snapshot()

    response, build = _call_answer_url(client, world, route)

    _assert_b028(response, code="answer_url_token_invalid")
    build.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize("route", ANSWER_ROUTES)
def test_answer_url_in_a_cell_without_a_token_secret_is_401(
    client, world, monkeypatch, route
):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", None)
    monkeypatch.setenv("CELL_ROLE", "call")
    before = world.snapshot()

    response, build = _call_answer_url(client, world, route)

    detail = _assert_b028(response, code="answer_url_token_unconfigured")
    assert "TELEPHONY_WS_TOKEN_SECRET" in detail["hint"] + detail["reason"]
    build.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize("route", ANSWER_ROUTES)
def test_answer_url_outside_a_cell_without_a_token_secret_is_unchanged(
    client, world, monkeypatch, route
):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", None)
    monkeypatch.delenv("CELL_ROLE", raising=False)

    response, build = _call_answer_url(client, world, route)

    assert response.status_code == 200, response.text
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


def test_twilio_dial_does_not_log_the_answer_url_token(monkeypatch, log_lines):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", secrets.token_hex(16))
    token = ws_auth.mint_answer_url_token(RUN_ID)
    url = f"https://backend.test/api/v1/telephony/twiml?workflow_run_id={RUN_ID}&t={token}"
    with patch(
        "api.services.telephony.providers.twilio.provider.aiohttp.ClientSession",
        side_effect=RuntimeError("test: stop before dialing"),
    ):
        with pytest.raises(RuntimeError):
            asyncio.run(
                _twilio().initiate_call(to_number="+34911000222", webhook_url=url)
            )

    text = "\n".join(log_lines)
    assert "/telephony/twiml" in text, "the dial log line is gone: nothing was checked"
    assert token not in text
