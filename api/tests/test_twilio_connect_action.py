"""Twilio ``<Connect action>`` callback: TwiML shape and the run-state decision.

When the media WebSocket ends, Twilio requests the ``action`` URL and plays what
it returns. The callback must decide from the run's state in our database (never
from Twilio parameters), only for a signed request, and the TwiML must not hold
the caller in a silent ``<Pause>``.
"""

import secrets
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from twilio.request_validator import RequestValidator

from api import constants
from api.services.telephony import ws_auth
from api.services.telephony.providers.twilio import provider as provider_module
from api.services.telephony.providers.twilio.provider import (
    TwilioProvider,
    build_stream_twiml,
)
from api.services.telephony.providers.twilio.routes import router

ROUTES = "api.services.telephony.providers.twilio.routes"
RUN_ID = 42
ACTION_PATH = f"/api/v1/telephony/twilio/connect-action/{RUN_ID}"
FORM = {"CallSid": "CA1"}
# Generated per run: the tests need values, not literals that look like credentials.
TOKEN_SECRET = secrets.token_hex(16)
TWILIO_AUTH_TOKEN = secrets.token_hex(16)


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", TOKEN_SECRET)


@pytest.fixture
def no_secret(monkeypatch):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", None)


@pytest.fixture
def twilio():
    return TwilioProvider(
        {
            "account_sid": "AC123",
            "auth_token": TWILIO_AUTH_TOKEN,
            "from_numbers": ["+15551230002"],
        }
    )


@pytest.fixture
def db(twilio):
    """Patched ``db_client`` plus the provider lookup, wired like the other Twilio route tests."""
    with (
        patch(f"{ROUTES}.db_client") as db_client,
        patch(
            f"{ROUTES}.get_telephony_provider_for_run",
            new_callable=AsyncMock,
            return_value=twilio,
        ),
    ):
        db_client.get_workflow_run_by_id = AsyncMock(
            return_value=SimpleNamespace(id=RUN_ID, workflow_id=7, state="running")
        )
        db_client.get_workflow_by_id = AsyncMock(
            return_value=SimpleNamespace(organization_id=11)
        )
        yield db_client


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/telephony")
    return TestClient(app)


def _post(client, twilio, *, query=None, signed=True):
    url = f"http://testserver{ACTION_PATH}"
    if query:
        url = f"{url}?{urlencode(query)}"
    headers = {}
    if signed:
        headers["X-Twilio-Signature"] = RequestValidator(
            twilio.auth_token
        ).compute_signature(url, FORM)
    return client.post(url, data=FORM, headers=headers)


def _set_state(db, state):
    db.get_workflow_run_by_id.return_value = SimpleNamespace(
        id=RUN_ID, workflow_id=7, state=state
    )


def _assert_no_writes(db):
    called = {call[0] for call in db.mock_calls}
    assert called <= {"get_workflow_run_by_id", "get_workflow_by_id"}


# --------------------------------------------------------------------------
# TwiML shape
# --------------------------------------------------------------------------


def test_stream_twiml_has_action_and_no_pause():
    xml = build_stream_twiml(
        ws_url="wss://cell/ws/x",
        action_url="https://cell/api/v1/telephony/twilio/connect-action/42?t=tok",
    )
    assert (
        '<Connect action="https://cell/api/v1/telephony/twilio/connect-action/42?t=tok">'
        in xml
    )
    assert '<Stream url="wss://cell/ws/x"></Stream>' in xml
    assert "<Pause" not in xml


def test_stream_twiml_status_callback_is_optional():
    without = build_stream_twiml(ws_url="wss://w", action_url="https://a")
    with_cb = build_stream_twiml(
        ws_url="wss://w", action_url="https://a", status_callback_url="https://s/1"
    )
    assert "statusCallback" not in without
    assert '<Stream url="wss://w" statusCallback="https://s/1"></Stream>' in with_cb


async def test_outbound_twiml_points_action_at_the_run(twilio, no_secret, monkeypatch):
    monkeypatch.setattr(
        provider_module,
        "get_backend_endpoints",
        AsyncMock(return_value=("https://api.test", "wss://api.test")),
    )
    body = await twilio.get_webhook_response(7, 3, RUN_ID)
    assert (
        '<Connect action="https://api.test/api/v1/telephony/twilio/connect-action/42">'
        in body
    )
    assert "<Pause" not in body


async def test_inbound_twiml_points_action_at_the_run_and_keeps_status_callback(
    twilio, secret
):
    response = await twilio.start_inbound_stream(
        websocket_url="wss://api.test/ws",
        workflow_run_id=RUN_ID,
        normalized_data=None,
        backend_endpoint="https://api.test",
    )
    body = response.body.decode()
    token = ws_auth.mint_connect_action_token(RUN_ID)
    assert (
        f'<Connect action="https://api.test/api/v1/telephony/twilio/connect-action/42?t={token}">'
        in body
    )
    assert (
        'statusCallback="https://api.test/api/v1/telephony/twilio/status-callback/42"'
        in body
    )
    assert "<Pause" not in body


# --------------------------------------------------------------------------
# Decision by run state
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "run_state,expected,forbidden",
    [
        ("completed", "<Hangup/>", "<Say"),  # (a) normal end: never the rejection
        ("running", "<Say", "<Connect"),  # (c) mid-call drop: message, never a new bot
        ("initialized", "<Say", "<Connect"),  # (d) anything else: audible message
    ],
)
def test_action_callback_decides_by_run_state(
    client, twilio, db, no_secret, run_state, expected, forbidden
):
    _set_state(db, run_state)
    resp = _post(client, twilio)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    assert expected in resp.text and forbidden not in resp.text
    assert "<Hangup/>" in resp.text
    if run_state != "completed":
        assert '<Say language="es-ES">' in resp.text
        assert "Ahora mismo no podemos atender tu llamada." in resp.text


def test_completed_run_gets_only_a_hangup(client, twilio, db, no_secret):
    _set_state(db, "completed")
    resp = _post(client, twilio)
    assert resp.text == "<Response><Hangup/></Response>"


def test_state_comes_from_the_database_not_from_twilio_parameters(
    client, twilio, db, no_secret
):
    _set_state(db, "running")
    url = f"http://testserver{ACTION_PATH}"
    form = {"CallSid": "CA1", "CallStatus": "completed", "state": "completed"}
    signature = RequestValidator(twilio.auth_token).compute_signature(url, form)
    resp = client.post(url, data=form, headers={"X-Twilio-Signature": signature})
    assert "<Say" in resp.text


# --------------------------------------------------------------------------
# Failing closed
# --------------------------------------------------------------------------


def test_action_callback_rejects_missing_signature(client, twilio, db, no_secret):
    resp = _post(client, twilio, signed=False)
    assert resp.status_code == 403
    _assert_no_writes(db)


def test_action_callback_rejects_wrong_signature(client, twilio, db, no_secret):
    url = f"http://testserver{ACTION_PATH}"
    resp = client.post(url, data=FORM, headers={"X-Twilio-Signature": "bogus"})
    assert resp.status_code == 403
    _assert_no_writes(db)


def test_action_callback_rejects_unknown_run(client, twilio, db, no_secret):
    db.get_workflow_run_by_id.return_value = None
    resp = _post(client, twilio)
    assert resp.status_code == 403
    _assert_no_writes(db)


def test_valid_token_and_signature_are_accepted(client, twilio, db, secret):
    token = ws_auth.mint_connect_action_token(RUN_ID)
    resp = _post(client, twilio, query={"t": token})
    assert resp.status_code == 200


@pytest.mark.parametrize("query", [None, {"t": "not-the-token"}, {"t": ""}])
def test_secret_set_requires_the_right_token(client, twilio, db, secret, query):
    """A valid Twilio signature is not enough once the token is opted in."""
    resp = _post(client, twilio, query=query)
    assert resp.status_code == 403
    _assert_no_writes(db)


def test_token_of_another_run_is_rejected(client, twilio, db, secret):
    resp = _post(client, twilio, query={"t": ws_auth.mint_connect_action_token(99)})
    assert resp.status_code == 403


def test_signature_is_checked_even_with_a_valid_token(client, twilio, db, secret):
    token = ws_auth.mint_connect_action_token(RUN_ID)
    resp = _post(client, twilio, query={"t": token}, signed=False)
    assert resp.status_code == 403
    _assert_no_writes(db)


def test_no_secret_means_no_token_in_the_url_and_no_token_check(
    client, twilio, db, no_secret
):
    assert ws_auth.mint_connect_action_token(RUN_ID) is None
    resp = _post(client, twilio, query={"t": "ignored"})
    assert resp.status_code == 200


# --------------------------------------------------------------------------
# The token itself
# --------------------------------------------------------------------------


def test_connect_action_token_is_not_interchangeable_with_other_tokens(secret):
    token = ws_auth.mint_connect_action_token(RUN_ID)
    events_token = ws_auth.mint_events_token(RUN_ID)
    assert token != events_token
    assert ws_auth.verify_connect_action_token(RUN_ID, token) is True
    assert ws_auth.verify_connect_action_token(RUN_ID, events_token) is False
    assert ws_auth.verify_events_token(RUN_ID, token) is False


def test_connect_action_token_verification_never_raises(secret):
    assert ws_auth.verify_connect_action_token(RUN_ID, "éè") is False
    assert ws_auth.verify_connect_action_token(RUN_ID, None) is False


def test_redact_token_masks_the_action_token(secret):
    token = ws_auth.mint_connect_action_token(RUN_ID)
    url = f"https://api.test/api/v1/telephony/twilio/connect-action/42?t={token}"
    redacted = ws_auth.redact_token(f'<Connect action="{url}">')
    assert token not in redacted
    assert "?t=[REDACTED]" in redacted
    assert "connect-action/42" in redacted


def test_redact_token_leaves_unrelated_t_params_alone():
    assert ws_auth.redact_token("a=1&format=x&t5=y") == "a=1&format=x&t5=y"


async def test_outbound_twiml_log_line_does_not_carry_the_action_token(
    twilio, secret, monkeypatch
):
    monkeypatch.setattr(
        provider_module,
        "get_backend_endpoints",
        AsyncMock(return_value=("https://api.test", "wss://api.test")),
    )
    logged = []
    monkeypatch.setattr(
        provider_module,
        "logger",
        MagicMock(info=lambda msg, *a, **k: logged.append(msg)),
    )
    await twilio.get_webhook_response(7, 3, RUN_ID)
    assert ws_auth.mint_connect_action_token(RUN_ID) not in "".join(logged)
