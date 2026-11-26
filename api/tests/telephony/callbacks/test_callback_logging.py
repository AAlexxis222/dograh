"""VOZ-N0-29 (VOZ-AC-B4-46b): what the callback logs never hold: the raw body, caller names, caller or called numbers, the media-WS token.

The fakes and request builders are shared: api.tests.support.carrier_callbacks and this folder's conftest.py.
"""

import secrets
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.tests.support.carrier_callbacks import (
    BODY_MARKER,
    CALLED,
    CALLER,
    ORG_ID,
    RUN_ID,
    UNAVAILABLE_ES,
    WORKFLOW_ID,
    _assert_b028,
    _post_vobiz_xml,
    _route_for_twilio,
)

# ---------------------------------------------------------------------------------------------------------
# What the log never holds: the raw body, caller names, caller or called numbers, the media-WS token
# ---------------------------------------------------------------------------------------------------------


NAME_MARKER = "caller-name-4471"


def _pii_form() -> dict:
    return {
        "CallSid": "CA-pii",
        "CallUUID": "uuid-pii",
        "AccountSid": "AC123",
        "From": CALLER,
        "To": CALLED,
        "CallerName": NAME_MARKER,
        "Marker": BODY_MARKER,
    }


def _post_legacy_inbound(client, world, *, signature_valid: bool):
    """The legacy per-workflow webhook with a body full of personal data, verified as ``signature_valid``."""
    normalized = SimpleNamespace(
        provider="twilio",
        direction="inbound",
        to_number=CALLED,
        from_number=CALLER,
        to_country="ES",
        from_country="ES",
        account_id="AC123",
        call_id="CA-pii",
        raw_data=_pii_form(),
        sip_headers={"X-Caller-Name": NAME_MARKER},
        sip_call_id=None,
    )
    _route_for_twilio(world)
    world.db.get_workflow_by_id = AsyncMock(
        return_value=SimpleNamespace(id=WORKFLOW_ID, organization_id=ORG_ID, user_id=1)
    )
    provider = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=signature_valid),
        start_inbound_stream=AsyncMock(),
    )
    with (
        patch("api.routes.telephony.normalize_webhook_data", return_value=normalized),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=TwilioProvider),
        ),
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider),
        ),
        patch("api.routes.telephony._create_inbound_workflow_run", new=AsyncMock()),
        patch("api.routes.telephony.call_concurrency") as concurrency,
    ):
        concurrency.acquire_org_slot = AsyncMock(
            side_effect=RuntimeError("test: stop after admission")
        )
        return client.post(f"/telephony/inbound/{WORKFLOW_ID}", data=_pii_form())


def _assert_logged_nothing_personal(log_lines, *, numbers=True):
    text = "\n".join(log_lines).lower()
    assert text, "the sink saw nothing: the test would pass for the wrong reason"
    assert BODY_MARKER not in text
    assert NAME_MARKER not in text
    if numbers:
        # Digits only: the redaction patcher misses numbers without "+", and this sink has no patcher at all.
        assert "612345678" not in text and "911000111" not in text


# Status callbacks of the carrier routes: run 999 does not exist, so each handler answers right after its first log.
STATUS_CALLBACKS = [
    ("plivo", "/telephony/plivo/hangup-callback/999", "form"),
    ("twilio", "/telephony/twilio/status-callback/999", "form"),
    ("vobiz", "/telephony/vobiz/hangup-callback/999", "form"),
    ("vobiz", "/telephony/vobiz/ring-callback/999", "form"),
    ("vobiz", "/telephony/vobiz/hangup-callback/workflow/999", "form"),
    ("exotel", "/telephony/exotel/status-callback/999", "form"),
    ("telnyx", "/telephony/telnyx/events/999", "json"),
    ("cloudonix", "/telephony/cloudonix/status-callback/999", "form"),
    ("cloudonix", "/telephony/cloudonix/cdr", "json"),
]


def test_legacy_inbound_does_not_log_the_body_or_the_caller_name(
    client, world, log_lines
):
    _post_legacy_inbound(client, world, signature_valid=True)

    _assert_logged_nothing_personal(log_lines, numbers=False)


@pytest.mark.parametrize(("carrier", "path", "kind"), STATUS_CALLBACKS)
def test_status_callbacks_do_not_log_the_body(
    client, world, monkeypatch, log_lines, carrier, path, kind
):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    module = f"api.services.telephony.providers.{carrier}.routes"
    body = {
        **_pii_form(),
        "data": {
            "event_type": "call.hangup",
            "payload": {"from": CALLER, "caller_id_name": NAME_MARKER},
        },
        "domain": "acme.cloudonix.net",
        "session": {"token": "session-pii"},
    }
    with patch(f"{module}.db_client", world.db):
        if kind == "json":
            client.post(path, json=body)
        else:
            client.post(path, data=_pii_form())

    _assert_logged_nothing_personal(log_lines)


def test_refused_legacy_inbound_does_not_log_the_caller_number(
    client, world, log_lines
):
    response = _post_legacy_inbound(client, world, signature_valid=False)

    _assert_b028(response)
    _assert_logged_nothing_personal(log_lines)


@pytest.mark.parametrize("routed", [False, True], ids=["no-route", "routed"])
def test_inbound_run_does_not_log_caller_or_called_numbers(
    client, world, log_lines, routed
):
    normalized = SimpleNamespace(
        provider="twilio",
        direction="inbound",
        to_number=CALLED,
        from_number=CALLER,
        to_country="ES",
        from_country="ES",
        account_id="AC123",
        call_id="CA-run",
    )
    if routed:
        _route_for_twilio(world)
        world.db.get_workflow = AsyncMock(return_value=None)
    with (
        patch("api.routes.telephony.normalize_webhook_data", return_value=normalized),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=TwilioProvider),
        ),
    ):
        response = client.post("/telephony/inbound/run", data=_pii_form())

    assert UNAVAILABLE_ES in response.text
    _assert_logged_nothing_personal(log_lines)


def test_vobiz_xml_does_not_log_the_media_url_or_its_token(client, world, log_lines):
    token = secrets.token_hex(16)
    stream_xml = (
        f"<Response><Stream>wss://media.example/ws/{WORKFLOW_ID}/{ORG_ID}/{RUN_ID}/{token}"
        "</Stream></Response>"
    )
    response, _ = _post_vobiz_xml(client, world, build=stream_xml)

    assert response.status_code == 200 and token in response.text
    text = "\n".join(log_lines)
    assert text, "the sink saw nothing: the test would pass for the wrong reason"
    assert token not in text and "wss://media.example" not in text


def test_refusal_log_cannot_be_split_by_a_request_value(client, world, log_lines):
    # The transfer_id path segment is the caller's; decoded, %0A is a newline that would start a forged line.
    response = client.post(
        "/telephony/transfer-result/x%0A2026-11-06 12:00:00 | INFO | carrier callback accepted",
        data={"CallStatus": "answered"},
    )

    _assert_b028(response)
    # No such transfer: logged as transfer_context_gone, with the same repr guard as a refusal.
    refusals = [line for line in log_lines if "code=" in line]
    assert refusals
    for line in refusals:
        assert "\n2026-11-06" not in line


# ---------------------------------------------------------------------------------------------------------
# RUL-r1-26: a value of an unauthenticated request is logged as its repr, so a newline in it cannot forge a line
# ---------------------------------------------------------------------------------------------------------


FORGED = "x\n2026-11-22 10:00:00.000 | INFO | FORGED-LINE"
FORGED_SEGMENT = "x%0AFORGED-LINE"

# (route module, path, body kind, body): each handler logs the request's values before any signature check.
UNSIGNED_VALUE_LOGS = [
    (
        "twilio",
        "/telephony/twilio/status-callback/999",
        "form",
        {"CallSid": FORGED, "CallStatus": FORGED},
    ),
    ("plivo", "/telephony/plivo/hangup-callback/999", "form", {"CallUUID": FORGED}),
    (
        "plivo",
        f"/telephony/plivo/transfer-xml/{FORGED_SEGMENT}/{FORGED_SEGMENT}",
        "form",
        {"CallUUID": FORGED},
    ),
    (
        "plivo",
        f"/telephony/plivo/transfer-result/{FORGED_SEGMENT}",
        "form",
        {"Event": FORGED, "CallUUID": FORGED, "HangupCause": FORGED},
    ),
    ("vobiz", "/telephony/vobiz/hangup-callback/999", "form", {"CallUUID": FORGED}),
    ("vobiz", "/telephony/vobiz/ring-callback/999", "form", {"CallUUID": FORGED}),
    (
        "vobiz",
        "/telephony/vobiz/hangup-callback/workflow/999",
        "form",
        {"CallUUID": FORGED},
    ),
    ("exotel", "/telephony/exotel/status-callback/999", "form", {"CallSid": FORGED}),
    (
        "vonage",
        "/telephony/vonage/events/999",
        "json",
        {"uuid": FORGED, "status": FORGED},
    ),
    ("vonage", "/telephony/vonage/events", "json", {"uuid": FORGED, "status": FORGED}),
    (
        "telnyx",
        "/telephony/telnyx/events/999",
        "json",
        {"data": {"event_type": FORGED}},
    ),
]


@pytest.mark.parametrize(("carrier", "path", "kind", "body"), UNSIGNED_VALUE_LOGS)
def test_unsigned_request_values_cannot_forge_a_log_line(
    client, world, monkeypatch, log_lines, carrier, path, kind, body
):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    module = f"api.services.telephony.providers.{carrier}.routes"
    world.db.get_workflow_run_by_call_id = AsyncMock(return_value=None)
    with (
        patch(f"{module}.db_client", world.db),
        patch(
            f"{module}.get_call_transfer_manager",
            new=AsyncMock(return_value=world.manager),
            create=True,
        ),
    ):
        if kind == "json":
            client.post(path, json=body)
        else:
            client.post(path, data=body)

    assert any("FORGED-LINE" in line for line in log_lines), "the value was not logged"
    for line in log_lines:
        assert "\nFORGED-LINE" not in line and "\n2026-11-22" not in line, line


# RUL-r2-2: the same rule in routes/telephony.py. (path, form, the account match): each request reaches a line that
# logs its values before any signature check.
UNSIGNED_INBOUND_VALUE_LOGS = {
    "run-direction": (
        "/telephony/inbound/run",
        {
            "CallSid": FORGED,
            "AccountSid": "AC123",
            "Direction": FORGED,
            "ToCountry": FORGED,
        },
        None,
    ),
    "run-no-route": (
        "/telephony/inbound/run",
        {"CallSid": FORGED, "AccountSid": FORGED, "Direction": "inbound", "To": CALLED},
        None,
    ),
    "legacy-direction": (
        f"/telephony/inbound/{WORKFLOW_ID}",
        {"CallSid": FORGED, "Direction": FORGED},
        None,
    ),
    "legacy-account": (
        f"/telephony/inbound/{WORKFLOW_ID}",
        {"CallSid": "CA-x", "AccountSid": FORGED, "Direction": "inbound", "To": CALLED},
        None,
    ),
    "legacy-number": (
        f"/telephony/inbound/{WORKFLOW_ID}",
        {
            "CallSid": "CA-x",
            "AccountSid": "AC123",
            "Direction": "inbound",
            "To": CALLED,
            "ToCountry": FORGED,
            "FromCountry": FORGED,
        },
        (5, None),
    ),
}


@pytest.mark.parametrize("case", sorted(UNSIGNED_INBOUND_VALUE_LOGS))
def test_unsigned_inbound_values_cannot_forge_a_log_line(
    client, world, log_lines, case
):
    path, form, account_match = UNSIGNED_INBOUND_VALUE_LOGS[case]
    world.db.get_workflow_by_id = AsyncMock(
        return_value=SimpleNamespace(id=WORKFLOW_ID, organization_id=ORG_ID, user_id=1)
    )
    world.db.list_telephony_configurations_by_provider = AsyncMock(
        return_value=[SimpleNamespace(id=5)]
    )
    world.db.find_active_phone_number_for_inbound = AsyncMock(return_value=None)
    world.db.list_phone_numbers_for_config = AsyncMock(return_value=[])
    with (
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=TwilioProvider),
        ),
        patch(
            "api.services.telephony.factory.find_telephony_config_for_inbound",
            new=AsyncMock(return_value=account_match),
        ),
    ):
        client.post(path, data=form)

    assert any("FORGED-LINE" in line for line in log_lines), "the value was not logged"
    for line in log_lines:
        assert "\nFORGED-LINE" not in line and "\n2026-11-22" not in line, line
