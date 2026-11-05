"""VOZ-N0-29 (VOZ-AC-B4-46b): every carrier callback verifies its signature and fails closed, and what a caller
hears on an error is Spanish.

"State" is what a forged request could move: the run rows (state, updated_at) and every write to them, the
TransferContext, the transfer events published, and the status updates handed to the processor. A rejected
request leaves all four exactly as they were. No test needs Postgres or Redis: the collaborators are fakes.
"""

import base64
import dataclasses
import hashlib
import hmac
import json
import pathlib
import re
import secrets
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlencode

import nacl.signing
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger
from twilio.request_validator import RequestValidator

from api.errors.telephony_errors import (
    CALLER_MESSAGES_ES,
    TELEPHONY_ERROR_MESSAGES,
    TelephonyError,
)
from api.routes.telephony import router
from api.services.runtime.cell_startup import is_cell
from api.services.telephony.providers.cloudonix.provider import CloudonixProvider
from api.services.telephony.providers.exotel.provider import ExotelProvider
from api.services.telephony.providers.plivo.provider import PlivoProvider
from api.services.telephony.providers.telnyx.provider import TelnyxProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.vobiz.provider import VobizProvider
from api.services.telephony.providers.vonage.provider import VonageProvider
from api.services.telephony.transfer_event_protocol import (
    TransferContext,
    TransferEventType,
)

API_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "http://testserver"
# Generated per run: the tests need values, not literals that look like credentials.
TWILIO_TOKEN = secrets.token_hex(16)
VOBIZ_TOKEN = secrets.token_hex(16)
OTHER_TOKEN = secrets.token_hex(16)
FAKE = secrets.token_hex(8)
RUN_ID = 7
WORKFLOW_ID = 33
ORG_ID = 11
TRANSFER_ID = "transfer-1"
CALLER = "+34612345678"

CAPACITY_ES = CALLER_MESSAGES_ES[TelephonyError.CONCURRENT_CALL_LIMIT]
UNAVAILABLE_ES = CALLER_MESSAGES_ES[TelephonyError.PHONE_NUMBER_NOT_CONFIGURED]

TELNYX_KEY = nacl.signing.SigningKey.generate()
TELNYX_PUBLIC = base64.b64encode(bytes(TELNYX_KEY.verify_key)).decode("ascii")


# ---------------------------------------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------------------------------------


def _twilio() -> TwilioProvider:
    return TwilioProvider(
        {
            "account_sid": "AC123",
            "auth_token": TWILIO_TOKEN,
            "from_numbers": ["+15551230002"],
        }
    )


def _cloudonix() -> CloudonixProvider:
    return CloudonixProvider({"domain_id": "acme.cloudonix.net", "bearer_token": FAKE})


def _vobiz() -> VobizProvider:
    return VobizProvider(
        {
            "auth_id": "MA123",
            "auth_token": VOBIZ_TOKEN,
            "application_id": None,
            "from_numbers": ["+15551230002"],
        }
    )


def _telnyx() -> TelnyxProvider:
    return TelnyxProvider(
        {
            "api_key": FAKE,
            "connection_id": "connection-id",
            "webhook_public_key": TELNYX_PUBLIC,
            "from_numbers": ["+15551230002"],
        }
    )


class FakeManager:
    """The Redis-backed CallTransferManager, in memory, recording what a request did to it."""

    def __init__(self, contexts: dict[str, TransferContext]):
        self.contexts = contexts
        self.published = []
        self.calls = []

    async def get_transfer_context(self, transfer_id):
        return self.contexts.get(transfer_id)

    async def publish_transfer_event(self, event):
        self.published.append(event)

    async def store_transfer_context(self, context):
        self.calls.append(("store", context.transfer_id))
        self.contexts[context.transfer_id] = context

    async def remove_transfer_context(self, transfer_id):
        self.calls.append(("remove", transfer_id))
        self.contexts.pop(transfer_id, None)

    async def claim_transfer_step(self, transfer_id, step):
        self.calls.append(("claim", transfer_id, step))
        return True


class FakeDb:
    def __init__(self):
        self.run = SimpleNamespace(
            id=RUN_ID,
            workflow_id=WORKFLOW_ID,
            state="running",
            updated_at="2026-11-05T10:00:00",
            initial_context={"telephony_configuration_id": 5},
            gathered_context={"call_id": "CA-original"},
            logs={},
        )
        self.workflow = SimpleNamespace(id=WORKFLOW_ID, organization_id=ORG_ID)
        self.writes = []
        self.route = (
            None  # what find_inbound_route_by_account answers (a (config, phone) pair)
        )

    async def get_workflow_run_by_id(self, run_id):
        return self.run if run_id == self.run.id else None

    async def get_workflow_by_id(self, workflow_id):
        return self.workflow if workflow_id == self.workflow.id else None

    async def update_workflow_run(self, **kwargs):
        self.writes.append(kwargs)

    async def find_inbound_route_by_account(self, **kwargs):
        return self.route

    async def find_inbound_route_by_called_number(self, **kwargs):
        return self.route


def _context(**overrides) -> TransferContext:
    fields = dict(
        transfer_id=TRANSFER_ID,
        call_sid="CA-transfer",
        target_number="+34911222333",
        tool_uuid="tool-1",
        original_call_sid="CA-original",
        conference_name="conf-1",
        initiated_at=1.0,
        workflow_run_id=RUN_ID,
    )
    fields.update(overrides)
    return TransferContext(**fields)


class World:
    """One test's universe: the fakes, the provider the run resolves to, and the state snapshot."""

    def __init__(self):
        self.db = FakeDb()
        self.manager = FakeManager({TRANSFER_ID: _context()})
        self.provider = _twilio()
        self.provider_error: Exception | None = None
        self.status_updates = []

    async def provider_for_run(self, run, organization_id):
        if self.provider_error:
            raise self.provider_error
        return self.provider

    def snapshot(self) -> dict:
        return {
            "run": (self.db.run.state, self.db.run.updated_at),
            "writes": json.dumps(self.db.writes, default=str),
            "contexts": {
                key: dataclasses.asdict(value)
                for key, value in self.manager.contexts.items()
            },
            "published": [dataclasses.asdict(e) for e in self.manager.published],
            "calls": list(self.manager.calls),
            "status_updates": list(self.status_updates),
        }


@pytest.fixture
def world():
    w = World()
    get_manager = AsyncMock(return_value=w.manager)
    with (
        patch("api.services.telephony.callback_auth.db_client", w.db),
        patch(
            "api.services.telephony.callback_auth.get_call_transfer_manager",
            get_manager,
        ),
        patch("api.routes.telephony.get_call_transfer_manager", get_manager),
        patch(
            "api.services.telephony.providers.telnyx.routes.get_call_transfer_manager",
            get_manager,
        ),
        patch(
            "api.services.telephony.callback_auth.get_telephony_provider_for_run",
            side_effect=w.provider_for_run,
        ),
        patch("api.routes.telephony.db_client", w.db),
        patch("api.services.telephony.providers.cloudonix.routes.db_client", w.db),
        patch(
            "api.services.telephony.providers.cloudonix.routes.get_call_transfer_manager",
            get_manager,
        ),
        patch(
            "api.services.telephony.providers.cloudonix.routes._process_status_update",
            side_effect=lambda run_id, update: w.status_updates.append(run_id),
        ),
        patch("api.services.telephony.providers.vonage.routes.db_client", w.db),
        patch(
            "api.services.telephony.providers.vonage.routes."
            "get_telephony_provider_for_run",
            side_effect=w.provider_for_run,
        ),
    ):
        yield w


@pytest.fixture
def client(world):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def log_lines():
    lines = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


def _twilio_signature(url: str, params: dict, token: str = TWILIO_TOKEN) -> str:
    return RequestValidator(token).compute_signature(url, params)


def _assert_b028(response, *, code: str | None = None) -> dict:
    """401 whose body says what failed (VOZ-AC-B0-28): stable code, where, reason and hint, none empty."""
    assert response.status_code == 401, response.text
    detail = response.json()["detail"]
    for field in ("code", "where", "reason", "hint"):
        assert detail.get(field), f"{field} missing or empty in {detail}"
    if code:
        assert detail["code"] == code
    return detail


# ---------------------------------------------------------------------------------------------------------
# /transfer-result/{transfer_id} (Twilio StatusCallback of the transfer leg)
# ---------------------------------------------------------------------------------------------------------

TRANSFER_PATH = f"/telephony/transfer-result/{TRANSFER_ID}"


def _post_transfer_result(
    client, *, signed: bool, form=None, token=TWILIO_TOKEN, headers=None
):
    form = form or {"CallStatus": "answered", "CallSid": "CA-transfer"}
    headers = dict(headers or {})
    if signed:
        headers["X-Twilio-Signature"] = _twilio_signature(
            BASE + TRANSFER_PATH, form, token
        )
    return client.post(TRANSFER_PATH, data=form, headers=headers)


def test_transfer_result_unsigned_forgery_is_401_and_moves_nothing(client, world):
    before = world.snapshot()

    response = _post_transfer_result(client, signed=False)

    _assert_b028(response)
    assert world.snapshot() == before
    assert world.manager.published == []


def test_transfer_result_signature_of_another_token_is_401(client, world):
    before = world.snapshot()

    response = _post_transfer_result(client, signed=True, token=OTHER_TOKEN)

    _assert_b028(response)
    assert world.snapshot() == before


@pytest.mark.parametrize(
    "scenario",
    [
        "no_transfer_context",
        "context_without_workflow_run_id",
        "run_not_found",
        "workflow_not_found",
        "provider_cannot_be_built",
        "run_belongs_to_another_provider",
    ],
)
def test_transfer_result_without_a_credential_context_is_401_even_if_signed(
    client, world, scenario
):
    if scenario == "no_transfer_context":
        world.manager.contexts.clear()
    elif scenario == "context_without_workflow_run_id":
        world.manager.contexts[TRANSFER_ID] = _context(workflow_run_id=None)
    elif scenario == "run_not_found":
        world.manager.contexts[TRANSFER_ID] = _context(workflow_run_id=999)
    elif scenario == "workflow_not_found":
        world.db.run.workflow_id = 404
    elif scenario == "provider_cannot_be_built":
        world.provider_error = ValueError("config parked")
    elif scenario == "run_belongs_to_another_provider":
        # Cloudonix's verifier accepts everything, so only the provider-name check can refuse this one.
        world.provider = _cloudonix()
    before = world.snapshot()

    response = _post_transfer_result(client, signed=True, headers={"x-cx-apikey": FAKE})

    _assert_b028(response)
    assert world.snapshot() == before


def test_transfer_result_signed_answer_publishes_destination_answered(client, world):
    response = _post_transfer_result(client, signed=True)

    assert response.status_code == 200
    assert [e.type for e in world.manager.published] == [
        TransferEventType.DESTINATION_ANSWERED
    ]
    assert world.manager.published[0].original_call_sid == "CA-original"


def test_transfer_result_signed_no_answer_publishes_transfer_failed(client, world):
    form = {"CallStatus": "no-answer", "CallSid": "CA-transfer"}

    response = _post_transfer_result(client, signed=True, form=form)

    assert response.status_code == 200
    assert [e.type for e in world.manager.published] == [
        TransferEventType.TRANSFER_FAILED
    ]


# ---------------------------------------------------------------------------------------------------------
# /inbound/fallback
# ---------------------------------------------------------------------------------------------------------

FALLBACK_PATH = "/telephony/inbound/fallback"
BODY_MARKER = "body-marker-9137"


def _fallback_form(account_sid="AC123") -> dict:
    return {
        "CallSid": "CA-fallback",
        "AccountSid": account_sid,
        "From": CALLER,
        "To": "+34911000111",
        "ApiVersion": "2010-04-01",
        "ErrorCode": "11200",
        "Marker": BODY_MARKER,
    }


def _post_fallback(client, form, *, signature="valid", token=TWILIO_TOKEN):
    headers = {
        "User-Agent": "TwilioProxy/1.1",
        "X-Custom-Probe": "custom-header-value",
    }
    if signature == "valid":
        headers["X-Twilio-Signature"] = _twilio_signature(
            BASE + FALLBACK_PATH, form, token
        )
    elif signature == "forged":
        headers["X-Twilio-Signature"] = "Zm9yZ2Vk"
    return client.post(FALLBACK_PATH, data=form, headers=headers)


def _route_for_twilio(world):
    world.db.route = (
        SimpleNamespace(id=5, organization_id=ORG_ID, name="Twilio"),
        SimpleNamespace(id=9, address="+34911000111", inbound_workflow_id=WORKFLOW_ID),
    )


@pytest.fixture
def fallback_provider(world):
    """The provider the matched config builds: real Twilio, so the real signature check runs."""
    _route_for_twilio(world)
    with patch(
        "api.routes.telephony.get_telephony_provider_by_id",
        new=AsyncMock(return_value=world.provider),
    ) as built:
        yield built


@pytest.mark.parametrize("signature", ["missing", "forged"])
def test_fallback_bad_signature_is_401_and_moves_nothing(
    client, world, fallback_provider, signature
):
    before = world.snapshot()

    response = _post_fallback(client, _fallback_form(), signature=signature)

    _assert_b028(response)
    assert world.snapshot() == before


def test_fallback_signed_with_another_token_is_401(client, world, fallback_provider):
    response = _post_fallback(client, _fallback_form(), token=OTHER_TOKEN)

    _assert_b028(response)


def test_fallback_unknown_account_is_401(client, world, fallback_provider):
    world.db.route = None

    response = _post_fallback(client, _fallback_form(account_sid="ACunknown"))

    _assert_b028(response)
    fallback_provider.assert_not_awaited()


def test_fallback_that_fails_to_build_its_provider_is_401(client, world):
    _route_for_twilio(world)
    with patch(
        "api.routes.telephony.get_telephony_provider_by_id",
        new=AsyncMock(side_effect=ValueError("config parked")),
    ):
        response = _post_fallback(client, _fallback_form())

    _assert_b028(response)


def test_fallback_with_no_body_is_401_not_500(client, world, fallback_provider):
    response = client.post(FALLBACK_PATH, headers={"User-Agent": "TwilioProxy/1.1"})

    _assert_b028(response)


def test_fallback_from_a_provider_nobody_recognises_is_401(
    client, world, fallback_provider
):
    response = client.post(FALLBACK_PATH, data={"Hello": "world"})

    _assert_b028(response)


def test_fallback_signed_answers_in_spanish(client, world, fallback_provider):
    response = _post_fallback(client, _fallback_form())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert re.search(r'<Say language="es-ES">([^<]+)</Say>', response.text)
    spoken = re.search(r"<Say[^>]*>([^<]+)</Say>", response.text).group(1)
    assert spoken in CALLER_MESSAGES_ES.values()
    assert "Sorry" not in response.text and "temporarily" not in response.text


@pytest.mark.parametrize("signature", ["valid", "forged", "missing"])
def test_fallback_does_not_log_headers_or_body(
    client, world, fallback_provider, log_lines, signature
):
    _post_fallback(client, _fallback_form(), signature=signature)

    text = "\n".join(log_lines).lower()
    assert text, "the sink saw nothing: the test would pass for the wrong reason"
    assert CALLER not in text and "612345678" not in text
    assert BODY_MARKER not in text
    # No header is dumped: not a header sent by the request, not a value, not the signature itself. (The
    # provider's own warning may name the header it expected when it is missing; that names none of the request.)
    signature_sent = {
        "valid": _twilio_signature(BASE + FALLBACK_PATH, _fallback_form()),
        "forged": "Zm9yZ2Vk",
        "missing": None,
    }[signature]
    for leaked in (
        "x-custom-probe",
        "custom-header-value",
        "twilioproxy",
        signature_sent,
    ):
        assert leaked is None or leaked.lower() not in text


def test_fallback_for_a_provider_nobody_recognises_does_not_log_headers_or_body(
    client, world, fallback_provider, log_lines
):
    client.post(
        FALLBACK_PATH,
        data={"Marker": BODY_MARKER, "From": CALLER},
        headers={"X-Custom-Probe": "custom-header-value"},
    )

    text = "\n".join(log_lines).lower()
    assert text
    assert CALLER not in text and BODY_MARKER not in text
    assert "x-custom-probe" not in text and "custom-header-value" not in text


# ---------------------------------------------------------------------------------------------------------
# /inbound/{workflow_id} (legacy) and Vonage events: they verified already; these pin it
# ---------------------------------------------------------------------------------------------------------


def test_legacy_inbound_bad_signature_creates_no_run_and_speaks_spanish(client, world):
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
    _route_for_twilio(world)
    world.db.get_workflow_by_id = AsyncMock(
        return_value=SimpleNamespace(id=WORKFLOW_ID, organization_id=ORG_ID, user_id=1)
    )
    provider = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=False),
        start_inbound_stream=AsyncMock(),
    )
    created = AsyncMock()
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
        patch("api.routes.telephony._create_inbound_workflow_run", new=created),
        patch("api.routes.telephony.call_concurrency") as concurrency,
    ):
        concurrency.acquire_org_slot = AsyncMock()
        response = client.post(
            f"/telephony/inbound/{WORKFLOW_ID}",
            data={"CallSid": "CA-legacy", "From": CALLER},
        )

    provider.verify_inbound_signature.assert_awaited_once()
    provider.start_inbound_stream.assert_not_awaited()
    created.assert_not_awaited()
    concurrency.acquire_org_slot.assert_not_awaited()
    assert UNAVAILABLE_ES in response.text


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


# ---------------------------------------------------------------------------------------------------------
# Telnyx transfer-result
# ---------------------------------------------------------------------------------------------------------

TELNYX_PATH = f"/telephony/telnyx/transfer-result/{TRANSFER_ID}"


def _telnyx_body(event_type="call.hangup", cause="busy") -> str:
    return json.dumps(
        {
            "data": {
                "event_type": event_type,
                "payload": {"call_control_id": "cc-1", "hangup_cause": cause},
            }
        },
        separators=(",", ":"),
    )


def _telnyx_headers(body: str, *, key=TELNYX_KEY, timestamp=None) -> dict:
    timestamp = timestamp or str(int(time.time()))
    signature = key.sign(f"{timestamp}|{body}".encode()).signature
    return {
        "telnyx-signature-ed25519": base64.b64encode(signature).decode("ascii"),
        "telnyx-timestamp": timestamp,
        "content-type": "application/json",
    }


def _post_telnyx(client, body, headers):
    return client.post(TELNYX_PATH, content=body, headers=headers)


@pytest.mark.parametrize("headers", ["none", "other_key", "stale", "body_tampered"])
def test_telnyx_transfer_result_bad_signature_is_401_and_moves_nothing(
    client, world, headers
):
    world.provider = _telnyx()
    body = _telnyx_body()
    if headers == "none":
        sent = {"content-type": "application/json"}
    elif headers == "other_key":
        sent = _telnyx_headers(body, key=nacl.signing.SigningKey.generate())
    elif headers == "stale":
        sent = _telnyx_headers(body, timestamp=str(int(time.time()) - 3600))
    else:
        sent = _telnyx_headers(_telnyx_body(cause="no_answer"))
    before = world.snapshot()

    response = _post_telnyx(client, body, sent)

    _assert_b028(response)
    assert world.snapshot() == before


def test_telnyx_transfer_result_without_context_is_401_even_if_signed(client, world):
    world.provider = _telnyx()
    world.manager.contexts.clear()
    body = _telnyx_body()

    response = _post_telnyx(client, body, _telnyx_headers(body))

    _assert_b028(response)
    assert world.manager.published == []


def test_telnyx_transfer_result_signed_hangup_publishes_transfer_failed(client, world):
    world.provider = _telnyx()
    body = _telnyx_body(cause="busy")

    response = _post_telnyx(client, body, _telnyx_headers(body))

    assert response.status_code == 200
    assert [e.type for e in world.manager.published] == [
        TransferEventType.TRANSFER_FAILED
    ]
    assert world.manager.published[0].reason == "busy"


# ---------------------------------------------------------------------------------------------------------
# Vobiz /vobiz-xml (the answer URL: its XML carries the media-stream URL and its token)
# ---------------------------------------------------------------------------------------------------------

VOBIZ_QUERY = {
    "workflow_id": WORKFLOW_ID,
    "workflow_run_id": RUN_ID,
    "organization_id": ORG_ID,
}


def _vobiz_headers(path_url: str, token: str = VOBIZ_TOKEN) -> dict:
    nonce = "12345678901234567890"
    digest = hmac.new(
        token.encode(), f"{path_url}.{nonce}".encode(), hashlib.sha256
    ).digest()
    return {
        "x-vobiz-signature-v3": base64.b64encode(digest).decode("ascii"),
        "x-vobiz-signature-v3-nonce": nonce,
    }


def _post_vobiz_xml(client, world, *, query=None, headers=None):
    world.provider = _vobiz()
    world.provider.get_webhook_response = AsyncMock(return_value="<Response/>")
    path = "/telephony/vobiz-xml"
    response = client.post(
        f"{path}?{urlencode(query or VOBIZ_QUERY)}",
        data={"CallUUID": "uuid-1"},
        headers=headers if headers is not None else _vobiz_headers(BASE + path),
    )
    return response, world.provider.get_webhook_response


def test_vobiz_xml_signed_returns_the_stream_xml(client, world):
    response, build = _post_vobiz_xml(client, world)

    assert response.status_code == 200
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


@pytest.mark.parametrize("headers", [{}, "forged", "other_token"])
def test_vobiz_xml_bad_signature_is_401_and_builds_no_stream_url(
    client, world, headers
):
    if headers == "forged":
        headers = {
            "x-vobiz-signature-v3": "Zm9yZ2Vk",
            "x-vobiz-signature-v3-nonce": "12345678901234567890",
        }
    elif headers == "other_token":
        headers = _vobiz_headers(BASE + "/telephony/vobiz-xml", token=OTHER_TOKEN)
    before = world.snapshot()

    response, build = _post_vobiz_xml(client, world, headers=headers)

    _assert_b028(response)
    build.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize(
    "tampered", [{"workflow_id": 34}, {"organization_id": 12}, {"workflow_run_id": 8}]
)
def test_vobiz_xml_query_that_disagrees_with_the_run_is_401_even_if_signed(
    client, world, tampered
):
    # Vobiz signs the URL without its query string, so a replayed signature covers any query. The run decides.
    response, build = _post_vobiz_xml(client, world, query={**VOBIZ_QUERY, **tampered})

    _assert_b028(response)
    build.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------
# Cloudonix: signature UNVERIFIED (verify_inbound_signature accepts everything), so a cell refuses
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
def test_cloudonix_callback_outside_a_cell_is_unchanged(
    client, world, monkeypatch, name, path, kind
):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    world.manager.contexts.clear()

    response = _post_cloudonix(client, path, kind)

    # The handler ran (it answers 200 with its own ignore/error body) instead of the guard answering 401.
    assert response.status_code == 200


# ---------------------------------------------------------------------------------------------------------
# R8: the cell marker
# ---------------------------------------------------------------------------------------------------------


def test_is_cell_is_true_only_when_cell_role_is_non_empty(monkeypatch):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    assert is_cell() is False
    monkeypatch.setenv("CELL_ROLE", "")
    assert is_cell() is False
    monkeypatch.setenv("CELL_ROLE", "call")
    assert is_cell() is True


def test_is_cell_ignores_the_carrier_region_policy(monkeypatch):
    monkeypatch.delenv("CELL_ROLE", raising=False)
    monkeypatch.setenv("CARRIER_REGION_POLICY", "eu")
    assert is_cell() is False


def test_cell_startup_uses_is_cell(monkeypatch):
    """assert_cell_startup_config decides 'in a cell' through is_cell(), not a second inline check."""
    from api.services.runtime import cell_startup

    monkeypatch.setenv("CELL_ROLE", "call")
    monkeypatch.delenv("DOGRAH_DEVOPS_SECRET", raising=False)
    # Not a cell as far as the module is concerned: only the master-key check runs, so no CellStartupError
    # about DOGRAH_DEVOPS_SECRET even though the real env says CELL_ROLE=call.
    with patch.object(cell_startup, "is_cell", return_value=False) as marker:
        cell_startup.assert_cell_startup_config("test")
    marker.assert_called()


# ---------------------------------------------------------------------------------------------------------
# What the caller hears
# ---------------------------------------------------------------------------------------------------------


def test_caller_messages_are_exactly_the_two_agreed_sentences():
    assert set(CALLER_MESSAGES_ES.values()) == {
        "Lo sentimos, en este momento no podemos atender su llamada. Por favor, inténtelo de nuevo más tarde.",
        "Lo sentimos, este número no está disponible en este momento. Por favor, inténtelo más tarde.",
    }


def test_every_error_has_a_caller_message_and_capacity_ones_say_capacity():
    assert set(CALLER_MESSAGES_ES) == set(TelephonyError) - {TelephonyError.VALID}
    capacity = {
        TelephonyError.CONCURRENT_CALL_LIMIT,
        TelephonyError.QUOTA_EXCEEDED,
        TelephonyError.ADMISSION_BACKEND_UNAVAILABLE,
    }
    for error in capacity:
        assert CALLER_MESSAGES_ES[error] == CAPACITY_ES
    for error in set(CALLER_MESSAGES_ES) - capacity:
        assert CALLER_MESSAGES_ES[error] == UNAVAILABLE_ES


SPOKEN = re.compile(r"<(?:Say|Speak)\b[^>]*>(.*?)</(?:Say|Speak)>", re.S)


def _spoken(body: str) -> list[str]:
    found = SPOKEN.findall(body)
    if found:
        return [t.strip() for t in found]
    # Vonage speaks through NCCO "talk" actions.
    return [a["text"] for a in json.loads(body) if a.get("action") == "talk"]


PROVIDER_CLASSES = {
    "twilio": TwilioProvider,
    "cloudonix": CloudonixProvider,
    "exotel": ExotelProvider,
    "plivo": PlivoProvider,
    "vobiz": VobizProvider,
    "vonage": VonageProvider,
}
# Vonage has generate_error_response only.
WITH_VALIDATION_RESPONSE = sorted(set(PROVIDER_CLASSES) - {"vonage"})


@pytest.mark.parametrize("provider", WITH_VALIDATION_RESPONSE)
@pytest.mark.parametrize(
    "error", [e for e in TelephonyError if e != TelephonyError.VALID]
)
def test_validation_error_is_spoken_in_spanish_from_the_table(provider, error):
    cls = PROVIDER_CLASSES[provider]

    body = cls.generate_validation_error_response(error).body.decode("utf-8")

    assert _spoken(body) == [CALLER_MESSAGES_ES[error]]
    assert TELEPHONY_ERROR_MESSAGES[error] not in body


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
@pytest.mark.parametrize(
    "error_type",
    ["SYSTEM_UNAVAILABLE", "auth_failed", "not_configured", "invalid_number", "other"],
)
def test_generic_error_is_spoken_in_spanish_and_never_speaks_the_debug_text(
    provider, error_type
):
    cls = PROVIDER_CLASSES[provider]
    debug_text = "Debug detail for the operator"

    response = cls.generate_error_response(error_type, debug_text)
    response = response[0] if isinstance(response, tuple) else response
    body = response.body.decode("utf-8")

    spoken = _spoken(body)
    assert spoken and all(t in CALLER_MESSAGES_ES.values() for t in spoken)
    assert "Debug detail" not in body


@pytest.mark.parametrize(
    ("provider", "marker"),
    [
        ("twilio", '<Say language="es-ES">'),
        ("cloudonix", '<Say language="es-ES">'),
        ("plivo", 'language="es-ES"'),
        ("vobiz", 'language="es-ES"'),
        ("vonage", '"language": "es-ES"'),
    ],
)
def test_spoken_errors_name_the_language(provider, marker):
    cls = PROVIDER_CLASSES[provider]

    response = cls.generate_error_response("SYSTEM_UNAVAILABLE", "x")
    response = response[0] if isinstance(response, tuple) else response

    assert marker in response.body.decode("utf-8")


def test_no_english_audible_error_texts_in_the_telephony_sources():
    """Grep guard: no <Say>/<Speak>/talk text with the old English error wording is left in routes or providers."""
    files = [API_ROOT / "routes" / "telephony.py"] + sorted(
        (API_ROOT / "services" / "telephony" / "providers").glob("*/*.py")
    )
    english = re.compile(
        r"(?:<(?:Say|Speak)\b[^>]*>[^<{]*"
        r"(?:Sorry|temporarily unavailable|error|failed|not configured|Invalid|occurred)"
        r"|\"text\":\s*f?\"[^\"]*(?:Sorry|error))",
        re.I,
    )
    offenders = [
        f"{path.relative_to(API_ROOT)}: {m.group(0)[:80]}"
        for path in files
        for m in english.finditer(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
