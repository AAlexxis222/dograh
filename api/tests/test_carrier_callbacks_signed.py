"""VOZ-N0-29 (VOZ-AC-B4-46b): the carrier callbacks of the closed route list verify their signature and fail closed,
and what a caller or a transfer destination hears is Spanish.

The closed route list: POST /inbound/fallback, POST /transfer-result/{id}, the legacy POST /inbound/{workflow_id},
the three Cloudonix callbacks (transfer-result, status-callback, cdr), POST /telnyx/transfer-result/{id}, and the
answer URLs /twiml, /plivo-xml, /vobiz-xml and GET /ncco (carrier signature, run agreement, run-bound answer-URL
token). Each refuses a forged request with the 401, the VOZ-AC-B0-28 body and no state change, behind the real
provider verifier (test_closed_route_refuses_a_forged_request_with_the_401_and_moves_nothing; the legacy route in
test_legacy_inbound_refused_before_authentication_is_401_and_creates_no_run). The inbound fallback answers a
database outage during its credential lookup with the Spanish unavailable sentence instead.

Outside the list, routes that already verified keep their own refusal and are not pinned to the B0-28 body here:
the Vobiz hangup and ring callbacks answer 403 (Vobiz's callback-validation docs), Vonage events their existing 401.

"State" is what a forged request could move: the run rows (state, updated_at) and every write to them, the
TransferContext, the transfer events published, and the status updates handed to the processor. A rejected
request leaves all four exactly as they were. No test needs Postgres or Redis: the collaborators are fakes.
"""

import asyncio
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

import asyncpg.exceptions
import jwt
import nacl.signing
import pytest
import sqlalchemy.exc
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger
from twilio.request_validator import RequestValidator

from api import constants
from api.errors.telephony_errors import (
    CALL_NOT_ATTENDED_MESSAGE_ES,
    CALLER_MESSAGES_ES,
    TELEPHONY_ERROR_MESSAGES,
    TelephonyError,
)
from api.routes.telephony import router
from api.services.runtime.cell_state import is_cell
from api.services.telephony import ws_auth
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
PLIVO_TOKEN = secrets.token_hex(16)
FAKE = secrets.token_hex(8)
RUN_ID = 7
WORKFLOW_ID = 33
ORG_ID = 11
TRANSFER_ID = "transfer-1"
CALLER = "+34612345678"

CAPACITY_ES = CALLER_MESSAGES_ES[TelephonyError.CONCURRENT_CALL_LIMIT]
UNAVAILABLE_ES = CALLER_MESSAGES_ES[TelephonyError.PHONE_NUMBER_NOT_CONFIGURED]
# The tag of every Spanish Twilio <Say>: a voice that speaks es-ES, and the language.
TWILIO_SPANISH_SAY = '<Say voice="Polly.Lucia" language="es-ES">'

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
        # Cloudonix's verifier compares x-cx-apikey with the bearer token, and this request sends the matching
        # key: only the provider-name check can refuse it.
        world.provider = _cloudonix()
    before = world.snapshot()

    response = _post_transfer_result(client, signed=True, headers={"x-cx-apikey": FAKE})

    _assert_b028(response)
    assert world.snapshot() == before


@pytest.mark.parametrize("carrier", ["twilio", "telnyx"])
def test_signed_callback_of_a_finished_transfer_is_401_logged_as_info_not_refused(
    client, world, log_lines, carrier
):
    # The strategy removes the context once the transfer completes; the carrier's later 'completed' or
    # 'call.hangup' is genuine. It keeps the 401, but it is not an attack and is not logged as one.
    world.manager.contexts.clear()
    before = world.snapshot()
    if carrier == "twilio":
        response = _post_transfer_result(
            client, signed=True, form={"CallStatus": "completed"}
        )
    else:
        world.provider = _telnyx()
        body = _telnyx_body()
        response = _post_telnyx(client, body, _telnyx_headers(body))

    _assert_b028(response, code="carrier_callback_unauthenticated")
    assert world.snapshot() == before
    gone = [line for line in log_lines if "code=transfer_context_gone" in line]
    assert len(gone) == 1 and "| INFO" in gone[0], log_lines
    assert not [line for line in log_lines if "carrier callback refused" in line]
    assert not [line for line in log_lines if "| WARNING" in line]


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
        "api.services.telephony.callback_auth.get_telephony_provider_by_id",
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
        "api.services.telephony.callback_auth.get_telephony_provider_by_id",
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
    assert re.search(re.escape(TWILIO_SPANISH_SAY) + r"([^<]+)</Say>", response.text)
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


def _db_error(cls, *, invalidated: bool):
    return cls(
        "SELECT 1", {}, Exception("driver error"), connection_invalidated=invalidated
    )


# What this stack raises when Postgres is down, restarting, saturated or drops the connection.
LOOKUP_OUTAGES = {
    "connection_refused": lambda: ConnectionRefusedError("connection refused"),
    "builtin_timeout": lambda: TimeoutError(),
    "pool_exhausted": lambda: sqlalchemy.exc.TimeoutError("QueuePool limit reached"),
    "interface_error": lambda: _db_error(
        sqlalchemy.exc.InterfaceError, invalidated=False
    ),
    "connection_invalidated": lambda: _db_error(
        sqlalchemy.exc.OperationalError, invalidated=True
    ),
    "cannot_connect_now": lambda: asyncpg.exceptions.CannotConnectNowError(
        "the database system is starting up"
    ),
    "too_many_connections": lambda: asyncpg.exceptions.TooManyConnectionsError(
        "too many connections"
    ),
    "admin_shutdown": lambda: asyncpg.exceptions.AdminShutdownError("shutting down"),
    "connection_lost": lambda: asyncpg.exceptions.PostgresConnectionError("lost"),
}


def _lookup_raises(world, stage: str, error: Exception):
    """Make the fallback's credential lookup raise ``error`` at ``stage``: the inbound route or the configuration."""
    _route_for_twilio(world)
    if stage == "route":
        world.db.find_inbound_route_by_account = AsyncMock(side_effect=error)
        return patch(
            "api.services.telephony.callback_auth.get_telephony_provider_by_id",
            new=AsyncMock(return_value=world.provider),
        )
    return patch(
        "api.services.telephony.callback_auth.get_telephony_provider_by_id",
        new=AsyncMock(side_effect=error),
    )


@pytest.mark.parametrize("stage", ["route", "configuration"])
@pytest.mark.parametrize("outage", sorted(LOOKUP_OUTAGES))
def test_fallback_during_a_database_outage_speaks_the_unavailable_sentence(
    client, world, log_lines, stage, outage
):
    # Twilio calls the fallback exactly when /inbound/run failed: in an outage the caller hears the Spanish
    # sentence, not the carrier's English application error. Nothing is written and nothing personal is logged.
    before = world.snapshot()
    with _lookup_raises(world, stage, LOOKUP_OUTAGES[outage]()):
        response = _post_fallback(client, _fallback_form())

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/xml")
    assert _spoken(response.text) == [UNAVAILABLE_ES]
    assert world.snapshot() == before
    text = "\n".join(log_lines).lower()
    assert "ca-fallback" in text
    assert CALLER not in text and "612345678" not in text and BODY_MARKER not in text
    assert "custom-header-value" not in text and "twilioproxy" not in text


@pytest.mark.parametrize(
    "error",
    [
        lambda: _db_error(sqlalchemy.exc.DataError, invalidated=False),
        lambda: _db_error(sqlalchemy.exc.ProgrammingError, invalidated=False),
        lambda: _db_error(sqlalchemy.exc.OperationalError, invalidated=False),
        lambda: ValueError("config parked"),
        lambda: AttributeError("'int' object has no attribute 'upper'"),
    ],
    ids=["data_error", "programming_error", "operational_alive", "value", "attribute"],
)
@pytest.mark.parametrize("stage", ["route", "configuration"])
def test_fallback_lookup_error_that_is_not_an_outage_stays_the_401(
    client, world, stage, error
):
    before = world.snapshot()
    with _lookup_raises(world, stage, error()):
        response = _post_fallback(client, _fallback_form())

    _assert_b028(response)
    assert world.snapshot() == before


# JSON bodies that parse but are not an object: no carrier sends one, and no field can be read from them.
NOT_AN_OBJECT = ["[1]", "3", '"text"', "true"]


@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_fallback_with_a_json_body_that_is_not_an_object_is_401_not_500(world, body):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)
    before = world.snapshot()

    response = client.post(
        FALLBACK_PATH, content=body, headers={"content-type": "application/json"}
    )

    _assert_b028(response)
    assert world.snapshot() == before


def test_fallback_with_a_malformed_unsigned_body_is_401(
    client, world, fallback_provider
):
    before = world.snapshot()

    response = client.post(
        FALLBACK_PATH,
        json={**_fallback_form(), "ToCountry": 123, "To": 34911000111},
        headers={"User-Agent": "TwilioProxy/1.1"},
    )

    _assert_b028(response)
    assert world.snapshot() == before


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


# ---------------------------------------------------------------------------------------------------------
# Vonage GET /ncco (the answer URL: its NCCO carries the media-stream URL and its token)
# ---------------------------------------------------------------------------------------------------------

NCCO_PATH = "/telephony/ncco"
VONAGE_SECRET = secrets.token_hex(16)
VONAGE_APP = "aaaaaaaa-bbbb-cccc-dddd-0123456789ab"
NCCO_TOKEN = "ws-capability-" + secrets.token_hex(8)
NCCO = [
    {
        "action": "connect",
        "endpoint": [
            {"type": "websocket", "uri": f"wss://backend.test/ws?token={NCCO_TOKEN}"}
        ],
    }
]
NCCO_QUERY = {
    "workflow_id": WORKFLOW_ID,
    "workflow_run_id": RUN_ID,
    "organization_id": ORG_ID,
}


def _vonage_jwt(secret: str = VONAGE_SECRET) -> dict:
    """The signed-webhook JWT Vonage sends in Authorization (a GET has no body, so no payload_hash)."""
    token = jwt.encode(
        {
            "iat": int(time.time()),
            "jti": "test-jti",
            "iss": "Vonage",
            "api_key": FAKE,
            "application_id": VONAGE_APP,
        },
        secret,
        algorithm="HS256",
    )
    return {"authorization": f"Bearer {token}"}


def _get_ncco(client, world, *, query=None, headers=None):
    world.provider = VonageProvider(
        {
            "api_key": FAKE,
            "api_secret": secrets.token_hex(8),
            "application_id": VONAGE_APP,
            "private_key": FAKE,
            "signature_secret": VONAGE_SECRET,
            "from_numbers": ["15551230002"],
        }
    )
    world.provider.get_webhook_response = AsyncMock(return_value=json.dumps(NCCO))
    response = client.get(
        NCCO_PATH,
        params=query or NCCO_QUERY,
        headers=headers if headers is not None else _vonage_jwt(),
    )
    return response, world.provider.get_webhook_response


def test_ncco_signed_returns_the_ncco(client, world):
    response, build = _get_ncco(client, world)

    assert response.status_code == 200
    assert response.json() == NCCO
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


@pytest.mark.parametrize("headers", [{}, "other_secret", "not_a_jwt"])
def test_ncco_bad_signature_is_401_without_ncco_or_token(client, world, headers):
    if headers == "other_secret":
        headers = _vonage_jwt(secret=OTHER_TOKEN)
    elif headers == "not_a_jwt":
        headers = {"authorization": "Bearer Zm9yZ2Vk"}
    before = world.snapshot()

    response, build = _get_ncco(client, world, headers=headers)

    _assert_b028(response)
    assert NCCO_TOKEN not in response.text and "websocket" not in response.text
    build.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize(
    "tampered", [{"workflow_id": 34}, {"organization_id": 12}, {"workflow_run_id": 8}]
)
def test_ncco_query_that_disagrees_with_the_run_is_401_even_if_signed(
    client, world, tampered
):
    response, build = _get_ncco(client, world, query={**NCCO_QUERY, **tampered})

    _assert_b028(response)
    assert NCCO_TOKEN not in response.text
    build.assert_not_awaited()


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


PUBLIC_BACKEND = "https://public.example"
# What Vobiz signs: the public URL the platform gave it (get_backend_endpoints), not the one the app sees.
VOBIZ_SIGNED_URL = f"{PUBLIC_BACKEND}/api/v1/telephony/vobiz-xml"


def _post_vobiz_xml(client, world, *, query=None, headers=None, build="<Response/>"):
    world.provider = _vobiz()
    world.provider.get_webhook_response = AsyncMock(return_value=build)
    path = "/telephony/vobiz-xml"
    with patch(
        "api.services.telephony.providers.vobiz.routes.get_backend_endpoints",
        new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
    ):
        response = client.post(
            f"{path}?{urlencode(query or VOBIZ_QUERY)}",
            data={"CallUUID": "uuid-1"},
            headers=headers
            if headers is not None
            else _vobiz_headers(VOBIZ_SIGNED_URL),
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
        headers = _vobiz_headers(VOBIZ_SIGNED_URL, token=OTHER_TOKEN)
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


def test_vobiz_xml_verifies_the_public_url_vobiz_signed_not_the_one_the_app_sees(
    client, world
):
    # Behind a TLS proxy or tunnel the app sees http://<internal host>; Vobiz signed the URL the platform gave it.
    with patch(
        "api.services.telephony.providers.vobiz.routes.get_backend_endpoints",
        new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
    ):
        response, build = _post_vobiz_xml(
            client,
            world,
            headers=_vobiz_headers(f"{PUBLIC_BACKEND}/api/v1/telephony/vobiz-xml"),
        )

    assert response.status_code == 200, response.text
    build.assert_awaited_once_with(WORKFLOW_ID, ORG_ID, RUN_ID)


def test_vobiz_xml_signed_over_the_internal_url_is_401(client, world):
    with patch(
        "api.services.telephony.providers.vobiz.routes.get_backend_endpoints",
        new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
    ):
        response, build = _post_vobiz_xml(
            client, world, headers=_vobiz_headers(BASE + "/telephony/vobiz-xml")
        )

    _assert_b028(response)
    build.assert_not_awaited()


@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_parse_webhook_request_refuses_a_json_body_that_is_not_an_object(body):
    from api.utils.telephony_helper import parse_webhook_request

    class JsonRequest:
        query_params = {}

        async def body(self):
            return body.encode()

        async def json(self):
            return json.loads(body)

        async def form(self):
            raise AssertionError("a JSON body is not read as a form")

    with pytest.raises(ValueError):
        asyncio.run(parse_webhook_request(JsonRequest()))


@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_vobiz_xml_with_a_body_that_is_not_an_object_is_401_before_any_lookup(
    client, world, body
):
    world.provider = _vobiz()
    world.provider.get_webhook_response = AsyncMock(return_value="<Response/>")
    lookup = AsyncMock(side_effect=world.db.get_workflow_run_by_id)
    world.db.get_workflow_run_by_id = lookup
    before = world.snapshot()
    with patch(
        "api.services.telephony.providers.vobiz.routes.get_backend_endpoints",
        new=AsyncMock(return_value=(PUBLIC_BACKEND, "wss://public.example")),
    ):
        response = client.post(
            f"/telephony/vobiz-xml?{urlencode(VOBIZ_QUERY)}",
            content=body,
            headers={
                **_vobiz_headers(VOBIZ_SIGNED_URL),
                "content-type": "application/json",
            },
        )

    _assert_b028(response)
    lookup.assert_not_awaited()
    world.provider.get_webhook_response.assert_not_awaited()
    assert world.snapshot() == before


@pytest.mark.parametrize("kind", ["hangup", "ring"])
@pytest.mark.parametrize("body", NOT_AN_OBJECT)
def test_vobiz_status_callback_with_a_body_that_is_not_an_object_is_403(
    world, kind, body
):
    # Each Vobiz callback keeps one refusal status: the 403 it answers for a bad signature (Vobiz docs).
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)
    lookup = AsyncMock(side_effect=world.db.get_workflow_run_by_id)
    world.db.get_workflow_run_by_id = lookup
    before = world.snapshot()
    with patch("api.services.telephony.providers.vobiz.routes.db_client", world.db):
        response = client.post(
            f"/telephony/vobiz/{kind}-callback/{RUN_ID}",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 403, response.text
    lookup.assert_not_awaited()
    assert world.snapshot() == before


# ---------------------------------------------------------------------------------------------------------
# /plivo-xml and /twiml: the answer URL; the run decides whose credentials verify it, never the query
# ---------------------------------------------------------------------------------------------------------

ATTACKER_ORG = 12
PLIVO_NONCE = "12345678901234567890"


def _plivo(token: str = PLIVO_TOKEN) -> PlivoProvider:
    return PlivoProvider(
        {"auth_id": "MA123", "auth_token": token, "from_numbers": ["+15551230002"]}
    )


def _plivo_headers(url: str, form: dict, provider: PlivoProvider) -> dict:
    payload = f"{provider._construct_post_url(url, form)}.{PLIVO_NONCE}"
    digest = hmac.new(
        provider.auth_token.encode(), payload.encode(), hashlib.sha256
    ).digest()
    return {
        "x-plivo-signature-v3": base64.b64encode(digest).decode("ascii"),
        "x-plivo-signature-v3-nonce": PLIVO_NONCE,
    }


ANSWER_URLS = {
    "plivo-xml": {
        "module": "api.services.telephony.providers.plivo.routes",
        "provider": _plivo,
        "form": {"CallUUID": "chosen-call-id"},
        "sign": lambda url, form, provider: _plivo_headers(url, form, provider),
    },
    "twiml": {
        "module": "api.services.telephony.providers.twilio.routes",
        "provider": lambda token: TwilioProvider(
            {"account_sid": "AC123", "auth_token": token, "from_numbers": []}
        ),
        "form": {"CallSid": "chosen-call-id"},
        "sign": lambda url, form, provider: {
            "X-Twilio-Signature": _twilio_signature(url, form, provider.auth_token)
        },
    },
}


def _answer_url_patches(module: str, world, resolve):
    """The answer route's own db and provider lookup, and callback_auth's, all resolving through ``resolve``."""
    return (
        patch(f"{module}.db_client", world.db),
        patch(f"{module}.get_telephony_provider_for_run", side_effect=resolve),
        patch(
            "api.services.telephony.callback_auth.get_telephony_provider_for_run",
            side_effect=resolve,
        ),
    )


@pytest.mark.parametrize("route", sorted(ANSWER_URLS))
def test_answer_url_naming_another_org_is_401_even_if_signed_by_that_org(
    client, world, route
):
    # The caller's own org has a configuration of the same carrier and signs with its own token; the run
    # belongs to ORG_ID. The query's organization_id must not choose the credentials.
    spec = ANSWER_URLS[route]
    own, attacker = spec["provider"](PLIVO_TOKEN), spec["provider"](OTHER_TOKEN)
    providers = {ORG_ID: own, ATTACKER_ORG: attacker}
    for provider in providers.values():
        provider.get_webhook_response = AsyncMock(return_value="<Response/>")

    async def resolve(run, organization_id):
        return providers[organization_id]

    query = {
        "workflow_id": WORKFLOW_ID,
        "workflow_run_id": RUN_ID,
        "organization_id": ATTACKER_ORG,
    }
    path = f"/telephony/{route}?{urlencode(query)}"
    before = world.snapshot()
    db, route_lookup, auth_lookup = _answer_url_patches(spec["module"], world, resolve)
    with db, route_lookup, auth_lookup:
        response = client.post(
            path,
            data=spec["form"],
            headers=spec["sign"](BASE + path, spec["form"], attacker),
        )

    _assert_b028(response)
    assert world.snapshot() == before
    own.get_webhook_response.assert_not_awaited()
    attacker.get_webhook_response.assert_not_awaited()


def test_plivo_xml_bad_signature_is_401_and_writes_nothing(client, world):
    world.provider = _plivo()
    world.provider.get_webhook_response = AsyncMock(return_value="<Response/>")
    query = {
        "workflow_id": WORKFLOW_ID,
        "workflow_run_id": RUN_ID,
        "organization_id": ORG_ID,
    }
    before = world.snapshot()
    db, route_lookup, auth_lookup = _answer_url_patches(
        ANSWER_URLS["plivo-xml"]["module"], world, world.provider_for_run
    )
    with db, route_lookup, auth_lookup:
        response = client.post(
            f"/telephony/plivo-xml?{urlencode(query)}",
            data={"CallUUID": "chosen-call-id"},
            headers={
                "x-plivo-signature-v3": "Zm9yZ2Vk",
                "x-plivo-signature-v3-nonce": PLIVO_NONCE,
            },
        )

    _assert_b028(response)
    assert world.snapshot() == before
    world.provider.get_webhook_response.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------
# Answer-URL token: a carrier signature binds no run (Vobiz signs the bare URL, the Vonage GET JWT has no
# payload hash and no freshness), so a captured one replays against every run of its org with a consistent
# query. The answer URL carries a run-bound token that the four answer handlers check.
# ---------------------------------------------------------------------------------------------------------

ANSWER_ROUTES = ["ncco", "plivo-xml", "twiml", "vobiz-xml"]
OTHER_RUN = 8


@pytest.fixture
def answer_secret(monkeypatch):
    monkeypatch.setattr(constants, "TELEPHONY_WS_TOKEN_SECRET", secrets.token_hex(16))
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
# What the log never holds: the raw body, caller names, caller or called numbers, the media-WS token
# ---------------------------------------------------------------------------------------------------------

CALLED = "+34911000111"
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


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
@pytest.mark.parametrize(
    "error", [e for e in TelephonyError if e != TelephonyError.VALID]
)
def test_validation_error_is_spoken_in_spanish_from_the_table(provider, error):
    cls = PROVIDER_CLASSES[provider]

    body = cls.generate_validation_error_response(error).body.decode("utf-8")

    assert _spoken(body) == [CALLER_MESSAGES_ES[error]]
    assert TELEPHONY_ERROR_MESSAGES[error] not in body


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
def test_validation_error_given_as_its_string_value_is_spoken_from_the_table(
    provider,
):
    cls = PROVIDER_CLASSES[provider]

    body = cls.generate_validation_error_response("QUOTA_EXCEEDED").body.decode("utf-8")

    assert _spoken(body) == [CAPACITY_ES]


def test_vonage_inbound_run_refusal_is_an_ncco_that_speaks_the_table_sentence(
    client, world
):
    # Vonage fetches /inbound/run as its answer URL and reads the answer as NCCO JSON: a TwiML <Hangup/> there
    # is an error on Vonage's side and the caller hears nothing.
    normalized = SimpleNamespace(
        provider="vonage",
        direction="inbound",
        to_number=CALLED,
        from_number=CALLER,
        to_country="ES",
        from_country="ES",
        account_id=VONAGE_APP,
        call_id="vonage-uuid",
    )
    with (
        patch("api.routes.telephony.normalize_webhook_data", return_value=normalized),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=VonageProvider),
        ),
    ):
        response = client.get(
            "/telephony/inbound/run", params={"uuid": "vonage-uuid", "to": "34911"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    talks = [a for a in response.json() if a["action"] == "talk"]
    assert [(a["text"], a["language"]) for a in talks] == [(UNAVAILABLE_ES, "es-ES")]


# The NCCO actions Vonage accepts; the call ends when the actions run out (there is no "hangup" action).
NCCO_ACTIONS = {
    "talk",
    "stream",
    "input",
    "record",
    "conversation",
    "connect",
    "notify",
    "wait",
}


@pytest.mark.parametrize(
    "build",
    [
        lambda: VonageProvider.generate_error_response("SYSTEM_UNAVAILABLE", "x"),
        lambda: VonageProvider.generate_validation_error_response(
            TelephonyError.QUOTA_EXCEEDED
        ),
    ],
    ids=["error", "validation"],
)
def test_vonage_error_ncco_has_only_actions_vonage_accepts(build):
    actions = [a["action"] for a in json.loads(build().body)]

    assert actions == ["talk"]
    assert set(actions) <= NCCO_ACTIONS


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
        ("twilio", TWILIO_SPANISH_SAY),
        ("cloudonix", '<Say language="es-ES">'),
        ("plivo", 'language="es-ES"'),
        ("vobiz", 'language="es-ES"'),
        ("vonage", '"language": "es-ES"'),
        pytest.param(
            "exotel",
            None,
            marks=pytest.mark.skip(
                reason="GATED exotel_language_unverified: Exotel's <Say> language and voice attributes are not "
                "documented offline and there is no account to try them (R9), so none is guessed; VOZ-N0-40 Step 2"
            ),
        ),
    ],
)
def test_spoken_errors_name_the_language(provider, marker):
    cls = PROVIDER_CLASSES[provider]

    response = cls.generate_error_response("SYSTEM_UNAVAILABLE", "x")
    response = response[0] if isinstance(response, tuple) else response

    assert marker in response.body.decode("utf-8")


def test_every_spanish_twilio_say_names_a_voice_that_speaks_es_es():
    # Twilio's default voice ("man") does not list es-ES and "alice" is retired: each Spanish <Say> names the
    # Polly voice, or Twilio reads the sentence with the wrong voice or drops it.
    from api.services.telephony.providers.twilio.routes import _connect_action_twiml

    bodies = {
        "error": TwilioProvider.generate_error_response("SYSTEM_UNAVAILABLE", "x"),
        "validation": TwilioProvider.generate_validation_error_response(
            TelephonyError.CONCURRENT_CALL_LIMIT
        ),
    }
    bodies = {name: r.body.decode("utf-8") for name, r in bodies.items()}
    bodies["connect-action"] = _connect_action_twiml(call_ended=False)

    for name, body in bodies.items():
        says = re.findall(r"<Say\b[^>]*>", body)
        assert says == [TWILIO_SPANISH_SAY], name


TRANSFER_GREETING = "Tiene una llamada transferida. Le conectamos ahora."
# What a caller or a transfer destination may hear, and the names that produce it in the sources.
AUDIBLE_TEXTS = set(CALLER_MESSAGES_ES.values()) | {
    TRANSFER_GREETING,
    CALL_NOT_ATTENDED_MESSAGE_ES,
}
AUDIBLE_PRODUCERS = (
    "caller_message_es(",
    "caller_error_text(",
    "UNAVAILABLE_MESSAGE_ES",
    "TRANSFER_GREETING_ES",
    "CALL_NOT_ATTENDED_MESSAGE_ES",
)
SOURCE_SPEECH = re.compile(r"<(Say|Speak)\b[^>]*>(.*?)</\1>", re.S)
SOURCE_TALK_TEXT = re.compile(r'"text":\s*([^\n]+)')
# The seam between two adjacent string literals of one implicitly concatenated string.
LITERAL_SEAM = re.compile(r"""["']\s*f?["']""")


def _audible(source: str) -> bool:
    """Whether ``source`` (what a <Say>/<Speak> or a talk "text" holds in the code) can only be an agreed text:
    a literal of AUDIBLE_TEXTS, or one expression of an AUDIBLE_PRODUCERS name."""
    text = LITERAL_SEAM.sub("", source).strip().rstrip(",").strip()
    if text[:1] == "{" and text[-1:] == "}" and text.count("{") == 1:
        text = text[1:-1]
    elif text[:1] in "\"'" and text[-1:] == text[:1]:
        return text[1:-1] in AUDIBLE_TEXTS
    elif text in AUDIBLE_TEXTS:
        return True
    return text.startswith(AUDIBLE_PRODUCERS) or any(
        f"({name}" in text for name in AUDIBLE_PRODUCERS
    )


def test_no_audible_text_outside_the_agreed_set_in_the_telephony_sources():
    """Grep guard: every literal inside <Say>/<Speak> and every talk text in the telephony routes and services is
    one of the agreed Spanish texts (VOZ-AC-B4-46b: no English, no free text)."""
    files = [API_ROOT / "routes" / "telephony.py"] + sorted(
        (API_ROOT / "services" / "telephony").rglob("*.py")
    )
    offenders = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        spoken = [m.group(2) for m in SOURCE_SPEECH.finditer(source)]
        spoken += [m.group(1) for m in SOURCE_TALK_TEXT.finditer(source)]
        offenders += [
            f"{path.relative_to(API_ROOT)}: {text.strip()[:80]}"
            for text in spoken
            if not _audible(text)
        ]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "Connecting you now.",
        '"Sorry, an error occurred",',
        "Hola {name}",
        "{escape(message)}",
        "{message}",
        "{UNAVAILABLE_MESSAGE_ES} {name}",
    ],
)
def test_the_audible_text_guard_refuses_free_or_english_text(source):
    assert not _audible(source)


def _twilio_transfer_twiml() -> str:
    """The inline TwiML Twilio's transfer_call sends for the destination leg, captured before it is sent."""
    sent = []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, data=None, **kwargs):
            sent.append(data)
            raise RuntimeError("test: stop after building the request")

    module = "api.services.telephony.providers.twilio.provider"
    with (
        patch(f"{module}.aiohttp.ClientSession", lambda **kwargs: Session()),
        patch(
            f"{module}.get_backend_endpoints",
            new=AsyncMock(return_value=("https://backend.test", "")),
        ),
    ):
        try:
            asyncio.run(
                _twilio().transfer_call(
                    destination="+34911222333",
                    transfer_id=TRANSFER_ID,
                    conference_name="conf-1",
                )
            )
        except Exception:
            pass
    return sent[0]["Twiml"]


def _plivo_transfer_xml() -> str:
    from api.services.telephony.providers.plivo.routes import _conference_xml_response

    return _conference_xml_response("conf-1").body.decode("utf-8")


@pytest.mark.parametrize(
    ("build", "tag"),
    [
        (_twilio_transfer_twiml, TWILIO_SPANISH_SAY),
        (_plivo_transfer_xml, '<Speak language="es-ES">'),
        (
            lambda: CloudonixProvider._conference_join_cxml(
                "conf-1", "https://backend.test/cb"
            ),
            '<Say language="es-ES">',
        ),
    ],
    ids=["twilio", "plivo", "cloudonix"],
)
def test_transfer_greeting_is_the_spanish_sentence_in_a_spanish_voice(build, tag):
    body = build()

    assert re.findall(r"<(?:Say|Speak)\b[^>]*>", body) == [tag]
    assert _spoken(body) == [TRANSFER_GREETING]


@pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
def test_generic_error_logs_the_operator_text(provider, log_lines):
    # The second argument of generate_error_response is for the operator: logged, never spoken.
    cls = PROVIDER_CLASSES[provider]

    cls.generate_error_response("auth_failed", "Debug detail for the operator")

    logged = [line for line in log_lines if "Debug detail for the operator" in line]
    assert len(logged) == 1
    assert "auth_failed" in logged[0] and provider in logged[0]
