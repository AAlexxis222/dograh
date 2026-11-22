"""Shared harness of the carrier-callback tests under api/tests/telephony/callbacks/ (fixtures in its conftest.py).

VOZ-N0-29 (VOZ-AC-B4-46b): the carrier callbacks of the closed route list verify their signature and fail closed,
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
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import jwt
import nacl.signing
from twilio.request_validator import RequestValidator

from api.errors.telephony_errors import CALLER_MESSAGES_ES, TelephonyError
from api.services.telephony.providers.cloudonix.provider import CloudonixProvider
from api.services.telephony.providers.plivo.provider import PlivoProvider
from api.services.telephony.providers.telnyx.provider import TelnyxProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.vobiz.provider import VobizProvider
from api.services.telephony.providers.vonage.provider import VonageProvider
from api.services.telephony.transfer_event_protocol import TransferContext

API_ROOT = pathlib.Path(__file__).resolve().parents[2]


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


# JSON bodies that parse but are not an object: no carrier sends one, and no field can be read from them.
NOT_AN_OBJECT = ["[1]", "3", '"text"', "true"]


# ---------------------------------------------------------------------------------------------------------
# Vonage GET /ncco (the answer URL: its NCCO carries the media-stream URL and its token)
# ---------------------------------------------------------------------------------------------------------


NCCO_PATH = "/telephony/ncco"


VONAGE_SECRET = secrets.token_hex(16)


VONAGE_APP = "aaaaaaaa-bbbb-cccc-dddd-0123456789ab"


NCCO_TOKEN = secrets.token_hex(16)


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


def _post_telnyx(client, body, headers):
    return client.post(TELNYX_PATH, content=body, headers=headers)


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


# ---------------------------------------------------------------------------------------------------------
# /plivo-xml and /twiml: the answer URL; the run decides whose credentials verify it, never the query
# ---------------------------------------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------------------------------------
# Answer-URL token: a carrier signature binds no run (Vobiz signs the bare URL, the Vonage GET JWT has no
# payload hash and no freshness), so a captured one replays against every run of its org with a consistent
# query. The answer URL carries a run-bound token that the four answer handlers check.
# ---------------------------------------------------------------------------------------------------------


ANSWER_ROUTES = ["ncco", "plivo-xml", "twiml", "vobiz-xml"]


# ---------------------------------------------------------------------------------------------------------
# What the log never holds: the raw body, caller names, caller or called numbers, the media-WS token
# ---------------------------------------------------------------------------------------------------------


CALLED = "+34911000111"


# ---------------------------------------------------------------------------------------------------------
# What the caller hears
# ---------------------------------------------------------------------------------------------------------


SPOKEN = re.compile(r"<(?:Say|Speak)\b[^>]*>(.*?)</(?:Say|Speak)>", re.S)


def _spoken(body: str) -> list[str]:
    found = SPOKEN.findall(body)
    if found:
        return [t.strip() for t in found]
    # Vonage speaks through NCCO "talk" actions.
    return [a["text"] for a in json.loads(body) if a.get("action") == "talk"]
