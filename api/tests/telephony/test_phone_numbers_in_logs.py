"""RUL-r1-25: a caller, callee or transfer-destination number written to a log line is masked to its last four digits
at the line that writes it ("***2671"). The sink redactor only knows "+", "00", "%2B" and Spanish 9-digit shapes, so a
number written without a prefix (here a North American one) would reach the log whole; these tests attach a sink with
no redactor at all, so the line itself must mask.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from api.services.telephony.providers.ari.provider import ARIProvider
from api.services.telephony.providers.cloudonix.provider import CloudonixProvider
from api.services.telephony.providers.exotel.provider import ExotelProvider
from api.services.telephony.providers.plivo.provider import PlivoProvider
from api.services.telephony.providers.telnyx.provider import TelnyxProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.vobiz.provider import VobizProvider
from api.services.telephony.providers.vonage.provider import VonageProvider
from api.services.workflow.tools import transfer_resolver
from api.services.workflow.tools.transfer_resolver import resolve_transfer_config

# No "+" and not a Spanish shape: the sink redactor leaves both whole.
OTHER_PARTY = "14155552671"
OWN_NUMBER = "19175550188"
DIGITS = ("4155552671", "9175550188")


@pytest.fixture
def log_lines():
    lines = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


def _assert_masked(log_lines, *tails):
    text = "\n".join(log_lines)
    for tail in tails:
        assert f"***{tail}" in text, (
            f"no masked ***{tail} in the log: the line did not run"
        )
    for digits in DIGITS:
        assert digits not in text, f"a full number reached the log: {text}"


def _session(status: int, payload: dict) -> MagicMock:
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value=payload)
    response.text = AsyncMock(return_value=json.dumps(payload))
    session = MagicMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False
    for verb in ("post", "get", "put"):
        getattr(session, verb).return_value.__aenter__.return_value = response
    return session


async def _drive(provider, method: str, payload: dict, status: int = 200, **kwargs):
    module = type(provider).__module__
    with (
        patch(
            f"{module}.aiohttp.ClientSession",
            MagicMock(return_value=_session(status, payload)),
        ),
        patch(
            f"{module}.get_backend_endpoints",
            new=AsyncMock(return_value=("https://backend.test", "wss://backend.test")),
            create=True,
        ),
        patch(
            "api.services.telephony.call_transfer_manager.get_call_transfer_manager",
            new=AsyncMock(return_value=AsyncMock()),
        ),
    ):
        await getattr(provider, method)(**kwargs)


def _transfer(**extra):
    return dict(
        destination=OTHER_PARTY, transfer_id="tx-1", conference_name="conf-1", **extra
    )


def _outbound(**extra):
    return dict(
        to_number=OTHER_PARTY,
        webhook_url="https://backend.test/hook",
        workflow_run_id=42,
        from_number=OWN_NUMBER,
        workflow_id=7,
        organization_id=9,
        **extra,
    )


def _ari():
    return ARIProvider(
        {
            "ari_endpoint": "http://asterisk.test:8088",
            "app_name": "dograh",
            "app_password": "pw",
            "stasis_app_name": "dograh-config-10",
        }
    )


def _telnyx():
    return TelnyxProvider(
        {
            "api_key": "k",
            "connection_id": "conn",
            "webhook_public_key": "pub",
            "from_numbers": [OWN_NUMBER],
        }
    )


def _plivo():
    return PlivoProvider(
        {"auth_id": "MA1", "auth_token": "t", "from_numbers": [OWN_NUMBER]}
    )


def _twilio():
    return TwilioProvider(
        {"account_sid": "AC1", "auth_token": "t", "from_numbers": [OWN_NUMBER]}
    )


def _cloudonix():
    return CloudonixProvider(
        {
            "bearer_token": "b" * 16,
            "domain_id": "acme.cloudonix.net",
            "from_numbers": [OWN_NUMBER],
        }
    )


@pytest.mark.asyncio
async def test_ari_transfer_masks_the_destination(log_lines):
    await _drive(_ari(), "transfer_call", {"id": "chan-1"}, **_transfer())

    _assert_masked(log_lines, "2671")


@pytest.mark.asyncio
async def test_ari_outbound_masks_the_callee(log_lines):
    with patch(
        "api.services.telephony.providers.ari.provider.register_channel",
        new=AsyncMock(),
    ):
        await _drive(_ari(), "initiate_call", {"state": "Down"}, **_outbound())

    _assert_masked(log_lines, "2671")


@pytest.mark.asyncio
async def test_telnyx_transfer_masks_the_destination_and_the_caller_id(log_lines):
    await _drive(
        _telnyx(), "transfer_call", {"data": {"call_control_id": "cc-1"}}, **_transfer()
    )

    _assert_masked(log_lines, "2671", "0188")


@pytest.mark.asyncio
async def test_telnyx_outbound_masks_both_numbers(log_lines):
    await _drive(
        _telnyx(), "initiate_call", {"data": {"call_control_id": "cc-1"}}, **_outbound()
    )

    _assert_masked(log_lines, "2671", "0188")


@pytest.mark.asyncio
async def test_plivo_transfer_masks_the_destination_and_the_caller_id(log_lines):
    await _drive(
        _plivo(), "transfer_call", {"request_uuid": "req-1"}, status=201, **_transfer()
    )

    _assert_masked(log_lines, "2671", "0188")


@pytest.mark.asyncio
async def test_exotel_outbound_masks_both_numbers(log_lines):
    provider = ExotelProvider(
        {
            "account_sid": "acc",
            "api_key": "k",
            "api_token": "t",
            "from_numbers": [OWN_NUMBER],
        }
    )

    await _drive(provider, "initiate_call", {"Call": {"Sid": "call-1"}}, **_outbound())

    _assert_masked(log_lines, "2671", "0188")


@pytest.mark.asyncio
async def test_twilio_outbound_and_transfer_mask_the_caller_id(log_lines):
    await _drive(_twilio(), "initiate_call", {"sid": "CA1"}, status=201, **_outbound())
    await _drive(_twilio(), "transfer_call", {"sid": "CA2"}, status=201, **_transfer())

    _assert_masked(log_lines, "0188")


@pytest.mark.asyncio
async def test_vobiz_and_vonage_outbound_mask_the_caller_id(log_lines, monkeypatch):
    vobiz = VobizProvider(
        {"auth_id": "MA1", "auth_token": "t", "from_numbers": [OWN_NUMBER]}
    )
    vonage = VonageProvider(
        {
            "application_id": "app",
            "private_key": "k",
            "signature_secret": "s",
            "from_numbers": [OWN_NUMBER],
        }
    )
    monkeypatch.setattr(vonage, "_generate_jwt", lambda: "jwt", raising=False)

    await _drive(
        vobiz, "initiate_call", {"request_uuid": "req-1"}, status=201, **_outbound()
    )
    await _drive(vonage, "initiate_call", {"uuid": "uuid-1"}, status=201, **_outbound())

    _assert_masked(log_lines, "0188")


@pytest.mark.asyncio
async def test_cloudonix_outbound_and_transfer_mask_both_numbers(log_lines):
    await _drive(
        _cloudonix(), "initiate_call", {"token": "sess-1", "domainId": 1}, **_outbound()
    )
    await _drive(_cloudonix(), "transfer_call", {"token": "sess-2"}, **_transfer())

    _assert_masked(log_lines, "2671", "0188")


def test_cloudonix_inbound_webhook_masks_the_caller_and_the_called_number(log_lines):
    CloudonixProvider.parse_inbound_webhook(
        {
            "CallSid": "c-1",
            "AccountSid": "acme.cloudonix.net",
            "From": OTHER_PARTY,
            "To": OWN_NUMBER,
        }
    )

    _assert_masked(log_lines, "2671", "0188")


def _transfer_config(source: str) -> dict:
    if source == "context_mapping":
        return {
            "destination_source": "context_mapping",
            "context_mapping": {
                "rules": [
                    {
                        "context_path": "department",
                        "routes": [
                            {"context_value": "sales", "destination": OTHER_PARTY}
                        ],
                    }
                ]
            },
        }
    if source == "dynamic":
        return {
            "destination_source": "dynamic",
            "resolver": {"type": "http", "url": "https://resolver.test"},
        }
    return {"destination": OTHER_PARTY}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["static", "context_mapping", "dynamic"])
async def test_transfer_resolver_masks_the_resolved_destination(log_lines, source):
    with patch.object(
        transfer_resolver,
        "_execute_http_resolver",
        new=AsyncMock(return_value={"transfer_context": {"destination": OTHER_PARTY}}),
    ):
        resolved = await resolve_transfer_config(
            tool=SimpleNamespace(tool_uuid="transfer-tool"),
            config=_transfer_config(source),
            arguments={},
            call_context_vars=None,
            gathered_context_vars={"department": "sales"},
            organization_id=None,
            workflow_run_id=11,
        )

    # No behaviour change in what is dialed: only the log line is masked.
    assert resolved.destination == OTHER_PARTY
    _assert_masked(log_lines, "2671")
