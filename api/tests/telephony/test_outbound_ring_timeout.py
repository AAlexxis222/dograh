"""Every outbound builder caps the ringing at durations.ring_timeout_s: the outbound pending lease of a concurrency
slot (outbound_pending_ttl_s = ring_timeout_s + pending_ttl_s) only holds if the carrier stops ringing by then.
Telnyx and ARI are pinned in their own provider tests."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.services.runtime.durations import cell_durations
from api.services.telephony.providers.plivo.provider import PlivoProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.vonage.provider import VonageProvider

FROM = ["+15551230002"]
RESPONSE = {"sid": "CA1", "request_uuid": "req-1", "uuid": "uuid-1"}


def _session(status: int) -> MagicMock:
    response = MagicMock(status=status)
    response.json = AsyncMock(return_value=RESPONSE)
    response.text = AsyncMock(return_value=json.dumps(RESPONSE))
    session = MagicMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False
    session.post.return_value.__aenter__.return_value = response
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("carrier", "provider", "status", "body_kwarg", "field"),
    [
        (
            "twilio",
            TwilioProvider(
                {"account_sid": "AC1", "auth_token": "t", "from_numbers": FROM}
            ),
            201,
            "data",
            "Timeout",
        ),
        (
            "plivo",
            PlivoProvider({"auth_id": "MA1", "auth_token": "t", "from_numbers": FROM}),
            201,
            "json",
            "ring_timeout",
        ),
        (
            "vonage",
            VonageProvider(
                {"application_id": "app", "private_key": "k", "from_numbers": FROM}
            ),
            201,
            "json",
            "ringing_timer",
        ),
    ],
    ids=["twilio", "plivo", "vonage"],
)
async def test_outbound_call_rings_at_most_ring_timeout(
    monkeypatch, carrier, provider, status, body_kwarg, field
):
    module = type(provider).__module__
    session = _session(status)
    monkeypatch.setattr(provider, "_generate_jwt", lambda: "jwt", raising=False)
    with patch(f"{module}.aiohttp.ClientSession", MagicMock(return_value=session)):
        await provider.initiate_call(
            "+15551230001", "https://hook", from_number="+15551230002"
        )

    body = session.post.call_args.kwargs[body_kwarg]
    assert body[field] == cell_durations().ring_timeout_for(carrier)
    assert (
        body[field] <= cell_durations().ring_timeout_s
    )  # within the outbound pending lease
