"""Each outbound builder that sends a ring field sends its carrier's own default ring (no behaviour change), the
bound the outbound pending lease (ring + pending_ttl_s) is sized for. Plivo states no default, so it sends none.
Telnyx (30) and ARI (30) are pinned in their own provider tests."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.services.runtime.durations import CARRIER_RING_TIMEOUT_S, cell_durations
from api.services.telephony import (  # noqa: F401  -- providers registers every carrier
    providers,
    registry,
)
from api.services.telephony.providers.plivo.provider import PlivoProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.vonage.provider import VonageProvider

FROM = ["+15551230002"]
# Carriers whose outbound builder has no ring field: they size the lease on the default ring.
NO_RING_FIELD = {"exotel", "vobiz", "cloudonix"}
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
    ("provider", "status", "body_kwarg", "field", "ring"),
    [
        (
            TwilioProvider(
                {"account_sid": "AC1", "auth_token": "t", "from_numbers": FROM}
            ),
            201,
            "data",
            "Timeout",
            60,
        ),
        (
            PlivoProvider({"auth_id": "MA1", "auth_token": "t", "from_numbers": FROM}),
            201,
            "json",
            "ring_timeout",
            None,
        ),
        (
            VonageProvider(
                {"application_id": "app", "private_key": "k", "from_numbers": FROM}
            ),
            201,
            "json",
            "ringing_timer",
            60,
        ),
    ],
    ids=["twilio", "plivo", "vonage"],
)
async def test_outbound_call_rings_the_carrier_default(
    monkeypatch, provider, status, body_kwarg, field, ring
):
    module = type(provider).__module__
    session = _session(status)
    monkeypatch.setattr(provider, "_generate_jwt", lambda: "jwt", raising=False)
    with patch(f"{module}.aiohttp.ClientSession", MagicMock(return_value=session)):
        await provider.initiate_call(
            "+15551230001", "https://hook", from_number="+15551230002"
        )

    body = session.post.call_args.kwargs[body_kwarg]
    assert body.get(field) == ring
    carrier = provider.PROVIDER_NAME
    assert cell_durations().ring_timeout_for(carrier) == (ring or 120)
    assert (
        cell_durations().outbound_pending_ttl_s(carrier)
        == cell_durations().ring_timeout_for(carrier) + cell_durations().pending_ttl_s
    )


def test_every_registered_provider_names_its_ring_for_the_outbound_lease():
    """An outbound admission looks its lease up by PROVIDER_NAME; a provider that leaves it empty would silently
    get the inbound lease on outbound calls."""
    for spec in registry.all_specs():
        name = spec.provider_cls.PROVIDER_NAME
        assert name, f"{spec.name}: PROVIDER_NAME is empty"
        assert name in CARRIER_RING_TIMEOUT_S or name in NO_RING_FIELD, (
            f"{spec.name}: {name!r} is neither in CARRIER_RING_TIMEOUT_S nor a no-ring-field carrier"
        )
