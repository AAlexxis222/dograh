"""Region-aware Twilio endpoints and regional credentials (VOZ-AC-B4-41)."""

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from twilio.request_validator import RequestValidator

from api.services.telephony.providers.twilio import SPEC, transport
from api.services.telephony.providers.twilio import provider as provider_module
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.providers.twilio.region import (
    RegionError,
    resolve_twilio_endpoint,
)

IE1 = {"account_sid": "AC1", "auth_token": "t-ie1"}
US1 = {"account_sid": "ACUS", "auth_token": "t-us1"}


def test_eu_cell_defaults_to_ie1():
    ep = resolve_twilio_endpoint({"credentials": {"ie1": IE1}}, cell_policy="eu")
    assert ep.base_url == "https://api.dublin.ie1.twilio.com/2010-04-01/Accounts/AC1"
    assert (ep.region, ep.edge, ep.auth_token) == ("ie1", "dublin", "t-ie1")


def test_us1_in_eu_cell_is_rejected_with_named_error():
    with pytest.raises(RegionError) as e:
        resolve_twilio_endpoint(
            {"region": "us1", "credentials": {"us1": US1}}, cell_policy="eu"
        )
    assert e.value.code == "carrier_region_not_eu" and e.value.hint and e.value.reason


def test_missing_regional_credentials_is_named_not_401():
    with pytest.raises(RegionError) as e:
        resolve_twilio_endpoint(
            {"region": "ie1", "credentials": {"us1": US1}}, cell_policy="eu"
        )
    assert e.value.code == "carrier_region_credentials_missing"
    assert e.value.hint and e.value.reason


def test_legacy_flat_config_is_us1_credentials_and_fails_closed_in_eu_cell():
    flat = {"account_sid": "ACUS", "auth_token": "t-us1"}
    ep = resolve_twilio_endpoint(flat, cell_policy=None)
    assert ep.base_url == "https://api.twilio.com/2010-04-01/Accounts/ACUS"
    assert (ep.region, ep.edge) == ("us1", None)
    with pytest.raises(RegionError) as e:
        resolve_twilio_endpoint(flat, cell_policy="eu")
    assert e.value.code == "carrier_region_credentials_missing"


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_non_eu_exception_needs_a_reason(reason):
    config = {
        "region": "us1",
        "credentials": {"us1": US1},
        "allow_non_eu_carrier_region": True,
        "allow_non_eu_carrier_region_reason": reason,
    }
    with pytest.raises(RegionError) as e:
        resolve_twilio_endpoint(config, cell_policy="eu")
    assert e.value.code == "carrier_region_not_eu"


def test_non_eu_exception_with_reason_is_allowed():
    config = {
        "region": "us1",
        "credentials": {"us1": US1},
        "allow_non_eu_carrier_region": True,
        "allow_non_eu_carrier_region_reason": "pilot with US-only number",
    }
    ep = resolve_twilio_endpoint(config, cell_policy="eu")
    assert ep.base_url == "https://api.twilio.com/2010-04-01/Accounts/ACUS"


def test_regional_credentials_win_over_flat_ones_for_us1():
    config = {**IE1, "credentials": {"us1": US1}}
    ep = resolve_twilio_endpoint(config, cell_policy=None)
    assert ep.account_sid == "ACUS"


def test_region_without_known_edge_is_a_named_error():
    with pytest.raises(RegionError) as e:
        resolve_twilio_endpoint(
            {"region": "xx9", "credentials": {"xx9": IE1}}, cell_policy=None
        )
    assert e.value.code == "carrier_region_edge_missing"


def test_unknown_policy_fails_closed():
    with pytest.raises(RegionError) as e:
        resolve_twilio_endpoint({"credentials": {"ie1": IE1}}, cell_policy="ue")
    assert e.value.code == "carrier_region_policy_invalid"


def test_no_hardcoded_us1_host_left():
    src = pathlib.Path(provider_module.__file__).read_text(encoding="utf-8")
    assert "https://api.twilio.com" not in src


def test_policy_unset_keeps_the_upstream_url(monkeypatch):
    monkeypatch.delenv("CARRIER_REGION_POLICY", raising=False)
    provider = TwilioProvider({"account_sid": "AC1", "auth_token": "t"})
    assert provider.base_url == "https://api.twilio.com/2010-04-01/Accounts/AC1"


def test_provider_in_eu_cell_uses_the_regional_host_and_credentials(monkeypatch):
    monkeypatch.setenv("CARRIER_REGION_POLICY", "eu")
    provider = TwilioProvider({"credentials": {"ie1": IE1}, "from_numbers": ["+34"]})
    assert provider.base_url.startswith("https://api.dublin.ie1.twilio.com/")
    assert (provider.account_sid, provider.auth_token) == ("AC1", "t-ie1")


def test_config_loader_forwards_the_region_keys():
    loaded = SPEC.config_loader(
        {
            "account_sid": "ACUS",
            "auth_token": "t-us1",
            "region": "ie1",
            "edge": "dublin",
            "credentials": {"ie1": IE1},
            "allow_non_eu_carrier_region": True,
            "allow_non_eu_carrier_region_reason": "r",
            "fallback_url": "https://fallback.example/twiml",
        }
    )
    assert loaded["credentials"] == {"ie1": IE1}
    assert loaded["region"] == "ie1" and loaded["fallback_url"]
    # Legacy configs produce exactly the keys they produced before.
    assert set(SPEC.config_loader({"account_sid": "ACUS", "auth_token": "x"})) == {
        "provider",
        "account_sid",
        "auth_token",
        "from_numbers",
    }


@pytest.mark.asyncio
async def test_webhook_signature_is_validated_with_the_resolved_region_token(
    monkeypatch,
):
    monkeypatch.setenv("CARRIER_REGION_POLICY", "eu")
    provider = TwilioProvider(
        {"auth_token": "t-us1", "credentials": {"ie1": IE1}, "from_numbers": ["+34"]}
    )
    url, params = "https://cell.example/twiml", {"CallSid": "CA1"}
    signed_by_ie1 = RequestValidator("t-ie1").compute_signature(url, params)
    signed_by_flat = RequestValidator("t-us1").compute_signature(url, params)
    assert await provider.verify_webhook_signature(url, params, signed_by_ie1)
    assert not await provider.verify_webhook_signature(url, params, signed_by_flat)


class _Response:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return ""


class _Session:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs.get("params")))
        return _Response(
            {"incoming_phone_numbers": [{"phone_number": "+34911222333", "sid": "PN1"}]}
        )

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs.get("data")))
        return _Response({})


async def _run_configure_inbound(monkeypatch, config, webhook_url):
    calls = []
    monkeypatch.setattr(
        provider_module.aiohttp, "ClientSession", lambda **kw: _Session(calls)
    )
    provider = TwilioProvider(config)
    result = await provider.configure_inbound("+34911222333", webhook_url)
    assert result.ok
    return calls


@pytest.mark.asyncio
async def test_configure_inbound_never_touches_the_us_host_in_an_eu_cell(monkeypatch):
    monkeypatch.setenv("CARRIER_REGION_POLICY", "eu")
    config = {
        "credentials": {"ie1": IE1},
        "from_numbers": ["+34911222333"],
        "fallback_url": "https://fallback.example/twiml",
    }
    calls = await _run_configure_inbound(monkeypatch, config, "https://cell/twiml")
    assert [c[0] for c in calls] == ["GET", "POST"]
    assert all(c[1].startswith("https://api.dublin.ie1.twilio.com/") for c in calls)
    post_data = calls[1][2]
    assert post_data["VoiceUrl"] == "https://cell/twiml"
    assert post_data["VoiceFallbackUrl"] == "https://fallback.example/twiml"


@pytest.mark.asyncio
async def test_configure_inbound_without_fallback_url_sends_no_fallback(monkeypatch):
    monkeypatch.delenv("CARRIER_REGION_POLICY", raising=False)
    config = {**US1, "from_numbers": ["+34911222333"]}
    calls = await _run_configure_inbound(monkeypatch, config, "https://cell/twiml")
    assert "VoiceFallbackUrl" not in calls[1][2]
    assert calls[1][1].startswith("https://api.twilio.com/")


@pytest.mark.asyncio
async def test_transport_passes_region_and_edge_to_the_serializer(monkeypatch):
    monkeypatch.setenv("CARRIER_REGION_POLICY", "eu")
    monkeypatch.setattr(
        transport,
        "load_credentials_for_transport",
        AsyncMock(return_value={"credentials": {"ie1": IE1}, "provider": "twilio"}),
    )
    serializer = MagicMock()
    monkeypatch.setattr(transport, "TwilioFrameSerializer", serializer)
    monkeypatch.setattr(transport, "build_audio_out_mixer", AsyncMock())
    monkeypatch.setattr(transport, "FastAPIWebsocketTransport", MagicMock())
    monkeypatch.setattr(transport, "FastAPIWebsocketParams", MagicMock())
    audio_config = MagicMock()

    await transport.create_transport(
        MagicMock(),
        1,
        audio_config,
        7,
        stream_sid="MZ1",
        call_sid="CA1",
    )

    kwargs = serializer.call_args.kwargs
    assert (kwargs["region"], kwargs["edge"]) == ("ie1", "dublin")
    assert (kwargs["account_sid"], kwargs["auth_token"]) == ("AC1", "t-ie1")
