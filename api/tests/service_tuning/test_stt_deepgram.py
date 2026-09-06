import pytest
from fastapi import HTTPException

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_stt_service
from api.tests.service_tuning._transport import (
    audio_config,
    capture_nova_connect,
    capture_ws_connect,
    query_params,
    user_config_stt,
)

FLUX = dict(
    provider=ServiceProviders.DEEPGRAM.value, model="flux-general-multi", language="es"
)


@pytest.mark.asyncio
async def test_flux_thresholds_numerals_and_hints_reach_the_query(monkeypatch):
    tuning = {
        "stt": {
            "deepgram": {
                "settings": {
                    "eot_timeout_ms": 5000,
                    "eot_threshold": 0.8,
                    "eager_eot_threshold": 0.6,
                    "numerals": True,
                    "min_confidence": 0.4,
                    "language_hints": ["en", "de"],
                },
                "ctor": {"mip_opt_out": True, "tag": ["xpand"]},
            }
        }
    }
    service = create_stt_service(
        user_config_stt(**FLUX), audio_config(), keyterms=["Marbella"], tuning=tuning
    )
    q = query_params((await capture_ws_connect(monkeypatch, service))["url"])
    assert (
        q["eot_timeout_ms"] == ["5000"]
        and q["eot_threshold"] == ["0.8"]
        and q["eager_eot_threshold"] == ["0.6"]
    )
    assert (
        q["numerals"] == ["true"]
        and q["mip_opt_out"] == ["true"]
        and q["tag"] == ["xpand"]
    )
    assert q["language_hint"] == [
        "en",
        "de",
    ]  # explicit hints replace the registry-derived one
    assert q["keyterm"] == ["Marbella"]  # dictionary keyterms untouched
    assert (
        service._settings.min_confidence == 0.4
    )  # local filter, not on the wire (flux base.py:766-771)


@pytest.mark.asyncio
async def test_flux_eager_null_turns_eager_off(monkeypatch):
    service = create_stt_service(
        user_config_stt(**FLUX),
        audio_config(),
        tuning={"stt": {"deepgram": {"settings": {"eager_eot_threshold": None}}}},
    )
    q = query_params((await capture_ws_connect(monkeypatch, service))["url"])
    assert "eager_eot_threshold" not in q and q["eot_threshold"] == ["0.7"]


@pytest.mark.asyncio
async def test_flux_language_hints_empty_list_means_autodetect(monkeypatch):
    service = create_stt_service(
        user_config_stt(**FLUX),
        audio_config(),
        tuning={"stt": {"deepgram": {"settings": {"language_hints": []}}}},
    )
    q = query_params((await capture_ws_connect(monkeypatch, service))["url"])
    assert "language_hint" not in q


@pytest.mark.asyncio
async def test_flux_scalar_keyterm_is_wrapped_not_spelled_out(monkeypatch):
    # The merged Deepgram row lets a scalar keyterm through because Nova
    # declares the field as ``Any`` ("str or list of str", deepgram/stt.py:215).
    # Flux appends one query parameter per element (flux/base.py:306), so the
    # string would be sent one character at a time.
    service = create_stt_service(
        user_config_stt(**FLUX),
        audio_config(),
        tuning={"stt": {"deepgram": {"settings": {"keyterm": "Marbella"}}}},
    )
    q = query_params((await capture_ws_connect(monkeypatch, service))["url"])
    assert q["keyterm"] == ["Marbella"]


@pytest.mark.asyncio
async def test_nova_keeps_a_scalar_keyterm_as_the_provider_declares_it():
    service = create_stt_service(
        user_config_stt(
            provider=ServiceProviders.DEEPGRAM.value, model="nova-3", language=None
        ),
        audio_config(),
        tuning={"stt": {"deepgram": {"settings": {"keyterm": "Marbella"}}}},
    )
    assert (await capture_nova_connect(service))["keyterm"] == "Marbella"


@pytest.mark.asyncio
async def test_flux_byok_url_ctor_changes_endpoint(monkeypatch):
    service = create_stt_service(
        user_config_stt(**FLUX),
        audio_config(),
        tuning={
            "stt": {"deepgram": {"ctor": {"url": "wss://proxy.example/v2/listen"}}}
        },
    )
    url = (await capture_ws_connect(monkeypatch, service))["url"]
    assert url.startswith("wss://proxy.example/v2/listen?")


def test_flux_byok_url_ctor_blocks_private_endpoint_in_saas(monkeypatch):
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "saas")
    with pytest.raises(HTTPException) as exc_info:
        create_stt_service(
            user_config_stt(**FLUX),
            audio_config(),
            tuning={"stt": {"deepgram": {"ctor": {"url": "wss://127.0.0.1/listen"}}}},
        )
    assert exc_info.value.status_code == 400
    assert "public IP" in exc_info.value.detail


@pytest.mark.asyncio
async def test_dograh_flux_gets_flux_settings_but_never_url(monkeypatch):
    tuning = {
        "stt": {"dograh": {"settings": {"eot_threshold": 0.9}, "ctor": {"tag": ["t"]}}}
    }
    service = create_stt_service(
        user_config_stt(
            provider=ServiceProviders.DOGRAH.value, model="x", language="es"
        ),
        audio_config(),
        correlation_id="c",
        tuning=tuning,
    )
    q = query_params((await capture_ws_connect(monkeypatch, service))["url"])
    assert (
        q["eot_threshold"] == ["0.9"]
        and q["tag"] == ["t"]
        and q["model"] == ["flux-general-multi"]
    )


@pytest.mark.asyncio
async def test_nova_settings_reach_connect_kwargs():
    tuning = {
        "stt": {
            "deepgram": {
                "settings": {
                    "endpointing": 300,
                    "smart_format": True,
                    "utterance_end_ms": 1200,
                    "profanity_filter": True,
                    "keywords": ["Marbella:2"],
                }
            }
        }
    }
    service = create_stt_service(
        user_config_stt(
            provider=ServiceProviders.DEEPGRAM.value, model="nova-3", language=None
        ),
        audio_config(),
        tuning=tuning,
    )
    kw = await capture_nova_connect(service)
    assert (
        kw["endpointing"] == "300"
        and kw["smart_format"] == "true"
        and kw["utterance_end_ms"] == "1200"
    )
    assert kw["profanity_filter"] == "true" and kw["keywords"] == ["Marbella:2"]
    assert service._settings.extra == {}


@pytest.mark.asyncio
async def test_nova_endpointing_null_omits_it():
    service = create_stt_service(
        user_config_stt(
            provider=ServiceProviders.DEEPGRAM.value, model="nova-3", language=None
        ),
        audio_config(),
        tuning={"stt": {"deepgram": {"settings": {"endpointing": None}}}},
    )
    assert "endpointing" not in await capture_nova_connect(service)
