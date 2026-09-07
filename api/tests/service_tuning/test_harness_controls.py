import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import (
    create_llm_service_from_provider,
    create_stt_service,
    create_tts_service,
)
from api.tests.service_tuning._transport import (
    audio_config,
    capture_elevenlabs_context_init,
    capture_nova_connect,
    capture_ws_connect,
    chat_payload,
    query_params,
    user_config_stt,
    user_config_tts,
)


@pytest.mark.asyncio
async def test_control_flux_query_carries_todays_hardcoded_thresholds(monkeypatch):
    service = create_stt_service(
        user_config_stt(
            ServiceProviders.DEEPGRAM.value, model="flux-general-multi", language="es"
        ),
        audio_config(),
        keyterms=["Marbella"],
    )
    captured = await capture_ws_connect(monkeypatch, service)
    q = query_params(captured["url"])
    assert q["eot_timeout_ms"] == ["3000"]
    assert q["eot_threshold"] == ["0.7"]
    assert q["eager_eot_threshold"] == ["0.5"]
    assert q["keyterm"] == ["Marbella"]
    assert q["language_hint"] == ["es"]
    assert "numerals" not in q and "min_confidence" not in q
    assert captured["headers"]["Authorization"] == "Token test-key"


@pytest.mark.asyncio
async def test_control_dograh_flux_query_uses_bearer_and_multi_model(monkeypatch):
    service = create_stt_service(
        user_config_stt(ServiceProviders.DOGRAH.value, model="ignored", language="en"),
        audio_config(),
        correlation_id="corr-1",
    )
    captured = await capture_ws_connect(monkeypatch, service)
    q = query_params(captured["url"])
    assert q["model"] == ["flux-general-multi"] and q["correlation_id"] == ["corr-1"]
    assert captured["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_control_nova_connect_kwargs_today():
    service = create_stt_service(
        user_config_stt(ServiceProviders.DEEPGRAM.value, model="nova-3", language=None),
        audio_config(),
    )
    kw = await capture_nova_connect(service)
    assert kw["endpointing"] == "100"
    assert kw["profanity_filter"] == "false"
    assert kw["language"] == "multi"


@pytest.mark.asyncio
async def test_control_elevenlabs_tts_voice_settings_today():
    service = create_tts_service(
        user_config_tts(
            ServiceProviders.ELEVENLABS.value,
            model="eleven_flash_v2_5",
            voice="Elena - abc",
            speed=1.0,
            base_url="https://api.elevenlabs.io",
        ),
        audio_config(),
    )
    init = await capture_elevenlabs_context_init(service)
    assert init["voice_settings"] == {
        "stability": 0.8,
        "similarity_boost": 0.75,
        "speed": 1.0,
    }


def test_control_openai_chat_payload_gpt41_has_temperature_and_no_extras():
    llm = create_llm_service_from_provider(
        provider="openai", model="gpt-4.1", api_key="k"
    )
    p = chat_payload(llm)
    assert p["temperature"] == 0.1
    assert "reasoning_effort" not in p and "verbosity" not in p


def test_control_openai_chat_payload_gpt5_has_extras_and_no_temperature():
    from openai import NOT_GIVEN

    llm = create_llm_service_from_provider(
        provider="openai", model="gpt-5-mini", api_key="k"
    )
    p = chat_payload(llm)
    assert p["reasoning_effort"] == "minimal" and p["verbosity"] == "low"
    assert p["temperature"] is NOT_GIVEN


def test_control_untuned_tts_pushes_no_silence_and_keeps_todays_silence_time():
    from unittest.mock import patch

    with patch("api.services.pipecat.service_factory.CartesiaTTSService") as mock:
        create_tts_service(user_config_tts("cartesia", model="sonic-2"), audio_config())
    kwargs = mock.call_args.kwargs
    assert kwargs["push_silence_after_stop"] is False
    assert kwargs["silence_time_s"] == 1.0
