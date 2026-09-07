import json
from pathlib import Path

import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import (
    create_llm_service_from_provider,
    create_stt_service,
    create_tts_service,
)
from api.tests.service_tuning import _golden
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


# ---------------------------------------------------------------------------
# «expose ≠ change» for every construction site (#15)
#
# ``golden/untuned_construction.json`` is ``_golden.capture()`` run against
# the base commit ``ac8ed47d`` (a throwaway ``git worktree`` of it, the
# capture module loaded by path so both trees ran the same code; the worktree
# was removed afterwards). Every provider the factory builds, one case per
# model- or language-dependent branch, with ``tuning=None``. The kwargs each
# service class receives today must be the golden's, except for the changes
# this branch declares — listed below, and checked to be *exactly* the diff,
# so an undeclared change fails and a declared one that stopped differing
# fails too (a stale allow-list is not a rubber stamp).
# ---------------------------------------------------------------------------
_GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "untuned_construction.json").read_text()
)
# Declared behaviour changes (docs/voice-agent/provider-tuning.mdx,
# "Behaviour changes"), as the kwarg paths they touch.
DECLARED_DELTAS: dict[str, set[str]] = {
    # B6: OpenAI TTS now passes the registry voice.
    "tts.openai": {"kwargs.settings.voice"},
    # C1: Cartesia ink-2 now receives a settings object.
    "stt.cartesia/ink-2": {"kwargs.settings"},
    # B12: the configured xAI speed reaches the service (1.0 is still omitted).
    "tts.xai/speed": {"kwargs.settings.speed"},
    # The segmented OpenAI STT sends the configured language (none: unchanged).
    "stt.openai/segments-language": {"kwargs.settings.language"},
    # gpt-5-chat is not a reasoning model: temperature, no extras.
    "llm.openai/gpt-5-chat": {
        "kwargs.settings.temperature",
        "kwargs.settings.extra.reasoning_effort",
        "kwargs.settings.extra.verbosity",
    },
    # Untuned gpt-5 reasoning on openrouter/azure: no default temperature.
    "llm.openrouter/gpt-5": {"kwargs.settings.temperature"},
    "llm.azure/gpt-5": {"kwargs.settings.temperature"},
    # #10: Camb built through settings= instead of the deprecated kwargs; the
    # effective values are pinned by test_tts_long_tail.py's Camb control.
    "tts.camb": {"kwargs.voice_id", "kwargs.model", "kwargs.settings"},
}
# ``push_silence_after_stop`` is passed explicitly now (False untuned) where
# the base left it to the constructor default, which is False
# (tts_service.py:159): same construction, spelled out. Every TTS branch but
# Camb, which takes neither silence kwarg (C8).
_PUSH_SILENCE = "kwargs.push_silence_after_stop"


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(set(a) | set(b))
            for d in _diff(
                a.get(key, "<absent>"),
                b.get(key, "<absent>"),
                f"{path}.{key}" if path else key,
            )
        ]
    return [] if a == b else [path]


@pytest.fixture(scope="module")
def current_construction():
    return _golden.capture()


def test_golden_and_capture_cover_the_same_cases(current_construction):
    assert set(current_construction) == set(_GOLDEN)


@pytest.mark.parametrize("case", sorted(_GOLDEN))
def test_untuned_construction_matches_base(case, current_construction):
    today = current_construction[case]
    assert today["class"] == _GOLDEN[case]["class"]
    expected = set(DECLARED_DELTAS.get(case, set()))
    if case.startswith("tts.") and case != "tts.camb":
        expected.add(_PUSH_SILENCE)
        assert today["kwargs"]["push_silence_after_stop"] is False
    assert set(_diff(_GOLDEN[case], today)) == expected
