"""``service_tuning`` on the realtime (speech-to-speech) branches.

Realtime is the one part of the plan without a pipecat ``Settings`` mapping
for the session: each branch builds provider pydantic objects
(``SessionProperties``, ``AudioInput``, ``Reasoning``, Ultravox
``OneShotInputParams``), so every knob is mapped onto an explicit destination
declared in ``service_factory.REALTIME_FIELDS``. Turn-detection knobs are
absent on purpose — they belong to the turn PR.

The service classes are patched (not the Settings classes), so what these
assert on is the object the factory actually hands the service.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pipecat.services.settings import NOT_GIVEN

from api.services.pipecat.service_factory import (
    REALTIME_FIELDS,
    create_realtime_llm_service,
    create_stt_service,
)
from api.services.pipecat.service_tuning_specs import SPECS
from api.tests.service_tuning._transport import audio_config, user_config_stt


def _rt(provider, model="gpt-realtime", **fields):
    return SimpleNamespace(
        realtime=SimpleNamespace(
            provider=provider,
            model=model,
            api_key="k",
            voice="alloy",
            language="es",
            **fields,
        ),
        is_realtime=True,
        llm=None,
    )


def test_openai_realtime_noise_reduction_speed_reasoning_and_max_output_tokens():
    tuning = {
        "realtime": {
            "openai_realtime": {
                "settings": {
                    "noise_reduction": "near_field",
                    "speed": 1.1,
                    "reasoning_effort": "low",
                    "max_output_tokens": 400,
                }
            }
        }
    }
    with patch(
        "api.services.pipecat.service_factory.DograhOpenAIRealtimeLLMService"
    ) as cls:
        create_realtime_llm_service(
            _rt("openai_realtime"), audio_config(), tuning=tuning
        )
    sp = cls.call_args.kwargs["settings"].session_properties
    assert (
        sp.audio.input.noise_reduction.type == "near_field"
        and sp.audio.output.speed == 1.1
    )
    assert sp.reasoning.effort == "low" and sp.max_output_tokens == 400
    # turn knobs untouched (PR turnos)
    assert sp.audio.input.turn_detection is None


def test_grok_realtime_language_hint():
    with patch(
        "api.services.pipecat.service_factory.DograhGrokRealtimeLLMService"
    ) as cls:
        create_realtime_llm_service(
            _rt("grok_realtime", model="grok-realtime"),
            audio_config(),
            tuning={
                "realtime": {"grok_realtime": {"settings": {"language_hint": "es"}}}
            },
        )
    sp = cls.call_args.kwargs["settings"].session_properties
    assert sp.audio.input.transcription.language_hint == "es"


def test_gemini_live_thinking_and_affective_dialog():
    tuning = {
        "realtime": {
            "google_realtime": {
                "settings": {
                    "thinking": {"thinking_budget": 256},
                    "enable_affective_dialog": True,
                }
            }
        }
    }
    with patch(
        "api.services.pipecat.service_factory.DograhGeminiLiveLLMService"
    ) as cls:
        create_realtime_llm_service(
            _rt("google_realtime", model="gemini-live"), audio_config(), tuning=tuning
        )
    s = cls.call_args.kwargs["settings"]
    assert s.enable_affective_dialog is True and s.thinking.thinking_budget == 256


def test_ultravox_extra_reaches_call_creation_params():
    # ``recordingEnabled`` is not one of the keys the call-creation body sets
    # (ultravox/llm.py:342-358) and the wrapper only rewrites
    # ``firstSpeakerSettings`` (realtime/ultravox_realtime.py:422-434), so it
    # survives the merge at ultravox/llm.py:361 into the request.
    with (
        patch(
            "api.services.pipecat.service_factory.DograhUltravoxOneShotInputParams"
        ) as params_cls,
        patch("api.services.pipecat.service_factory.DograhUltravoxRealtimeLLMService"),
    ):
        create_realtime_llm_service(
            _rt("ultravox_realtime", model="fixie-ai/ultravox"),
            audio_config(),
            tuning={
                "realtime": {
                    "ultravox_realtime": {
                        "settings": {"extra": {"recordingEnabled": True}}
                    }
                }
            },
        )
    assert params_cls.call_args.kwargs["extra"] == {"recordingEnabled": True}


def test_openai_stt_realtime_api_option_switches_service():
    with patch("api.services.pipecat.service_factory.OpenAIRealtimeSTTService") as cls:
        create_stt_service(
            user_config_stt("openai", model="gpt-realtime-whisper", language="es"),
            audio_config(),
            tuning={
                "stt": {
                    "openai": {
                        "options": {"api": "realtime"},
                        "settings": {
                            "noise_reduction": "far_field",
                            "prompt": "Marbella",
                        },
                    }
                }
            },
        )
    s = cls.call_args.kwargs["settings"]
    assert s.noise_reduction == "far_field" and s.prompt == "Marbella"
    # Both OpenAI STT services seed English, so the configured language has to
    # travel with the delta or the session transcribes Spanish as English.
    assert s.language == "es"
    # turn knobs belong to the turn PR
    assert cls.call_args.kwargs["turn_detection"] is False


def test_azure_realtime_shares_the_openai_session_knobs():
    tuning = {
        "realtime": {
            "azure_realtime": {
                "settings": {"noise_reduction": "far_field", "tool_choice": "required"}
            }
        }
    }
    with patch(
        "api.services.pipecat.service_factory.DograhAzureRealtimeLLMService"
    ) as cls:
        create_realtime_llm_service(
            _rt("azure_realtime", endpoint="https://example.openai.azure.com"),
            audio_config(),
            tuning=tuning,
        )
    sp = cls.call_args.kwargs["settings"].session_properties
    assert sp.audio.input.noise_reduction.type == "far_field"
    assert sp.tool_choice == "required"


def test_ultravox_temperature_and_max_duration_reach_the_params():
    with (
        patch(
            "api.services.pipecat.service_factory.DograhUltravoxOneShotInputParams"
        ) as params_cls,
        patch("api.services.pipecat.service_factory.DograhUltravoxRealtimeLLMService"),
    ):
        create_realtime_llm_service(
            _rt("ultravox_realtime", model="fixie-ai/ultravox"),
            audio_config(),
            tuning={
                "realtime": {
                    "ultravox_realtime": {
                        "settings": {"temperature": 0.4, "max_duration": 900}
                    }
                }
            },
        )
    kwargs = params_cls.call_args.kwargs
    assert kwargs["temperature"] == 0.4 and kwargs["max_duration"] == 900


def test_gemini_vertex_realtime_takes_the_same_knobs():
    with patch(
        "api.services.pipecat.service_factory.DograhGeminiLiveVertexLLMService"
    ) as cls:
        create_realtime_llm_service(
            _rt("google_vertex_realtime", model="gemini-live", project_id="p"),
            audio_config(),
            tuning={
                "realtime": {
                    "google_vertex_realtime": {
                        "settings": {
                            "temperature": 0.9,
                            "context_window_compression": {"trigger_tokens": 16000},
                        }
                    }
                }
            },
        )
    s = cls.call_args.kwargs["settings"]
    assert s.temperature == 0.9 and s.context_window_compression.trigger_tokens == 16000


def _leaf(root, path):
    for step in path:
        root = getattr(root, step)
    return root


@pytest.mark.parametrize(
    "provider,service",
    [
        ("openai_realtime", "DograhOpenAIRealtimeLLMService"),
        ("azure_realtime", "DograhAzureRealtimeLLMService"),
        ("grok_realtime", "DograhGrokRealtimeLLMService"),
        ("google_realtime", "DograhGeminiLiveLLMService"),
    ],
)
def test_untuned_realtime_builds_todays_session(provider, service):
    """Exposing a knob must not change what an untuned run gets: every
    destination is still the value the branch itself built."""
    fields = {"endpoint": "https://example.openai.azure.com"}
    with patch(f"api.services.pipecat.service_factory.{service}") as cls:
        create_realtime_llm_service(_rt(provider, **fields), audio_config())
    settings = cls.call_args.kwargs["settings"]
    root = getattr(settings, "session_properties", settings)
    # A session leaves its knobs at None; a Settings dataclass at NOT_GIVEN.
    untouched = NOT_GIVEN if root is settings else None
    for name, path in REALTIME_FIELDS[provider].items():
        assert _leaf(root, path) is untouched, name


def test_every_realtime_knob_has_a_destination():
    """No orphan knob: the coherence test in test_service_tuning_schema.py
    skips ``settings_cls=None`` rows, so the realtime allow-lists are checked
    against the field map that actually delivers them."""
    rows = {
        provider: spec for (kind, provider), spec in SPECS.items() if kind == "realtime"
    }
    assert set(rows) == set(REALTIME_FIELDS)
    for provider, spec in rows.items():
        assert set(spec.settings_allowed) == set(REALTIME_FIELDS[provider]), provider
        assert set(spec.settings_types) == set(spec.settings_allowed), provider


# --- OpenAI STT, options.api=realtime with a BYOK base_url (#7) -------------
#
# The segmented service takes an http(s) base and the realtime one a wss URL
# (openai/stt.py:242); reusing the configured http(s) base_url for the
# websocket would fail the connect, so the factory derives the websocket URL
# from an OpenAI-shaped base or fails the run start by name.


def _openai_realtime_stt(base_url):
    with patch("api.services.pipecat.service_factory.OpenAIRealtimeSTTService") as cls:
        create_stt_service(
            user_config_stt(
                "openai", model="gpt-realtime-whisper", language="es", base_url=base_url
            ),
            audio_config(),
            tuning={"stt": {"openai": {"options": {"api": "realtime"}}}},
        )
    return cls.call_args.kwargs


def test_openai_realtime_stt_derives_wss_from_an_openai_shaped_base_url():
    assert (
        _openai_realtime_stt("https://proxy.example/v1")["base_url"]
        == "wss://proxy.example/v1/realtime"
    )


def test_openai_realtime_stt_rejects_a_base_url_that_is_not_openai_shaped():
    with pytest.raises(HTTPException) as info:
        _openai_realtime_stt("https://proxy.example/custom")
    assert info.value.status_code == 400
    assert (
        info.value.detail
        == "stt.openai base_url https://proxy.example/custom is not usable with "
        "options.api=realtime; expected an https://…/v1 OpenAI-compatible base or unset"
    )


def test_openai_realtime_stt_derived_url_goes_through_the_egress_guard(monkeypatch):
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "saas")
    seen = []
    monkeypatch.setattr(
        "api.services.pipecat.service_factory.validate_user_configured_service_url",
        lambda url, *, field_name: seen.append(url),
    )
    _openai_realtime_stt("https://proxy.example/v1")
    assert "wss://proxy.example/v1/realtime" in seen


def test_control_openai_realtime_stt_without_base_url_keeps_pipecats_default():
    assert "base_url" not in _openai_realtime_stt(None)


# --- OpenAI segmented STT and the configured language (§B) -----------------


def _openai_segmented_stt(language):
    with patch("api.services.pipecat.service_factory.OpenAISTTService") as cls:
        create_stt_service(
            user_config_stt("openai", model="gpt-4o-transcribe", language=language),
            audio_config(),
        )
    return cls.call_args.kwargs["settings"]


def test_openai_segmented_stt_sends_the_configured_language():
    # The service seeds ``Language.EN`` (openai/stt.py:120), so without this a
    # Spanish call was transcribed as English.
    assert _openai_segmented_stt("es").language == "es"


def test_control_openai_segmented_stt_without_language_is_unchanged():
    from pipecat.services.openai.stt import OpenAISTTSettings

    # No language configured: the settings object carries the dataclass
    # default exactly as before, and the service seeds English from it.
    assert _openai_segmented_stt(None).language == OpenAISTTSettings().language
