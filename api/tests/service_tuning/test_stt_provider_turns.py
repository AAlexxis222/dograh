"""Provider-side turn knobs on the STT services that detect turns themselves.

Layer 2 again (no isolated query builder); the ctor half checks the kwargs the
factory forwards from ``plan.ctor``.
"""

from unittest.mock import patch

import pytest

from api.services.pipecat.service_factory import create_stt_service
from api.tests.service_tuning._transport import audio_config, user_config_stt

TURN_CASES = [
    (
        "assemblyai",
        "universal-streaming",
        {
            "end_of_turn_confidence_threshold": 0.6,
            "min_turn_silence": 200,
            "max_turn_silence": 1500,
            "vad_threshold": 0.4,
            "interruption_delay": 300,
            "mode": "balanced",
        },
        {"vad_force_turn_endpoint": False},
        "AssemblyAISTTService",
    ),
    (
        "speechmatics",
        "enhanced",
        {
            "turn_detection_mode": "adaptive",
            "end_of_utterance_silence_trigger": 0.5,
            "end_of_utterance_max_delay": 2.0,
            "max_delay": 1.0,
        },
        {},
        "SpeechmaticsSTTService",
    ),
    (
        "gladia",
        "solaria-1",
        {
            "enable_vad": True,
            "endpointing": 0.4,
            "maximum_duration_without_endpointing": 8,
        },
        {},
        "GladiaSTTService",
    ),
    (
        "smallest",
        "lightning",
        {"endpointing": True, "numerals": "on"},
        {},
        "SmallestSTTService",
    ),
    (
        "sarvam",
        "saarika:v2",
        {"vad_signals": True, "high_vad_sensitivity": True, "min_speech_frames": 3},
        {},
        "SarvamSTTService",
    ),
]


@pytest.mark.parametrize("provider,model,settings,ctor,cls", TURN_CASES)
def test_turn_knobs_reach_settings_and_ctor(provider, model, settings, ctor, cls):
    with patch(f"api.services.pipecat.service_factory.{cls}") as mock:
        create_stt_service(
            user_config_stt(provider, model=model, language="en"),
            audio_config(),
            tuning={"stt": {provider: {"settings": settings, "ctor": ctor}}},
        )
    got = mock.call_args.kwargs["settings"]
    for k, v in settings.items():
        assert getattr(got, k) == v or getattr(getattr(got, k), "value", None) == v, k
    for k, v in ctor.items():
        assert mock.call_args.kwargs[k] == v


def test_speechmatics_turn_detection_mode_is_the_service_enum():
    tuning = {
        "stt": {"speechmatics": {"settings": {"turn_detection_mode": "adaptive"}}}
    }
    with patch("api.services.pipecat.service_factory.SpeechmaticsSTTService") as mock:
        create_stt_service(
            user_config_stt("speechmatics", model="enhanced", language="en"),
            audio_config(),
            tuning=tuning,
        )
    # _build_config reads ``turn_detection_mode.value`` (speechmatics/stt.py:752),
    # which a plain string does not have.
    assert mock.call_args.kwargs["settings"].turn_detection_mode.value == "adaptive"
