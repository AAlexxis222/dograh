"""Every STT provider exposes its keyterm-style biasing field (D9/B7).

Layer-2 tests: these services have no isolated query builder to drive, so the
assertion is on the Settings object handed to the service class.
"""

from unittest.mock import patch

import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_stt_service
from api.tests.service_tuning._transport import audio_config, user_config_stt

CASES = [
    # provider, model, settings key, value, class name in service_factory
    (
        ServiceProviders.CARTESIA.value,
        "ink-whisper",
        "keyterm",
        ["Marbella"],
        "CartesiaSTTService",
    ),
    (
        ServiceProviders.CARTESIA.value,
        "ink-2",
        "keyterm",
        ["Marbella"],
        "CartesiaTurnsSTTService",
    ),
    (
        ServiceProviders.ELEVENLABS.value,
        "scribe_v2_realtime",
        "keyterms",
        ["Marbella"],
        "ElevenLabsRealtimeSTTService",
    ),
    (
        ServiceProviders.SMALLEST.value,
        "lightning",
        "keywords",
        "Marbella",
        "SmallestSTTService",
    ),
    (
        ServiceProviders.OPENAI.value,
        "whisper-1",
        "prompt",
        "Marbella, parasailing",
        "OpenAISTTService",
    ),
    (
        ServiceProviders.SPEACHES.value,
        "whisper",
        "prompt",
        "Marbella",
        "SpeachesSTTService",
    ),
    (
        ServiceProviders.SARVAM.value,
        "saarika:v2",
        "prompt",
        "Marbella",
        "SarvamSTTService",
    ),
]


@pytest.mark.parametrize("provider,model,key,value,cls", CASES)
def test_keyterm_field_reaches_settings(provider, model, key, value, cls):
    tuning = {"stt": {provider: {"settings": {key: value}}}}
    with patch(f"api.services.pipecat.service_factory.{cls}") as mock:
        create_stt_service(
            user_config_stt(
                provider,
                model=model,
                language="en",
                base_url="https://api.elevenlabs.io",
            ),
            audio_config(),
            tuning=tuning,
        )
    assert getattr(mock.call_args.kwargs["settings"], key) == value


def test_gladia_custom_vocabulary_reaches_realtime_processing():
    tuning = {
        "stt": {
            "gladia": {
                "settings": {
                    "realtime_processing": {
                        "custom_vocabulary": True,
                        "custom_vocabulary_config": {"vocabulary": ["Marbella"]},
                    }
                }
            }
        }
    }
    with patch("api.services.pipecat.service_factory.GladiaSTTService") as mock:
        create_stt_service(
            user_config_stt("gladia", model="solaria-1", language="en"),
            audio_config(),
            tuning=tuning,
        )
    rp = mock.call_args.kwargs["settings"].realtime_processing
    # RealtimeProcessingConfig (gladia/config.py:100); custom_vocabulary_config:
    # CustomVocabularyConfig | None (:117, class :56).
    assert rp.custom_vocabulary_config.vocabulary == ["Marbella"]


def test_speechmatics_additional_vocab_is_coerced_from_json():
    tuning = {
        "stt": {
            "speechmatics": {
                "settings": {"additional_vocab": [{"content": "Marbella"}]}
            }
        }
    }
    with patch("api.services.pipecat.service_factory.SpeechmaticsSTTService") as mock:
        create_stt_service(
            user_config_stt("speechmatics", model="enhanced", language="en"),
            audio_config(),
            tuning=tuning,
        )
    entry = mock.call_args.kwargs["settings"].additional_vocab[0]
    # VoiceAgentConfig does not validate on assignment, so a raw dict would
    # reach the Speechmatics SDK unconverted (speechmatics/stt.py:775).
    assert entry.content == "Marbella"


def test_cartesia_ink2_now_receives_settings_at_all():
    with patch("api.services.pipecat.service_factory.CartesiaTurnsSTTService") as mock:
        create_stt_service(
            user_config_stt("cartesia", model="ink-2", language="es"),
            audio_config(),
        )
    # Today sf:333-338 passes no settings at all (C1).
    assert mock.call_args.kwargs["settings"].language == "es"
