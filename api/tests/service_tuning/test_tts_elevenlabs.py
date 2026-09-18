from unittest.mock import patch

import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_tts_service
from api.tests.service_tuning._transport import (
    audio_config,
    capture_elevenlabs_context_init,
    capture_ws_connect,
    user_config_tts,
)


@pytest.mark.asyncio
async def test_elevenlabs_voice_settings_and_speed_reach_the_context_init():
    tuning = {
        "tts": {
            "elevenlabs": {
                "settings": {
                    "stability": 0.4,
                    "similarity_boost": 0.9,
                    "style": 0.2,
                    "use_speaker_boost": True,
                    "speed": 1.1,
                    "apply_text_normalization": "on",
                }
            }
        }
    }
    service = create_tts_service(
        user_config_tts(
            ServiceProviders.ELEVENLABS.value,
            model="eleven_flash_v2_5",
            voice="Elena - abc",
            speed=1.0,
            base_url="https://api.elevenlabs.io",
        ),
        audio_config(),
        tuning=tuning,
    )
    init = await capture_elevenlabs_context_init(service)
    assert init["voice_settings"] == {
        "stability": 0.4,
        "similarity_boost": 0.9,
        "style": 0.2,
        "use_speaker_boost": True,
        "speed": 1.1,
    }


@pytest.mark.asyncio
async def test_elevenlabs_text_normalization_reaches_the_url(monkeypatch):
    service = create_tts_service(
        user_config_tts(
            ServiceProviders.ELEVENLABS.value,
            model="eleven_flash_v2_5",
            voice="Elena - abc",
            base_url="https://api.elevenlabs.io",
        ),
        audio_config(),
        tuning={
            "tts": {"elevenlabs": {"settings": {"apply_text_normalization": "on"}}}
        },
    )
    url = (await capture_ws_connect(monkeypatch, service))["url"]
    assert "apply_text_normalization=on" in url


def test_tts_all_silence_time_applies_to_every_tts():
    with patch("api.services.pipecat.service_factory.ElevenLabsTTSService") as cls:
        create_tts_service(
            user_config_tts(
                ServiceProviders.ELEVENLABS.value,
                model="m",
                voice="Elena - abc",
                base_url="https://api.elevenlabs.io",
            ),
            audio_config(),
            tuning={"tts": {"_all": {"ctor": {"silence_time_s": 0.4}}}},
        )
    assert cls.call_args.kwargs["silence_time_s"] == 0.4
