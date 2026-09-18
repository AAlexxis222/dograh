import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_tts_service
from api.tests.service_tuning._transport import (
    audio_config,
    capture_openai_tts_request,
    user_config_tts,
)


@pytest.mark.asyncio
async def test_openai_tts_voice_instructions_and_speed_reach_the_request():
    tuning = {
        "tts": {
            "openai": {
                "settings": {
                    "voice": "nova",
                    "instructions": "Speak warmly",
                    "speed": 1.2,
                }
            }
        }
    }
    service = create_tts_service(
        user_config_tts(
            ServiceProviders.OPENAI.value, model="gpt-4o-mini-tts", voice="alloy"
        ),
        audio_config(),
        tuning=tuning,
    )
    req = await capture_openai_tts_request(service)
    assert req["voice"] == "nova" and req["instructions"] == "Speak warmly"
    assert req["speed"] == 1.2


@pytest.mark.asyncio
async def test_openai_tts_registry_voice_is_used_when_not_tuned():
    service = create_tts_service(
        user_config_tts(
            ServiceProviders.OPENAI.value, model="gpt-4o-mini-tts", voice="shimmer"
        ),
        audio_config(),
    )
    # B6: before this task the factory always hardcoded "alloy", ignoring the
    # registry voice.
    assert (await capture_openai_tts_request(service))["voice"] == "shimmer"
