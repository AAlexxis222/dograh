import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_stt_service
from api.tests.service_tuning._transport import (
    audio_config,
    capture_ws_connect,
    query_params,
    user_config_stt,
)


@pytest.mark.asyncio
async def test_elevenlabs_stt_keyterms_vad_and_filter_reach_the_query(monkeypatch):
    tuning = {
        "stt": {
            "elevenlabs": {
                "settings": {
                    "keyterms": ["Marbella", "parasailing"],
                    "vad_threshold": 0.6,
                    "min_silence_duration_ms": 400,
                    "filter_background_audio": True,
                },
                "ctor": {"include_timestamps": True},
            }
        }
    }
    service = create_stt_service(
        user_config_stt(
            ServiceProviders.ELEVENLABS.value,
            model="scribe_v2_realtime",
            language="es",
            base_url="https://api.elevenlabs.io",
        ),
        audio_config(),
        tuning=tuning,
    )
    q = query_params((await capture_ws_connect(monkeypatch, service))["url"])
    assert q["keyterms"] == ["Marbella", "parasailing"] and q["vad_threshold"] == [
        "0.6"
    ]
    assert q["min_silence_duration_ms"] == ["400"]
    assert q["filter_background_audio"] == ["true"] and q["commit_strategy"] == ["vad"]


@pytest.mark.asyncio
async def test_elevenlabs_stt_manual_commit_drops_vad_params(monkeypatch):
    tuning = {
        "stt": {
            "elevenlabs": {
                "settings": {"vad_threshold": 0.6},
                "ctor": {"commit_strategy": "manual"},
            }
        }
    }
    service = create_stt_service(
        user_config_stt(
            ServiceProviders.ELEVENLABS.value,
            model="scribe_v2_realtime",
            language="es",
            base_url="https://api.elevenlabs.io",
        ),
        audio_config(),
        tuning=tuning,
    )
    q = query_params((await capture_ws_connect(monkeypatch, service))["url"])
    # layer-3 loss, documented (elevenlabs/stt.py:793-805): the VAD params are
    # only appended to the query when commit_strategy is VAD.
    assert q["commit_strategy"] == ["manual"] and "vad_threshold" not in q
