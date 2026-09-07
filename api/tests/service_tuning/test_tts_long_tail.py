from unittest.mock import patch

import pytest
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.pipecat.service_factory import create_tts_service
from api.tests.service_tuning._transport import audio_config, user_config_tts

TTS_CASES = [
    (
        "cartesia",
        "sonic-2",
        {
            "pronunciation_dict_id": "pd_1",
            "generation_config": {"speed": 1.2, "volume": 0.9},
        },
        {"max_buffer_delay_ms": 200},
        "CartesiaTTSService",
    ),
    ("inworld", "inworld-tts-1", {"temperature": 0.7}, {}, "InworldTTSService"),
    (
        "rime",
        "arcana",
        {
            "reduceLatency": True,
            "temperature": 0.5,
            "top_p": 0.9,
            "phonemizeBetweenBrackets": True,
        },
        {},
        "RimeTTSService",
    ),
    (
        "sarvam",
        "bulbul:v2",
        {
            "pitch": 0.2,
            "loudness": 1.1,
            "enable_preprocessing": True,
            "temperature": 0.6,
        },
        {},
        "SarvamTTSService",
    ),
    (
        "minimax",
        "speech-02-hd",
        {"volume": 1.2, "pitch": 1, "emotion": "happy", "language_boost": "Spanish"},
        {},
        "MiniMaxOwnedSessionTTSService",
    ),
    (
        "azure_speech",
        "neural",
        {
            # ``style_degree`` is declared ``str`` upstream (azure/tts.py:99),
            # so a float would be a 422 at the PUT: this layer-2 test has to
            # send what a stored document can actually hold.
            "style": "cheerful",
            "style_degree": "1.5",
            "pitch": "+5%",
            "volume": "+10%",
        },
        {},
        "AzureTTSService",
    ),
    (
        "xai",
        "grok-tts",
        {"speed": 1.3, "text_normalization": True},
        {},
        "XAITTSService",
    ),
    ("smallest", "lightning-v2", {"speed": 1.1}, {}, "SmallestTTSService"),
    ("google", "chirp_3_hd", {"speaking_rate": 1.4}, {}, "GoogleTTSService"),
]


@pytest.mark.parametrize("provider,model,settings,ctor,cls", TTS_CASES)
def test_tts_knobs_reach_settings(provider, model, settings, ctor, cls):
    with (
        # MiniMax opens its own session before constructing the service.
        patch("api.services.pipecat.service_factory.aiohttp.ClientSession"),
        patch(f"api.services.pipecat.service_factory.{cls}") as mock,
    ):
        create_tts_service(
            user_config_tts(
                provider,
                model=model,
                voice="Name - v1",
                base_url="https://api.example",
                group_id="grp-1",
            ),
            audio_config(),
            tuning={"tts": {provider: {"settings": settings, "ctor": ctor}}},
        )
    got = mock.call_args.kwargs["settings"]
    for k, v in settings.items():
        actual = getattr(got, k)
        assert (
            actual == v
            or getattr(actual, "value", None) == v
            or (
                hasattr(actual, "model_dump")
                and actual.model_dump(exclude_none=True) == v
            )
        ), k
    for k, v in ctor.items():
        assert mock.call_args.kwargs[k] == v


def test_dograh_tts_pitch_and_volume_are_dropped_by_the_service_and_documented():
    """Layer-3 loss (dograh/tts.py:143-150): the service only forwards speed.

    We keep the knob out of the allow-list so the user gets a 422 instead of a
    no-op.
    """
    with pytest.raises(
        ValidationError, match="tts.dograh.settings.pitch: unknown setting"
    ):
        WorkflowConfigurationDefaults.model_validate(
            {"service_tuning": {"tts": {"dograh": {"settings": {"pitch": 2}}}}}
        )


def test_xai_registry_speed_reaches_the_service():
    """B12: the xAI branch dropped ``user_config.tts.speed`` before this task."""
    with patch("api.services.pipecat.service_factory.XAITTSService") as mock:
        create_tts_service(
            user_config_tts("xai", model="grok-tts", voice="eve", speed=1.3),
            audio_config(),
        )
    assert mock.call_args.kwargs["settings"].speed == 1.3


def test_camb_timeout_ctor_reaches_the_service():
    # Camb is the one TTS branch that keeps building by direct kwargs (it has
    # no ``settings=`` today), so its row exposes the ctor timeout only.
    with patch("pipecat.services.camb.tts.CambTTSService") as mock:
        create_tts_service(
            user_config_tts("camb", model="mars-flash", voice="147320"),
            audio_config(),
            tuning={"tts": {"camb": {"ctor": {"timeout": 30.0}}}},
        )
    assert mock.call_args.kwargs["timeout"] == 30.0


def test_camb_timeout_accepts_a_json_int():
    # ``timeout`` is declared ``float`` on the constructor, but JSON has one
    # number type: 30 must not be a false 422.
    WorkflowConfigurationDefaults.model_validate(
        {"service_tuning": {"tts": {"camb": {"ctor": {"timeout": 30}}}}}
    )
    with pytest.raises(ValidationError, match="tts.camb.ctor.timeout: wrong type"):
        WorkflowConfigurationDefaults.model_validate(
            {"service_tuning": {"tts": {"camb": {"ctor": {"timeout": "30"}}}}}
        )


def test_all_section_silence_time_is_not_splatted_into_the_provider_ctor():
    # tuning_for merges tts._all into every provider plan, and each branch
    # already passes silence_time_s by hand: splatting the plan's ctor sent it
    # twice (TypeError on Cartesia) and would have handed it to Camb, the one
    # branch built without it (C8).
    tuning = {
        "tts": {
            "_all": {"ctor": {"silence_time_s": 0.4}},
            "cartesia": {"ctor": {"max_buffer_delay_ms": 200}},
        }
    }
    with patch("api.services.pipecat.service_factory.CartesiaTTSService") as mock:
        create_tts_service(
            user_config_tts("cartesia", model="sonic-2"), audio_config(), tuning=tuning
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["silence_time_s"] == 0.4 and kwargs["max_buffer_delay_ms"] == 200

    with patch("pipecat.services.camb.tts.CambTTSService") as mock:
        create_tts_service(
            user_config_tts("camb", model="mars-flash", voice="147320"),
            audio_config(),
            tuning=tuning,
        )
    assert "silence_time_s" not in mock.call_args.kwargs


def test_all_section_silence_time_enables_push_silence_after_stop():
    # ``silence_time_s`` is only read under ``push_silence_after_stop``
    # (tts_service.py:902-903), so setting the knob has to switch that on or
    # it sizes a silence nobody pushes.
    tuning = {"tts": {"_all": {"ctor": {"silence_time_s": 0.4}}}}
    with patch("api.services.pipecat.service_factory.CartesiaTTSService") as mock:
        create_tts_service(
            user_config_tts("cartesia", model="sonic-2"), audio_config(), tuning=tuning
        )
    kwargs = mock.call_args.kwargs
    assert kwargs["push_silence_after_stop"] is True and kwargs["silence_time_s"] == 0.4
    # Camb stays the one branch built without either (C8).
    with patch("pipecat.services.camb.tts.CambTTSService") as mock:
        create_tts_service(
            user_config_tts("camb", model="mars-flash", voice="147320"),
            audio_config(),
            tuning=tuning,
        )
    assert "push_silence_after_stop" not in mock.call_args.kwargs
