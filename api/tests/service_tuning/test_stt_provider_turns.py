"""Provider-side turn knobs on the STT services that detect turns themselves.

Two halves. The knobs that only tune the provider's own segmentation reach the
Settings object (layer 2 again — no isolated query builder; the ctor half checks
the kwargs the factory forwards from ``plan.ctor``). The values that hand turn
detection to the provider are refused at the PUT while this build still runs its
own VAD and passes its own ``user_turn_strategies`` (run_pipeline.py:968-979,
which makes the service's recommendation a no-op in
llm_response_universal.py:966-974).
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.pipecat import service_tuning_specs as specs
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
        # True is the value that keeps this build's own turn detection; it is
        # the row's only ctor kwarg, so this is the ctor half of the test.
        {"vad_force_turn_endpoint": True},
        "AssemblyAISTTService",
    ),
    (
        "speechmatics",
        "enhanced",
        {
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
        {"endpointing": 0.4, "maximum_duration_without_endpointing": 8},
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
        {"high_vad_sensitivity": True, "min_speech_frames": 3},
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


# provider, tuning section, field, value that hands turns over, value that does not
HANDOVER_CASES = [
    # speechmatics/stt.py:550-554 (any mode but external) -> :915/:936 broadcast
    # UserStarted/StoppedSpeakingFrame.
    ("speechmatics", "settings", "turn_detection_mode", "adaptive", "external"),
    # gladia/stt.py:365-367; the broadcasts at :620-640 are guarded by enable_vad.
    ("gladia", "settings", "enable_vad", True, False),
    # sarvam/stt.py:815-826 broadcasts on the server's VAD events, :450 stops
    # honouring pipecat's VAD frames and :641 drops the flush signal.
    ("sarvam", "settings", "vad_signals", True, False),
    # assemblyai/stt.py:664-666; :1128-1133 and :1194-1196 emit turn frames only
    # in AssemblyAI's own turn-detection mode.
    ("assemblyai", "ctor", "vad_force_turn_endpoint", False, True),
]


@pytest.mark.parametrize("provider,section,field,handover,keeps", HANDOVER_CASES)
def test_provider_turn_handover_is_a_named_422(
    provider, section, field, handover, keeps
):
    def validate(value):
        WorkflowConfigurationDefaults.model_validate(
            {"service_tuning": {"stt": {provider: {section: {field: value}}}}}
        )

    validate(keeps)
    with pytest.raises(
        ValidationError,
        match=(
            f"stt.{provider}.{section}.{field}: {handover} hands turn detection "
            "to the provider"
        ),
    ):
        validate(handover)


# provider, field, a well-typed value. Whole fields, not values: these tune the
# provider's own turn detection, which this build's forced mode overwrites or
# ignores; the PUT is model-blind, so the set is too.
FORCED_MODE_FIELDS = [
    # assemblyai/stt.py:601-645 (_configure_pipecat_turn_mode) under
    # vad_force_turn_endpoint=True, the only value the gate lets in: u3-rt-pro
    # keeps min_turn_silence and overwrites max_turn_silence (:628-641),
    # universal-streaming keeps max_turn_silence and overwrites the other two
    # (:643-645). No single model honours all three.
    ("assemblyai", "end_of_turn_confidence_threshold", 0.6),
    ("assemblyai", "min_turn_silence", 200),
    ("assemblyai", "max_turn_silence", 1500),
    # speechmatics: turn_detection_mode=external is the only mode the gate lets
    # in, and its preset sets end_of_utterance_mode=EXTERNAL (speechmatics
    # voice/_presets.py:165-176), under which the SDK's end-of-utterance timers
    # (voice/_client.py:1463,1553-1560) never run.
    ("speechmatics", "end_of_utterance_silence_trigger", 0.5),
    ("speechmatics", "end_of_utterance_max_delay", 2.0),
]


@pytest.mark.parametrize("provider,field,value", FORCED_MODE_FIELDS)
def test_forced_mode_turn_fields_are_a_named_422_until_the_turn_pr(
    provider, field, value, monkeypatch
):
    doc = {"service_tuning": {"stt": {provider: {"settings": {field: value}}}}}
    with pytest.raises(
        ValidationError,
        match=(
            f"stt.{provider}.settings.{field}: .*PROVIDER_TURN_DETECTION_AVAILABLE"
            ".*turn PR"
        ),
    ):
        WorkflowConfigurationDefaults.model_validate(doc)
    monkeypatch.setattr(specs, "PROVIDER_TURN_DETECTION_AVAILABLE", True)
    WorkflowConfigurationDefaults.model_validate(doc)


@pytest.mark.parametrize("value", [0, None, "", []])
def test_falsy_ctor_values_cannot_slip_past_the_turn_gate(value):
    # The service tests this kwarg by truthiness (assemblyai/stt.py:665,
    # :1129, :1194), so anything falsy hands turns over exactly as False does.
    # An unhashable value must land as a 422, not a TypeError.
    with pytest.raises(
        ValidationError,
        match="stt.assemblyai.ctor.vad_force_turn_endpoint: wrong type",
    ):
        WorkflowConfigurationDefaults.model_validate(
            {
                "service_tuning": {
                    "stt": {"assemblyai": {"ctor": {"vad_force_turn_endpoint": value}}}
                }
            }
        )


def test_speechmatics_turn_detection_mode_is_the_service_enum():
    tuning = {
        "stt": {"speechmatics": {"settings": {"turn_detection_mode": "external"}}}
    }
    with patch("api.services.pipecat.service_factory.SpeechmaticsSTTService") as mock:
        create_stt_service(
            user_config_stt("speechmatics", model="enhanced", language="en"),
            audio_config(),
            tuning=tuning,
        )
    # _build_config reads ``turn_detection_mode.value`` (speechmatics/stt.py:752),
    # which a plain string does not have.
    assert mock.call_args.kwargs["settings"].turn_detection_mode.value == "external"
