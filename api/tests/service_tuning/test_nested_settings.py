"""Settings whose value is a provider object (pydantic model or dataclass).

``from_mapping`` stores the JSON object unchanged and the branch converts it
(``model_validate`` / ``SpeakerIdentifier(**e)``) at run creation, so before
this a misspelt key inside the object was silently dropped (those models
ignore unknown keys) and a wrong scalar was a ValidationError when the run
was created. Both are a named 422 at the PUT now (#11, #12).
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.pipecat import service_tuning_specs as specs
from api.services.pipecat.service_factory import create_tts_service
from api.tests.service_tuning._transport import audio_config, user_config_tts

_SAFETY = {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
_SPEAKER = {"label": "agent", "speaker_identifiers": ["S1"]}
# (kind, provider, name, valid, typo, wrong scalar). A wrong scalar has to be
# one pydantic's lax mode does not coerce: 7 is not a bool, "lots" is not an
# int, an int is not a str.
NESTED = [
    ("tts", "cartesia", "generation_config", {"speed": 1.2}, {"sped": 1.2}, {"speed": "zoom"}),
    ("stt", "gladia", "pre_processing", {"audio_enhancer": True}, {"enhancer": True}, {"speech_threshold": "loud"}),
    ("stt", "gladia", "realtime_processing", {"custom_vocabulary": True}, {"custom_vocab": True}, {"translation": 7}),
    ("stt", "gladia", "messages_config", {"receive_errors": True}, {"receive_error": True}, {"receive_errors": 7}),
    ("stt", "speechmatics", "additional_vocab", [{"content": "Marbella"}], [{"contnet": "Marbella"}], [{"content": 7}]),
    ("stt", "speechmatics", "known_speakers", [_SPEAKER], [{"label": "agent", "speakers": ["S1"]}], [{"label": "agent", "speaker_identifiers": "S1"}]),
    ("llm", "google", "thinking", {"thinking_budget": 1024}, {"budget": 1024}, {"thinking_budget": "lots"}),
    # genai enums are case-insensitive and admit an unknown *string* member
    # (a forward-compatibility choice the run creation shares), so the wrong
    # scalar here is a number.
    ("llm", "google", "safety_settings", [_SAFETY], [{**_SAFETY, "categry": "x"}], [{"category": 7}]),
    ("llm", "google_vertex", "thinking", {"include_thoughts": True}, {"thoughts": True}, {"include_thoughts": 7}),
    ("llm", "google_vertex", "safety_settings", [_SAFETY], [{"catgory": "x"}], [{"threshold": 7}]),
    ("realtime", "google_realtime", "thinking", {"include_thoughts": True}, {"thoughts": True}, {"thinking_budget": "lots"}),
    ("realtime", "google_realtime", "proactivity", {"proactive_audio": True}, {"proactive": True}, {"proactive_audio": 7}),
    ("realtime", "google_realtime", "context_window_compression", {"enabled": True, "trigger_tokens": 1000}, {"enable": True}, {"enabled": 7}),
    ("realtime", "google_vertex_realtime", "thinking", {"thinking_budget": 512}, {"budget": 512}, {"include_thoughts": 7}),
    ("realtime", "google_vertex_realtime", "proactivity", {"proactive_audio": False}, {"audio": False}, {"proactive_audio": "loud"}),
    ("realtime", "google_vertex_realtime", "context_window_compression", {"enabled": False}, {"on": False}, {"trigger_tokens": "many"}),
]  # fmt: skip


def _doc(kind, provider, name, value):
    return {"service_tuning": {kind: {provider: {"settings": {name: value}}}}}


@pytest.mark.parametrize("kind,provider,name,valid,typo,bad", NESTED)
def test_nested_object_keys_and_scalars_are_checked_at_the_put(
    kind, provider, name, valid, typo, bad
):
    WorkflowConfigurationDefaults.model_validate(_doc(kind, provider, name, valid))
    with pytest.raises(
        ValidationError, match=rf"{kind}\.{provider}\.settings\.{name}.*: unknown key"
    ):
        WorkflowConfigurationDefaults.model_validate(_doc(kind, provider, name, typo))
    with pytest.raises(ValidationError, match=rf"{kind}\.{provider}\.settings\.{name}"):
        WorkflowConfigurationDefaults.model_validate(_doc(kind, provider, name, bad))


def test_unknown_key_inside_a_nested_model_is_named():
    with pytest.raises(
        ValidationError,
        match=r"stt\.gladia\.settings\.realtime_processing\.custom_vocabulary_config\.vocab: unknown key",
    ):
        WorkflowConfigurationDefaults.model_validate(
            _doc(
                "stt",
                "gladia",
                "realtime_processing",
                {"custom_vocabulary_config": {"vocab": ["Marbella"]}},
            )
        )


def test_boolean_never_rides_a_numeric_field_inside_a_nested_object():
    # pydantic's lax mode coerces ``true`` to ``1.0`` on a float field; the
    # same rule as top-level settings applies inside the object.
    with pytest.raises(
        ValidationError,
        match=r"tts\.cartesia\.settings\.generation_config\.speed: wrong type",
    ):
        WorkflowConfigurationDefaults.model_validate(
            _doc("tts", "cartesia", "generation_config", {"speed": True})
        )


def test_genai_alias_spelling_is_accepted_like_the_model_accepts_it():
    # The realtime ``thinking`` is google.genai's model, which takes camelCase
    # aliases; the chat one is pipecat's own and does not.
    WorkflowConfigurationDefaults.model_validate(
        _doc("realtime", "google_realtime", "thinking", {"thinkingBudget": 256})
    )
    with pytest.raises(ValidationError, match="thinking.thinkingBudget: unknown key"):
        WorkflowConfigurationDefaults.model_validate(
            _doc("llm", "google", "thinking", {"thinkingBudget": 256})
        )


def test_every_nested_model_setting_is_declared_and_allow_listed():
    # Every field whose type is a pydantic model or dataclass has a row in
    # ``settings_models`` (the scan is the same one that found NESTED), and no
    # ``settings_models`` name is outside the allow-list.
    declared = {
        (k, p, n) for (k, p), s in specs.SPECS.items() for n in s.settings_models
    }
    assert {(k, p, n) for k, p, n, *_ in NESTED} == declared
    for (kind, provider), spec in specs.SPECS.items():
        assert set(spec.settings_models) <= spec.settings_allowed, (kind, provider)


def test_valid_nested_object_reaches_the_service():
    with patch("api.services.pipecat.service_factory.CartesiaTTSService") as mock:
        create_tts_service(
            user_config_tts("cartesia", model="sonic-2"),
            audio_config(),
            tuning={
                "tts": {"cartesia": {"settings": {"generation_config": {"speed": 1.2}}}}
            },
        )
    assert mock.call_args.kwargs["settings"].generation_config.speed == 1.2
