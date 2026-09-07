"""Settings whose value is a provider object (pydantic model or dataclass).

``from_mapping`` stores the JSON object unchanged and the branch converts it
(``model_validate`` / ``SpeakerIdentifier(**e)``) at run creation, so before
this a misspelt key inside the object was silently dropped (those models
ignore unknown keys) and a wrong scalar was a ValidationError when the run
was created. Both are a named 422 at the PUT now (#11, #12).
"""

import dataclasses
import inspect
import sys
import typing
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.pipecat import service_factory
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


def test_list_index_is_spelled_the_same_by_both_checks():
    # The key walk and pydantic's own error location agree on ``[i]``.
    with pytest.raises(
        ValidationError, match=r"known_speakers\[0\]\.speakers: unknown key"
    ):
        WorkflowConfigurationDefaults.model_validate(
            _doc("stt", "speechmatics", "known_speakers", [{"speakers": ["S1"]}])
        )
    with pytest.raises(
        ValidationError, match=r"known_speakers\[0\]\.speaker_identifiers: Input"
    ):
        WorkflowConfigurationDefaults.model_validate(
            _doc(
                "stt",
                "speechmatics",
                "known_speakers",
                [{**_SPEAKER, "speaker_identifiers": "S1"}],
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


def _resolve(annotation, namespace):
    """Evaluate a string / ForwardRef annotation in its module's namespace.

    ``GoogleLLMSettings.thinking`` is ``Union["GoogleLLMService.ThinkingConfig",
    ...]``: a dotted forward reference ``get_type_hints`` cannot resolve, but
    the module namespace can, since the class exists once the module loaded.
    """
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        try:
            return eval(annotation, dict(namespace))  # noqa: S307 - test-only
        except Exception:  # a name the module cannot see: nothing to check
            return None
    return annotation


def _model_leaves(annotation, namespace):
    """Every pydantic model or dataclass reachable inside ``annotation``,
    through Optional / Union / list / ForwardRef."""
    annotation = _resolve(annotation, namespace)
    if annotation is None:
        return set()
    if typing.get_args(annotation):
        return {
            leaf
            for arg in typing.get_args(annotation)
            for leaf in _model_leaves(arg, namespace)
        }
    if inspect.isclass(annotation) and (
        issubclass(annotation, BaseModel) or dataclasses.is_dataclass(annotation)
    ):
        return {annotation}
    return set()


def _object_valued_settings():
    """(kind, provider, name) for every allow-listed setting whose declared
    type contains a provider object, read off the Settings dataclasses.

    The realtime rows have no dataclass: their object knobs are the ones the
    factory coerces with a model (``_GEMINI_REALTIME_COERCIONS``), taken from
    the factory rather than restated here.
    """
    found = set()
    for (kind, provider), spec in specs.SPECS.items():
        for cls in spec.settings_classes():
            namespace = vars(sys.modules[cls.__module__])
            for f in dataclasses.fields(cls):
                if f.name in spec.settings_allowed and _model_leaves(f.type, namespace):
                    found.add((kind, provider, f.name))
        if kind == "realtime":
            for name in service_factory._GEMINI_REALTIME_COERCIONS:
                if name in spec.settings_allowed:
                    found.add((kind, provider, name))
    return found


def test_every_object_valued_setting_has_a_settings_models_row():
    # ``settings_allowed`` is derived from the pinned pipecat dataclasses, so
    # a submodule bump that adds an object-valued field would put it in the
    # allow-list with no model to check against and reopen #11/#12 silently.
    # This scan reads the annotations, not a hand-kept list.
    declared = {
        (k, p, n) for (k, p), s in specs.SPECS.items() for n in s.settings_models
    }
    assert _object_valued_settings() == declared
    assert {(k, p, n) for k, p, n, *_ in NESTED} == declared  # the cases above
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


def test_shape_walk_resolves_string_annotations_on_a_dataclass():
    # Under ``from __future__ import annotations`` a dataclass field type is
    # a string; the boolean-on-numeric rule must still see the float.
    @dataclasses.dataclass
    class Synthetic:
        speed: "float"
        label: "str"

    assert specs._model_shape_errors(Synthetic, {"speed": True}, "p") == [
        "p.speed: wrong type"
    ]
    assert specs._model_shape_errors(Synthetic, {"speed": 1.5}, "p") == []
