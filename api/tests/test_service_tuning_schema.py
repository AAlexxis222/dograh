import dataclasses
import inspect

import pytest
from pydantic import ValidationError

from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.configuration import service_tuning_bounds as bounds
from api.services.pipecat import service_tuning_specs as specs


def _validate(doc):
    return WorkflowConfigurationDefaults.model_validate({"service_tuning": doc})


def test_flux_thresholds_accepted_and_kept_sparse():
    cfg = _validate({"stt": {"deepgram": {"settings": {"eot_threshold": 0.8}}}})
    assert cfg.model_dump(exclude_unset=True)["service_tuning"] == {
        "stt": {"deepgram": {"settings": {"eot_threshold": 0.8}}}
    }


def test_unknown_provider_is_422():
    with pytest.raises(ValidationError, match="stt.nope: unknown provider"):
        _validate({"stt": {"nope": {"settings": {}}}})


def test_unknown_settings_key_is_422_not_silent_extra():
    with pytest.raises(
        ValidationError, match="stt.deepgram.settings.eot_treshold: unknown"
    ):
        _validate({"stt": {"deepgram": {"settings": {"eot_treshold": 0.8}}}})


def test_registry_owned_fields_are_rejected():
    for key in ("model", "language", "api_key", "extra"):
        with pytest.raises(ValidationError, match=f"stt.deepgram.settings.{key}"):
            _validate({"stt": {"deepgram": {"settings": {key: "x"}}}})


def test_null_only_on_nullable_fields():
    _validate(
        {"stt": {"deepgram": {"settings": {"eager_eot_threshold": None}}}}
    )  # eager OFF
    with pytest.raises(
        ValidationError, match="stt.deepgram.settings.keyterm: null not allowed"
    ):
        _validate({"stt": {"deepgram": {"settings": {"keyterm": None}}}})


def test_ctor_kwarg_must_be_allow_listed():
    _validate({"stt": {"deepgram": {"ctor": {"mip_opt_out": True}}}})
    with pytest.raises(
        ValidationError, match="stt.deepgram.ctor.sample_rate: not allowed"
    ):
        _validate({"stt": {"deepgram": {"ctor": {"sample_rate": 8000}}}})


def test_ctor_kwarg_values_are_checked_where_the_set_is_closed():
    _validate({"stt": {"elevenlabs": {"ctor": {"commit_strategy": "manual"}}}})
    for value in ("bogus", ["vad"]):
        # An unhashable value must not raise TypeError out of the closed-set
        # check — pydantic would surface that as a 500, not a 422.
        with pytest.raises(
            ValidationError,
            match="stt.elevenlabs.ctor.commit_strategy: must be one of manual, vad",
        ):
            _validate({"stt": {"elevenlabs": {"ctor": {"commit_strategy": value}}}})


STT_KEYTERM_KNOBS = {
    "assemblyai": {"keyterms_prompt": ["Marbella"]},
    "azure_speech": {"profanity": "raw"},
    "cartesia": {"keyterm": ["Marbella"]},
    "elevenlabs": {"keyterms": ["Marbella"]},
    "gladia": {"realtime_processing": {"custom_vocabulary": True}},
    "google": {"enable_automatic_punctuation": True},
    "huggingface": {"return_timestamps": True},
    "openai": {"prompt": "Marbella"},
    "sarvam": {"prompt": "Marbella"},
    "smallest": {"keywords": "Marbella:2"},
    "speaches": {"prompt": "Marbella"},
    "speechmatics": {"additional_vocab": [{"content": "Marbella"}]},
}


@pytest.mark.parametrize("provider,settings", sorted(STT_KEYTERM_KNOBS.items()))
def test_every_stt_provider_has_a_row_and_takes_its_biasing_knob(provider, settings):
    _validate({"stt": {provider: {"settings": settings}}})


def test_speechmatics_operating_point_stays_registry_owned():
    # It *is* the model: sf maps user_config.stt.model to an OperatingPoint and
    # the service writes it back into settings.model (speechmatics/stt.py:512).
    with pytest.raises(
        ValidationError, match="stt.speechmatics.settings.operating_point: unknown"
    ):
        _validate(
            {"stt": {"speechmatics": {"settings": {"operating_point": "enhanced"}}}}
        )


def test_speechmatics_extra_params_stays_closed():
    # _build_config splats it onto the SDK config by hasattr
    # (speechmatics/stt.py:795-799), so a free-form dict would re-open every
    # excluded field — operating_point included, which then crashes at :512.
    with pytest.raises(
        ValidationError, match="stt.speechmatics.settings.extra_params: unknown"
    ):
        _validate(
            {
                "stt": {
                    "speechmatics": {
                        "settings": {"extra_params": {"operating_point": "enhanced"}}
                    }
                }
            }
        )


def test_enum_typed_setting_checks_the_enum_values():
    _validate({"stt": {"speechmatics": {"settings": {"focus_mode": "retain"}}}})
    for field, value in (
        ("focus_mode", "bogus"),
        # An unhashable value must not raise TypeError out of the membership
        # check, and a bad mode must not reach TurnDetectionMode(...) in the
        # factory as a ValueError.
        ("focus_mode", ["retain"]),
        ("turn_detection_mode", "bogus"),
        ("turn_detection_mode", ["external"]),
    ):
        with pytest.raises(
            ValidationError,
            match=f"stt.speechmatics.settings.{field}: wrong type",
        ):
            _validate({"stt": {"speechmatics": {"settings": {field: value}}}})


def test_ctor_choices_and_types_are_declared_ctor_kwargs():
    for (kind, provider), spec in specs.SPECS.items():
        for declared in (set(spec.ctor_choices), set(spec.ctor_types)):
            assert declared <= spec.ctor_allowed, (
                kind,
                provider,
                declared - spec.ctor_allowed,
            )


def test_scalar_ctor_kwargs_declare_their_type():
    """A ctor kwarg whose constructor annotation is a plain scalar must be in
    ``ctor_types``; otherwise any JSON value reaches the constructor raw (the
    falsy ``vad_force_turn_endpoint`` bypass). Ambiguous annotations
    (``bool | None``, enums, ``Any``) are skipped — those rows declare their
    type by hand."""
    for (kind, provider), spec in specs.SPECS.items():
        for name in spec.ctor_allowed:
            annotations = set()
            for cls in spec.service_classes:
                for klass in cls.__mro__:
                    init = klass.__dict__.get("__init__")
                    if init is None:
                        continue
                    param = inspect.signature(init).parameters.get(name)
                    if param is not None:
                        annotations.add(param.annotation)
            if len(annotations) == 1 and next(iter(annotations)) in (
                bool,
                str,
                int,
                float,
            ):
                assert spec.ctor_types.get(name) == next(iter(annotations)), (
                    kind,
                    provider,
                    name,
                )


def test_language_aliases_stay_registry_owned():
    for provider, key in (
        ("assemblyai", "language_code"),
        ("assemblyai", "language_codes"),
        ("google", "languages"),
        ("google", "language_codes"),
        ("gladia", "language_config"),
    ):
        with pytest.raises(
            ValidationError, match=f"stt.{provider}.settings.{key}: unknown"
        ):
            _validate({"stt": {provider: {"settings": {key: ["en"]}}}})


def test_openai_stt_api_option_selects_the_service_and_its_settings():
    _validate({"stt": {"openai": {"options": {"api": "segments"}}}})
    _validate(
        {
            "stt": {
                "openai": {
                    "options": {"api": "realtime"},
                    "settings": {"noise_reduction": "near_field"},
                }
            }
        }
    )
    for value in ("grpc", ["realtime"], None):
        with pytest.raises(
            ValidationError, match="stt.openai.options.api: must be one of"
        ):
            _validate({"stt": {"openai": {"options": {"api": value}}}})


def test_openai_stt_settings_must_match_the_selected_api():
    # from_mapping sends a field the selected service doesn't declare to
    # ``extra``, which neither service reads: a silent no-op either way.
    with pytest.raises(
        ValidationError,
        match="stt.openai.settings.noise_reduction: only available with options.api=realtime",
    ):
        _validate({"stt": {"openai": {"settings": {"noise_reduction": "near_field"}}}})
    with pytest.raises(
        ValidationError,
        match="stt.openai.settings.temperature: only available with options.api=segments",
    ):
        _validate(
            {
                "stt": {
                    "openai": {
                        "options": {"api": "realtime"},
                        "settings": {"temperature": 0.2},
                    }
                }
            }
        )
    # ``prompt`` is declared by both services, so it needs no pairing.
    _validate({"stt": {"openai": {"settings": {"prompt": "Marbella"}}}})


def test_realtime_rows_check_types_and_closed_sets_without_a_settings_class():
    _validate(
        {
            "realtime": {
                "openai_realtime": {
                    "settings": {"noise_reduction": "far_field", "speed": 1}
                }
            }
        }
    )
    for settings, message in (
        ({"noise_reduction": "loud"}, "noise_reduction: must be one of"),
        # Unhashable values must not raise TypeError out of the closed-set
        # check — pydantic surfaces that as a 500, not a 422.
        ({"noise_reduction": ["far_field"]}, "noise_reduction: must be one of"),
        ({"reasoning_effort": "extreme"}, "reasoning_effort: must be one of"),
        ({"tool_choice": "maybe"}, "tool_choice: must be one of"),
        ({"speed": "fast"}, "speed: wrong type"),
        ({"max_output_tokens": True}, "max_output_tokens: wrong type"),
        # Every realtime destination already defaults to "unset", so null
        # would be a no-op; omitting the key is how a default is written.
        ({"speed": None}, "speed: null not allowed"),
        ({"turn_detection": False}, "turn_detection: unknown setting"),
    ):
        with pytest.raises(
            ValidationError, match=f"realtime.openai_realtime.settings.{message}"
        ):
            _validate({"realtime": {"openai_realtime": {"settings": settings}}})


def test_ultravox_extra_cannot_reopen_what_the_call_body_owns():
    # ``request_body | params.extra`` (ultravox/llm.py:361) puts extra last, so
    # a key the service already set would win over the run's configuration —
    # and the wrapper rewrites firstSpeakerSettings on every call anyway.
    _validate(
        {
            "realtime": {
                "ultravox_realtime": {"settings": {"extra": {"recordingEnabled": True}}}
            }
        }
    )
    for key, owner in (
        ("model", "the model configuration"),
        ("voice", "the model configuration"),
        ("medium", "the transport"),
        ("selectedTools", "the workflow tools"),
        ("systemPrompt", "the workflow node"),
        ("firstSpeakerSettings", "the greeting decision"),
        # The wrapper strips and rewrites only ``firstSpeakerSettings``
        # (realtime/ultravox_realtime.py:425-436), so Ultravox's older
        # spelling would survive the merge and re-decide who opens the call.
        ("firstSpeaker", "the greeting decision"),
        # Ultravox's own endpointing: who ends the user turn (spec §6).
        ("vadSettings", "turn handling"),
    ):
        with pytest.raises(
            ValidationError,
            match=f"realtime.ultravox_realtime.settings.extra.{key}: owned by {owner}",
        ):
            _validate(
                {"realtime": {"ultravox_realtime": {"settings": {"extra": {key: "x"}}}}}
            )


def test_realtime_provider_must_be_known():
    with pytest.raises(ValidationError, match="realtime.openai: unknown provider"):
        _validate({"realtime": {"openai": {"settings": {}}}})


def test_options_only_where_declared():
    _validate({"llm": {"openai": {"options": {"reasoning": {"effort": "low"}}}}})
    with pytest.raises(ValidationError, match="stt.deepgram.options.api: not allowed"):
        _validate({"stt": {"deepgram": {"options": {"api": "x"}}}})


def test_openai_option_values_are_checked_not_just_their_names():
    _validate(
        {
            "llm": {
                "openai": {
                    "options": {
                        "api": "chat_completions",
                        "reasoning": {"effort": "high"},
                        "verbosity": "high",
                    }
                }
            }
        }
    )
    for options, message in (
        ({"api": "grpc"}, "llm.openai.options.api: must be one of"),
        # Ruling A: explicit null is not one of the choices either.
        ({"api": None}, "llm.openai.options.api: must be one of"),
        ({"reasoning": "high"}, "llm.openai.options.reasoning: wrong type"),
        (
            {"reasoning": {"effort": "extreme"}},
            "llm.openai.options.reasoning.effort: must be one of",
        ),
        (
            {"reasoning": {"depth": "low"}},
            "llm.openai.options.reasoning.depth: unknown key",
        ),
        # Ruling B: only the Responses API carries a reasoning summary, and
        # that path is parked — so the knob is a named 422, not a no-op.
        (
            {"reasoning": {"summary": "concise"}},
            "llm.openai.options.reasoning.summary: not available in this build",
        ),
        ({"verbosity": "loud"}, "llm.openai.options.verbosity: must be one of"),
        # Unhashable values: a bare `value in frozenset` raises TypeError,
        # which pydantic does NOT convert — the PUT would 500 instead of 422.
        ({"verbosity": ["low"]}, "llm.openai.options.verbosity: must be one of"),
        ({"api": ["responses"]}, "llm.openai.options.api: must be one of"),
        (
            {"reasoning": {"effort": ["low"]}},
            "llm.openai.options.reasoning.effort: must be one of",
        ),
        (
            {"reasoning": {"effort": {"a": 1}}},
            "llm.openai.options.reasoning.effort: must be one of",
        ),
        (
            {"reasoning": {"summary": ["auto"]}},
            "llm.openai.options.reasoning.summary: not available in this build",
        ),
    ):
        with pytest.raises(ValidationError, match=message):
            _validate({"llm": {"openai": {"options": options}}})


def test_llm_temperature_bounds_track_every_llm_row():
    # Ruling 2 (controller): the bounds module can't import pipecat, so it
    # repeats the provider names; this is the guard against them drifting.
    assert set(bounds.LLM_PROVIDERS) == {
        provider for (kind, provider) in specs.SPECS if kind == "llm"
    }


def test_all_section_uses_common_fields_only():
    _validate({"llm": {"_all": {"settings": {"temperature": 0.3, "max_tokens": 200}}}})
    with pytest.raises(ValidationError, match="llm._all.settings.reasoning_effort"):
        _validate({"llm": {"_all": {"settings": {"reasoning_effort": "low"}}}})


def test_tts_all_silence_time_is_gated_while_no_service_pushes_silence():
    # The only ``tts._all`` ctor kwarg, and inert: ``silence_time_s`` is read
    # solely under ``push_silence_after_stop`` (tts_service.py:902), which
    # nothing sets. Declared and rejected by name rather than stored as a
    # placebo; the row itself stays, so the name survives the flip.
    assert specs.SPECS[("tts", specs.ALL)].ctor_allowed == {"silence_time_s"}
    with pytest.raises(ValidationError, match="not wired in this build"):
        _validate({"tts": {"_all": {"ctor": {"silence_time_s": 0.4}}}})


def test_scope_flags_and_forbid_extra():
    cfg = _validate({"scope": {"extraction": True}})
    assert (
        cfg.service_tuning.scope.extraction is True
        and cfg.service_tuning.scope.voicemail is False
    )
    with pytest.raises(ValidationError):
        _validate({"scope": {"qa": True}})


def test_root_null_on_service_tuning_is_unset():
    cfg = WorkflowConfigurationDefaults.model_validate({"service_tuning": None})
    assert "service_tuning" not in cfg.model_dump(exclude_unset=True)


def test_typing_union_literal_field_is_nullable():
    # ElevenLabsTTSSettings.apply_text_normalization renders (no
    # `from __future__ import annotations` in elevenlabs/tts.py) as
    # ``typing.Union[Literal['auto','on','off'], NoneType, _NotGiven]`` —
    # str(f.type) shows "NoneType", not "None", so a plain "|" string split
    # would miss it. Regression for the false 422 this caused.
    spec = specs.spec_for("tts", "elevenlabs")
    assert "apply_text_normalization" in spec.nullable()
    _validate({"tts": {"elevenlabs": {"settings": {"apply_text_normalization": None}}}})


def test_wrong_type_is_rejected_not_silently_accepted():
    with pytest.raises(
        ValidationError, match="stt.deepgram.settings.eot_threshold: wrong type"
    ):
        _validate({"stt": {"deepgram": {"settings": {"eot_threshold": "banana"}}}})
    with pytest.raises(
        ValidationError, match="stt.deepgram.settings.numerals: wrong type"
    ):
        _validate({"stt": {"deepgram": {"settings": {"numerals": "yes"}}}})
    with pytest.raises(
        ValidationError,
        match="tts.elevenlabs.settings.apply_text_normalization: wrong type",
    ):
        _validate(
            {
                "tts": {
                    "elevenlabs": {
                        "settings": {"apply_text_normalization": "sometimes"}
                    }
                }
            }
        )


BOOL_ON_A_NON_BOOL_FIELD = [
    ("stt", "dograh", "eot_threshold"),
    ("stt", "deepgram", "eot_threshold"),
    ("llm", "openai", "temperature"),
    ("tts", "elevenlabs", "stability"),
    # Nova declares this one as ``Any``, so the field says nothing about the
    # value; the row declares ``int`` instead.
    ("stt", "deepgram", "endpointing"),
]


@pytest.mark.parametrize("kind,provider,name", BOOL_ON_A_NON_BOOL_FIELD)
def test_bool_is_rejected_on_a_numeric_setting(kind, provider, name):
    # ``isinstance(True, int)`` is True, so without an explicit rejection a
    # bool passes the type check, then skips the cascade clamp (cascade.py:196,
    # which excludes bools) and reaches the wire as ``eot_threshold=True``.
    with pytest.raises(
        ValidationError, match=f"{kind}.{provider}.settings.{name}: wrong type"
    ):
        _validate({kind: {provider: {"settings": {name: True}}}})


def test_bool_still_accepted_where_the_field_is_a_bool():
    _validate({"stt": {"deepgram": {"settings": {"numerals": True}}}})
    _validate({"stt": {"dograh": {"settings": {"numerals": False}}}})
    _validate({"tts": {"elevenlabs": {"settings": {"use_speaker_boost": True}}}})
    _validate({"stt": {"deepgram": {"settings": {"endpointing": 250}}}})


def test_pydantic_typed_settings_take_a_json_object_only():
    # Fields whose declared type is a provider model carry no member this
    # module can check, so anything used to pass the PUT and then raise out of
    # the branch's own ``model_validate``/``ThinkingConfig(**value)`` at run
    # creation — a 500 on the call, not a 422 on the save.
    _validate({"llm": {"google": {"settings": {"thinking": {"thinking_budget": 512}}}}})
    for kind, provider, name in (
        ("llm", "google", "thinking"),
        ("llm", "google_vertex", "thinking"),
        ("stt", "gladia", "pre_processing"),
        ("stt", "gladia", "realtime_processing"),
        ("stt", "gladia", "messages_config"),
        ("tts", "cartesia", "generation_config"),
    ):
        for value in ("x", True):
            with pytest.raises(
                ValidationError, match=f"{kind}.{provider}.settings.{name}: wrong type"
            ):
                _validate({kind: {provider: {"settings": {name: value}}}})


def test_openai_tts_voice_cannot_be_nulled():
    # The field is ``str | None`` upstream, but None is not "unset" here: the
    # service yields an ErrorFrame instead of audio on every turn
    # (openai/tts.py:254-256).
    _validate({"tts": {"openai": {"settings": {"voice": "nova"}}}})
    with pytest.raises(
        ValidationError, match="tts.openai.settings.voice: null not allowed"
    ):
        _validate({"tts": {"openai": {"settings": {"voice": None}}}})


def test_rime_inline_speed_alpha_is_rejected_by_name():
    # The websocket service reads it neither in _build_ws_params nor in
    # _build_msg, so accepting it would store a placebo.
    with pytest.raises(
        ValidationError, match="tts.rime.settings.inlineSpeedAlpha: unknown setting"
    ):
        _validate({"tts": {"rime": {"settings": {"inlineSpeedAlpha": 1.2}}}})


def test_wrong_type_check_does_not_false_reject_valid_values():
    _validate({"stt": {"deepgram": {"settings": {"language_hints": ["en"]}}}})
    _validate({"stt": {"deepgram": {"settings": {"keyterm": ["a"]}}}})
    _validate(
        {"stt": {"deepgram": {"settings": {"eot_threshold": 1}}}}
    )  # int for float


def test_specs_allow_lists_are_subsets_of_real_settings_fields():
    # Ruling 2 (controller): the allow-list may re-admit a REGISTRY_OWNED name
    # (tts.openai exposes "voice", B6) — that single exception is exempted
    # from the "never re-admits an owned field" half of this assertion.
    allowed_owned = {("tts", "openai"): {"voice"}}
    for (kind, provider), spec in specs.SPECS.items():
        if spec.settings_cls is None:
            continue
        real = {f.name for f in dataclasses.fields(spec.settings_cls)}
        assert spec.settings_allowed <= real, (
            kind,
            provider,
            spec.settings_allowed - real,
        )
        exempt = allowed_owned.get((kind, provider), set())
        assert not (spec.settings_allowed & specs.REGISTRY_OWNED - exempt), (
            kind,
            provider,
        )


def _ctor_params(cls: type) -> set[str]:
    """Union of ``__init__`` parameters across the MRO: a service's own
    ``__init__`` commonly forwards unrecognised kwargs to ``super().__init__``
    (e.g. ``ElevenLabsTTSService`` -> ``WebsocketTTSService`` ->
    ``WebsocketService.reconnect_on_error``), so a single class's own
    signature under-counts what it actually accepts."""
    params: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is not None:
            params |= set(inspect.signature(init).parameters)
    return params


def test_specs_ctor_allow_lists_exist_on_constructors():
    # Ruling 3 (controller): checked against the UNION of constructor params
    # over spec.service_classes, not per class — Deepgram is one provider
    # with two services (Flux takes "url", Nova takes "base_url"); the
    # applier filters per class in Task 5.
    for (kind, provider), spec in specs.SPECS.items():
        if not spec.service_classes:
            continue
        union: set[str] = set()
        for cls in spec.service_classes:
            union |= _ctor_params(cls)
        assert spec.ctor_allowed <= union, (kind, provider, spec.ctor_allowed - union)


@pytest.mark.parametrize(
    "provider,name",
    [
        # google/llm.py:385-397 (_build_generation_params) reads temperature,
        # top_p, top_k, max_tokens, safety_settings and thinking only; Vertex
        # inherits it.
        ("google", "frequency_penalty"),
        ("google", "presence_penalty"),
        ("google", "seed"),
        ("google_vertex", "frequency_penalty"),
        ("google_vertex", "presence_penalty"),
        ("google_vertex", "seed"),
        # aws/llm.py:265-271 (_build_inference_config) reads max_tokens,
        # temperature, top_p and stop_sequences only.
        ("aws_bedrock", "frequency_penalty"),
        ("aws_bedrock", "presence_penalty"),
        ("aws_bedrock", "seed"),
        # sarvam/llm.py:137 pops it off the chat-completions payload.
        ("sarvam", "max_completion_tokens"),
    ],
)
def test_llm_knobs_the_serializer_never_reads_are_rejected_by_name(provider, name):
    with pytest.raises(
        ValidationError, match=f"llm.{provider}.settings.{name}: unknown setting"
    ):
        _validate({"llm": {provider: {"settings": {name: 1}}}})


def test_google_stt_separate_recognition_per_channel_is_rejected_by_name():
    # google/stt.py:846 hard-codes audio_channel_count=1 and _connect
    # (:849-861) never reads the field.
    with pytest.raises(
        ValidationError,
        match="stt.google.settings.use_separate_recognition_per_channel: unknown setting",
    ):
        _validate(
            {
                "stt": {
                    "google": {
                        "settings": {"use_separate_recognition_per_channel": True}
                    }
                }
            }
        )


def test_scope_filler_is_gated_until_the_filler_role_exists(monkeypatch):
    _validate({"scope": {"filler": False}})
    with pytest.raises(ValidationError, match="scope.filler: .*FILLER_ROLE_AVAILABLE"):
        _validate({"scope": {"filler": True}})
    monkeypatch.setattr(specs, "FILLER_ROLE_AVAILABLE", True)
    assert _validate({"scope": {"filler": True}}).service_tuning.scope.filler is True


@pytest.mark.parametrize("name", ["keyterm", "keywords", "replace", "search"])
@pytest.mark.parametrize("value", [7, {"a": True}, True])
def test_deepgram_list_or_string_knobs_reject_other_shapes(name, value):
    # Nova declares these as ``Any`` (deepgram/stt.py:215-222), so nothing on
    # the field constrains the value; the wire takes a string or a list.
    with pytest.raises(
        ValidationError, match=f"stt.deepgram.settings.{name}: wrong type"
    ):
        _validate({"stt": {"deepgram": {"settings": {name: value}}}})


def test_deepgram_list_or_string_knobs_take_both_shapes():
    _validate({"stt": {"deepgram": {"settings": {"keyterm": "Marbella"}}}})
    _validate({"stt": {"deepgram": {"settings": {"keyterm": ["Marbella"]}}}})
    _validate({"stt": {"deepgram": {"settings": {"redact": True}}}})
    _validate({"stt": {"deepgram": {"settings": {"redact": "pci"}}}})
    _validate({"stt": {"deepgram": {"settings": {"redact": ["pci", "ssn"]}}}})
    with pytest.raises(
        ValidationError, match="stt.deepgram.settings.redact: wrong type"
    ):
        _validate({"stt": {"deepgram": {"settings": {"redact": 3}}}})
