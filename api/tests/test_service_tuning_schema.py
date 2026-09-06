import dataclasses
import inspect

import pytest
from pydantic import ValidationError

from api.schemas.service_tuning import ServiceTuning
from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
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
    with pytest.raises(ValidationError, match="stt.deepgram.settings.eot_treshold: unknown"):
        _validate({"stt": {"deepgram": {"settings": {"eot_treshold": 0.8}}}})


def test_registry_owned_fields_are_rejected():
    for key in ("model", "language", "api_key", "extra"):
        with pytest.raises(ValidationError, match=f"stt.deepgram.settings.{key}"):
            _validate({"stt": {"deepgram": {"settings": {key: "x"}}}})


def test_null_only_on_nullable_fields():
    _validate({"stt": {"deepgram": {"settings": {"eager_eot_threshold": None}}}})  # eager OFF
    with pytest.raises(ValidationError, match="stt.deepgram.settings.keyterm: null not allowed"):
        _validate({"stt": {"deepgram": {"settings": {"keyterm": None}}}})


def test_ctor_kwarg_must_be_allow_listed():
    _validate({"stt": {"deepgram": {"ctor": {"mip_opt_out": True}}}})
    with pytest.raises(ValidationError, match="stt.deepgram.ctor.sample_rate: not allowed"):
        _validate({"stt": {"deepgram": {"ctor": {"sample_rate": 8000}}}})


def test_options_only_where_declared():
    _validate({"llm": {"openai": {"options": {"reasoning": {"effort": "low"}}}}})
    with pytest.raises(ValidationError, match="stt.deepgram.options.api: not allowed"):
        _validate({"stt": {"deepgram": {"options": {"api": "x"}}}})


def test_all_section_uses_common_fields_only():
    _validate({"llm": {"_all": {"settings": {"temperature": 0.3, "max_tokens": 200}}}})
    _validate({"tts": {"_all": {"ctor": {"silence_time_s": 0.4}}}})
    with pytest.raises(ValidationError, match="llm._all.settings.reasoning_effort"):
        _validate({"llm": {"_all": {"settings": {"reasoning_effort": "low"}}}})


def test_scope_flags_and_forbid_extra():
    cfg = _validate({"scope": {"extraction": True}})
    assert cfg.service_tuning.scope.extraction is True and cfg.service_tuning.scope.voicemail is False
    with pytest.raises(ValidationError):
        _validate({"scope": {"qa": True}})


def test_root_null_on_service_tuning_is_unset():
    cfg = WorkflowConfigurationDefaults.model_validate({"service_tuning": None})
    assert "service_tuning" not in cfg.model_dump(exclude_unset=True)


def test_specs_allow_lists_are_subsets_of_real_settings_fields():
    # Ruling 2 (controller): the allow-list may re-admit a REGISTRY_OWNED name
    # (tts.openai exposes "voice", B6) — that single exception is exempted
    # from the "never re-admits an owned field" half of this assertion.
    allowed_owned = {("tts", "openai"): {"voice"}}
    for (kind, provider), spec in specs.SPECS.items():
        if spec.settings_cls is None:
            continue
        real = {f.name for f in dataclasses.fields(spec.settings_cls)}
        assert spec.settings_allowed <= real, (kind, provider, spec.settings_allowed - real)
        exempt = allowed_owned.get((kind, provider), set())
        assert not (spec.settings_allowed & specs.REGISTRY_OWNED - exempt), (kind, provider)


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
