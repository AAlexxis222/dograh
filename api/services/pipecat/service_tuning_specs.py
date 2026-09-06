"""Allow-lists that make ``service_tuning`` explicit per service.

pipecat's ``Settings.from_mapping`` sends unknown keys to ``extra`` silently
and most services never read ``extra`` (pass2/service-factory.md B1). This
table is what the PUT validates against so a misspelt knob is a 422, not a
no-op. ``settings_allowed`` must be a subset of the real dataclass fields
(guarded by test_service_tuning_schema.py); identity fields owned by the
model registry (model/voice/language/api_key) are never tunable here.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass
from typing import Any, Literal

from pipecat.services.deepgram.flux.base import DeepgramFluxSTTSettings
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.dograh.flux.stt import DograhFluxSTTService
from pipecat.services.elevenlabs.stt import (
    ElevenLabsRealtimeSTTService,
    ElevenLabsRealtimeSTTSettings,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings
from pipecat.services.settings import LLMSettings

ALL = "_all"
KINDS = ("stt", "tts", "llm", "realtime")
REGISTRY_OWNED = frozenset(
    {"model", "voice", "language", "api_key", "extra", "system_instruction"}
)


@dataclass(frozen=True)
class TuningSpec:
    kind: str
    provider: str
    settings_cls: type | None
    settings_allowed: frozenset[str]
    service_classes: tuple[type, ...] = ()
    ctor_allowed: frozenset[str] = frozenset()
    options_allowed: frozenset[str] = frozenset()
    nullable_extra: frozenset[str] = frozenset()

    def nullable(self) -> frozenset[str]:
        if self.settings_cls is None:
            return self.nullable_extra
        return nullable_fields(self.settings_cls) | self.nullable_extra

    def settings_classes(self) -> tuple[type, ...]:
        """The Settings dataclass(es) a field's type is checked against.

        Most rows have a single ``settings_cls``. A merged row (only
        ``stt.deepgram`` today) sets ``settings_cls=None`` because its
        allow-list unions two wire protocols (Flux/Nova) whose Settings
        dataclasses disagree on some field types; for those, fall back to
        each service class's own ``Settings`` attribute so a value is still
        checked against whichever class actually declares that field name.
        """
        if self.settings_cls is not None:
            return (self.settings_cls,)
        seen: list[type] = []
        for cls in self.service_classes:
            settings_cls = getattr(cls, "Settings", None)
            if settings_cls is not None and settings_cls not in seen:
                seen.append(settings_cls)
        return tuple(seen)


def _is_union(ann: Any) -> bool:
    return typing.get_origin(ann) is typing.Union or isinstance(ann, types.UnionType)


def nullable_fields(settings_cls: type) -> frozenset[str]:
    # Most pipecat service modules do NOT use
    # ``from __future__ import annotations``, so ``f.type`` is an evaluated
    # type object there and ``None`` is reachable via get_args. Only
    # settings.py itself deferred annotations (get_type_hints raises
    # NameError on LLMSettings, a TYPE_CHECKING-only import), so ``f.type``
    # is a literal string there and needs the string-split fallback; every
    # settings.py field used by this module is a plain scalar/_NotGiven
    # union with no Literal member, so a bare "|" split is enough for it.
    out = set()
    for f in dataclasses.fields(settings_cls):
        ann = f.type
        if isinstance(ann, str):
            parts = {p.strip() for p in ann.split("|")}
            is_nullable = "None" in parts
        else:
            is_nullable = _is_union(ann) and type(None) in typing.get_args(ann)
        if is_nullable:
            out.add(f.name)
    return frozenset(out)


def _fields(settings_cls: type, *exclude: str) -> frozenset[str]:
    names = {f.name for f in dataclasses.fields(settings_cls)}
    return frozenset(names - REGISTRY_OWNED - set(exclude))


FLUX_SETTINGS = _fields(
    DeepgramFluxSTTSettings
)  # eager_eot_threshold, eot_threshold, eot_timeout_ms, keyterm, min_confidence, numerals, language_hints

SPECS: dict[tuple[str, str], TuningSpec] = {}
# Deepgram is one provider with two wire protocols; the allow-list is the
# union and the applier picks the right Settings class by model (Task 5).
SPECS[("stt", "deepgram")] = TuningSpec(
    "stt",
    "deepgram",
    None,
    FLUX_SETTINGS | _fields(DeepgramSTTSettings),
    service_classes=(DeepgramFluxSTTService, DeepgramSTTService),
    ctor_allowed=frozenset({"url", "mip_opt_out", "tag"}),
    # "keyterm" deliberately excluded: test_null_only_on_nullable_fields
    # requires stt.deepgram.settings.keyterm=None to be rejected even though
    # DeepgramSTTService (Nova) itself defaults keyterm=None at connect time
    # (deepgram/stt.py:362) — the Flux settings field (DeepgramFluxSTTSettings
    # .keyterm: list | _NotGiven) has no None in its declared type either way.
    nullable_extra=nullable_fields(DeepgramFluxSTTSettings)
    | frozenset(
        {
            "endpointing",
            "keywords",
            "redact",
            "replace",
            "search",
            "utterance_end_ms",
        }
    ),
)
SPECS[("stt", "dograh")] = TuningSpec(
    "stt",
    "dograh",
    DeepgramFluxSTTSettings,
    FLUX_SETTINGS,
    service_classes=(DograhFluxSTTService,),
    ctor_allowed=frozenset({"mip_opt_out", "tag"}),
)
SPECS[("stt", "elevenlabs")] = TuningSpec(
    "stt",
    "elevenlabs",
    ElevenLabsRealtimeSTTSettings,
    _fields(ElevenLabsRealtimeSTTSettings),
    service_classes=(ElevenLabsRealtimeSTTService,),
    ctor_allowed=frozenset({"commit_strategy", "include_timestamps"}),
)
SPECS[("tts", "elevenlabs")] = TuningSpec(
    "tts",
    "elevenlabs",
    ElevenLabsTTSSettings,
    _fields(ElevenLabsTTSSettings),
    service_classes=(ElevenLabsTTSService,),
    ctor_allowed=frozenset({"auto_mode", "enable_ssml_parsing", "reconnect_on_error"}),
)
SPECS[("tts", "openai")] = TuningSpec(
    "tts",
    "openai",
    OpenAITTSSettings,
    _fields(OpenAITTSSettings) | frozenset({"voice"}),
    service_classes=(OpenAITTSService,),
)
SPECS[("tts", ALL)] = TuningSpec(
    "tts", ALL, None, frozenset(), ctor_allowed=frozenset({"silence_time_s"})
)
SPECS[("llm", ALL)] = TuningSpec(
    "llm", ALL, LLMSettings, frozenset({"temperature", "max_tokens"})
)
SPECS[("llm", "openai")] = TuningSpec(
    "llm",
    "openai",
    OpenAILLMSettings,
    _fields(
        OpenAILLMSettings,
        "filter_incomplete_user_turns",
        "user_turn_completion_config",
        "top_k",
    ),
    service_classes=(OpenAILLMService,),
    options_allowed=frozenset({"api", "reasoning", "verbosity"}),
)


def spec_for(kind: str, provider: str) -> TuningSpec | None:
    return SPECS.get((kind, provider))


_SCALAR_TYPES = {
    float: (int, float),
    int: (int,),
    bool: (bool,),
    str: (str,),
    list: (list,),
    dict: (dict,),
}
_SCALAR_NAMES = {
    "float": (int, float),
    "int": (int,),
    "bool": (bool,),
    "str": (str,),
    "list": (list,),
    "dict": (dict,),
}
# Sentinel/placeholder names that never carry a checkable value shape.
_UNCHECKED_NAMES = frozenset({"NoneType", "_NotGiven", "NotGiven"})


def _scalar_verdict(expected: tuple[type, ...], value: Any) -> bool | None:
    """True/False if ``expected`` settles the check; None to keep scanning
    other union members (a bool value never satisfies a non-bool numeric
    member, but a later member — e.g. another class's field variant — might
    still accept it)."""
    if isinstance(value, bool) and expected != (bool,):
        return None
    return isinstance(value, expected)


def _member_verdict(member: Any, value: Any) -> bool | None:
    """True: this member matches the value. False: this member is a
    checkable type (scalar or Literal) that rejects the value. None: this
    member neither confirms nor denies — it's a sentinel/None placeholder
    (skip) or a type this module doesn't verify (enum, pydantic model,
    ``Language``, TypeVar, ``typing.Any`` — left to the provider at connect
    time, so it doesn't veto a match found elsewhere, but also never
    single-handedly excuses an otherwise-wrong-typed value)."""
    if isinstance(member, str):
        # settings.py string-annotation path (see nullable_fields).
        head = member.split("[")[0].strip()
        if head in ("None", "Any") or head in _UNCHECKED_NAMES:
            return None
        expected = _SCALAR_NAMES.get(head)
        if expected is None:
            return None
        return _scalar_verdict(expected, value)

    if member is Any or member is type(None):
        return None
    name = getattr(member, "__name__", None)
    if name in _UNCHECKED_NAMES:
        return None

    origin = typing.get_origin(member)
    if origin is Literal:
        return value in typing.get_args(member)
    checked_type = origin if origin is not None else member
    expected = _SCALAR_TYPES.get(checked_type)
    if expected is None:
        return None  # enum / pydantic model / Language / TypeVar / etc.
    return _scalar_verdict(expected, value)


def _field_type_ok(f: dataclasses.Field, value: Any) -> bool | None:
    """True/False if this field's declared type settles whether ``value``
    matches; None if the field carries no checkable member at all (so the
    caller should try another class in a merged spec, or fall back to
    accepting)."""
    ann = f.type
    members = (
        ann.split("|")
        if isinstance(ann, str)
        else (typing.get_args(ann) if _is_union(ann) else (ann,))
    )
    saw_checkable = False
    for member in members:
        verdict = _member_verdict(member, value)
        if verdict is True:
            return True
        if verdict is False:
            saw_checkable = True
    if saw_checkable:
        return False
    return None


def _type_ok(settings_classes: tuple[type, ...], name: str, value: Any) -> bool:
    if value is None:
        return True
    found_field = False
    for settings_cls in settings_classes:
        f = next(
            (fld for fld in dataclasses.fields(settings_cls) if fld.name == name), None
        )
        if f is None:
            continue
        found_field = True
        verdict = _field_type_ok(f, value)
        if verdict is not False:
            return True  # a match, or no checkable member on this class
    return not found_field


def validate_service_tuning(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for kind in KINDS:
        providers = document.get(kind) or {}
        for provider, tuning in providers.items():
            spec = spec_for(kind, provider)
            if spec is None:
                errors.append(f"{kind}.{provider}: unknown provider")
                continue
            tuning = tuning or {}
            nullable = spec.nullable()
            for name, value in (tuning.get("settings") or {}).items():
                path = f"{kind}.{provider}.settings.{name}"
                # A name in the allow-list wins even when it is also a
                # REGISTRY_OWNED identity field elsewhere (e.g. tts.openai
                # exposes "voice" as a tunable setting, B6).
                if name in spec.settings_allowed:
                    if value is None and name not in nullable:
                        errors.append(f"{path}: null not allowed")
                    elif not _type_ok(spec.settings_classes(), name, value):
                        errors.append(f"{path}: wrong type")
                elif name in REGISTRY_OWNED:
                    errors.append(f"{path}: owned by the model configuration")
                else:
                    errors.append(f"{path}: unknown setting")
            for name in tuning.get("ctor") or {}:
                if name not in spec.ctor_allowed:
                    errors.append(f"{kind}.{provider}.ctor.{name}: not allowed")
            for name in tuning.get("options") or {}:
                if name not in spec.options_allowed:
                    errors.append(f"{kind}.{provider}.options.{name}: not allowed")
    return errors
