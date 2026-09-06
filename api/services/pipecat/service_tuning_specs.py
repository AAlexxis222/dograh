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
from dataclasses import dataclass
from typing import Any

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
from pipecat.services.settings import LLMSettings, TTSSettings

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


def nullable_fields(settings_cls: type) -> frozenset[str]:
    # settings.py uses ``from __future__ import annotations`` so ``f.type`` is
    # the literal annotation string; get_type_hints raises NameError on
    # LLMSettings (TYPE_CHECKING-only import), so parse the string instead.
    out = set()
    for f in dataclasses.fields(settings_cls):
        parts = {p.strip() for p in str(f.type).split("|")}
        if "None" in parts:
            out.add(f.name)
    return frozenset(out)


def _fields(settings_cls: type, *exclude: str) -> frozenset[str]:
    names = {f.name for f in dataclasses.fields(settings_cls)}
    return frozenset(names - REGISTRY_OWNED - set(exclude))


FLUX_SETTINGS = _fields(DeepgramFluxSTTSettings)  # eager_eot_threshold, eot_threshold, eot_timeout_ms, keyterm, min_confidence, numerals, language_hints

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


_SCALARS = {"float": (int, float), "int": int, "bool": bool, "str": str, "list": list, "dict": dict}


def _type_ok(settings_cls: type | None, name: str, value: Any) -> bool:
    if settings_cls is None or value is None:
        return True
    ann = next((str(f.type) for f in dataclasses.fields(settings_cls) if f.name == name), "")
    parts = [p.strip() for p in ann.split("|")]
    for part in parts:
        head = part.split("[")[0]
        expected = _SCALARS.get(head)
        if expected is None:
            return True  # Literal/enum/pydantic: left to the provider at connect time
        if isinstance(value, bool) and expected is not bool:
            continue
        if isinstance(value, expected):
            return True
    return False


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
                    elif not _type_ok(spec.settings_cls, name, value):
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
