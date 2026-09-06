"""Allow-lists that make ``service_tuning`` explicit per service.

pipecat's ``Settings.from_mapping`` sends unknown keys to ``extra`` silently
and most services never read ``extra`` (pass2/service-factory.md B1). This
table is what the PUT validates against so a misspelt knob is a 422, not a
no-op. ``settings_allowed`` must be a subset of the real dataclass fields
(guarded by test_service_tuning_schema.py); identity fields owned by the
model registry (model/voice/language/api_key) are never tunable here.

Three build gates below name a knob that this build cannot honour rather than
letting it through as a silent no-op or a broken run (§1.2):
``RESPONSES_API_AVAILABLE``, ``OPENAI_REALTIME_STT_AVAILABLE`` and
``PROVIDER_TURN_DETECTION_AVAILABLE``. Each is a constant the PR that wires
the feature flips.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pipecat.services.assemblyai.stt import AssemblyAISTTService, AssemblyAISTTSettings
from pipecat.services.aws.llm import AWSBedrockLLMSettings
from pipecat.services.azure.llm import AzureLLMSettings
from pipecat.services.azure.stt import AzureSTTSettings
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaSTTSettings
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.services.deepgram.flux.base import DeepgramFluxSTTSettings
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.dograh.flux.stt import DograhFluxSTTService
from pipecat.services.elevenlabs.stt import (
    ElevenLabsRealtimeSTTService,
    ElevenLabsRealtimeSTTSettings,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
from pipecat.services.gladia.stt import GladiaSTTSettings
from pipecat.services.google.llm import GoogleLLMSettings
from pipecat.services.google.stt import GoogleSTTSettings
from pipecat.services.google.vertex.llm import GoogleVertexLLMSettings
from pipecat.services.groq.llm import GroqLLMSettings
from pipecat.services.huggingface.llm import HuggingFaceLLMSettings
from pipecat.services.huggingface.stt import HuggingFaceSTTSettings
from pipecat.services.minimax.llm import MiniMaxLLMSettings
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTSettings
from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings
from pipecat.services.openrouter.llm import OpenRouterLLMSettings
from pipecat.services.sarvam.llm import SarvamLLMSettings
from pipecat.services.sarvam.stt import SarvamSTTSettings
from pipecat.services.settings import LLMSettings
from pipecat.services.smallest.stt import SmallestSTTSettings
from pipecat.services.speaches.llm import SpeachesLLMSettings
from pipecat.services.speaches.stt import SpeachesSTTSettings
from pipecat.services.speechmatics.stt import SpeechmaticsSTTSettings

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
    ctor_choices: dict[str, frozenset[str]] = dataclasses.field(default_factory=dict)
    """Closed value sets for the ctor kwargs that take an enum, keyed by name.

    ``ctor_allowed`` only says a kwarg may be sent; a kwarg whose value the
    branch feeds straight to an enum constructor (``CommitStrategy(...)``,
    service_factory.py:610) needs the value checked too, or the 422 arrives as
    a ValueError at run creation instead.
    """
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
    ctor_choices={"commit_strategy": frozenset({"vad", "manual"})},
)
# ink-whisper and ink-2 speak different wire protocols behind one provider, but
# their Settings classes declare the same fields (cartesia/stt.py:84-99,
# cartesia/turns/stt.py:41-56), so one allow-list covers both; the factory
# builds each branch's own class.
SPECS[("stt", "cartesia")] = TuningSpec(
    "stt",
    "cartesia",
    CartesiaSTTSettings,
    _fields(CartesiaSTTSettings),
    service_classes=(CartesiaSTTService, CartesiaTurnsSTTService),
)
SPECS[("stt", "smallest")] = TuningSpec(
    "stt", "smallest", SmallestSTTSettings, _fields(SmallestSTTSettings)
)
# The Responses-style realtime transcription session is a different service
# class this factory does not build; ``options.api`` is declared so the knob
# has one name from the start (validated in _validate_openai_stt_options).
SPECS[("stt", "openai")] = TuningSpec(
    "stt",
    "openai",
    OpenAISTTSettings,
    _fields(OpenAISTTSettings),
    options_allowed=frozenset({"api"}),
)
SPECS[("stt", "speaches")] = TuningSpec(
    "stt", "speaches", SpeachesSTTSettings, _fields(SpeachesSTTSettings)
)
SPECS[("stt", "sarvam")] = TuningSpec(
    "stt", "sarvam", SarvamSTTSettings, _fields(SarvamSTTSettings)
)
# ``language_code``/``language_codes`` are the registry's language under
# another name (the factory fills ``language`` from it, service_factory.py:519).
SPECS[("stt", "assemblyai")] = TuningSpec(
    "stt",
    "assemblyai",
    AssemblyAISTTSettings,
    _fields(AssemblyAISTTSettings, "language_code", "language_codes"),
    service_classes=(AssemblyAISTTService,),
    ctor_allowed=frozenset({"vad_force_turn_endpoint"}),
)
# ``operating_point`` is excluded for the same reason: for Speechmatics it *is*
# the model — the factory derives it from ``user_config.stt.model`` and the
# service writes it back into ``settings.model`` (speechmatics/stt.py:512).
# ``extra_params`` is excluded as a second ``extra``: _build_config splats it
# onto the SDK config by hasattr (speechmatics/stt.py:795-799), which would
# re-open every name this row excludes, operating_point included.
SPECS[("stt", "speechmatics")] = TuningSpec(
    "stt",
    "speechmatics",
    SpeechmaticsSTTSettings,
    _fields(SpeechmaticsSTTSettings, "operating_point", "extra_params"),
)
SPECS[("stt", "gladia")] = TuningSpec(
    "stt",
    "gladia",
    GladiaSTTSettings,
    _fields(GladiaSTTSettings, "language_config"),
)
SPECS[("stt", "google")] = TuningSpec(
    "stt",
    "google",
    GoogleSTTSettings,
    _fields(GoogleSTTSettings, "languages", "language_codes"),
)
SPECS[("stt", "azure_speech")] = TuningSpec(
    "stt", "azure_speech", AzureSTTSettings, _fields(AzureSTTSettings)
)
SPECS[("stt", "huggingface")] = TuningSpec(
    "stt", "huggingface", HuggingFaceSTTSettings, _fields(HuggingFaceSTTSettings)
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
# Deprecated on LLMSettings (1.7.0) and owned by the turn strategies, not by
# the model configuration.
_TURN_COMPLETION = ("filter_incomplete_user_turns", "user_turn_completion_config")
# ``top_k`` on top of those: no LLM branch this factory builds sends it except
# Google (google/llm.py:387). The OpenAI client library drops it
# (openai/base_llm.py:349-361, inherited by every OpenAI-compatible provider
# below) and AWS Bedrock only ever stores None (aws/llm.py:180), so exposing
# it there would be a knob that silently does nothing — the one failure this
# table exists to prevent.
_LLM_EXCLUDED = _TURN_COMPLETION + ("top_k",)


def _llm_row(provider: str, settings_cls: type, **kwargs: Any) -> TuningSpec:
    return TuningSpec(
        "llm", provider, settings_cls, _fields(settings_cls, *_LLM_EXCLUDED), **kwargs
    )


SPECS[("llm", "openai")] = _llm_row(
    "openai",
    OpenAILLMSettings,
    service_classes=(OpenAILLMService,),
    options_allowed=frozenset({"api", "reasoning", "verbosity"}),
)
# AtlasCloud is served by the OpenAI branch of the factory (service_factory
# .py:1079-1101) and so shares its Settings class; the gpt-5 extras are an
# OpenAI-model rule, so it gets no ``options``.
SPECS[("llm", "atlascloud")] = _llm_row("atlascloud", OpenAILLMSettings)
SPECS[("llm", "google")] = TuningSpec(
    "llm",
    "google",
    GoogleLLMSettings,
    _fields(GoogleLLMSettings, *_TURN_COMPLETION),
)
SPECS[("llm", "google_vertex")] = TuningSpec(
    "llm",
    "google_vertex",
    GoogleVertexLLMSettings,
    _fields(GoogleVertexLLMSettings, *_TURN_COMPLETION),
)
SPECS[("llm", "groq")] = _llm_row("groq", GroqLLMSettings)
SPECS[("llm", "openrouter")] = _llm_row("openrouter", OpenRouterLLMSettings)
SPECS[("llm", "azure")] = _llm_row("azure", AzureLLMSettings)
SPECS[("llm", "huggingface")] = _llm_row("huggingface", HuggingFaceLLMSettings)
SPECS[("llm", "speaches")] = _llm_row("speaches", SpeachesLLMSettings)
SPECS[("llm", "minimax")] = _llm_row("minimax", MiniMaxLLMSettings)
SPECS[("llm", "sarvam")] = _llm_row("sarvam", SarvamLLMSettings)
# DograhLLMService declares no Settings class of its own: it inherits
# OpenAILLMService's (dograh/llm.py:47,62).
SPECS[("llm", "dograh")] = _llm_row("dograh", OpenAILLMSettings)
SPECS[("llm", "aws_bedrock")] = _llm_row("aws_bedrock", AWSBedrockLLMSettings)


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
    checkable type (scalar, Literal or Enum) that rejects the value. None: this
    member neither confirms nor denies — it's a sentinel/None placeholder
    (skip) or a type this module doesn't verify (pydantic model,
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
    if isinstance(member, type) and issubclass(member, enum.Enum):
        # The factory feeds these straight to the enum constructor
        # (service_factory.py, speechmatics turn_detection_mode), so an
        # out-of-set value has to be a 422 here, not a ValueError at run
        # creation. Membership over a tuple compares by equality, so an
        # unhashable JSON list/object is a plain False, not a TypeError.
        return value in tuple(m.value for m in member)
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


RESPONSES_API_AVAILABLE = False
"""Whether this build can serve the OpenAI Responses API.

The Responses service needs the node-transition deferral ported to it before
it can replace chat completions; until that lands, both knobs that only exist
there (``options.api="responses"`` and ``options.reasoning.summary``) are named
422s rather than a silent downgrade or a knob that reaches nothing.
"""

PROVIDER_TURN_DETECTION_AVAILABLE = False
"""Whether this build can let an STT provider own turn detection.

The services below hand turns over by broadcasting their own
``UserStarted/StoppedSpeakingFrame`` and asking the aggregator for
``ExternalUserTurnStrategies``. That request is a no-op here:
``run_pipeline.py:968-979`` always passes its own ``user_turn_strategies``,
which wins (llm_response_universal.py:966-974), so the pipeline would keep
running local VAD while the provider endpoints server-side. Turn handling
belongs to the turn PR (spec §6); until then the *values* that flip it over
are a named 422, while the fields stay declared so the knobs keep one name.
"""

_TURN_HANDOVER = "hands turn detection to the provider"
_PROVIDER_TURN_KNOBS: dict[tuple[str, str, str, str], Callable[[Any], bool]] = {
    # Any mode but EXTERNAL (speechmatics/stt.py:550-554) makes the service
    # broadcast the turn frames at :915/:936.
    ("stt", "speechmatics", "settings", "turn_detection_mode"): (
        lambda v: v != "external"
    ),
    # gladia/stt.py:365-367; the broadcasts at :620-640 are guarded by it.
    ("stt", "gladia", "settings", "enable_vad"): lambda v: v is True,
    # sarvam/stt.py:815-826 broadcasts on the server's VAD events, :450 stops
    # honouring pipecat's VAD frames and :641 drops the flush signal. Sarvam
    # never even recommends external strategies, so nothing would notice.
    ("stt", "sarvam", "settings", "vad_signals"): lambda v: v is True,
    # assemblyai/stt.py:664-666; :1128-1133 and :1194-1196 emit turn frames
    # only in AssemblyAI's own turn-detection mode.
    ("stt", "assemblyai", "ctor", "vad_force_turn_endpoint"): lambda v: v is False,
}


def _provider_turn_error(
    kind: str, provider: str, section: str, name: str, value: Any
) -> str | None:
    if PROVIDER_TURN_DETECTION_AVAILABLE:
        return None
    hands_over = _PROVIDER_TURN_KNOBS.get((kind, provider, section, name))
    if hands_over is None or not hands_over(value):
        return None
    return (
        f"{kind}.{provider}.{section}.{name}: {value} {_TURN_HANDOVER}, "
        "not wired in this build (turn PR)"
    )


OPENAI_REALTIME_STT_AVAILABLE = False
"""Whether this build can serve OpenAI's realtime transcription session.

``create_stt_service`` builds the segmented ``OpenAISTTService`` for every
OpenAI model; the realtime session is a different service class that is not
wired yet, so asking for it is a named 422 rather than a silent downgrade to
segments.
"""

_OPENAI_STT_APIS = frozenset({"segments", "realtime"})
_OPENAI_APIS = frozenset({"chat_completions", "responses"})
_REASONING_KEYS = {
    "effort": frozenset({"none", "minimal", "low", "medium", "high", "xhigh"}),
    "summary": frozenset({"auto", "concise", "detailed"}),
}
_VERBOSITIES = frozenset({"low", "medium", "high"})


def _one_of(path: str, value: Any, allowed: frozenset[str]) -> str | None:
    """Closed-set check that survives an unhashable value.

    ``options`` is ``dict[str, Any]``, so a JSON list or object reaches here;
    a bare ``value not in allowed`` would raise TypeError, and pydantic only
    converts ValueError/AssertionError — the PUT would 500 instead of telling
    the caller which knob is wrong. ``None`` fails the same way as any other
    non-choice: null is not one of the choices.
    """
    if isinstance(value, str) and value in allowed:
        return None
    return f"{path}: must be one of {', '.join(sorted(allowed))}"


def _validate_openai_llm_options(options: dict[str, Any]) -> list[str]:
    """Value checks for ``llm.openai.options`` (spec §5.4).

    The generic loop below only checks that an option *name* is declared;
    these three carry closed value sets, and ``reasoning`` is a nested object,
    so a bad value would otherwise reach the provider as a 400 at call time.
    """
    errors: list[str] = []
    if "api" in options:
        error = _one_of("llm.openai.options.api", options["api"], _OPENAI_APIS)
        if error:
            errors.append(error)
        elif options["api"] == "responses" and not RESPONSES_API_AVAILABLE:
            errors.append(
                "llm.openai.options.api: responses is not available in this build "
                "(node-transition deferral not ported)"
            )
    if "reasoning" in options:
        reasoning = options["reasoning"]
        if not isinstance(reasoning, dict):
            errors.append("llm.openai.options.reasoning: wrong type")
        else:
            for name, value in reasoning.items():
                path = f"llm.openai.options.reasoning.{name}"
                if name not in _REASONING_KEYS:
                    errors.append(f"{path}: unknown key")
                elif name == "summary" and not RESPONSES_API_AVAILABLE:
                    # A chat-completions payload has no reasoning summary; only
                    # the Responses API does. Accepting it would be a knob that
                    # reaches nothing (§1.2).
                    errors.append(
                        f"{path}: not available in this build "
                        "(Responses API not ported)"
                    )
                else:
                    error = _one_of(path, value, _REASONING_KEYS[name])
                    if error:
                        errors.append(error)
    if "verbosity" in options:
        error = _one_of(
            "llm.openai.options.verbosity", options["verbosity"], _VERBOSITIES
        )
        if error:
            errors.append(error)
    return errors


def _validate_openai_stt_options(options: dict[str, Any]) -> list[str]:
    """Value check for ``stt.openai.options.api`` (reserved for the realtime
    transcription session, which this build does not construct)."""
    if "api" not in options:
        return []
    error = _one_of("stt.openai.options.api", options["api"], _OPENAI_STT_APIS)
    if error:
        return [error]
    if options["api"] == "realtime" and not OPENAI_REALTIME_STT_AVAILABLE:
        return ["stt.openai.options.api: realtime is not wired in this build"]
    return []


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
                    else:
                        error = _provider_turn_error(
                            kind, provider, "settings", name, value
                        )
                        if error:
                            errors.append(error)
                elif name in REGISTRY_OWNED:
                    errors.append(f"{path}: owned by the model configuration")
                else:
                    errors.append(f"{path}: unknown setting")
            for name, value in (tuning.get("ctor") or {}).items():
                path = f"{kind}.{provider}.ctor.{name}"
                if name not in spec.ctor_allowed:
                    errors.append(f"{path}: not allowed")
                    continue
                if name in spec.ctor_choices:
                    error = _one_of(path, value, spec.ctor_choices[name])
                    if error:
                        errors.append(error)
                        continue
                error = _provider_turn_error(kind, provider, "ctor", name, value)
                if error:
                    errors.append(error)
            options = tuning.get("options") or {}
            for name in options:
                if name not in spec.options_allowed:
                    errors.append(f"{kind}.{provider}.options.{name}: not allowed")
            if kind == "llm" and provider == "openai":
                errors.extend(_validate_openai_llm_options(options))
            if kind == "stt" and provider == "openai":
                errors.extend(_validate_openai_stt_options(options))
    return errors
