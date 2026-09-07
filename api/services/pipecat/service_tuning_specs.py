"""Allow-lists that make ``service_tuning`` explicit per service.

pipecat's ``Settings.from_mapping`` sends unknown keys to ``extra`` silently
and most services never read ``extra`` (pass2/service-factory.md B1). This
table is what the PUT validates against so a misspelt knob is a 422, not a
no-op. ``settings_allowed`` must be a subset of the real dataclass fields
(guarded by test_service_tuning_schema.py); identity fields owned by the
model registry (model/voice/language/api_key) are never tunable here.

Five build gates below name a knob that this build cannot honour rather than
letting it through as a silent no-op or a broken run (§1.2):
``RESPONSES_API_AVAILABLE``, ``OPENAI_REALTIME_STT_AVAILABLE``,
``PROVIDER_TURN_DETECTION_AVAILABLE``, ``TTS_SILENCE_AFTER_STOP_AVAILABLE`` and
``FILLER_ROLE_AVAILABLE``. Each is a constant the PR that wires the feature
flips.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from google.genai.types import ProactivityConfig, SafetySetting, ThinkingConfig
from pydantic import BaseModel, TypeAdapter, ValidationError

from pipecat.services.assemblyai.stt import AssemblyAISTTService, AssemblyAISTTSettings
from pipecat.services.aws.llm import AWSBedrockLLMSettings
from pipecat.services.azure.llm import AzureLLMSettings
from pipecat.services.azure.stt import AzureSTTSettings
from pipecat.services.azure.tts import AzureTTSSettings
from pipecat.services.camb.tts import CambTTSService, CambTTSSettings
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaSTTSettings
from pipecat.services.cartesia.tts import (
    CartesiaTTSService,
    CartesiaTTSSettings,
    GenerationConfig,
)
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.services.deepgram.flux.base import DeepgramFluxSTTSettings
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.deepgram.tts import DeepgramTTSSettings
from pipecat.services.dograh.flux.stt import DograhFluxSTTService
from pipecat.services.dograh.tts import DograhTTSSettings
from pipecat.services.elevenlabs.stt import (
    ElevenLabsRealtimeSTTService,
    ElevenLabsRealtimeSTTSettings,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
from pipecat.services.gladia.config import (
    MessagesConfig,
    PreProcessingConfig,
    RealtimeProcessingConfig,
)
from pipecat.services.gladia.stt import GladiaSTTSettings
from pipecat.services.google.gemini_live.llm import ContextWindowCompressionParams
from pipecat.services.google.llm import GoogleLLMService, GoogleLLMSettings
from pipecat.services.google.stt import GoogleSTTSettings
from pipecat.services.google.tts import GoogleTTSSettings
from pipecat.services.google.vertex.llm import GoogleVertexLLMSettings
from pipecat.services.groq.llm import GroqLLMSettings
from pipecat.services.huggingface.llm import HuggingFaceLLMSettings
from pipecat.services.huggingface.stt import HuggingFaceSTTSettings
from pipecat.services.inworld.tts import InworldTTSSettings
from pipecat.services.lmnt.tts import LmntTTSSettings
from pipecat.services.minimax.llm import MiniMaxLLMSettings
from pipecat.services.minimax.tts import MiniMaxTTSSettings
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import (
    OpenAIRealtimeSTTService,
    OpenAIRealtimeSTTSettings,
    OpenAISTTService,
    OpenAISTTSettings,
)
from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings
from pipecat.services.openrouter.llm import OpenRouterLLMSettings
from pipecat.services.rime.tts import RimeTTSSettings
from pipecat.services.sarvam.llm import SarvamLLMSettings
from pipecat.services.sarvam.stt import SarvamSTTSettings
from pipecat.services.sarvam.tts import SarvamTTSSettings
from pipecat.services.settings import LLMSettings
from pipecat.services.smallest.stt import SmallestSTTSettings
from pipecat.services.smallest.tts import SmallestTTSSettings
from pipecat.services.speaches.llm import SpeachesLLMSettings
from pipecat.services.speaches.stt import SpeachesSTTSettings
from pipecat.services.speaches.tts import SpeachesTTSSettings
from pipecat.services.speechmatics.stt import (
    AdditionalVocabEntry,
    SpeakerIdentifier,
    SpeechmaticsSTTSettings,
)
from pipecat.services.xai.tts import XAIWebsocketTTSSettings

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
    ctor_types: dict[str, type | tuple[type, ...]] = dataclasses.field(
        default_factory=dict
    )
    """Accepted value type per ctor kwarg, for the kwargs that carry a scalar.

    A ctor value is forwarded to the constructor verbatim (``**plan.ctor``),
    unlike a setting it is not checked against a dataclass field, so without
    this any JSON value reaches the service: ``vad_force_turn_endpoint: null``
    is falsy, which is how AssemblyAI reads it (assemblyai/stt.py:665), so it
    hands turns over exactly as ``false`` does while dodging a check written
    for ``false``. ``None`` is not a member of any entry: in a sparse document
    "leave it at the constructor default" is written by omitting the key.
    Guarded by test_scalar_ctor_kwargs_declare_their_type.
    """
    settings_types: dict[str, type | tuple[type, ...]] = dataclasses.field(
        default_factory=dict
    )
    """Accepted value type per setting, for rows with no Settings dataclass.

    ``_type_ok`` checks a setting against the field that declares it; the
    ``realtime`` rows have no such field (the branch maps the knob onto a
    provider pydantic object instead of a pipecat ``Settings``), so without
    this any JSON value would reach the session. Same rule as ``ctor_types``,
    checked with the same helper. A knob whose set is closed also carries a
    ``settings_choices`` entry, which is checked first; the type here is what
    remains true if that set ever goes away. Completeness for the realtime
    rows is guarded by test_realtime.py.
    """
    settings_choices: dict[str, frozenset[str]] = dataclasses.field(
        default_factory=dict
    )
    """Closed value sets for the settings that take an enum, keyed by name.

    The realtime counterpart of ``ctor_choices``: the branch feeds these to a
    provider model whose ``Literal`` pipecat widens with ``| str`` for forward
    compatibility (openai/realtime/events.py:182), so a typo would be a
    provider-side 400 mid-call rather than a 422 at the PUT.
    """
    settings_models: dict[str, Any] = dataclasses.field(default_factory=dict)
    """The provider object a setting is built into, for settings whose value
    is an object: a pydantic model, a dataclass, or a ``list[...]`` of one.

    ``from_mapping`` stores the JSON object unchanged and the branch converts
    it at run creation (``model_validate``, ``SpeakerIdentifier(**e)``).
    Those models ignore unknown keys (``model_config`` is not ``forbid`` on
    most of them), so a misspelt key inside the object was silently dropped,
    and a wrong scalar was a ValidationError at run creation rather than a
    422 at the PUT (#11, #12). ``_model_error`` walks the object against the
    model's fields for unknown keys and then validates it with the model
    itself; ``settings_types`` still names the JSON shape (dict/list), which
    is checked first. Completeness is guarded by test_nested_settings.py.
    """
    options_allowed: frozenset[str] = frozenset()
    nullable_extra: frozenset[str] = frozenset()
    non_nullable: frozenset[str] = frozenset()
    """Settings whose field is nullable upstream but which this build requires.

    Nullability is read off the dataclass, and a field can be ``| None`` for a
    reason the branch does not share: ``OpenAITTSSettings.voice`` is optional
    because pipecat lets the constructor supply it, while this factory always
    resolves it from the model configuration, so an explicit ``null`` would
    reach the request as ``voice=None`` and fail the call mid-turn. Subtracted
    from ``nullable()``.
    """

    def nullable(self) -> frozenset[str]:
        if self.settings_cls is None:
            return self.nullable_extra - self.non_nullable
        return (
            nullable_fields(self.settings_cls) | self.nullable_extra
        ) - self.non_nullable

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
    # A misspelt exclude would subtract nothing and leave the knob it meant
    # to close wide open, with every "unknown setting" test still green.
    unknown = set(exclude) - names
    if unknown:
        raise ValueError(f"{settings_cls.__name__}: not fields: {sorted(unknown)}")
    return frozenset(names - REGISTRY_OWNED - set(exclude))


FLUX_SETTINGS = _fields(
    DeepgramFluxSTTSettings
)  # eager_eot_threshold, eot_threshold, eot_timeout_ms, keyterm, min_confidence, numerals, language_hints
NOVA_SETTINGS = _fields(DeepgramSTTSettings)

SPECS: dict[tuple[str, str], TuningSpec] = {}
# Deepgram is one provider with two wire protocols; the allow-list is the
# union and the applier picks the right Settings class by model (Task 5).
SPECS[("stt", "deepgram")] = TuningSpec(
    "stt",
    "deepgram",
    None,
    FLUX_SETTINGS | NOVA_SETTINGS,
    service_classes=(DeepgramFluxSTTService, DeepgramSTTService),
    ctor_allowed=frozenset({"url", "mip_opt_out", "tag"}),
    ctor_types={"url": str, "mip_opt_out": bool, "tag": list},
    # Nova declares ``endpointing`` as ``Any`` (deepgram/stt.py:213), so
    # nothing about the field constrains the value and ``endpointing: true``
    # would reach the connection as a nonsense query parameter. Deepgram's
    # other documented value, ``false`` ("disable endpointing"), is not
    # exposed: it removes Nova's end-of-speech signal, which is who-owns-the-
    # turn territory (spec §6, same posture as _PROVIDER_TURN_KNOBS), and it
    # is indistinguishable at this seam from the ``true`` this check exists to
    # reject. Milliseconds only. The biasing/redaction knobs are ``Any`` on
    # Nova too (deepgram/stt.py:215-222) while the wire takes a string or a
    # list of strings (query params, :572-574); ``redact`` also takes ``true``.
    settings_types={
        "endpointing": int,
        "keyterm": (str, list),
        "keywords": (str, list),
        "replace": (str, list),
        "search": (str, list),
        "redact": (bool, str, list),
    },
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
    ctor_types={"mip_opt_out": bool, "tag": list},
)
SPECS[("stt", "elevenlabs")] = TuningSpec(
    "stt",
    "elevenlabs",
    ElevenLabsRealtimeSTTSettings,
    _fields(ElevenLabsRealtimeSTTSettings),
    service_classes=(ElevenLabsRealtimeSTTService,),
    ctor_allowed=frozenset({"commit_strategy", "include_timestamps"}),
    ctor_choices={"commit_strategy": frozenset({"vad", "manual"})},
    ctor_types={"include_timestamps": bool},
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
# OpenAI is one provider with two transcription services: the segmented Audio
# API one and the realtime transcription session, selected by ``options.api``.
# Like Deepgram above, the allow-list is the union and ``settings_cls=None``
# lets each field be type-checked against whichever Settings class declares
# it; _validate_openai_stt then pairs a field with the api that has it, so a
# knob the selected service never reads is a 422 and not a silent ``extra``.
SPECS[("stt", "openai")] = TuningSpec(
    "stt",
    "openai",
    None,
    _fields(OpenAISTTSettings) | _fields(OpenAIRealtimeSTTSettings),
    service_classes=(OpenAISTTService, OpenAIRealtimeSTTService),
    options_allowed=frozenset({"api"}),
    nullable_extra=nullable_fields(OpenAISTTSettings)
    | nullable_fields(OpenAIRealtimeSTTSettings),
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
    ctor_types={"vad_force_turn_endpoint": bool},
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
    # Both lists are handed to the SDK config, which does not validate on
    # assignment (speechmatics/stt.py:772-775); ``SpeakerIdentifier`` is a
    # dataclass, so an unknown key was a TypeError at run creation (#12).
    settings_models={
        "additional_vocab": list[AdditionalVocabEntry],
        "known_speakers": list[SpeakerIdentifier],
    },
)
SPECS[("stt", "gladia")] = TuningSpec(
    "stt",
    "gladia",
    GladiaSTTSettings,
    _fields(GladiaSTTSettings, "language_config"),
    # Three pydantic models the branch builds with ``model_validate``
    # (service_factory.py, gladia branch): the field type is a model, which
    # ``_type_ok`` cannot check, so without these a scalar would pass the PUT
    # and raise a pydantic ValidationError at run creation. JSON object only,
    # and the object's keys and scalars are checked against the model.
    settings_types={
        "pre_processing": dict,
        "realtime_processing": dict,
        "messages_config": dict,
    },
    settings_models={
        "pre_processing": PreProcessingConfig,
        "realtime_processing": RealtimeProcessingConfig,
        "messages_config": MessagesConfig,
    },
)
# ``use_separate_recognition_per_channel`` is excluded: ``_connect`` hard-codes
# ``audio_channel_count=1`` (google/stt.py:846) and reads every other field of
# the row into the recognition features (:851-861) but never this one.
SPECS[("stt", "google")] = TuningSpec(
    "stt",
    "google",
    GoogleSTTSettings,
    _fields(
        GoogleSTTSettings,
        "languages",
        "language_codes",
        "use_separate_recognition_per_channel",
    ),
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
    ctor_types={
        "auto_mode": bool,
        "enable_ssml_parsing": bool,
        "reconnect_on_error": bool,
    },
)
SPECS[("tts", "openai")] = TuningSpec(
    "tts",
    "openai",
    OpenAITTSSettings,
    _fields(OpenAITTSSettings) | frozenset({"voice"}),
    service_classes=(OpenAITTSService,),
    # ``voice`` is the one REGISTRY_OWNED name this table re-admits (B6). The
    # field is ``str | None`` upstream, but the service does not treat None as
    # "unset": every turn yields an ErrorFrame instead of audio
    # (openai/tts.py:254-256). A 422 at the PUT, not a mute call.
    non_nullable=frozenset({"voice"}),
)
SPECS[("tts", "cartesia")] = TuningSpec(
    "tts",
    "cartesia",
    CartesiaTTSSettings,
    _fields(CartesiaTTSSettings),
    service_classes=(CartesiaTTSService,),
    # Sent verbatim as the ``max_buffer_delay_ms`` field of every synthesis
    # message (cartesia/tts.py:509-510).
    ctor_allowed=frozenset({"max_buffer_delay_ms"}),
    ctor_types={"max_buffer_delay_ms": int},
    # ``GenerationConfig`` is a pydantic model the branch builds with
    # ``model_validate``; same hole as gladia's three above.
    settings_types={"generation_config": dict},
    settings_models={"generation_config": GenerationConfig},
)
SPECS[("tts", "inworld")] = TuningSpec(
    "tts", "inworld", InworldTTSSettings, _fields(InworldTTSSettings)
)
# ``pitch`` and ``volume`` are excluded on purpose: the service copies only
# ``speed`` into the payload it sends (dograh/tts.py:143-150), so exposing them
# would be a knob that silently does nothing. A 422 says so instead (B9).
SPECS[("tts", "dograh")] = TuningSpec(
    "tts", "dograh", DograhTTSSettings, _fields(DograhTTSSettings, "pitch", "volume")
)
# ``inlineSpeedAlpha`` is excluded for the same reason: the websocket service
# never reads it off the settings — neither ``_build_ws_params``
# (rime/tts.py:324-369) nor ``_build_msg`` (:414-420), which sends only what
# the ``INLINE_SPEED`` text helper put in ``_extra_msg_fields`` (:391-397).
# The rest are model-conditional, not dropped, so they stay in.
SPECS[("tts", "rime")] = TuningSpec(
    "tts", "rime", RimeTTSSettings, _fields(RimeTTSSettings, "inlineSpeedAlpha")
)
SPECS[("tts", "sarvam")] = TuningSpec(
    "tts", "sarvam", SarvamTTSSettings, _fields(SarvamTTSSettings)
)
SPECS[("tts", "minimax")] = TuningSpec(
    "tts", "minimax", MiniMaxTTSSettings, _fields(MiniMaxTTSSettings)
)
SPECS[("tts", "azure_speech")] = TuningSpec(
    "tts", "azure_speech", AzureTTSSettings, _fields(AzureTTSSettings)
)
SPECS[("tts", "xai")] = TuningSpec(
    "tts", "xai", XAIWebsocketTTSSettings, _fields(XAIWebsocketTTSSettings)
)
SPECS[("tts", "smallest")] = TuningSpec(
    "tts", "smallest", SmallestTTSSettings, _fields(SmallestTTSSettings)
)
SPECS[("tts", "google")] = TuningSpec(
    "tts", "google", GoogleTTSSettings, _fields(GoogleTTSSettings)
)
# Deepgram and LMNT declare no TTS settings of their own beyond the identity
# fields the registry owns, so both allow-lists are empty today; the rows exist
# so the providers are known (a misspelt knob is "unknown setting", not
# "unknown provider") and so a field added upstream is honoured by the branch,
# which already routes through ``build_settings``.
SPECS[("tts", "deepgram")] = TuningSpec(
    "tts", "deepgram", DeepgramTTSSettings, _fields(DeepgramTTSSettings)
)
SPECS[("tts", "lmnt")] = TuningSpec(
    "tts", "lmnt", LmntTTSSettings, _fields(LmntTTSSettings)
)
# ``user_instructions`` is the one field Camb declares beyond the identity
# ones (camb/tts.py:142-153); the request timeout is a constructor argument.
SPECS[("tts", "camb")] = TuningSpec(
    "tts",
    "camb",
    CambTTSSettings,
    _fields(CambTTSSettings),
    service_classes=(CambTTSService,),
    ctor_allowed=frozenset({"timeout"}),
    ctor_types={"timeout": float},
)
# Speaches speaks OpenAI's audio API and inherits its Settings unchanged
# (speaches/tts.py:14-18); ``run_tts`` reads ``instructions`` and ``speed``
# off them (:62-68).
SPECS[("tts", "speaches")] = TuningSpec(
    "tts", "speaches", SpeachesTTSSettings, _fields(SpeachesTTSSettings)
)
SPECS[("tts", ALL)] = TuningSpec(
    "tts",
    ALL,
    None,
    frozenset(),
    ctor_allowed=frozenset({"silence_time_s"}),
    ctor_types={"silence_time_s": (int, float)},
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
# Declared on every OpenAI-shaped Settings class but read by only one wire:
# openai/base_llm.py:353-355 sends them, so the OpenAI-compatible rows keep
# them; Google's ``_build_generation_params`` (google/llm.py:385-397, inherited
# by Vertex) and Bedrock's ``_build_inference_config`` (aws/llm.py:265-271)
# never look at them.
_PENALTIES_AND_SEED = ("frequency_penalty", "presence_penalty", "seed")


def _llm_row(
    provider: str, settings_cls: type, *exclude: str, **kwargs: Any
) -> TuningSpec:
    return TuningSpec(
        "llm",
        provider,
        settings_cls,
        _fields(settings_cls, *_LLM_EXCLUDED, *exclude),
        **kwargs,
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
# ``thinking`` is declared as ``ThinkingConfig | None | _NotGiven``
# (google/llm.py:123) — a pydantic model, which this module cannot check off
# the field, so a scalar would pass the PUT and blow up in ``from_mapping``'s
# own coercion (``ThinkingConfig(**value)``, google/llm.py:140-141). It arrives
# as a JSON object or not at all.
_GOOGLE_LLM_TYPES: dict[str, type | tuple[type, ...]] = {"thinking": dict}
# ``safety_settings`` entries are converted the same way (``SafetySetting(
# **entry)``, google/llm.py:142-145); the genai models take camelCase aliases
# too, which the key walk honours.
_GOOGLE_LLM_MODELS: dict[str, Any] = {
    "thinking": GoogleLLMService.ThinkingConfig,
    "safety_settings": list[SafetySetting],
}
SPECS[("llm", "google")] = TuningSpec(
    "llm",
    "google",
    GoogleLLMSettings,
    _fields(GoogleLLMSettings, *_TURN_COMPLETION, *_PENALTIES_AND_SEED),
    settings_types=_GOOGLE_LLM_TYPES,
    settings_models=_GOOGLE_LLM_MODELS,
)
SPECS[("llm", "google_vertex")] = TuningSpec(
    "llm",
    "google_vertex",
    GoogleVertexLLMSettings,
    _fields(GoogleVertexLLMSettings, *_TURN_COMPLETION, *_PENALTIES_AND_SEED),
    settings_types=_GOOGLE_LLM_TYPES,
    settings_models=_GOOGLE_LLM_MODELS,
)
SPECS[("llm", "groq")] = _llm_row("groq", GroqLLMSettings)
SPECS[("llm", "openrouter")] = _llm_row("openrouter", OpenRouterLLMSettings)
SPECS[("llm", "azure")] = _llm_row("azure", AzureLLMSettings)
SPECS[("llm", "huggingface")] = _llm_row("huggingface", HuggingFaceLLMSettings)
SPECS[("llm", "speaches")] = _llm_row("speaches", SpeachesLLMSettings)
SPECS[("llm", "minimax")] = _llm_row("minimax", MiniMaxLLMSettings)
# Sarvam's payload builder pops ``max_completion_tokens`` (sarvam/llm.py:137).
SPECS[("llm", "sarvam")] = _llm_row(
    "sarvam", SarvamLLMSettings, "max_completion_tokens"
)
# DograhLLMService declares no Settings class of its own: it inherits
# OpenAILLMService's (dograh/llm.py:47,62).
SPECS[("llm", "dograh")] = _llm_row("dograh", OpenAILLMSettings)
SPECS[("llm", "aws_bedrock")] = _llm_row(
    "aws_bedrock", AWSBedrockLLMSettings, *_PENALTIES_AND_SEED
)

# ---------------------------------------------------------------------------
# realtime (speech-to-speech)
#
# These rows are the one part of the table with no pipecat ``Settings``
# dataclass behind them: a realtime branch builds the session as provider
# pydantic objects (``SessionProperties``, ``AudioInput``, ``Reasoning``,
# Ultravox ``OneShotInputParams``), so the allow-list is written out and the
# value shape is declared in ``settings_types``/``settings_choices`` instead of
# being read off a field. ``service_factory.REALTIME_FIELDS`` says where each
# knob lands, and test_realtime.py keeps the two in step so no knob is orphaned.
#
# No realtime row is nullable: every destination already defaults to "unset",
# so an explicit null would be a knob that does nothing — a default is written
# by omitting the key (same rule as ``ctor_types``).
#
# Turn detection is absent on purpose. ``turn_detection``, server VAD and
# ``semantic_vad`` decide who owns the user turn, which is the turn PR's
# subject (spec §6); leaving them out of the allow-list makes them "unknown
# setting" until it lands.
_NOISE_REDUCTIONS = frozenset({"near_field", "far_field"})
# openai/realtime/events.py:182 (Reasoning.effort). Narrower than the chat
# completions set above: the realtime enum has no "none".
_REALTIME_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
_TOOL_CHOICES = frozenset({"auto", "none", "required"})
# Azure Realtime speaks the OpenAI wire protocol and reuses its events module
# (azure/realtime/llm.py), so both rows expose the same session knobs.
_OPENAI_REALTIME_TYPES: dict[str, type | tuple[type, ...]] = {
    "noise_reduction": str,
    "speed": (int, float),
    "reasoning_effort": str,
    "max_output_tokens": int,
    "tool_choice": str,
}
_OPENAI_REALTIME_CHOICES = {
    "noise_reduction": _NOISE_REDUCTIONS,
    "reasoning_effort": _REALTIME_REASONING_EFFORTS,
    "tool_choice": _TOOL_CHOICES,
}
# ``thinking``/``proactivity``/``context_window_compression`` arrive as JSON
# objects and the branch converts them to their provider models; ``temperature``
# is the one knob the realtime configuration already carried.
_GEMINI_REALTIME_TYPES: dict[str, type | tuple[type, ...]] = {
    "thinking": dict,
    "enable_affective_dialog": bool,
    "proactivity": dict,
    "context_window_compression": dict,
    "temperature": (int, float),
}
# The models ``service_factory._GEMINI_REALTIME_COERCIONS`` builds them into.
_GEMINI_REALTIME_MODELS: dict[str, Any] = {
    "thinking": ThinkingConfig,
    "proactivity": ProactivityConfig,
    "context_window_compression": ContextWindowCompressionParams,
}
# ``extra`` is the one declared exception to "nothing goes through extra": it
# is ``OneShotInputParams.extra``, a field Ultravox merges into the
# call-creation request (ultravox/llm.py:361), not the ``Settings.extra``
# overflow that no service reads. ``max_duration`` is a pydantic timedelta
# that would also take an ISO-8601 string, but a string skips the numeric
# clamp (cascade.py:194) and would only fail at run creation; seconds only.
_ULTRAVOX_REALTIME_TYPES: dict[str, type | tuple[type, ...]] = {
    "temperature": (int, float),
    "max_duration": (int, float),
    "extra": dict,
}
# ``extra`` is merged over the call-creation body *last*
# (``request_body = request_body | params.extra``, ultravox/llm.py:361), so a
# key the service already put there wins over the run's own configuration —
# which is how a free-form ``extra`` would otherwise re-open every identity
# field this table owns, the transport's ``medium`` included. The body keys
# are built at ultravox/llm.py:342-358; ``firstSpeakerSettings`` is rewritten
# on every call by the Dograh wrapper from the greeting decision
# (realtime/ultravox_realtime.py:422-434), so setting it here is a plain
# no-op. Anything Ultravox accepts that is not in this table still passes.
_ULTRAVOX_EXTRA_OWNED = {
    "systemPrompt": "the workflow node",
    "model": "the model configuration",
    "voice": "the model configuration",
    "temperature": "the model configuration",
    "maxDuration": "the model configuration",
    "initialOutputMedium": "the model configuration",
    "metadata": "the service",
    "selectedTools": "the workflow tools",
    "medium": "the transport",
    "firstSpeakerSettings": "the greeting decision",
    # ``firstSpeaker`` is Ultravox's older spelling of the same decision. The
    # wrapper only strips and rewrites ``firstSpeakerSettings``
    # (realtime/ultravox_realtime.py:425-436), so this alias would survive the
    # merge and re-decide who opens the call behind the greeting logic.
    "firstSpeaker": "the greeting decision",
    # ``vadSettings`` tunes Ultravox's own endpointing, i.e. who ends the user
    # turn — the turn PR's subject (spec §6), and the same reason no turn knob
    # appears in the realtime allow-lists.
    "vadSettings": "turn handling",
}


def _realtime_row(
    provider: str,
    types: dict[str, type | tuple[type, ...]],
    choices: dict[str, frozenset[str]] | None = None,
    models: dict[str, Any] | None = None,
) -> TuningSpec:
    return TuningSpec(
        "realtime",
        provider,
        None,
        frozenset(types),
        settings_types=types,
        settings_choices=choices or {},
        settings_models=models or {},
    )


SPECS[("realtime", "openai_realtime")] = _realtime_row(
    "openai_realtime", _OPENAI_REALTIME_TYPES, _OPENAI_REALTIME_CHOICES
)
SPECS[("realtime", "azure_realtime")] = _realtime_row(
    "azure_realtime", _OPENAI_REALTIME_TYPES, _OPENAI_REALTIME_CHOICES
)
# Grok's session carries neither speed, reasoning nor an output token cap
# (xai/realtime/events.py:218-243); the transcription language hint is the
# only knob it has that the registry doesn't already own.
SPECS[("realtime", "grok_realtime")] = _realtime_row(
    "grok_realtime", {"language_hint": str}
)
SPECS[("realtime", "google_realtime")] = _realtime_row(
    "google_realtime", _GEMINI_REALTIME_TYPES, models=_GEMINI_REALTIME_MODELS
)
SPECS[("realtime", "google_vertex_realtime")] = _realtime_row(
    "google_vertex_realtime", _GEMINI_REALTIME_TYPES, models=_GEMINI_REALTIME_MODELS
)
SPECS[("realtime", "ultravox_realtime")] = _realtime_row(
    "ultravox_realtime", _ULTRAVOX_REALTIME_TYPES
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


def _scalar_verdict(expected: tuple[type, ...], value: Any) -> bool:
    """True/False for this member; never None.

    ``isinstance(True, int)`` is True, so a bool has to be rejected explicitly
    against a numeric member or ``temperature: true`` rides a float field all
    the way to the wire (the cascade clamp skips bools, cascade.py:196). The
    rejection is a *member* verdict, not a field verdict: ``_field_type_ok``
    keeps scanning, so a union that also declares ``bool`` — or another
    class's variant of the field in a merged row — still accepts it.
    """
    if isinstance(value, bool) and bool not in expected:
        return False
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
which wins (llm_response_universal.py:966-974). The provider's frames are
*not* discarded, though: the aggregator forwards them
(llm_response_universal.py:1567-1572) while the local strategies keep
emitting their own, so the pipeline would see two turn signals for one
utterance. Turn handling belongs to the turn PR (spec §6); until then the
*values* that flip it over are a named 422, and so are the *fields* that
only tune the provider's own turn detection (``_PROVIDER_TURN_FIELDS``),
while everything stays declared so the knobs keep one name.
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
# Whole fields, any value: they tune the provider's own turn detection, and
# the one mode the gate above lets through overwrites or ignores them. The
# PUT cannot see the model, so this set is model-blind by design until the
# turn PR decides per model at build time.
_PROVIDER_TURN_FIELDS: frozenset[tuple[str, str, str, str]] = frozenset(
    {
        # assemblyai/stt.py:601-645 (_configure_pipecat_turn_mode), under
        # vad_force_turn_endpoint=True: on u3-rt-pro max_turn_silence is
        # overwritten with min_turn_silence and the threshold is left to the
        # API (:628-641, min_turn_silence survives); on universal-streaming
        # the threshold and min_turn_silence are overwritten (:643-645,
        # max_turn_silence survives). No single model honours all three.
        ("stt", "assemblyai", "settings", "end_of_turn_confidence_threshold"),
        ("stt", "assemblyai", "settings", "min_turn_silence"),
        ("stt", "assemblyai", "settings", "max_turn_silence"),
        # turn_detection_mode=external loads a preset with
        # end_of_utterance_mode=EXTERNAL (speechmatics voice/_presets.py:
        # 165-176), under which the SDK's end-of-utterance timers
        # (voice/_client.py:1463,1553-1560) never run.
        ("stt", "speechmatics", "settings", "end_of_utterance_silence_trigger"),
        ("stt", "speechmatics", "settings", "end_of_utterance_max_delay"),
    }
)


def _provider_turn_error(
    kind: str, provider: str, section: str, name: str, value: Any
) -> str | None:
    if PROVIDER_TURN_DETECTION_AVAILABLE:
        return None
    path = f"{kind}.{provider}.{section}.{name}"
    if (kind, provider, section, name) in _PROVIDER_TURN_FIELDS:
        return (
            f"{path}: only read when the provider owns turn detection "
            "(PROVIDER_TURN_DETECTION_AVAILABLE), not wired in this build (turn PR)"
        )
    hands_over = _PROVIDER_TURN_KNOBS.get((kind, provider, section, name))
    if hands_over is None or not hands_over(value):
        return None
    return f"{path}: {value} {_TURN_HANDOVER}, not wired in this build (turn PR)"


FILLER_ROLE_AVAILABLE = False
"""Whether this build builds a filler LLM that ``scope.filler`` can reach.

The flag is declared on ``LLMScope`` so the document keeps one name, but no
factory call passes ``role="filler"`` yet: the role arrives with the tools PR
(spec §8.1). Until then ``scope.filler: true`` is a named 422, not a stored
placebo; ``false`` and absent are the same "no such role" and pass.
"""


def _scope_error(document: dict[str, Any]) -> list[str]:
    if FILLER_ROLE_AVAILABLE or not (document.get("scope") or {}).get("filler"):
        return []
    return [
        "scope.filler: no filler LLM role in this build (FILLER_ROLE_AVAILABLE), "
        "it lands with the tools PR"
    ]


TTS_SILENCE_AFTER_STOP_AVAILABLE = True
"""Whether this build pushes silence after a TTS turn ends.

``silence_time_s`` sizes that silence, and it is read in exactly one place,
under ``if self._push_silence_after_stop`` (tts_service.py:902-903), a
constructor parameter defaulting to False (:159). Wired: ``create_tts_service``
passes ``push_silence_after_stop=True`` to every TTS branch but Camb whenever
``tts._all.ctor.silence_time_s`` is set, and leaves it False otherwise. The
gate stays so the knob has one place to be turned off again.
"""

_SILENCE_AFTER_STOP_KNOB = ("tts", ALL, "ctor", "silence_time_s")


def _silence_after_stop_error(
    kind: str, provider: str, section: str, name: str
) -> str | None:
    if TTS_SILENCE_AFTER_STOP_AVAILABLE:
        return None
    if (kind, provider, section, name) != _SILENCE_AFTER_STOP_KNOB:
        return None
    return (
        f"{kind}.{provider}.{section}.{name}: not wired in this build "
        "(TTS services do not push silence after stop)"
    )


OPENAI_REALTIME_STT_AVAILABLE = True
"""Whether this build can serve OpenAI's realtime transcription session.

Wired: ``create_stt_service`` builds ``OpenAIRealtimeSTTService`` when
``options.api`` is ``"realtime"`` and the segmented ``OpenAISTTService``
otherwise. The gate stays so the knob has one place to be turned off again.
"""

_OPENAI_STT_APIS = frozenset({"segments", "realtime"})
_DEFAULT_OPENAI_STT_API = "segments"
_OPENAI_STT_API_FIELDS = {
    "segments": _fields(OpenAISTTSettings),
    "realtime": _fields(OpenAIRealtimeSTTSettings),
}
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


def _ctor_type_ok(expected: type | tuple[type, ...], value: Any) -> bool:
    """Whether a ctor value matches its declared type.

    Same rule as ``_scalar_verdict`` for settings — a bool never satisfies a
    non-bool numeric type, since ``isinstance(True, int)`` is True, and a
    declared ``float`` also takes an int (``_SCALAR_TYPES``), because JSON has
    a single number type and ``"timeout": 30`` must not be a false 422 — but
    definitive: there is no second union member to fall through to.
    """
    expected = expected if isinstance(expected, tuple) else (expected,)
    if float in expected:
        expected += (int,)
    if isinstance(value, bool) and bool not in expected:
        return False
    return isinstance(value, expected)


def _model_fields(model: Any) -> dict[str, Any] | None:
    """``{accepted key: annotation}`` for a pydantic model or a dataclass.

    A pydantic field is reachable by its alias as well (the genai models
    take camelCase), so both names are accepted.
    """
    if isinstance(model, type) and issubclass(model, BaseModel):
        fields: dict[str, Any] = {}
        for name, f in model.model_fields.items():
            fields[name] = f.annotation
            if f.alias:
                fields[f.alias] = f.annotation
        return fields
    if dataclasses.is_dataclass(model):
        return {f.name: f.type for f in dataclasses.fields(model)}
    return None


def _model_shape_errors(model: Any, value: Any, path: str) -> list[str]:
    """Unknown keys and booleans on numeric fields, anywhere inside ``value``.

    Neither is something the model's own validation reports: unknown keys
    are ignored unless ``extra="forbid"``, and lax mode coerces ``true`` to
    ``1.0`` on a float field — the same hole ``_scalar_verdict`` closes for
    top-level settings.
    """
    origin = typing.get_origin(model)
    if origin is list:
        if not isinstance(value, list):
            return []
        (item,) = typing.get_args(model)
        return [
            e
            for i, v in enumerate(value)
            for e in _model_shape_errors(item, v, f"{path}[{i}]")
        ]
    if _is_union(model):
        return [
            e
            for member in typing.get_args(model)
            for e in _model_shape_errors(member, value, path)
        ]
    fields = _model_fields(model)
    if fields is None:
        if isinstance(value, bool) and origin is None and model in (int, float):
            return [f"{path}: wrong type"]
        return []
    if not isinstance(value, dict):
        return []
    errors = []
    for key, sub in value.items():
        if key not in fields:
            errors.append(f"{path}.{key}: unknown key")
        else:
            errors.extend(_model_shape_errors(fields[key], sub, f"{path}.{key}"))
    return errors


@cache
def _adapter(model: Any) -> TypeAdapter:
    return TypeAdapter(model)


def _model_error(path: str, model: Any, value: Any) -> str | None:
    """Whether ``value`` builds the provider object the branch will build.

    Keys and booleans first (``_model_shape_errors``), then the model's own
    validation in the lax mode the branch uses, so a value the PUT accepts is
    one ``model_validate`` accepts at run creation, with the first failing
    location named.
    """
    errors = _model_shape_errors(model, value, path)
    if errors:
        return "; ".join(errors)
    try:
        _adapter(model).validate_python(value)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(step) for step in first["loc"])
        return f"{path}{'.' + where if where else ''}: {first['msg']}"
    return None


def _settings_value_error(
    spec: TuningSpec,
    nullable: frozenset[str],
    path: str,
    name: str,
    value: Any,
) -> str | None:
    """Whether an allow-listed setting's *value* is acceptable.

    Four checks in order of how definitive they are: null against the
    nullable set, then the row's own declaration (a closed set first, since a
    ``Literal`` widened with ``| str`` upstream is not checkable from the
    field), then the declaring dataclass, then — for an object-valued
    setting — the provider model it is built into.
    """
    if value is None:
        return None if name in nullable else f"{path}: null not allowed"
    if name in spec.settings_choices:
        return _one_of(path, value, spec.settings_choices[name])
    if name in spec.settings_types and not _ctor_type_ok(
        spec.settings_types[name], value
    ):
        return f"{path}: wrong type"
    if not _type_ok(spec.settings_classes(), name, value):
        return f"{path}: wrong type"
    if name in spec.settings_models:
        return _model_error(path, spec.settings_models[name], value)
    return None


def _validate_ultravox_extra(settings: dict[str, Any]) -> list[str]:
    """Keep ``realtime.ultravox_realtime.settings.extra`` out of the keys the
    call-creation body already owns (see ``_ULTRAVOX_EXTRA_OWNED``)."""
    extra = settings.get("extra")
    if not isinstance(extra, dict):
        return []  # a non-dict is already "wrong type" from the shape check
    return [
        f"realtime.ultravox_realtime.settings.extra.{key}: owned by {owner}"
        for key, owner in _ULTRAVOX_EXTRA_OWNED.items()
        if key in extra
    ]


def _validate_openai_stt(tuning: dict[str, Any]) -> list[str]:
    """Check ``stt.openai.options.api`` and pair each setting with it.

    The two OpenAI transcription services declare different fields, and
    ``from_mapping`` sends a field the selected one doesn't declare to
    ``extra``, which neither reads — so a knob written for the other api is a
    silent no-op. Name it instead (§1.2).
    """
    options = tuning.get("options") or {}
    api = options.get("api", _DEFAULT_OPENAI_STT_API)
    if "api" in options:
        error = _one_of("stt.openai.options.api", options["api"], _OPENAI_STT_APIS)
        if error:
            return [error]
        if api == "realtime" and not OPENAI_REALTIME_STT_AVAILABLE:
            return ["stt.openai.options.api: realtime is not wired in this build"]
    other = "realtime" if api == "segments" else "segments"
    return [
        f"stt.openai.settings.{name}: only available with options.api={other}"
        for name in tuning.get("settings") or {}
        if name in _OPENAI_STT_API_FIELDS[other]
        and name not in _OPENAI_STT_API_FIELDS[api]
    ]


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
                    error = _settings_value_error(spec, nullable, path, name, value)
                    if error is None:
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
                elif name in spec.ctor_types and not _ctor_type_ok(
                    spec.ctor_types[name], value
                ):
                    errors.append(f"{path}: wrong type")
                    continue
                error = _provider_turn_error(
                    kind, provider, "ctor", name, value
                ) or _silence_after_stop_error(kind, provider, "ctor", name)
                if error:
                    errors.append(error)
            options = tuning.get("options") or {}
            for name in options:
                if name not in spec.options_allowed:
                    errors.append(f"{kind}.{provider}.options.{name}: not allowed")
            if kind == "llm" and provider == "openai":
                errors.extend(_validate_openai_llm_options(options))
            if kind == "stt" and provider == "openai":
                errors.extend(_validate_openai_stt(tuning))
            if kind == "realtime" and provider == "ultravox_realtime":
                errors.extend(_validate_ultravox_extra(tuning.get("settings") or {}))
    errors.extend(_scope_error(document))
    return errors
