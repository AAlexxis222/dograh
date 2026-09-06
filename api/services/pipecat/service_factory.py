from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlparse, urlunparse

import aiohttp
from fastapi import HTTPException
from google.genai.types import ProactivityConfig, ThinkingConfig
from loguru import logger

from api.constants import MPS_API_URL
from api.errors.failure import (
    ErrorSource,
    annotate_failure_metadata,
    classify_exception,
    log_failure,
)
from api.services.configuration.options import (
    DEEPGRAM_FLUX_MODELS,
    DEEPGRAM_FLUX_MULTILINGUAL_LANGUAGE_OPTIONS,
)
from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.gemini_json_schema_adapter import (
    DograhGeminiJSONSchemaAdapter,
)
from api.services.pipecat.minimax_tts import MiniMaxOwnedSessionTTSService

# Module level, unlike the rest of the realtime module's contents, so the
# branches below name the service class the way every other branch in this
# file does — and so a test can patch one.
from api.services.pipecat.realtime.azure_realtime import DograhAzureRealtimeLLMService
from api.services.pipecat.realtime.gemini_live import DograhGeminiLiveLLMService
from api.services.pipecat.realtime.gemini_live_vertex import (
    DograhGeminiLiveVertexLLMService,
)
from api.services.pipecat.realtime.grok_realtime import DograhGrokRealtimeLLMService
from api.services.pipecat.realtime.openai_realtime import DograhOpenAIRealtimeLLMService
from api.services.pipecat.realtime.ultravox_realtime import (
    DograhUltravoxOneShotInputParams,
    DograhUltravoxRealtimeLLMService,
)
from api.services.pipecat.service_tuning import (
    ALL,
    LLMRole,
    build_settings,
    coerce_settings,
    llm_document_for_role,
    tuning_for,
)
from api.services.pipecat.service_tuning_specs import FLUX_SETTINGS
from api.utils.url_security import validate_user_configured_service_url
from pipecat.services.assemblyai.stt import AssemblyAISTTService, AssemblyAISTTSettings
from pipecat.services.aws.llm import AWSBedrockLLMService, AWSBedrockLLMSettings
from pipecat.services.azure.llm import AzureLLMService, AzureLLMSettings
from pipecat.services.azure.realtime.llm import AzureRealtimeLLMSettings
from pipecat.services.azure.stt import AzureSTTService, AzureSTTSettings
from pipecat.services.azure.tts import AzureTTSService, AzureTTSSettings
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaSTTSettings
from pipecat.services.cartesia.tts import (
    CartesiaTTSService,
    CartesiaTTSSettings,
    GenerationConfig,
)
from pipecat.services.cartesia.turns.stt import (
    CartesiaTurnsSTTService,
    CartesiaTurnsSTTSettings,
)
from pipecat.services.deepgram.flux.stt import (
    DeepgramFluxSTTService,
    DeepgramFluxSTTSettings,
)
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.deepgram.tts import DeepgramTTSService, DeepgramTTSSettings
from pipecat.services.dograh.flux.stt import DograhFluxSTTService
from pipecat.services.dograh.llm import DograhLLMService
from pipecat.services.dograh.stt import DograhSTTService, DograhSTTSettings
from pipecat.services.dograh.tts import DograhTTSService, DograhTTSSettings
from pipecat.services.elevenlabs.stt import (
    CommitStrategy,
    ElevenLabsRealtimeSTTService,
    ElevenLabsRealtimeSTTSettings,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
from pipecat.services.gladia.stt import GladiaSTTService, GladiaSTTSettings
from pipecat.services.google.gemini_live.llm import (
    ContextWindowCompressionParams,
    GeminiLiveLLMSettings,
)
from pipecat.services.google.gemini_live.vertex.llm import GeminiLiveVertexLLMSettings
from pipecat.services.google.llm import GoogleLLMService, GoogleLLMSettings
from pipecat.services.google.stt import GoogleSTTService, GoogleSTTSettings
from pipecat.services.google.tts import GoogleTTSService, GoogleTTSSettings
from pipecat.services.google.vertex.llm import (
    GoogleVertexLLMService,
    GoogleVertexLLMSettings,
)
from pipecat.services.groq.llm import GroqLLMService, GroqLLMSettings
from pipecat.services.huggingface.llm import (
    HuggingFaceLLMService,
    HuggingFaceLLMSettings,
)
from pipecat.services.huggingface.stt import (
    HuggingFaceSTTService,
    HuggingFaceSTTSettings,
)
from pipecat.services.inworld.tts import InworldTTSService, InworldTTSSettings
from pipecat.services.lmnt.tts import LmntTTSService, LmntTTSSettings
from pipecat.services.minimax.llm import MiniMaxLLMService
from pipecat.services.minimax.tts import MiniMaxTTSSettings
from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.realtime.events import InputAudioNoiseReduction, Reasoning
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMSettings
from pipecat.services.openai.stt import (
    OpenAIRealtimeSTTService,
    OpenAIRealtimeSTTSettings,
    OpenAISTTService,
    OpenAISTTSettings,
)
from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings
from pipecat.services.openrouter.llm import OpenRouterLLMService, OpenRouterLLMSettings
from pipecat.services.rime.tts import RimeTTSService, RimeTTSSettings
from pipecat.services.sarvam.llm import SarvamLLMService, SarvamLLMSettings
from pipecat.services.sarvam.stt import SarvamSTTService, SarvamSTTSettings
from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSettings
from pipecat.services.smallest.stt import SmallestSTTService, SmallestSTTSettings
from pipecat.services.smallest.tts import SmallestTTSService, SmallestTTSSettings
from pipecat.services.speaches.llm import SpeachesLLMService, SpeachesLLMSettings
from pipecat.services.speaches.stt import SpeachesSTTService, SpeachesSTTSettings
from pipecat.services.speaches.tts import SpeachesTTSService, SpeachesTTSSettings
from pipecat.services.speechmatics.stt import (
    SpeechmaticsSTTService,
    SpeechmaticsSTTSettings,
)
from pipecat.services.ultravox.llm import UltravoxRealtimeLLMSettings
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMSettings
from pipecat.services.xai.tts import XAITTSService, XAIWebsocketTTSSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter

if TYPE_CHECKING:
    from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
    from api.services.pipecat.audio_config import AudioConfig


def _report_service_factory_failures(
    source: ErrorSource,
    *,
    config_section: str | None = None,
    provider_argument: int | None = None,
):
    """Classify constructor failures and tag successful services for ErrorFrames."""

    def decorator(factory):
        @wraps(factory)
        def wrapped(*args, **kwargs):
            provider = None
            if config_section:
                user_config = args[0] if args else kwargs.get("user_config")
                config = getattr(user_config, config_section, None)
                provider = getattr(config, "provider", None)
            elif provider_argument is not None:
                if len(args) > provider_argument:
                    provider = args[provider_argument]
                else:
                    provider = kwargs.get("provider")

            provider_value = getattr(provider, "value", provider)
            error_owner = (
                "operator" if str(provider_value).lower() == "dograh" else "user"
            )
            try:
                service = factory(*args, **kwargs)
            except Exception as exc:
                log_failure(
                    classify_exception(
                        exc,
                        source=source,
                        provider=provider,
                        error_owner=error_owner,
                    )
                )
                raise

            return annotate_failure_metadata(
                service,
                source=source,
                provider=provider,
                error_owner=error_owner,
            )

        return wrapped

    return decorator


DEEPGRAM_FLUX_LANGUAGE_HINTS = {
    "de": Language.DE,
    "en": Language.EN,
    "es": Language.ES,
    "fr": Language.FR,
    "hi": Language.HI,
    "it": Language.IT,
    "ja": Language.JA,
    "nl": Language.NL,
    "pt": Language.PT,
    "ru": Language.RU,
}


def _as_language(code: Language | str) -> Language:
    """Coerce a ``language_hints`` entry to ``Language``.

    Registry-derived hints are already ``Language`` (see
    ``DEEPGRAM_FLUX_LANGUAGE_HINTS`` above); ``service_tuning``-derived hints
    arrive as plain strings from the validated JSON document.
    """
    return code if isinstance(code, Language) else Language(code)


def dograh_stt_uses_flux_language(language: str | None) -> bool:
    language = language or "multi"
    return language in DEEPGRAM_FLUX_MULTILINGUAL_LANGUAGE_OPTIONS


def _resolve_elevenlabs_stt_language(
    language_code: str | None,
) -> Language | str | None:
    if not language_code or language_code == "auto":
        return None
    try:
        return Language(language_code)
    except ValueError:
        return language_code


def _elevenlabs_websocket_url(base_url: str) -> str:
    """Normalize an ElevenLabs API base URL for WebSocket clients."""
    base_url = base_url.strip()
    parsed = urlparse(base_url)
    if not parsed.netloc:
        return base_url.rstrip("/")

    websocket_scheme = {
        "http": "ws",
        "https": "wss",
    }.get(parsed.scheme, parsed.scheme)
    return urlunparse(
        parsed._replace(
            scheme=websocket_scheme,
            path=parsed.path.rstrip("/"),
        )
    )


def _elevenlabs_realtime_stt_host(base_url: str) -> str:
    """Return the host/path prefix Pipecat's ElevenLabs realtime STT expects.

    Pipecat's realtime STT service builds
    ``wss://{host}/v1/speech-to-text/realtime`` internally, so remove the scheme
    from the same normalized WebSocket URL used by ElevenLabs TTS. Preserve
    netloc (including optional ports) and any path prefix used by BYOK proxies.
    """
    websocket_url = _elevenlabs_websocket_url(base_url)
    parsed = urlparse(websocket_url)
    if parsed.netloc:
        path = parsed.path
        return f"{parsed.netloc}{path}" if path else parsed.netloc
    return websocket_url


def stt_uses_external_turns(user_config) -> bool:
    if user_config.stt.provider == ServiceProviders.DEEPGRAM.value:
        return user_config.stt.model in DEEPGRAM_FLUX_MODELS
    if user_config.stt.provider == ServiceProviders.DOGRAH.value:
        return dograh_stt_uses_flux_language(getattr(user_config.stt, "language", None))
    if user_config.stt.provider == ServiceProviders.CARTESIA.value:
        return user_config.stt.model == "ink-2"
    return False


class DograhGoogleLLMService(GoogleLLMService):
    adapter_class = DograhGeminiJSONSchemaAdapter


class DograhGoogleVertexLLMService(GoogleVertexLLMService):
    adapter_class = DograhGeminiJSONSchemaAdapter


def _validate_runtime_service_url(url: str, field_name: str) -> None:
    try:
        validate_user_configured_service_url(
            url,
            field_name=field_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _tts_provider_ctor(plan) -> dict:
    """A TTS plan's own ctor kwargs, without the ``tts._all`` one.

    ``tuning_for`` merges the ``_all`` section into every provider plan, and
    ``silence_time_s`` is the only kwarg that section carries: each branch
    already passes it by hand, so splatting the plan would be a duplicate
    keyword argument — and it would hand it to Camb, which is the one branch
    deliberately built without it (C8).
    """
    return {
        name: value for name, value in plan.ctor.items() if name != "silence_time_s"
    }


@_report_service_factory_failures(ErrorSource.STT, config_section="stt")
def create_stt_service(
    user_config,
    audio_config: "AudioConfig",
    keyterms: list[str] | None = None,
    correlation_id: str | None = None,
    *,
    tuning: dict | None = None,
):
    """Create and return appropriate STT service based on user configuration

    Args:
        user_config: User configuration containing STT settings
        keyterms: Optional list of keyterms for speech recognition boosting (Deepgram only)
        tuning: The run's ``service_tuning`` document (validated at the PUT and
            clamped by the cascade resolver). ``None`` means "build exactly the
            defaults below".
    """
    logger.info(
        f"Creating STT service: provider={user_config.stt.provider}, model={user_config.stt.model}"
    )
    if user_config.stt.provider == ServiceProviders.DEEPGRAM.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.DEEPGRAM.value)
        if user_config.stt.model in DEEPGRAM_FLUX_MODELS:
            settings_kwargs = {
                "model": user_config.stt.model,
                "eot_timeout_ms": 3000,
                "eot_threshold": 0.7,
                "eager_eot_threshold": 0.5,
                "keyterm": keyterms or [],
            }
            if user_config.stt.model == "flux-general-multi":
                language = getattr(user_config.stt, "language", None)
                language_hint = DEEPGRAM_FLUX_LANGUAGE_HINTS.get(language)
                if language_hint:
                    settings_kwargs["language_hints"] = [language_hint]

            flux_settings = {
                k: v for k, v in plan.settings.items() if k in FLUX_SETTINGS
            }
            settings = build_settings(
                DeepgramFluxSTTSettings,
                settings_kwargs,
                replace(plan, settings=flux_settings),
            )
            if isinstance(settings.language_hints, list):
                settings.language_hints = [
                    _as_language(h) for h in settings.language_hints
                ]
            if "url" in plan.ctor:
                _validate_runtime_service_url(plan.ctor["url"], "stt.deepgram.ctor.url")

            return DeepgramFluxSTTService(
                api_key=user_config.stt.api_key,
                settings=settings,
                should_interrupt=False,  # Let UserAggregator take care of sending InterruptionFrame
                sample_rate=audio_config.transport_in_sample_rate,
                **plan.ctor,
            )

        # Other models than flux
        # Use language from user config, defaulting to "multi" for multilingual support
        language = getattr(user_config.stt, "language", None) or "multi"
        nova_settings = {
            k: v
            for k, v in plan.settings.items()
            if k not in FLUX_SETTINGS or k in ("numerals", "keyterm")
        }
        settings = build_settings(
            DeepgramSTTSettings,
            {
                "language": language,
                "profanity_filter": False,
                "endpointing": 100,
                "model": user_config.stt.model,
                "keyterm": keyterms or [],
            },
            replace(plan, settings=nova_settings),
        )
        # Nova takes base_url, not url; url is Flux-only, so drop it here even
        # though the shared allow-list accepts ctor.url for either model.
        ctor = {k: v for k, v in plan.ctor.items() if k in ("mip_opt_out", "tag")}
        return DeepgramSTTService(
            api_key=user_config.stt.api_key,
            settings=settings,
            should_interrupt=False,  # Let UserAggregator take care of sending InterruptionFrame
            sample_rate=audio_config.transport_in_sample_rate,
            **ctor,
        )
    elif user_config.stt.provider == ServiceProviders.OPENAI.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.OPENAI.value)
        kwargs = {}
        base_url = getattr(user_config.stt, "base_url", None)
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        if plan.options.get("api") == "realtime":
            # A different service and wire protocol behind the same provider:
            # the Realtime API in transcription-only mode, over a websocket.
            # ``turn_detection=False`` is passed explicitly rather than left to
            # the constructor default: server-side turns would make the service
            # broadcast its own user-turn frames while the pipeline still runs
            # local VAD, and who owns the turn is the turn PR's subject.
            settings_kwargs = {"model": user_config.stt.model}
            # Both OpenAI STT services seed English (openai/stt.py:119,313), so
            # the configured language has to be sent or a Spanish call is
            # transcribed as English. Sent as the raw code like the other STT
            # branches: ``Language`` is a StrEnum and the session serialises
            # either (openai/stt.py:563-568), while ``Language(code)`` would
            # raise on a code the enum doesn't carry.
            language = getattr(user_config.stt, "language", None)
            if language:
                settings_kwargs["language"] = language
            return OpenAIRealtimeSTTService(
                api_key=user_config.stt.api_key,
                settings=build_settings(
                    OpenAIRealtimeSTTSettings, settings_kwargs, plan
                ),
                turn_detection=False,
                should_interrupt=False,
                **kwargs,
            )
        return OpenAISTTService(
            api_key=user_config.stt.api_key,
            settings=build_settings(
                OpenAISTTSettings, {"model": user_config.stt.model}, plan
            ),
            should_interrupt=False,  # Let UserAggregator own interruption confirmation.
            **kwargs,
        )
    elif user_config.stt.provider == ServiceProviders.GOOGLE.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.GOOGLE.value)
        language = getattr(user_config.stt, "language", None) or "en-US"
        location = getattr(user_config.stt, "location", None) or "global"
        credentials = getattr(user_config.stt, "credentials", None)

        settings_kwargs = {"model": user_config.stt.model}
        try:
            settings_kwargs["languages"] = [Language(language)]
        except ValueError:
            settings_kwargs["language_codes"] = [language]

        return GoogleSTTService(
            credentials=credentials,
            location=location,
            settings=build_settings(GoogleSTTSettings, settings_kwargs, plan),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.CARTESIA.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.CARTESIA.value)
        language = getattr(user_config.stt, "language", None) or "en"
        if user_config.stt.model == "ink-2":
            # ink-2 got no settings at all until now (C1), so keyterms — the
            # one biasing knob its wire protocol carries (cartesia/turns/stt
            # .py:276) — could never reach it.
            return CartesiaTurnsSTTService(
                api_key=user_config.stt.api_key,
                settings=build_settings(
                    CartesiaTurnsSTTSettings,
                    {"model": user_config.stt.model, "language": language},
                    plan,
                ),
                should_interrupt=False,  # Let UserAggregator emit interruption frames.
                sample_rate=audio_config.transport_in_sample_rate,
            )

        return CartesiaSTTService(
            api_key=user_config.stt.api_key,
            settings=build_settings(
                CartesiaSTTSettings,
                {"model": user_config.stt.model, "language": language},
                plan,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.DOGRAH.value:
        base_url = MPS_API_URL.replace("http://", "ws://").replace("https://", "wss://")
        language = getattr(user_config.stt, "language", None) or "multi"

        if dograh_stt_uses_flux_language(language):
            # Dograh's Flux proxy only supports multilingual auto-detect and the
            # same language hint subset as Deepgram Flux multilingual.
            settings_kwargs = {
                "model": "flux-general-multi",
                "eot_timeout_ms": 3000,
                "eot_threshold": 0.7,
                "eager_eot_threshold": 0.5,
                "keyterm": keyterms or [],
            }
            language_hint = DEEPGRAM_FLUX_LANGUAGE_HINTS.get(language)
            if language_hint:
                settings_kwargs["language_hints"] = [language_hint]

            plan = tuning_for(tuning, "stt", ServiceProviders.DOGRAH.value)
            flux_settings = {
                k: v for k, v in plan.settings.items() if k in FLUX_SETTINGS
            }
            settings = build_settings(
                DeepgramFluxSTTSettings,
                settings_kwargs,
                replace(plan, settings=flux_settings),
            )
            if isinstance(settings.language_hints, list):
                settings.language_hints = [
                    _as_language(h) for h in settings.language_hints
                ]

            return DograhFluxSTTService(
                base_url=base_url,
                api_key=user_config.stt.api_key,
                correlation_id=correlation_id,
                settings=settings,
                should_interrupt=False,  # external turn strategies own interruption
                sample_rate=audio_config.transport_in_sample_rate,
                **plan.ctor,  # dograh's ctor allow-list has no "url" (SPECS)
            )

        return DograhSTTService(
            base_url=base_url,
            api_key=user_config.stt.api_key,
            correlation_id=correlation_id,
            settings=DograhSTTSettings(
                model=user_config.stt.model,
                language=language,
            ),
            keyterms=keyterms,
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SARVAM.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.SARVAM.value)
        language = getattr(user_config.stt, "language", None)
        language_mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "pa-IN": Language.PA_IN,
            "od-IN": Language.OR_IN,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
            "ur-IN": Language.UR_IN,
            "kok-IN": Language.KOK_IN,
            "mai-IN": Language.MAI_IN,
            "sd-IN": Language.SD_IN,
        }
        if not language or language == "unknown":
            pipecat_language = None
        elif language in language_mapping:
            pipecat_language = language_mapping[language]
        else:
            # Unmapped BCP-47 codes pass through; Sarvam accepts them per https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
            pipecat_language = language
        return SarvamSTTService(
            api_key=user_config.stt.api_key,
            settings=build_settings(
                SarvamSTTSettings,
                {"model": user_config.stt.model, "language": pipecat_language},
                plan,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SPEACHES.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.SPEACHES.value)
        language = getattr(user_config.stt, "language", None)
        _validate_runtime_service_url(user_config.stt.base_url, "base_url")
        return SpeachesSTTService(
            base_url=user_config.stt.base_url,
            api_key=user_config.stt.api_key or "none",
            settings=build_settings(
                SpeachesSTTSettings,
                {"model": user_config.stt.model, "language": language},
                plan,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.HUGGINGFACE.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.HUGGINGFACE.value)
        base_url = (
            getattr(user_config.stt, "base_url", None)
            or "https://router.huggingface.co/hf-inference"
        )
        _validate_runtime_service_url(base_url, "base_url")
        return HuggingFaceSTTService(
            api_key=user_config.stt.api_key,
            base_url=base_url,
            bill_to=getattr(user_config.stt, "bill_to", None),
            settings=build_settings(
                HuggingFaceSTTSettings,
                {
                    "model": user_config.stt.model,
                    "return_timestamps": getattr(
                        user_config.stt, "return_timestamps", False
                    ),
                },
                plan,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.ASSEMBLYAI.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.ASSEMBLYAI.value)
        language = getattr(user_config.stt, "language", None)
        settings_kwargs = {"model": user_config.stt.model, "language": language}
        if keyterms:
            settings_kwargs["keyterms_prompt"] = keyterms
        return AssemblyAISTTService(
            api_key=user_config.stt.api_key,
            settings=build_settings(AssemblyAISTTSettings, settings_kwargs, plan),
            sample_rate=audio_config.transport_in_sample_rate,
            **plan.ctor,
        )
    elif user_config.stt.provider == ServiceProviders.GLADIA.value:
        from pipecat.services.gladia.config import (
            LanguageConfig,
            MessagesConfig,
            PreProcessingConfig,
            RealtimeProcessingConfig,
        )

        plan = tuning_for(tuning, "stt", ServiceProviders.GLADIA.value)
        language = getattr(user_config.stt, "language", None) or "en"
        settings_kwargs = {
            "model": user_config.stt.model,
            "language_config": LanguageConfig(
                languages=[language], code_switching=False
            ),
        }
        return GladiaSTTService(
            api_key=user_config.stt.api_key,
            settings=build_settings(
                GladiaSTTSettings,
                settings_kwargs,
                coerce_settings(
                    plan,
                    {
                        "pre_processing": PreProcessingConfig.model_validate,
                        "realtime_processing": RealtimeProcessingConfig.model_validate,
                        "messages_config": MessagesConfig.model_validate,
                    },
                ),
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SPEECHMATICS.value:
        from pipecat.services.speechmatics.stt import (
            AdditionalVocabEntry,
            OperatingPoint,
            SpeakerIdentifier,
            TurnDetectionMode,
        )

        plan = tuning_for(tuning, "stt", ServiceProviders.SPEECHMATICS.value)
        language = getattr(user_config.stt, "language", None) or "en"
        # Map model field to operating point (standard or enhanced)
        operating_point = (
            OperatingPoint.ENHANCED
            if user_config.stt.model == "enhanced"
            else OperatingPoint.STANDARD
        )
        # Convert keyterms to AdditionalVocabEntry objects for Speechmatics
        additional_vocab = []
        if keyterms:
            additional_vocab = [AdditionalVocabEntry(content=term) for term in keyterms]
        return SpeechmaticsSTTService(
            api_key=user_config.stt.api_key,
            settings=build_settings(
                SpeechmaticsSTTSettings,
                {
                    "language": language,
                    "operating_point": operating_point,
                    "additional_vocab": additional_vocab,
                },
                coerce_settings(
                    plan,
                    {
                        # _build_config reads .value off the mode
                        # (speechmatics/stt.py:752); the vocabulary and speaker
                        # lists are handed to the SDK config, which does not
                        # validate on assignment (:772-775).
                        "turn_detection_mode": TurnDetectionMode,
                        "additional_vocab": lambda entries: [
                            AdditionalVocabEntry.model_validate(e) for e in entries
                        ],
                        "known_speakers": lambda entries: [
                            SpeakerIdentifier(**e) if isinstance(e, dict) else e
                            for e in entries
                        ],
                    },
                ),
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.AZURE_SPEECH.value:
        from pipecat.transcriptions.language import Language as PipecatLanguage

        plan = tuning_for(tuning, "stt", ServiceProviders.AZURE_SPEECH.value)
        language_code = getattr(user_config.stt, "language", None) or "en-US"
        region = getattr(user_config.stt, "region", None) or "eastus"
        try:
            pipecat_language = PipecatLanguage(language_code)
        except ValueError:
            pipecat_language = language_code
        return AzureSTTService(
            api_key=user_config.stt.api_key,
            region=region,
            settings=build_settings(
                AzureSTTSettings, {"language": pipecat_language}, plan
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SMALLEST.value:
        plan = tuning_for(tuning, "stt", ServiceProviders.SMALLEST.value)
        language_code = getattr(user_config.stt, "language", None) or "en"
        try:
            pipecat_language = Language(language_code)
        except ValueError:
            pipecat_language = Language.EN
        return SmallestSTTService(
            api_key=user_config.stt.api_key,
            settings=build_settings(
                SmallestSTTSettings,
                {"model": user_config.stt.model, "language": pipecat_language},
                plan,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.ELEVENLABS.value:
        language_code = getattr(user_config.stt, "language", None)
        pipecat_language = _resolve_elevenlabs_stt_language(language_code)

        _validate_runtime_service_url(user_config.stt.base_url, "base_url")
        elevenlabs_host = _elevenlabs_realtime_stt_host(user_config.stt.base_url)

        plan = tuning_for(tuning, "stt", ServiceProviders.ELEVENLABS.value)
        # CommitStrategy is a StrEnum whose values are the lowercase wire
        # strings ("manual"/"vad" — elevenlabs/stt.py:167-171), so the plan's
        # string constructs it directly.
        commit_strategy = CommitStrategy(plan.ctor.get("commit_strategy", "vad"))
        ctor = {k: v for k, v in plan.ctor.items() if k != "commit_strategy"}
        settings = build_settings(
            ElevenLabsRealtimeSTTSettings,
            {
                "model": user_config.stt.model,
                "language": pipecat_language,
            },
            plan,
        )

        return ElevenLabsRealtimeSTTService(
            api_key=user_config.stt.api_key,
            base_url=elevenlabs_host,
            commit_strategy=commit_strategy,
            settings=settings,
            should_interrupt=False,
            sample_rate=audio_config.transport_in_sample_rate,
            **ctor,
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid STT provider {user_config.stt.provider}"
        )


@_report_service_factory_failures(ErrorSource.TTS, config_section="tts")
def create_tts_service(
    user_config,
    audio_config: "AudioConfig",
    correlation_id: str | None = None,
    *,
    tuning: dict | None = None,
):
    """Create and return appropriate TTS service based on user configuration

    Args:
        user_config: User configuration containing TTS settings
        transport_type: Type of transport (e.g., 'twilio', 'webrtc')
        tuning: The run's ``service_tuning`` document; ``None`` builds the
            defaults below unchanged.
    """
    logger.info(
        f"Creating TTS service: provider={user_config.tts.provider}, model={user_config.tts.model}"
    )
    # Create function call filter to prevent TTS from speaking function call tags
    xml_function_tag_filter = XMLFunctionTagFilter()
    # Applies to every TTS branch below (except Camb, which doesn't take
    # silence_time_s — C8).
    common = tuning_for(tuning, "tts", ALL)
    silence_time_s = common.ctor.get("silence_time_s", 1.0)
    if user_config.tts.provider == ServiceProviders.DEEPGRAM.value:
        plan = tuning_for(tuning, "tts", ServiceProviders.DEEPGRAM.value)
        return DeepgramTTSService(
            api_key=user_config.tts.api_key,
            settings=build_settings(
                DeepgramTTSSettings, {"voice": user_config.tts.voice}, plan
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.OPENAI.value:
        kwargs = {}
        base_url = getattr(user_config.tts, "base_url", None)
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        plan = tuning_for(tuning, "tts", ServiceProviders.OPENAI.value)
        settings = build_settings(
            OpenAITTSSettings,
            {"model": user_config.tts.model, "voice": user_config.tts.voice},
            plan,
        )
        return OpenAITTSService(
            api_key=user_config.tts.api_key,
            sample_rate=OPENAI_SAMPLE_RATE,
            settings=settings,
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
            **kwargs,
        )
    elif user_config.tts.provider == ServiceProviders.GOOGLE.value:
        model = getattr(user_config.tts, "model", None) or "chirp_3_hd"
        language = getattr(user_config.tts, "language", None) or "en-US"
        voice = getattr(user_config.tts, "voice", None) or "en-US-Chirp3-HD-Charon"
        speed = getattr(user_config.tts, "speed", None)
        location = getattr(user_config.tts, "location", None) or None
        credentials = getattr(user_config.tts, "credentials", None)

        settings_kwargs = {
            "model": model,
            "voice": voice,
            "language": language,
        }
        if speed is not None and speed != 1.0:
            settings_kwargs["speaking_rate"] = speed

        plan = tuning_for(tuning, "tts", ServiceProviders.GOOGLE.value)
        return GoogleTTSService(
            credentials=credentials,
            location=location,
            settings=build_settings(GoogleTTSSettings, settings_kwargs, plan),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.ELEVENLABS.value:
        # Backward compatible with older configuration "Name - voice_id"
        try:
            voice_id = user_config.tts.voice.split(" - ")[1]
        except IndexError:
            voice_id = user_config.tts.voice
        # ElevenLabs TTS consumes the full normalized WebSocket URL. Realtime
        # STT uses the same normalization before adapting it to Pipecat's
        # scheme-less base_url contract.
        _validate_runtime_service_url(user_config.tts.base_url, "base_url")
        elevenlabs_url = _elevenlabs_websocket_url(user_config.tts.base_url)
        plan = tuning_for(tuning, "tts", ServiceProviders.ELEVENLABS.value)
        settings = build_settings(
            ElevenLabsTTSSettings,
            {
                "voice": voice_id,
                "model": user_config.tts.model,
                "stability": 0.8,
                "speed": user_config.tts.speed,
                "similarity_boost": 0.75,
            },
            plan,
        )
        ctor = {"reconnect_on_error": plan.ctor.get("reconnect_on_error", False)}
        if "auto_mode" in plan.ctor:
            ctor["auto_mode"] = plan.ctor["auto_mode"]
        if "enable_ssml_parsing" in plan.ctor:
            ctor["enable_ssml_parsing"] = plan.ctor["enable_ssml_parsing"]
        return ElevenLabsTTSService(
            api_key=user_config.tts.api_key,
            url=elevenlabs_url,
            settings=settings,
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
            **ctor,
        )
    elif user_config.tts.provider == ServiceProviders.CARTESIA.value:
        speed = getattr(user_config.tts, "speed", None)
        volume = getattr(user_config.tts, "volume", None)
        gen_config_kwargs = {}
        if speed and speed != 1.0:
            gen_config_kwargs["speed"] = speed
        if volume and volume != 1.0:
            gen_config_kwargs["volume"] = volume
        generation_config = (
            GenerationConfig(**gen_config_kwargs) if gen_config_kwargs else None
        )
        language = getattr(user_config.tts, "language", None) or "en"
        settings_kwargs = {
            "voice": user_config.tts.voice,
            "model": user_config.tts.model,
            "language": language,
        }
        if generation_config:
            settings_kwargs["generation_config"] = generation_config
        plan = tuning_for(tuning, "tts", ServiceProviders.CARTESIA.value)
        return CartesiaTTSService(
            api_key=user_config.tts.api_key,
            settings=build_settings(
                CartesiaTTSSettings,
                settings_kwargs,
                # _build_msg calls .model_dump() on it (cartesia/tts.py:515-517),
                # so the tuning document's plain object has to become the model.
                coerce_settings(
                    plan, {"generation_config": GenerationConfig.model_validate}
                ),
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
            **_tts_provider_ctor(plan),
        )
    elif user_config.tts.provider == ServiceProviders.INWORLD.value:
        voice = getattr(user_config.tts, "voice", None) or "Ashley"
        model = getattr(user_config.tts, "model", None) or "inworld-tts-2"
        speed = getattr(user_config.tts, "speed", None)
        language = getattr(user_config.tts, "language", None) or "en-US"
        delivery_mode = getattr(user_config.tts, "delivery_mode", None) or "BALANCED"
        plan = tuning_for(tuning, "tts", ServiceProviders.INWORLD.value)
        return InworldTTSService(
            api_key=user_config.tts.api_key,
            settings=build_settings(
                InworldTTSSettings,
                {
                    "voice": voice,
                    "model": model,
                    "language": language,
                    "speaking_rate": speed,
                    "delivery_mode": delivery_mode,
                },
                plan,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.DOGRAH.value:
        # Convert HTTP URL to WebSocket URL for TTS
        base_url = MPS_API_URL.replace("http://", "ws://").replace("https://", "wss://")
        plan = tuning_for(tuning, "tts", ServiceProviders.DOGRAH.value)
        return DograhTTSService(
            base_url=base_url,
            api_key=user_config.tts.api_key,
            correlation_id=correlation_id,
            settings=build_settings(
                DograhTTSSettings,
                {
                    "model": user_config.tts.model,
                    "voice": user_config.tts.voice,
                    "speed": user_config.tts.speed,
                },
                plan,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.CAMB.value:
        from pipecat.services.camb.tts import CambTTSService

        voice_id = int(getattr(user_config.tts, "voice", None) or "147320")
        language = getattr(user_config.tts, "language", None) or "en-us"
        plan = tuning_for(tuning, "tts", ServiceProviders.CAMB.value)
        tts = CambTTSService(
            api_key=user_config.tts.api_key,
            voice_id=voice_id,
            model=user_config.tts.model,
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            **_tts_provider_ctor(plan),
        )
        # Set language directly as BCP-47 code (bypasses Language enum conversion)
        tts._settings.language = language
        return tts
    elif user_config.tts.provider == ServiceProviders.SPEACHES.value:
        _validate_runtime_service_url(user_config.tts.base_url, "base_url")
        return SpeachesTTSService(
            base_url=user_config.tts.base_url,
            api_key=user_config.tts.api_key or "none",
            settings=SpeachesTTSSettings(
                model=user_config.tts.model,
                voice=user_config.tts.voice,
                speed=user_config.tts.speed,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.RIME.value:
        speed = getattr(user_config.tts, "speed", None)
        language_code = getattr(user_config.tts, "language", None) or "en"
        rime_language_mapping = {
            "en": Language.EN,
            "de": Language.DE,
            "fr": Language.FR,
            "es": Language.ES,
            "hi": Language.HI,
        }
        pipecat_language = rime_language_mapping.get(language_code, Language.EN)
        settings_kwargs = {
            "voice": user_config.tts.voice,
            "model": user_config.tts.model,
            "language": pipecat_language,
        }
        if speed and speed != 1.0:
            settings_kwargs["speedAlpha"] = speed
        plan = tuning_for(tuning, "tts", ServiceProviders.RIME.value)
        return RimeTTSService(
            api_key=user_config.tts.api_key,
            settings=build_settings(RimeTTSSettings, settings_kwargs, plan),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.SARVAM.value:
        # Map Sarvam language code to pipecat Language enum for TTS
        language_mapping = {
            "bn-IN": Language.BN,
            "en-IN": Language.EN,
            "gu-IN": Language.GU,
            "hi-IN": Language.HI,
            "kn-IN": Language.KN,
            "ml-IN": Language.ML,
            "mr-IN": Language.MR,
            "od-IN": Language.OR,
            "pa-IN": Language.PA,
            "ta-IN": Language.TA,
            "te-IN": Language.TE,
        }
        language = getattr(user_config.tts, "language", None)
        pipecat_language = language_mapping.get(language, Language.HI)

        voice = (
            getattr(user_config.tts, "voice", None) or ""
        ).strip().lower() or "anushka"
        speed = getattr(user_config.tts, "speed", None)
        settings_kwargs = {
            "model": user_config.tts.model,
            "voice": voice,
            "language": pipecat_language,
        }
        if speed and speed != 1.0:
            settings_kwargs["pace"] = speed
        plan = tuning_for(tuning, "tts", ServiceProviders.SARVAM.value)
        return SarvamTTSService(
            api_key=user_config.tts.api_key,
            settings=build_settings(SarvamTTSSettings, settings_kwargs, plan),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.MINIMAX.value:
        group_id = getattr(user_config.tts, "group_id", None)
        if not group_id:
            raise HTTPException(
                status_code=400,
                detail="MiniMax TTS requires a group_id. Configure it in your TTS settings.",
            )
        voice = getattr(user_config.tts, "voice", None) or "English_Graceful_Lady"
        speed = getattr(user_config.tts, "speed", None) or 1.0

        # Pipecat appends "?GroupId=..." to base_url as-is, so /t2a_v2 must
        # already be in the path.
        base_url = (
            getattr(user_config.tts, "base_url", None)
            or "https://api.minimax.io/v1/t2a_v2"
        ).rstrip("/")
        if not base_url.endswith("/t2a_v2"):
            base_url = f"{base_url}/t2a_v2"
        _validate_runtime_service_url(base_url, "base_url")

        plan = tuning_for(tuning, "tts", ServiceProviders.MINIMAX.value)
        session = aiohttp.ClientSession()
        return MiniMaxOwnedSessionTTSService(
            api_key=user_config.tts.api_key,
            group_id=group_id,
            base_url=base_url,
            aiohttp_session=session,
            settings=build_settings(
                MiniMaxTTSSettings,
                {
                    "model": user_config.tts.model,
                    "voice": voice,
                    "speed": speed,
                },
                plan,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.AZURE_SPEECH.value:
        region = getattr(user_config.tts, "region", None) or "eastus"
        voice = getattr(user_config.tts, "voice", None) or "en-US-AriaNeural"
        language = getattr(user_config.tts, "language", None) or "en-US"
        speed = getattr(user_config.tts, "speed", None) or 1.0
        # Map speed multiplier (0.5–2.0) to Azure SSML rate string (e.g. "1.25")
        rate = str(speed) if speed != 1.0 else None
        settings_kwargs: dict = {
            "voice": voice,
            "language": language,
        }
        if rate:
            settings_kwargs["rate"] = rate
        plan = tuning_for(tuning, "tts", ServiceProviders.AZURE_SPEECH.value)
        return AzureTTSService(
            api_key=user_config.tts.api_key,
            region=region,
            settings=build_settings(AzureTTSSettings, settings_kwargs, plan),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.SMALLEST.value:
        language_code = getattr(user_config.tts, "language", None) or "en"
        try:
            pipecat_language = Language(language_code)
        except ValueError:
            pipecat_language = Language.EN
        speed = getattr(user_config.tts, "speed", None)
        model = user_config.tts.model.replace("lightning-v", "lightning_v")
        settings_kwargs = {
            "model": model,
            "voice": user_config.tts.voice,
            "language": pipecat_language,
        }
        if speed and speed != 1.0:
            settings_kwargs["speed"] = speed
        plan = tuning_for(tuning, "tts", ServiceProviders.SMALLEST.value)
        return SmallestTTSService(
            api_key=user_config.tts.api_key,
            settings=build_settings(SmallestTTSSettings, settings_kwargs, plan),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.XAI.value:
        voice = getattr(user_config.tts, "voice", None) or "eve"
        language_code = getattr(user_config.tts, "language", None) or "en"
        if language_code.lower() == "auto":
            pipecat_language = "auto"
        else:
            try:
                pipecat_language = Language(language_code)
            except ValueError:
                pipecat_language = Language.EN
        settings_kwargs = {"voice": voice, "language": pipecat_language}
        # B12: the configured speed used to stop here — the branch built the
        # settings without it, so xAI always spoke at 1.0.
        speed = getattr(user_config.tts, "speed", None)
        if speed and speed != 1.0:
            settings_kwargs["speed"] = speed
        plan = tuning_for(tuning, "tts", ServiceProviders.XAI.value)
        return XAITTSService(
            api_key=user_config.tts.api_key,
            settings=build_settings(XAIWebsocketTTSSettings, settings_kwargs, plan),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    elif user_config.tts.provider == ServiceProviders.LMNT.value:
        voice = getattr(user_config.tts, "voice", None) or "lily"
        model = getattr(user_config.tts, "model", None) or "aurora"
        language_code = getattr(user_config.tts, "language", None) or "en"
        try:
            pipecat_language = Language(language_code)
        except ValueError:
            pipecat_language = Language.EN
        plan = tuning_for(tuning, "tts", ServiceProviders.LMNT.value)
        return LmntTTSService(
            api_key=user_config.tts.api_key,
            sample_rate=audio_config.transport_out_sample_rate,
            # LMNT's streaming `format` field expects "raw" for signed 16-bit PCM
            # at the requested sample rate, which is what the output transport
            # consumes; "pcm_s16le" is not a valid LMNT format value.
            output_format="raw",
            settings=build_settings(
                LmntTTSSettings,
                {"voice": voice, "language": pipecat_language, "model": model},
                plan,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=silence_time_s,
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid TTS provider {user_config.tts.provider}"
        )


def _migrate_deprecated_google_model(model: str) -> str:
    """Google removed the ``gemini-2.0-flash*`` models. Transparently upgrade
    any stored config that still references them to the 2.5 equivalent so old
    user configurations keep working instead of failing at runtime."""
    if model and model.startswith("gemini-2.0-flash"):
        migrated = model.replace("gemini-2.0-", "gemini-2.5-", 1)
        logger.warning(
            f"Google model '{model}' is no longer supported; using '{migrated}' instead"
        )
        return migrated
    return model


@_report_service_factory_failures(ErrorSource.LLM, provider_argument=0)
def create_llm_service_from_provider(
    provider: str,
    model: str,
    api_key: str | None,
    *,
    correlation_id: str | None = None,
    base_url: str | None = None,
    endpoint: str | None = None,
    aws_access_key: str | None = None,
    aws_secret_key: str | None = None,
    aws_region: str | None = None,
    project_id: str | None = None,
    location: str | None = None,
    credentials: str | None = None,
    temperature: float | None = None,
    bill_to: str | None = None,
    usage_context: str | None = None,
    tuning: dict | None = None,
):
    """Create an LLM service from explicit provider/model/api_key.

    Also used by create_llm_service which extracts these from user_config.

    Args:
        usage_context: Optional tag describing what the LLM instance is used for
            (e.g. "voicemail_detection"). Sent as request metadata by the Dograh
            provider; ignored by other providers.
        tuning: The run's ``service_tuning`` document, already narrowed by the
            caller to the roles it applies to (``llm_document_for_role``);
            ``None`` builds the defaults below unchanged.
    """
    logger.info(f"Creating LLM service: provider={provider}, model={model}")
    plan = tuning_for(tuning, "llm", provider)
    if provider in (
        ServiceProviders.OPENAI.value,
        ServiceProviders.ATLASCLOUD.value,
    ):
        kwargs = {}
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        if "gpt-5" in model:
            # Today's behaviour, declared: two extras and no temperature
            # (spec §5.4). ``extra`` is REGISTRY_OWNED, so these two keys are
            # only ever reachable through ``options`` — never through
            # ``from_mapping``'s overflow.
            reasoning = plan.options.get("reasoning") or {}
            settings = build_settings(OpenAILLMSettings, {"model": model}, plan)
            settings.extra = {
                "reasoning_effort": reasoning.get("effort", "minimal"),
                "verbosity": plan.options.get("verbosity", "low"),
            }
            return OpenAILLMService(api_key=api_key, settings=settings, **kwargs)
        return OpenAILLMService(
            api_key=api_key,
            settings=build_settings(
                OpenAILLMSettings, {"model": model, "temperature": 0.1}, plan
            ),
            **kwargs,
        )
    elif provider == ServiceProviders.GROQ.value:
        return GroqLLMService(
            api_key=api_key,
            settings=build_settings(
                GroqLLMSettings, {"model": model, "temperature": 0.1}, plan
            ),
        )
    elif provider == ServiceProviders.OPENROUTER.value:
        kwargs = {}
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        return OpenRouterLLMService(
            api_key=api_key,
            settings=build_settings(
                OpenRouterLLMSettings, {"model": model, "temperature": 0.1}, plan
            ),
            **kwargs,
        )
    elif provider == ServiceProviders.GOOGLE.value:
        model = _migrate_deprecated_google_model(model)
        return DograhGoogleLLMService(
            api_key=api_key,
            settings=build_settings(
                GoogleLLMSettings, {"model": model, "temperature": 0.1}, plan
            ),
        )
    elif provider == ServiceProviders.GOOGLE_VERTEX.value:
        return DograhGoogleVertexLLMService(
            credentials=credentials,
            project_id=project_id,
            location=location or "us-east4",
            settings=build_settings(
                GoogleVertexLLMSettings, {"model": model, "temperature": 0.1}, plan
            ),
        )
    elif provider == ServiceProviders.AZURE.value:
        if endpoint:
            _validate_runtime_service_url(endpoint, "endpoint")
        return AzureLLMService(
            api_key=api_key,
            endpoint=endpoint,
            settings=build_settings(
                AzureLLMSettings, {"model": model, "temperature": 0.1}, plan
            ),
        )
    elif provider == ServiceProviders.DOGRAH.value:
        return DograhLLMService(
            base_url=f"{MPS_API_URL}/api/v1/llm",
            api_key=api_key,
            correlation_id=correlation_id,
            usage_context=usage_context,
            settings=build_settings(OpenAILLMSettings, {"model": model}, plan),
        )
    elif provider == ServiceProviders.AWS_BEDROCK.value:
        return AWSBedrockLLMService(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
            settings=build_settings(AWSBedrockLLMSettings, {"model": model}, plan),
        )
    elif provider == ServiceProviders.SPEACHES.value:
        base_url = base_url or "http://localhost:11434/v1"
        _validate_runtime_service_url(base_url, "base_url")
        return SpeachesLLMService(
            base_url=base_url,
            api_key=api_key or "none",
            settings=build_settings(SpeachesLLMSettings, {"model": model}, plan),
        )
    elif provider == ServiceProviders.HUGGINGFACE.value:
        base_url = base_url or "https://router.huggingface.co/v1"
        _validate_runtime_service_url(base_url, "base_url")
        return HuggingFaceLLMService(
            api_key=api_key,
            base_url=base_url,
            bill_to=bill_to,
            settings=build_settings(
                HuggingFaceLLMSettings, {"model": model, "temperature": 0.1}, plan
            ),
        )
    elif provider == ServiceProviders.MINIMAX.value:
        base_url = base_url or "https://api.minimax.io/v1"
        _validate_runtime_service_url(base_url, "base_url")
        return MiniMaxLLMService(
            api_key=api_key,
            base_url=base_url,
            settings=build_settings(
                MiniMaxLLMService.Settings,
                {
                    "model": model,
                    "temperature": temperature if temperature is not None else 1.0,
                },
                plan,
            ),
        )
    elif provider == ServiceProviders.SARVAM.value:
        return SarvamLLMService(
            api_key=api_key,
            settings=build_settings(
                SarvamLLMSettings,
                {
                    "model": model,
                    "temperature": temperature if temperature is not None else 0.5,
                },
                plan,
            ),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid LLM provider {provider}")


REALTIME_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    # Where each ``realtime`` knob lands. The realtime branches are the one
    # part of the plan with no pipecat ``Settings`` mapping for the session:
    # they build provider pydantic objects, so a knob is written to an
    # explicit destination instead of merged through ``from_mapping``. Paths
    # are rooted at the object the branch hands ``_apply_realtime_fields``:
    # the provider ``SessionProperties`` (OpenAI/Azure/Grok), or the kwargs
    # dict the branch is about to construct (Gemini settings, Ultravox
    # params). Kept in step with the allow-lists by
    # test_realtime.py::test_every_realtime_knob_has_a_destination.
    #
    # No turn-detection knob appears here: who owns the user turn is the turn
    # PR's subject, and none of them is in an allow-list.
    "openai_realtime": {
        "noise_reduction": ("audio", "input", "noise_reduction"),
        "speed": ("audio", "output", "speed"),
        # The destination is the whole ``Reasoning`` model, not its ``effort``
        # field: the session leaves ``reasoning`` unset, so there is no object
        # to write into until a run asks for one (_REALTIME_WRAPPERS).
        "reasoning_effort": ("reasoning",),
        "max_output_tokens": ("max_output_tokens",),
        "tool_choice": ("tool_choice",),
    },
    "grok_realtime": {
        "language_hint": ("audio", "input", "transcription", "language_hint"),
    },
    "google_realtime": {
        "thinking": ("thinking",),
        "enable_affective_dialog": ("enable_affective_dialog",),
        "proactivity": ("proactivity",),
        "context_window_compression": ("context_window_compression",),
        "temperature": ("temperature",),
    },
    "ultravox_realtime": {
        "temperature": ("temperature",),
        "max_duration": ("max_duration",),
        "extra": ("extra",),
    },
}
# Azure Realtime is the OpenAI wire protocol against an Azure deployment and
# reuses its events module, so the two sessions take the same knobs; Vertex is
# Gemini Live against Vertex AI.
REALTIME_FIELDS["azure_realtime"] = REALTIME_FIELDS["openai_realtime"]
REALTIME_FIELDS["google_vertex_realtime"] = REALTIME_FIELDS["google_realtime"]

_OPENAI_REALTIME_WRAPPERS: dict[str, Callable[[Any], Any]] = {
    # The session models these two as objects; the knob is the one field
    # inside each that a run configures.
    "noise_reduction": lambda value: InputAudioNoiseReduction(type=value),
    "reasoning_effort": lambda value: Reasoning(effort=value),
}
_GEMINI_REALTIME_COERCIONS: dict[str, Callable[[Any], Any]] = {
    # These arrive from the tuning document as JSON objects. Gemini Live reads
    # ``thinking`` and ``proactivity`` straight off the settings into its
    # connect config (google/gemini_live/llm.py:1222,1230), so a bare dict
    # would reach the SDK; ``model_validate`` also accepts an already-built
    # model, as ``coerce_settings`` requires.
    "thinking": ThinkingConfig.model_validate,
    "proactivity": ProactivityConfig.model_validate,
    "context_window_compression": ContextWindowCompressionParams.model_validate,
}


def _apply_realtime_fields(
    root,
    provider: str,
    plan,
    wrappers: dict[str, Callable[[Any], Any]] | None = None,
) -> None:
    """Write each tuned knob to the destination ``REALTIME_FIELDS`` declares.

    Only the knobs the plan actually carries are touched, so an untuned run
    gets exactly the objects the branch just built. Every node on the way to a
    leaf exists by then — a knob whose destination is a nested model the
    session leaves unset is wrapped instead of walked into.
    """
    fields = REALTIME_FIELDS[provider]
    for name, value in plan.settings.items():
        wrap = (wrappers or {}).get(name)
        *parents, leaf = fields[name]
        target = root
        for step in parents:
            target = target[step] if isinstance(target, dict) else getattr(target, step)
        value = wrap(value) if wrap else value
        if isinstance(target, dict):
            target[leaf] = value
        else:
            setattr(target, leaf, value)


@_report_service_factory_failures(ErrorSource.LLM, config_section="realtime")
def create_realtime_llm_service(
    user_config, audio_config: "AudioConfig", *, tuning: dict | None = None
):
    """Create a realtime (speech-to-speech) LLM service that handles STT+LLM+TTS.

    These services bypass separate STT/TTS and handle audio directly via
    a bidirectional WebSocket connection. Reads from user_config.realtime.

    Args:
        tuning: The run's ``service_tuning`` document; ``None`` builds the
            defaults below unchanged.
    """
    realtime_config = user_config.realtime
    provider = realtime_config.provider
    model = realtime_config.model
    api_key = realtime_config.api_key
    voice = getattr(realtime_config, "voice", None)
    language = getattr(realtime_config, "language", None)

    logger.info(
        f"Creating realtime LLM service: provider={provider}, model={model}, voice={voice}, language={language}"
    )
    plan = tuning_for(tuning, "realtime", provider)

    if provider == ServiceProviders.OPENAI_REALTIME.value:
        from pipecat.services.openai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            AudioOutput,
            InputAudioTranscription,
            SessionProperties,
        )

        # Pin the transcription language when configured. Without it the model
        # auto-detects per utterance, which misfires on short/noisy telephony
        # audio (e.g. Portuguese transcribed as English or Chinese).
        transcription_kwargs = {}
        if language:
            transcription_kwargs["language"] = language

        session_properties = SessionProperties(
            audio=AudioConfiguration(
                input=AudioInput(
                    transcription=InputAudioTranscription(**transcription_kwargs),
                ),
                output=AudioOutput(
                    voice=voice or "alloy",
                ),
            ),
        )
        _apply_realtime_fields(
            session_properties, provider, plan, _OPENAI_REALTIME_WRAPPERS
        )
        return DograhOpenAIRealtimeLLMService(
            api_key=api_key,
            settings=OpenAIRealtimeLLMSettings(
                model=model,
                session_properties=session_properties,
            ),
        )
    elif provider == ServiceProviders.GROK_REALTIME.value:
        from pipecat.services.xai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            InputAudioTranscription,
            SessionProperties,
        )

        grok_voice = voice or "ara"
        if grok_voice.lower() in {"ara", "rex", "sal", "eve", "leo"}:
            grok_voice = grok_voice.lower()

        session_properties = SessionProperties(
            voice=grok_voice,
            audio=AudioConfiguration(
                input=AudioInput(
                    transcription=InputAudioTranscription(),
                ),
            ),
        )
        _apply_realtime_fields(session_properties, provider, plan)
        return DograhGrokRealtimeLLMService(
            api_key=api_key,
            settings=GrokRealtimeLLMSettings(
                model=model,
                session_properties=session_properties,
            ),
        )
    elif provider == ServiceProviders.ULTRAVOX_REALTIME.value:
        # Ultravox is the one branch whose knobs are call-creation parameters
        # rather than session settings: the service posts them when it opens
        # the call (ultravox/llm.py:341-358).
        params_kwargs = {
            "api_key": api_key,
            "model": model,
            "voice": voice,
            "output_medium": "voice",
        }
        _apply_realtime_fields(params_kwargs, provider, plan)
        return DograhUltravoxRealtimeLLMService(
            params=DograhUltravoxOneShotInputParams(**params_kwargs),
            settings=UltravoxRealtimeLLMSettings(
                model=model,
                output_medium="voice",
            ),
        )
    elif provider == ServiceProviders.GOOGLE_REALTIME.value:
        # Gemini Live enables input/output audio transcription by default
        # in its _connect() method — no need to configure it explicitly.
        settings_kwargs = {
            "model": model,
            "voice": voice or "Puck",
        }
        if language:
            settings_kwargs["language"] = language
        temperature = getattr(realtime_config, "temperature", None)
        if temperature is not None:
            settings_kwargs["temperature"] = temperature
        _apply_realtime_fields(
            settings_kwargs,
            provider,
            coerce_settings(plan, _GEMINI_REALTIME_COERCIONS),
        )
        return DograhGeminiLiveLLMService(
            api_key=api_key,
            settings=GeminiLiveLLMSettings(**settings_kwargs),
        )
    elif provider == ServiceProviders.GOOGLE_VERTEX_REALTIME.value:
        project_id = getattr(realtime_config, "project_id", None)
        location = getattr(realtime_config, "location", None) or "us-east4"
        credentials = getattr(realtime_config, "credentials", None)

        settings_kwargs = {
            "model": model,
            "voice": voice or "Charon",
        }
        if language:
            settings_kwargs["language"] = language
        temperature = getattr(realtime_config, "temperature", None)
        if temperature is not None:
            settings_kwargs["temperature"] = temperature
        _apply_realtime_fields(
            settings_kwargs,
            provider,
            coerce_settings(plan, _GEMINI_REALTIME_COERCIONS),
        )
        return DograhGeminiLiveVertexLLMService(
            credentials=credentials,
            project_id=project_id,
            location=location,
            settings=GeminiLiveVertexLLMSettings(**settings_kwargs),
        )
    elif provider == ServiceProviders.AZURE_REALTIME.value:
        from pipecat.services.openai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            AudioOutput,
            InputAudioTranscription,
            SessionProperties,
        )

        endpoint = getattr(realtime_config, "endpoint", None) or ""
        if not endpoint:
            raise HTTPException(
                status_code=400,
                detail="Azure Realtime requires an endpoint.",
            )
        _validate_runtime_service_url(endpoint, "endpoint")
        api_version = getattr(realtime_config, "api_version", None) or "v1"
        parsed_endpoint = urlparse(endpoint)
        if api_version == "v1":
            # Azure's GA Realtime API uses the deployment name as `model` and
            # deliberately has no date-based api-version query parameter.
            path = "/openai/v1/realtime"
            query = urlencode({"model": model})
        else:
            # Preserve explicitly configured preview deployments while users
            # migrate. Microsoft deprecated this protocol on April 30, 2026.
            path = "/openai/realtime"
            query = urlencode({"api-version": api_version, "deployment": model})
        wss_url = urlunparse(
            (
                "wss",
                parsed_endpoint.netloc,
                path,
                "",
                query,
                "",
            )
        )
        session_properties = SessionProperties(
            audio=AudioConfiguration(
                input=AudioInput(
                    transcription=InputAudioTranscription(),
                ),
                output=AudioOutput(
                    voice=voice or "alloy",
                ),
            ),
        )
        _apply_realtime_fields(
            session_properties, provider, plan, _OPENAI_REALTIME_WRAPPERS
        )
        return DograhAzureRealtimeLLMService(
            api_key=api_key,
            base_url=wss_url,
            settings=AzureRealtimeLLMSettings(
                model=model,
                session_properties=session_properties,
            ),
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid realtime LLM provider {provider}"
        )


def create_llm_service(
    user_config,
    correlation_id: str | None = None,
    usage_context: str | None = None,
    *,
    tuning: dict | None = None,
    role: LLMRole = "conversation",
):
    """Create and return appropriate LLM service based on user configuration.

    Args:
        tuning: The run's ``service_tuning`` document.
        role: Which LLM instance this is. ``llm`` tuning reaches the
            conversation LLM always and the secondary instances only when
            ``service_tuning.scope`` opts them in.
    """
    provider = user_config.llm.provider
    model = user_config.llm.model
    api_key = user_config.llm.api_key

    kwargs = {}
    if provider in (
        ServiceProviders.OPENAI.value,
        ServiceProviders.ATLASCLOUD.value,
    ):
        kwargs["base_url"] = user_config.llm.base_url
    elif provider == ServiceProviders.OPENROUTER.value:
        kwargs["base_url"] = user_config.llm.base_url
    elif provider == ServiceProviders.AZURE.value:
        kwargs["endpoint"] = user_config.llm.endpoint
    elif provider == ServiceProviders.SPEACHES.value:
        kwargs["base_url"] = user_config.llm.base_url
    elif provider == ServiceProviders.HUGGINGFACE.value:
        kwargs["base_url"] = user_config.llm.base_url
        kwargs["bill_to"] = user_config.llm.bill_to
    elif provider == ServiceProviders.AWS_BEDROCK.value:
        kwargs["aws_access_key"] = user_config.llm.aws_access_key
        kwargs["aws_secret_key"] = user_config.llm.aws_secret_key
        kwargs["aws_region"] = user_config.llm.aws_region
    elif provider == ServiceProviders.GOOGLE_VERTEX.value:
        kwargs["project_id"] = user_config.llm.project_id
        kwargs["location"] = user_config.llm.location
        kwargs["credentials"] = user_config.llm.credentials
    elif provider == ServiceProviders.MINIMAX.value:
        kwargs["base_url"] = user_config.llm.base_url
        kwargs["temperature"] = user_config.llm.temperature
    elif provider == ServiceProviders.SARVAM.value:
        kwargs["temperature"] = user_config.llm.temperature

    return create_llm_service_from_provider(
        provider,
        model,
        api_key,
        correlation_id=correlation_id,
        usage_context=usage_context,
        tuning=llm_document_for_role(tuning, role),
        **kwargs,
    )


def create_llm_service_with_model_override(
    user_config: "EffectiveAIModelConfiguration",
    model_override: str | None,
    correlation_id: str | None = None,
    usage_context: str | None = None,
):
    """Create an LLM service with an optional model override.

    The copied configuration is delegated to ``create_llm_service`` so provider-
    specific settings continue to be extracted in one place.
    """
    if model_override is None:
        return create_llm_service(
            user_config,
            correlation_id=correlation_id,
            usage_context=usage_context,
        )

    if user_config.llm is None:
        raise ValueError("Cannot override the model without an LLM configuration")

    llm_config = user_config.llm.model_copy(update={"model": model_override})
    overridden_config = user_config.model_copy(update={"llm": llm_config})
    return create_llm_service(
        overridden_config,
        correlation_id=correlation_id,
        usage_context=usage_context,
    )
