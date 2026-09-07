"""Capture what the factory hands every service constructor, untuned.

«expose ≠ change» (spec §1.3) says a workflow with no ``service_tuning``
builds exactly the services it built before. The control tests in
test_harness_controls.py pin a handful of wire values; this module pins the
whole construction: for every provider the factory builds — one case per
model-dependent branch — the service class is replaced by a mock and the
kwargs it receives are serialised. ``golden/untuned_construction.json`` is
that capture taken ONCE from the base commit (``ac8ed47d``, see
test_harness_controls.py for the method), and the test compares today's
capture against it with an explicit allow-list of the declared changes.

Self-contained on purpose: the same file is loaded by path against the base
worktree to produce the golden, so it must not import anything this branch
added (``_transport``, the specs table).
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, patch

# (defining module, class name) for every service class a factory branch
# constructs. A branch may hold the name at module level (patched on the
# factory) or import it locally (patched at the source); both are patched
# with one mock so either style is captured.
SERVICE_CLASSES = [
    ("pipecat.services.deepgram.flux.stt", "DeepgramFluxSTTService"),
    ("pipecat.services.deepgram.stt", "DeepgramSTTService"),
    ("pipecat.services.openai.stt", "OpenAISTTService"),
    ("pipecat.services.openai.stt", "OpenAIRealtimeSTTService"),
    ("pipecat.services.google.stt", "GoogleSTTService"),
    ("pipecat.services.cartesia.stt", "CartesiaSTTService"),
    ("pipecat.services.cartesia.turns.stt", "CartesiaTurnsSTTService"),
    ("pipecat.services.dograh.flux.stt", "DograhFluxSTTService"),
    ("pipecat.services.dograh.stt", "DograhSTTService"),
    ("pipecat.services.sarvam.stt", "SarvamSTTService"),
    ("pipecat.services.speaches.stt", "SpeachesSTTService"),
    ("pipecat.services.huggingface.stt", "HuggingFaceSTTService"),
    ("pipecat.services.assemblyai.stt", "AssemblyAISTTService"),
    ("pipecat.services.gladia.stt", "GladiaSTTService"),
    ("pipecat.services.speechmatics.stt", "SpeechmaticsSTTService"),
    ("pipecat.services.azure.stt", "AzureSTTService"),
    ("pipecat.services.smallest.stt", "SmallestSTTService"),
    ("pipecat.services.elevenlabs.stt", "ElevenLabsRealtimeSTTService"),
    ("pipecat.services.deepgram.tts", "DeepgramTTSService"),
    ("pipecat.services.openai.tts", "OpenAITTSService"),
    ("pipecat.services.google.tts", "GoogleTTSService"),
    ("pipecat.services.elevenlabs.tts", "ElevenLabsTTSService"),
    ("pipecat.services.cartesia.tts", "CartesiaTTSService"),
    ("pipecat.services.inworld.tts", "InworldTTSService"),
    ("pipecat.services.dograh.tts", "DograhTTSService"),
    ("pipecat.services.camb.tts", "CambTTSService"),
    ("pipecat.services.speaches.tts", "SpeachesTTSService"),
    ("pipecat.services.rime.tts", "RimeTTSService"),
    ("pipecat.services.sarvam.tts", "SarvamTTSService"),
    ("api.services.pipecat.minimax_tts", "MiniMaxOwnedSessionTTSService"),
    ("pipecat.services.azure.tts", "AzureTTSService"),
    ("pipecat.services.smallest.tts", "SmallestTTSService"),
    ("pipecat.services.xai.tts", "XAITTSService"),
    ("pipecat.services.lmnt.tts", "LmntTTSService"),
    ("pipecat.services.openai.llm", "OpenAILLMService"),
    ("pipecat.services.groq.llm", "GroqLLMService"),
    ("pipecat.services.openrouter.llm", "OpenRouterLLMService"),
    ("api.services.pipecat.service_factory", "DograhGoogleLLMService"),
    ("api.services.pipecat.service_factory", "DograhGoogleVertexLLMService"),
    ("pipecat.services.azure.llm", "AzureLLMService"),
    ("pipecat.services.dograh.llm", "DograhLLMService"),
    ("pipecat.services.aws.llm", "AWSBedrockLLMService"),
    ("pipecat.services.speaches.llm", "SpeachesLLMService"),
    ("pipecat.services.huggingface.llm", "HuggingFaceLLMService"),
    ("pipecat.services.minimax.llm", "MiniMaxLLMService"),
    ("pipecat.services.sarvam.llm", "SarvamLLMService"),
    ("api.services.pipecat.realtime.openai_realtime", "DograhOpenAIRealtimeLLMService"),
    ("api.services.pipecat.realtime.grok_realtime", "DograhGrokRealtimeLLMService"),
    ("api.services.pipecat.realtime.ultravox_realtime", "DograhUltravoxRealtimeLLMService"),
    ("api.services.pipecat.realtime.gemini_live", "DograhGeminiLiveLLMService"),
    ("api.services.pipecat.realtime.gemini_live_vertex", "DograhGeminiLiveVertexLLMService"),
    ("api.services.pipecat.realtime.azure_realtime", "DograhAzureRealtimeLLMService"),
]  # fmt: skip


def _cfg(section: str, provider: str, **fields) -> SimpleNamespace:
    return SimpleNamespace(**{section: SimpleNamespace(provider=provider, **fields)})


def _stt(provider: str, model: str, language=None, base_url=None, **fields):
    return _cfg(
        "stt",
        provider,
        model=model,
        api_key="test-key",
        language=language,
        base_url=base_url,
        **fields,
    )


def _tts(provider: str, model: str, voice="Name - voice-1", speed=1.0, **fields):
    return _cfg(
        "tts",
        provider,
        model=model,
        voice=voice,
        api_key="test-key",
        speed=speed,
        base_url=fields.pop("base_url", None),
        **fields,
    )


def _realtime(provider: str, model: str, **fields):
    return _cfg("realtime", provider, model=model, api_key="test-key", **fields)


_AUDIO = SimpleNamespace(
    transport_in_sample_rate=16000, transport_out_sample_rate=16000
)
_KEYTERMS = ["Marbella"]
_EL = "https://api.elevenlabs.io"

# case id -> user config (or, for llm, the create_llm_service_from_provider
# kwargs). One case per branch the factory chooses by model or language.
STT_CASES: dict[str, Any] = {
    "stt.deepgram/flux-multi": _stt("deepgram", "flux-general-multi", "es"),
    "stt.deepgram/flux-en": _stt("deepgram", "flux-general-en", "en"),
    "stt.deepgram/nova": _stt("deepgram", "nova-3-general"),
    "stt.openai/segments": _stt("openai", "gpt-4o-transcribe"),
    "stt.openai/segments-language": _stt("openai", "gpt-4o-transcribe", "es"),
    "stt.google": _stt("google", "latest_long", "en-US"),
    "stt.cartesia/ink-whisper": _stt("cartesia", "ink-whisper", "es"),
    "stt.cartesia/ink-2": _stt("cartesia", "ink-2", "es"),
    "stt.dograh/flux": _stt("dograh", "ignored", "en"),
    "stt.dograh/non-flux": _stt("dograh", "nova-3", "zz"),
    "stt.sarvam": _stt("sarvam", "saarika:v2", "hi-IN"),
    "stt.speaches": _stt("speaches", "whisper-1", "en", base_url="https://api.example"),
    "stt.huggingface": _stt("huggingface", "openai/whisper-large-v3"),
    "stt.assemblyai": _stt("assemblyai", "universal-streaming", "en"),
    "stt.gladia": _stt("gladia", "solaria-1", "en"),
    "stt.speechmatics": _stt("speechmatics", "enhanced", "en"),
    "stt.azure_speech": _stt("azure_speech", "default", "en-US"),
    "stt.smallest": _stt("smallest", "lightning", "en"),
    "stt.elevenlabs": _stt("elevenlabs", "scribe_v2_realtime", "es", base_url=_EL),
}
TTS_CASES: dict[str, Any] = {
    "tts.deepgram": _tts("deepgram", "aura-2-thalia-en", voice="aura-2-thalia-en"),
    "tts.openai": _tts("openai", "gpt-4o-mini-tts", voice="alloy"),
    "tts.google": _tts("google", "chirp_3_hd", voice="en-US-Chirp3-HD-Charon", speed=1.2, language="es-ES"),
    "tts.elevenlabs": _tts("elevenlabs", "eleven_flash_v2_5", voice="Elena - abc", base_url=_EL),
    "tts.cartesia": _tts("cartesia", "sonic-2", language="es"),
    "tts.cartesia/speed-volume": _tts("cartesia", "sonic-2", speed=1.2, volume=0.8, language="es"),
    "tts.inworld": _tts("inworld", "inworld-tts-1", voice="Ashley", speed=1.1, language="es-ES"),
    "tts.dograh": _tts("dograh", "aura-2", speed=1.1),
    "tts.camb": _tts("camb", "mars-flash", voice="147320", language="es-es"),
    "tts.speaches": _tts("speaches", "kokoro", voice="fettah", speed=1.1, base_url="https://api.example"),
    "tts.rime": _tts("rime", "arcana", voice="luna", speed=1.2, language="es"),
    "tts.sarvam": _tts("sarvam", "bulbul:v2", voice="Anushka", speed=1.2, language="hi-IN"),
    "tts.minimax": _tts("minimax", "speech-02-hd", voice="Wise_Woman", speed=1.1, group_id="grp-1"),
    "tts.azure_speech": _tts("azure_speech", "neural", voice="es-ES-ElviraNeural", speed=1.25, language="es-ES"),
    "tts.smallest": _tts("smallest", "lightning-v2", voice="emily", speed=1.1, language="es"),
    "tts.xai": _tts("xai", "grok-tts", voice="eve", language="es"),
    "tts.xai/speed": _tts("xai", "grok-tts", voice="eve", speed=1.3, language="es"),
    "tts.lmnt": _tts("lmnt", "aurora", voice="lily", language="es"),
}  # fmt: skip
LLM_CASES: dict[str, dict[str, Any]] = {
    "llm.openai/gpt-4.1": {"provider": "openai", "model": "gpt-4.1"},
    "llm.openai/gpt-5": {"provider": "openai", "model": "gpt-5-mini"},
    "llm.openai/gpt-5-chat": {"provider": "openai", "model": "gpt-5-chat-latest"},
    "llm.openai/base_url": {"provider": "openai", "model": "gpt-4.1", "base_url": "https://api.example/v1"},
    "llm.atlascloud": {"provider": "atlascloud", "model": "some-model", "base_url": "https://api.example/v1"},
    "llm.groq": {"provider": "groq", "model": "llama-3.3-70b"},
    "llm.openrouter/gpt-4.1": {"provider": "openrouter", "model": "openai/gpt-4.1"},
    "llm.openrouter/gpt-5": {"provider": "openrouter", "model": "openai/gpt-5-mini"},
    "llm.google": {"provider": "google", "model": "gemini-2.5-flash"},
    "llm.google/migrated": {"provider": "google", "model": "gemini-2.0-flash"},
    "llm.google_vertex": {"provider": "google_vertex", "model": "gemini-2.5-flash", "project_id": "p", "credentials": "c"},
    "llm.azure/gpt-4.1": {"provider": "azure", "model": "gpt-4.1", "endpoint": "https://x.openai.azure.com"},
    "llm.azure/gpt-5": {"provider": "azure", "model": "gpt-5-mini", "endpoint": "https://x.openai.azure.com"},
    "llm.dograh": {"provider": "dograh", "model": "gpt-4.1", "correlation_id": "corr-1", "usage_context": "conversation"},
    "llm.aws_bedrock": {"provider": "aws_bedrock", "model": "anthropic.claude", "aws_access_key": "a", "aws_secret_key": "s", "aws_region": "eu-west-1"},
    "llm.speaches": {"provider": "speaches", "model": "qwen", "base_url": "https://api.example/v1"},
    "llm.huggingface": {"provider": "huggingface", "model": "meta-llama/x", "bill_to": "org"},
    "llm.minimax": {"provider": "minimax", "model": "MiniMax-M1"},
    "llm.minimax/temperature": {"provider": "minimax", "model": "MiniMax-M1", "temperature": 0.4},
    "llm.sarvam": {"provider": "sarvam", "model": "sarvam-m"},
}  # fmt: skip
REALTIME_CASES: dict[str, Any] = {
    "realtime.openai_realtime": _realtime("openai_realtime", "gpt-realtime", voice="alloy", language="es"),
    "realtime.grok_realtime": _realtime("grok_realtime", "grok-realtime", voice="Ara", language=None),
    "realtime.ultravox_realtime": _realtime("ultravox_realtime", "ultravox-v0.6", voice="Jessica", language=None),
    "realtime.google_realtime": _realtime("google_realtime", "gemini-live", voice="Puck", language="es", temperature=0.7),
    "realtime.google_vertex_realtime": _realtime("google_vertex_realtime", "gemini-live", voice=None, language=None, project_id="p", location=None, credentials="c"),
    "realtime.azure_realtime": _realtime("azure_realtime", "gpt-realtime", voice="alloy", language="es", endpoint="https://x.openai.azure.com", api_version=None),
}  # fmt: skip


def _serial(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, MagicMock):
        return "<mock>"
    if type(value).__name__ == "_NotGiven":
        return "NOT_GIVEN"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__class__": type(value).__name__,
            **{
                f.name: _serial(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    if hasattr(value, "model_dump"):
        return {"__class__": type(value).__name__, **_serial(value.model_dump())}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    return f"<{type(value).__name__}>"


def _builders(factory) -> dict[str, Callable[[], Any]]:
    out: dict[str, Callable[[], Any]] = {}
    for case, cfg in STT_CASES.items():
        out[case] = lambda cfg=cfg: factory.create_stt_service(
            cfg, _AUDIO, keyterms=_KEYTERMS, correlation_id="corr-1"
        )
    for case, cfg in TTS_CASES.items():
        out[case] = lambda cfg=cfg: factory.create_tts_service(
            cfg, _AUDIO, correlation_id="corr-1"
        )
    for case, kw in LLM_CASES.items():
        out[case] = lambda kw=kw: factory.create_llm_service_from_provider(
            api_key="k", **kw
        )
    for case, cfg in REALTIME_CASES.items():
        out[case] = lambda cfg=cfg: factory.create_realtime_llm_service(cfg, _AUDIO)
    return out


def capture() -> dict[str, dict[str, Any]]:
    """``{case id: {"class": name, "kwargs": {...}}}`` for every case."""
    factory = importlib.import_module("api.services.pipecat.service_factory")
    out: dict[str, dict[str, Any]] = {}
    # Import everything before patching anything: a module imported after
    # its base class was mocked would subclass the mock.
    modules = {m: importlib.import_module(m) for m, _ in SERVICE_CLASSES}
    with ExitStack() as stack:
        mocks: dict[str, MagicMock] = {}
        for module_name, name in SERVICE_CLASSES:
            mock = MagicMock(name=name)
            module = modules[module_name]
            # A branch may build the settings through the class
            # (``Cls.Settings(...)``); keep that a real object, not a mock.
            real = getattr(module, name, None)
            if real is not None and hasattr(real, "Settings"):
                mock.Settings = real.Settings
            if hasattr(module, name):
                stack.enter_context(patch.object(module, name, mock))
            if module is not factory and hasattr(factory, name):
                stack.enter_context(patch.object(factory, name, mock))
            mocks[name] = mock
        # MiniMax opens its own session before constructing the service.
        stack.enter_context(patch.object(factory.aiohttp, "ClientSession"))
        for case, build in _builders(factory).items():
            for mock in mocks.values():
                mock.reset_mock()
            build()
            called = [(name, mock) for name, mock in mocks.items() if mock.called]
            assert len(called) == 1, (case, [name for name, _ in called])
            name, mock = called[0]
            assert not mock.call_args.args, case  # every branch passes kwargs only
            out[case] = {"class": name, "kwargs": _serial(mock.call_args.kwargs)}
    return out


if __name__ == "__main__":  # pragma: no cover - golden generation
    import json
    import sys

    json.dump(capture(), sys.stdout, indent=2, sort_keys=True)
