"""Every URL-checking service build, with the REAL provider classes, through build_resolving_service_hosts on a
running loop with the guard enforced and a cold cache (review r3, RUL-r3-12 (3)).

What it proves: each cold host is looked up off the loop, the build runs once more per cold host and no more, and
what the builder does before its URL checks (sessions, failure reports) is not repeated by the re-run. No provider
class is patched: the factory tests mock them, so they cannot say whether a real constructor tolerates a re-run.
"""

import ast
import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pipecat.processors.frame_processor import FrameProcessor

from api.services.gen_ai.embedding.azure_openai_service import (
    AzureOpenAIEmbeddingService,
)
from api.services.gen_ai.embedding.base import BaseEmbeddingService
from api.services.gen_ai.embedding.openai_service import OpenAIEmbeddingService
from api.services.pipecat import service_factory
from api.services.security.egress_guard import build_resolving_service_hosts
from api.tests.service_tuning._golden import (
    _AUDIO,
    _KEYTERMS,
    LLM_CASES,
    REALTIME_CASES,
    STT_CASES,
    TTS_CASES,
    _stt,
    _tts,
)

PUBLIC_IP = "93.184.216.34"
_DG_EU = "https://api.eu.deepgram.com"
_FLUX_URL = "wss://flux.example.com/v2/listen"


def _stt_build(cfg, tuning=None):
    return (
        service_factory.create_stt_service,
        (cfg, _AUDIO),
        {"keyterms": _KEYTERMS, "correlation_id": "corr-1", "tuning": tuning},
    )


def _tts_build(cfg):
    return (
        service_factory.create_tts_service,
        (cfg, _AUDIO),
        {"correlation_id": "corr-1"},
    )


def _llm_build(kwargs):
    return (
        service_factory.create_llm_service_from_provider,
        (),
        {"api_key": "k", **kwargs},
    )


def _realtime_build(cfg):
    return (service_factory.create_realtime_llm_service, (cfg, _AUDIO), {})


# One case per URL check site in the create_* functions (test_every_check_site_has_a_case keeps the count honest).
FACTORY_CASES = {
    "stt.deepgram/base_url": _stt_build(_stt("deepgram", "nova-3-general", base_url=_DG_EU)),
    "stt.deepgram/flux-ctor-url": _stt_build(
        _stt("deepgram", "flux-general-en", "en", base_url=_DG_EU),
        tuning={"stt": {"deepgram": {"ctor": {"url": _FLUX_URL}}}},
    ),
    "stt.openai/realtime": _stt_build(
        _stt("openai", "gpt-realtime-whisper", "es", base_url="https://stt.example.com/v1"),
        tuning={"stt": {"openai": {"options": {"api": "realtime"}}}},
    ),
    "stt.openai/base_url": _stt_build(
        _stt("openai", "gpt-4o-transcribe", base_url="https://stt.example.com/v1")
    ),
    "stt.speaches": _stt_build(STT_CASES["stt.speaches"]),
    "stt.huggingface/default": _stt_build(STT_CASES["stt.huggingface"]),
    "stt.elevenlabs": _stt_build(STT_CASES["stt.elevenlabs"]),
    "tts.deepgram/base_url": _tts_build(
        _tts("deepgram", "aura-2-thalia-en", voice="aura-2-thalia-en", base_url=_DG_EU)
    ),
    "tts.openai/base_url": _tts_build(
        _tts("openai", "gpt-4o-mini-tts", voice="alloy", base_url="https://tts.example.com/v1")
    ),
    "tts.elevenlabs": _tts_build(TTS_CASES["tts.elevenlabs"]),
    "tts.speaches": _tts_build(TTS_CASES["tts.speaches"]),
    "tts.minimax/default": _tts_build(TTS_CASES["tts.minimax"]),
    "llm.openai/base_url": _llm_build(LLM_CASES["llm.openai/base_url"]),
    "llm.atlascloud": _llm_build(LLM_CASES["llm.atlascloud"]),
    "llm.openrouter/base_url": _llm_build(
        {"provider": "openrouter", "model": "openai/gpt-4.1", "base_url": "https://or.example.com/v1"}
    ),
    "llm.azure": _llm_build(LLM_CASES["llm.azure/gpt-4.1"]),
    "llm.speaches": _llm_build(LLM_CASES["llm.speaches"]),
    "llm.huggingface/default": _llm_build(LLM_CASES["llm.huggingface"]),
    "llm.minimax/default": _llm_build(LLM_CASES["llm.minimax"]),
    # The golden's "sarvam-m" only suits a mocked class; the real one validates the model.
    "llm.sarvam/default": _llm_build({**LLM_CASES["llm.sarvam"], "model": "sarvam-105b"}),
    "realtime.azure_realtime": _realtime_build(REALTIME_CASES["realtime.azure_realtime"]),
}  # fmt: skip
EMBEDDING_CASES = {
    "embeddings.openai": (
        OpenAIEmbeddingService,
        (),
        {"db_client": SimpleNamespace(), "api_key": "k", "base_url": "https://emb.example.com/v1"},
    ),
    "embeddings.azure": (
        AzureOpenAIEmbeddingService,
        (),
        {"db_client": SimpleNamespace(), "api_key": "k", "endpoint": "https://x.openai.azure.com"},
    ),
}  # fmt: skip
# The checks that run inside the create_* functions, directly or through the Deepgram base URL helper.
_CHECKS = frozenset(
    {"_validate_runtime_service_url", "_validate_deepgram_url", "_deepgram_base_url"}
)


class _Spies:
    """Wraps, never replaces: DNS answers, aiohttp sessions and failure reports, with the thread of each lookup."""

    def __init__(self, monkeypatch):
        self.lookups: list[tuple[str, int, int]] = []
        self.sessions: list[aiohttp.ClientSession] = []
        self.failures: list = []

        def getaddrinfo(host, port, *a, **k):
            self.lookups.append((host, port, threading.get_ident()))
            return [(socket.AF_INET, 1, 6, "", (PUBLIC_IP, port))]

        monkeypatch.setattr("socket.getaddrinfo", getaddrinfo)
        real_init = aiohttp.ClientSession.__init__

        def session_init(session, *a, **k):
            self.sessions.append(session)
            real_init(session, *a, **k)

        monkeypatch.setattr(aiohttp.ClientSession, "__init__", session_init)
        real_log_failure = service_factory.log_failure

        def log_failure(failure, *a, **k):
            self.failures.append(failure)
            return real_log_failure(failure, *a, **k)

        monkeypatch.setattr(service_factory, "log_failure", log_failure)
        monkeypatch.setattr("api.errors.failure.log_failure", log_failure)

    async def close(self):
        for session in self.sessions:
            await session.close()


async def _build(builder, args, kwargs):
    attempts = []

    def counted(*a, **k):
        attempts.append(1)
        return builder(*a, **k)

    service = await build_resolving_service_hosts(counted, *args, **kwargs)
    return service, len(attempts)


@pytest.mark.parametrize(
    "case", [*FACTORY_CASES, *EMBEDDING_CASES], ids=lambda case: case
)
async def test_a_real_service_is_built_with_its_cold_hosts_resolved_off_the_loop(
    enforced, monkeypatch, case
):
    spies = _Spies(monkeypatch)
    builder, args, kwargs = FACTORY_CASES.get(case) or EMBEDDING_CASES[case]
    try:
        service, attempts = await _build(builder, args, kwargs)
    finally:
        await spies.close()
    expected = BaseEmbeddingService if case in EMBEDDING_CASES else FrameProcessor
    assert isinstance(service, expected), type(service)
    cold_hosts = {(host, port) for host, port, _ in spies.lookups}
    assert cold_hosts, "the build checked no URL"
    assert threading.get_ident() not in {thread for *_, thread in spies.lookups}
    assert attempts == 1 + len(cold_hosts)
    # A re-run does not open a second session: every check comes before the session (MiniMax opens one).
    assert (
        len(spies.sessions) == 1
        if case == "tts.minimax/default"
        else len(spies.sessions) <= 1
    )
    assert spies.failures == []


def test_every_check_site_has_a_case():
    tree = ast.parse(Path(service_factory.__file__).read_text(encoding="utf-8"))
    sites = [
        node.lineno
        for function in tree.body
        if isinstance(function, ast.FunctionDef) and function.name.startswith("create_")
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CHECKS
    ]
    assert len(FACTORY_CASES) >= len(sites), (
        f"{len(sites)} URL checks in the create_* functions, {len(FACTORY_CASES)} cases: add one per new site"
    )


def _service_account_json() -> str:
    """A throwaway service account: its RSA key is generated here and never leaves the test."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "p",
            "private_key_id": "k",
            "private_key": pem,
            "client_email": "svc@p.iam.gserviceaccount.com",
            "client_id": "1",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


@pytest.mark.parametrize("role", ["stt", "tts"])
async def test_google_services_build_on_the_loop_through_the_wrapper(
    enforced, monkeypatch, role
):
    # The gRPC asyncio clients these constructors create need the running loop (review r3, RUL-r3-1 refuted).
    credentials = _service_account_json()
    if role == "stt":
        build = _stt_build(
            _stt("google", "latest_long", "en-US", credentials=credentials)
        )
    else:
        build = _tts_build(
            _tts(
                "google",
                "chirp_3_hd",
                voice="en-US-Chirp3-HD-Charon",
                language="es-ES",
                credentials=credentials,
            )
        )
    service, attempts = await _build(*build)
    assert isinstance(service, FrameProcessor)
    assert type(service).__name__ == (
        "GoogleSTTService" if role == "stt" else "GoogleTTSService"
    )
    assert attempts == 1
