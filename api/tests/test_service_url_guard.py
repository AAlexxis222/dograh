"""VOZ-N0-41: service URLs honour URL_GUARD_ENFORCE (VOZ-AC-B5-91, VOZ-AC-B0-30)."""

import asyncio
import contextlib
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from secrets import token_hex
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import httpx2
import pytest
from fastapi import HTTPException
from loguru import logger

from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.registry import (
    DeepgramSTTConfiguration,
    ElevenlabsSTTConfiguration,
    ElevenlabsTTSConfiguration,
    OpenAILLMService,
    ServiceProviders,
    SpeachesLLMConfiguration,
    SpeachesSTTConfiguration,
    SpeachesTTSConfiguration,
)
from api.services.gen_ai.embedding.azure_openai_service import (
    AzureOpenAIEmbeddingService,
)
from api.services.gen_ai.embedding.openai_service import OpenAIEmbeddingService
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.service_factory import (
    create_llm_service_from_provider,
    create_stt_service,
    create_tts_service,
)
from api.services.security import egress_guard
from api.services.security.egress_guard import (
    EgressBlocked,
    EgressTarget,
    _ResolutionCache,
    check_service_url,
    check_tool_url,
)
from pipecat.services.huggingface.llm import HuggingFaceLLMService

PUBLIC_IP = "93.184.216.34"  # what conftest's public_dns resolves every name to
IMDS = "http://169.254.169.254/latest/meta-data/"
KEY = token_hex(12)  # generated per run
AUDIO = AudioConfig(transport_in_sample_rate=16000, transport_out_sample_rate=16000)

pytestmark = pytest.mark.usefixtures("public_dns")


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data",
        "http://redis:6379",
        "http://minio:9000/v1",
        "http://postgres:5432",
        "http://10.0.0.10/v1",
        "http://localhost:11434/v1",
    ],
)
def test_factory_refuses_internal_base_url_in_a_cell(enforced, url):
    with pytest.raises(
        HTTPException
    ) as e:  # red today: DID NOT RAISE (oss returns early)
        create_llm_service_from_provider(
            provider=ServiceProviders.OPENAI.value,
            model="gpt-4.1",
            api_key=KEY,
            base_url=url,
        )
    assert e.value.status_code == 400 and "tool_url_blocked" in e.value.detail


def test_validator_refuses_internal_url_in_a_cell(enforced):
    result = UserConfigurationValidator()._validate_service(
        SpeachesLLMConfiguration(), "llm"
    )
    assert (
        len(result) == 1 and "tool_url_blocked" in result[0]["message"]
    )  # red today: result == []


def test_embeddings_refuse_internal_url_in_a_cell(enforced):
    with pytest.raises(
        ValueError, match="tool_url_blocked"
    ):  # red today: DID NOT RAISE
        OpenAIEmbeddingService(
            db_client=SimpleNamespace(), api_key=KEY, base_url="http://10.0.0.10/v1"
        )


def test_azure_embeddings_refuse_internal_endpoint_in_a_cell(enforced):
    with pytest.raises(ValueError, match="tool_url_blocked"):
        AzureOpenAIEmbeddingService(
            db_client=SimpleNamespace(), api_key=KEY, endpoint="http://10.0.0.10"
        )


def test_azure_embeddings_keep_internal_endpoint_with_guard_off_in_oss(monkeypatch):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    AzureOpenAIEmbeddingService(
        db_client=SimpleNamespace(), api_key=KEY, endpoint="http://10.0.0.10"
    )


def test_control_flag_off_oss_keeps_local_urls(
    monkeypatch,
):  # regression (today's OSS behaviour)
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    create_llm_service_from_provider(
        provider=ServiceProviders.OPENAI.value,
        model="gpt-4.1",
        api_key=KEY,
        base_url="http://localhost:11434/v1",
    )


@pytest.mark.parametrize("url", ["wss://el.example.com/x", "ws://el.example.com/x"])
def test_check_service_url_accepts_ws_and_wss(enforced, url):
    target = check_service_url(url, field="base_url")
    assert target == EgressTarget(host="el.example.com", address=PUBLIC_IP)


def test_check_service_url_refuses_other_schemes(enforced):
    with pytest.raises(EgressBlocked) as e:
        check_service_url("ftp://example.com", field="base_url")
    assert e.value.reason == "scheme 'ftp' is not one of http, https, ws, wss"


@pytest.mark.parametrize("url", [None, ""])
def test_check_service_url_refuses_empty(enforced, url):
    with pytest.raises(EgressBlocked) as e:
        check_service_url(url, field="base_url")
    assert e.value.reason == "the URL has no host"


@pytest.mark.parametrize(
    "url, port", [("wss://el.example.com/x", 443), ("ws://el.example.com/x", 80)]
)
def test_check_service_url_default_port_follows_the_scheme(
    enforced, monkeypatch, url, port
):
    seen = []

    def fake_getaddrinfo(host, requested_port, *a, **k):
        seen.append((host, requested_port))
        return [(socket.AF_INET, 1, 6, "", (PUBLIC_IP, requested_port))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    check_service_url(url, field="base_url")
    assert seen == [("el.example.com", port)]


def test_check_service_url_guard_off_returns_empty_target(monkeypatch):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    assert (
        check_service_url("http://localhost:11434/v1", field="base_url")
        == EgressTarget()
    )


def test_tool_urls_still_refuse_ws(enforced):
    with pytest.raises(EgressBlocked):
        check_tool_url("wss://example.com")


# --- no DNS lookup on the event loop (live calls share it) ---


def _lookup_threads(monkeypatch) -> list[int]:
    """Record the thread of every DNS lookup; every name resolves to the public address."""
    threads = []

    def fake_getaddrinfo(host, port, *a, **k):
        threads.append(threading.get_ident())
        return [(socket.AF_INET, 1, 6, "", (PUBLIC_IP, port))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    return threads


async def test_validator_resolves_no_service_host_on_the_event_loop(
    enforced, monkeypatch
):
    threads = _lookup_threads(monkeypatch)
    configuration = SimpleNamespace(
        llm=SpeachesLLMConfiguration(base_url="https://llm.example.com/v1"),
        stt=SpeachesSTTConfiguration(base_url="https://stt.example.com/v1"),
        tts=SpeachesTTSConfiguration(base_url="https://tts.example.com/v1"),
        embeddings=None,
        realtime=None,
        is_realtime=False,
    )
    await UserConfigurationValidator().validate(configuration)
    assert threads  # the hosts were checked
    assert threading.get_ident() not in threads  # red today: all three on the loop


class _LoopWatch:
    """The longest time, in seconds, the event loop went without running a ticker task of the test."""

    def __init__(self):
        self.longest = 0.0
        self._last = time.monotonic()
        self._ticker = None

    async def _tick(self):
        while True:
            await asyncio.sleep(0.01)
            self._note()

    def _note(self):
        now = time.monotonic()
        self.longest = max(self.longest, now - self._last)
        self._last = now

    async def __aenter__(self):
        self._ticker = asyncio.create_task(self._tick())
        await asyncio.sleep(0.02)
        self.longest, self._last = 0.0, time.monotonic()
        return self

    async def __aexit__(self, *exc):
        self._note()  # a block that ended the body is counted although the ticker did not run after it
        self._ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._ticker


@contextlib.asynccontextmanager
async def _dripping_server(seconds: float, every: float):
    """A local HTTP server that answers every request one byte of a response every ``every`` s for ``seconds``."""
    handlers = set()
    response = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\n\r\n{}"

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        await reader.readuntil(b"\r\n\r\n")
        for byte in response[: int(seconds / every)]:
            if writer.is_closing():
                break
            writer.write(bytes([byte]))
            await asyncio.sleep(every)
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
    finally:
        for handler in handlers:
            handler.cancel()
        server.close()
        server.close_clients()
        await server.wait_closed()


async def test_a_dripping_tenant_server_holds_neither_the_loop_nor_the_save(
    monkeypatch,
):
    # Review r3 (9ec1ad5d6899, 71e822e75122): the key check's timeout bounded each read, not the request, and it
    # ran on the event loop of the async validate(): a server that sends a byte inside every window held the loop.
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    monkeypatch.setattr(
        "api.services.configuration.check_validity._KEY_CHECK_TIMEOUT_SECONDS", 1.0
    )
    async with _dripping_server(seconds=5.0, every=0.3) as base_url:
        configuration = SimpleNamespace(
            llm=OpenAILLMService(api_key=KEY, base_url=base_url),
            stt=None,
            tts=None,
            embeddings=None,
            realtime=None,
            is_realtime=False,
        )
        started = time.monotonic()
        async with _LoopWatch() as watch:
            with pytest.raises(ValueError) as e:
                await UserConfigurationValidator().validate(configuration)
        elapsed = time.monotonic() - started
    assert watch.longest < 0.2  # red today: the loop is held while the check reads
    assert elapsed < 2.5  # red without the request deadline: the check reads for 5 s
    (llm,) = [status for status in e.value.args[0] if status["model"] == "llm"]
    assert "did not answer" in llm["message"]


async def test_an_async_build_with_a_cold_host_does_not_hold_the_event_loop(
    enforced, monkeypatch
):
    # Review r3 (1cedb2e1c0d4, 4e8385aab6c4, 6a4402c01244): a host that no list of the configuration's URLs names,
    # here the HuggingFace default the factory supplies, met by a service build an async caller runs on the loop.
    from api.services.workflow.qa.llm_config import create_qa_llm_service

    def slow_getaddrinfo(host, port, *a, **k):
        time.sleep(1.0)
        return [(socket.AF_INET, 1, 6, "", (PUBLIC_IP, port))]

    monkeypatch.setattr("socket.getaddrinfo", slow_getaddrinfo)
    qa = SimpleNamespace(
        qa_use_workflow_llm=False,
        qa_provider=ServiceProviders.HUGGINGFACE.value,
        qa_model="meta-llama/Llama-3.1-8B-Instruct",
        qa_api_key=f"hf_{KEY}",
        qa_endpoint=None,
    )
    async with _LoopWatch() as watch:
        try:
            outcome = await create_qa_llm_service(qa, None)
        except HTTPException as e:
            outcome = e
    # red today: the loop waits 1 s for the lookup, then the build is refused
    assert watch.longest < 0.2, f"the event loop was held {watch.longest:.2f} s"
    llm, _ = outcome
    assert isinstance(llm, HuggingFaceLLMService)


async def test_factory_build_after_the_prewarm_resolves_nothing_on_the_event_loop(
    enforced, monkeypatch
):
    from api.services.pipecat.service_factory import (
        create_llm_service,
        prewarm_config_service_urls,
    )

    threads = _lookup_threads(monkeypatch)
    user_config = SimpleNamespace(
        llm=OpenAILLMService(api_key=KEY, base_url="https://llm.example.com/v1"),
        stt=None,
        tts=None,
        realtime=None,
    )
    await prewarm_config_service_urls(user_config)
    create_llm_service(user_config)
    assert threads
    assert threading.get_ident() not in threads


async def test_an_on_loop_lookup_that_does_not_answer_is_refused_within_the_bound(
    enforced, monkeypatch
):
    # Review r2 (8cee9264e264, 4e8385aab6c4, 1cedb2e1c0d4): a URL no prewarm covers, here the HuggingFace default
    # the factory supplies itself, checked on the loop with a resolver that does not answer.
    gate = threading.Event()

    def hung_getaddrinfo(host, port, *a, **k):
        gate.wait(4)
        return [(socket.AF_INET, 1, 6, "", (PUBLIC_IP, port))]

    monkeypatch.setattr("socket.getaddrinfo", hung_getaddrinfo)
    started = time.monotonic()
    try:
        with pytest.raises(HTTPException) as e:  # red today: the loop waits 4 s
            create_llm_service_from_provider(
                provider=ServiceProviders.HUGGINGFACE.value,
                model="meta-llama/Llama-3.1-8B-Instruct",
                api_key=f"hf_{KEY}",
            )
    finally:
        elapsed = time.monotonic() - started
        gate.set()
    assert elapsed < 2.5
    assert e.value.status_code == 400 and "could not be resolved" in e.value.detail


async def test_a_service_built_on_the_loop_without_the_prewarm_is_logged(
    enforced, monkeypatch
):
    # Review r2 (47dd71ea571f): a caller that forgot the prewarm is visible, and its lookup still left the loop.
    from api.services.pipecat.service_factory import create_llm_service

    threads = _lookup_threads(monkeypatch)
    user_config = SimpleNamespace(
        llm=OpenAILLMService(api_key=KEY, base_url="https://llm.example.com/v1"),
        stt=None,
        tts=None,
        realtime=None,
    )
    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), level="WARNING")
    try:
        create_llm_service(user_config)
    finally:
        logger.remove(sink)
    assert any("prewarm missing" in ln and "llm.example.com" in ln for ln in lines)
    assert threads and threading.get_ident() not in threads


# --- the validator's key check connects where the guard looked ---


class SyncWire:
    """Stands in for the network under every synchronous client, of both httpx lines (the openai SDK builds its
    default client on httpx2): records the requests put on the wire and answers each with one fixed response."""

    def __init__(self, status: int, headers: dict[str, str]):
        self.requests = []
        self._status = status
        self._headers = headers

    def patch(self, monkeypatch):
        for line in (httpx, httpx2):

            def handle(request, line=line):
                self.requests.append(request)
                return line.Response(self._status, headers=self._headers)

            mock = line.MockTransport(handle)
            for target in ("HTTPTransport", "_client.HTTPTransport"):
                monkeypatch.setattr(
                    f"{line.__name__}.{target}", lambda mock=mock, **kw: mock
                )


def test_validator_key_check_is_pinned_to_the_validated_address(enforced, monkeypatch):
    # The public base_url answers with a redirect to the metadata endpoint.
    wire = SyncWire(302, {"location": IMDS})
    wire.patch(monkeypatch)
    config = OpenAILLMService(api_key=KEY, base_url="https://llm.example.com/v1")
    result = UserConfigurationValidator()._validate_service(config, "llm")
    # red today: the SDK resolves the name again and follows the 302 to 169.254.169.254
    assert [request.url.host for request in wire.requests] == [PUBLIC_IP]
    assert len(result) == 1  # the unfollowed redirect is an error, not a valid key


@pytest.mark.parametrize("guard", ["enforced", "off"])
def test_validator_key_check_waits_10_s_at_most_and_does_not_retry(
    monkeypatch, request, guard
):
    # Review r2 (71e822e75122): the check runs inside the async validate(), so a tenant URL that never answers
    # held the worker's loop for the SDK's 600 s read timeout, three times.
    if guard == "enforced":
        request.getfixturevalue("enforced")
    else:
        monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
        monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    wire = SyncWire(500, {})
    wire.patch(monkeypatch)
    config = OpenAILLMService(api_key=KEY, base_url="https://llm.example.com/v1")
    result = UserConfigurationValidator()._validate_service(config, "llm")
    assert len(result) == 1
    (sent,) = wire.requests  # red today: two retries follow the 500
    assert sent.extensions["timeout"]["read"] == 10.0  # red today: 600


# --- the resolution cache: answers, never verdicts ---


class _Resolver:
    """A DNS that answers what the test sets, counting lookups; None makes the lookup fail."""

    def __init__(self, monkeypatch, address):
        self.address = address
        self.lookups = 0
        monkeypatch.setattr("socket.getaddrinfo", self)

    def __call__(self, host, port, *a, **k):
        self.lookups += 1
        if self.address is None:
            raise socket.gaierror(socket.EAI_AGAIN, "temporary failure")
        return [(socket.AF_INET, 1, 6, "", (self.address, port))]


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _cache(**settings) -> tuple[_ResolutionCache, _Clock]:
    """A cache of the test's own on a hand-moved clock; ``settings`` override the service check's defaults."""
    clock = _Clock()
    return _ResolutionCache(clock=clock, **settings), clock


def test_a_host_that_turns_internal_is_answered_anew_once_its_entry_expires(
    monkeypatch,
):
    cache, clock = _cache()
    dns = _Resolver(monkeypatch, PUBLIC_IP)
    assert cache.resolve("llm.example.com", 443) == (PUBLIC_IP,)
    dns.address = "10.0.0.10"
    clock.now += 59
    assert cache.resolve("llm.example.com", 443) == (PUBLIC_IP,)  # cached answer
    assert dns.lookups == 1
    clock.now += 2
    # The new answer, not a cached verdict: the check judges it and refuses the internal address.
    assert cache.resolve("llm.example.com", 443) == ("10.0.0.10",)
    assert dns.lookups == 2


def test_a_failed_lookup_is_cached_and_refused_again(enforced, monkeypatch):
    dns = _Resolver(monkeypatch, None)
    for _ in range(2):
        with pytest.raises(EgressBlocked) as e:
            check_service_url("https://llm.example.com/v1", field="base_url")
        # The refusal says a resolver failure may be temporary (review r1, 98e6b3bd0ccb).
        assert e.value.reason == (
            "the host name could not be resolved (DNS failure; retry if the host is correct)"
        )
    assert dns.lookups == 1


def test_the_cache_keeps_at_most_its_bound(monkeypatch):
    cache, _ = _cache(size=2)
    dns = _Resolver(monkeypatch, PUBLIC_IP)
    for host in ("h0.example.com", "h1.example.com", "h2.example.com"):
        cache.resolve(host, 443)
    cache.resolve("h1.example.com", 443)
    cache.resolve("h2.example.com", 443)
    assert dns.lookups == 3  # the two most recent are still held
    cache.resolve("h0.example.com", 443)
    assert dns.lookups == 4  # the least recently used went


async def test_prewarm_swallows_a_refusal_and_the_check_reports_it(
    enforced, monkeypatch
):
    dns = _Resolver(monkeypatch, "10.0.0.10")
    await egress_guard.prewarm_service_urls(["https://llm.example.com/v1"])
    with pytest.raises(EgressBlocked, match="internal address"):
        check_service_url("https://llm.example.com/v1", field="base_url")
    assert dns.lookups == 1


async def test_prewarm_with_the_guard_off_resolves_nothing(monkeypatch):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    dns = _Resolver(monkeypatch, PUBLIC_IP)
    await egress_guard.prewarm_service_urls(["https://llm.example.com/v1"])
    assert dns.lookups == 0  # and with no lookup, nothing was cached


# --- the guard judges the exact URL the client uses (review r1 bf601ab39259, aa5d2aac9673; r2 9f7d8d355b63) ---


@pytest.mark.parametrize(
    "config, role",
    [
        # Deepgram's documented regional switch: its client gets https://api.eu.deepgram.com.
        (DeepgramSTTConfiguration(api_key=KEY, base_url="api.eu.deepgram.com"), "stt"),
        # Pipecat's ElevenLabs STT builds wss://{host}/... from the bare host.
        (
            ElevenlabsSTTConfiguration(
                api_key=KEY, base_url="api.eu.residency.elevenlabs.io"
            ),
            "stt",
        ),
        (
            ElevenlabsTTSConfiguration(
                api_key=KEY, voice="v", base_url=" https://api.elevenlabs.io "
            ),
            "tts",
        ),
    ],
)
def test_a_bare_or_padded_value_the_runtime_accepts_is_saved(
    enforced, monkeypatch, config, role
):
    monkeypatch.setattr(
        UserConfigurationValidator, "_check_deepgram_api_key", lambda *a: True
    )
    assert UserConfigurationValidator()._validate_service(config, role) == []


def test_a_bare_elevenlabs_stt_host_builds_in_a_cell(enforced):
    user_config = SimpleNamespace(
        stt=SimpleNamespace(
            provider=ServiceProviders.ELEVENLABS.value,
            api_key=KEY,
            model="scribe_v2_realtime",
            language="en",
            base_url="api.eu.residency.elevenlabs.io",
        )
    )
    with patch(
        "api.services.pipecat.service_factory.ElevenLabsRealtimeSTTService"
    ) as stt_service:
        create_stt_service(user_config, AUDIO)
    assert stt_service.call_args.kwargs["base_url"] == "api.eu.residency.elevenlabs.io"


def test_a_padded_elevenlabs_tts_url_builds_in_a_cell(enforced):
    user_config = SimpleNamespace(
        tts=SimpleNamespace(
            provider=ServiceProviders.ELEVENLABS.value,
            api_key=KEY,
            model="eleven_flash_v2_5",
            voice="test-voice",
            speed=1.0,
            base_url=" https://api.elevenlabs.io ",
        )
    )
    with patch(
        "api.services.pipecat.service_factory.ElevenLabsTTSService"
    ) as tts_service:
        create_tts_service(user_config, AUDIO)
    assert tts_service.call_args.kwargs["url"] == "wss://api.elevenlabs.io"


def test_a_bare_speaches_host_is_refused_at_call_start_too(enforced):
    user_config = SimpleNamespace(
        stt=SpeachesSTTConfiguration(base_url="speaches.example.com/v1")
    )
    with pytest.raises(HTTPException) as e:
        create_stt_service(user_config, AUDIO)
    assert "scheme '' is not one of" in e.value.detail


def test_a_bare_host_is_refused_at_save_when_the_client_takes_it_as_is(enforced):
    # Review r2 (9f7d8d355b63): Speaches hands the raw base_url to its client, which needs a scheme.
    result = UserConfigurationValidator()._validate_service(
        SpeachesSTTConfiguration(base_url="speaches.example.com/v1"), "stt"
    )
    (status,) = result  # red today: saved, and STT fails once the call is answered
    assert "scheme '' is not one of http, https, ws, wss" in status["message"]


@pytest.mark.parametrize("url", ["redis", "10.0.0.10", " localhost:11434 "])
def test_a_bare_internal_host_is_still_refused(enforced, url):
    with pytest.raises(EgressBlocked):
        check_service_url(url, field="base_url")


def test_a_blank_service_url_is_refused(enforced):
    with pytest.raises(EgressBlocked):
        check_service_url("   ", field="base_url")


def test_a_failure_entry_expires_after_5_s_while_a_success_is_still_served(
    monkeypatch,
):
    cache, clock = _cache()
    dns = _Resolver(monkeypatch, PUBLIC_IP)
    cache.resolve("ok.example.com", 443)
    dns.address = None
    assert cache.resolve("flaky.example.com", 443) is None
    clock.now += 4
    # A burst within 5 s does not look up again.
    assert cache.resolve("flaky.example.com", 443) is None
    assert dns.lookups == 2
    dns.address = PUBLIC_IP
    clock.now += 2  # 6 s after the failure: the failure expired, the success did not
    assert cache.resolve("flaky.example.com", 443) == (PUBLIC_IP,)
    assert cache.resolve("ok.example.com", 443) == (PUBLIC_IP,)
    assert dns.lookups == 3  # with failures kept 60 s, the flaky host stays unresolved


# --- a miss on the event loop: bounded wait, lookups shared in flight (review r2, RUL-r2-1) ---


class _Lookup:
    """A pool lookup the test finishes by hand. Records every bounded wait; ``on_wait`` runs once the wait has
    decided it timed out, so the test can land the answer between that decision and the cache's reaction."""

    def __init__(self, on_wait):
        self.waits = []
        self._on_wait = on_wait
        self._callbacks = []
        self._done = False
        self._value = None

    def add_done_callback(self, callback):
        if self._done:
            callback(self)
        else:
            self._callbacks.append(callback)

    def result(self, timeout=None):
        if timeout is None:  # the cache reading a finished lookup
            return self._value
        done = self._done
        self.waits.append(timeout)
        if self._on_wait:
            self._on_wait(self)
        if not done:
            raise FutureTimeout()
        return self._value

    def exception(self):
        return None

    def finish(self, value):
        self._done, self._value = True, value
        for callback in self._callbacks:
            callback(self)


class _Pool:
    """Stands in for the lookup pool: hands out lookups the test finishes."""

    def __init__(self):
        self.lookups = []
        self.on_wait = None

    def submit(self, fn, *args):
        lookup = _Lookup(self.on_wait)
        self.lookups.append(lookup)
        return lookup


def _loop_cache() -> tuple[_ResolutionCache, _Clock, _Pool]:
    """A cache of the test's own with the service check's settings, a hand-moved clock and a hand-run pool."""
    clock, pool = _Clock(), _Pool()
    return _ResolutionCache(clock=clock, pool=pool), clock, pool


async def test_an_on_loop_miss_waits_1_s_at_most():
    cache, _, pool = _loop_cache()
    assert cache.resolve("slow.example.com", 443) is None
    (lookup,) = pool.lookups
    assert lookup.waits == [1.0]


async def test_a_timed_out_host_is_refused_for_60_s_unless_its_late_answer_lands():
    cache, clock, pool = _loop_cache()
    assert cache.resolve("slow.example.com", 443) is None
    clock.now += 59
    assert cache.resolve("slow.example.com", 443) is None
    (lookup,) = pool.lookups
    assert len(lookup.waits) == 1  # refused from the cache, no second wait
    clock.now += 2
    assert cache.resolve("slow.example.com", 443) is None
    # Expired: it waits again, on the lookup still in flight, never a second one.
    assert len(pool.lookups) == 1 and len(lookup.waits) == 2
    lookup.finish((PUBLIC_IP,))
    assert cache.resolve("slow.example.com", 443) == (PUBLIC_IP,)
    assert len(lookup.waits) == 2


async def test_an_answer_that_lands_as_the_wait_times_out_is_not_overwritten():
    cache, _, pool = _loop_cache()
    pool.on_wait = lambda lookup: lookup.finish((PUBLIC_IP,))
    assert cache.resolve("slow.example.com", 443) is None  # this check timed out
    assert cache.resolve("slow.example.com", 443) == (PUBLIC_IP,)
    assert len(pool.lookups) == 1


async def test_clear_drops_the_lookups_in_flight():
    cache, _, pool = _loop_cache()
    assert cache.resolve("slow.example.com", 443) is None
    cache.clear()
    pool.lookups[0].finish(("10.0.0.10",))  # the dropped lookup's answer is not stored
    assert cache.resolve("slow.example.com", 443) is None
    assert len(pool.lookups) == 2  # a fresh lookup, not the dropped one


def test_a_miss_off_the_event_loop_resolves_inline(monkeypatch):
    cache, _, pool = _loop_cache()
    dns = _Resolver(monkeypatch, PUBLIC_IP)
    assert cache.resolve("h.example.com", 443) == (PUBLIC_IP,)
    assert dns.lookups == 1 and pool.lookups == []


async def test_hung_lookups_fill_the_pool_and_then_a_fast_host_times_out_too(
    monkeypatch,
):
    # The bound the pool's size buys (egress_guard._LOOKUP_WORKERS): as many hung hosts as workers.
    gate = threading.Event()

    def getaddrinfo(host, port, *a, **k):
        if host.startswith("blackhole"):
            gate.wait(5)
        return [(socket.AF_INET, 1, 6, "", (PUBLIC_IP, port))]

    monkeypatch.setattr("socket.getaddrinfo", getaddrinfo)
    pool = ThreadPoolExecutor(max_workers=2)
    cache = _ResolutionCache(pool=pool, loop_timeout=0.2)
    try:
        assert cache.resolve("blackhole-1.example.com", 443) is None
        assert cache.resolve("blackhole-2.example.com", 443) is None
        assert cache.resolve("fast.example.com", 443) is None  # queued behind them
    finally:
        gate.set()
        pool.shutdown(wait=True)
    assert cache.resolve("fast.example.com", 443) == (PUBLIC_IP,)  # its late answer
    assert egress_guard._LOOKUP_WORKERS == 16
