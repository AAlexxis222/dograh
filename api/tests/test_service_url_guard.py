"""VOZ-N0-41: service URLs honour URL_GUARD_ENFORCE (VOZ-AC-B5-91, VOZ-AC-B0-30)."""

import socket
import threading
from secrets import token_hex
from types import SimpleNamespace

import httpx
import httpx2
import pytest
from fastapi import HTTPException

from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.registry import (
    OpenAILLMService,
    ServiceProviders,
    SpeachesLLMConfiguration,
    SpeachesSTTConfiguration,
    SpeachesTTSConfiguration,
)
from api.services.gen_ai.embedding.azure_openai_service import AzureOpenAIEmbeddingService
from api.services.gen_ai.embedding.openai_service import OpenAIEmbeddingService
from api.services.pipecat.service_factory import create_llm_service_from_provider
from api.services.security.egress_guard import (
    EgressTarget,
    ToolUrlBlocked,
    check_service_url,
    check_tool_url,
)

PUBLIC_IP = "93.184.216.34"  # what conftest's public_dns resolves every name to
IMDS = "http://169.254.169.254/latest/meta-data/"
KEY = token_hex(12)  # generated per run

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
    with pytest.raises(HTTPException) as e:  # red today: DID NOT RAISE (oss returns early)
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
    assert len(result) == 1 and "tool_url_blocked" in result[0]["message"]  # red today: result == []


def test_embeddings_refuse_internal_url_in_a_cell(enforced):
    with pytest.raises(ValueError, match="tool_url_blocked"):  # red today: DID NOT RAISE
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


def test_control_flag_off_oss_keeps_local_urls(monkeypatch):  # regression (today's OSS behaviour)
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
    with pytest.raises(ToolUrlBlocked) as e:
        check_service_url("ftp://example.com", field="base_url")
    assert e.value.reason == "scheme 'ftp' is not one of http, https, ws, wss"


@pytest.mark.parametrize("url", [None, ""])
def test_check_service_url_refuses_empty(enforced, url):
    with pytest.raises(ToolUrlBlocked) as e:
        check_service_url(url, field="base_url")
    assert e.value.reason == "the URL has no host"


@pytest.mark.parametrize(
    "url, port", [("wss://el.example.com/x", 443), ("ws://el.example.com/x", 80)]
)
def test_check_service_url_default_port_follows_the_scheme(enforced, monkeypatch, url, port):
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
    assert check_service_url("http://localhost:11434/v1", field="base_url") == EgressTarget()


def test_tool_urls_still_refuse_ws(enforced):
    with pytest.raises(ToolUrlBlocked):
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
