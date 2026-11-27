"""VOZ-N0-41: service URLs honour URL_GUARD_ENFORCE (VOZ-AC-B5-91, VOZ-AC-B0-30)."""

import socket
from secrets import token_hex
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.registry import SpeachesLLMConfiguration, ServiceProviders
from api.services.gen_ai.embedding.azure_openai_service import AzureOpenAIEmbeddingService
from api.services.gen_ai.embedding.openai_service import OpenAIEmbeddingService
from api.services.pipecat.service_factory import create_llm_service_from_provider
from api.services.security.egress_guard import (
    EgressTarget,
    ToolUrlBlocked,
    check_service_url,
    check_tool_url,
)

PUBLIC_IP = "93.184.216.34"
KEY = token_hex(12)  # generated per run


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, 1, 6, "", (PUBLIC_IP, 443))],
    )


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")  # the cell's mode


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
