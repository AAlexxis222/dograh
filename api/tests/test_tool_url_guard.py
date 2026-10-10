"""VOZ-N0-28: the custom-tool URL guard is enforced by its own flag (VOZ-AC-B5-91, VOZ-AC-B0-30)."""

import socket
from types import SimpleNamespace

import httpx
import pytest

from api.services.workflow.tools.custom_tool import (
    ToolUrlBlocked,
    check_tool_url,
    execute_http_tool,
)

PUBLIC_IP = "93.184.216.34"
PUBLIC_IPV6 = "2606:4700::1111"
IMDS = "http://169.254.169.254/latest/meta-data/"


def _addrinfo(*ips):
    return [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, 1, 6, "", (ip, 443))
        for ip in ips
    ]


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """No real DNS: every name resolves to a public address unless the test says otherwise."""
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: _addrinfo(PUBLIC_IP))


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    monkeypatch.setenv("DEPLOYMENT_MODE", "oss")


def _tool(url, method="GET"):
    config = {"method": method, "url": url}
    return SimpleNamespace(
        tool_uuid="t-1",
        name="Probe",
        definition={"schema_version": 1, "type": "http_api", "config": config},
    )


class Wire:
    """Stands in for the network: records every request httpx would put on the wire."""

    def __init__(self, handler=None):
        self.requests = []
        self._handler = handler or (lambda request: httpx.Response(200, json={}))

    def handle(self, request):
        self.requests.append(request)
        return self._handler(request)

    def patch(self, monkeypatch):
        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            "api.services.workflow.tools.custom_tool.httpx.AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(self.handle), **kw),
        )


# --- the plan's tests (binding) ---


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://redis:6379/",
        "http://postgres:5432",
        "http://10.0.0.5/",
        "http://127.0.0.1:8000/",
        "http://[::1]/",
        "http://localhost/",
    ],
)
def test_blocked_when_enforced_even_in_oss_mode(monkeypatch, url):
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    monkeypatch.setenv("DEPLOYMENT_MODE", "oss")
    with pytest.raises(ToolUrlBlocked) as e:
        check_tool_url(url)
    assert e.value.code == "tool_url_blocked" and e.value.hint


def test_public_https_allowed(monkeypatch):
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    check_tool_url("https://api.example.com/booking")


def test_dns_rebinding_to_private_ip_is_blocked(monkeypatch):
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.1.2.3", 443))]
    )
    with pytest.raises(ToolUrlBlocked):
        check_tool_url("https://looks-public.example.com/")


# --- today's guard, characterized (flag off): these pass before and after the change ---


async def test_flag_off_oss_mode_sends_the_request_to_imds_unchanged(monkeypatch):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    wire = Wire()
    wire.patch(monkeypatch)
    result = await execute_http_tool(_tool(IMDS), {})
    assert result["status"] == "success"
    (request,) = wire.requests
    assert str(request.url) == IMDS and "sni_hostname" not in request.extensions


async def test_flag_off_saas_mode_blocks_private_hosts_with_the_legacy_error(
    monkeypatch,
):
    monkeypatch.setenv("URL_GUARD_ENFORCE", "false")
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "saas")
    wire = Wire()
    wire.patch(monkeypatch)
    result = await execute_http_tool(_tool(IMDS), {})
    assert result["status"] == "error"
    assert result["error"].startswith("URL validation failed:")
    assert "code" not in result and wire.requests == []


def test_flag_off_check_delegates_to_the_legacy_guard(monkeypatch):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    assert check_tool_url(IMDS) is None
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "saas")
    with pytest.raises(ValueError, match="public IP"):
        check_tool_url(IMDS)


# --- what is blocked when enforced (ruling 3) ---


@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:169.254.169.254]/",
        "http://[::ffff:7f00:1]/",
        "http://[::ffff:10.0.0.5]/",
        "http://[64:ff9b::a9fe:a9fe]/",
        "http://100.64.0.1/",
        "http://224.0.0.1/",
        "http://[ff02::1]/",
        "http://100.127.255.254/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[fd12:3456::1]/",
        "http://172.17.0.2:5432/",
        "http://192.168.1.1/",
        "http://0.0.0.0/",
        "http://[::]/",
        "http://minio:9000/",
        "http://api:8000/",
        "http://call:8001/",
        "http://ui:3000/",
        "http://nginx/",
        "http://coturn:3478/",
        "http://REDIS./",
        "http://user:pw@postgres:5432/",
        "http://localhost./",
        "http://example.com\\@169.254.169.254/",
    ],
)
def test_other_internal_targets_are_blocked(enforced, url):
    with pytest.raises(ToolUrlBlocked) as e:
        check_tool_url(url)
    assert e.value.code == "tool_url_blocked"


@pytest.mark.parametrize(
    "url", ["ftp://api.example.com/", "file:///etc/passwd", "gopher://api.example.com/"]
)
def test_non_http_schemes_are_blocked(enforced, url):
    with pytest.raises(ToolUrlBlocked):
        check_tool_url(url)


def test_one_private_record_among_public_ones_blocks(enforced, monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *a, **k: _addrinfo(PUBLIC_IP, "169.254.169.254")
    )
    with pytest.raises(ToolUrlBlocked):
        check_tool_url("https://mixed.example.com/")


def test_an_unresolvable_host_is_blocked_not_waved_through(enforced, monkeypatch):
    def fail(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr("socket.getaddrinfo", fail)
    with pytest.raises(ToolUrlBlocked):
        check_tool_url("https://nowhere.example.com/")


def test_a_host_with_no_address_is_blocked(enforced, monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [])
    with pytest.raises(ToolUrlBlocked):
        check_tool_url("https://empty.example.com/")


def test_the_flag_is_read_when_called_and_only_true_enforces(monkeypatch):
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    monkeypatch.setenv("URL_GUARD_ENFORCE", "false")
    assert check_tool_url(IMDS) is None
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    with pytest.raises(ToolUrlBlocked):
        check_tool_url(IMDS)


def test_the_error_names_the_destination_and_leaks_no_address(enforced, monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: _addrinfo("10.1.2.3"))
    with pytest.raises(ToolUrlBlocked) as e:
        check_tool_url("https://looks-public.example.com/pay?token=s3cr3t")
    err = e.value
    assert err.what and err.where and err.reason and err.hint
    assert "looks-public.example.com" in err.where
    assert "10.1.2.3" not in str(err) and "s3cr3t" not in str(err)


# --- the request uses the validated address (ruling 4) ---


async def test_blocked_tool_call_never_reaches_the_wire(enforced, monkeypatch):
    wire = Wire()
    wire.patch(monkeypatch)
    result = await execute_http_tool(_tool("http://redis:6379/"), {})
    assert wire.requests == []
    assert result["status"] == "error" and result["code"] == "tool_url_blocked"
    assert result["reason"] and result["hint"] and "redis" in result["where"]


async def test_request_connects_to_the_validated_ip_and_resolves_once(
    enforced, monkeypatch
):
    answers = [_addrinfo(PUBLIC_IP), _addrinfo("127.0.0.1")]  # a rebinding resolver
    calls = []

    def resolver(*a, **k):
        calls.append(a)
        return answers[min(len(calls), 2) - 1]

    monkeypatch.setattr("socket.getaddrinfo", resolver)
    wire = Wire()
    wire.patch(monkeypatch)
    result = await execute_http_tool(
        _tool("https://api.example.com:8443/a/b?x=1", method="POST"), {"k": "v"}
    )
    assert result["status"] == "success"
    (request,) = wire.requests
    assert str(request.url) == f"https://{PUBLIC_IP}:8443/a/b?x=1"
    assert request.headers["host"] == "api.example.com:8443"
    assert request.extensions["sni_hostname"] == "api.example.com"
    assert len(calls) == 1  # the guard resolved once and nothing resolves again


async def test_http_request_keeps_the_host_header_without_sni(enforced, monkeypatch):
    wire = Wire()
    wire.patch(monkeypatch)
    await execute_http_tool(_tool("http://api.example.com/x"), {"q": "1"})
    (request,) = wire.requests
    assert str(request.url) == f"http://{PUBLIC_IP}/x?q=1"
    assert request.headers["host"] == "api.example.com"


async def test_ipv6_target_is_pinned_in_brackets(enforced, monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: _addrinfo(PUBLIC_IPV6))
    wire = Wire()
    wire.patch(monkeypatch)
    await execute_http_tool(_tool("https://v6.example.com/x"), {})
    (request,) = wire.requests
    assert request.url.host == PUBLIC_IPV6 and request.url.scheme == "https"
    assert request.headers["host"] == "v6.example.com"


async def test_ip_literal_target_needs_no_sni_override(enforced, monkeypatch):
    wire = Wire()
    wire.patch(monkeypatch)
    await execute_http_tool(_tool(f"https://{PUBLIC_IP}/x"), {})
    (request,) = wire.requests
    assert request.url.host == PUBLIC_IP and "sni_hostname" not in request.extensions


async def test_a_redirect_to_imds_is_not_followed(enforced, monkeypatch):
    wire = Wire(
        lambda request: httpx.Response(302, headers={"location": IMDS}, text="moved")
    )
    wire.patch(monkeypatch)
    result = await execute_http_tool(_tool("https://api.example.com/go"), {})
    assert len(wire.requests) == 1  # the second hop never leaves
    assert result["status"] == "error" and result["status_code"] == 302
