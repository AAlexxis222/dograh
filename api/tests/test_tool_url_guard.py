"""VOZ-N0-28: the tool URL guard (custom tools, transfer resolver, MCP) is enforced by its own flag (VOZ-AC-B5-91, VOZ-AC-B0-30)."""

import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import httpx2
import pytest
from loguru import logger

from api.services.security.egress_guard import (
    EgressTarget,
    ToolUrlBlocked,
    check_tool_url,
)
from api.services.workflow.mcp_tool_session import McpToolSession, discover_mcp_tools
from api.services.workflow.tools.custom_tool import execute_http_tool
from api.services.workflow.tools.transfer_resolver import (
    TransferResolutionError,
    resolve_transfer_config,
)

REAL_GETADDRINFO = socket.getaddrinfo
PUBLIC_IP = "93.184.216.34"
PUBLIC_IPV6 = "2606:4700::1111"
IMDS = "http://169.254.169.254/latest/meta-data/"
# An invalid A-label: httpx parses the URL and raises IDNAError only when the host is read.
BAD_PUNYCODE = "http://xn--zz.example.com/x"


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
        self.clients = []  # the keyword arguments of every client built
        self._handler = handler or (lambda request: httpx.Response(200, json={}))

    def handle(self, request):
        self.requests.append(request)
        return self._handler(request)

    def patch(self, monkeypatch):
        real_client = httpx.AsyncClient

        def client(**kw):
            self.clients.append(kw)
            return real_client(**kw)

        # httpx is one module object, so this covers every client a tool builds; the transport stands in for the
        # network under it, under the guard's pinned transport too.
        monkeypatch.setattr("httpx.AsyncClient", client)
        for target in ("httpx.AsyncHTTPTransport", "httpx._client.AsyncHTTPTransport"):
            monkeypatch.setattr(target, lambda **kw: httpx.MockTransport(self.handle))


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
        # deprecated IPv6 forms that embed an internal IPv4 or are site-local: is_global is True for them on 3.13
        "http://[::169.254.169.254]/",
        "http://[::a9fe:a9fe]/",
        "http://[::ffff:0:a9fe:a9fe]/",
        "http://[fec0::1]/",
        # 6to4 and Teredo embed an IPv4 address inside 2000::/3
        "http://[2002:a9fe:a9fe::1]/",
        "http://[2001::1]/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_other_internal_targets_are_blocked(enforced, url):
    with pytest.raises(ToolUrlBlocked) as e:
        check_tool_url(url)
    assert e.value.code == "tool_url_blocked"


def test_an_ipv4_mapped_public_address_is_judged_by_the_ipv4_it_carries(enforced):
    assert check_tool_url("http://[::ffff:93.184.216.34]/") == "::ffff:93.184.216.34"


@pytest.mark.parametrize(
    "url", ["http:/169.254.169.254/", "http:169.254.169.254", "https:///path"]
)
def test_a_url_without_host_says_so_not_that_the_scheme_is_wrong(enforced, url):
    with pytest.raises(ToolUrlBlocked) as e:
        check_tool_url(url)
    assert "no host" in e.value.reason and "scheme" not in e.value.reason


def test_a_bad_scheme_names_the_scheme(enforced):
    with pytest.raises(ToolUrlBlocked) as e:
        check_tool_url("ftp://api.example.com/")
    assert "scheme 'ftp'" in e.value.reason


@pytest.mark.parametrize(
    "host",
    ["a" * 64 + ".example.com", "a..example.com"],  # label too long / empty label
)
async def test_a_host_the_resolver_cannot_encode_is_blocked_in_the_b0_28_shape(
    enforced, monkeypatch, host
):
    # The real resolver: it fails while IDNA-encoding the name, before any lookup leaves the machine.
    monkeypatch.setattr("socket.getaddrinfo", REAL_GETADDRINFO)
    wire = Wire()
    wire.patch(monkeypatch)
    result = await execute_http_tool(_tool(f"http://{host}/"), {})
    assert wire.requests == []
    assert result["code"] == "tool_url_blocked"
    assert result["reason"] == "the host name could not be resolved"
    assert result["hint"]


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


async def test_the_pinned_request_ignores_proxy_env_and_the_unpinned_one_keeps_it(
    monkeypatch,
):
    # An env proxy would tunnel to the pinned IP, and httpcore's tunnel ignores sni_hostname: the cert check fails.
    wire = Wire()
    wire.patch(monkeypatch)
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    await execute_http_tool(_tool("https://api.example.com/x"), {})
    monkeypatch.delenv("URL_GUARD_ENFORCE")
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    await execute_http_tool(_tool("https://api.example.com/x"), {})
    pinned, unpinned = wire.clients
    assert pinned["trust_env"] is False
    assert unpinned.get("trust_env", True) is True


# --- the transfer resolver is a tool too (VOZ-AC-B5-91, fix r1 item 6) ---


def _resolve_transfer(url):
    return resolve_transfer_config(
        tool=SimpleNamespace(tool_uuid="tr-1"),
        config={
            "destination_source": "dynamic",
            "destination": "",
            "timeout": 30,
            "resolver": {"type": "http", "url": url, "timeout_ms": 3000},
        },
        arguments={},
        call_context_vars={},
        gathered_context_vars={},
        organization_id=None,
        workflow_run_id=None,
    )


def _transfer_wire():
    return Wire(
        lambda request: httpx.Response(
            200, json={"transfer_context": {"destination": "+14155550123"}}
        )
    )


@pytest.mark.parametrize("url", [IMDS, "http://redis:6379"])
async def test_transfer_resolver_to_an_internal_target_is_blocked_before_the_wire(
    enforced, monkeypatch, url
):
    wire = _transfer_wire()
    wire.patch(monkeypatch)
    with pytest.raises(TransferResolutionError) as e:
        await _resolve_transfer(url)
    assert wire.requests == []
    assert e.value.reason == "tool_url_blocked"
    assert "config.resolver.url" in e.value.message and "hint=" in e.value.message


async def test_transfer_resolver_connects_to_the_validated_ip(enforced, monkeypatch):
    wire = _transfer_wire()
    wire.patch(monkeypatch)
    resolved = await _resolve_transfer("https://crm.example.com/resolve")
    assert resolved.destination == "+14155550123"
    (request,) = wire.requests
    assert str(request.url) == f"https://{PUBLIC_IP}/resolve"
    assert request.headers["host"] == "crm.example.com"
    assert request.extensions["sni_hostname"] == "crm.example.com"
    (client,) = wire.clients
    assert client["trust_env"] is False and client["follow_redirects"] is False


async def test_transfer_resolver_flag_off_oss_reaches_imds_unchanged(monkeypatch):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "oss")
    wire = _transfer_wire()
    wire.patch(monkeypatch)
    await _resolve_transfer(IMDS)
    (request,) = wire.requests
    assert str(request.url) == IMDS and "sni_hostname" not in request.extensions
    (client,) = wire.clients
    assert client["follow_redirects"] is False  # stated, not left to the httpx default


async def test_transfer_resolver_flag_off_saas_keeps_the_legacy_error(monkeypatch):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "saas")
    wire = _transfer_wire()
    wire.patch(monkeypatch)
    with pytest.raises(ValueError, match="config.resolver.url must resolve") as e:
        await _resolve_transfer(IMDS)
    assert not isinstance(e.value, TransferResolutionError)
    assert wire.requests == []


# --- an invalid punycode host is blocked in the B0-28 shape on every path (fix r2 N1) ---


async def test_custom_tool_with_an_invalid_punycode_host_is_blocked(
    enforced, monkeypatch
):
    wire = Wire()
    wire.patch(monkeypatch)
    result = await execute_http_tool(_tool(BAD_PUNYCODE), {})
    assert wire.requests == []
    assert result["code"] == "tool_url_blocked"
    assert result["reason"] == "the URL cannot be parsed" and result["hint"]


async def test_transfer_resolver_with_an_invalid_punycode_host_is_blocked(
    enforced, monkeypatch
):
    wire = _transfer_wire()
    wire.patch(monkeypatch)
    with pytest.raises(TransferResolutionError) as e:
        await _resolve_transfer(BAD_PUNYCODE)
    assert e.value.reason == "tool_url_blocked" and wire.requests == []


# --- MCP servers are tools too (VOZ-AC-B5-91, fix r1 item 6, fix r2 M1): guarded, pinned, no redirect ---


class FakeMcpServer:
    """Stands in for the network under every httpx2 client: a minimal streamable-HTTP MCP server.

    It records every request that reaches the transport (host, Host header, SNI), whatever client sent it.
    """

    def __init__(self, redirect_to=None):
        self.requests = []
        self._redirect_to = redirect_to

    def handle(self, request):
        self.requests.append(request)
        if self._redirect_to:
            return httpx2.Response(307, headers={"location": self._redirect_to})
        if request.method != "POST":
            return httpx2.Response(405)
        message = json.loads(request.content)
        if "id" not in message:  # a notification
            return httpx2.Response(202)
        if message["method"] == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        else:  # tools/list
            echo = {
                "name": "echo",
                "description": "Echo",
                "inputSchema": {"type": "object", "properties": {}},
            }
            result = {"tools": [echo]}
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    def patch(self, monkeypatch):
        # Both names: the one a client builds by default and the one a caller can build explicitly.
        for target in (
            "httpx2.AsyncHTTPTransport",
            "httpx2._client.AsyncHTTPTransport",
        ):
            monkeypatch.setattr(target, lambda **kw: httpx2.MockTransport(self.handle))


@pytest.fixture
def mcp_server(monkeypatch):
    server = FakeMcpServer()
    server.patch(monkeypatch)
    return server


def _mcp_session(url):
    return McpToolSession(
        tool_uuid="mcp-1",
        tool_name="Probe MCP",
        url=url,
        credential=None,
        tools_filter=[],
        timeout_secs=5,
        sse_read_timeout_secs=5,
    )


@pytest.mark.parametrize("url", [IMDS, "http://redis:6379"])
async def test_mcp_call_start_to_an_internal_target_degrades_without_connecting(
    enforced, mcp_server, url
):
    session = _mcp_session(url)
    # The call-start path (pipecat_engine): it must not raise.
    await session.start_managed()
    try:
        assert session.available is False and session.function_schemas() == []
        assert mcp_server.requests == []
    finally:
        await session.close_managed()


async def test_mcp_call_start_to_a_public_server_is_pinned_to_the_validated_ip(
    enforced, mcp_server
):
    session = _mcp_session("https://mcp.example.com/mcp")
    await session.start()
    try:
        assert session.available is True
        assert [s.name for s in session.function_schemas()] == ["mcp__probe_mcp__echo"]
    finally:
        await session.close()
    assert mcp_server.requests
    for request in mcp_server.requests:
        assert request.url.host == PUBLIC_IP and request.url.path == "/mcp"
        assert request.headers["host"] == "mcp.example.com"
        assert request.extensions["sni_hostname"] == "mcp.example.com"


async def test_mcp_server_redirecting_to_imds_is_not_followed(enforced, monkeypatch):
    server = FakeMcpServer(redirect_to=IMDS)
    server.patch(monkeypatch)
    session = _mcp_session("https://mcp.example.com/mcp")
    await session.start_managed()
    try:
        assert session.available is False  # degrades like an unreachable server
    finally:
        await session.close_managed()
    assert server.requests
    assert all(r.url.host == PUBLIC_IP for r in server.requests)  # never IMDS


async def test_mcp_pinned_client_ignores_env_and_redirects(
    enforced, mcp_server, monkeypatch
):
    built = []
    real_client = httpx2.AsyncClient

    def client(**kw):
        built.append(kw)
        return real_client(**kw)

    monkeypatch.setattr("httpx2.AsyncClient", client)
    session = _mcp_session("https://mcp.example.com/mcp")
    await session.start()
    await session.close()
    (kw,) = built
    assert kw["trust_env"] is False and kw["follow_redirects"] is False


@pytest.mark.parametrize("lib", [httpx, httpx2])
async def test_a_pinned_client_refuses_any_other_host(
    enforced, monkeypatch, mcp_server, lib
):
    wire = Wire()
    wire.patch(monkeypatch)
    target = EgressTarget("mcp.example.com", PUBLIC_IP)
    client = target.httpx_client() if lib is httpx else target.httpx2_client()
    async with client:
        with pytest.raises(lib.ConnectError):
            await client.post("http://169.254.169.254/latest/api/token")
    assert wire.requests == [] and mcp_server.requests == []


@pytest.mark.parametrize("mode", ["oss", "saas"])
async def test_mcp_flag_off_connects_as_today(monkeypatch, mcp_server, mode):
    # MCP had no URL check in any mode: flag off adds none (not even the legacy saas one) and does not pin.
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", mode)
    session = _mcp_session(IMDS)
    await session.start()
    try:
        assert session.available is True
    finally:
        await session.close()
    assert mcp_server.requests
    assert all(str(r.url) == IMDS for r in mcp_server.requests)
    assert all("sni_hostname" not in r.extensions for r in mcp_server.requests)


@pytest.mark.parametrize("url", [IMDS, "http://redis:6379"])
async def test_mcp_discovery_of_an_internal_target_returns_the_b0_28_error(
    enforced, mcp_server, url
):
    with pytest.raises(ToolUrlBlocked) as e:
        await discover_mcp_tools(
            url=url, credential=None, timeout_secs=5, sse_read_timeout_secs=5
        )
    assert e.value.code == "tool_url_blocked" and e.value.hint
    assert mcp_server.requests == []


async def test_mcp_discovery_of_a_public_server_is_unchanged(enforced, mcp_server):
    tools = await discover_mcp_tools(
        url="https://mcp.example.com/mcp",
        credential=None,
        timeout_secs=5,
        sse_read_timeout_secs=5,
    )
    assert tools == [{"name": "echo", "description": "Echo"}]


@pytest.mark.parametrize("url", [IMDS, BAD_PUNYCODE])
async def test_mcp_refresh_reports_the_block_and_keeps_the_cache(
    enforced, monkeypatch, mcp_server, url
):
    from api.services import tool_management

    tool = SimpleNamespace(
        tool_uuid="tu-mcp",
        category="mcp",
        definition={"schema_version": 1, "type": "mcp", "config": {"url": url}},
    )
    monkeypatch.setattr(
        tool_management.db_client, "get_tool_by_uuid", AsyncMock(return_value=tool)
    )
    update = AsyncMock()
    monkeypatch.setattr(tool_management.db_client, "update_tool", update)
    response = await tool_management.refresh_mcp_tool_for_user(
        "tu-mcp", SimpleNamespace(selected_organization_id=1)
    )
    assert response.discovered_tools == []
    assert "code=tool_url_blocked" in response.error and "hint=" in response.error
    assert response.code == "tool_url_blocked" and response.hint
    update.assert_not_awaited()
    assert mcp_server.requests == []


# --- fix r3: one exception type, one resolution per discovery, the logged cause, flag-off parity, IDN ---


@pytest.mark.parametrize("field", ["config.url", "config.resolver.url"])
def test_the_guard_raises_one_type_in_both_modes(monkeypatch, field):
    monkeypatch.setattr("api.utils.url_security.DEPLOYMENT_MODE", "saas")
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    with pytest.raises(ToolUrlBlocked) as off:
        check_tool_url(IMDS, field=field)
    # Guard off: the legacy refusal, its text unchanged.
    assert off.value.enforced is False
    assert str(off.value) == f"{field} must resolve to a public IP address in SaaS mode"
    monkeypatch.setenv("URL_GUARD_ENFORCE", "true")
    with pytest.raises(ToolUrlBlocked) as on:
        check_tool_url(IMDS, field=field)
    assert on.value.enforced is True and on.value.code == "tool_url_blocked"


async def test_mcp_discovery_resolves_the_host_once(enforced, mcp_server, monkeypatch):
    names = []

    def resolver(host, *a, **k):
        names.append(host)
        return _addrinfo(PUBLIC_IP)

    monkeypatch.setattr("socket.getaddrinfo", resolver)
    tools = await discover_mcp_tools(
        url="https://mcp.example.com/mcp",
        credential=None,
        timeout_secs=5,
        sse_read_timeout_secs=5,
    )
    assert tools and names == ["mcp.example.com"]


async def test_mcp_degrade_logs_the_real_cause_once(enforced, monkeypatch):
    def refuse(request):
        raise httpx2.ConnectError("connection refused by the probe")

    for target in ("httpx2.AsyncHTTPTransport", "httpx2._client.AsyncHTTPTransport"):
        monkeypatch.setattr(target, lambda **kw: httpx2.MockTransport(refuse))
    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), level="WARNING")
    try:
        session = _mcp_session("https://mcp.example.com/mcp")
        await session.start_managed()
        assert session.available is False
        await session.close_managed()
    finally:
        logger.remove(sink)
    (line,) = [ln for ln in lines if "MCP session unavailable" in ln]
    assert "ConnectError" in line and "connection refused by the probe" in line, line
    assert "Cancelled via cancel scope" not in line, line
    # The cause itself, not the task group wrapping it.
    assert "ExceptionGroup" not in line, line
    assert not any("Error closing MCP session" in ln for ln in lines), lines


async def test_mcp_flag_off_client_is_the_one_pipecat_built(monkeypatch, mcp_server):
    monkeypatch.delenv("URL_GUARD_ENFORCE", raising=False)
    built = []
    real_client = httpx2.AsyncClient

    def client(**kw):
        built.append(kw)
        return real_client(**kw)

    monkeypatch.setattr("httpx2.AsyncClient", client)
    credential = MagicMock()
    credential.credential_type = "bearer_token"
    credential.credential_data = {"token": "abc"}
    session = McpToolSession(
        tool_uuid="mcp-1",
        tool_name="Probe MCP",
        url="https://mcp.example.com/mcp",
        credential=credential,
        tools_filter=[],
        timeout_secs=7,
        sse_read_timeout_secs=11,
    )
    await session.start()
    await session.close()
    (kw,) = built
    # Pipecat's _streamable_http_transport: headers, Timeout(timeout, read=sse_read_timeout), follow_redirects=True.
    assert kw["headers"] == {"Authorization": "Bearer abc"}
    assert kw["timeout"].connect == 7 and kw["timeout"].read == 11
    assert kw["follow_redirects"] is True
    assert "trust_env" not in kw and "transport" not in kw


async def test_an_idn_host_resolves_and_pins_with_the_name_httpx_sends(
    enforced, monkeypatch
):
    names = []

    def resolver(host, *a, **k):
        names.append(host)
        return _addrinfo(PUBLIC_IP)

    monkeypatch.setattr("socket.getaddrinfo", resolver)
    wire = Wire()
    wire.patch(monkeypatch)
    # Sharp s: IDNA 2003 (getaddrinfo on the Unicode name) maps it to "strasse.de", httpx sends IDNA 2008.
    result = await execute_http_tool(_tool("https://stra\u00dfe.de/x"), {})
    assert result["status"] == "success"
    assert names == ["xn--strae-oqa.de"]
    (request,) = wire.requests
    assert request.url.host == PUBLIC_IP
    assert request.headers["host"] == "xn--strae-oqa.de"
    assert request.extensions["sni_hostname"] == "xn--strae-oqa.de"
