"""Egress URL guard (VOZ-AC-B5-91 layer 2): a request to a tenant-configured URL never reaches IMDS or the internal network.

One entry point, ``guard_egress()``: it takes the mode decision (``URL_GUARD_ENFORCE``, egress_flag.py), resolves
the host off the event loop and returns an ``EgressTarget`` whose client is ready to use: pinned to the validated
address, redirects off, no env settings. Every refusal is a ``ToolUrlBlocked``, in both modes. Callers only choose
the wording (field, hint) and map the refusal to their own result shape.
"""

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Any

import httpx
import httpx2  # the httpx line the MCP SDK is built on
from loguru import logger

from api.services.security.egress_flag import url_guard_enforced
from api.utils.url_security import validate_user_configured_service_url

# Compose service names (docker-compose.yaml): inside the Docker network they resolve to private addresses.
_INTERNAL_HOSTNAMES = frozenset(
    {"localhost", "postgres", "redis", "minio", "api", "call", "ui", "nginx", "coturn"}
)
# Global unicast IPv6. Everything outside it (IPv4-compatible ::/96, IPv4-translated ::ffff:0:0:0/96, NAT64
# 64:ff9b::/96, site-local fec0::/10, ULA, link-local, multicast) is refused whatever is_global says.
_GLOBAL_UNICAST_V6 = ipaddress.ip_network("2000::/3")
_DEFAULT_HINT = (
    "point it at a public http(s) endpoint: private, loopback, link-local and internal "
    "service addresses are not reachable"
)


class ToolUrlBlocked(ValueError):
    """A URL the guard refuses, in the VOZ-AC-B0-28 shape: stable code, what, where, reason, hint.

    With the guard off (``enforced`` False) it carries the legacy DEPLOYMENT_MODE check's refusal and its message
    is that check's text, unchanged, so the flag-off outcome of every caller stays what it was.
    """

    code = "tool_url_blocked"
    what = "outbound request blocked before it left the platform"

    def __init__(
        self, host: str, reason: str, *, field: str, hint: str, enforced: bool = True
    ):
        self.where = f"{field} host '{host}'"
        self.reason = reason
        self.hint = hint
        self.enforced = enforced
        super().__init__(
            f"code={self.code} what={self.what} where={self.where} reason={reason} hint={hint}"
            if enforced
            else reason
        )


def _is_internal_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Anything that is not public unicast: private, loopback, link-local, 100.64.0.0/10, unspecified, multicast.

    An IPv4-mapped address (::ffff:a.b.c.d) is judged by the IPv4 it carries, explicitly: the stdlib's answer for
    it changed between versions (3.13 delegates is_global to the embedded IPv4). Other IPv6 must be global unicast
    (2000::/3) and is_global, which also refuses 6to4 and Teredo, the forms that embed an IPv4 inside 2000::/3.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip not in _GLOBAL_UNICAST_V6:
            return True
    return not ip.is_global or ip.is_multicast


def _ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _tool_url_addresses(url: str, field: str, hint: str) -> tuple[str, list[str]]:
    """Parse the URL the way the request will, and return its host with every address that host maps to."""

    def blocked(host: str, reason: str) -> ToolUrlBlocked:
        return ToolUrlBlocked(host, reason, field=field, hint=hint)

    try:
        parsed = httpx.URL(url)
        # httpx decodes punycode when the host is read: an invalid A-label raises IDNAError (a UnicodeError) here.
        host = parsed.host.rstrip(".")  # httpx already lower-cases it
        # The name the request will use (IDNA 2008, as httpx encodes it). getaddrinfo on the Unicode name would
        # encode IDNA 2003 and resolve another domain for deviation characters (sharp s).
        ascii_host = parsed.raw_host.decode("ascii").rstrip(".")
    except (httpx.InvalidURL, UnicodeError):
        raise blocked("(unparseable)", "the URL cannot be parsed") from None
    if parsed.scheme not in ("http", "https"):
        raise blocked(
            host or "(none)", f"scheme '{parsed.scheme}' is not http or https"
        )
    if not host:
        raise blocked("(none)", "the URL has no host (expected http(s)://host/...)")
    if host in _INTERNAL_HOSTNAMES:
        raise blocked(host, "the host is an internal service name")
    if _ip_literal(host):
        return host, [host]
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(ascii_host, port, type=socket.SOCK_STREAM)
    # UnicodeError: a label the IDNA codec refuses (too long, empty) fails before any lookup.
    except (socket.gaierror, UnicodeError):
        raise blocked(host, "the host name could not be resolved") from None
    return host, [str(info[4][0]) for info in infos]


def check_tool_url(
    url: str,
    *,
    field: str = "url",
    hint: str = _DEFAULT_HINT,
    legacy_check: bool = True,
) -> str | None:
    """Guard a URL; return the address the request must connect to, or None to connect as usual. Blocking: it
    resolves the host name (guard_egress runs it off the event loop).

    URL_GUARD_ENFORCE=true enforces whatever DEPLOYMENT_MODE says: every address the host maps to must be public
    (one bad record blocks), and the returned one is the address to connect to, so the request does not resolve
    the name again. Guard off: the pre-existing DEPLOYMENT_MODE check runs, unchanged, when ``legacy_check`` (a path
    that never had it, MCP, passes False and gets no check).
    """
    if not url_guard_enforced():
        if legacy_check:
            try:
                validate_user_configured_service_url(url, field_name=field)
            except ValueError as e:
                raise ToolUrlBlocked(
                    "(not checked)", str(e), field=field, hint=hint, enforced=False
                ) from e
        return None
    host, addresses = _tool_url_addresses(url, field, hint)
    if not addresses:
        raise ToolUrlBlocked(
            host, "the host name has no address", field=field, hint=hint
        )
    for address in addresses:
        ip = _ip_literal(address)
        if ip is None or _is_internal_address(ip):  # unreadable is not vouched for
            logger.warning(f"Egress guard blocked '{host}': it maps to {address}")
            raise ToolUrlBlocked(
                host,
                "the host maps to a private, loopback, link-local or otherwise internal address",
                field=field,
                hint=hint,
            )
    return addresses[0]


class _PinnedTransportBase:
    """The transport of a guarded client: every request goes to the one validated address.

    The Host header stays as the client built it (the host name) and, for a name host, SNI and the certificate
    check stay on it. A request for any other host is refused, so no redirect can take the client elsewhere. No
    env settings: trust_env=False here and on the client, which turns off proxies (an env proxy would tunnel to
    the IP and ignore sni_hostname, failing the certificate check) and also SSL_CERT_FILE / SSL_CERT_DIR and
    .netrc; no cell sets them today.
    """

    _http: Any  # the httpx line: httpx (custom tools, transfer resolver) or httpx2 (the MCP SDK)

    def __init__(self, host: str, address: str):
        self._host = host  # ASCII, as the request URL carries it
        self._address = address
        self._inner = self._http.AsyncHTTPTransport(trust_env=False)

    async def handle_async_request(self, request: Any) -> Any:
        host = request.url.raw_host.decode("ascii")
        if host != self._host:
            raise self._http.ConnectError(
                f"'{host}' is not the validated host '{self._host}'", request=request
            )
        extensions = dict(request.extensions)
        if not _ip_literal(self._host):
            extensions["sni_hostname"] = self._host
        # A new request object: the client keeps the original (host name URL) to judge redirects against.
        pinned = self._http.Request(
            request.method,
            request.url.copy_with(host=self._address),
            headers=request.headers,
            stream=request.stream,
            extensions=extensions,
        )
        return await self._inner.handle_async_request(pinned)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _PinnedHttpxTransport(_PinnedTransportBase, httpx.AsyncBaseTransport):
    _http = httpx


class _PinnedHttpx2Transport(_PinnedTransportBase, httpx2.AsyncBaseTransport):
    _http = httpx2


@dataclass(frozen=True)
class EgressTarget:
    """Where a guarded request goes: the URL's ASCII host and the validated address the client is pinned to.

    No address: the guard is off, the client connects as usual.
    """

    host: str | None = None
    address: str | None = None

    def httpx_client(self, **kwargs: Any) -> httpx.AsyncClient:
        """An httpx client that never follows redirects (a 3xx would go to a URL the guard has not checked)."""
        if self.address is None:
            return httpx.AsyncClient(follow_redirects=False, **kwargs)
        return httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            transport=_PinnedHttpxTransport(self.host, self.address),
            **kwargs,
        )

    def httpx2_client(self, **kwargs: Any) -> httpx2.AsyncClient:
        """An httpx2 client for the MCP SDK. Guard off: follow_redirects=True, the client Pipecat's MCPClient built
        (the SDK follows same-origin redirects itself and does not consult it)."""
        if self.address is None:
            return httpx2.AsyncClient(follow_redirects=True, **kwargs)
        return httpx2.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            transport=_PinnedHttpx2Transport(self.host, self.address),
            **kwargs,
        )


async def guard_egress(
    url: str, *, field: str, hint: str, legacy_check: bool = True
) -> EgressTarget:
    """Guard ``url`` and return where its request goes; raises ToolUrlBlocked (both modes). See check_tool_url."""
    address = await asyncio.to_thread(
        check_tool_url, url, field=field, hint=hint, legacy_check=legacy_check
    )
    if address is None:
        return EgressTarget()
    return EgressTarget(httpx.URL(url).raw_host.decode("ascii"), address)
