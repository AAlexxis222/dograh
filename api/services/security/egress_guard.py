"""Egress URL guard (VOZ-AC-B5-91 layer 2): a request to a tenant-configured URL never reaches IMDS or the internal network.

Two entry points. ``guard_egress()`` (async, for tool URLs): it takes the mode decision (``URL_GUARD_ENFORCE``, egress_flag.py), resolves
the host off the event loop and returns an ``EgressTarget`` whose client is ready to use: pinned to the validated
address, redirects off, no env settings. Every refusal is an ``EgressBlocked``, in both modes. Callers only choose
the wording (field, hint) and map the refusal to their own result shape.
``check_service_url()`` is the synchronous entry point for the service URLs a tenant configures (model providers,
embeddings, STT/TTS); it also admits ws and wss. Its lookups go through a short-lived cache. On an event loop a cache
miss never looks up: it raises ``ServiceHostUnresolved``, and async callers build their services through
``await build_resolving_service_hosts(builder, ...)``, which resolves the host off the loop and runs the build again.
"""

import asyncio
import ipaddress
import socket
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
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
_SERVICE_HINT = (
    "point it at a public http(s) or ws(s) endpoint: private, loopback, link-local and internal "
    "service addresses are not reachable"
)
_TOOL_SCHEMES = ("http", "https")
_SERVICE_SCHEMES = ("http", "https", "ws", "wss")
# How long a service host's DNS answer is reused, and how many hosts are kept.
_RESOLUTION_TTL_SECONDS = 60.0
# A failure is kept only long enough to cover the build that runs again after the lookup, so a resolver hiccup does
# not refuse a host for a full minute.
_FAILED_RESOLUTION_TTL_SECONDS = 5.0
_RESOLUTION_CACHE_SIZE = 256
# A build runs once, plus once per host it meets uncached (Deepgram Flux, with two URLs, needs 3).
_MAX_BUILD_ATTEMPTS = 4

# A failed lookup may be a resolver hiccup: the refusal says so instead of reading like a bad configuration.
_UNRESOLVED = (
    "the host name could not be resolved (DNS failure; retry if the host is correct)"
)

# What a lookup answered: the addresses, or None when the name could not be resolved.
Resolution = tuple[str, ...] | None


class EgressBlocked(ValueError):
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


# The name the tool callers (custom tools, transfer resolver, MCP, tool management) import; the same class.
ToolUrlBlocked = EgressBlocked


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


def _resolve(ascii_host: str, port: int) -> Resolution:
    """Every address the name maps to, or None when it cannot be resolved. Blocking, so never on an event loop:
    the cache raises its miss first, so this catches any other path (a tool URL checked on the loop)."""
    if _on_event_loop():
        raise RuntimeError(
            "blocking DNS lookup on the event loop: run this check in a worker thread"
        )
    try:
        infos = socket.getaddrinfo(ascii_host, port, type=socket.SOCK_STREAM)
    # UnicodeError: a label the IDNA codec refuses (too long, empty) fails before any lookup.
    except (socket.gaierror, UnicodeError):
        return None
    return tuple(str(info[4][0]) for info in infos)


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


_Key = tuple[str, int]


class ServiceHostUnresolved(Exception):
    """A service URL check met, on an event loop, a host the cache has no answer for.

    Not a ValueError: no caller may read it as a refusal. ``build_resolving_service_hosts`` resolves the host off
    the loop and runs the build again.
    """

    def __init__(self, host: str, port: int):
        super().__init__(
            f"service host '{host}' port {port} is not resolved yet: build the service through "
            "build_resolving_service_hosts"
        )
        self.host = host
        self.port = port


class _ResolutionCache:
    """What DNS answered for a (host, port), reused for a short TTL, LRU-bounded, thread-safe.

    It stores answers, never verdicts: every address is judged again on each check, and an expired entry is looked
    up again. A failure is kept too, for a shorter TTL, so the build that runs again after the lookup refuses it
    without a second one. A miss on a thread with no running event loop resolves inline; a miss on an event loop
    (the synchronous service builders run there) raises ``ServiceHostUnresolved`` at once, with no lookup. The
    defaults are the service URL check's settings.
    """

    def __init__(
        self,
        *,
        ttl: float = _RESOLUTION_TTL_SECONDS,
        failure_ttl: float = _FAILED_RESOLUTION_TTL_SECONDS,
        size: int = _RESOLUTION_CACHE_SIZE,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._ttl = ttl
        self._failure_ttl = failure_ttl
        self._size = size
        self._clock = clock
        self._entries: OrderedDict[_Key, tuple[float, Resolution]] = OrderedDict()
        self._lock = threading.Lock()

    def resolve(self, ascii_host: str, port: int) -> Resolution:
        key = (ascii_host, port)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] > self._clock():
                self._entries.move_to_end(key)
                return entry[1]
        if _on_event_loop():
            raise ServiceHostUnresolved(ascii_host, port)
        # Outside the lock: lookups of different hosts run in parallel.
        resolution = _resolve(ascii_host, port)
        with self._lock:
            ttl = self._failure_ttl if resolution is None else self._ttl
            self._entries[key] = (self._clock() + ttl, resolution)
            self._entries.move_to_end(key)
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)
        return resolution

    def clear(self) -> None:
        """Forget every answer."""
        with self._lock:
            self._entries.clear()


_service_resolutions = _ResolutionCache()


def reset_service_resolutions() -> None:
    """Forget every service host answer the check has cached (a test starts from an empty cache)."""
    _service_resolutions.clear()


def _url_addresses(
    url: str | None,
    field: str,
    hint: str,
    schemes: tuple[str, ...],
    resolve: Callable[[str, int], Resolution],
) -> tuple[str, tuple[str, ...]]:
    """Parse the URL the way the request will, and return its host with every address that host maps to."""

    def blocked(host: str, reason: str) -> EgressBlocked:
        return EgressBlocked(host, reason, field=field, hint=hint)

    if not url:
        raise blocked("(none)", "the URL has no host")
    try:
        parsed = httpx.URL(url)
        # httpx decodes punycode when the host is read: an invalid A-label raises IDNAError (a UnicodeError) here.
        host = parsed.host.rstrip(".")  # httpx already lower-cases it
        # The name the request will use (IDNA 2008, as httpx encodes it). getaddrinfo on the Unicode name would
        # encode IDNA 2003 and resolve another domain for deviation characters (sharp s).
        ascii_host = parsed.raw_host.decode("ascii").rstrip(".")
    except (httpx.InvalidURL, UnicodeError):
        raise blocked("(unparseable)", "the URL cannot be parsed") from None
    if parsed.scheme not in schemes:
        raise blocked(
            host or "(none)",
            f"scheme '{parsed.scheme}' is not one of {', '.join(schemes)}",
        )
    if not host:
        raise blocked("(none)", "the URL has no host")
    if host in _INTERNAL_HOSTNAMES:
        raise blocked(host, "the host is an internal service name")
    if _ip_literal(host):
        return host, (host,)
    port = parsed.port or (443 if parsed.scheme in ("https", "wss") else 80)
    addresses = resolve(ascii_host, port)
    if addresses is None:
        raise blocked(host, _UNRESOLVED)
    return host, addresses


def _check_url(
    url: str | None,
    *,
    field: str,
    hint: str,
    legacy_check: bool,
    schemes: tuple[str, ...],
    resolve: Callable[[str, int], Resolution] = _resolve,
) -> str | None:
    """The guard core shared by tool and service URLs; return the address the request must connect to, or None
    to connect as usual. Blocking: it resolves the host name.

    URL_GUARD_ENFORCE=true enforces whatever DEPLOYMENT_MODE says: the scheme must be one of ``schemes``, every
    address the host maps to must be public (one bad record blocks), and the returned one is the address to
    connect to, so the request does not resolve the name again. Guard off: the pre-existing DEPLOYMENT_MODE check
    runs, unchanged, when ``legacy_check`` (a path that never had it, MCP, passes False and gets no check).
    """
    if not url_guard_enforced():
        if legacy_check:
            try:
                validate_user_configured_service_url(url, field_name=field)
            except ValueError as e:
                raise EgressBlocked(
                    "(not checked)", str(e), field=field, hint=hint, enforced=False
                ) from e
        return None
    host, addresses = _url_addresses(url, field, hint, schemes, resolve)
    if not addresses:
        raise EgressBlocked(
            host, "the host name has no address", field=field, hint=hint
        )
    for address in addresses:
        ip = _ip_literal(address)
        if ip is None or _is_internal_address(ip):  # unreadable is not vouched for
            logger.warning(f"Egress guard blocked '{host}': it maps to {address}")
            raise EgressBlocked(
                host,
                "the host maps to a private, loopback, link-local or otherwise internal address",
                field=field,
                hint=hint,
            )
    return addresses[0]


def check_tool_url(
    url: str | None,
    *,
    field: str = "url",
    hint: str = _DEFAULT_HINT,
    legacy_check: bool = True,
) -> str | None:
    """Guard a tool URL (custom tools, transfer resolver, MCP): http and https only. See ``_check_url``; guard_egress
    runs it off the event loop."""
    return _check_url(
        url, field=field, hint=hint, legacy_check=legacy_check, schemes=_TOOL_SCHEMES
    )


class _Pinning:
    """What a guarded client's transport does to each request: send it to the one validated address.

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

    def _pin(self, request: Any) -> Any:
        host = request.url.raw_host.decode("ascii")
        if host != self._host:
            raise self._http.ConnectError(
                f"'{host}' is not the validated host '{self._host}'", request=request
            )
        extensions = dict(request.extensions)
        if not _ip_literal(self._host):
            extensions["sni_hostname"] = self._host
        # A new request object: the client keeps the original (host name URL) to judge redirects against.
        return self._http.Request(
            request.method,
            request.url.copy_with(host=self._address),
            headers=request.headers,
            stream=request.stream,
            extensions=extensions,
        )


class _PinnedTransportBase(_Pinning):
    """The async transport of a guarded client (see ``_Pinning``)."""

    def __init__(self, host: str, address: str):
        super().__init__(host, address)
        self._inner = self._http.AsyncHTTPTransport(trust_env=False)

    async def handle_async_request(self, request: Any) -> Any:
        return await self._inner.handle_async_request(self._pin(request))

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


def _egress_target(url: str, address: str | None) -> EgressTarget:
    """The target for a checked URL: no address (guard off) is the empty target."""
    if address is None:
        return EgressTarget()
    return EgressTarget(httpx.URL(url).raw_host.decode("ascii"), address)


def check_service_url(url: str | None, *, field: str) -> EgressTarget:
    """Guard a service URL (model providers, embeddings, STT/TTS endpoints) a tenant configured.

    Same rules as ``check_tool_url`` with ws and wss admitted, for the providers that connect by websocket. Off the
    loop it may wait for DNS unless the host's answer is cached; on an event loop a cache miss raises
    ``ServiceHostUnresolved`` (async callers build through ``build_resolving_service_hosts``). Guard off: the DEPLOYMENT_MODE check runs and the target is empty. ``url`` is the exact
    string the client receives: a caller whose client normalises the stored value passes the normalised one
    (``api.services.configuration.client_urls``).
    """
    address = _check_url(
        url,
        field=field,
        hint=_SERVICE_HINT,
        legacy_check=True,
        schemes=_SERVICE_SCHEMES,
        resolve=_service_resolutions.resolve,
    )
    return _egress_target(url, address)


async def build_resolving_service_hosts[T](
    builder: Callable[..., T], /, *args: Any, **kwargs: Any
) -> T:
    """Run a synchronous service builder on the event loop; each host its URL checks meet uncached is resolved off
    the loop, and the builder runs again. Every async caller of a builder that checks service URLs goes through it.

    A builder must check its URLs before it builds anything with side effects (a session, a cache): a re-run repeats
    what comes before the check. Guard off: no lookup, no miss, one run.
    """
    attempt = 1
    while True:
        try:
            return builder(*args, **kwargs)
        except ServiceHostUnresolved as miss:
            if attempt == _MAX_BUILD_ATTEMPTS:
                name = getattr(builder, "__name__", repr(builder))
                raise RuntimeError(
                    f"{name} still met an unresolved service host after {attempt} attempts"
                ) from miss
            await asyncio.to_thread(_service_resolutions.resolve, miss.host, miss.port)
        attempt += 1


async def guard_egress(
    url: str, *, field: str, hint: str, legacy_check: bool = True
) -> EgressTarget:
    """Guard ``url`` and return where its request goes; raises EgressBlocked (both modes). See check_tool_url."""
    address = await asyncio.to_thread(
        check_tool_url, url, field=field, hint=hint, legacy_check=legacy_check
    )
    return _egress_target(url, address)
