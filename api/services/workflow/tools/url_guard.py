"""Tool URL guard (VOZ-AC-B5-91 layer 2): no tool reaches IMDS or the internal network.

Shared by every tool that calls a tenant-configured URL (custom HTTP tools, the transfer resolver, MCP servers),
so the tools depend on the guard and never on each other.
"""

import ipaddress
import os
import socket
from typing import Any

import httpx
from loguru import logger

from api.utils.url_security import validate_user_configured_service_url

# The guard has its own flag, so a cell (DEPLOYMENT_MODE=oss) can enforce it without changing the other branches
# that mode governs. Read at call time; only the exact value "true" enforces.
URL_GUARD_FLAG = "URL_GUARD_ENFORCE"
# Compose service names (docker-compose.yaml): inside the Docker network they resolve to private addresses.
_INTERNAL_HOSTNAMES = frozenset(
    {"localhost", "postgres", "redis", "minio", "api", "call", "ui", "nginx", "coturn"}
)
# Global unicast IPv6. Everything outside it (IPv4-compatible ::/96, IPv4-translated ::ffff:0:0:0/96, NAT64
# 64:ff9b::/96, site-local fec0::/10, ULA, link-local, multicast) is refused whatever is_global says.
_GLOBAL_UNICAST_V6 = ipaddress.ip_network("2000::/3")
_BLOCKED_HINT = (
    "point the tool at a public http(s) endpoint: private, loopback, link-local and internal "
    "service addresses are not reachable from tools"
)


class ToolUrlBlocked(ValueError):
    """A tool URL the guard refuses, in the VOZ-AC-B0-28 shape: stable code, what, where, reason, hint."""

    code = "tool_url_blocked"
    what = "tool request blocked before it left the platform"

    def __init__(self, host: str, reason: str, field: str = "config.url"):
        self.where = f"{field} host '{host}'"
        self.reason = reason
        self.hint = _BLOCKED_HINT
        super().__init__(
            f"code={self.code} what={self.what} where={self.where} "
            f"reason={reason} hint={self.hint}"
        )

    def as_result(self) -> dict[str, str]:
        return {
            "status": "error",
            "error": str(self),
            "code": self.code,
            "what": self.what,
            "where": self.where,
            "reason": self.reason,
            "hint": self.hint,
        }


def url_guard_enforced() -> bool:
    return os.environ.get(URL_GUARD_FLAG) == "true"


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


def _tool_url_addresses(url: str, field: str) -> tuple[str, list[str]]:
    """Parse the URL the way the request will, and return its host with every address that host maps to."""
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        raise ToolUrlBlocked(
            "(unparseable)", "the URL cannot be parsed", field
        ) from None
    host = parsed.host.rstrip(".")  # httpx already lower-cases it
    if parsed.scheme not in ("http", "https"):
        raise ToolUrlBlocked(
            host or "(none)", f"scheme '{parsed.scheme}' is not http or https", field
        )
    if not host:
        raise ToolUrlBlocked(
            "(none)", "the URL has no host (expected http(s)://host/...)", field
        )
    if host in _INTERNAL_HOSTNAMES:
        raise ToolUrlBlocked(host, "the host is an internal service name", field)
    if _ip_literal(host):
        return host, [host]
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    # UnicodeError: a label the IDNA codec refuses (too long, empty) fails before any lookup.
    except (socket.gaierror, UnicodeError):
        raise ToolUrlBlocked(
            host, "the host name could not be resolved", field
        ) from None
    return host, [str(info[4][0]) for info in infos]


def check_tool_url(url: str, field: str = "config.url") -> str | None:
    """Guard a tool URL; return the address the request must connect to, or None to connect as usual.

    URL_GUARD_ENFORCE=true enforces whatever DEPLOYMENT_MODE says: every address the host maps to must be public
    (one bad record blocks), and the returned one is the address to connect to, so the request does not resolve
    the name again. Otherwise the pre-existing DEPLOYMENT_MODE-based check runs, unchanged (it raises ValueError).
    """
    if not url_guard_enforced():
        validate_user_configured_service_url(url, field_name=field)
        return None
    host, addresses = _tool_url_addresses(url, field)
    if not addresses:
        raise ToolUrlBlocked(host, "the host name has no address", field)
    for address in addresses:
        ip = _ip_literal(address)
        if ip is None or _is_internal_address(ip):  # unreadable is not vouched for
            logger.warning(f"Tool URL guard blocked '{host}': it maps to {address}")
            raise ToolUrlBlocked(
                host,
                "the host maps to a private, loopback, link-local or otherwise internal address",
                field,
            )
    return addresses[0]


def pinned_request_target(
    url: str, headers: dict[str, str], address: str | None
) -> dict[str, Any]:
    """The url, headers and extensions httpx needs to connect to the validated address without resolving again.

    The Host header and, for https, the SNI name and certificate check stay on the original host name. The client
    that sends a pinned request must be built with trust_env=False: an env proxy would tunnel to the IP, and
    httpcore's tunnel ignores sni_hostname, so the certificate check would fail.
    """
    if address is None:
        return {"url": url, "headers": headers}
    parsed = httpx.URL(url)
    if not any(name.lower() == "host" for name in headers):
        headers = {**headers, "Host": parsed.netloc.decode("ascii")}
    target: dict[str, Any] = {
        "url": str(parsed.copy_with(host=address)),
        "headers": headers,
    }
    if not _ip_literal(parsed.host):
        target["extensions"] = {"sni_hostname": parsed.host}
    return target
