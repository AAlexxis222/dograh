"""Single unit that knows the MCP protocol + credentials.

Drives the MCP SDK's streamable-HTTP transport and ``ClientSession`` on an
HTTP client it builds itself (so the tool URL guard can pin it), builds
streamable-HTTP params from a Dograh credential, exposes namespaced
``FunctionSchema``s, and proxies tool calls. Connection failures degrade
(``available = False``) instead of raising — the call must survive a
dead MCP server.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import httpx2
from loguru import logger
from mcp.client.session import ClientSession
from mcp.client.session_group import StreamableHttpParameters
from mcp.client.streamable_http import streamable_http_client
from pipecat.adapters.schemas.function_schema import FunctionSchema

from api.services.security.egress_guard import ToolUrlBlocked, guard_egress
from api.services.workflow.tools.mcp_tool import namespace_function_name
from api.utils.credential_auth import build_auth_header

if TYPE_CHECKING:
    from api.db.models import ExternalCredentialModel


_MCP_URL_HINT = (
    "point the MCP server URL at a public http(s) endpoint: private, loopback, link-local and "
    "internal service addresses are not reachable from tools"
)


def _sole_cause(error: BaseException) -> BaseException:
    """The one exception a group of one wraps, at any depth: a group names neither the type nor the cause."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def build_streamable_http_params(
    *,
    url: str,
    credential: Optional["ExternalCredentialModel"],
    timeout_secs: int,
    sse_read_timeout_secs: int,
) -> StreamableHttpParameters:
    """Build Pipecat/MCP streamable-HTTP params, injecting the auth header
    from an ExternalCredentialModel (reuses the http_api credential path)."""
    headers: Optional[Dict[str, str]] = None
    if credential is not None:
        auth = build_auth_header(credential)
        headers = auth or None
    return StreamableHttpParameters(
        url=url,
        headers=headers,
        timeout=float(timeout_secs),
        sse_read_timeout=float(sse_read_timeout_secs),
    )


class McpToolSession:
    """One live MCP server connection for the duration of a call."""

    def __init__(
        self,
        *,
        tool_uuid: str,
        tool_name: str,
        url: str,
        credential: Optional["ExternalCredentialModel"],
        tools_filter: List[str],
        timeout_secs: int,
        sse_read_timeout_secs: int,
    ) -> None:
        self._tool_uuid = tool_uuid
        self._tool_name = tool_name
        self._url = url
        self._credential = credential
        # An empty list is intentionally treated as "no filter (expose all
        # tools)": the filter applies only when this is a non-empty list, so
        # [] and None are equivalent ("all tools").
        self._tools_filter = tools_filter or None
        self._timeout_secs = timeout_secs
        self._sse_read_timeout_secs = sse_read_timeout_secs

        # Transport, HTTP client and ClientSession, closed together (in the task that opened them).
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Any = None  # mcp.ClientSession
        self._schemas: List[FunctionSchema] = []
        # namespaced LLM name -> original MCP tool name
        self._name_map: Dict[str, str] = {}
        self.available: bool = False
        # Set when the egress guard refused the URL at start(): discovery raises it instead of checking again.
        self.blocked: ToolUrlBlocked | None = None
        self._owner_task: asyncio.Task | None = None
        self._ready: asyncio.Future | None = None
        self._stop = asyncio.Event()

    async def start_managed(self) -> None:
        """Keep MCP's task-affine scopes inside one connection owner task."""
        if self._owner_task is None:
            self._ready = asyncio.get_running_loop().create_future()
            self._owner_task = asyncio.create_task(
                self._own_connection(), name=f"mcp:{self._tool_uuid}"
            )
        try:
            await asyncio.shield(self._ready)
        except BaseException:
            await self.close_managed()
            raise

    async def _own_connection(self) -> None:
        try:
            await self.start()
            self._ready.set_result(None)
            await self._stop.wait()
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            raise
        finally:
            await self.close()

    async def close_managed(self) -> None:
        task = self._owner_task
        if task is None:
            return
        self._stop.set()
        if not self._ready.done():
            task.cancel()
        # The owner closes in the task that opened the connection, even when
        # the preparation task waiting for it was cancelled.
        await asyncio.gather(task, return_exceptions=True)
        if self._ready.done() and not self._ready.cancelled():
            self._ready.exception()  # Consume a cancelled/failed startup result.
        self._owner_task = None

    async def start(self) -> None:
        """Connect, initialize, and cache the tool list.

        Never raises on a connect failure — a dead/unreachable MCP server
        leaves the session marked unavailable (``available = False``). Genuine
        external cancellation, KeyboardInterrupt, and SystemExit are re-raised
        (see the CancelledError handling below and ``_degrade``). A URL the egress
        guard refuses degrades the same way, before any connection, and is kept
        in ``blocked``."""
        try:
            # An MCP server is a tool (VOZ-AC-B5-91). MCP never had the legacy DEPLOYMENT_MODE check, so with the
            # guard off nothing is checked or pinned, as before.
            target = await guard_egress(
                self._url, field="config.url", hint=_MCP_URL_HINT, legacy_check=False
            )
            params = build_streamable_http_params(
                url=self._url,
                credential=self._credential,
                timeout_secs=self._timeout_secs,
                sse_read_timeout_secs=self._sse_read_timeout_secs,
            )
            self._exit_stack = stack = AsyncExitStack()
            client = await stack.enter_async_context(
                target.httpx2_client(
                    headers=params.headers,
                    timeout=httpx2.Timeout(
                        params.timeout, read=params.sse_read_timeout
                    ),
                )
            )
            read_stream, write_stream = await stack.enter_async_context(
                streamable_http_client(
                    params.url,
                    http_client=client,
                    terminate_on_close=params.terminate_on_close,
                )
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            listed = await session.list_tools()

            fallback = self._tool_uuid[:8] if self._tool_uuid else "server"
            for tool in listed.tools:
                if self._tools_filter and tool.name not in self._tools_filter:
                    continue
                ns_name = namespace_function_name(
                    self._tool_name, tool.name, fallback=fallback
                )
                self._name_map[ns_name] = tool.name
                self._schemas.append(
                    FunctionSchema(
                        name=ns_name,
                        description=tool.description,
                        properties=tool.input_schema.get("properties", {}),
                        required=tool.input_schema.get("required", []),
                    )
                )
            self._session = session
            self.available = True
            logger.info(
                f"MCP session ready for tool '{self._tool_name}' "
                f"({self._tool_uuid}): {sorted(self._name_map)}"
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except ToolUrlBlocked as e:
            self.blocked = e
            await self._degrade(e)
        except asyncio.CancelledError as e:
            # A dead/unreachable MCP server may not surface as a plain
            # Exception here. The real failure is httpx.ConnectError, but
            # anyio's streamablehttp_client task group, while tearing down that
            # ConnectError, re-surfaces it to our frame as an *internal*
            # cancel-scope CancelledError carrying the signature message
            # "Cancelled via cancel scope <id>". A genuine *external*
            # cancellation (call teardown / shutdown) is a bare CancelledError
            # (empty args) or one with an application-chosen message. Type, MRO,
            # context chain, and asyncio task.cancelling() are all identical
            # between the two, so the anyio scope-signature message is the only
            # reliable discriminator. Re-raise genuine external cancellation to
            # preserve structured concurrency; degrade only on the anyio
            # connect-teardown artifact.
            msg = "" if not e.args else str(e.args[0] or "")
            if not msg.startswith("Cancelled via cancel scope"):
                raise
            await self._degrade(e)
        except Exception as e:  # noqa: BLE001 — see _degrade docstring
            # The connect failure surfaced directly (an MCPError, an HTTP
            # error) instead of via the anyio cancel-scope artifact above:
            # degrade gracefully.
            await self._degrade(e)

    async def _degrade(self, e: BaseException) -> None:
        """Mark this session unavailable and tear down a half-open connection so
        start() leaves the session either fully usable or closed. The contract
        requires graceful degradation on any *connect* failure (never raising
        for a dead MCP server) while genuine external cancellation /
        KeyboardInterrupt / SystemExit are re-raised by the caller."""
        self.available = False
        self._schemas = []
        self._name_map = {}
        # Self-contained cleanup: the transport may be open when a later step
        # (initialize, list_tools) failed. Closing it also exits the anyio task
        # group whose cancel scope fired, before anything else is awaited, and
        # that exit raises the real cause (ConnectError, ConnectTimeout, a
        # certificate mismatch) when ``e`` is only the cancel-scope artifact.
        closing_error = await self._close_stack()
        cause = _sole_cause(e if isinstance(e, Exception) else (closing_error or e))
        logger.warning(
            f"MCP session unavailable for tool '{self._tool_name}' "
            f"({self._tool_uuid}) at {self._url}: {cause!r}. "
            f"Call proceeds without these tools."
        )

    @property
    def call_timeout_secs(self) -> float:
        """Pipecat function-call timeout for this server's tools. Slightly
        longer than the transport read timeout so a slow MCP call surfaces
        as a structured tool error (handled in the handler) rather than a
        hard pipeline timeout."""
        return float(self._sse_read_timeout_secs) + 5.0

    def function_schemas(
        self, allowed_raw_names: Optional[Set[str]] = None
    ) -> List[FunctionSchema]:
        """Return cached FunctionSchemas, optionally filtered by raw MCP tool name.

        ``allowed_raw_names=None`` returns all schemas. An empty set returns none.
        Raw names are the pre-namespace MCP tool names (e.g. ``echo``, not
        ``mcp__slug__echo``).
        """
        if allowed_raw_names is None:
            return list(self._schemas)
        return [
            s for s in self._schemas if self._name_map.get(s.name) in allowed_raw_names
        ]

    def discovered_tools(self) -> List[Dict[str, str]]:
        """Raw MCP tool catalog for UI/cache: ``[{name, description}]``
        using the *raw* server names (not the namespaced LLM names).
        Empty if the session is unavailable."""
        out: List[Dict[str, str]] = []
        for s in self._schemas:
            raw = self._name_map.get(s.name)
            if raw is None:
                continue
            out.append({"name": raw, "description": s.description or ""})
        return out

    async def call(self, namespaced_name: str, arguments: Dict[str, Any]) -> str:
        """Invoke an MCP tool by its namespaced LLM name. Returns a string
        (flattened text content). Raises if the session is unavailable so
        the caller can map it to a structured error for the LLM."""
        if not self.available or self._session is None:
            raise RuntimeError(f"MCP session unavailable for {namespaced_name}")
        original = self._name_map.get(namespaced_name)
        if original is None:
            raise RuntimeError(f"Unknown MCP function {namespaced_name}")
        result = await self._session.call_tool(original, arguments=arguments)
        text = ""
        for content in getattr(result, "content", []) or []:
            if getattr(content, "text", None):
                text += content.text
        return text or "Sorry, the MCP tool returned no content."

    async def _close_stack(self) -> Exception | None:
        """Close the connection; return what closing it raised (None if nothing)."""
        stack, self._exit_stack, self._session = self._exit_stack, None, None
        if stack is None:
            return None
        try:
            await stack.aclose()
        except Exception as e:
            return e
        return None

    async def close(self) -> None:
        error = await self._close_stack()
        if error is not None:
            logger.warning(f"Error closing MCP session {self._tool_uuid}: {error}")


async def discover_mcp_tools(
    *,
    url: str,
    credential: Optional["ExternalCredentialModel"],
    timeout_secs: int,
    sse_read_timeout_secs: int,
) -> List[Dict[str, str]]:
    """Open an ephemeral MCP session, list its tools, close it. Returns
    ``[{name, description}]`` (raw names). Raises ``ToolUrlBlocked`` when the
    egress guard refuses the URL; on any connect failure returns ``[]``."""
    session = McpToolSession(
        tool_uuid="discover",
        tool_name="discover",
        url=url,
        credential=credential,
        tools_filter=[],
        timeout_secs=timeout_secs,
        sse_read_timeout_secs=sse_read_timeout_secs,
    )
    await session.start()
    try:
        if session.blocked is not None:
            raise session.blocked
        if not session.available:
            return []
        return session.discovered_tools()
    finally:
        await session.close()
