"""Mark a workflow run as "ending on purpose", in the control plane (Redis).

The run row only turns ``completed`` at the very end of teardown, after the call
was hung up and the media socket closed. The carrier's callback asks what to do
within milliseconds of that, so the engine marks the run here, before it hangs
up, for the callback to read. This is run-lifecycle state: it has nothing to do
with call transfers.
"""

import asyncio

import redis.asyncio as aioredis
from loguru import logger

from api.constants import REDIS_URL

# Remembered for an hour: far longer than the gap between the media socket
# closing and the carrier's callback asking about it.
RUN_ENDING_TTL_SECONDS = 3600
# ``from_url`` sets no socket timeout, so an unreachable Redis would otherwise
# hold the call open: the mark must cost at most this long.
RUN_ENDING_TIMEOUT_SECONDS = 2.0


def _key(workflow_run_id: int) -> str:
    return f"run:ending:{workflow_run_id}"


class RunEnding:
    """Reads and writes the "ending on purpose" mark through the given Redis client."""

    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client

    async def mark(self, workflow_run_id: int) -> None:
        """Record that the run is ending on purpose.

        Never raises and never waits long: the call must still end when Redis is
        slow or down, and without the mark the caller hears the message, not
        silence.
        """
        try:
            async with asyncio.timeout(RUN_ENDING_TIMEOUT_SECONDS):
                await self._redis.setex(
                    _key(workflow_run_id), RUN_ENDING_TTL_SECONDS, "1"
                )
        except Exception as e:
            logger.error(f"[run {workflow_run_id}] Failed to mark run ending: {e}")

    async def is_marked(self, workflow_run_id: int) -> bool:
        """True when :meth:`mark` was called for the run; False on any error."""
        try:
            async with asyncio.timeout(RUN_ENDING_TIMEOUT_SECONDS):
                return bool(await self._redis.get(_key(workflow_run_id)))
        except Exception as e:
            logger.error(f"[run {workflow_run_id}] Failed to read run-ending mark: {e}")
            return False


_run_ending: RunEnding | None = None


async def get_run_ending() -> RunEnding:
    """The process-wide :class:`RunEnding`, on a lazily created Redis client."""
    global _run_ending
    if _run_ending is None:
        _run_ending = RunEnding(
            await aioredis.from_url(REDIS_URL, decode_responses=True)
        )
    return _run_ending
