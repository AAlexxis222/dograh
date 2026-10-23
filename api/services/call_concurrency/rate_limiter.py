import time
import uuid
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger

from api.constants import REDIS_URL
from api.services.call_concurrency import keys
from api.services.call_concurrency.slots import (
    ConcurrentSlotAcquisition,
    SlotBackendError,
    SlotStore,
    slot_store,
)
from api.services.runtime.durations import Durations, cell_durations


class RateLimiter:
    """Sliding window rate limiter to enforce strict per-second limits, plus caller-ID rotation.

    The call slots live in ``slots.SlotStore`` (``self.slots``); the upstream slot methods below forward to it.
    """

    def __init__(
        self, durations: Durations | None = None, *, slots: SlotStore | None = None
    ):
        self.redis_client: Optional[aioredis.Redis] = None
        self.durations = durations or cell_durations()
        # compat: the upstream name of slot_ttl (the slot outlives the longest call by the slot margin).
        self.stale_call_timeout = self.durations.slot_ttl
        self.slots = slots or SlotStore(self.durations)

    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection"""
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(
                REDIS_URL, decode_responses=True
            )
        return self.redis_client

    async def acquire_token(
        self,
        organization_id: int,
        rate_limit: int = 1,
        *,
        scope_key: str | None = None,
    ) -> bool:
        """
        Enforces strict rate limit: max N calls per rolling second window
        Returns True if allowed, False if rate limited

        ``scope_key`` creates an isolated rate-limit bucket without touching
        organization concurrency counters or fleet-wide call metrics.
        """
        redis_client = await self._get_redis()

        key = f"rate_limit:{scope_key or organization_id}"
        now = time.time()
        window_start = now - 1.0  # 1 second sliding window

        # Lua script for atomic sliding window operation
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])
        
        -- Remove timestamps older than window
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        
        -- Count requests in current window
        local current_requests = redis.call('ZCARD', key)
        
        if current_requests < max_requests then
            -- Add current timestamp
            redis.call('ZADD', key, now, ARGV[4])
            redis.call('EXPIRE', key, 2)  -- Expire after 2 seconds
            return 1
        else
            return 0
        end
        """

        try:
            result = await redis_client.eval(
                lua_script, 1, key, now, window_start, rate_limit, uuid.uuid4().hex
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # On error, be conservative and deny
            return False

    async def get_next_available_slot(
        self, organization_id: int, rate_limit: int = 1, *, scope_key: str | None = None
    ) -> float:
        """
        Returns seconds until next available slot
        Useful for implementing retry with backoff
        """
        redis_client = await self._get_redis()

        key = f"rate_limit:{scope_key or organization_id}"

        try:
            # Get oldest timestamp in current window
            oldest = await redis_client.zrange(key, 0, 0, withscores=True)
            if not oldest:
                return 0.0  # Can call immediately

            oldest_time = oldest[0][1]
            next_available = oldest_time + 1.0  # 1 second after oldest
            wait_time = max(0, next_available - time.time())

            return wait_time
        except Exception as e:
            logger.error(f"Rate limiter get_next_available_slot error: {e}")
            return 1.0  # Default wait time on error

    async def try_acquire_concurrent_slot(
        self, organization_id: int, max_concurrent: int = 20
    ) -> Optional[str]:
        """compat: prefer acquire_slot/release_slot (slots.SlotStore).

        Try to acquire a concurrent call slot. Returns a unique slot_id if
        successful, None if limit reached; raises ``SlotBackendError`` when
        Redis cannot answer.
        """
        acquisition = await self.try_acquire_concurrent_slot_details(
            organization_id, max_concurrent
        )
        return acquisition.slot_id if acquisition else None

    async def try_acquire_concurrent_slot_details(
        self,
        organization_id: int,
        max_concurrent: int = 20,
        *,
        scope_key: str | None = None,
        scope_max_concurrent: int | None = None,
    ) -> Optional[ConcurrentSlotAcquisition]:
        """compat: prefer acquire_slot/release_slot (slots.SlotStore); this one
        cannot pass the outbound carrier, so an outbound call gets the inbound
        pending lease.

        Acquire a concurrent call slot under a fresh attempt id. Returns the
        slot_id and post-acquire active count if successful, or None if the
        limit is reached; raises ``SlotBackendError`` when Redis cannot answer.
        """
        return await self.slots.acquire_slot(
            org_id=organization_id,
            attempt_id=keys.new_attempt_id(),
            max_concurrent=max_concurrent,
            scope_key=scope_key,
            scope_max_concurrent=scope_max_concurrent,
        )

    async def release_concurrent_slot(
        self,
        organization_id: int,
        slot_id: str,
        scope_key: str | None = None,
    ) -> bool:
        """compat: prefer acquire_slot/release_slot (slots.SlotStore).

        Release a slot through the funnel: True if released, False if it was
        already gone; raises ``SlotBackendError`` when Redis cannot answer.
        """
        return await self.slots.release_slot(
            org_id=organization_id, attempt_id=slot_id, scope_key=scope_key
        )

    async def get_concurrent_count(
        self, organization_id: int, *, raise_on_error: bool = False
    ) -> int:
        """compat: prefer slots.SlotStore.get_concurrent_count, which always
        raises ``SlotBackendError`` when Redis cannot answer.

        Active calls of an organization. Keeps the upstream contract of its
        flag: ``raise_on_error`` raises the backend error, the default logs it
        and reads 0.
        """
        try:
            return await self.slots.get_concurrent_count(organization_id)
        except SlotBackendError as e:
            logger.error(f"Error getting concurrent count: {e}")
            if raise_on_error:
                raise
            return 0

    async def get_fleet_concurrent_count(self) -> int:
        """compat: prefer slots.SlotStore.get_fleet_concurrent_count.

        Total active calls across every org; raises ``SlotBackendError`` when
        Redis cannot answer (an idle fleet must not be reported on an error).
        """
        return await self.slots.get_fleet_concurrent_count()

    async def store_workflow_slot_mapping(
        self,
        workflow_run_id: int,
        organization_id: int,
        slot_id: str,
        *,
        scope_key: str | None = None,
        max_concurrent: int | None = None,
        scope_max_concurrent: int | None = None,
    ) -> bool:
        """compat: prefer acquire_slot/release_slot (slots.SlotStore).

        Bind a run to its slot through the if-absent writer (False when the
        run already has a mapping). Pass the admission limits, or a late claim
        of this run re-adds its slot without re-checking them.
        """
        return await self.slots.store_workflow_slot_mapping_if_absent(
            workflow_run_id,
            organization_id,
            slot_id,
            scope_key,
            max_concurrent=max_concurrent,
            scope_max_concurrent=scope_max_concurrent,
        )

    async def store_workflow_slot_mapping_if_absent(
        self,
        workflow_run_id: int,
        organization_id: int,
        slot_id: str,
        scope_key: str | None = None,
        *,
        max_concurrent: int | None = None,
        scope_max_concurrent: int | None = None,
    ) -> bool:
        """compat: prefer acquire_slot/release_slot (slots.SlotStore).

        Bind a run to its slot unless a mapping already exists (False then);
        raises ``SlotBackendError`` when Redis cannot answer.
        """
        return await self.slots.store_workflow_slot_mapping_if_absent(
            workflow_run_id,
            organization_id,
            slot_id,
            scope_key,
            max_concurrent=max_concurrent,
            scope_max_concurrent=scope_max_concurrent,
        )

    async def get_workflow_slot_mapping(
        self, workflow_run_id: int
    ) -> Optional[tuple[int, str, str | None]]:
        """compat: prefer acquire_slot/release_slot (slots.SlotStore).

        (organization_id, slot_id, scope_key) of the run, or None when it has
        no mapping; raises ``SlotBackendError`` when Redis cannot answer.
        """
        return await self.slots.get_workflow_slot_mapping(workflow_run_id)

    async def reconcile_workflow_slot_mapping(
        self,
        workflow_run_id: int,
        *,
        organization_id: int,
        slot_id: str | None = None,
        scope_key: str | None = None,
    ) -> None:
        """compat: prefer acquire_slot/release_slot (slots.SlotStore).

        Durable cleanup of a run's slots and mapping; raises (``SlotBackendError``
        on a backend error) so the DB cleanup marker stays pending.
        """
        await self.slots.reconcile_workflow_slot_mapping(
            workflow_run_id,
            organization_id=organization_id,
            slot_id=slot_id,
            scope_key=scope_key,
        )

    async def select_from_number(
        self,
        organization_id: int,
        telephony_configuration_id: int | None,
        from_numbers: list[str],
    ) -> str | None:
        """Rotate active caller IDs without reserving them for a call's lifetime."""
        numbers = sorted(set(from_numbers))
        if not numbers:
            return None
        key = f"caller_id_rotation:{organization_id}:{telephony_configuration_id}"
        try:
            redis_client = await self._get_redis()
            index = await redis_client.eval(
                "local n = redis.call('INCR', KEYS[1]); "
                "redis.call('EXPIRE', KEYS[1], 86400); return n - 1",
                1,
                key,
            )
            return numbers[int(index) % len(numbers)]
        except Exception as exc:
            logger.warning(
                f"Caller-ID rotation unavailable for config {telephony_configuration_id}: {exc}"
            )
            return numbers[0]

    async def close(self):
        """Close Redis connections (this limiter's and its slot store's)"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        await self.slots.close()


# Global rate limiter instance; its slot methods share the global slot store.
rate_limiter = RateLimiter(slots=slot_store)
