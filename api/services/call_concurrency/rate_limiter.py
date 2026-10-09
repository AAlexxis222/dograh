import time
import uuid
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger
from redis.exceptions import RedisError

from api.constants import REDIS_URL
from api.services.call_concurrency import keys
from api.services.runtime.durations import Durations, cell_durations

# Fleet-wide mirror of every live slot (member = attempt_id, score = expiry in
# Redis time, see keys.py), maintained by the acquire/renew/release paths
# alongside the per-org sets so the autoscaling scrape
# (get_fleet_concurrent_count) is a single ZCOUNT instead of a keyspace scan.
# Deliberately outside the "concurrent_calls:" prefix so it can never collide
# with an org or scope counter key.
FLEET_CONCURRENT_KEY = "concurrent_calls_fleet"


# "now" of every slot script: Redis's own clock, never a process clock (VOZ-AC-B3-65). Redis >= 7 and Valkey
# replicate a script's effects, not the script, so reading TIME before writing needs no
# redis.replicate_commands(). KEYS[1] is a test-only clock; production passes '' and never reads one.
_REDIS_NOW = """
local function redis_now(clock)
    if clock ~= '' then
        local fake = redis.call('GET', clock)
        if fake then
            return tonumber(fake)
        end
    end
    local t = redis.call('TIME')
    return tonumber(t[1]) + tonumber(t[2]) / 1000000
end
"""

# Admission (phase 1 of the lease): the member is added with score now + pending_ttl; claim/renew moves it to
# now + slot_ttl. Expired members (score <= now) are purged before counting.
_ACQUIRE_SLOT = (
    _REDIS_NOW
    + """
local now = redis_now(KEYS[1])
local org, scope, fleet = KEYS[2], KEYS[3], KEYS[4]
local member = ARGV[1]
local max_concurrent = tonumber(ARGV[2])
local scope_max_concurrent = tonumber(ARGV[3])
local pending_ttl = tonumber(ARGV[4])
local slot_ttl = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', org, '-inf', now)
local current_count = redis.call('ZCARD', org)
if current_count >= max_concurrent then
    return nil
end

local expires = now + pending_ttl
if scope ~= '' then
    redis.call('ZREMRANGEBYSCORE', scope, '-inf', now)
    if redis.call('ZCARD', scope) >= scope_max_concurrent then
        return nil
    end
    redis.call('ZADD', scope, expires, member)
    redis.call('EXPIRE', scope, slot_ttl)
end

redis.call('ZADD', org, expires, member)
redis.call('EXPIRE', org, slot_ttl)

-- Mirror into the fleet-wide set (autoscaling signal); expired members are pruned here.
redis.call('ZREMRANGEBYSCORE', fleet, '-inf', now)
redis.call('ZADD', fleet, expires, member)
redis.call('EXPIRE', fleet, slot_ttl)
return {member, current_count + 1}
"""
)

# Claim (phase 2) and every renewal (VOZ-AC-B3-65-bis, -66): ZADD without XX, so a slot a Redis restart lost
# comes back, and the mapping is re-created if missing;
# every EXPIRE is refreshed.
_RENEW_SLOT = (
    _REDIS_NOW
    + """
local now = redis_now(KEYS[1])
local mapping = KEYS[5]
local member = ARGV[1]
local slot_ttl = tonumber(ARGV[2])
local expires = now + slot_ttl

for _, set in ipairs({KEYS[2], KEYS[3], KEYS[4]}) do
    if set ~= '' then
        redis.call('ZADD', set, expires, member)
        redis.call('EXPIRE', set, slot_ttl)
    end
end
if mapping ~= '' then
    if redis.call('EXISTS', mapping) == 0 then
        redis.call('HSET', mapping, 'org_id', ARGV[3], 'slot_id', member)
        if ARGV[4] ~= '' then
            redis.call('HSET', mapping, 'scope_key', ARGV[4])
        end
    end
    redis.call('EXPIRE', mapping, slot_ttl)
end
return 1
"""
)

# The single release (VOZ-AC-B3-67): the slot in every set, the semaphore members registered in the mapping
# ("sem:<zset>" fields, none written yet) and the mapping itself, atomically. The mapping goes only with the
# attempt that owns it. Idempotent: returns 0 when the org member was already gone.
_RELEASE_SLOT = """
local org, scope, fleet, mapping = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local member = ARGV[1]
local removed = redis.call('ZREM', org, member)
if scope ~= '' then
    redis.call('ZREM', scope, member)
end
redis.call('ZREM', fleet, member)
if mapping ~= '' and redis.call('HGET', mapping, 'slot_id') == member then
    local fields = redis.call('HGETALL', mapping)
    for i = 1, #fields, 2 do
        if string.sub(fields[i], 1, 4) == 'sem:' then
            redis.call('ZREM', string.sub(fields[i], 5), fields[i + 1])
        end
    end
    redis.call('DEL', mapping)
end
return removed
"""

_COUNT_SLOTS = (
    _REDIS_NOW
    + """
local now = redis_now(KEYS[1])
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
return redis.call('ZCARD', KEYS[2])
"""
)

# Read-only: counts the unexpired members without purging.
_COUNT_FLEET = (
    _REDIS_NOW
    + """
local now = redis_now(KEYS[1])
return redis.call('ZCOUNT', KEYS[2], string.format('(%.17g', now), '+inf')
"""
)


class AdmissionBackendUnavailable(Exception):
    """VOZ-AC-B3-65-bis: admission fails closed when Redis cannot answer, with its own reason
    (not "concurrent call limit", which would be false). VOZ-AC-B0-28 shape: reason, where, hint."""

    reason = "admission_backend_unavailable"
    where = "call_concurrency.acquire_slot"
    hint = "check Redis (REDIS_URL): admission stays closed until it answers"


@dataclass(frozen=True)
class ConcurrentSlotAcquisition:
    slot_id: str
    active_count: int


@dataclass(frozen=True)
class SlotLease:
    """What a worker holds for a claimed slot: enough to renew it even after Redis lost every key."""

    org_id: int
    attempt_id: str
    scope_key: str | None


class RateLimiter:
    """Sliding window rate limiter to enforce strict per-second limits and concurrent call limits"""

    def __init__(
        self, durations: Durations | None = None, *, test_clock: str | None = None
    ):
        self.redis_client: Optional[aioredis.Redis] = None
        self.durations = durations or cell_durations()
        # The slot outlives the longest call by the slot margin (durations.py): never purged mid-call.
        self.stale_call_timeout = self.durations.slot_ttl
        # Tests only: the slot scripts read "now" from this Redis key instead of TIME. Nothing in the app passes
        # it, so production scripts get '' and never read a clock key, whatever exists in Redis.
        self._test_clock = test_clock or ""

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
        """
        Try to acquire a concurrent call slot.
        Returns a unique slot_id if successful, None if limit reached.
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
        """
        Try to acquire a concurrent call slot under a fresh attempt id
        (delegates to ``acquire_slot``). Returns the slot_id and post-acquire
        active count if successful, or None if the limit is reached; raises
        ``AdmissionBackendUnavailable`` when Redis cannot answer.
        """
        return await self.acquire_slot(
            org_id=organization_id,
            attempt_id=keys.new_attempt_id(),
            max_concurrent=max_concurrent,
            scope_key=scope_key,
            scope_max_concurrent=scope_max_concurrent,
        )

    async def acquire_slot(
        self,
        *,
        org_id: int,
        attempt_id: str,
        max_concurrent: int,
        scope_key: str | None = None,
        scope_max_concurrent: int | None = None,
    ) -> Optional[ConcurrentSlotAcquisition]:
        """Admit an attempt: a pending slot (score = Redis TIME + pending_ttl_s) in the org set, the optional
        scope set (``campaign:<id>``, bounded by ``scope_max_concurrent``) and the fleet set, atomically.

        None when a limit is reached. A Redis error fails closed with ``AdmissionBackendUnavailable``
        (VOZ-AC-B3-65-bis) instead of reading as a full org. The worker must ``claim_slot`` within
        pending_ttl_s or the slot expires on its own (two-phase lease).
        """
        try:
            redis_client = await self._get_redis()
            result = await redis_client.eval(
                _ACQUIRE_SLOT,
                4,
                self._test_clock,
                keys.org_key(org_id),
                keys.scope_key(scope_key),
                FLEET_CONCURRENT_KEY,
                attempt_id,
                max_concurrent,
                scope_max_concurrent if scope_max_concurrent is not None else 0,
                self.durations.pending_ttl_s,
                self.durations.slot_ttl,
            )
        except (RedisError, OSError) as e:
            raise AdmissionBackendUnavailable(
                f"slot acquisition for org {org_id} failed: {e}"
            ) from e
        if not result:
            return None
        acquired_slot_id, active_count = result
        return ConcurrentSlotAcquisition(
            slot_id=str(acquired_slot_id), active_count=int(active_count)
        )

    async def claim_slot(self, workflow_run_id: int) -> SlotLease | None:
        """Phase 2 of the lease, when the worker takes the call: the run's slot moves from pending to
        TIME + slot_ttl. None when the run holds no slot. Redis errors propagate (the caller retries)."""
        redis_client = await self._get_redis()
        mapping = await redis_client.hgetall(keys.mapping_key(workflow_run_id))
        if "slot_id" not in mapping:
            return None
        lease = SlotLease(
            org_id=int(mapping["org_id"]),
            attempt_id=mapping["slot_id"],
            scope_key=mapping.get("scope_key") or None,
        )
        await self.renew_slot(
            org_id=lease.org_id,
            attempt_id=lease.attempt_id,
            scope_key=lease.scope_key,
            workflow_run_id=workflow_run_id,
        )
        return lease

    async def renew_slot(
        self,
        *,
        org_id: int,
        attempt_id: str,
        scope_key: str | None = None,
        workflow_run_id: int | None = None,
    ) -> None:
        """Score = TIME + slot_ttl on every set, re-added if Redis lost it; EXPIRE of the sets and of the
        mapping refreshed, the mapping re-created if missing. Redis errors propagate."""
        redis_client = await self._get_redis()
        await redis_client.eval(
            _RENEW_SLOT,
            5,
            self._test_clock,
            keys.org_key(org_id),
            keys.scope_key(scope_key),
            FLEET_CONCURRENT_KEY,
            keys.mapping_key(workflow_run_id),
            attempt_id,
            self.durations.slot_ttl,
            org_id,
            scope_key or "",
        )

    async def release_slot(
        self,
        *,
        org_id: int,
        attempt_id: str,
        scope_key: str | None = None,
        workflow_run_id: int | None = None,
    ) -> bool | None:
        """The release funnel (VOZ-AC-B3-67): one atomic script for the org, scope and fleet members, the
        semaphore members in the run's mapping and the mapping itself (only if this attempt owns it).

        True if the slot was released, False if it was already gone (released or expired), None on a Redis
        error: nothing was released, so callers keep the mapping around for a retry.
        """
        if not attempt_id:
            return False
        try:
            redis_client = await self._get_redis()
            removed = await redis_client.eval(
                _RELEASE_SLOT,
                4,
                keys.org_key(org_id),
                keys.scope_key(scope_key),
                FLEET_CONCURRENT_KEY,
                keys.mapping_key(workflow_run_id),
                attempt_id,
            )
        except Exception as e:
            logger.error(f"Error releasing concurrent slot: {e}")
            return None
        if removed:
            logger.debug(f"Released concurrent slot {attempt_id} for org {org_id}")
        return bool(removed)

    async def release_concurrent_slot(
        self,
        organization_id: int,
        slot_id: str,
        scope_key: str | None = None,
    ) -> bool | None:
        """Release a slot through the funnel (``release_slot``); same return contract."""
        return await self.release_slot(
            org_id=organization_id, attempt_id=slot_id, scope_key=scope_key
        )

    async def get_concurrent_count(
        self, organization_id: int, *, raise_on_error: bool = False
    ) -> int:
        """
        Get current number of active concurrent calls for an organization.
        Purges the expired members (score <= Redis TIME) before counting.

        Public status reads set ``raise_on_error`` so an unavailable count is
        not reported as zero. The default preserves existing admission logging.
        """
        try:
            redis_client = await self._get_redis()
            return await redis_client.eval(
                _COUNT_SLOTS, 2, self._test_clock, keys.org_key(organization_id)
            )
        except Exception as e:
            logger.error(f"Error getting concurrent count: {e}")
            if raise_on_error:
                raise
            return 0

    async def get_fleet_concurrent_count(self) -> int:
        """Total active calls across every org — the fleet-wide autoscaling signal.

        One ZCOUNT over FLEET_CONCURRENT_KEY, the fleet-wide mirror the
        acquire/release paths maintain alongside the per-org counters — no
        keyspace scan, no per-org fan-out, and scrape cost is independent of
        whatever else lives in this (shared) Redis. Counting by score (not
        ZCARD) excludes expired slots (score = expiry <= Redis TIME) without
        writing, so an orphaned call can't keep the metric high and block
        scale-down, matching the org counters' expiry semantics.

        Unlike the sibling methods, Redis errors are NOT swallowed here: for an
        autoscaling signal, 0 is the most aggressive scale-down instruction, so
        a failed read must surface as an error (the autoscale-metric endpoint
        turns it into a 503) rather than masquerade as an idle fleet.
        """
        redis_client = await self._get_redis()
        return await redis_client.eval(
            _COUNT_FLEET, 2, self._test_clock, FLEET_CONCURRENT_KEY
        )

    async def store_workflow_slot_mapping(
        self, workflow_run_id: int, organization_id: int, slot_id: str
    ) -> bool:
        """
        Store the mapping between workflow_run_id and its concurrent slot.
        Used for cleanup when calls complete.
        """
        redis_client = await self._get_redis()
        mapping_key = keys.mapping_key(workflow_run_id)

        try:
            # Store as a hash with TTL
            await redis_client.hset(
                mapping_key, mapping={"org_id": organization_id, "slot_id": slot_id}
            )
            # Set expiry to match stale timeout
            await redis_client.expire(mapping_key, self.stale_call_timeout)
            return True
        except Exception as e:
            logger.error(f"Error storing workflow slot mapping: {e}")
            return False

    async def store_workflow_slot_mapping_if_absent(
        self,
        workflow_run_id: int,
        organization_id: int,
        slot_id: str,
        scope_key: str | None = None,
    ) -> bool:
        """
        Store the workflow_run_id -> concurrent slot mapping only if no mapping
        already exists. This prevents duplicate public/WebRTC starts for the
        same workflow run from overwriting the cleanup pointer.
        """
        redis_client = await self._get_redis()
        mapping_key = keys.mapping_key(workflow_run_id)

        lua_script = """
        local key = KEYS[1]
        local org_id = ARGV[1]
        local slot_id = ARGV[2]
        local ttl = tonumber(ARGV[3])
        local scope_key = ARGV[4]

        if redis.call('EXISTS', key) == 1 then
            return 0
        end

        redis.call('HSET', key, 'org_id', org_id, 'slot_id', slot_id)
        if scope_key ~= '' then
            redis.call('HSET', key, 'scope_key', scope_key)
        end
        redis.call('EXPIRE', key, ttl)
        return 1
        """

        try:
            stored = await redis_client.eval(
                lua_script,
                1,
                mapping_key,
                organization_id,
                slot_id,
                self.stale_call_timeout,
                scope_key or "",
            )
            return bool(stored)
        except Exception as e:
            logger.error(f"Error storing workflow slot mapping if absent: {e}")
            return False

    async def get_workflow_slot_mapping(
        self, workflow_run_id: int
    ) -> Optional[tuple[int, str, str | None]]:
        """
        Get the concurrent slot mapping for a workflow run.
        Returns (organization_id, slot_id, scope_key) or None if not found;
        scope_key is None for slots acquired without a scope counter.
        """
        redis_client = await self._get_redis()
        mapping_key = keys.mapping_key(workflow_run_id)

        try:
            mapping = await redis_client.hgetall(mapping_key)
            if mapping and "org_id" in mapping and "slot_id" in mapping:
                return (
                    int(mapping["org_id"]),
                    mapping["slot_id"],
                    mapping.get("scope_key") or None,
                )
            return None
        except Exception as e:
            logger.error(f"Error getting workflow slot mapping: {e}")
            return None

    async def reconcile_workflow_slot_mapping(
        self,
        workflow_run_id: int,
        *,
        organization_id: int,
        slot_id: str | None = None,
        scope_key: str | None = None,
    ) -> None:
        """Durable cleanup, including after the workflow mapping's TTL expires.

        Errors propagate so the DB cleanup marker stays pending until every
        counter and the mapping have been cleared. Repeating cleanup is safe:
        every slot goes through the release funnel, which drops the mapping
        together with the slot that owns it.
        """
        redis_client = await self._get_redis()
        mapping = await redis_client.hgetall(keys.mapping_key(workflow_run_id))
        slots = {(slot_id, scope_key)} if slot_id else set()
        if mapping:
            if int(mapping["org_id"]) != organization_id:
                raise ValueError(
                    f"Slot mapping organization mismatch for run {workflow_run_id}"
                )
            slots.add((mapping["slot_id"], mapping.get("scope_key") or None))
        for reserved_slot_id, reserved_scope in slots:
            released = await self.release_slot(
                org_id=organization_id,
                attempt_id=reserved_slot_id,
                scope_key=reserved_scope,
                workflow_run_id=workflow_run_id,
            )
            if released is None:
                raise ConnectionError(f"Slot cleanup failed for run {workflow_run_id}")

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
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None


# Global rate limiter instance
rate_limiter = RateLimiter()
