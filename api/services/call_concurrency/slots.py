"""The slot-lease lifecycle in Redis (VOZ-AC-B3-22, -65, -66, -67): admit, claim, extend, renew, release, count and
the run-to-slot mapping, each one atomic Lua script on Redis's clock. Key schema: keys.py.

Error policy at this boundary: every primitive raises ``SlotBackendError`` when Redis does not answer (``RedisError``
or ``OSError``), and nothing else stands for "Redis is down". ``None`` / ``False`` / ``0`` are always real answers
(limit reached, no slot, nothing released, nobody on a call). The caller decides what a backend error means at its
boundary (call_concurrency/service.py): admission fails closed, teardown logs and keeps the mapping.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis
from loguru import logger
from redis.exceptions import RedisError

from api.constants import REDIS_URL
from api.services.call_concurrency import keys
from api.services.runtime.durations import Durations, cell_durations

# "now" of every slot script: Redis's own clock, never a process clock (VOZ-AC-B3-65). Redis >= 7 and Valkey
# replicate a script's effects, not the script, so reading TIME before writing needs no
# redis.replicate_commands(). The clock key is test-only; production passes '' and never reads one.
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

-- Unexpired v2 members of a set, after purging the expired ones.
local function live(set, now)
    redis.call('ZREMRANGEBYSCORE', set, '-inf', now)
    return redis.call('ZCARD', set)
end

-- Members of a LEGACY (pre-v2) set still live under the old rule: score = acquisition time, live while
-- score > now - slot_ttl. Read-only; the legacy sets die on their own expiry (keys.py, follow-up VOZ-N0-21-F1).
local function legacy_live(set, now, slot_ttl)
    if set == '' then
        return 0
    end
    return redis.call('ZCOUNT', set, string.format('(%.17g', now - slot_ttl), '+inf')
end

-- Claim and renewal: ZADD without XX, so a slot a Redis restart lost comes back (VOZ-AC-B3-65-bis), the mapping
-- is re-created if missing, and every expiry of the slot and of the mapping is refreshed (VOZ-AC-B3-66).
local function readd(now, sets, mapping, member, slot_ttl, org_id, scope_field)
    local expires = now + slot_ttl
    for _, set in ipairs(sets) do
        if set ~= '' then
            redis.call('ZADD', set, expires, member)
            redis.call('EXPIRE', set, slot_ttl)
        end
    end
    if mapping ~= '' then
        if redis.call('EXISTS', mapping) == 0 then
            redis.call('HSET', mapping, 'org_id', org_id, 'slot_id', member)
            if scope_field ~= '' then
                redis.call('HSET', mapping, 'scope_key', scope_field)
            end
        end
        redis.call('EXPIRE', mapping, slot_ttl)
    end
end
"""

# Admission (phase 1 of the lease): the member is added with score now + pending_ttl (inbound or outbound);
# claim/renew moves it to now + slot_ttl. Legacy calls still running count toward both limits.
_ACQUIRE_SLOT = (
    _REDIS_NOW
    + """
local clock, org, scope, fleet, legacy_org, legacy_scope = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5], KEYS[6]
local member = ARGV[1]
local max_concurrent = tonumber(ARGV[2])
local scope_max_concurrent = tonumber(ARGV[3])
local pending_ttl = tonumber(ARGV[4])
local slot_ttl = tonumber(ARGV[5])
local now = redis_now(clock)

local current_count = live(org, now) + legacy_live(legacy_org, now, slot_ttl)
if current_count >= max_concurrent then
    return nil
end

local expires = now + pending_ttl
if scope ~= '' then
    if live(scope, now) + legacy_live(legacy_scope, now, slot_ttl) >= scope_max_concurrent then
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

# Claim (phase 2), when the worker takes the call. A member still pending just moves to now + slot_ttl. A member
# that already expired (a late claim) is re-added through the same limit check: the answered call is kept either
# way, but returns overcommit = 1 when the limit was already full. A legacy call is left to its legacy life.
# Returns {late, overcommit, org count after}.
_CLAIM_SLOT = (
    _REDIS_NOW
    + """
local clock, org, scope, fleet, mapping = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5]
local legacy_org, legacy_scope = KEYS[6], KEYS[7]
local member = ARGV[1]
local slot_ttl = tonumber(ARGV[2])
local org_id, scope_field = ARGV[3], ARGV[4]
local max_concurrent, scope_max_concurrent = ARGV[5], ARGV[6]
local now = redis_now(clock)
if legacy_org ~= '' and redis.call('ZSCORE', legacy_org, member) then
    return {0, 0, 0}
end

local score = redis.call('ZSCORE', org, member)
local late = (not score) or tonumber(score) <= now
local overcommit = 0
if late then
    if max_concurrent ~= ''
        and live(org, now) + legacy_live(legacy_org, now, slot_ttl) >= tonumber(max_concurrent) then
        overcommit = 1
    end
    if scope ~= '' and scope_max_concurrent ~= ''
        and live(scope, now) + legacy_live(legacy_scope, now, slot_ttl) >= tonumber(scope_max_concurrent) then
        overcommit = 1
    end
end
readd(now, {org, scope, fleet}, mapping, member, slot_ttl, org_id, scope_field)
return {late and 1 or 0, overcommit, live(org, now) + legacy_live(legacy_org, now, slot_ttl)}
"""
)

# A carrier's initiated/ringing callback for an outbound call that is still pending: its lease restarts at
# now + outbound pending lease (a CPS queue before the ringing can outlast the lease sized at dial). ZADD GT never
# shortens a claimed slot; an already expired member is left to the late claim, which re-checks the limit.
_EXTEND_PENDING = (
    _REDIS_NOW
    + """
local clock, org, scope, fleet = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local member = ARGV[1]
local pending_ttl = tonumber(ARGV[2])
local slot_ttl = ARGV[3]
local now = redis_now(clock)
local expires = now + pending_ttl
local extended = 0
for _, set in ipairs({org, scope, fleet}) do
    local score = set ~= '' and redis.call('ZSCORE', set, member)
    if score and tonumber(score) > now then
        extended = extended + redis.call('ZADD', set, 'GT', 'CH', expires, member)
        redis.call('EXPIRE', set, slot_ttl)
    end
end
return extended
"""
)

# Every renewal of a claimed call. A slot the carrier's terminal callback already released while this pipeline is
# still live comes back here (the call is still holding capacity); the pipeline's own release at its end removes
# it for good. A legacy call is left to its legacy life.
_RENEW_SLOT = (
    _REDIS_NOW
    + """
local clock, org, scope, fleet, mapping, legacy_org = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5], KEYS[6]
local member = ARGV[1]
local slot_ttl = tonumber(ARGV[2])
local org_id, scope_field = ARGV[3], ARGV[4]
local now = redis_now(clock)
if legacy_org ~= '' and redis.call('ZSCORE', legacy_org, member) then
    return 0
end
readd(now, {org, scope, fleet}, mapping, member, slot_ttl, org_id, scope_field)
return 1
"""
)

# The single release (VOZ-AC-B3-67): the slot in every set (and in the legacy sets, for a call the previous code
# admitted), the semaphore members registered in the mapping ("sem:<zset>" fields, none written yet) and the
# mapping itself, atomically. The mapping goes only with the attempt that owns it. Idempotent: returns 0 when the
# member was already gone.
_RELEASE_SLOT = """
local org, scope, fleet, mapping = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local legacy_org, legacy_scope, legacy_fleet = KEYS[5], KEYS[6], KEYS[7]
local member = ARGV[1]
local legacy_fleet_member = ARGV[2]
local removed = redis.call('ZREM', org, member) + redis.call('ZREM', legacy_org, member)
for _, set in ipairs({scope, legacy_scope}) do
    if set ~= '' then
        redis.call('ZREM', set, member)
    end
end
redis.call('ZREM', fleet, member)
redis.call('ZREM', legacy_fleet, legacy_fleet_member)
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
local clock, org, legacy_org = KEYS[1], KEYS[2], KEYS[3]
local slot_ttl = tonumber(ARGV[1])
local now = redis_now(clock)
return live(org, now) + legacy_live(legacy_org, now, slot_ttl)
"""
)

# Read-only: counts the unexpired members without purging.
_COUNT_FLEET = (
    _REDIS_NOW
    + """
local clock, fleet, legacy_fleet = KEYS[1], KEYS[2], KEYS[3]
local slot_ttl = tonumber(ARGV[1])
local now = redis_now(clock)
return redis.call('ZCOUNT', fleet, string.format('(%.17g', now), '+inf')
    + legacy_live(legacy_fleet, now, slot_ttl)
"""
)

# Bind a run to its slot only if no mapping exists yet, so a duplicate start cannot overwrite the cleanup pointer.
# The admission limits ride along so a late claim can re-check them.
_BIND_MAPPING = """
local mapping = KEYS[1]
local org_id, slot_id = ARGV[1], ARGV[2]
local ttl = tonumber(ARGV[3])
local scope_field = ARGV[4]
local max_concurrent, scope_max_concurrent = ARGV[5], ARGV[6]

if redis.call('EXISTS', mapping) == 1 then
    return 0
end

redis.call('HSET', mapping, 'org_id', org_id, 'slot_id', slot_id)
if scope_field ~= '' then
    redis.call('HSET', mapping, 'scope_key', scope_field)
end
if max_concurrent ~= '' then
    redis.call('HSET', mapping, 'max_concurrent', max_concurrent)
end
if scope_max_concurrent ~= '' then
    redis.call('HSET', mapping, 'scope_max_concurrent', scope_max_concurrent)
end
redis.call('EXPIRE', mapping, ttl)
return 1
"""

LATE_CLAIM_OVERCOMMIT = "slot_late_claim_overcommit"


class SlotBackendError(Exception):
    """The slot backend (Redis) did not answer: the one error every slot primitive raises (VOZ-AC-B3-65-bis)."""


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


class SlotStore:
    """The call slots of every org, scope and the fleet, and the run-to-slot mappings, in one Redis."""

    def __init__(
        self, durations: Durations | None = None, *, test_clock: str | None = None
    ):
        self.redis_client: aioredis.Redis | None = None
        self.durations = durations or cell_durations()
        # Tests only: the slot scripts read "now" from this Redis key instead of TIME. Nothing in the app passes
        # it, so production scripts get '' and never read a clock key, whatever exists in Redis.
        self._test_clock = test_clock or ""

    async def _get_redis(self) -> aioredis.Redis:
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(
                REDIS_URL, decode_responses=True
            )
        return self.redis_client

    async def _redis(
        self, operation: str, command: Callable[[aioredis.Redis], Awaitable[Any]]
    ) -> Any:
        """Run one Redis command (``command(client)``); a backend failure becomes ``SlotBackendError``."""
        try:
            client = await self._get_redis()
            return await command(client)
        except (RedisError, OSError) as e:
            raise SlotBackendError(f"{operation} failed: {e}") from e

    def _eval(
        self, operation: str, script: str, slot_keys: list[str], *args: Any
    ) -> Awaitable[Any]:
        return self._redis(
            operation,
            lambda client: client.eval(script, len(slot_keys), *slot_keys, *args),
        )

    def _mapping(self, workflow_run_id: int) -> Awaitable[dict]:
        return self._redis(
            "slot mapping read",
            lambda client: client.hgetall(keys.mapping_key(workflow_run_id)),
        )

    async def acquire_slot(
        self,
        *,
        org_id: int,
        attempt_id: str,
        max_concurrent: int,
        scope_key: str | None = None,
        scope_max_concurrent: int | None = None,
        outbound_carrier: str | None = None,
    ) -> ConcurrentSlotAcquisition | None:
        """Admit an attempt: a pending slot in the org set, the optional scope set (``campaign:<id>``, bounded by
        ``scope_max_concurrent``) and the fleet set, atomically. Score = Redis TIME + ``pending_ttl_s``, or
        + ``outbound_pending_ttl_s(outbound_carrier)`` for an outbound call, which reaches a worker only once
        answered.

        None when a limit is reached. The worker must ``claim_slot`` within the pending lease or the slot expires
        on its own (two-phase lease).
        """
        pending_ttl = (
            self.durations.outbound_pending_ttl_s(outbound_carrier)
            if outbound_carrier
            else self.durations.pending_ttl_s
        )
        result = await self._eval(
            f"slot acquisition for org {org_id}",
            _ACQUIRE_SLOT,
            [
                self._test_clock,
                keys.org_key(org_id),
                keys.scope_key(scope_key),
                keys.fleet_key(),
                keys.legacy_org_key(org_id),
                keys.legacy_scope_key(scope_key),
            ],
            attempt_id,
            max_concurrent,
            scope_max_concurrent if scope_max_concurrent is not None else 0,
            pending_ttl,
            self.durations.slot_ttl,
        )
        if not result:
            return None
        acquired_slot_id, active_count = result
        return ConcurrentSlotAcquisition(
            slot_id=str(acquired_slot_id), active_count=int(active_count)
        )

    async def claim_slot(self, workflow_run_id: int) -> SlotLease | None:
        """Phase 2 of the lease, when the worker takes the call: the run's slot moves from pending to
        TIME + slot_ttl. None when the run holds no slot.

        A late claim (the pending lease already expired) re-adds the slot through the admission limits stored in
        the mapping: the answered call is always kept, and a full limit is logged as LATE_CLAIM_OVERCOMMIT.
        """
        mapping = await self._mapping(workflow_run_id)
        if "slot_id" not in mapping:
            return None
        lease = SlotLease(
            org_id=int(mapping["org_id"]),
            attempt_id=mapping["slot_id"],
            scope_key=mapping.get("scope_key") or None,
        )
        _late, overcommit, count = await self._eval(
            f"slot claim for workflow run {workflow_run_id}",
            _CLAIM_SLOT,
            [
                self._test_clock,
                keys.org_key(lease.org_id),
                keys.scope_key(lease.scope_key),
                keys.fleet_key(),
                keys.mapping_key(workflow_run_id),
                keys.legacy_org_key(lease.org_id),
                keys.legacy_scope_key(lease.scope_key),
            ],
            lease.attempt_id,
            self.durations.slot_ttl,
            lease.org_id,
            lease.scope_key or "",
            mapping.get("max_concurrent", ""),
            mapping.get("scope_max_concurrent", ""),
        )
        if overcommit:
            logger.warning(
                f"{LATE_CLAIM_OVERCOMMIT}: org={lease.org_id} "
                f"attempt_id={lease.attempt_id} workflow_run_id={workflow_run_id} "
                f"count={count} max_concurrent={mapping.get('max_concurrent')} "
                f"scope={lease.scope_key} scope_max_concurrent="
                f"{mapping.get('scope_max_concurrent')}: the answered call is kept"
            )
        return lease

    async def extend_pending_slot(
        self,
        *,
        org_id: int,
        attempt_id: str,
        carrier: str,
        scope_key: str | None = None,
    ) -> bool:
        """Restart the pending lease of a still-pending outbound slot on ``carrier``'s initiated/ringing callback.
        True if a set was extended."""
        extended = await self._eval(
            f"pending slot extension for org {org_id}",
            _EXTEND_PENDING,
            [
                self._test_clock,
                keys.org_key(org_id),
                keys.scope_key(scope_key),
                keys.fleet_key(),
            ],
            attempt_id,
            self.durations.outbound_pending_ttl_s(carrier),
            self.durations.slot_ttl,
        )
        return bool(extended)

    async def renew_slot(
        self,
        *,
        org_id: int,
        attempt_id: str,
        scope_key: str | None = None,
        workflow_run_id: int | None = None,
    ) -> None:
        """Score = TIME + slot_ttl on every set, re-added if Redis lost it; EXPIRE of the sets and of the
        mapping refreshed, the mapping re-created if missing."""
        await self._eval(
            f"slot renewal for org {org_id}",
            _RENEW_SLOT,
            [
                self._test_clock,
                keys.org_key(org_id),
                keys.scope_key(scope_key),
                keys.fleet_key(),
                keys.mapping_key(workflow_run_id),
                keys.legacy_org_key(org_id),
            ],
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
    ) -> bool:
        """The release funnel (VOZ-AC-B3-67): one atomic script for the org, scope and fleet members, the
        semaphore members in the run's mapping and the mapping itself (only if this attempt owns it).

        True if the slot was released, False if it was already gone (released or expired). On
        ``SlotBackendError`` the script changed nothing, so the mapping is still there for a retry.
        """
        if not attempt_id:
            return False
        removed = await self._eval(
            f"slot release for org {org_id}",
            _RELEASE_SLOT,
            [
                keys.org_key(org_id),
                keys.scope_key(scope_key),
                keys.fleet_key(),
                keys.mapping_key(workflow_run_id),
                keys.legacy_org_key(org_id),
                keys.legacy_scope_key(scope_key),
                keys.legacy_fleet_key(),
            ],
            attempt_id,
            keys.legacy_fleet_member(org_id, attempt_id),
        )
        if removed:
            logger.debug(f"Released concurrent slot {attempt_id} for org {org_id}")
        return bool(removed)

    async def get_concurrent_count(self, organization_id: int) -> int:
        """Active calls of an organization: purges the expired members (score <= Redis TIME), then counts."""
        return await self._eval(
            f"concurrent count for org {organization_id}",
            _COUNT_SLOTS,
            [
                self._test_clock,
                keys.org_key(organization_id),
                keys.legacy_org_key(organization_id),
            ],
            self.durations.slot_ttl,
        )

    async def get_fleet_concurrent_count(self) -> int:
        """Total active calls across every org — the fleet-wide autoscaling signal.

        One ZCOUNT over keys.fleet_key() (plus the legacy fleet set during the transition, keys.py), the
        fleet-wide mirror the acquire/renew/release paths maintain alongside the per-org counters: no keyspace
        scan, no per-org fan-out. Counting by score (not ZCARD) excludes expired slots (score = expiry <= Redis
        TIME) without writing, so an orphaned call can't keep the metric high and block scale-down.

        A failed read raises ``SlotBackendError`` like every primitive here: for an autoscaling signal, 0 is the
        most aggressive scale-down instruction, so the autoscale-metric endpoint turns the error into a 503.
        """
        return await self._eval(
            "fleet concurrent count",
            _COUNT_FLEET,
            [self._test_clock, keys.fleet_key(), keys.legacy_fleet_key()],
            self.durations.slot_ttl,
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
        """Bind a run to its slot unless a mapping already exists (False then), so a duplicate public/WebRTC start
        cannot overwrite the cleanup pointer. The admission limits ride along so a late claim can re-check them."""
        stored = await self._eval(
            f"slot mapping write for workflow run {workflow_run_id}",
            _BIND_MAPPING,
            [keys.mapping_key(workflow_run_id)],
            organization_id,
            slot_id,
            self.durations.slot_ttl,
            scope_key or "",
            "" if max_concurrent is None else max_concurrent,
            "" if scope_max_concurrent is None else scope_max_concurrent,
        )
        return bool(stored)

    async def get_workflow_slot_mapping(
        self, workflow_run_id: int
    ) -> tuple[int, str, str | None] | None:
        """(organization_id, slot_id, scope_key) of the run, or None when the run has no mapping; scope_key is
        None for slots acquired without a scope counter."""
        mapping = await self._mapping(workflow_run_id)
        if "org_id" in mapping and "slot_id" in mapping:
            return (
                int(mapping["org_id"]),
                mapping["slot_id"],
                mapping.get("scope_key") or None,
            )
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

        ``SlotBackendError`` propagates, so the DB cleanup marker stays pending until every counter and the
        mapping have been cleared. Repeating cleanup is safe: every slot goes through the release funnel, which
        drops the mapping together with the slot that owns it.
        """
        mapping = await self._mapping(workflow_run_id)
        slots = {(slot_id, scope_key)} if slot_id else set()
        if mapping:
            if int(mapping["org_id"]) != organization_id:
                raise ValueError(
                    f"Slot mapping organization mismatch for run {workflow_run_id}"
                )
            slots.add((mapping["slot_id"], mapping.get("scope_key") or None))
        for reserved_slot_id, reserved_scope in slots:
            await self.release_slot(
                org_id=organization_id,
                attempt_id=reserved_slot_id,
                scope_key=reserved_scope,
                workflow_run_id=workflow_run_id,
            )

    async def close(self) -> None:
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None


slot_store = SlotStore()
